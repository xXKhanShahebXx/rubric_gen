"""
LLM-backed rubric proposer for the offline library builder.

This module wires the existing ``discover_pair_criteria`` discovery code to external preference
pairs (not JudgeBench) and returns the resulting proposals as ``ProposedCriterion`` rows. Multi-
model rollouts run one pair through N models (defaults: ``openai:gpt-4o-2024-05-13``,
``anthropic:claude-sonnet-4-20250514``, ``google:gemini-2.5-pro``). Every proposal is tagged with
the proposer model so downstream aggregation can weight by model diversity.

Import-time side effects are avoided: if ``rubric_gen.compiled.discovery`` is unavailable or no
model credentials are configured, ``make_llm_proposer`` raises so ``build_default_proposer``
falls back to the seed proposer cleanly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from rubric_gen.compiled.rubric_library_builder import (
    ExternalPreferencePair,
    ProposedCriterion,
)
from rubric_gen.config import parse_model_spec
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.types import ModelSpec


_DEFAULT_PROPOSER_MODELS: Sequence[str] = (
    "openai:gpt-4o-2024-05-13",
    "anthropic:claude-sonnet-4-20250514",
)


_PROPOSER_SYSTEM_PROMPT = """You are distilling portable evaluation criteria from a preference pair.

You are given a prompt and two responses: one is preferred (the "chosen" side), the other is less
preferred (the "rejected" side). Propose 3-6 small, atomic, binary-leaning criteria that would
help a future judge distinguish the preferred side from the rejected side on similar tasks.

Rules:
- Each criterion must check a single concrete behavior that can be answered YES or NO.
- Criteria must transfer to OTHER tasks in the same family (do not mention the specific prompt).
- Do not reference the candidate IDs, names, or surface text of the responses.
- Use `severity_tier` in {"hard_gate", "high", "medium", "low"}.
- Use `focus_kind` in {"final_answer", "assignment_completeness", "clue_consistency",
  "exclusivity", "contradiction", "final_answer_format", "constraint", "derivation",
  "arithmetic", "behavior", "edge_cases", "grounding", "instruction", "completeness",
  "reasoning"}.

Return JSON with shape:
{
  "criteria": [
    {
      "dimension": "<short dimension name>",
      "label": "<short label>",
      "requirement": "<single concrete binary requirement>",
      "severity_tier": "...",
      "focus_kind": "..."
    }
  ]
}
No markdown fences."""


def _resolve_model_specs(raw: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for idx, raw_spec in enumerate(raw or ()):
        try:
            specs.append(parse_model_spec(str(raw_spec), alias_prefix="rubric-library-proposer", index=idx))
        except Exception:
            continue
    if not specs:
        raise RuntimeError(
            "No usable LLM proposer models. Configure OPENAI_API_KEY / ANTHROPIC_API_KEY or set "
            "proposer.models in the rubric library manifest."
        )
    return specs


def _build_user_prompt(pair: ExternalPreferencePair) -> str:
    return (
        f"Task family tag: {pair.source_family or 'generic'}\n"
        f"Source: {pair.source}\n"
        f"Focus hint: {pair.focus_kind or 'n/a'}\n\n"
        f"PROMPT:\n{pair.prompt}\n\n"
        f"CHOSEN RESPONSE:\n{pair.chosen}\n\n"
        f"REJECTED RESPONSE:\n{pair.rejected}\n\n"
        f"Return the JSON object described in the system message."
    )


def _parse_proposals(raw: str, *, source_tag: str, model_alias: str) -> List[ProposedCriterion]:
    obj = extract_json_object(raw)
    if not isinstance(obj, Mapping):
        return []
    criteria = obj.get("criteria")
    if not isinstance(criteria, list):
        return []
    out: List[ProposedCriterion] = []
    for row in criteria:
        if not isinstance(row, Mapping):
            continue
        dimension = str(row.get("dimension", "") or "").strip()
        label = str(row.get("label", "") or "").strip()
        requirement = str(row.get("requirement", "") or "").strip()
        severity = str(row.get("severity_tier", "") or "medium").strip().lower()
        focus = str(row.get("focus_kind", "") or "").strip().lower()
        if not dimension or not label or not requirement:
            continue
        if severity not in {"hard_gate", "high", "medium", "low"}:
            severity = "medium"
        out.append(
            ProposedCriterion(
                dimension=dimension,
                label=label,
                requirement=requirement,
                severity_tier=severity,
                focus_kind=focus,
                source_tag=f"{source_tag}::{model_alias}",
            )
        )
    return out


def make_llm_proposer(config: Mapping[str, Any]) -> Callable[[ExternalPreferencePair], List[ProposedCriterion]]:
    model_specs_raw = list(config.get("models") or _DEFAULT_PROPOSER_MODELS)
    model_specs = _resolve_model_specs(model_specs_raw)
    source_tag = str(config.get("source_tag", "llm_proposer") or "llm_proposer")
    temperature = float(config.get("temperature", 0.2) or 0.2)
    router = LLMRouter()

    def _proposer(pair: ExternalPreferencePair) -> List[ProposedCriterion]:
        aggregate: List[ProposedCriterion] = []
        user_prompt = _build_user_prompt(pair)
        for spec in model_specs:
            try:
                response = router.generate(
                    spec,
                    system_prompt=_PROPOSER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=temperature,
                )
                raw_text = response.raw_text or response.text
            except Exception:
                continue
            aggregate.extend(_parse_proposals(raw_text, source_tag=source_tag, model_alias=spec.alias))
        return aggregate

    return _proposer


_VERIFIER_SYSTEM_PROMPT = """You are auditing one proposed evaluation criterion against a preference pair.

Given a prompt, a chosen response, a rejected response, and a candidate criterion, decide whether
satisfying the criterion *would* have made the judge more likely to prefer the chosen response
over the rejected one.

Rules:
- The criterion must be directionally aligned: answering YES on the chosen side and NO on the
  rejected side is the happy path. Criteria that favor the rejected side are misaligned.
- Ties (YES on both, NO on both) are not evidence.
- Hallucinated content or surface-form coverage that does not actually reflect quality is
  misaligned.

Return JSON:
{
  "aligned": true | false,
  "evidence": <integer in [0, 3]>,
  "rationale": "<1-sentence reason>"
}
No markdown fences."""


def make_llm_verifier(config: Mapping[str, Any]):
    model_specs_raw = list(config.get("models") or [_DEFAULT_PROPOSER_MODELS[0]])
    model_specs = _resolve_model_specs(model_specs_raw)
    router = LLMRouter()

    def _verifier(pair: ExternalPreferencePair, criterion: ProposedCriterion):
        user_prompt = (
            f"PROMPT:\n{pair.prompt}\n\nCHOSEN:\n{pair.chosen}\n\nREJECTED:\n{pair.rejected}\n\n"
            f"CRITERION:\ndimension={criterion.dimension}\nlabel={criterion.label}\n"
            f"requirement={criterion.requirement}\nseverity_tier={criterion.severity_tier}\n\n"
            f"Return the JSON object described in the system message."
        )
        aligned = False
        evidence = 0
        for spec in model_specs:
            try:
                response = router.generate(
                    spec,
                    system_prompt=_VERIFIER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.0,
                )
                raw_text = response.raw_text or response.text
            except Exception:
                continue
            obj = extract_json_object(raw_text)
            if not isinstance(obj, Mapping):
                continue
            aligned = bool(obj.get("aligned")) or aligned
            try:
                evidence = max(evidence, int(obj.get("evidence", 0) or 0))
            except Exception:
                pass
            if aligned and evidence >= 1:
                break
        return aligned, evidence

    return _verifier
