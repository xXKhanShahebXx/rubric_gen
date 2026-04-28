"""
LLM-backed analytic judge for compiled case rubrics (starter prototype).

One call per candidate evaluates all criteria; overall decision is aggregated locally.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from rubric_gen.compiled.heuristic_judge import aggregate_overall_decision, dimension_scores_from_results
from rubric_gen.compiled.schema import (
    CaseEvaluationRecord,
    CaseRubric,
    CompiledCriterion,
    CriterionResult,
)
from rubric_gen.config import discover_compiled_llm_judge_model, parse_model_spec
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


_VERDICTS = frozenset({"MET", "UNMET", "CANNOT_ASSESS"})

_SYSTEM_PROMPT = """You evaluate a candidate artifact against a compiled rubric for ONE task instance.
Rules:
- Ground judgments in the task context and candidate artifact only; do not invent facts.
- Verdict per criterion must be exactly one of: MET, UNMET, CANNOT_ASSESS.
- MET: the artifact clearly satisfies the requirement.
- UNMET: the requirement is clearly not satisfied or violated.
- CANNOT_ASSESS: insufficient information in the text to decide (say why briefly).
- For structure criteria: count content as satisfied if clearly present under equivalent
  headings or narrative blocks, not only if literal heading strings match.
Return a single JSON object as specified in the user message. No markdown fences."""


def resolve_compiled_judge_spec(explicit_model: Optional[str]) -> ModelSpec:
    """Resolve model from CLI override, then env-based defaults."""
    if explicit_model and explicit_model.strip():
        return parse_model_spec(explicit_model.strip(), default_alias="compiled-llm-judge")
    spec = discover_compiled_llm_judge_model()
    if spec is None:
        raise ValueError(
            "No LLM judge model configured. Set RUBRIC_GEN_COMPILED_JUDGE_MODEL=provider:model "
            "or ensure a default judge API key exists (see discover_default_judge_model)."
        )
    return spec


def _criteria_payload(case_rubric: CaseRubric) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def one(c: CompiledCriterion, *, is_hard: bool) -> None:
        out.append(
            {
                "criterion_id": c.criterion_id,
                "is_hard_gate": is_hard,
                "label": c.label,
                "requirement": c.requirement,
                "severity_tier": c.severity_tier,
            }
        )

    for c in case_rubric.hard_gates:
        one(c, is_hard=True)
    for c in case_rubric.soft_checks:
        one(c, is_hard=False)
    return out


def _parse_llm_payload(
    raw: str,
    expected_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    obj = extract_json_object(raw)
    if not obj:
        raise ValueError("LLM response did not contain parseable JSON.")
    criteria = obj.get("criteria")
    if not isinstance(criteria, list):
        raise ValueError("JSON must contain a 'criteria' array.")
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in criteria:
        if not isinstance(row, dict):
            continue
        cid = row.get("criterion_id")
        if isinstance(cid, str):
            by_id[cid] = row
    missing = [i for i in expected_ids if i not in by_id]
    if missing:
        raise ValueError(f"LLM JSON missing criterion results for: {missing[:8]}")
    return by_id


def _row_to_result(cid: str, row: Dict[str, Any]) -> CriterionResult:
    v_raw = str(row.get("verdict", "")).strip().upper()
    if v_raw not in _VERDICTS:
        v_raw = "CANNOT_ASSESS"
    rationale = str(row.get("rationale", row.get("reasoning", ""))).strip()
    err = row.get("error_codes")
    codes: List[str] = []
    if isinstance(err, list):
        codes = [str(x) for x in err if x is not None]
    conf_raw = row.get("confidence")
    confidence: Optional[float] = None
    if isinstance(conf_raw, (int, float)):
        confidence = max(0.0, min(1.0, float(conf_raw)))
    score: float
    if v_raw == "MET":
        score = 1.0
    elif v_raw == "UNMET":
        score = 0.0
    else:
        score = 0.5
    return CriterionResult(
        criterion_id=cid,
        verdict=v_raw,
        rationale=f"[llm_analytic] {rationale}" if rationale else "[llm_analytic]",
        score_value=score,
        confidence=confidence,
        error_codes=codes,
        evidence_used=[],
    )


def _build_user_prompt(
    *,
    dialogue: str,
    note_text: str,
    criteria: List[Dict[str, Any]],
    note_family_label: str,
    artifact_label: str,
) -> str:
    dlim = int(os.getenv("RUBRIC_GEN_COMPILED_JUDGE_MAX_DIALOGUE_CHARS", "14000"))
    nlim = int(os.getenv("RUBRIC_GEN_COMPILED_JUDGE_MAX_NOTE_CHARS", "12000"))
    d = dialogue if len(dialogue) <= dlim else dialogue[:dlim] + "\n…[truncated]"
    n = note_text if len(note_text) <= nlim else note_text[:nlim] + "\n…[truncated]"
    crit_json = json.dumps(criteria, ensure_ascii=False, indent=2)
    artifact_name = artifact_label or "artifact"
    return f"""Task family (case-level scaffold): {note_family_label}

TASK CONTEXT:
{d}

CANDIDATE {artifact_name.upper()}:
{n}

CRITERIA (evaluate every id below):
{crit_json}

Return JSON with this exact shape:
{{
  "criteria": [
    {{
      "criterion_id": "<id>",
      "verdict": "MET" | "UNMET" | "CANNOT_ASSESS",
      "rationale": "<one short sentence>",
      "confidence": <number 0-1>,
      "error_codes": []
    }}
  ]
}}
Include exactly one object per criterion_id listed above. Use empty error_codes unless assigning a taxonomy code."""


def evaluate_note_with_llm_judge(
    *,
    candidate_id: str,
    note_text: str,
    dialogue: str,
    case_rubric: CaseRubric,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache] = None,
    evaluation_suffix: str = "eval_llm_v0",
) -> Tuple[CaseEvaluationRecord, bool]:
    """
    Returns the evaluation record and whether the result was served from cache.
    """
    criteria = _criteria_payload(case_rubric)
    expected_ids = [c["criterion_id"] for c in criteria]
    cache_hit = False

    payload_for_key = {
        "criteria_ids": expected_ids,
        "dialogue_hash": stable_hash(dialogue),
        "note_hash": stable_hash(note_text),
        "rubric_id": case_rubric.rubric_id,
        "rubric_version": case_rubric.version,
        "model": f"{model_spec.provider}:{model_spec.model}",
    }
    cache_key = make_cache_key("compiled_llm_judge_v1", payload_for_key)

    raw_text = ""
    if cache and cache.enabled:
        cache.load()
        hit = cache.get(cache_key)
        if hit and isinstance(hit.get("raw_response"), str):
            raw_text = hit["raw_response"]
            cache_hit = True

    if not raw_text:
        user_prompt = _build_user_prompt(
            dialogue=dialogue,
            note_text=note_text,
            criteria=criteria,
            note_family_label=case_rubric.note_family_id,
            artifact_label=case_rubric.artifact_label,
        )
        resp = router.generate(
            model_spec,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        raw_text = resp.raw_text or resp.text
        if cache and cache.enabled:
            cache.set(
                cache_key,
                {
                    "raw_response": raw_text,
                    "candidate_id": candidate_id,
                    "example_id": case_rubric.example_id,
                },
            )

    by_id = _parse_llm_payload(raw_text, expected_ids)

    hard_results: List[CriterionResult] = []
    soft_results: List[CriterionResult] = []
    for c in case_rubric.hard_gates:
        hard_results.append(_row_to_result(c.criterion_id, by_id[c.criterion_id]))
    for c in case_rubric.soft_checks:
        soft_results.append(_row_to_result(c.criterion_id, by_id[c.criterion_id]))

    dims = dimension_scores_from_results(case_rubric, hard_results, soft_results)
    overall = aggregate_overall_decision(case_rubric, hard_results, soft_results)

    meta: Dict[str, Any] = {
        "mode": "llm_analytic_single_call",
        "model_alias": model_spec.alias,
        "provider": model_spec.provider,
        "model": model_spec.model,
        "prompt_version": "compiled_llm_judge_v1",
    }
    if cache_hit:
        meta["cache"] = "hit"
    else:
        meta["cache"] = "miss"

    return (
        CaseEvaluationRecord(
            evaluation_id=f"{case_rubric.example_id}_{candidate_id}_{evaluation_suffix}",
            rubric_id=case_rubric.rubric_id,
            example_id=case_rubric.example_id,
            candidate_id=candidate_id,
            note_family_id=case_rubric.note_family_id,
            rubric_version=case_rubric.version,
            hard_gate_results=hard_results,
            soft_results=soft_results,
            dimension_scores=dims,
            overall_decision=overall,
            judge_metadata=meta,
            task_profile_id=case_rubric.task_profile_id,
            task_family_id=case_rubric.task_family_id,
            artifact_label=case_rubric.artifact_label,
        ),
        cache_hit,
    )


def evaluate_artifact_with_llm_judge(
    *,
    candidate_id: str,
    artifact_text: str,
    task_context: str,
    case_rubric: CaseRubric,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache] = None,
    evaluation_suffix: str = "eval_llm_v0",
) -> Tuple[CaseEvaluationRecord, bool]:
    return evaluate_note_with_llm_judge(
        candidate_id=candidate_id,
        note_text=artifact_text,
        dialogue=task_context,
        case_rubric=case_rubric,
        model_spec=model_spec,
        router=router,
        cache=cache,
        evaluation_suffix=evaluation_suffix,
    )
