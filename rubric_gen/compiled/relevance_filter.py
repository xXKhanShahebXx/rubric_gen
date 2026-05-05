"""
Per-pair relevance filter for retrieved rubrics.

Family-based retrieval (``RubricLibrary.filter_by_family``) is intentionally coarse: it
keeps every criterion tagged with the source family, ignoring whether that criterion is
about the same *topic* as the prompt. A "cardiology" criterion can be retrieved for an
endocrinology prompt; a "geometry" rubric can land on a probability question. Both pass
family-membership but fail to discriminate the actual response, polluting downstream
satisfaction scoring with rubrics the candidate could never have addressed.

This module makes one batched LLM call per example. For each retrieved criterion, the
filter model returns one of:

- ``APPLICABLE``: the criterion is on-topic for this prompt+response pair.
- ``IRRELEVANT``: the criterion clearly does not apply (different topic / different
  sub-domain / asks for evidence the response could not possibly contain).
- ``UNCERTAIN``: the model is unsure (off-by-a-cousin, ambiguous wording, partially
  applies).

Two gating policies:

- ``conservative`` (default): drop only ``IRRELEVANT``. ``UNCERTAIN`` survives so we
  do not throw away rubrics on borderline-applicable cases.
- ``aggressive``: keep only ``APPLICABLE``. ``UNCERTAIN`` is dropped along with
  ``IRRELEVANT``. Higher precision in the surviving rubric set; risks discarding
  signal on hard cases.

The filter call is cached via :class:`rubric_gen.storage.JsonlCache` keyed on the
prompt text, candidate texts, sorted criterion ids, model spec, and strictness, so
re-running the same example with the same configuration costs zero LLM calls.

Module is dependency-free of JudgeBench so it can be reused for any retrieval path
that yields a list of :class:`RubricLibraryCriterion`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.rubric_library import RubricLibraryCriterion
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


RELEVANCE_FILTER_PROMPT_VERSION = "rubric_relevance_filter_v1"

DEFAULT_FILTER_MODEL = ModelSpec(
    alias="relevance_filter_sonnet45",
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    api_key_env="ANTHROPIC_API_KEY",
    max_tokens=4096,
)

CONSERVATIVE = "conservative"
AGGRESSIVE = "aggressive"
ALLOWED_STRICTNESS = (CONSERVATIVE, AGGRESSIVE)

VERDICT_APPLICABLE = "APPLICABLE"
VERDICT_IRRELEVANT = "IRRELEVANT"
VERDICT_UNCERTAIN = "UNCERTAIN"
ALLOWED_VERDICTS = (VERDICT_APPLICABLE, VERDICT_IRRELEVANT, VERDICT_UNCERTAIN)


_SYSTEM_PROMPT = """You are a rubric relevance filter.

You are given a prompt-response pair from an evaluation set, plus a numbered list of
candidate rubric criteria that were retrieved from a rubric library. Each criterion is a
short statement of something the response should (or should not) demonstrate.

Your job is to decide, for each criterion, whether it is applicable to scoring THIS
particular prompt-response pair. A criterion is APPLICABLE only when scoring the
response against it produces a meaningful signal about the response's quality on this
prompt. A criterion is IRRELEVANT when it is clearly about a different topic, a
different sub-domain, or asks for evidence the response could not contain even if
correct. Use UNCERTAIN sparingly, only when the criterion is borderline.

Examples:
- Prompt about pancreatic cancer staging; criterion about breast cancer screening
  intervals -> IRRELEVANT (different cancer; staging vs screening).
- Prompt about asthma exacerbation management; criterion about avoiding unsupported
  medication doses -> APPLICABLE (general medication-grounding criterion).
- Prompt about a math word problem; criterion about citing source studies ->
  IRRELEVANT (math response would never cite studies).
- Prompt about endocarditis antibiotics; criterion about culture-directed therapy ->
  APPLICABLE (clearly on-topic).
- Prompt that tangentially mentions a side topic; criterion narrowly about that side
  topic -> UNCERTAIN.

Output format: a single JSON object whose keys are the criterion indices (as strings)
and whose values are objects with ``verdict`` and ``reason``. Example:

{
  "0": {"verdict": "APPLICABLE", "reason": "general medication grounding"},
  "1": {"verdict": "IRRELEVANT", "reason": "criterion is about pediatrics; prompt is geriatric"},
  "2": {"verdict": "UNCERTAIN", "reason": "criterion narrowly about side-topic"}
}

Output ONLY the JSON object. Do not include any other text."""


@dataclass(frozen=True)
class RelevanceFilterConfig:
    """Configuration for the relevance filter.

    Attributes:
        enabled: Master switch. When ``False``, ``filter_relevant_criteria`` short-
            circuits, returning the input unchanged.
        model_spec: The judge model. Defaults to Anthropic Sonnet 4.5.
        strictness: ``"conservative"`` keeps APPLICABLE + UNCERTAIN. ``"aggressive"``
            keeps APPLICABLE only.
        max_criteria_per_call: Cap on how many criteria are sent in one batched call.
            If more criteria are passed in, they are processed in sequential batches
            of this size (cache hits make subsequent runs free).
        prompt_version: Bumped when the filter prompt changes; included in cache key
            to avoid stale hits on prompt revisions.
        temperature: Sampling temperature for the filter model. ``0.0`` for
            determinism.
    """

    enabled: bool = False
    model_spec: ModelSpec = field(default_factory=lambda: DEFAULT_FILTER_MODEL)
    strictness: str = CONSERVATIVE
    max_criteria_per_call: int = 16
    prompt_version: str = RELEVANCE_FILTER_PROMPT_VERSION
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.strictness not in ALLOWED_STRICTNESS:
            raise ValueError(
                f"strictness must be one of {ALLOWED_STRICTNESS}; got {self.strictness!r}."
            )
        if self.max_criteria_per_call < 1:
            raise ValueError("max_criteria_per_call must be >= 1.")


def _criterion_text(criterion: RubricLibraryCriterion) -> str:
    """Render a criterion as a single readable line for the filter prompt."""
    parts: List[str] = []
    label = (criterion.label or "").strip()
    requirement = (criterion.requirement or "").strip()
    if label and requirement:
        parts.append(f"{label}: {requirement}")
    elif requirement:
        parts.append(requirement)
    elif label:
        parts.append(label)
    else:
        parts.append(criterion.criterion_id)
    if criterion.dimension:
        parts.append(f"[dimension={criterion.dimension}]")
    return " ".join(parts)


def _render_user_prompt(
    prompt_text: str,
    candidate_texts: Sequence[str],
    batch: Sequence[RubricLibraryCriterion],
) -> str:
    candidate_blocks: List[str] = []
    for index, text in enumerate(candidate_texts):
        if not text:
            continue
        label = "Response" if len(candidate_texts) == 1 else f"Response {chr(ord('A') + index)}"
        candidate_blocks.append(f"<{label}>\n{text.strip()}\n</{label}>")
    candidates_section = "\n\n".join(candidate_blocks) if candidate_blocks else "<Response>\n(no response provided)\n</Response>"

    criteria_lines = [
        f"{index}. {_criterion_text(criterion)}"
        for index, criterion in enumerate(batch)
    ]
    criteria_section = "\n".join(criteria_lines)

    return (
        f"<Prompt>\n{prompt_text.strip()}\n</Prompt>\n\n"
        f"{candidates_section}\n\n"
        f"<Criteria>\n{criteria_section}\n</Criteria>\n\n"
        "Return your JSON object now."
    )


def _normalize_verdict(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if text in ALLOWED_VERDICTS:
        return text
    if text in {"YES", "Y", "TRUE", "RELEVANT", "APPLY", "APPLIES"}:
        return VERDICT_APPLICABLE
    if text in {"NO", "N", "FALSE", "NOT_RELEVANT", "OFF_TOPIC", "OFFTOPIC"}:
        return VERDICT_IRRELEVANT
    if text in {"MAYBE", "UNSURE", "UNKNOWN", "PARTIAL"}:
        return VERDICT_UNCERTAIN
    return ""


def _parse_filter_response(
    text: str,
    batch_size: int,
) -> Tuple[List[Dict[str, str]], str]:
    """Parse the model's JSON output into a per-index list of {verdict, reason}.

    Returns ``(verdicts, parse_error)`` where ``verdicts[i]`` defaults to UNCERTAIN if
    the model failed to provide that index. ``parse_error`` is a short tag describing
    the failure mode (or empty when parsing succeeded fully).
    """
    payload = extract_json_object(text)
    parse_error = ""
    if not isinstance(payload, dict):
        return (
            [{"verdict": VERDICT_UNCERTAIN, "reason": "parse_failed"} for _ in range(batch_size)],
            "no_json_object",
        )

    verdicts: List[Dict[str, str]] = []
    missing_indices: List[int] = []
    for idx in range(batch_size):
        entry = payload.get(str(idx))
        if entry is None:
            entry = payload.get(idx)
        if not isinstance(entry, Mapping):
            missing_indices.append(idx)
            verdicts.append({"verdict": VERDICT_UNCERTAIN, "reason": "missing_in_response"})
            continue
        verdict = _normalize_verdict(entry.get("verdict") or entry.get("label") or entry.get("decision"))
        if not verdict:
            missing_indices.append(idx)
            verdicts.append({"verdict": VERDICT_UNCERTAIN, "reason": "unrecognized_verdict"})
            continue
        reason = str(entry.get("reason") or entry.get("rationale") or "").strip()
        verdicts.append({"verdict": verdict, "reason": reason})
    if missing_indices:
        parse_error = f"missing_or_unrecognized:{len(missing_indices)}"
    return verdicts, parse_error


def _gate_keeps(verdict: str, strictness: str) -> bool:
    """Apply the gating policy: which verdicts survive?"""
    if strictness == AGGRESSIVE:
        return verdict == VERDICT_APPLICABLE
    return verdict in {VERDICT_APPLICABLE, VERDICT_UNCERTAIN}


def _build_cache_key(
    *,
    config: RelevanceFilterConfig,
    prompt_text: str,
    candidate_texts: Sequence[str],
    batch: Sequence[RubricLibraryCriterion],
) -> str:
    payload: Dict[str, Any] = {
        "prompt_version": config.prompt_version,
        "model": f"{config.model_spec.provider}:{config.model_spec.model}",
        "strictness": config.strictness,
        "temperature": round(float(config.temperature), 4),
        "prompt_hash": stable_hash(prompt_text or ""),
        "candidates_hash": stable_hash([(text or "") for text in candidate_texts]),
        "criterion_ids": [c.criterion_id for c in batch],
        "criterion_text_hash": stable_hash([_criterion_text(c) for c in batch]),
    }
    return make_cache_key(config.prompt_version, payload)


def _run_filter_batch(
    *,
    config: RelevanceFilterConfig,
    prompt_text: str,
    candidate_texts: Sequence[str],
    batch: Sequence[RubricLibraryCriterion],
    router: LLMRouter,
    cache: Optional[JsonlCache],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Run one batched filter call (or hit the cache). Returns per-criterion verdicts and a debug payload."""
    cache_key = _build_cache_key(
        config=config,
        prompt_text=prompt_text,
        candidate_texts=candidate_texts,
        batch=batch,
    )
    cache_hit = False
    cached_payload: Optional[Mapping[str, Any]] = None
    if cache is not None and cache.enabled:
        cache.load()
        cached = cache.get(cache_key)
        if cached and isinstance(cached.get("verdicts"), list):
            cached_payload = cached
            cache_hit = True

    if cached_payload is not None:
        verdicts_payload = cached_payload.get("verdicts") or []
        verdicts: List[Dict[str, str]] = []
        for entry in verdicts_payload:
            if isinstance(entry, Mapping):
                verdicts.append(
                    {
                        "verdict": _normalize_verdict(entry.get("verdict")) or VERDICT_UNCERTAIN,
                        "reason": str(entry.get("reason") or "").strip(),
                    }
                )
            else:
                verdicts.append({"verdict": VERDICT_UNCERTAIN, "reason": "cached_malformed_entry"})
        # Pad / truncate to batch size in case the cached entry shape drifted.
        if len(verdicts) < len(batch):
            verdicts.extend(
                [{"verdict": VERDICT_UNCERTAIN, "reason": "cached_short"}]
                * (len(batch) - len(verdicts))
            )
        verdicts = verdicts[: len(batch)]
        return verdicts, {
            "cache_hit": True,
            "parse_error": str(cached_payload.get("parse_error") or ""),
            "raw_response": str(cached_payload.get("raw_response") or ""),
        }

    user_prompt = _render_user_prompt(prompt_text, candidate_texts, batch)
    try:
        response = router.generate(
            config.model_spec,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=float(config.temperature),
        )
    except Exception as exc:
        # Conservative fallback: treat everything as UNCERTAIN so the conservative gate
        # keeps every criterion. The router_error tag surfaces in the debug payload so
        # downstream diagnostics can detect filter outages.
        verdicts = [
            {"verdict": VERDICT_UNCERTAIN, "reason": f"router_error:{type(exc).__name__}"}
            for _ in batch
        ]
        return verdicts, {
            "cache_hit": False,
            "parse_error": "router_error",
            "raw_response": "",
            "router_error": f"{type(exc).__name__}: {exc}",
        }

    raw_text = response.raw_text or response.text or ""
    verdicts, parse_error = _parse_filter_response(raw_text, len(batch))
    if cache is not None and cache.enabled:
        cache.set(
            cache_key,
            {
                "kind": "rubric_relevance_filter",
                "verdicts": verdicts,
                "parse_error": parse_error,
                "raw_response": raw_text,
            },
        )
    return verdicts, {
        "cache_hit": cache_hit,
        "parse_error": parse_error,
        "raw_response": raw_text,
    }


def filter_relevant_criteria(
    criteria: Sequence[RubricLibraryCriterion],
    prompt_text: str,
    candidate_texts: Sequence[str],
    *,
    config: RelevanceFilterConfig,
    router: Optional[LLMRouter],
    cache: Optional[JsonlCache] = None,
) -> Tuple[List[RubricLibraryCriterion], Dict[str, Any]]:
    """Filter a list of retrieved criteria down to the relevance-applicable subset.

    Args:
        criteria: The retrieved candidate criteria. Order is preserved among survivors.
        prompt_text: The example's prompt / question text.
        candidate_texts: One or two response texts (e.g. ``[response_A, response_B]``
            for JudgeBench pairs, or ``[response]`` for medical Q&A).
        config: The filter configuration.
        router: An LLMRouter for issuing filter calls. May be ``None`` only when
            ``config.enabled`` is False or when running with caches that are guaranteed
            to be fully warm.
        cache: Optional cache for storing/retrieving filter verdicts. When ``None`` or
            disabled, every call hits the LLM.

    Returns:
        ``(kept, debug)`` where ``kept`` is the surviving subset (in input order) and
        ``debug`` records counts, per-criterion verdicts/reasons, batch metadata, and
        any router/parse errors.
    """
    criteria_list = list(criteria)
    if not config.enabled or not criteria_list:
        return criteria_list, {
            "enabled": bool(config.enabled),
            "input_count": len(criteria_list),
            "kept_count": len(criteria_list),
            "dropped_count": 0,
            "verdict_counts": {VERDICT_APPLICABLE: 0, VERDICT_IRRELEVANT: 0, VERDICT_UNCERTAIN: 0},
            "decisions": [],
            "batches": [],
            "strictness": config.strictness,
            "model": f"{config.model_spec.provider}:{config.model_spec.model}",
        }

    if router is None:
        raise ValueError(
            "filter_relevant_criteria requires a non-None LLMRouter when config.enabled=True."
        )

    decisions: List[Dict[str, Any]] = []
    batch_records: List[Dict[str, Any]] = []
    kept: List[RubricLibraryCriterion] = []
    counts: Dict[str, int] = {
        VERDICT_APPLICABLE: 0,
        VERDICT_IRRELEVANT: 0,
        VERDICT_UNCERTAIN: 0,
    }

    batch_size = max(1, int(config.max_criteria_per_call))
    for batch_start in range(0, len(criteria_list), batch_size):
        batch = criteria_list[batch_start : batch_start + batch_size]
        verdicts, batch_debug = _run_filter_batch(
            config=config,
            prompt_text=prompt_text,
            candidate_texts=candidate_texts,
            batch=batch,
            router=router,
            cache=cache,
        )
        for offset, (criterion, verdict_payload) in enumerate(zip(batch, verdicts)):
            verdict = verdict_payload.get("verdict", VERDICT_UNCERTAIN)
            reason = verdict_payload.get("reason", "")
            counts[verdict] = counts.get(verdict, 0) + 1
            survives = _gate_keeps(verdict, config.strictness)
            decisions.append(
                {
                    "criterion_id": criterion.criterion_id,
                    "verdict": verdict,
                    "reason": reason,
                    "kept": survives,
                    "batch_index": batch_start // batch_size,
                    "position_in_batch": offset,
                }
            )
            if survives:
                kept.append(criterion)
        batch_records.append(
            {
                "batch_index": batch_start // batch_size,
                "size": len(batch),
                "cache_hit": bool(batch_debug.get("cache_hit")),
                "parse_error": str(batch_debug.get("parse_error") or ""),
                "router_error": str(batch_debug.get("router_error") or ""),
            }
        )

    debug: Dict[str, Any] = {
        "enabled": True,
        "input_count": len(criteria_list),
        "kept_count": len(kept),
        "dropped_count": len(criteria_list) - len(kept),
        "verdict_counts": counts,
        "decisions": decisions,
        "batches": batch_records,
        "strictness": config.strictness,
        "model": f"{config.model_spec.provider}:{config.model_spec.model}",
    }
    return kept, debug


def parse_strictness(value: Any) -> str:
    """Normalize a CLI / config string into a valid strictness value.

    Raises ValueError if the input is non-empty and not recognised; returns the
    default (``conservative``) when the input is empty / None.
    """
    text = str(value or "").strip().lower()
    if not text:
        return CONSERVATIVE
    if text not in ALLOWED_STRICTNESS:
        raise ValueError(
            f"Unknown relevance filter strictness {value!r}; expected one of "
            f"{', '.join(ALLOWED_STRICTNESS)}."
        )
    return text


def build_default_config(
    *,
    enabled: bool,
    model_spec: Optional[ModelSpec] = None,
    strictness: str = CONSERVATIVE,
    max_criteria_per_call: int = 16,
) -> RelevanceFilterConfig:
    """Convenience constructor used by callers that don't want to wire a ModelSpec."""
    spec = model_spec or DEFAULT_FILTER_MODEL
    return RelevanceFilterConfig(
        enabled=bool(enabled),
        model_spec=spec,
        strictness=parse_strictness(strictness),
        max_criteria_per_call=int(max_criteria_per_call),
    )


def filter_cache_path(cache_root) -> "os.PathLike[str]":
    """Return the canonical cache path under a cache root directory."""
    from pathlib import Path

    return Path(cache_root) / "rubric_relevance_filter.jsonl"
