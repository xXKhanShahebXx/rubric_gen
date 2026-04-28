"""
Single-call RaR-style holistic GPT-4o pair judge.

This module implements the implicit-holistic aggregation path from
*Rubrics as Rewards* (`2507.17746v2`, §5): rather than computing a weighted sum over per-criterion
YES/NO satisfactions, we give the judge the full rubric text plus both candidate responses in
neutral X/Y labels and ask it to return a single pairwise decision with a confidence and a short
justification.

The holistic judge is run in parallel with the WU-weighted path and is consumed by the hybrid
aggregator in :mod:`rubric_gen.compiled.judgebench_eval`. To keep the cost bounded we cache
responses by (pair hash, rubric hash, model) and run order-swap as a second call.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from rubric_gen.compiled.task_profiles import get_task_profile
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec, RubricCriterion


HOLISTIC_JUDGE_PROMPT_VERSION = "judgebench_holistic_pair_judge_v1"


_SYSTEM_PROMPT = """You compare two candidate responses against a shared rubric and decide which one better satisfies the rubric overall.

Rules:
- Read the prompt, the rubric criteria, and the two candidate responses labelled X and Y.
- Evaluate each criterion mentally; do not write per-criterion verdicts.
- Output a single pairwise decision: X > Y, Y > X, or X = Y (use = only when the responses are genuinely indistinguishable on the rubric).
- Ground your judgement in the rubric; do not reward verbosity, style, or personal preference.
- Return a single JSON object with the exact shape:

{
  "decision": "X>Y" | "Y>X" | "X=Y",
  "confidence": "high" | "medium" | "low",
  "distinguishing_behavior": "<one-sentence reason tied to the rubric>"
}
No markdown fences."""


def _criteria_summary(rubrics: Sequence[RubricCriterion]) -> str:
    if not rubrics:
        return "(no rubric criteria supplied)"
    lines: List[str] = []
    for idx, criterion in enumerate(rubrics, start=1):
        text = str(getattr(criterion, "text", "") or "").strip()
        if not text:
            continue
        lines.append(f"{idx}. {text}")
    return "\n".join(lines) or "(no rubric criteria supplied)"


def _build_user_prompt(
    *,
    example: ExampleRecord,
    rubrics: Sequence[RubricCriterion],
    left_text: str,
    right_text: str,
    left_id: str,
    right_id: str,
    task_profile_id: str,
) -> str:
    profile = get_task_profile(task_profile_id)
    artifact_label = profile.artifact_label or "response"
    max_prompt = 6000
    max_response = 6000
    prompt = (example.task_prompt or example.conversation or "").strip()
    if len(prompt) > max_prompt:
        prompt = prompt[:max_prompt] + "\n...[truncated]"
    left_blob = left_text or ""
    right_blob = right_text or ""
    if len(left_blob) > max_response:
        left_blob = left_blob[:max_response] + "\n...[truncated]"
    if len(right_blob) > max_response:
        right_blob = right_blob[:max_response] + "\n...[truncated]"
    rubric_block = _criteria_summary(rubrics)
    return f"""Task family: {profile.task_profile_id}

PROMPT:
{prompt}

RUBRIC (apply holistically; do not return per-criterion verdicts):
{rubric_block}

CANDIDATE {left_id}:
{left_blob}

CANDIDATE {right_id}:
{right_blob}

Return the JSON object described in the system message. Compare the two candidate {artifact_label}s overall on the rubric."""


def _normalize_decision(raw_decision: str, *, left_id: str, right_id: str, left_pair_position: str) -> str:
    text = (raw_decision or "").strip().upper().replace(" ", "")
    left_id = str(left_id).strip().upper()
    right_id = str(right_id).strip().upper()
    if text in {f"{left_id}={right_id}", f"{right_id}={left_id}", "A=B"}:
        return "A=B"
    if text in {f"{left_id}>{right_id}", "A>B"}:
        return "A>B" if str(left_pair_position).strip().upper() == "A" else "B>A"
    if text in {f"{right_id}>{left_id}", "B>A"}:
        return "B>A" if str(left_pair_position).strip().upper() == "A" else "A>B"
    return ""


def _run_holistic_once(
    *,
    example_record: ExampleRecord,
    rubrics: Sequence[RubricCriterion],
    left_candidate: CandidateNote,
    right_candidate: CandidateNote,
    left_pair_position: str,
    task_profile_id: str,
    judge_model: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    order_label: str,
    left_id: str = "X",
    right_id: str = "Y",
) -> Dict[str, Any]:
    user_prompt = _build_user_prompt(
        example=example_record,
        rubrics=rubrics,
        left_text=left_candidate.text,
        right_text=right_candidate.text,
        left_id=left_id,
        right_id=right_id,
        task_profile_id=task_profile_id,
    )
    payload = {
        "pair_id": example_record.metadata.get("pair_id", example_record.example_id),
        "task_profile_id": task_profile_id,
        "candidate_left": stable_hash(left_candidate.text),
        "candidate_right": stable_hash(right_candidate.text),
        "left_pair_position": str(left_pair_position),
        "left_id": left_id,
        "right_id": right_id,
        "rubric_hash": stable_hash("|".join(str(getattr(c, "text", "")) for c in rubrics)),
        "model": f"{judge_model.provider}:{judge_model.model}",
    }
    cache_key = make_cache_key(HOLISTIC_JUDGE_PROMPT_VERSION, payload)
    raw_text = ""
    cache_hit = False
    if cache and cache.enabled:
        cache.load()
        hit = cache.get(cache_key)
        if hit and isinstance(hit.get("raw_response"), str):
            raw_text = hit["raw_response"]
            cache_hit = True
    if not raw_text:
        response = router.generate(
            judge_model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        raw_text = response.raw_text or response.text
        if cache and cache.enabled:
            cache.set(cache_key, {"raw_response": raw_text, "kind": "holistic_judge"})
    obj = extract_json_object(raw_text)
    raw_decision = ""
    confidence = ""
    distinguishing_behavior = ""
    if isinstance(obj, Mapping):
        raw_decision = str(obj.get("decision", "") or "").strip()
        confidence = str(obj.get("confidence", "") or "").strip().lower()
        distinguishing_behavior = str(obj.get("distinguishing_behavior", "") or "").strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = ""
    decision = _normalize_decision(
        raw_decision,
        left_id=left_id,
        right_id=right_id,
        left_pair_position=left_pair_position,
    )
    parse_error = "" if decision in {"A>B", "B>A", "A=B"} else "holistic_judge_invalid_decision"
    return {
        "order": order_label,
        "decision": decision,
        "raw_decision": raw_decision,
        "distinguishing_behavior": distinguishing_behavior,
        "confidence": confidence,
        "raw_response": raw_text,
        "cache_hit": cache_hit,
        "parse_error": parse_error,
    }


def _aggregate_holistic_attempts(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    decisions = [str(attempt.get("decision", "") or "").strip() for attempt in attempts]
    directional = {d for d in decisions if d in {"A>B", "B>A"}}
    if directional and "A=B" not in decisions and len(directional) == 1:
        decision = next(iter(directional))
        consistent = True
    elif directional and len(directional) == 1:
        decision = next(iter(directional))
        consistent = True
    elif directional and len(directional) == 2:
        decision = ""
        consistent = False
    else:
        decision = "A=B" if decisions and all(d == "A=B" for d in decisions) else ""
        consistent = True
    confidences = [str(attempt.get("confidence", "") or "").strip().lower() for attempt in attempts if attempt.get("confidence")]
    if consistent and decision in {"A>B", "B>A"}:
        if all(c == "high" for c in confidences) and confidences:
            confidence = "high"
        elif any(c in {"high", "medium"} for c in confidences):
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = ""
    explanation = ""
    for attempt in attempts:
        if str(attempt.get("decision", "")) == decision:
            explanation = str(attempt.get("distinguishing_behavior", "") or "").strip()
            if explanation:
                break
    return {
        "decision": decision,
        "confidence": confidence,
        "distinguishing_behavior": explanation,
        "order_consistent": bool(consistent),
        "attempts": [dict(attempt) for attempt in attempts],
        "schema": "compiled_judgebench_holistic_judge_v1",
    }


def run_holistic_pair_judge(
    *,
    example_record: ExampleRecord,
    rubrics: Sequence[RubricCriterion],
    pair_candidates: Sequence[CandidateNote],
    task_profile_id: str,
    judge_model: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
) -> Dict[str, Any]:
    if len(pair_candidates) < 2:
        return {
            "decision": "",
            "confidence": "",
            "distinguishing_behavior": "",
            "order_consistent": False,
            "attempts": [],
            "schema": "compiled_judgebench_holistic_judge_v1",
        }
    attempts = [
        _run_holistic_once(
            example_record=example_record,
            rubrics=rubrics,
            left_candidate=pair_candidates[0],
            right_candidate=pair_candidates[1],
            left_pair_position="A",
            task_profile_id=task_profile_id,
            judge_model=judge_model,
            router=router,
            cache=cache,
            order_label="AB",
        ),
        _run_holistic_once(
            example_record=example_record,
            rubrics=rubrics,
            left_candidate=pair_candidates[1],
            right_candidate=pair_candidates[0],
            left_pair_position="B",
            task_profile_id=task_profile_id,
            judge_model=judge_model,
            router=router,
            cache=cache,
            order_label="BA",
        ),
    ]
    return _aggregate_holistic_attempts(attempts)


def apply_holistic_judge_to_scoring(
    *,
    scoring: Mapping[str, Any],
    holistic: Mapping[str, Any],
    low_margin_threshold: float = 0.005,
    require_high_confidence: bool = True,
    require_order_consistent: bool = True,
) -> Dict[str, Any]:
    """
    Apply a holistic judgment to an existing WU scoring result only on the tightest ties.

    The defaults are intentionally conservative: experiments on the v2 stack found that the
    holistic judge regressed transport accuracy on JudgeBench's blind-350 by ~11 points when it
    fired on the previous (loose) gate (margin <= 0.05, medium+ confidence). The current defaults
    (margin <= 0.005, HIGH confidence required, order-consistent required) limit the holistic
    override to true ties where the LLM was strongly confident in both directional attempts.

    Callers that want the looser legacy behaviour can pass ``low_margin_threshold=0.05`` and
    ``require_high_confidence=False`` explicitly.
    """
    import copy as _copy

    updated = _copy.deepcopy(dict(scoring))
    updated["holistic_judge"] = dict(holistic)
    decision = str(holistic.get("decision", "") or "").strip()
    confidence = str(holistic.get("confidence", "") or "").strip().lower()
    allowed_confidences = {"high"} if require_high_confidence else {"medium", "high"}
    if decision not in {"A>B", "B>A"} or confidence not in allowed_confidences:
        return updated
    if require_order_consistent and not bool(holistic.get("order_consistent", False)):
        return updated
    for method_name in ("uniform", "whitened_uniform"):
        result = (updated.get(method_name) or {}).get("result") or {}
        current_decision = str(result.get("decision", "") or "").strip()
        margin = abs(float(result.get("score_A", 0.0) or 0.0) - float(result.get("score_B", 0.0) or 0.0))
        if current_decision in {"A>B", "B>A"}:
            if current_decision == decision:
                continue
            if margin > low_margin_threshold:
                continue
        if current_decision == "A=B" and not holistic.get("order_consistent", False):
            continue
        result["base_decision"] = current_decision
        result["base_decision_policy"] = str(result.get("decision_policy", "")).strip()
        result["decision_policy"] = "holistic_judge"
        result["decision"] = decision
        result["tie_break_reason"] = "holistic_judge"
        result["holistic_judge_confidence"] = confidence
        result["holistic_judge_explanation"] = str(holistic.get("distinguishing_behavior", "") or "").strip()
    return updated
