from __future__ import annotations

from statistics import mean
from typing import Dict, Sequence

from rubric_gen.config import PipelineConfig
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key
from rubric_gen.types import CandidateNote, ExampleRecord


BANK_JUDGE_SYSTEM = """You are an expert evaluator of rubric banks for an LLM-as-a-judge system.

You will receive:
- a clinical transcript,
- a summary of candidate notes for the same case,
- a rubric bank produced for evaluating those notes.

Score the rubric bank from 0 to 10 on:
- coverage: does it cover the important dimensions needed to distinguish note quality for this case?
- atomicity: are the rubric items single-criterion and non-stacked?
- redundancy: is the bank non-overlapping and non-repetitive? Higher means less redundant.
- directionality: are the rubrics correctly polarized and appropriately applicable to the case?
- executability: can another judge apply these rubrics consistently and concretely?
- overall_usefulness: how useful is this bank for evaluating candidate notes in this case?

Return JSON only:
{
  "coverage": 0-10,
  "atomicity": 0-10,
  "redundancy": 0-10,
  "directionality": 0-10,
  "executability": 0-10,
  "overall_usefulness": 0-10,
  "brief_reasoning": "1-3 sentences"
}
"""


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _candidate_pool_summary(candidates: Sequence[CandidateNote]) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            f"[{candidate.candidate_id} | {candidate.source_label} | {candidate.quality_bucket}]\n"
            f"{_truncate(candidate.text, 420)}"
        )
    return "\n\n".join(lines)


def _rubric_bank_text(rubrics: Sequence[Dict[str, object]]) -> str:
    lines = []
    for rubric in rubrics:
        lines.append(f"- {rubric.get('text', '')}")
    return "\n".join(lines)


def _pairwise_overlap_ratio(rubrics: Sequence[Dict[str, object]]) -> float:
    texts = [str(rubric.get("text", "")).lower().split() for rubric in rubrics]
    if len(texts) < 2:
        return 0.0
    overlaps = []
    for i in range(len(texts)):
        set_i = set(texts[i])
        for j in range(i + 1, len(texts)):
            set_j = set(texts[j])
            denom = max(1, len(set_i | set_j))
            overlaps.append(len(set_i & set_j) / denom)
    return mean(overlaps) if overlaps else 0.0


def heuristic_bank_judgment(rubrics: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rubric_count = len(rubrics)
    if rubric_count == 0:
        return {
            "coverage": 0.0,
            "atomicity": 0.0,
            "redundancy": 0.0,
            "directionality": 0.0,
            "executability": 0.0,
            "overall_usefulness": 0.0,
            "brief_reasoning": "Empty rubric bank.",
        }

    stacked_count = sum(1 for rubric in rubrics if " and " in str(rubric.get("text", "")).lower())
    overlap = _pairwise_overlap_ratio(rubrics)
    rubric_count_score = min(10.0, 4.0 + (rubric_count / 3.0))
    atomicity = max(0.0, 9.0 - stacked_count)
    redundancy = max(0.0, 10.0 - (overlap * 12.0))
    directionality = 8.0
    executability = max(
        0.0,
        8.5 - (sum(1 for rubric in rubrics if len(str(rubric.get("text", "")).split()) > 35) * 0.5),
    )
    overall = mean([rubric_count_score, atomicity, redundancy, directionality, executability])
    return {
        "coverage": round(rubric_count_score, 2),
        "atomicity": round(atomicity, 2),
        "redundancy": round(redundancy, 2),
        "directionality": round(directionality, 2),
        "executability": round(executability, 2),
        "overall_usefulness": round(overall, 2),
        "brief_reasoning": "Heuristic fallback based on rubric count, overlap, and stacked phrasing.",
    }


def judge_rubric_bank(
    config: PipelineConfig,
    router: LLMRouter | None,
    cache: JsonlCache,
    example: ExampleRecord,
    candidates: Sequence[CandidateNote],
    rubrics: Sequence[Dict[str, object]],
    proposer_label: str,
    stage: str,
) -> Dict[str, object]:
    cache_key = make_cache_key(
        "rubric_bank_judgment_v2",
        {
            "example_id": example.example_id,
            "proposer_label": proposer_label,
            "stage": stage,
            "judge_model": config.rubric_bank_judge.model if config.rubric_bank_judge else "heuristic",
            "rubrics": [rubric.get("text", "") for rubric in rubrics],
        },
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached["judgment"]

    if config.dry_run or router is None or config.rubric_bank_judge is None:
        judgment = heuristic_bank_judgment(rubrics)
        cache.set(cache_key, {"judgment": judgment})
        return judgment

    user_prompt = (
        f"Proposer label: {proposer_label}\n"
        f"Stage: {stage}\n\n"
        f"Transcript:\n{_truncate(example.conversation, 7000)}\n\n"
        f"Candidate note pool:\n{_candidate_pool_summary(candidates)}\n\n"
        f"Rubric bank:\n{_rubric_bank_text(rubrics)}\n"
    )
    try:
        response = router.generate(
            config.rubric_bank_judge,
            system_prompt=BANK_JUDGE_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=700,
        )
        payload = extract_json_object(response.text) or heuristic_bank_judgment(rubrics)
    except Exception:
        payload = heuristic_bank_judgment(rubrics)

    normalized = {
        "coverage": float(payload.get("coverage", 0.0)),
        "atomicity": float(payload.get("atomicity", 0.0)),
        "redundancy": float(payload.get("redundancy", 0.0)),
        "directionality": float(payload.get("directionality", 0.0)),
        "executability": float(payload.get("executability", 0.0)),
        "overall_usefulness": float(payload.get("overall_usefulness", 0.0)),
        "brief_reasoning": str(payload.get("brief_reasoning", "")),
    }
    cache.set(cache_key, {"judgment": normalized})
    return normalized
