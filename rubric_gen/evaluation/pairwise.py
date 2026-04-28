from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence

from rubric_gen.config import PipelineConfig
from rubric_gen.dataio import ExampleRecord
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key


PAIRWISE_JUDGE_SYSTEM = """You are an expert judge comparing two candidate clinical notes for the same case.

Given the task context and two notes, choose which note is better overall.
Focus on:
- clinical fidelity
- completeness
- safety
- organization
- avoidance of unsupported content

Do not choose based on style alone if the weaker note is less faithful or less complete.
Do not output a tie. Choose exactly one.

Output JSON only:
{"preferred":"A or B","reasoning":"brief explanation"}
"""


def build_proxy_pairwise_preferences(
    candidates: Sequence[Dict[str, object]],
    label_mode: str = "reference_proxy",
) -> List[Dict[str, object]]:
    if label_mode != "reference_proxy":
        return []

    preferred_candidate = None
    for candidate in candidates:
        if candidate.get("source_label") == "reference_note":
            preferred_candidate = candidate
            break
    if preferred_candidate is None:
        for candidate in candidates:
            if candidate.get("source_label") == "augmented_note":
                preferred_candidate = candidate
                break
    if preferred_candidate is None:
        return []

    pairs: List[Dict[str, object]] = []
    for candidate in candidates:
        if candidate["candidate_id"] == preferred_candidate["candidate_id"]:
            continue
        pairs.append(
            {
                "preferred_id": preferred_candidate["candidate_id"],
                "other_id": candidate["candidate_id"],
                "label_source": preferred_candidate["source_label"],
            }
        )
    return pairs


def build_generated_note_pairs(candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    generated = [
        candidate
        for candidate in candidates
        if candidate.get("origin_kind") == "generated"
    ]
    pairs: List[Dict[str, object]] = []
    for left, right in combinations(generated, 2):
        pairs.append(
            {
                "candidate_a_id": left["candidate_id"],
                "candidate_b_id": right["candidate_id"],
                "label_source": "judge_proxy",
            }
        )
    return pairs


def _parse_preferred_choice(payload: Dict[str, object]) -> str | None:
    preferred = str(payload.get("preferred", "")).strip().upper()
    if preferred in {"A", "B"}:
        return preferred
    return None


def build_judge_pairwise_preferences(
    config: PipelineConfig,
    router: LLMRouter | None,
    cache: JsonlCache,
    example: ExampleRecord,
    candidates: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    if config.dry_run or router is None or config.paper_pairwise_judge is None:
        return []

    candidate_lookup = {candidate["candidate_id"]: candidate for candidate in candidates}
    pairs = build_generated_note_pairs(candidates)
    judged_pairs: List[Dict[str, object]] = []
    for pair in pairs:
        candidate_a = candidate_lookup[pair["candidate_a_id"]]
        candidate_b = candidate_lookup[pair["candidate_b_id"]]
        cache_key = make_cache_key(
            "paper_pairwise_judge",
            {
                "example_id": example.example_id,
                "judge_model": config.paper_pairwise_judge.model,
                "candidate_a_id": candidate_a["candidate_id"],
                "candidate_b_id": candidate_b["candidate_id"],
                "candidate_a_text": candidate_a["text"],
                "candidate_b_text": candidate_b["text"],
            },
        )
        cached = cache.get(cache_key)
        if cached is None:
            user_prompt = (
                f"Task prompt:\n{example.task_prompt}\n\n"
                f"Task instance / transcript:\n{example.conversation}\n\n"
                f"Candidate A:\n{candidate_a['text']}\n\n"
                f"Candidate B:\n{candidate_b['text']}\n"
            )
            try:
                response = router.generate(
                    config.paper_pairwise_judge,
                    system_prompt=PAIRWISE_JUDGE_SYSTEM,
                    user_prompt=user_prompt,
                    max_tokens=500,
                )
                payload = extract_json_object(response.text) or {}
            except Exception:
                payload = {}
            preferred = _parse_preferred_choice(payload) or "A"
            cached = cache.set(
                cache_key,
                {
                    "pair_label": {
                        "candidate_a_id": candidate_a["candidate_id"],
                        "candidate_b_id": candidate_b["candidate_id"],
                        "preferred": preferred,
                        "reasoning": str(payload.get("reasoning", "")),
                        "label_source": "judge_proxy",
                    }
                },
            )
        judged_pairs.append(cached["pair_label"])
    return judged_pairs


def evaluate_pairwise_preferences(
    ranking: List[Dict[str, object]],
    pairs: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    lookup = {row["candidate_id"]: row for row in ranking}
    total = 0
    correct = 0
    margins: List[float] = []
    for pair in pairs:
        if "preferred_id" in pair:
            preferred = lookup.get(pair["preferred_id"])
            other = lookup.get(pair["other_id"])
            expected_preferred_score = lambda preferred_row, other_row: float(preferred_row.get("score", 0.0)) > float(other_row.get("score", 0.0))
            preferred_score = float(preferred.get("score", 0.0)) if preferred else 0.0
            other_score = float(other.get("score", 0.0)) if other else 0.0
        else:
            candidate_a = lookup.get(pair["candidate_a_id"])
            candidate_b = lookup.get(pair["candidate_b_id"])
            preferred = candidate_a
            other = candidate_b
            if pair.get("preferred") == "B":
                preferred, other = candidate_b, candidate_a
            expected_preferred_score = lambda preferred_row, other_row: float(preferred_row.get("score", 0.0)) > float(other_row.get("score", 0.0))
            preferred_score = float(preferred.get("score", 0.0)) if preferred else 0.0
            other_score = float(other.get("score", 0.0)) if other else 0.0
        if preferred is None or other is None:
            continue
        total += 1
        if expected_preferred_score(preferred, other):
            correct += 1
        margins.append(preferred_score - other_score)
    return {
        "pair_count": float(total),
        "pairwise_accuracy": (float(correct) / total) if total else 0.0,
        "mean_margin": (sum(margins) / len(margins)) if margins else 0.0,
    }
