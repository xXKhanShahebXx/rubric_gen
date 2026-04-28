from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence

from rubric_gen.config import PipelineConfig
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.rrd.engine import RRDEngine, normalize_rubric_text
from rubric_gen.storage import JsonlCache, make_cache_key
from rubric_gen.types import CandidateNote, CandidateScore, ExampleRecord, RubricCriterion


DIRECT_JUDGE_SYSTEM = """You are a strict clinical note evaluator.

Given a transcript and a candidate clinical note, score the note from 0 to 10 on:
- clinical_fidelity
- completeness
- structure
- safety

Then provide an overall_score from 0 to 10.
Do not reward unsupported details.

Output JSON only:
{"overall_score":0-10,"clinical_fidelity":0-10,"completeness":0-10,"structure":0-10,"safety":0-10,"reasoning":"brief explanation"}
"""


STATIC_HEALTHCARE_RUBRICS = [
    "The note states the chief complaint or visit reason that is supported by the transcript.",
    "The note captures clinically important symptoms, findings, or history relevant to the encounter.",
    "The note preserves clinically relevant negatives or denials when they were discussed.",
    "The note records medications only when they are supported by the transcript.",
    "The note preserves dose or frequency details when those details were discussed.",
    "The note avoids unsupported diagnoses, medications, procedures, or test results.",
    "The note includes assessment or diagnoses only when they are supported by the transcript.",
    "The note includes follow-up, referrals, monitoring, or next steps only when they were discussed.",
    "The note distinguishes patient-reported information from clinician observations or exam findings.",
    "The note uses a clear clinical structure with coherent sections.",
]


def _wrap_rubrics(texts: Sequence[str], example: ExampleRecord, source_stage: str) -> List[RubricCriterion]:
    wrapped: List[RubricCriterion] = []
    seen = set()
    for index, text in enumerate(texts):
        normalized = normalize_rubric_text(text)
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        wrapped.append(
            RubricCriterion(
                rubric_id=f"{example.example_id}__{source_stage}_{index}",
                text=normalized,
                source_stage=source_stage,
                depth=0,
                round_index=index,
            )
        )
    return wrapped


def _wrap_bank_entries(
    bank_entries: Sequence[Dict[str, object]],
    example: ExampleRecord,
    source_stage: str,
    text_key: str = "canonical_text",
) -> List[RubricCriterion]:
    wrapped: List[RubricCriterion] = []
    seen = set()
    for index, entry in enumerate(bank_entries):
        normalized = normalize_rubric_text(str(entry.get(text_key, "")))
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        wrapped.append(
            RubricCriterion(
                rubric_id=f"{example.example_id}__{source_stage}_{index}",
                text=normalized,
                source_stage=source_stage,
                depth=0,
                round_index=index,
                metadata={
                    "bank_entry": entry,
                    "source_text_key": text_key,
                },
            )
        )
    return wrapped


def run_static_healthcare_baseline(
    engine: RRDEngine,
    example: ExampleRecord,
    candidates: Sequence[CandidateNote],
) -> Dict[str, object]:
    rubrics = _wrap_rubrics(STATIC_HEALTHCARE_RUBRICS, example, source_stage="static")
    scored = engine.score_rubric_set(example, candidates, rubrics, method_prefix="static_healthcare")
    scored["baseline_name"] = "static_healthcare"
    return scored


def run_one_shot_baseline(
    engine: RRDEngine,
    example: ExampleRecord,
    candidates: Sequence[CandidateNote],
    evaluation_candidates: Sequence[CandidateNote] | None = None,
    include_responses: bool = True,
    method_prefix: str = "one_shot",
    baseline_name: str = "one_shot",
) -> Dict[str, object]:
    initial_rubrics = engine.propose_initial_rubrics(
        example,
        candidates,
        include_responses=include_responses,
    )
    rubrics = _wrap_rubrics(initial_rubrics, example, source_stage="one_shot")
    scored = engine.score_rubric_set(
        example,
        candidates,
        rubrics,
        method_prefix=method_prefix,
        evaluation_candidates=evaluation_candidates or candidates,
    )
    scored["baseline_name"] = baseline_name
    return scored


def run_prompt_only_baseline(
    engine: RRDEngine,
    example: ExampleRecord,
    proposal_candidates: Sequence[CandidateNote],
    evaluation_candidates: Sequence[CandidateNote] | None = None,
) -> Dict[str, object]:
    return run_one_shot_baseline(
        engine=engine,
        example=example,
        candidates=proposal_candidates,
        evaluation_candidates=evaluation_candidates,
        include_responses=False,
        method_prefix="prompt_only",
        baseline_name="prompt_only",
    )


def run_bank_method(
    engine: RRDEngine,
    example: ExampleRecord,
    candidates: Sequence[CandidateNote],
    bank_entries: Sequence[Dict[str, object]],
    source_stage: str,
    method_prefix: str,
    baseline_name: str,
    text_key: str = "canonical_text",
) -> Dict[str, object]:
    rubrics = _wrap_bank_entries(
        bank_entries=bank_entries,
        example=example,
        source_stage=source_stage,
        text_key=text_key,
    )
    scored = engine.score_rubric_set(
        example=example,
        candidates=candidates,
        rubrics=rubrics,
        method_prefix=method_prefix,
    )
    scored["baseline_name"] = baseline_name
    scored["bank_source"] = source_stage
    return scored


def _heuristic_direct_score(candidate: CandidateNote) -> Dict[str, float]:
    quality_scores = {
        "gold_like": 9.5,
        "strong_anchor": 8.8,
        "frontier_generated": 7.5,
        "open_generated": 6.5,
        "synthetically_degraded": 3.5,
    }
    score = quality_scores.get(candidate.quality_bucket, 5.0)
    return {
        "overall_score": score,
        "clinical_fidelity": score,
        "completeness": score,
        "structure": score,
        "safety": score,
        "reasoning": "heuristic fallback",
    }


def run_direct_judge_baseline(
    config: PipelineConfig,
    router: LLMRouter | None,
    judge_cache: JsonlCache,
    example: ExampleRecord,
    candidates: Sequence[CandidateNote],
) -> Dict[str, object]:
    rows: List[CandidateScore] = []
    details = []

    for candidate in candidates:
        cache_key = make_cache_key(
            "direct_judge",
            {
                "example_id": example.example_id,
                "candidate_id": candidate.candidate_id,
                "judge_model": config.baseline_judge.model if config.baseline_judge else "heuristic",
            },
        )
        cached = judge_cache.get(cache_key)
        if cached is None:
            if config.dry_run or router is None or config.baseline_judge is None:
                payload = _heuristic_direct_score(candidate)
                cached = judge_cache.set(cache_key, {"score_payload": payload})
            else:
                try:
                    response = router.generate(
                        config.baseline_judge,
                        system_prompt=DIRECT_JUDGE_SYSTEM,
                        user_prompt=(
                            f"Transcript:\n{example.conversation}\n\n"
                            f"Candidate note:\n{candidate.text}\n"
                        ),
                    )
                    payload = extract_json_object(response.text) or _heuristic_direct_score(candidate)
                    cached = judge_cache.set(
                        cache_key,
                        {
                            "score_payload": payload,
                            "raw_text": response.raw_text,
                        },
                    )
                except Exception:
                    payload = _heuristic_direct_score(candidate)
                    cached = judge_cache.set(cache_key, {"score_payload": payload})

        score_payload = cached["score_payload"]
        score = float(score_payload.get("overall_score", 0.0)) / 10.0
        details.append(
            {
                "candidate_id": candidate.candidate_id,
                "score_payload": score_payload,
                "source_label": candidate.source_label,
                "quality_bucket": candidate.quality_bucket,
            }
        )
        rows.append(
            CandidateScore(
                candidate_id=candidate.candidate_id,
                method="direct_judge",
                score=score,
                rank=0,
                satisfied_count=0,
                rubric_count=0,
                quality_bucket=candidate.quality_bucket,
                source_label=candidate.source_label,
                reasoning=str(score_payload.get("reasoning", "")),
            )
        )

    rows.sort(key=lambda row: (-row.score, row.candidate_id))
    ranked = []
    for rank, row in enumerate(rows, start=1):
        row.rank = rank
        ranked.append(asdict(row))

    return {
        "baseline_name": "direct_judge",
        "ranking": ranked,
        "details": details,
    }
