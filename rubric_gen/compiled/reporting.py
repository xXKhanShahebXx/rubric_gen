"""Lightweight aggregates for compiled pilot runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rubric_gen.compiled.schema import CaseEvaluationRecord


@dataclass
class PilotRunStats:
    """Counters for a single compiled pilot run (starter scaffold)."""

    examples_processed: int = 0
    candidates_evaluated: int = 0
    hard_gate_failures: int = 0
    synthetic_candidates: int = 0
    original_candidates: int = 0
    decision_gold_sft: int = 0
    decision_repair: int = 0
    decision_do_not_train: int = 0
    note_family_counts: Dict[str, int] = field(default_factory=dict)
    task_family_counts: Dict[str, int] = field(default_factory=dict)
    task_profile_counts: Dict[str, int] = field(default_factory=dict)
    judge_disagreements: int = 0
    llm_cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples_processed": self.examples_processed,
            "candidates_evaluated": self.candidates_evaluated,
            "hard_gate_failures": self.hard_gate_failures,
            "synthetic_candidates": self.synthetic_candidates,
            "original_candidates": self.original_candidates,
            "decision_bucket_counts": {
                "gold_sft": self.decision_gold_sft,
                "repair": self.decision_repair,
                "do_not_train": self.decision_do_not_train,
            },
            "note_family_counts": dict(sorted(self.note_family_counts.items())),
            "task_family_counts": dict(sorted(self.task_family_counts.items())),
            "task_profile_counts": dict(sorted(self.task_profile_counts.items())),
            "judge_disagreements": self.judge_disagreements,
            "llm_cache_hits": self.llm_cache_hits,
        }


def eval_hard_failed(record: CaseEvaluationRecord) -> bool:
    return any(getattr(r, "verdict", "") != "MET" for r in record.hard_gate_results)


def average_confidence(record: CaseEvaluationRecord) -> Optional[float]:
    vals: List[float] = []
    for r in record.hard_gate_results + record.soft_results:
        c = getattr(r, "confidence", None)
        if isinstance(c, (int, float)):
            vals.append(float(c))
    if not vals:
        return None
    return sum(vals) / len(vals)


def review_queue_reasons(
    *,
    heuristic_ev: Optional[CaseEvaluationRecord],
    llm_ev: Optional[CaseEvaluationRecord],
    judge_mode: str,
    low_conf_threshold: float = 0.6,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return human-readable reasons and a small detail dict for adjudication follow-up."""
    reasons: List[str] = []
    detail: Dict[str, Any] = {}

    if judge_mode in {"llm", "both"} and llm_ev is not None:
        for r in llm_ev.hard_gate_results + llm_ev.soft_results:
            if getattr(r, "verdict", "") == "CANNOT_ASSESS":
                reasons.append("llm_cannot_assess")
                break
        avg = average_confidence(llm_ev)
        detail["llm_avg_confidence"] = avg
        if avg is not None and avg < low_conf_threshold:
            reasons.append("low_llm_confidence")

    if judge_mode == "both" and heuristic_ev is not None and llm_ev is not None:
        ho = heuristic_ev.overall_decision
        lo = llm_ev.overall_decision
        hh = eval_hard_failed(heuristic_ev)
        lh = eval_hard_failed(llm_ev)
        detail["heuristic_overall"] = ho
        detail["llm_overall"] = lo
        detail["heuristic_hard_failed"] = hh
        detail["llm_hard_failed"] = lh
        if ho != lo or hh != lh:
            reasons.append("judge_disagreement")

    # De-duplicate preserving order
    seen = set()
    uniq = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq, detail
