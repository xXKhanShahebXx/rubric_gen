from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


_CONFIDENCE_RANK = {"": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class JudgeBenchVerifierCandidateSignal:
    extracted_value: str = ""
    extracted_source: str = ""
    explicit: bool = False
    exact_match: bool = False
    format_ok: bool = False
    consistent: bool = False
    conflicting_markers: bool = False
    option_map_available: bool = False
    marker_count: int = 0
    choice_letter: str = ""
    choice_value: str = ""
    final_line: str = ""


@dataclass(frozen=True)
class JudgeBenchVerifierOutcome:
    source_family: str = ""
    triggered: bool = False
    recommended_decision: str = ""
    confidence: str = ""
    reason: str = ""
    margin: float = 0.0
    candidate_signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)


def _signal_payload(signal: JudgeBenchVerifierCandidateSignal) -> Dict[str, Any]:
    return asdict(signal)


def _family_threshold(source_family: str) -> float:
    if source_family == "mmlu-pro":
        return 1.0
    if source_family in {"livebench-reasoning", "livebench-math"}:
        return 1.25
    return 1.5


def _candidate_score(
    source_family: str,
    signal: JudgeBenchVerifierCandidateSignal,
    *,
    features: Mapping[str, Any],
) -> float:
    score = 0.0
    if signal.exact_match:
        score += 5.0
    if signal.consistent:
        score += 2.0
    if signal.format_ok:
        score += 1.25
    if signal.explicit:
        score += 1.0
    if signal.choice_letter:
        score += 0.5
    if signal.choice_value:
        score += 0.35
    if signal.conflicting_markers:
        score -= 2.75
    if signal.marker_count > 1:
        score -= 0.5

    if source_family == "mmlu-pro":
        if signal.exact_match:
            score += 1.5
        if signal.choice_letter and signal.choice_value:
            score += 0.65
        if signal.option_map_available and not signal.explicit:
            score -= 0.5
    elif source_family == "livebench-reasoning":
        if signal.consistent:
            score += 0.75
        if signal.explicit:
            score += 0.5
        if not signal.explicit and not signal.extracted_value:
            score -= 0.5
    elif source_family == "livebench-math":
        if signal.format_ok:
            score += 0.5
        if signal.extracted_value:
            score += 0.35

    if not bool(features.get("exact_answer_task")) and not signal.consistent:
        score -= 0.25
    return score


def _decision_confidence(
    source_family: str,
    signal_a: JudgeBenchVerifierCandidateSignal,
    signal_b: JudgeBenchVerifierCandidateSignal,
    *,
    margin: float,
) -> str:
    if signal_a.exact_match != signal_b.exact_match:
        return "high"
    if signal_a.consistent != signal_b.consistent and margin >= 1.5:
        return "high"
    if margin >= 3.0:
        return "high"
    if margin >= 1.75:
        return "medium"
    if source_family == "mmlu-pro" and margin >= 1.0:
        return "medium"
    return "low"


def _decision_reason(
    source_family: str,
    better_label: str,
    better: JudgeBenchVerifierCandidateSignal,
    weaker: JudgeBenchVerifierCandidateSignal,
) -> str:
    if better.exact_match != weaker.exact_match:
        return f"{better_label.lower()}_exact_answer_match"
    if better.consistent != weaker.consistent:
        return f"{better_label.lower()}_answer_consistency"
    if better.conflicting_markers != weaker.conflicting_markers:
        return f"{better_label.lower()}_marker_conflict"
    if better.format_ok != weaker.format_ok:
        return f"{better_label.lower()}_answer_format"
    if source_family == "livebench-reasoning":
        return f"{better_label.lower()}_reasoning_state_tracking"
    if source_family == "mmlu-pro":
        return f"{better_label.lower()}_choice_alignment"
    return f"{better_label.lower()}_verifier_margin"


def evaluate_pair_verifier(
    *,
    source_family: str,
    features: Mapping[str, Any],
    signal_a: JudgeBenchVerifierCandidateSignal,
    signal_b: JudgeBenchVerifierCandidateSignal,
) -> JudgeBenchVerifierOutcome:
    score_a = _candidate_score(source_family, signal_a, features=features)
    score_b = _candidate_score(source_family, signal_b, features=features)
    margin = round(abs(score_a - score_b), 6)
    threshold = _family_threshold(source_family)
    if margin < threshold:
        return JudgeBenchVerifierOutcome(
            source_family=source_family,
            triggered=False,
            recommended_decision="",
            confidence="low" if margin > 0.0 else "",
            reason="insufficient_verifier_margin",
            margin=margin,
            candidate_signals={"A": _signal_payload(signal_a), "B": _signal_payload(signal_b)},
            features=dict(features),
        )

    decision = "A>B" if score_a > score_b else "B>A"
    better = signal_a if decision == "A>B" else signal_b
    weaker = signal_b if decision == "A>B" else signal_a
    confidence = _decision_confidence(
        source_family,
        signal_a,
        signal_b,
        margin=margin,
    )
    return JudgeBenchVerifierOutcome(
        source_family=source_family,
        triggered=_CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK["medium"],
        recommended_decision=decision if confidence in {"medium", "high"} else "",
        confidence=confidence,
        reason=_decision_reason(source_family, "A" if decision == "A>B" else "B", better, weaker),
        margin=margin,
        candidate_signals={"A": _signal_payload(signal_a), "B": _signal_payload(signal_b)},
        features=dict(features),
    )
