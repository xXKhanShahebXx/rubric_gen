"""
Utility to compare JudgeBench train-only runs with stricter freeze-selection gates.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_TRAIN_ONLY_SUMMARY_SCHEMA = "compiled_judgebench_train_only_summary_v1"
_SPLIT_SUMMARY_SCHEMA = "compiled_judgebench_split_summary_v1"
_FAILURE_ANALYSIS_SCHEMA = "compiled_judgebench_failure_analysis_v1"

_DEFAULT_MIN_FAMILY_WU = 65.0
_DEFAULT_MAX_FAILURE_COUNT = 22
_DEFAULT_MAX_EXACT_ANSWER_FAILURES = 20
_DEFAULT_MAX_TIE_FAILURES = 4
_DEFAULT_MAX_WU_UNIFORM_GAP = 6.0
_DEFAULT_BLIND_GATE_MIN_MEAN_OOF_WU = 80.0
_DEFAULT_BLIND_GATE_MIN_TARGET_MIX_WU = 78.0
_DEFAULT_BLIND_GATE_MIN_WORST_FAMILY_WU = 70.0
_DEFAULT_BLIND_GATE_MIN_FOCUS_FAMILY_WU = 75.0
_DEFAULT_BLIND_GATE_MAX_FAILURE_RATE = 0.16
_DEFAULT_BLIND_GATE_MAX_EXACT_ANSWER_FAILURE_RATE = 0.10
_DEFAULT_BLIND_GATE_MAX_TIE_FAILURE_RATE = 0.08
_DEFAULT_BLIND_GATE_MAX_LOCKED_POLICY_WU_GAP = 8.0
_DEFAULT_BLIND_GATE_MIN_FOCUS_VERIFIER_COVERAGE = 0.75
_DEFAULT_BLIND_GATE_MIN_LOW_CONFIDENCE_ACCURACY = 0.65
_DEFAULT_BLIND_GATE_MIN_EXACT_PARSER_SUCCESS_RATE = 0.65

# v2 promotion gates. These are stricter than the v1 gates above and are required alongside
# external-slice transport signals before a candidate can spend a blind-350 shot. They are checked
# in :func:`evaluate_v2_promotion_gates`; the baseline gates in this file remain unchanged.
_V2_GATE_MIN_OVERALL_WU = 86.0
_V2_GATE_MIN_FAMILY_WU_MMLU_PRO = 82.0
_V2_GATE_MIN_FAMILY_WU_REASONING = 82.0
_V2_GATE_MIN_FAMILY_WU_MATH = 82.0
_V2_GATE_MAX_TIE_FAILURE_RATE = 0.05
_V2_GATE_MAX_EXACT_ANSWER_FAILURE_RATE = 0.06
_V2_GATE_MIN_REASONING_VERIFIER_TRIGGER_RATE = 0.10
_V2_GATE_MAX_DISCRIMINATOR_ORDER_DISAGREEMENT_RATE = 0.05
_V2_GATE_MIN_EXTERNAL_SLICE_WU = 75.0

_RECOVERY_SWEEP_FAMILIES = ("mmlu-pro", "livebench-reasoning")


@dataclass(frozen=True)
class _RunRecord:
    label: str
    run_dir: str
    pair_count: int
    overall_wu: float
    overall_uniform: float
    family_wu: Dict[str, float]
    worst_family_wu: float
    worst_family_name: str
    failure_count: int
    failure_rate: float
    exact_answer_failures: int
    exact_answer_failure_rate: float
    tie_failures: int
    tie_failure_rate: float
    fold_overall_stddev: float
    train_fit_available: bool
    train_fit_pair_count: int
    train_fit_overall_wu: float
    train_fit_overall_uniform: float
    train_fit_family_wu: Dict[str, float]
    train_fit_worst_family_wu: float
    train_fit_worst_family_name: str
    train_fit_failure_count: int
    train_fit_failure_rate: float
    train_fit_exact_answer_failures: int
    train_fit_exact_answer_failure_rate: float
    train_fit_tie_failures: int
    train_fit_tie_failure_rate: float
    locked_policy_gap_available: bool
    locked_policy_export_strategy: str
    locked_policy_train_oof_wu_gap: float
    locked_policy_train_oof_uniform_gap: float
    locked_policy_max_family_gap: float
    locked_policy_max_family_gap_name: str
    blind_parity_bootstrap: bool
    verifier_coverage_rate: float
    focus_verifier_coverage_rate: float
    low_confidence_bucket_accuracy: float
    exact_answer_parser_success_rate: float
    discriminator_usage_rate: float


def _summary_path_for_candidate(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        summaries_dir = candidate / "summaries"
        summary_file = summaries_dir / "summary.json"
        if summary_file.exists():
            return summary_file
        oof_file = summaries_dir / "oof_summary.json"
        if oof_file.exists():
            return oof_file
        return summary_file
    return candidate


def _load_split_summary_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    """
    Load an OOF-style ``compiled_judgebench_split_summary_v1`` summary directly from
    ``summaries/oof_summary.json`` and adapt it into the audit's expected ``oof_summary`` /
    ``failure_analysis`` shape by pulling the sibling failure-analysis artifact. Used when a run
    predates the top-level ``summary.json`` schema or when only OOF summaries are present.
    """
    summaries_dir = run_dir / "summaries"
    oof_summary_path = summaries_dir / "oof_summary.json"
    oof_failure_path = summaries_dir / "oof_failure_analysis.json"
    train_fit_summary_path = summaries_dir / "train_fit_summary.json"
    train_fit_failure_path = summaries_dir / "train_fit_failure_analysis.json"
    locked_alignment_path = summaries_dir / "locked_policy_alignment.json"
    external_slice_path = summaries_dir / "external_slice_summary.json"
    if not oof_summary_path.exists():
        raise FileNotFoundError(f"oof_summary.json missing at {summaries_dir}")
    oof_summary = json.loads(oof_summary_path.read_text(encoding="utf-8"))
    if oof_summary.get("schema") != _SPLIT_SUMMARY_SCHEMA:
        raise ValueError(
            f"Unsupported OOF summary schema at {oof_summary_path}: {oof_summary.get('schema')!r}"
        )
    failure_analysis = (
        json.loads(oof_failure_path.read_text(encoding="utf-8")) if oof_failure_path.exists() else {}
    )
    synthetic = {
        "schema": _TRAIN_ONLY_SUMMARY_SCHEMA,
        "source": "adapted_from_oof_summary",
        "oof_summary": oof_summary,
        "failure_analysis": failure_analysis,
        "mechanism_hash": str(oof_summary.get("policy_hash", "") or ""),
    }
    if train_fit_summary_path.exists():
        synthetic["train_fit_summary"] = json.loads(train_fit_summary_path.read_text(encoding="utf-8"))
    if train_fit_failure_path.exists():
        synthetic["train_fit_failure_analysis"] = json.loads(train_fit_failure_path.read_text(encoding="utf-8"))
    if locked_alignment_path.exists():
        synthetic["locked_policy_alignment"] = json.loads(locked_alignment_path.read_text(encoding="utf-8"))
    if external_slice_path.exists():
        synthetic["external_slice_summary"] = json.loads(external_slice_path.read_text(encoding="utf-8"))
    elif "external_slice_summary" in oof_summary:
        synthetic["external_slice_summary"] = oof_summary["external_slice_summary"]
    return synthetic


def _load_summary(path: Path) -> Dict[str, Any]:
    summary_path = _summary_path_for_candidate(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema == _TRAIN_ONLY_SUMMARY_SCHEMA:
        return payload
    if schema == _SPLIT_SUMMARY_SCHEMA:
        run_dir = summary_path.parent.parent
        return _load_split_summary_from_run_dir(run_dir)
    raise ValueError(f"Unsupported summary schema at {summary_path}: {schema!r}")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pstdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _iter_family_wu_metrics(summary: Mapping[str, Any]) -> Iterable[Tuple[str, float]]:
    wu_metrics = dict((summary.get("wu_metrics", {}) or {}))
    for name, value in wu_metrics.items():
        if name == "overall":
            continue
        yield str(name), _safe_float(value)


def _empty_split_metrics() -> Dict[str, Any]:
    return {
        "pair_count": 0,
        "overall_wu": 0.0,
        "overall_uniform": 0.0,
        "family_wu": {},
        "worst_family_wu": 0.0,
        "worst_family_name": "",
        "failure_count": 0,
        "failure_rate": 0.0,
        "exact_answer_failures": 0,
        "exact_answer_failure_rate": 0.0,
        "tie_failures": 0,
        "tie_failure_rate": 0.0,
    }


def _build_split_metrics(
    split_summary: Optional[Mapping[str, Any]],
    failure_analysis: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    split_summary = dict(split_summary or {})
    failure_analysis = dict(failure_analysis or {})
    family_wu = dict(_iter_family_wu_metrics(split_summary))
    if family_wu:
        worst_family_name, worst_family_wu = min(family_wu.items(), key=lambda item: item[1])
    else:
        worst_family_name, worst_family_wu = "", 0.0
    pair_count = max(
        1,
        _safe_int(split_summary.get("pair_count")),
        _safe_int(failure_analysis.get("pair_count")),
    )
    failure_count = _safe_int(failure_analysis.get("failure_count"))
    exact_answer_failures = _safe_int(failure_analysis.get("exact_answer_failures"))
    tie_failures = _safe_int(failure_analysis.get("tie_failures"))
    failure_rate = _safe_float(failure_analysis.get("failure_rate")) or (failure_count / pair_count)
    return {
        "pair_count": pair_count,
        "overall_wu": _safe_float((split_summary.get("wu_metrics", {}) or {}).get("overall")),
        "overall_uniform": _safe_float((split_summary.get("uniform_metrics", {}) or {}).get("overall")),
        "family_wu": family_wu,
        "worst_family_wu": worst_family_wu,
        "worst_family_name": worst_family_name,
        "failure_count": failure_count,
        "failure_rate": failure_rate,
        "exact_answer_failures": exact_answer_failures,
        "exact_answer_failure_rate": exact_answer_failures / pair_count,
        "tie_failures": tie_failures,
        "tie_failure_rate": tie_failures / pair_count,
    }


def _normalize_family_weights(weights: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for raw_name, raw_weight in dict(weights or {}).items():
        name = str(raw_name).strip()
        weight = _safe_float(raw_weight)
        if not name or weight <= 0.0:
            continue
        normalized[name] = weight
    return dict(sorted(normalized.items()))


def _weighted_family_wu(
    family_metrics: Mapping[str, float],
    *,
    weights: Mapping[str, float],
    default_value: float,
) -> float:
    normalized_weights = _normalize_family_weights(weights)
    if not normalized_weights:
        return default_value
    weighted_total = 0.0
    weight_total = 0.0
    for family, weight in normalized_weights.items():
        if family not in family_metrics:
            continue
        weighted_total += _safe_float(family_metrics.get(family)) * weight
        weight_total += weight
    if weight_total <= 0.0:
        return default_value
    return weighted_total / weight_total


def _locked_policy_gap_metrics(
    *,
    summary: Mapping[str, Any],
    oof_metrics: Mapping[str, Any],
    train_fit_metrics: Mapping[str, Any],
    train_fit_available: bool,
) -> Dict[str, Any]:
    alignment = dict(summary.get("locked_policy_alignment", {}) or {})
    export_strategy = str(alignment.get("export_strategy", "")).strip() or "unknown"
    train_oof_wu_gap = alignment.get("locked_train_fit_minus_oof_wu")
    train_oof_uniform_gap = alignment.get("locked_train_fit_minus_oof_uniform")
    family_wu_gap = dict((alignment.get("family_wu_gap", {}) or {}))
    if train_fit_available:
        if train_oof_wu_gap is None:
            train_oof_wu_gap = _safe_float(train_fit_metrics.get("overall_wu")) - _safe_float(oof_metrics.get("overall_wu"))
        if train_oof_uniform_gap is None:
            train_oof_uniform_gap = _safe_float(train_fit_metrics.get("overall_uniform")) - _safe_float(
                oof_metrics.get("overall_uniform")
            )
        if not family_wu_gap:
            train_fit_family_wu = dict(train_fit_metrics.get("family_wu", {}) or {})
            oof_family_wu = dict(oof_metrics.get("family_wu", {}) or {})
            for family_name in sorted(set(train_fit_family_wu) & set(oof_family_wu)):
                family_wu_gap[family_name] = round(
                    _safe_float(train_fit_family_wu.get(family_name)) - _safe_float(oof_family_wu.get(family_name)),
                    6,
                )
    max_family_gap_name = ""
    max_family_gap = 0.0
    if family_wu_gap:
        max_family_gap_name, max_family_gap = max(family_wu_gap.items(), key=lambda item: item[1])
    return {
        "available": bool(train_fit_available or alignment),
        "export_strategy": export_strategy,
        "train_oof_wu_gap": _safe_float(train_oof_wu_gap),
        "train_oof_uniform_gap": _safe_float(train_oof_uniform_gap),
        "max_family_gap": _safe_float(max_family_gap),
        "max_family_gap_name": str(max_family_gap_name),
    }


def _calibration_metrics(split_summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = dict(((split_summary or {}).get("calibration_metrics", {}) or {}))
    return {
        "verifier_coverage_rate": _safe_float(payload.get("verifier_coverage_rate")),
        "focus_verifier_coverage_rate": _safe_float(payload.get("focus_verifier_coverage_rate")),
        "low_confidence_bucket_accuracy": _safe_float(payload.get("low_confidence_bucket_accuracy")),
        "exact_answer_parser_success_rate": _safe_float(payload.get("exact_answer_parser_success_rate")),
        "discriminator_usage_rate": _safe_float(payload.get("discriminator_usage_rate")),
    }


def _build_run_record(label: str, summary: Mapping[str, Any]) -> _RunRecord:
    oof_metrics = _build_split_metrics(summary.get("oof_summary"), summary.get("failure_analysis"))
    fold_overalls = [
        _safe_float(((fold.get("wu_metrics", {}) or {}).get("overall")))
        for fold in list(summary.get("fold_summaries", []) or [])
    ]
    train_fit_summary = summary.get("train_fit_summary")
    train_fit_failure_analysis = summary.get("train_fit_failure_analysis")
    train_fit_available = bool(train_fit_summary)
    train_fit_metrics = (
        _build_split_metrics(train_fit_summary, train_fit_failure_analysis)
        if train_fit_available
        else _empty_split_metrics()
    )
    locked_policy_gap_metrics = _locked_policy_gap_metrics(
        summary=summary,
        oof_metrics=oof_metrics,
        train_fit_metrics=train_fit_metrics,
        train_fit_available=train_fit_available,
    )
    calibration_metrics = _calibration_metrics(summary.get("oof_summary"))
    return _RunRecord(
        label=label,
        run_dir=str(summary.get("run_dir", "")),
        pair_count=int(oof_metrics["pair_count"]),
        overall_wu=float(oof_metrics["overall_wu"]),
        overall_uniform=float(oof_metrics["overall_uniform"]),
        family_wu=dict(oof_metrics["family_wu"]),
        worst_family_wu=float(oof_metrics["worst_family_wu"]),
        worst_family_name=str(oof_metrics["worst_family_name"]),
        failure_count=int(oof_metrics["failure_count"]),
        failure_rate=float(oof_metrics["failure_rate"]),
        exact_answer_failures=int(oof_metrics["exact_answer_failures"]),
        exact_answer_failure_rate=float(oof_metrics["exact_answer_failure_rate"]),
        tie_failures=int(oof_metrics["tie_failures"]),
        tie_failure_rate=float(oof_metrics["tie_failure_rate"]),
        fold_overall_stddev=_pstdev(fold_overalls),
        train_fit_available=train_fit_available,
        train_fit_pair_count=int(train_fit_metrics["pair_count"]),
        train_fit_overall_wu=float(train_fit_metrics["overall_wu"]),
        train_fit_overall_uniform=float(train_fit_metrics["overall_uniform"]),
        train_fit_family_wu=dict(train_fit_metrics["family_wu"]),
        train_fit_worst_family_wu=float(train_fit_metrics["worst_family_wu"]),
        train_fit_worst_family_name=str(train_fit_metrics["worst_family_name"]),
        train_fit_failure_count=int(train_fit_metrics["failure_count"]),
        train_fit_failure_rate=float(train_fit_metrics["failure_rate"]),
        train_fit_exact_answer_failures=int(train_fit_metrics["exact_answer_failures"]),
        train_fit_exact_answer_failure_rate=float(train_fit_metrics["exact_answer_failure_rate"]),
        train_fit_tie_failures=int(train_fit_metrics["tie_failures"]),
        train_fit_tie_failure_rate=float(train_fit_metrics["tie_failure_rate"]),
        locked_policy_gap_available=bool(locked_policy_gap_metrics["available"]),
        locked_policy_export_strategy=str(locked_policy_gap_metrics["export_strategy"]),
        locked_policy_train_oof_wu_gap=float(locked_policy_gap_metrics["train_oof_wu_gap"]),
        locked_policy_train_oof_uniform_gap=float(locked_policy_gap_metrics["train_oof_uniform_gap"]),
        locked_policy_max_family_gap=float(locked_policy_gap_metrics["max_family_gap"]),
        locked_policy_max_family_gap_name=str(locked_policy_gap_metrics["max_family_gap_name"]),
        blind_parity_bootstrap=bool(summary.get("blind_parity_bootstrap", summary.get("train_reference_answer_access") is False)),
        verifier_coverage_rate=float(calibration_metrics["verifier_coverage_rate"]),
        focus_verifier_coverage_rate=float(calibration_metrics["focus_verifier_coverage_rate"]),
        low_confidence_bucket_accuracy=float(calibration_metrics["low_confidence_bucket_accuracy"]),
        exact_answer_parser_success_rate=float(calibration_metrics["exact_answer_parser_success_rate"]),
        discriminator_usage_rate=float(calibration_metrics["discriminator_usage_rate"]),
    )


def _selection_penalty(
    *,
    mean_fold_stddev: float,
    run_wu_stdev: float,
    min_family_wu: float,
    max_failure_count: int,
    max_exact_answer_failures: int,
    max_tie_failures: int,
    max_failure_rate: float,
    max_exact_answer_failure_rate: float,
    max_tie_failure_rate: float,
    mean_wu_uniform_gap: float,
    min_family_threshold: float,
    max_failure_threshold: Optional[int],
    max_exact_answer_failure_threshold: Optional[int],
    max_tie_threshold: Optional[int],
    max_failure_rate_threshold: Optional[float],
    max_exact_answer_failure_rate_threshold: Optional[float],
    max_tie_rate_threshold: Optional[float],
    max_gap_threshold: float,
) -> float:
    penalty = 0.0
    penalty += max(0.0, min_family_threshold - min_family_wu) * 2.0
    penalty += max(0.0, mean_wu_uniform_gap - max_gap_threshold) * 1.0
    if max_failure_threshold is not None:
        penalty += max(0, max_failure_count - max_failure_threshold) * 0.50
    if max_exact_answer_failure_threshold is not None:
        penalty += max(0, max_exact_answer_failures - max_exact_answer_failure_threshold) * 0.35
    if max_tie_threshold is not None:
        penalty += max(0, max_tie_failures - max_tie_threshold) * 0.50
    if max_failure_rate_threshold is not None:
        penalty += max(0.0, (max_failure_rate - max_failure_rate_threshold) * 100.0) * 0.50
    if max_exact_answer_failure_rate_threshold is not None:
        penalty += max(0.0, (max_exact_answer_failure_rate - max_exact_answer_failure_rate_threshold) * 100.0) * 0.35
    if max_tie_rate_threshold is not None:
        penalty += max(0.0, (max_tie_failure_rate - max_tie_rate_threshold) * 100.0) * 0.50
    penalty += mean_fold_stddev * 0.25
    penalty += run_wu_stdev * 0.50
    return penalty


def _aggregate_records(
    *,
    label: str,
    records: Sequence[_RunRecord],
    target_family_weights: Mapping[str, float],
    hard_family_weights: Mapping[str, float],
    min_family_threshold: float,
    max_failure_threshold: Optional[int],
    max_exact_answer_failure_threshold: Optional[int],
    max_tie_threshold: Optional[int],
    max_failure_rate_threshold: Optional[float],
    max_exact_answer_failure_rate_threshold: Optional[float],
    max_tie_rate_threshold: Optional[float],
    max_gap_threshold: float,
) -> Dict[str, Any]:
    if not records:
        raise ValueError(f"No run records provided for {label!r}.")
    train_fit_available = all(record.train_fit_available for record in records)
    locked_policy_gap_available = all(record.locked_policy_gap_available for record in records)
    blind_parity_bootstrap = all(record.blind_parity_bootstrap for record in records)
    overall_values = [record.overall_wu for record in records]
    uniform_gaps = [record.overall_wu - record.overall_uniform for record in records]
    target_mix_values = [
        _weighted_family_wu(record.family_wu, weights=target_family_weights, default_value=record.overall_wu)
        for record in records
    ]
    hard_proxy_values = [
        _weighted_family_wu(record.family_wu, weights=hard_family_weights, default_value=record.overall_wu)
        for record in records
    ]
    worst_family_record = min(records, key=lambda record: record.worst_family_wu)
    max_failure_count = max(record.failure_count for record in records)
    max_exact_answer_failures = max(record.exact_answer_failures for record in records)
    max_tie_failures = max(record.tie_failures for record in records)
    max_failure_rate = max(record.failure_rate for record in records)
    max_exact_answer_failure_rate = max(record.exact_answer_failure_rate for record in records)
    max_tie_failure_rate = max(record.tie_failure_rate for record in records)
    mean_fold_stddev = statistics.fmean(record.fold_overall_stddev for record in records)
    run_wu_stdev = _pstdev(overall_values)
    mean_overall_wu = statistics.fmean(overall_values)
    mean_overall_uniform = statistics.fmean(record.overall_uniform for record in records)
    mean_target_mix_wu = statistics.fmean(target_mix_values)
    mean_hard_proxy_wu = statistics.fmean(hard_proxy_values)
    mean_wu_uniform_gap = statistics.fmean(uniform_gaps)
    mean_family_wu: Dict[str, float] = {}
    for family in sorted({name for record in records for name in record.family_wu}):
        mean_family_wu[family] = round(
            statistics.fmean(_safe_float(record.family_wu.get(family)) for record in records),
            6,
        )
    mean_train_fit_wu: Optional[float] = None
    mean_train_fit_uniform: Optional[float] = None
    mean_train_fit_wu_uniform_gap: Optional[float] = None
    mean_train_fit_family_wu: Dict[str, float] = {}
    train_fit_worst_family_record: Optional[_RunRecord] = None
    max_train_fit_failure_count: Optional[int] = None
    max_train_fit_failure_rate: Optional[float] = None
    max_train_fit_exact_answer_failures: Optional[int] = None
    max_train_fit_exact_answer_failure_rate: Optional[float] = None
    max_train_fit_tie_failures: Optional[int] = None
    max_train_fit_tie_failure_rate: Optional[float] = None
    mean_locked_policy_train_oof_wu_gap: Optional[float] = None
    mean_locked_policy_train_oof_uniform_gap: Optional[float] = None
    mean_locked_policy_max_family_gap: Optional[float] = None
    locked_policy_max_family_gap_record: Optional[_RunRecord] = None
    locked_policy_export_strategy = ""
    mean_verifier_coverage_rate = statistics.fmean(record.verifier_coverage_rate for record in records)
    mean_focus_verifier_coverage_rate = statistics.fmean(record.focus_verifier_coverage_rate for record in records)
    mean_low_confidence_bucket_accuracy = statistics.fmean(record.low_confidence_bucket_accuracy for record in records)
    mean_exact_answer_parser_success_rate = statistics.fmean(
        record.exact_answer_parser_success_rate for record in records
    )
    mean_discriminator_usage_rate = statistics.fmean(record.discriminator_usage_rate for record in records)
    if train_fit_available:
        train_fit_overall_values = [record.train_fit_overall_wu for record in records]
        train_fit_uniform_gaps = [record.train_fit_overall_wu - record.train_fit_overall_uniform for record in records]
        train_fit_worst_family_record = min(records, key=lambda record: record.train_fit_worst_family_wu)
        mean_train_fit_wu = statistics.fmean(train_fit_overall_values)
        mean_train_fit_uniform = statistics.fmean(record.train_fit_overall_uniform for record in records)
        mean_train_fit_wu_uniform_gap = statistics.fmean(train_fit_uniform_gaps)
        for family in sorted({name for record in records for name in record.train_fit_family_wu}):
            mean_train_fit_family_wu[family] = round(
                statistics.fmean(_safe_float(record.train_fit_family_wu.get(family)) for record in records),
                6,
            )
        max_train_fit_failure_count = max(record.train_fit_failure_count for record in records)
        max_train_fit_failure_rate = max(record.train_fit_failure_rate for record in records)
        max_train_fit_exact_answer_failures = max(record.train_fit_exact_answer_failures for record in records)
        max_train_fit_exact_answer_failure_rate = max(
            record.train_fit_exact_answer_failure_rate for record in records
        )
        max_train_fit_tie_failures = max(record.train_fit_tie_failures for record in records)
        max_train_fit_tie_failure_rate = max(record.train_fit_tie_failure_rate for record in records)
    if locked_policy_gap_available:
        mean_locked_policy_train_oof_wu_gap = statistics.fmean(
            record.locked_policy_train_oof_wu_gap for record in records
        )
        mean_locked_policy_train_oof_uniform_gap = statistics.fmean(
            record.locked_policy_train_oof_uniform_gap for record in records
        )
        mean_locked_policy_max_family_gap = statistics.fmean(record.locked_policy_max_family_gap for record in records)
        locked_policy_max_family_gap_record = max(records, key=lambda record: record.locked_policy_max_family_gap)
        export_strategies = {
            record.locked_policy_export_strategy
            for record in records
            if str(record.locked_policy_export_strategy).strip()
        }
        if len(export_strategies) == 1:
            locked_policy_export_strategy = next(iter(export_strategies))
    gate_results: Dict[str, bool] = {
        "min_family_wu": worst_family_record.worst_family_wu >= min_family_threshold,
        "max_wu_uniform_gap": mean_wu_uniform_gap <= max_gap_threshold,
        "blind_parity_bootstrap": blind_parity_bootstrap,
        "min_focus_verifier_coverage": (
            mean_focus_verifier_coverage_rate >= _DEFAULT_BLIND_GATE_MIN_FOCUS_VERIFIER_COVERAGE
        ),
        "min_exact_answer_parser_success_rate": (
            mean_exact_answer_parser_success_rate >= _DEFAULT_BLIND_GATE_MIN_EXACT_PARSER_SUCCESS_RATE
        ),
        "min_low_confidence_bucket_accuracy": (
            mean_low_confidence_bucket_accuracy >= _DEFAULT_BLIND_GATE_MIN_LOW_CONFIDENCE_ACCURACY
        ),
    }
    if train_fit_available and train_fit_worst_family_record is not None and mean_train_fit_wu_uniform_gap is not None:
        gate_results["min_train_fit_family_wu"] = (
            train_fit_worst_family_record.train_fit_worst_family_wu >= min_family_threshold
        )
        gate_results["max_train_fit_wu_uniform_gap"] = mean_train_fit_wu_uniform_gap <= max_gap_threshold
    if max_failure_threshold is not None:
        gate_results["max_failure_count"] = max_failure_count <= max_failure_threshold
    if max_exact_answer_failure_threshold is not None:
        gate_results["max_exact_answer_failures"] = max_exact_answer_failures <= max_exact_answer_failure_threshold
    if max_tie_threshold is not None:
        gate_results["max_tie_failures"] = max_tie_failures <= max_tie_threshold
    if max_failure_rate_threshold is not None:
        gate_results["max_failure_rate"] = max_failure_rate <= max_failure_rate_threshold
    if max_exact_answer_failure_rate_threshold is not None:
        gate_results["max_exact_answer_failure_rate"] = (
            max_exact_answer_failure_rate <= max_exact_answer_failure_rate_threshold
        )
    if max_tie_rate_threshold is not None:
        gate_results["max_tie_failure_rate"] = max_tie_failure_rate <= max_tie_rate_threshold
    passes_all_gates = all(gate_results.values())
    penalty = _selection_penalty(
        mean_fold_stddev=mean_fold_stddev,
        run_wu_stdev=run_wu_stdev,
        min_family_wu=worst_family_record.worst_family_wu,
        max_failure_count=max_failure_count,
        max_exact_answer_failures=max_exact_answer_failures,
        max_tie_failures=max_tie_failures,
        max_failure_rate=max_failure_rate,
        max_exact_answer_failure_rate=max_exact_answer_failure_rate,
        max_tie_failure_rate=max_tie_failure_rate,
        mean_wu_uniform_gap=mean_wu_uniform_gap,
        min_family_threshold=min_family_threshold,
        max_failure_threshold=max_failure_threshold,
        max_exact_answer_failure_threshold=max_exact_answer_failure_threshold,
        max_tie_threshold=max_tie_threshold,
        max_failure_rate_threshold=max_failure_rate_threshold,
        max_exact_answer_failure_rate_threshold=max_exact_answer_failure_rate_threshold,
        max_tie_rate_threshold=max_tie_rate_threshold,
        max_gap_threshold=max_gap_threshold,
    )
    if train_fit_available and train_fit_worst_family_record is not None and mean_train_fit_wu_uniform_gap is not None:
        penalty += max(0.0, min_family_threshold - train_fit_worst_family_record.train_fit_worst_family_wu) * 1.50
        penalty += max(0.0, mean_train_fit_wu_uniform_gap - max_gap_threshold) * 0.75
    if mean_locked_policy_train_oof_wu_gap is not None:
        penalty += max(0.0, mean_locked_policy_train_oof_wu_gap - _DEFAULT_BLIND_GATE_MAX_LOCKED_POLICY_WU_GAP) * 0.75
    if mean_locked_policy_max_family_gap is not None:
        penalty += max(0.0, mean_locked_policy_max_family_gap - _DEFAULT_BLIND_GATE_MAX_LOCKED_POLICY_WU_GAP) * 0.35
    if not blind_parity_bootstrap:
        penalty += 4.0
    penalty += max(0.0, _DEFAULT_BLIND_GATE_MIN_FOCUS_VERIFIER_COVERAGE - mean_focus_verifier_coverage_rate) * 8.0
    penalty += max(
        0.0,
        _DEFAULT_BLIND_GATE_MIN_EXACT_PARSER_SUCCESS_RATE - mean_exact_answer_parser_success_rate,
    ) * 6.0
    penalty += max(
        0.0,
        _DEFAULT_BLIND_GATE_MIN_LOW_CONFIDENCE_ACCURACY - mean_low_confidence_bucket_accuracy,
    ) * 5.0
    weighted_selection_base = (
        (mean_overall_wu * 0.40)
        + (mean_target_mix_wu * 0.20)
        + (mean_hard_proxy_wu * 0.15)
        + (worst_family_record.worst_family_wu * 0.20)
        + (((mean_train_fit_wu or mean_overall_wu)) * 0.05)
    )
    selection_score = weighted_selection_base - penalty
    return {
        "label": label,
        "run_count": len(records),
        "run_dirs": [record.run_dir for record in records],
        "metrics": {
            "train_fit_available": train_fit_available,
            "mean_train_fit_wu": round(mean_train_fit_wu, 6) if mean_train_fit_wu is not None else None,
            "mean_train_fit_uniform": (
                round(mean_train_fit_uniform, 6) if mean_train_fit_uniform is not None else None
            ),
            "mean_train_fit_wu_uniform_gap": (
                round(mean_train_fit_wu_uniform_gap, 6) if mean_train_fit_wu_uniform_gap is not None else None
            ),
            "mean_train_fit_family_wu": mean_train_fit_family_wu,
            "train_fit_worst_family_wu": (
                round(train_fit_worst_family_record.train_fit_worst_family_wu, 6)
                if train_fit_worst_family_record is not None
                else None
            ),
            "train_fit_worst_family_name": (
                train_fit_worst_family_record.train_fit_worst_family_name
                if train_fit_worst_family_record is not None
                else ""
            ),
            "max_train_fit_failure_count": max_train_fit_failure_count,
            "max_train_fit_failure_rate": (
                round(max_train_fit_failure_rate, 6) if max_train_fit_failure_rate is not None else None
            ),
            "max_train_fit_exact_answer_failures": max_train_fit_exact_answer_failures,
            "max_train_fit_exact_answer_failure_rate": (
                round(max_train_fit_exact_answer_failure_rate, 6)
                if max_train_fit_exact_answer_failure_rate is not None
                else None
            ),
            "max_train_fit_tie_failures": max_train_fit_tie_failures,
            "max_train_fit_tie_failure_rate": (
                round(max_train_fit_tie_failure_rate, 6) if max_train_fit_tie_failure_rate is not None else None
            ),
            "mean_overall_wu": round(mean_overall_wu, 6),
            "mean_overall_uniform": round(mean_overall_uniform, 6),
            "mean_target_mix_wu": round(mean_target_mix_wu, 6),
            "mean_hard_proxy_wu": round(mean_hard_proxy_wu, 6),
            "mean_wu_uniform_gap": round(mean_wu_uniform_gap, 6),
            "blind_parity_bootstrap": blind_parity_bootstrap,
            "locked_policy_gap_available": locked_policy_gap_available,
            "locked_policy_export_strategy": locked_policy_export_strategy,
            "mean_locked_policy_train_oof_wu_gap": (
                round(mean_locked_policy_train_oof_wu_gap, 6)
                if mean_locked_policy_train_oof_wu_gap is not None
                else None
            ),
            "mean_locked_policy_train_oof_uniform_gap": (
                round(mean_locked_policy_train_oof_uniform_gap, 6)
                if mean_locked_policy_train_oof_uniform_gap is not None
                else None
            ),
            "mean_locked_policy_max_family_gap": (
                round(mean_locked_policy_max_family_gap, 6) if mean_locked_policy_max_family_gap is not None else None
            ),
            "mean_verifier_coverage_rate": round(mean_verifier_coverage_rate, 6),
            "mean_focus_verifier_coverage_rate": round(mean_focus_verifier_coverage_rate, 6),
            "mean_low_confidence_bucket_accuracy": round(mean_low_confidence_bucket_accuracy, 6),
            "mean_exact_answer_parser_success_rate": round(mean_exact_answer_parser_success_rate, 6),
            "mean_discriminator_usage_rate": round(mean_discriminator_usage_rate, 6),
            "locked_policy_max_family_gap_name": (
                locked_policy_max_family_gap_record.locked_policy_max_family_gap_name
                if locked_policy_max_family_gap_record is not None
                else ""
            ),
            "weighted_selection_base": round(weighted_selection_base, 6),
            "run_wu_stdev": round(run_wu_stdev, 6),
            "mean_fold_overall_stdev": round(mean_fold_stddev, 6),
            "worst_family_wu": round(worst_family_record.worst_family_wu, 6),
            "worst_family_name": worst_family_record.worst_family_name,
            "mean_family_wu": mean_family_wu,
            "max_failure_count": max_failure_count,
            "max_failure_rate": round(max_failure_rate, 6),
            "max_exact_answer_failures": max_exact_answer_failures,
            "max_exact_answer_failure_rate": round(max_exact_answer_failure_rate, 6),
            "max_tie_failures": max_tie_failures,
            "max_tie_failure_rate": round(max_tie_failure_rate, 6),
        },
        "gates": {
            "thresholds": {
                "min_family_wu": min_family_threshold,
                "min_train_fit_family_wu": min_family_threshold if train_fit_available else None,
                "max_failure_count": max_failure_threshold,
                "max_failure_rate": max_failure_rate_threshold,
                "max_exact_answer_failures": max_exact_answer_failure_threshold,
                "max_exact_answer_failure_rate": max_exact_answer_failure_rate_threshold,
                "max_tie_failures": max_tie_threshold,
                "max_tie_failure_rate": max_tie_rate_threshold,
                "max_wu_uniform_gap": max_gap_threshold,
                "max_train_fit_wu_uniform_gap": max_gap_threshold if train_fit_available else None,
            },
            "results": gate_results,
            "passes_all": passes_all_gates,
        },
        "selection_score": round(selection_score, 6),
    }


def _blind_validation_gate_thresholds(
    *,
    min_family_threshold: float,
    max_gap_threshold: float,
    max_failure_rate_threshold: Optional[float],
    max_exact_answer_failure_rate_threshold: Optional[float],
    max_tie_rate_threshold: Optional[float],
) -> Dict[str, float]:
    return {
        "min_mean_oof_wu": _DEFAULT_BLIND_GATE_MIN_MEAN_OOF_WU,
        "min_target_mix_wu": _DEFAULT_BLIND_GATE_MIN_TARGET_MIX_WU,
        "min_worst_family_wu": max(min_family_threshold, _DEFAULT_BLIND_GATE_MIN_WORST_FAMILY_WU),
        "min_focus_family_wu": _DEFAULT_BLIND_GATE_MIN_FOCUS_FAMILY_WU,
        "min_focus_verifier_coverage_rate": _DEFAULT_BLIND_GATE_MIN_FOCUS_VERIFIER_COVERAGE,
        "min_low_confidence_bucket_accuracy": _DEFAULT_BLIND_GATE_MIN_LOW_CONFIDENCE_ACCURACY,
        "min_exact_answer_parser_success_rate": _DEFAULT_BLIND_GATE_MIN_EXACT_PARSER_SUCCESS_RATE,
        "max_failure_rate": (
            max_failure_rate_threshold
            if max_failure_rate_threshold is not None
            else _DEFAULT_BLIND_GATE_MAX_FAILURE_RATE
        ),
        "max_exact_answer_failure_rate": (
            max_exact_answer_failure_rate_threshold
            if max_exact_answer_failure_rate_threshold is not None
            else _DEFAULT_BLIND_GATE_MAX_EXACT_ANSWER_FAILURE_RATE
        ),
        "max_tie_failure_rate": (
            max_tie_rate_threshold if max_tie_rate_threshold is not None else _DEFAULT_BLIND_GATE_MAX_TIE_FAILURE_RATE
        ),
        "max_wu_uniform_gap": min(max_gap_threshold, _DEFAULT_MAX_WU_UNIFORM_GAP),
        "max_locked_policy_train_oof_wu_gap": _DEFAULT_BLIND_GATE_MAX_LOCKED_POLICY_WU_GAP,
    }


def _attach_blind_validation_gate(
    candidate: Dict[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    metrics = dict(candidate.get("metrics", {}) or {})
    mean_family_wu = dict(metrics.get("mean_family_wu", {}) or {})
    results: Dict[str, bool] = {
        "min_mean_oof_wu": _safe_float(metrics.get("mean_overall_wu")) >= _safe_float(thresholds.get("min_mean_oof_wu")),
        "min_target_mix_wu": _safe_float(metrics.get("mean_target_mix_wu"))
        >= _safe_float(thresholds.get("min_target_mix_wu")),
        "min_worst_family_wu": _safe_float(metrics.get("worst_family_wu"))
        >= _safe_float(thresholds.get("min_worst_family_wu")),
        "min_focus_verifier_coverage_rate": _safe_float(metrics.get("mean_focus_verifier_coverage_rate"))
        >= _safe_float(thresholds.get("min_focus_verifier_coverage_rate")),
        "min_low_confidence_bucket_accuracy": _safe_float(metrics.get("mean_low_confidence_bucket_accuracy"))
        >= _safe_float(thresholds.get("min_low_confidence_bucket_accuracy")),
        "min_exact_answer_parser_success_rate": _safe_float(metrics.get("mean_exact_answer_parser_success_rate"))
        >= _safe_float(thresholds.get("min_exact_answer_parser_success_rate")),
        "max_failure_rate": _safe_float(metrics.get("max_failure_rate"))
        <= _safe_float(thresholds.get("max_failure_rate")),
        "max_exact_answer_failure_rate": _safe_float(metrics.get("max_exact_answer_failure_rate"))
        <= _safe_float(thresholds.get("max_exact_answer_failure_rate")),
        "max_tie_failure_rate": _safe_float(metrics.get("max_tie_failure_rate"))
        <= _safe_float(thresholds.get("max_tie_failure_rate")),
        "max_wu_uniform_gap": _safe_float(metrics.get("mean_wu_uniform_gap"))
        <= _safe_float(thresholds.get("max_wu_uniform_gap")),
        "blind_parity_bootstrap": bool(metrics.get("blind_parity_bootstrap")),
    }
    if metrics.get("locked_policy_gap_available"):
        results["max_locked_policy_train_oof_wu_gap"] = _safe_float(
            metrics.get("mean_locked_policy_train_oof_wu_gap")
        ) <= _safe_float(thresholds.get("max_locked_policy_train_oof_wu_gap"))
    for family in _RECOVERY_SWEEP_FAMILIES:
        if family in mean_family_wu:
            gate_name = f"min_{family.replace('-', '_')}_wu"
            results[gate_name] = _safe_float(mean_family_wu.get(family)) >= _safe_float(
                thresholds.get("min_focus_family_wu")
            )
    candidate["blind_validation_gate"] = {
        "thresholds": dict(thresholds),
        "results": results,
        "passes_all": all(results.values()),
    }
    return candidate


def _recommended_sweep_families(candidate: Mapping[str, Any]) -> List[str]:
    metrics = dict((candidate.get("metrics", {}) or {}))
    family_scores = dict(metrics.get("mean_family_wu", {}) or {})
    ordered = [name for name, _ in sorted(family_scores.items(), key=lambda item: item[1])]
    prioritized = [family for family in _RECOVERY_SWEEP_FAMILIES if family in family_scores]
    result: List[str] = []
    for family in prioritized + ordered:
        if family not in result:
            result.append(family)
    return result[: max(2, min(len(result), 3))]


def _build_recovery_guidance(
    *,
    recommended_candidate: Optional[Mapping[str, Any]],
    blind_gate_thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    candidate = dict(recommended_candidate or {})
    focus_families = _recommended_sweep_families(candidate)
    return {
        "focus_families": focus_families,
        "selection_targets": [
            "mean_overall_wu",
            "mean_target_mix_wu",
            "worst_family_wu",
            "max_failure_rate",
            "mean_wu_uniform_gap",
            "mean_locked_policy_train_oof_wu_gap",
        ],
        "blind_validation_gate": dict(blind_gate_thresholds),
        "oof_first_sweeps": [
            {
                "name": "family_retrieval_sweep",
                "families": focus_families,
                "objective": "Test retrieval off versus seeded per-family retrieval before spending another blind shot.",
                "grid": {
                    "retrieval_profile": ["off"],
                    "retrieval_profile_by_family": {family: ["off", "family_question_seed_v1"] for family in focus_families},
                    "retrieval_top_k_by_family": {family: [1, 2, 4] for family in focus_families},
                },
            },
            {
                "name": "family_discriminator_sweep",
                "families": focus_families,
                "objective": "Search blind discriminator strictness on the weak families while watching failure rates.",
                "grid": {
                    "blind_discriminator_mode_by_family": {
                        family: ["default", "off", "strict"] for family in focus_families
                    }
                },
            },
            {
                "name": "capacity_parse_robustness_sweep",
                "families": focus_families,
                "objective": "Trade a little train-fit for lower parse failures and smaller WU-uniform gaps.",
                "grid": {
                    "max_criteria": [4, 6, 8],
                    "max_pairs_per_example": [2, 4, 8],
                    "max_depth": [1, 2],
                },
            },
        ],
    }


def build_selection_audit(
    candidate_runs: Mapping[str, Sequence[Path]],
    *,
    target_family_weights: Optional[Mapping[str, Any]] = None,
    hard_family_weights: Optional[Mapping[str, Any]] = None,
    min_family_threshold: float = _DEFAULT_MIN_FAMILY_WU,
    max_failure_threshold: Optional[int] = _DEFAULT_MAX_FAILURE_COUNT,
    max_exact_answer_failure_threshold: Optional[int] = _DEFAULT_MAX_EXACT_ANSWER_FAILURES,
    max_tie_threshold: Optional[int] = _DEFAULT_MAX_TIE_FAILURES,
    max_failure_rate_threshold: Optional[float] = None,
    max_exact_answer_failure_rate_threshold: Optional[float] = None,
    max_tie_rate_threshold: Optional[float] = None,
    max_gap_threshold: float = _DEFAULT_MAX_WU_UNIFORM_GAP,
) -> Dict[str, Any]:
    normalized_target_family_weights = _normalize_family_weights(target_family_weights)
    normalized_hard_family_weights = _normalize_family_weights(hard_family_weights)
    aggregated: List[Dict[str, Any]] = []
    for label, paths in sorted(candidate_runs.items()):
        records = [_build_run_record(label, _load_summary(path)) for path in paths]
        aggregated.append(
            _aggregate_records(
                label=label,
                records=records,
                target_family_weights=normalized_target_family_weights,
                hard_family_weights=normalized_hard_family_weights,
                min_family_threshold=min_family_threshold,
                max_failure_threshold=max_failure_threshold,
                max_exact_answer_failure_threshold=max_exact_answer_failure_threshold,
                max_tie_threshold=max_tie_threshold,
                max_failure_rate_threshold=max_failure_rate_threshold,
                max_exact_answer_failure_rate_threshold=max_exact_answer_failure_rate_threshold,
                max_tie_rate_threshold=max_tie_rate_threshold,
                max_gap_threshold=max_gap_threshold,
            )
        )
    blind_gate_thresholds = _blind_validation_gate_thresholds(
        min_family_threshold=min_family_threshold,
        max_gap_threshold=max_gap_threshold,
        max_failure_rate_threshold=max_failure_rate_threshold,
        max_exact_answer_failure_rate_threshold=max_exact_answer_failure_rate_threshold,
        max_tie_rate_threshold=max_tie_rate_threshold,
    )
    aggregated = [
        _attach_blind_validation_gate(candidate, thresholds=blind_gate_thresholds) for candidate in aggregated
    ]
    ranked = sorted(
        aggregated,
        key=lambda record: (
            int((record.get("blind_validation_gate", {}) or {}).get("passes_all", False)),
            int(record["gates"]["passes_all"]),
            sum(1 for passed in record["gates"]["results"].values() if passed),
            record["selection_score"],
            sum(
                1
                for passed in ((record.get("blind_validation_gate", {}) or {}).get("results", {}) or {}).values()
                if passed
            ),
            record["metrics"]["mean_overall_wu"],
            record["metrics"]["mean_target_mix_wu"],
            record["metrics"]["worst_family_wu"],
            -_safe_float(record["metrics"].get("mean_locked_policy_train_oof_wu_gap")),
            -record["metrics"]["max_failure_rate"],
            -record["metrics"]["max_tie_failure_rate"],
            -record["metrics"]["mean_wu_uniform_gap"],
            _safe_float(record["metrics"].get("mean_train_fit_wu")),
        ),
        reverse=True,
    )
    recommendation = ranked[0]["label"] if ranked else ""
    recovery_guidance = _build_recovery_guidance(
        recommended_candidate=(ranked[0] if ranked else None),
        blind_gate_thresholds=blind_gate_thresholds,
    )
    return {
        "schema": "compiled_judgebench_selection_audit_v1",
        "candidate_count": len(ranked),
        "recommendation": recommendation,
        "config": {
            "target_family_weights": normalized_target_family_weights,
            "hard_family_weights": normalized_hard_family_weights,
            "thresholds": {
                "min_family_wu": min_family_threshold,
                "min_train_fit_family_wu": min_family_threshold,
                "max_failure_count": max_failure_threshold,
                "max_failure_rate": max_failure_rate_threshold,
                "max_exact_answer_failures": max_exact_answer_failure_threshold,
                "max_exact_answer_failure_rate": max_exact_answer_failure_rate_threshold,
                "max_tie_failures": max_tie_threshold,
                "max_tie_failure_rate": max_tie_rate_threshold,
                "max_wu_uniform_gap": max_gap_threshold,
                "max_train_fit_wu_uniform_gap": max_gap_threshold,
            },
            "blind_validation_gate_thresholds": blind_gate_thresholds,
        },
        "candidates": ranked,
        "recovery_guidance": recovery_guidance,
    }


def _format_count_rate(count: Any, rate: Any) -> str:
    return f"{_safe_int(count)} ({_safe_float(rate) * 100.0:.1f}%)"


def render_selection_audit_markdown(audit: Mapping[str, Any]) -> str:
    config = dict(audit.get("config", {}) or {})
    lines = [
        "# JudgeBench Strict Selection Audit",
        "",
        f"Recommended candidate: `{audit.get('recommendation', '')}`",
    ]
    target_family_weights = dict(config.get("target_family_weights", {}) or {})
    hard_family_weights = dict(config.get("hard_family_weights", {}) or {})
    if target_family_weights:
        lines.append("")
        lines.append(f"Target family weights: `{json.dumps(target_family_weights, sort_keys=True)}`")
    if hard_family_weights:
        lines.append("")
        lines.append(f"Hard proxy weights: `{json.dumps(hard_family_weights, sort_keys=True)}`")
    blind_gate_thresholds = dict(config.get("blind_validation_gate_thresholds", {}) or {})
    if blind_gate_thresholds:
        lines.append("")
        lines.append(f"Blind validation gate: `{json.dumps(blind_gate_thresholds, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "| Candidate | Runs | Train Fit WU | OOF WU | Target Mix | Hard Proxy | Worst Family | Locked Gap | Failures | Exact Failures | Tie Failures | WU-Uniform Gap | Fold Stddev | Blind Gate | Selection Score |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for candidate in list(audit.get("candidates", []) or []):
        metrics = dict(candidate.get("metrics", {}) or {})
        gates = dict((candidate.get("gates", {}) or {}).get("results", {}) or {})
        gate_text = ", ".join(
            f"{name}={'pass' if passed else 'fail'}"
            for name, passed in sorted(gates.items())
        )
        blind_gate = dict(candidate.get("blind_validation_gate", {}) or {})
        blind_gate_text = "pass" if blind_gate.get("passes_all") else "fail"
        if blind_gate.get("results"):
            blind_gate_text += f" ({sum(1 for passed in blind_gate['results'].values() if passed)}/{len(blind_gate['results'])})"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate.get('label', '')}`",
                    str(candidate.get("run_count", 0)),
                    (
                        f"{_safe_float(metrics.get('mean_train_fit_wu')):.2f}"
                        if metrics.get("train_fit_available")
                        else "n/a"
                    ),
                    f"{_safe_float(metrics.get('mean_overall_wu')):.2f}",
                    f"{_safe_float(metrics.get('mean_target_mix_wu')):.2f}",
                    f"{_safe_float(metrics.get('mean_hard_proxy_wu')):.2f}",
                    f"{metrics.get('worst_family_name', '')} ({_safe_float(metrics.get('worst_family_wu')):.2f})",
                    (
                        f"{_safe_float(metrics.get('mean_locked_policy_train_oof_wu_gap')):.2f}"
                        if metrics.get("locked_policy_gap_available")
                        else "n/a"
                    ),
                    _format_count_rate(metrics.get("max_failure_count"), metrics.get("max_failure_rate")),
                    _format_count_rate(
                        metrics.get("max_exact_answer_failures"),
                        metrics.get("max_exact_answer_failure_rate"),
                    ),
                    _format_count_rate(metrics.get("max_tie_failures"), metrics.get("max_tie_failure_rate")),
                    f"{_safe_float(metrics.get('mean_wu_uniform_gap')):.2f}",
                    f"{_safe_float(metrics.get('mean_fold_overall_stdev')):.2f}",
                    blind_gate_text,
                    f"{_safe_float(candidate.get('selection_score')):.2f}",
                ]
            )
            + " |"
        )
        if gate_text:
            lines.append(f"  - Selection gates for `{candidate.get('label', '')}`: {gate_text}")
    recovery_guidance = dict(audit.get("recovery_guidance", {}) or {})
    if recovery_guidance:
        lines.append("")
        lines.append("## Recovery Guidance")
        focus_families = list(recovery_guidance.get("focus_families", []) or [])
        if focus_families:
            lines.append(f"Focus families: `{', '.join(focus_families)}`")
        for sweep in list(recovery_guidance.get("oof_first_sweeps", []) or []):
            lines.append("")
            lines.append(
                f"- `{sweep.get('name', '')}`: {str(sweep.get('objective', '')).strip()} "
                f"`{json.dumps(sweep.get('grid', {}), sort_keys=True)}`"
            )
    return "\n".join(lines) + "\n"


def evaluate_v2_promotion_gates(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a run's summary against the v2 promotion gates.

    The summary can either be a ``compiled_judgebench_train_only_summary_v1`` payload (in which
    case the OOF summary is extracted from ``oof_summary``) or the OOF split summary directly.
    Returns a dict with per-gate pass/fail booleans, the aggregate ``passes_all`` flag, and the
    concrete metric values used so callers can log audit trails.
    """
    summary = dict(summary or {})
    schema = str(summary.get("schema", "") or "")
    if schema == _TRAIN_ONLY_SUMMARY_SCHEMA:
        oof = dict(summary.get("oof_summary", {}) or {})
        failure = dict(summary.get("failure_analysis", {}) or {})
        external = dict(summary.get("external_slice_summary", {}) or {})
    elif schema == _SPLIT_SUMMARY_SCHEMA:
        oof = summary
        failure = {}
        external = dict(summary.get("external_slice_summary", {}) or {})
    else:
        raise ValueError(f"Unsupported summary schema for v2 gate evaluation: {schema!r}")

    wu_metrics = dict(oof.get("wu_metrics", {}) or {})
    calibration = dict(oof.get("calibration_metrics", {}) or {})
    pair_count = max(1, _safe_int(oof.get("pair_count") or failure.get("pair_count")))
    tie_failures = _safe_int(failure.get("tie_failures") or oof.get("tie_failures"))
    exact_answer_failures = _safe_int(
        failure.get("exact_answer_failures") or oof.get("exact_answer_failures")
    )
    verifier_trigger_rate_by_family = dict(calibration.get("verifier_trigger_rate_by_family", {}) or {})
    discriminator_order_disagreement_rate = _safe_float(
        calibration.get("discriminator_order_disagreement_rate")
    )

    tie_rate = tie_failures / pair_count
    exact_rate = exact_answer_failures / pair_count
    reasoning_verifier_trigger = _safe_float(verifier_trigger_rate_by_family.get("livebench-reasoning"))

    slices_payload = dict(external.get("slices", {}) or {})
    external_slice_wus: List[float] = []
    for slice_info in slices_payload.values():
        if not isinstance(slice_info, Mapping):
            continue
        if slice_info.get("available") and "wu_score" in slice_info:
            external_slice_wus.append(_safe_float(slice_info.get("wu_score")))
    external_min_wu = min(external_slice_wus) if external_slice_wus else None

    gates = {
        "overall_wu": _safe_float(wu_metrics.get("overall")) >= _V2_GATE_MIN_OVERALL_WU,
        "mmlu_pro_wu": _safe_float(wu_metrics.get("mmlu-pro")) >= _V2_GATE_MIN_FAMILY_WU_MMLU_PRO,
        "reasoning_wu": _safe_float(wu_metrics.get("livebench-reasoning"))
        >= _V2_GATE_MIN_FAMILY_WU_REASONING,
        "math_wu": _safe_float(wu_metrics.get("livebench-math")) >= _V2_GATE_MIN_FAMILY_WU_MATH,
        "tie_failure_rate": tie_rate <= _V2_GATE_MAX_TIE_FAILURE_RATE,
        "exact_answer_failure_rate": exact_rate <= _V2_GATE_MAX_EXACT_ANSWER_FAILURE_RATE,
        "reasoning_verifier_trigger_rate": reasoning_verifier_trigger
        >= _V2_GATE_MIN_REASONING_VERIFIER_TRIGGER_RATE,
        "discriminator_order_disagreement_rate": discriminator_order_disagreement_rate
        <= _V2_GATE_MAX_DISCRIMINATOR_ORDER_DISAGREEMENT_RATE,
    }
    if external_min_wu is not None:
        gates["external_slice_wu"] = external_min_wu >= _V2_GATE_MIN_EXTERNAL_SLICE_WU
    passes_all = bool(gates) and all(gates.values())
    return {
        "schema": "compiled_judgebench_v2_promotion_gates_v1",
        "passes_all": passes_all,
        "gates": gates,
        "metrics": {
            "overall_wu": _safe_float(wu_metrics.get("overall")),
            "mmlu_pro_wu": _safe_float(wu_metrics.get("mmlu-pro")),
            "reasoning_wu": _safe_float(wu_metrics.get("livebench-reasoning")),
            "math_wu": _safe_float(wu_metrics.get("livebench-math")),
            "livecodebench_wu": _safe_float(wu_metrics.get("livecodebench")),
            "tie_failure_rate": round(tie_rate, 6),
            "exact_answer_failure_rate": round(exact_rate, 6),
            "reasoning_verifier_trigger_rate": round(reasoning_verifier_trigger, 6),
            "discriminator_order_disagreement_rate": round(discriminator_order_disagreement_rate, 6),
            "external_slice_wus": external_slice_wus,
            "external_slice_wu_min": external_min_wu,
        },
        "gate_thresholds": {
            "overall_wu": _V2_GATE_MIN_OVERALL_WU,
            "mmlu_pro_wu": _V2_GATE_MIN_FAMILY_WU_MMLU_PRO,
            "reasoning_wu": _V2_GATE_MIN_FAMILY_WU_REASONING,
            "math_wu": _V2_GATE_MIN_FAMILY_WU_MATH,
            "tie_failure_rate": _V2_GATE_MAX_TIE_FAILURE_RATE,
            "exact_answer_failure_rate": _V2_GATE_MAX_EXACT_ANSWER_FAILURE_RATE,
            "reasoning_verifier_trigger_rate": _V2_GATE_MIN_REASONING_VERIFIER_TRIGGER_RATE,
            "discriminator_order_disagreement_rate": _V2_GATE_MAX_DISCRIMINATOR_ORDER_DISAGREEMENT_RATE,
            "external_slice_wu": _V2_GATE_MIN_EXTERNAL_SLICE_WU,
        },
    }


def _parse_candidate_args(values: Sequence[str]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected candidate in 'label=path' form, got: {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"Expected candidate in 'label=path' form, got: {value!r}")
        grouped.setdefault(label, []).append(Path(raw_path))
    return grouped


def _parse_weight_args(values: Sequence[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected family weight in 'family=value' form, got: {value!r}")
        family, raw_weight = value.split("=", 1)
        family = family.strip()
        weight = _safe_float(raw_weight)
        if not family or weight <= 0.0:
            raise ValueError(f"Expected positive family weight in 'family=value' form, got: {value!r}")
        parsed[family] = weight
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare JudgeBench train-only runs with stricter freeze-selection gates.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate run in 'label=run_dir_or_summary_json' form. Repeat to compare multiple runs or replicates.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write selection audit artifacts.")
    parser.add_argument("--min-family-wu", type=float, default=_DEFAULT_MIN_FAMILY_WU)
    parser.add_argument("--max-failure-count", type=int, default=_DEFAULT_MAX_FAILURE_COUNT)
    parser.add_argument("--max-exact-answer-failures", type=int, default=_DEFAULT_MAX_EXACT_ANSWER_FAILURES)
    parser.add_argument("--max-tie-failures", type=int, default=_DEFAULT_MAX_TIE_FAILURES)
    parser.add_argument("--max-failure-rate", type=float, default=None)
    parser.add_argument("--max-exact-answer-failure-rate", type=float, default=None)
    parser.add_argument("--max-tie-failure-rate", type=float, default=None)
    parser.add_argument("--max-wu-uniform-gap", type=float, default=_DEFAULT_MAX_WU_UNIFORM_GAP)
    parser.add_argument(
        "--target-family-weight",
        action="append",
        default=[],
        help="Family weight in 'family=value' form for the target family-mix proxy metric.",
    )
    parser.add_argument(
        "--hard-family-weight",
        action="append",
        default=[],
        help="Family weight in 'family=value' form for the hard-slice proxy metric.",
    )
    args = parser.parse_args(argv)

    candidates = _parse_candidate_args(args.candidate)
    audit = build_selection_audit(
        candidates,
        target_family_weights=_parse_weight_args(list(args.target_family_weight or [])),
        hard_family_weights=_parse_weight_args(list(args.hard_family_weight or [])),
        min_family_threshold=float(args.min_family_wu),
        max_failure_threshold=int(args.max_failure_count),
        max_exact_answer_failure_threshold=int(args.max_exact_answer_failures),
        max_tie_threshold=int(args.max_tie_failures),
        max_failure_rate_threshold=(
            None if args.max_failure_rate is None else float(args.max_failure_rate)
        ),
        max_exact_answer_failure_rate_threshold=(
            None if args.max_exact_answer_failure_rate is None else float(args.max_exact_answer_failure_rate)
        ),
        max_tie_rate_threshold=(
            None if args.max_tie_failure_rate is None else float(args.max_tie_failure_rate)
        ),
        max_gap_threshold=float(args.max_wu_uniform_gap),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "selection_audit.md").write_text(render_selection_audit_markdown(audit), encoding="utf-8")
    print(f"Wrote JudgeBench selection audit to: {out_dir}")
    print(f"recommended={audit.get('recommendation', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
