"""
Archived-replay of the v2 stack over failures from a completed OOF run.

Operationalises the handoff recommendation:
``docs/workflows/judgebench_350_recovery_handoff.md`` §"What Could Work Next" #4 -- "use archived
example replay before spending a full 240 or 320". The replay loads the failure list from a run,
locates the per-example fold artifact (so we have the full candidate text, prompt, rubric) and
replays just the *deterministic* parts of the v2 stack -- the reasoning process verifier, the
widened discriminator gate, and the library retrieval -- on those failures. Any LLM-backed v2
component (multi-sample self-consistency, holistic judge) is not invoked here because the replay
runs without calling the network; the module outputs what those components *would* be asked to
resolve, so a follow-up pass can be planned.

The replay returns a structured report per failure plus aggregate counts::

    {
      "total_failures": 46,
      "tie_failures_in_run": 31,
      "process_verifier_would_resolve": 9,
      "widened_gate_would_route_to_discriminator": 18,
      "library_criteria_added_per_example": 4.1,
      ...
      "details": [ {"pair_id": ..., "resolution": ...}, ... ]
    }

It is intended to be run before any new ``train_240`` OOF; if fewer than the required number of
ties can be plausibly resolved, the train-only shot should not be spent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from rubric_gen.compiled.reasoning_process_verifier import (
    ReasoningProcessVerifier,
    ReasoningProcessVerifierConfig,
)
from rubric_gen.compiled.rubric_library import (
    RubricLibrary,
    load_rubric_library,
    maybe_load_default_library,
)


@dataclass
class ReplayFailureResolution:
    pair_id: str
    source_family: str
    label: str
    prior_decision: str
    prior_tie: bool
    prior_margin: float
    process_verifier_decision: str
    process_verifier_confidence: str
    process_verifier_margin: float
    process_verifier_resolves: bool
    widened_gate_routes_to_discriminator: bool
    library_criteria_count: int
    library_criteria_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source_family": self.source_family,
            "label": self.label,
            "prior_decision": self.prior_decision,
            "prior_tie": bool(self.prior_tie),
            "prior_margin": round(float(self.prior_margin), 6),
            "process_verifier_decision": self.process_verifier_decision,
            "process_verifier_confidence": self.process_verifier_confidence,
            "process_verifier_margin": round(float(self.process_verifier_margin), 6),
            "process_verifier_resolves": bool(self.process_verifier_resolves),
            "widened_gate_routes_to_discriminator": bool(self.widened_gate_routes_to_discriminator),
            "library_criteria_count": int(self.library_criteria_count),
            "library_criteria_ids": list(self.library_criteria_ids),
            "notes": list(self.notes),
        }


@dataclass
class ReplayReport:
    run_dir: str
    total_failures: int
    tie_failures_in_run: int
    exact_answer_failures_in_run: int
    process_verifier_would_resolve: int
    widened_gate_would_route_to_discriminator: int
    library_criteria_avg_per_example: float
    fold_artifacts_missing: int
    source_family_counts: Dict[str, int]
    details: List[ReplayFailureResolution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "compiled_judgebench_archived_replay_v1",
            "run_dir": self.run_dir,
            "total_failures": int(self.total_failures),
            "tie_failures_in_run": int(self.tie_failures_in_run),
            "exact_answer_failures_in_run": int(self.exact_answer_failures_in_run),
            "process_verifier_would_resolve": int(self.process_verifier_would_resolve),
            "widened_gate_would_route_to_discriminator": int(self.widened_gate_would_route_to_discriminator),
            "library_criteria_avg_per_example": round(float(self.library_criteria_avg_per_example), 4),
            "fold_artifacts_missing": int(self.fold_artifacts_missing),
            "source_family_counts": dict(self.source_family_counts),
            "details": [d.to_dict() for d in self.details],
        }


_V2_WIDER_GATE_LOW_MARGIN = 0.02
_V2_WIDER_GATE_REASONING_MARGIN = 0.05


def _find_fold_artifact(run_dir: Path, pair_id: str) -> Optional[Path]:
    folds_root = run_dir / "folds"
    if not folds_root.exists():
        return None
    for candidate in folds_root.glob(f"fold_*/dev/examples/{pair_id}.json"):
        return candidate
    return None


def _load_failures(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "summaries" / "oof_failures.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _load_failure_analysis(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "summaries" / "oof_failure_analysis.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _library_criteria_ids_for(
    library: Optional[RubricLibrary],
    *,
    source_family: str,
    limit: int = 6,
) -> List[str]:
    if library is None:
        return []
    criteria = library.filter_by_family(source_family, limit=limit)
    return [c.criterion_id for c in criteria]


def _decision_is_tie(decision: str) -> bool:
    return str(decision or "").strip() == "A=B"


def replay_run(
    run_dir: Path,
    *,
    library: Optional[RubricLibrary] = None,
    library_top_k: int = 6,
    wide_gate_low_margin: float = _V2_WIDER_GATE_LOW_MARGIN,
    wide_gate_reasoning_margin: float = _V2_WIDER_GATE_REASONING_MARGIN,
    process_verifier_config: Optional[ReasoningProcessVerifierConfig] = None,
) -> ReplayReport:
    run_dir = Path(run_dir)
    failures = _load_failures(run_dir)
    failure_analysis = _load_failure_analysis(run_dir)
    process_verifier = ReasoningProcessVerifier(
        config=process_verifier_config or ReasoningProcessVerifierConfig(),
    )

    tie_failures = sum(1 for f in failures if _decision_is_tie(f.get("decision")))
    exact_answer_failures = sum(1 for f in failures if bool(f.get("exact_answer_task")))

    process_resolves = 0
    widened_routes = 0
    library_total = 0
    missing = 0
    source_family_counts: Dict[str, int] = {}
    details: List[ReplayFailureResolution] = []

    for failure in failures:
        pair_id = str(failure.get("pair_id", "") or "")
        source_family = str(failure.get("source_family", "") or "")
        label = str(failure.get("label", "") or "")
        prior_decision = str(failure.get("decision", "") or "")
        prior_score_a = float(failure.get("score_A", 0.0) or 0.0)
        prior_score_b = float(failure.get("score_B", 0.0) or 0.0)
        prior_margin = abs(prior_score_a - prior_score_b)
        is_tie = _decision_is_tie(prior_decision)
        notes: List[str] = []
        source_family_counts[source_family] = source_family_counts.get(source_family, 0) + 1

        fold_path = _find_fold_artifact(run_dir, pair_id)
        if fold_path is None:
            missing += 1
            details.append(
                ReplayFailureResolution(
                    pair_id=pair_id,
                    source_family=source_family,
                    label=label,
                    prior_decision=prior_decision,
                    prior_tie=is_tie,
                    prior_margin=prior_margin,
                    process_verifier_decision="",
                    process_verifier_confidence="",
                    process_verifier_margin=0.0,
                    process_verifier_resolves=False,
                    widened_gate_routes_to_discriminator=False,
                    library_criteria_count=0,
                    library_criteria_ids=[],
                    notes=["fold_artifact_missing"],
                )
            )
            continue

        try:
            artifact = json.loads(fold_path.read_text(encoding="utf-8"))
        except Exception:
            missing += 1
            notes.append("fold_artifact_unreadable")
            details.append(
                ReplayFailureResolution(
                    pair_id=pair_id,
                    source_family=source_family,
                    label=label,
                    prior_decision=prior_decision,
                    prior_tie=is_tie,
                    prior_margin=prior_margin,
                    process_verifier_decision="",
                    process_verifier_confidence="",
                    process_verifier_margin=0.0,
                    process_verifier_resolves=False,
                    widened_gate_routes_to_discriminator=False,
                    library_criteria_count=0,
                    library_criteria_ids=[],
                    notes=notes,
                )
            )
            continue

        candidates = artifact.get("candidates") or []
        response_a = ""
        response_b = ""
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            meta = dict(candidate.get("metadata", {}) or {})
            pair_position = str(meta.get("pair_position", "")).strip().upper()
            if pair_position == "A" and not response_a:
                response_a = str(candidate.get("text", "") or "")
            elif pair_position == "B" and not response_b:
                response_b = str(candidate.get("text", "") or "")

        if source_family == "livebench-reasoning" and response_a and response_b:
            pv_outcome = process_verifier.evaluate(
                source_family=source_family,
                response_a=response_a,
                response_b=response_b,
            )
            pv_decision = pv_outcome.recommended_decision
            pv_confidence = pv_outcome.confidence
            pv_margin = pv_outcome.margin
            pv_resolves = bool(pv_outcome.triggered and pv_decision == label)
        else:
            pv_decision = ""
            pv_confidence = ""
            pv_margin = 0.0
            pv_resolves = False
            if source_family != "livebench-reasoning":
                notes.append("process_verifier_not_applicable_to_family")

        wide_gate_threshold = (
            wide_gate_reasoning_margin
            if source_family == "livebench-reasoning"
            else wide_gate_low_margin
        )
        widened_routes_to_disc = (
            is_tie
            or prior_margin <= wide_gate_threshold
            or bool(
                ((artifact.get("scoring") or {}).get("whitened_uniform", {}) or {}).get("result", {}).get("whitening_unstable")
            )
        )
        if widened_routes_to_disc:
            widened_routes += 1

        library_ids = _library_criteria_ids_for(library, source_family=source_family, limit=library_top_k)
        library_total += len(library_ids)

        if pv_resolves:
            process_resolves += 1

        details.append(
            ReplayFailureResolution(
                pair_id=pair_id,
                source_family=source_family,
                label=label,
                prior_decision=prior_decision,
                prior_tie=is_tie,
                prior_margin=prior_margin,
                process_verifier_decision=pv_decision,
                process_verifier_confidence=pv_confidence,
                process_verifier_margin=pv_margin,
                process_verifier_resolves=pv_resolves,
                widened_gate_routes_to_discriminator=widened_routes_to_disc,
                library_criteria_count=len(library_ids),
                library_criteria_ids=library_ids,
                notes=notes,
            )
        )

    total_failures = len(failures)
    library_avg = library_total / total_failures if total_failures else 0.0

    return ReplayReport(
        run_dir=str(run_dir),
        total_failures=total_failures,
        tie_failures_in_run=tie_failures,
        exact_answer_failures_in_run=exact_answer_failures,
        process_verifier_would_resolve=process_resolves,
        widened_gate_would_route_to_discriminator=widened_routes,
        library_criteria_avg_per_example=library_avg,
        fold_artifacts_missing=missing,
        source_family_counts=source_family_counts,
        details=details,
    )


def write_replay_report(report: ReplayReport, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _read_run_dir_from_args(argv: Sequence[str]) -> Path:
    import argparse

    parser = argparse.ArgumentParser(
        description="Archived-replay of the v2 deterministic components over a completed OOF run's failures."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=False)
    parser.add_argument("--rubric-library-path", type=Path, required=False)
    parser.add_argument("--library-top-k", type=int, default=6)
    parser.add_argument("--min-ties-resolved", type=int, default=20)
    parser.add_argument("--wide-gate-low-margin", type=float, default=_V2_WIDER_GATE_LOW_MARGIN)
    parser.add_argument("--wide-gate-reasoning-margin", type=float, default=_V2_WIDER_GATE_REASONING_MARGIN)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    import sys as _sys

    args = _read_run_dir_from_args(list(argv) if argv is not None else _sys.argv[1:])
    library: Optional[RubricLibrary] = None
    if args.rubric_library_path:
        try:
            library = load_rubric_library(args.rubric_library_path)
        except Exception:
            library = None
    if library is None:
        library = maybe_load_default_library(Path(__file__).resolve().parents[2])
    report = replay_run(
        args.run_dir,
        library=library,
        library_top_k=int(args.library_top_k or 6),
        wide_gate_low_margin=float(args.wide_gate_low_margin),
        wide_gate_reasoning_margin=float(args.wide_gate_reasoning_margin),
    )
    out_path = args.out_path or (args.run_dir / "summaries" / "archived_replay.json")
    write_replay_report(report, out_path)
    print(json.dumps(
        {
            "run_dir": report.run_dir,
            "total_failures": report.total_failures,
            "tie_failures_in_run": report.tie_failures_in_run,
            "process_verifier_would_resolve": report.process_verifier_would_resolve,
            "widened_gate_would_route_to_discriminator": report.widened_gate_would_route_to_discriminator,
            "library_criteria_avg_per_example": round(report.library_criteria_avg_per_example, 4),
            "fold_artifacts_missing": report.fold_artifacts_missing,
            "out_path": str(out_path),
        },
        indent=2,
    ))
    if report.process_verifier_would_resolve >= args.min_ties_resolved:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
