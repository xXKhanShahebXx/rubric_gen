"""Diagnostic: show what the relevance filter dropped per validation example.

Reads each ``examples/<id>.json`` artifact in a run directory and looks for the
relevance-filter debug payload that the medical pipeline (or JudgeBench compiled
pipeline) writes when ``--medical-rubric-index`` is set with the filter enabled.
For each example, prints a one-line summary plus a configurable number of sample
drop-reasons.

Usage:

  python scripts/inspect_filter_drops.py artifacts/medical_rl/runs/medical_v47_validation
  python scripts/inspect_filter_drops.py <run_dir> --samples-per-example 5
  python scripts/inspect_filter_drops.py <run_dir> --only-with-drops
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a pipeline run directory containing examples/.",
    )
    parser.add_argument(
        "--samples-per-example",
        type=int,
        default=3,
        help="Per-example sample count for printed drop-reasons (default: 3).",
    )
    parser.add_argument(
        "--only-with-drops",
        action="store_true",
        help="Skip examples where the filter dropped zero criteria.",
    )
    parser.add_argument(
        "--method-key",
        type=str,
        default="rrd_uniform",
        help="Method whose retrieval_debug to inspect (medical pipeline only). Default: rrd_uniform.",
    )
    return parser


def _retrieval_debug_from_artifact(
    artifact: Mapping[str, Any],
    *,
    method_key: str,
) -> Optional[Dict[str, Any]]:
    """Return the retrieval+filter debug payload from a per-example artifact.

    Two artifact shapes are supported:

    1. Medical RubricPipeline: ``methods.<method_key>.artifact.retrieval_debug``
       (the pipeline writes ``rrd_artifact`` to each method via
       ``_expand_weighted_methods``, and we attach ``retrieval_debug`` onto
       ``rrd_result`` before that fan-out).
    2. JudgeBench compiled pipeline: ``library_relevance_filter`` lives on the
       merged proposals payload; it surfaces inside the per-example artifact
       under whatever the JudgeBench writer produces (pair/route-shape).

    For (2) we look for a top-level ``library_relevance_filter`` key, then fall
    back to a recursive search over the artifact dict.
    """
    methods = artifact.get("methods")
    if isinstance(methods, Mapping):
        method_payload = methods.get(method_key)
        if isinstance(method_payload, Mapping):
            artifact_section = method_payload.get("artifact") or {}
            retrieval = artifact_section.get("retrieval_debug") if isinstance(artifact_section, Mapping) else None
            if isinstance(retrieval, Mapping):
                return dict(retrieval)
            # Older path: retrieval_debug stored directly on method_payload.
            retrieval = method_payload.get("retrieval_debug")
            if isinstance(retrieval, Mapping):
                return dict(retrieval)

    # JudgeBench shape: library_relevance_filter directly on the artifact.
    library_filter = artifact.get("library_relevance_filter")
    if isinstance(library_filter, Mapping):
        return {"filter": dict(library_filter)}

    return None


def _filter_payload(retrieval_debug: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not retrieval_debug:
        return None
    filter_payload = retrieval_debug.get("filter")
    if isinstance(filter_payload, Mapping):
        return filter_payload
    # If the retrieval debug *is* itself the filter payload (JudgeBench branch),
    # check for filter-shaped fields.
    if {"input_count", "kept_count", "decisions"}.issubset(retrieval_debug.keys()):
        return retrieval_debug
    return None


def _format_decision_summary(decision: Mapping[str, Any]) -> str:
    rubric_id = str(decision.get("criterion_id") or "?")
    verdict = str(decision.get("verdict") or "?")
    reason = str(decision.get("reason") or "").strip().replace("\n", " ")
    if len(reason) > 96:
        reason = reason[:93] + "..."
    return f"      [{verdict}] {rubric_id}: {reason}"


def _iter_dropped_decisions(
    filter_payload: Mapping[str, Any],
    sample_count: int,
) -> List[Mapping[str, Any]]:
    decisions = filter_payload.get("decisions") or []
    if not isinstance(decisions, list):
        return []
    dropped = [d for d in decisions if isinstance(d, Mapping) and d.get("kept") is False]
    if sample_count <= 0:
        return dropped
    return dropped[: int(sample_count)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    examples_dir = Path(args.run_dir) / "examples"
    if not examples_dir.is_dir():
        print(f"No examples/ directory under {args.run_dir}.")
        return 1

    artifacts = sorted(examples_dir.glob("*.json"))
    if not artifacts:
        print(f"No example artifacts in {examples_dir}.")
        return 1

    overall_input = 0
    overall_kept = 0
    overall_dropped = 0
    examples_with_filter = 0
    examples_with_drops = 0
    drop_count_distribution: Dict[int, int] = {}

    for path in artifacts:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        retrieval_debug = _retrieval_debug_from_artifact(artifact, method_key=args.method_key)
        filter_payload = _filter_payload(retrieval_debug)
        if filter_payload is None:
            continue

        examples_with_filter += 1
        try:
            input_count = int(filter_payload.get("input_count") or 0)
            kept_count = int(filter_payload.get("kept_count") or 0)
            dropped_count = int(filter_payload.get("dropped_count") or 0)
        except (TypeError, ValueError):
            continue
        overall_input += input_count
        overall_kept += kept_count
        overall_dropped += dropped_count
        drop_count_distribution[dropped_count] = drop_count_distribution.get(dropped_count, 0) + 1
        if args.only_with_drops and dropped_count == 0:
            continue
        if dropped_count > 0:
            examples_with_drops += 1

        source_id = str((artifact.get("example") or {}).get("source_id") or path.stem)
        strictness = str(filter_payload.get("strictness") or "?")
        print(
            f"{source_id[:42]:<44}  in={input_count:>2}  kept={kept_count:>2}  dropped={dropped_count:>2}  ({strictness})"
        )
        for decision in _iter_dropped_decisions(filter_payload, args.samples_per_example):
            print(_format_decision_summary(decision))

    print()
    print(f"Aggregate ({len(artifacts)} artifacts scanned)")
    print(f"  Examples with filter payload : {examples_with_filter}")
    print(f"  Examples with drops          : {examples_with_drops}")
    print(f"  Total criteria input         : {overall_input}")
    print(f"  Total criteria kept          : {overall_kept}")
    print(f"  Total criteria dropped       : {overall_dropped}")
    if overall_input > 0:
        rate = overall_dropped / overall_input
        print(f"  Drop rate                    : {rate:.1%}")
    if drop_count_distribution:
        print("  Drop-count distribution      :")
        for count in sorted(drop_count_distribution):
            print(f"    {count} drops -> {drop_count_distribution[count]} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
