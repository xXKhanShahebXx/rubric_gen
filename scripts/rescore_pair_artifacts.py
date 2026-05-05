"""Recompute pair-preference accuracy on existing rubric-scored artifacts using new labels.

Reads a relabeled JSONL (output of ``scripts/relabel_pair_dataset.py``) and a run
directory containing per-example artifacts produced by the medical RubricPipeline
(``examples/<example_id>.json`` files). For each artifact we:

1. Look up the relabeled row by the artifact's ``source_id`` (= the JSONL ``id``).
2. Override the ``pair_correct_label`` on the in-memory pair_a / pair_b candidate
   metadata with the relabeled verdict (``a`` / ``b``). Skip ties / unparseable.
3. Re-run ``aggregate_method_metrics`` from the existing reporting layer.

Prints a side-by-side comparison of the original vs relabeled pair-preference
accuracy per method, plus diagnostics: tie count, label-flip count, per-method
counts of agreed / disagreed / abstained.

The original artifact files on disk are NOT modified.

Usage:
  python scripts/rescore_pair_artifacts.py \\
    --run-dir artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt41 \\
    --relabeled-jsonl data/medical_gpt41_answers_rl_relabeled_phase1.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from rubric_gen.evaluation.reporting import aggregate_method_metrics


VERDICT_TO_LABEL = {
    "A": "a",
    "B": "b",
    "a": "a",
    "b": "b",
    "reference_answer_a": "a",
    "reference_answer_b": "b",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Pipeline run directory (must contain examples/<id>.json artifacts).",
    )
    parser.add_argument(
        "--relabeled-jsonl",
        type=Path,
        required=True,
        help="Relabeled JSONL produced by scripts/relabel_pair_dataset.py.",
    )
    parser.add_argument(
        "--include-ties",
        action="store_true",
        help=(
            "By default rows with judge verdict TIE are EXCLUDED from the relabeled "
            "metric (their candidate metadata is NOT overridden, so they keep the "
            "original label). Set this flag to also override them with the original "
            "label explicitly so they're counted under the relabeled aggregate."
        ),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Optional path to write the comparison summary as JSON.",
    )
    parser.add_argument(
        "--per-method-only",
        type=str,
        default="",
        help="Comma-separated method names to focus on; empty = show all.",
    )
    return parser


def _load_relabeled_index(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Return (id -> relabeled_row, id -> normalized_new_label).

    ``normalized_new_label`` is "a" / "b" / "" (tie or unparseable).
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    new_label_by_id: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                continue
            by_id[row_id] = row
            verdict = row.get("judge_verdict") or ""
            new_label_by_id[row_id] = VERDICT_TO_LABEL.get(str(verdict), "")
    return by_id, new_label_by_id


def _override_pair_label(artifact: Mapping[str, Any], new_label: str) -> Dict[str, Any]:
    """Return a deep-copied artifact with the pair_a / pair_b candidate label overridden."""
    out = copy.deepcopy(dict(artifact))
    candidates = out.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("source_label") in {"pair_response_a", "pair_response_b"}:
            metadata = candidate.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["pair_correct_label"] = new_label
            candidate["metadata"] = metadata
    # Mirror the override onto the example block too so future readers see a
    # consistent copy of the artifact.
    example = out.get("example") or {}
    if isinstance(example, dict):
        example["pair_correct_label"] = new_label
    return out


def _collect_method_names(artifacts: List[Mapping[str, Any]]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for art in artifacts:
        for name in (art.get("methods") or {}).keys():
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return sorted(names)


def _row_lookup_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["method"]: row for row in rows}


def _format_comparison_row(method: str, before: Dict[str, Any], after: Dict[str, Any]) -> str:
    def cells(row: Dict[str, Any]) -> Tuple[str, str]:
        acc = float(row.get("pair_preference_accuracy") or 0.0)
        c = int(row.get("pair_preference_correct") or 0)
        n = int(row.get("pair_preference_evaluable") or 0)
        return f"{acc:6.3f}", f"{c}/{n}"

    b_acc, b_count = cells(before)
    a_acc, a_count = cells(after)
    delta = float(after.get("pair_preference_accuracy") or 0.0) - float(
        before.get("pair_preference_accuracy") or 0.0
    )
    delta_str = f"{delta:+.3f}"
    return f"  {method:<32}  {b_acc:>9}  {b_count:>11}    {a_acc:>9}  {a_count:>11}    {delta_str:>7}"


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"ERROR: run dir not found: {args.run_dir}", file=sys.stderr)
        return 1
    if not args.relabeled_jsonl.exists():
        print(f"ERROR: relabeled JSONL not found: {args.relabeled_jsonl}", file=sys.stderr)
        return 1

    examples_dir = args.run_dir / "examples"
    if not examples_dir.is_dir():
        print(f"ERROR: no examples/ dir under {args.run_dir}", file=sys.stderr)
        return 1

    by_id, new_label_by_id = _load_relabeled_index(args.relabeled_jsonl)
    print(f"Loaded {len(by_id)} relabeled rows from {args.relabeled_jsonl}")

    # Load all per-example artifacts (these already carry the rubric scores).
    artifact_paths = sorted(examples_dir.glob("*.json"))
    print(f"Found {len(artifact_paths)} per-example artifacts under {examples_dir}")

    original_artifacts: List[Dict[str, Any]] = []
    relabeled_artifacts: List[Dict[str, Any]] = []
    matched = 0
    unmatched: List[str] = []
    ties_or_unparsed = 0
    label_agreements = 0
    label_flips = 0

    for path in artifact_paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        original_artifacts.append(artifact)

        source_id = str((artifact.get("example") or {}).get("source_id") or "")
        if not source_id or source_id not in by_id:
            unmatched.append(path.stem)
            continue
        matched += 1
        new_label = new_label_by_id.get(source_id, "")
        original_label = ""
        for candidate in artifact.get("candidates") or []:
            if isinstance(candidate, dict) and candidate.get("source_label") == "pair_response_a":
                original_label = (candidate.get("metadata") or {}).get("pair_correct_label", "")
                break

        if not new_label:
            # TIE or unparseable verdict.
            ties_or_unparsed += 1
            if args.include_ties:
                # Keep the original label so this row stays evaluable in the
                # relabeled aggregate. No flip; counts as agreement-by-default.
                label_agreements += int(bool(original_label))
                relabeled_artifacts.append(_override_pair_label(artifact, original_label))
            else:
                # Override with empty label -> _pair_preference_outcome will mark
                # the row not-evaluable in the relabeled aggregate.
                relabeled_artifacts.append(_override_pair_label(artifact, ""))
            continue

        if original_label == new_label:
            label_agreements += 1
        else:
            label_flips += 1
        relabeled_artifacts.append(_override_pair_label(artifact, new_label))

    if unmatched:
        print(
            f"WARNING: {len(unmatched)} artifacts had no relabeled row in the JSONL "
            f"(skipped from the relabeled aggregate). First few: {unmatched[:5]}",
            file=sys.stderr,
        )

    print(f"\nMatched {matched} artifacts to relabeled rows.")
    print(
        f"  ties / unparseable verdicts : {ties_or_unparsed} "
        f"({'kept under original label' if args.include_ties else 'excluded from relabeled aggregate'})"
    )
    print(f"  label agreements           : {label_agreements}")
    print(f"  label flips (A <-> B)      : {label_flips}")

    # Aggregate both ways using the existing reporting helper.
    before_rows = aggregate_method_metrics(original_artifacts)
    after_rows = aggregate_method_metrics(relabeled_artifacts)
    before_by_method = _row_lookup_by_id(before_rows)
    after_by_method = _row_lookup_by_id(after_rows)

    method_filter = {m.strip() for m in args.per_method_only.split(",") if m.strip()}
    method_names = _collect_method_names(original_artifacts)
    if method_filter:
        method_names = [m for m in method_names if m in method_filter]

    print()
    print(
        f"  {'method':<32}  {'ORIG ACC':>9}  {'ORIG c/n':>11}    "
        f"{'NEW ACC':>9}  {'NEW c/n':>11}    {'DELTA':>7}"
    )
    print("  " + "-" * 92)
    for method in method_names:
        before = before_by_method.get(method, {})
        after = after_by_method.get(method, {})
        # Skip methods that weren't pair-evaluated under either label set.
        before_n = int(before.get("pair_preference_evaluable") or 0)
        after_n = int(after.get("pair_preference_evaluable") or 0)
        if before_n == 0 and after_n == 0:
            continue
        print(_format_comparison_row(method, before, after))

    summary = {
        "run_dir": str(args.run_dir),
        "relabeled_jsonl": str(args.relabeled_jsonl),
        "artifacts_loaded": len(original_artifacts),
        "matched_to_relabel": matched,
        "unmatched_count": len(unmatched),
        "ties_or_unparsed": ties_or_unparsed,
        "include_ties_in_aggregate": bool(args.include_ties),
        "label_agreements": label_agreements,
        "label_flips": label_flips,
        "label_flip_rate_among_decided": (
            label_flips / max(1, label_agreements + label_flips)
        ),
        "before": before_rows,
        "after": after_rows,
    }
    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote summary JSON to {args.out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
