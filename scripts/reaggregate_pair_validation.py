"""Re-aggregate a pair-preference validation run from existing per-example artifacts.

Useful after changing the pair-preference scoring rule in
``rubric_gen/evaluation/reporting.py`` -- there's no need to re-run the
LLM pipeline. Just reload the already-written artifacts under
``<run_dir>/examples/`` and recompute the aggregate metrics.

Writes the rebuilt summary alongside the original (suffixed
``_v2`` so you can diff before/after) and prints the headline
pair-preference accuracy for every method, with the delta relative
to the existing ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from rubric_gen.evaluation.reporting import aggregate_method_metrics, _write_csv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to artifacts/.../runs/<run-name>/")
    parser.add_argument(
        "--suffix",
        type=str,
        default="_v2",
        help="Suffix appended to the new summary filenames (default: _v2).",
    )
    return parser


def _read_existing_pair_pref(summary_md: Path) -> Dict[str, Dict[str, float]]:
    """Pull (correct, n) for each method from the existing summary.md."""
    out: Dict[str, Dict[str, float]] = {}
    if not summary_md.exists():
        return out
    text = summary_md.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        method = cells[0]
        if method.lower() == "method" or method.startswith("---"):
            continue
        m = re.match(r"^([\d.]+)\s*$", cells[1])
        m2 = re.match(r"^(\d+)\s*/\s*(\d+)$", cells[2])
        if m and m2:
            out[method] = {
                "old_acc": float(m.group(1)),
                "old_correct": int(m2.group(1)),
                "old_n": int(m2.group(2)),
            }
    return out


def main() -> int:
    args = _build_parser().parse_args()
    run_dir: Path = args.run_dir
    examples_dir = run_dir / "examples"
    files = sorted(examples_dir.glob("*.json"))
    print(f"Run dir       : {run_dir}")
    print(f"Examples found: {len(files)}")
    if not files:
        print("ERROR: no per-example artifacts to aggregate.", file=sys.stderr)
        return 1

    # Read existing headlines for delta reporting.
    summary_md = run_dir / "summaries" / "summary.md"
    old_metrics = _read_existing_pair_pref(summary_md)

    # Re-aggregate using the current (post-fix) reporting code.
    artifacts: List[Dict[str, Any]] = []
    for p in files:
        artifacts.append(json.loads(p.read_text(encoding="utf-8")))
    rows = aggregate_method_metrics(artifacts)

    # Write the rebuilt outputs alongside the originals (don't overwrite
    # the audit trail of the original run).
    suffix = args.suffix
    reports_dir = run_dir / "reports"
    summaries_dir = run_dir / "summaries"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    new_csv = reports_dir / f"method_metrics{suffix}.csv"
    new_json = summaries_dir / f"summary{suffix}.json"
    new_md = summaries_dir / f"summary{suffix}.md"
    _write_csv(new_csv, rows)
    new_json.write_text(
        json.dumps({"method_metrics": rows}, indent=2),
        encoding="utf-8",
    )

    # Render a focused markdown summary (just pair-pref) to make the
    # before/after diff easy to read.
    lines = ["# Re-aggregated pair-preference summary", ""]
    lines.append(
        "| Method | New pair-pref acc | New correct/n | Old pair-pref acc | Old correct/n | Delta (pp) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        method = row["method"]
        new_correct = int(row.get("pair_preference_correct", 0) or 0)
        new_n = int(row.get("pair_preference_evaluable", 0) or 0)
        new_acc = float(row.get("pair_preference_accuracy", 0.0) or 0.0)
        old = old_metrics.get(method, {})
        old_acc = old.get("old_acc", 0.0)
        old_correct = int(old.get("old_correct", 0))
        old_n = int(old.get("old_n", 0))
        delta_pp = (new_acc - old_acc) * 100 if (new_n and old_n) else 0.0
        lines.append(
            f"| {method} | {new_acc:.4f} | {new_correct}/{new_n} | "
            f"{old_acc:.4f} | {old_correct}/{old_n} | {delta_pp:+.2f} |"
        )
    new_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 96)
    print(f"{'Method':<26} {'NEW acc':>10} {'NEW c/n':>14}    {'OLD acc':>10} {'OLD c/n':>14}    {'Δ pp':>7}")
    print("-" * 96)
    for row in rows:
        method = row["method"]
        new_correct = int(row.get("pair_preference_correct", 0) or 0)
        new_n = int(row.get("pair_preference_evaluable", 0) or 0)
        new_acc = float(row.get("pair_preference_accuracy", 0.0) or 0.0)
        old = old_metrics.get(method, {})
        old_acc = old.get("old_acc", 0.0)
        old_correct = int(old.get("old_correct", 0))
        old_n = int(old.get("old_n", 0))
        delta_pp = (new_acc - old_acc) * 100 if (new_n and old_n) else 0.0
        print(
            f"{method:<26} {new_acc:>10.4f} {f'{new_correct}/{new_n}':>14}    "
            f"{old_acc:>10.4f} {f'{old_correct}/{old_n}':>14}    {delta_pp:>+7.2f}"
        )
    print()
    print(f"New artifacts:")
    print(f"  {new_csv}")
    print(f"  {new_json}")
    print(f"  {new_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
