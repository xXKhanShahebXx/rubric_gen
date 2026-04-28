"""
CLI that evaluates a completed train-only run against the v2 promotion gates.

Usage::

    python -m rubric_gen.compiled.v2_promotion_gate_cli \
        --run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29

Exits 0 when every v2 gate is satisfied (the candidate may be promoted to the next experimental
phase); exits 2 otherwise. Prints a JSON summary of each gate's pass/fail plus the concrete
metric values, so the output can be fed into automated promotion scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rubric_gen.compiled.judgebench_selection_audit import (
    _load_summary,
    evaluate_v2_promotion_gates,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether a JudgeBench train-only run clears the v2 promotion gates."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a completed train-only run directory (should contain summaries/summary.json "
        "or summaries/oof_summary.json).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Optional path to write the gate-evaluation JSON. Defaults to "
        "<run-dir>/summaries/v2_promotion_gates.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the stdout JSON dump (still writes --out-path).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = _build_parser().parse_args(raw)
    summary = _load_summary(args.run_dir)
    result = evaluate_v2_promotion_gates(summary)
    out_path = args.out_path or (args.run_dir / "summaries" / "v2_promotion_gates.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("passes_all") else 2


if __name__ == "__main__":
    raise SystemExit(main())
