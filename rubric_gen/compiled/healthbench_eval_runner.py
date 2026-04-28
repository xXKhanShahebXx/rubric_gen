"""
CLI for HealthBench-based external evaluation of the compiled rubric discovery pipeline.

Run:
    python -m rubric_gen.compiled.healthbench_eval_runner
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from rubric_gen.compiled.healthbench_eval import run_healthbench_evaluation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a HealthBench external comparison for the compiled rubric discovery pipeline. "
            "The runner routes HealthBench examples through the generalized task-profile system, generates "
            "local criteria from ideal-vs-alt completions, compares them to physician criteria, and samples "
            "disagreements while preserving the old note-only regression slice."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="HealthBench JSON sample path (default: <repo>/healthbench_eval_sample.json)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under artifacts/compiled_healthbench_runs/",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override artifacts root (default: <repo>/artifacts/compiled_healthbench_runs)",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Max raw examples to load before filtering (0 = all)")
    parser.add_argument(
        "--discovery-model",
        type=str,
        default=None,
        help="Override discovery model as provider:model (defaults to compiled judge discovery model).",
    )
    parser.add_argument(
        "--alignment-model",
        type=str,
        default=None,
        help="Override rubric-alignment model as provider:model (defaults to comparison judge model).",
    )
    parser.add_argument(
        "--adjudication-model",
        type=str,
        default=None,
        help="Override disagreement-adjudication model as provider:model (defaults to comparison judge model).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable JSONL cache for all LLM calls.")
    parser.add_argument(
        "--max-pairs-per-example",
        type=int,
        default=4,
        help="Cap alt completions compared against the ideal completion per example (default: 4).",
    )
    parser.add_argument(
        "--max-criteria",
        type=int,
        default=8,
        help="Max local criteria requested per ideal-vs-alt pair (default: 8).",
    )
    parser.add_argument(
        "--disagreement-sample-size",
        type=int,
        default=12,
        help="How many disagreements to adjudicate after automatic comparison (default: 12).",
    )
    parser.add_argument(
        "--gold-provider",
        type=str,
        default="healthbench",
        help="Gold rubric provider backend to compare against (default: healthbench).",
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=1,
        help="How many gold-guided local refinement passes to run per example (default: 1).",
    )
    parser.add_argument(
        "--apply-calibration",
        type=Path,
        default=None,
        help="Optional calibration_hints.json file from a previous run to feed into discovery and family mapping.",
    )
    parser.add_argument(
        "--no-emit-calibration-hints",
        action="store_true",
        help="Skip writing calibration_hints.json for this run.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small run for verification: limit=30, max 2 pairs/example, at most 4 criteria/pair, 4 adjudications.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    dataset = args.dataset or (root / "healthbench_eval_sample.json")
    out_root = args.out_root or (root / "artifacts" / "compiled_healthbench_runs")
    run_name = args.run_name or f"healthbench_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_name

    limit = args.limit
    max_pairs = args.max_pairs_per_example
    max_criteria = max(1, min(args.max_criteria, 24))
    disagreement_sample_size = max(0, args.disagreement_sample_size)

    if args.smoke:
        if limit == 0:
            limit = 30
        max_pairs = min(max_pairs, 2) if max_pairs is not None else 2
        max_criteria = min(max_criteria, 4)
        disagreement_sample_size = min(disagreement_sample_size or 4, 4)

    _, summary = run_healthbench_evaluation(
        dataset_path=dataset,
        run_dir=run_dir,
        start=args.start,
        limit=limit,
        discovery_model_override=args.discovery_model,
        alignment_model_override=args.alignment_model,
        adjudication_model_override=args.adjudication_model,
        use_cache=not args.no_cache,
        max_criteria=max_criteria,
        max_pairs_per_example=max_pairs,
        disagreement_sample_size=disagreement_sample_size,
        gold_provider=args.gold_provider,
        refine_iterations=max(0, args.refine_iterations),
        apply_calibration=args.apply_calibration,
        emit_calibration_hints=not args.no_emit_calibration_hints,
    )

    print(f"Wrote HealthBench evaluation artifacts to: {run_dir}")
    print(
        f"selected_examples={summary['routing']['counts']['selected']} "
        f"note_regression_examples={summary['subset']['counts']['selected']} "
        f"pairs_total={summary['discovery']['pairs_total']} "
        f"pre_weighted_recall={summary['pre_refinement_alignment']['weighted_recall']:.3f} "
        f"post_weighted_recall={summary['alignment']['weighted_recall']:.3f} "
        f"post_generated_precision={summary['alignment']['generated_precision']:.3f}"
    )


if __name__ == "__main__":
    main()
