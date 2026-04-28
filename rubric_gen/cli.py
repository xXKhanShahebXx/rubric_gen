from __future__ import annotations

import argparse
import json

from rubric_gen.config import build_config
from rubric_gen.comparison_pipeline import ProposerComparisonPipeline
from rubric_gen.pipeline import RubricPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RRD rubric generation pipeline.")
    parser.add_argument("--dataset-path", default=None, help="Path to the input JSON dataset.")
    parser.add_argument("--output-dir", default=None, help="Root directory for artifacts and caches.")
    parser.add_argument("--run-name", default=None, help="Name for this pipeline run.")
    parser.add_argument(
        "--comparison-mode",
        action="store_true",
        help="Run proposer comparison mode with before/after RRD evaluation.",
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Run a paper-style evaluation path with pure generated candidates and pairwise reporting.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable all JSONL cache reads/writes for this run so every model call is fresh.",
    )
    parser.add_argument(
        "--rubrics-only",
        action="store_true",
        help="Generate and write rubric banks without running the extra downstream comparison methods.",
    )
    parser.add_argument(
        "--skip-bank-utility",
        action="store_true",
        help="Skip rubric-bank utility judging to reduce runtime.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start offset into the dataset.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process this many examples.")
    parser.add_argument("--source-filter", default=None, help="Only process examples whose source contains this string.")
    parser.add_argument("--resume", action="store_true", help="Skip examples that already have per-example artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Run with heuristic fallbacks and no provider calls.")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent rubric-evaluation workers.")
    parser.add_argument("--target-candidates", type=int, default=None, help="Override the target number of candidate notes per example.")
    parser.add_argument("--decomposition-threshold", type=int, default=None, help="Override the RRD decomposition threshold.")
    parser.add_argument("--max-initial-rubrics", type=int, default=None, help="Override the initial rubric cap.")
    parser.add_argument("--max-final-rubrics", type=int, default=None, help="Override the final RRD rubric cap.")
    parser.add_argument("--max-decomposition-depth", type=int, default=None, help="Override the maximum RRD decomposition depth.")
    parser.add_argument(
        "--writer-model",
        dest="writer_models",
        action="append",
        default=None,
        help="Repeatable provider:model spec for note generation, e.g. openai:gpt-4.1-mini.",
    )
    parser.add_argument(
        "--rubric-model",
        default=None,
        help="Provider:model spec for rubric proposal and decomposition.",
    )
    parser.add_argument(
        "--proposer-model",
        dest="proposer_models",
        action="append",
        default=None,
        help=(
            "Repeatable proposer spec for comparison mode. "
            "Supports optional alias syntax like proposer_openai=openai:gpt-4.1-mini."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Provider:model spec for rubric satisfaction and direct judging.",
    )
    parser.add_argument(
        "--bank-judge-model",
        default=None,
        help="Provider:model spec for rubric-bank judging in comparison mode.",
    )
    parser.add_argument(
        "--downstream-judge-model",
        default=None,
        help="Optional provider:model spec for downstream note judging / manifests.",
    )
    parser.add_argument(
        "--paper-pairwise-label-mode",
        choices=["reference_proxy", "judge_proxy"],
        default=None,
        help="Pairwise label mode for paper_mode.",
    )
    parser.add_argument(
        "--paper-pairwise-judge-model",
        default=None,
        help="Provider:model spec for paper_mode pairwise pseudo-label judging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        comparison_mode=args.comparison_mode,
        paper_mode=args.paper_mode,
        no_cache=args.no_cache,
        rubrics_only=args.rubrics_only,
        skip_bank_utility=args.skip_bank_utility,
        start=args.start,
        limit=args.limit,
        source_filter=args.source_filter,
        resume=args.resume,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
        target_candidate_count=args.target_candidates,
        decomposition_threshold=args.decomposition_threshold,
        max_initial_rubrics=args.max_initial_rubrics,
        max_final_rubrics=args.max_final_rubrics,
        max_decomposition_depth=args.max_decomposition_depth,
        writer_models=args.writer_models,
        proposer_models=args.proposer_models,
        rubric_model=args.rubric_model,
        judge_model=args.judge_model,
        bank_judge_model=args.bank_judge_model,
        downstream_judge_model=args.downstream_judge_model,
        paper_pairwise_label_mode=args.paper_pairwise_label_mode,
        paper_pairwise_judge_model=args.paper_pairwise_judge_model,
    )
    if config.paper_mode:
        from rubric_gen.paper_pipeline import PaperModePipeline

        result = PaperModePipeline(config).run()
    elif config.comparison_mode:
        result = ProposerComparisonPipeline(config).run()
    else:
        result = RubricPipeline(config).run()
    print(json.dumps(result, indent=2))
    return 0
