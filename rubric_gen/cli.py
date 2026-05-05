from __future__ import annotations

import argparse
import json

from rubric_gen.config import (
    ALLOWED_PRESETS,
    ALLOWED_SPLITS,
    DEFAULT_NUM_SHARDS,
    DEFAULT_SHARD_INDEX,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    build_config,
)
from rubric_gen.comparison_pipeline import ProposerComparisonPipeline
from rubric_gen.pipeline import RubricPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RRD rubric generation pipeline.")
    parser.add_argument("--dataset-path", default=None, help="Path to the input JSON or JSONL dataset.")
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
    parser.add_argument(
        "--split",
        choices=list(ALLOWED_SPLITS),
        default=None,
        help=(
            "Deterministic train/val split selector applied before --start/--limit. "
            "'train' takes the first --train-size rows; 'val' takes the next --val-size "
            "rows; 'all' (default when no preset is active) keeps every row."
        ),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help=(
            f"Number of rows reserved for training. Defaults to the preset's value "
            f"or {DEFAULT_TRAIN_SIZE} when --preset judgebench-v47-medical is set."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help=(
            f"Number of rows reserved for validation, taken from the rows after the "
            f"training span. Defaults to {DEFAULT_VAL_SIZE} under the medical preset."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help=(
            "Number of equal-sized contiguous shards to split the training span into. "
            f"Defaults to {DEFAULT_NUM_SHARDS} under the medical preset. --train-size "
            "must be evenly divisible by this value."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=DEFAULT_SHARD_INDEX,
        help=(
            "Which shard of the training span this run should process (0-indexed). "
            "Three team members run shard 0 / 1 / 2 on separate machines for the "
            "default 3000-row training span."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=list(ALLOWED_PRESETS),
        default=None,
        help=(
            "Lock in a known-good configuration. 'judgebench-v47-medical' applies the "
            "rubric-judge core that JudgeBench v4.7 (80.57%% on val_350) is built on, "
            "adapted for single-response medical Q&A: GPT-4o for both rubric proposal "
            "and rubric satisfaction, multi-model writers for candidate diversity, and "
            "the 3000/2000/3-shard training layout. User-supplied flags still win over "
            "the preset."
        ),
    )
    parser.add_argument(
        "--reference-field",
        dest="reference_fields",
        action="append",
        default=None,
        help=(
            "Repeatable column name to promote into the gold/reference slot during "
            "loading. The first matching column wins, then the loader falls back to "
            "the built-in reference field list (reference_artifact, reference_output, "
            "gold_output, target, ...). Use this to mark a generic field like "
            "'response' as gold so reference_top1_rate becomes meaningful. The "
            "judgebench-v47-medical preset sets this to 'response' by default."
        ),
    )
    parser.add_argument(
        "--medical-rubric-index",
        dest="medical_rubric_index_path",
        default=None,
        help=(
            "Path to a medical RRD-rubric embedding index built by "
            "scripts/build_medical_rubric_index.py. When set AND --split val, the "
            "pipeline embeds each validation prompt, retrieves the top-K nearest "
            "training rubrics, runs the relevance filter (unless "
            "--no-relevance-filter is set), and seeds RRD discovery with the "
            "survivors. Has no effect outside validation."
        ),
    )
    parser.add_argument(
        "--medical-rubric-retrieval-top-k",
        type=int,
        default=None,
        help=(
            "Number of nearest training rubrics to retrieve per validation example "
            "before applying the relevance filter (default: 8). Set to 0 to disable "
            "retrieval even when --medical-rubric-index is set."
        ),
    )
    parser.add_argument(
        "--no-relevance-filter",
        dest="medical_rubric_filter_enabled",
        action="store_false",
        default=None,
        help=(
            "Disable the Sonnet 4.5 relevance filter on retrieved rubrics. By "
            "default the filter is on whenever --medical-rubric-index is set. Use "
            "this when you want to compare retrieval-only vs retrieval+filter."
        ),
    )
    parser.add_argument(
        "--relevance-filter-strictness",
        dest="medical_rubric_filter_strictness",
        choices=["conservative", "aggressive"],
        default=None,
        help=(
            "How strictly the relevance filter prunes. 'conservative' (default) "
            "drops only IRRELEVANT verdicts and keeps UNCERTAIN. 'aggressive' "
            "keeps only APPLICABLE."
        ),
    )
    parser.add_argument(
        "--relevance-filter-model",
        dest="medical_rubric_filter_model",
        default=None,
        help=(
            "Optional provider:model override for the relevance filter (e.g. "
            "anthropic:claude-opus-4-7). Defaults to anthropic:claude-sonnet-4-5-20250929."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip examples that already have per-example artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Run with heuristic fallbacks and no provider calls.")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent rubric-evaluation workers (parallelism within one example).")
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help=(
            "Number of examples to process concurrently. Default 1 keeps the "
            "original sequential behaviour. Cross-sample parallelism multiplies "
            "with --max-workers, so --sample-workers 4 --max-workers 4 issues up "
            "to 16 concurrent rubric-satisfaction calls. Per-example artifact "
            "writes and the JSONL caches are already thread-safe, so a clean "
            "Ctrl+C is recoverable via --resume."
        ),
    )
    parser.add_argument("--target-candidates", type=int, default=None, help="Override the target number of candidate notes per example.")
    parser.add_argument("--decomposition-threshold", type=int, default=None, help="Override the RRD decomposition threshold.")
    parser.add_argument("--max-initial-rubrics", type=int, default=None, help="Override the initial rubric cap.")
    parser.add_argument("--max-final-rubrics", type=int, default=None, help="Override the final RRD rubric cap.")
    parser.add_argument("--max-decomposition-depth", type=int, default=None, help="Override the maximum RRD decomposition depth.")
    # v2 Tier A6: surface the new RRD tunables introduced in A4 + A5.
    parser.add_argument(
        "--rubric-satisfaction-samples",
        type=int,
        default=None,
        help=(
            "Number of LLM samples per rubric-satisfaction call (majority vote). "
            "Default 1 preserves single-sample behaviour and cache hits. "
            "Set to 3 to attack the score-tie regime by adding signal."
        ),
    )
    parser.add_argument(
        "--rubric-satisfaction-temperature",
        type=float,
        default=None,
        help=(
            "Temperature used for samples 1..N-1 when --rubric-satisfaction-samples > 1. "
            "Sample 0 always uses temperature 0.0 (so its cache key matches the legacy single-sample key)."
        ),
    )
    parser.add_argument(
        "--discrimination-min-pq",
        type=float,
        default=None,
        help=(
            "Drop rubrics with min(p, 1-p) below this threshold from the scoring weights/rankings "
            "(post-RRD discrimination filter). Default 0.0 keeps all rubrics. "
            "Recommended 0.05 to drop the ~12 pct 'useless' rubrics found by shard 0 v2 angle D."
        ),
    )
    parser.add_argument(
        "--decomposition-min-recall",
        type=float,
        default=None,
        help=(
            "Override the decomposition acceptance min_recall (default 0.85). "
            "Lower values (e.g. 0.70) accept more decompositions, increasing depth>0 rubric count."
        ),
    )
    parser.add_argument(
        "--decomposition-min-discrimination-gain",
        type=float,
        default=None,
        help=(
            "Override the decomposition min_discrimination_gain (default 0.03). "
            "Lower values (e.g. 0.01) accept more decompositions whose children only modestly "
            "improve discrimination over the parent."
        ),
    )
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
        sample_workers=args.sample_workers,
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
        split=args.split,
        train_size=args.train_size,
        val_size=args.val_size,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        preset=args.preset,
        reference_fields=args.reference_fields,
        medical_rubric_index_path=args.medical_rubric_index_path,
        medical_rubric_retrieval_top_k=args.medical_rubric_retrieval_top_k,
        medical_rubric_filter_enabled=args.medical_rubric_filter_enabled,
        medical_rubric_filter_strictness=args.medical_rubric_filter_strictness,
        medical_rubric_filter_model=args.medical_rubric_filter_model,
        rubric_satisfaction_samples=args.rubric_satisfaction_samples,
        rubric_satisfaction_temperature=args.rubric_satisfaction_temperature,
        discrimination_min_pq=args.discrimination_min_pq,
        decomposition_min_recall=args.decomposition_min_recall,
        decomposition_min_discrimination_gain=args.decomposition_min_discrimination_gain,
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


if __name__ == "__main__":
    raise SystemExit(main())
