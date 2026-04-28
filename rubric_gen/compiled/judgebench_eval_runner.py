"""
CLI for JudgeBench evaluation of the compiled recursive rubric discovery pipeline.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rubric_gen.compiled.discovery import RecursiveDiscoveryConfig
from rubric_gen.compiled.judgebench_eval import (
    _ALLOWED_BLIND_BUDGET_PROFILES,
    _ALLOWED_BLIND_GUIDANCE_PROFILES,
    _ALLOWED_BLIND_SCORING_PROFILES,
    _ALLOWED_BLIND_WU_PROFILES,
    _ALLOWED_RETRIEVAL_PROFILES,
    _DEFAULT_PROTOCOL_MODE,
    _PROTOCOL_MODE_GENERIC_BASELINE,
    _PROTOCOL_MODE_JUDGEBENCH_TUNED,
    run_judgebench_final_evaluation,
    run_judgebench_recursive_evaluation,
    run_judgebench_train_only_development,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under the chosen artifact root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override the artifact root directory for this command.",
    )
    parser.add_argument(
        "--discovery-model",
        type=str,
        default=None,
        help="Override compiled discovery model as provider:model.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override JudgeBench rubric-satisfaction judge model as provider:model.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable JSONL caches for discovery and scoring.")
    parser.add_argument(
        "--max-criteria",
        type=int,
        default=8,
        help="Max criteria requested per discovery pair (default: 8).",
    )
    parser.add_argument(
        "--max-pairs-per-example",
        type=int,
        default=4,
        help="Cap total weak candidates per example including A/B (default: 4).",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=3,
        help="Bootstrap refinement passes when learning source-family routing (default: 3).",
    )
    parser.add_argument("--max-depth", type=int, default=1, help="Recursive discovery max depth (default: 1).")
    parser.add_argument(
        "--max-recursive-parents-per-pair",
        type=int,
        default=2,
        help="Recursive parent expansion cap per pair (default: 2).",
    )
    parser.add_argument(
        "--max-children-per-parent",
        type=int,
        default=3,
        help="Recursive child cap per parent (default: 3).",
    )
    parser.add_argument(
        "--max-recursive-calls-per-pair",
        type=int,
        default=2,
        help="Recursive call cap per pair (default: 2).",
    )
    parser.add_argument(
        "--covariance-ridge",
        type=float,
        default=1e-3,
        help="Ridge term for whitened-uniform weighting (default: 1e-3).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel JudgeBench examples to process per split (default: 4).",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse already-written per-example artifacts where present.")


def _build_recursive_config(args: argparse.Namespace) -> RecursiveDiscoveryConfig:
    return RecursiveDiscoveryConfig(
        max_depth=max(0, args.max_depth),
        max_recursive_parents_per_pair=max(0, args.max_recursive_parents_per_pair),
        max_children_per_parent=max(1, args.max_children_per_parent),
        max_recursive_calls_per_pair=max(0, args.max_recursive_calls_per_pair),
    )


def _default_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _parse_family_overrides(values: Optional[List[str]], *, value_kind: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for raw_value in values or []:
        if "=" not in str(raw_value):
            raise ValueError(f"Expected {value_kind} override in 'family=value' form, got: {raw_value!r}")
        family, value = str(raw_value).split("=", 1)
        family = family.strip()
        value = value.strip()
        if not family or not value:
            raise ValueError(f"Expected {value_kind} override in 'family=value' form, got: {raw_value!r}")
        parsed[family] = int(value) if value_kind == "retrieval top-k" else value
    return parsed


def _v2_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "self_consistency_n": max(1, int(getattr(args, "self_consistency_n", 1) or 1)),
        "self_consistency_temperature": max(0.0, float(getattr(args, "self_consistency_temperature", 0.7) or 0.7)),
        "v2_wide_discriminator_gate": bool(getattr(args, "v2_wide_discriminator_gate", False)),
        "holistic_judge_enabled": bool(getattr(args, "holistic_judge", False)),
        "library_retrieval_top_k": max(0, int(getattr(args, "library_retrieval_top_k", 0) or 0)),
        "rubric_library_path": str(getattr(args, "rubric_library_path", "") or ""),
        "enable_rrd_filters": bool(getattr(args, "enable_rrd_filters", False)),
        "rrd_redundancy_threshold": float(getattr(args, "rrd_redundancy_threshold", 0.9) or 0.9),
    }


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run JudgeBench recursive rubric discovery with an 80-example design split and a 270-example "
            "held-out validation split."
        ),
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=None,
        help="Local JudgeBench design split JSON (default: <repo>/data/judgebench_80_human.json).",
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help="Local JudgeBench validation split JSON (default: <repo>/data/judgebench_270_generated.json).",
    )
    parser.add_argument(
        "--train-split-name",
        type=str,
        default="train_80",
        help="Artifact directory / summary name for the training split (default: train_80).",
    )
    parser.add_argument(
        "--validation-split-name",
        type=str,
        default="validation_270",
        help="Artifact directory / summary name for the validation split (default: validation_270).",
    )
    parser.add_argument(
        "--official-dataset",
        type=Path,
        default=None,
        help="Official JudgeBench pairwise JSONL. Downloaded if omitted.",
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=2,
        help="Accepted refinement attempts on the 80-example design split (default: 2).",
    )
    parser.add_argument(
        "--protocol-mode",
        choices=[_PROTOCOL_MODE_GENERIC_BASELINE, _PROTOCOL_MODE_JUDGEBENCH_TUNED],
        default=_DEFAULT_PROTOCOL_MODE,
        help="JudgeBench evaluation protocol mode.",
    )
    _add_shared_runtime_args(parser)
    return parser


def _build_train_only_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run JudgeBench train-only development on the 80-example split using balanced inner folds.",
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=None,
        help="Local JudgeBench train split JSON (default: <repo>/data/judgebench_80_human.json).",
    )
    parser.add_argument(
        "--train-split-name",
        type=str,
        default="train_80",
        help="Artifact directory / summary name for the train-only split (default: train_80).",
    )
    parser.add_argument(
        "--official-dataset",
        type=Path,
        default=None,
        help="Official JudgeBench pairwise JSONL. Downloaded if omitted.",
    )
    parser.add_argument(
        "--fold-count",
        type=int,
        default=4,
        help="Balanced fold count for inner train-only evaluation (default: 4).",
    )
    parser.add_argument(
        "--fold-shuffle-seed",
        type=int,
        default=None,
        help="Optional seed to reshuffle balanced fold assignment before OOF evaluation.",
    )
    parser.add_argument(
        "--protocol-mode",
        choices=[_PROTOCOL_MODE_GENERIC_BASELINE, _PROTOCOL_MODE_JUDGEBENCH_TUNED],
        default=_PROTOCOL_MODE_GENERIC_BASELINE,
        help="JudgeBench evaluation protocol mode for train-only development.",
    )
    parser.add_argument(
        "--allow-train-reference-answer",
        action="store_true",
        help="Expose reference answers during train-side routing/bootstrap. Off by default so blind-parity bootstrap is used.",
    )
    parser.add_argument(
        "--no-train-reference-answer",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-dev-reference-answer",
        action="store_true",
        help="Expose reference answers during fold-dev OOF scoring. Off by default so train-only model selection stays blind.",
    )
    parser.add_argument(
        "--write-train-fit",
        action="store_true",
        help="Score the locked full-train policy on the full train split and write train-fit artifacts.",
    )
    parser.add_argument(
        "--allow-train-fit-reference-answer",
        action="store_true",
        help="Expose reference answers during full-train train-fit scoring. Off by default so train-fit stays blind.",
    )
    parser.add_argument(
        "--blind-scoring-profile",
        choices=sorted(_ALLOWED_BLIND_SCORING_PROFILES),
        default="baseline",
        help="Blind scoring profile for OOF/final selection.",
    )
    parser.add_argument(
        "--blind-budget-profile",
        choices=sorted(_ALLOWED_BLIND_BUDGET_PROFILES),
        default="family_v1",
        help="Blind candidate-budget / mutation-ordering profile.",
    )
    parser.add_argument(
        "--blind-guidance-profile",
        choices=sorted(_ALLOWED_BLIND_GUIDANCE_PROFILES),
        default="off",
        help="Blind-safe discovery guidance profile for generic_baseline runs.",
    )
    parser.add_argument(
        "--blind-wu-profile",
        choices=sorted(_ALLOWED_BLIND_WU_PROFILES),
        default="raw",
        help="Blind whitened-uniform stabilization profile.",
    )
    parser.add_argument(
        "--retrieval-profile",
        choices=sorted(_ALLOWED_RETRIEVAL_PROFILES),
        default="off",
        help="Retrieval augmentation profile. Keep `off` for the strict-blind benchmark path.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=2,
        help="Number of frozen train exemplars to retrieve when retrieval is enabled.",
    )
    parser.add_argument(
        "--retrieval-family-profile",
        action="append",
        default=[],
        help="Per-family retrieval override in 'family=profile' form. Repeatable.",
    )
    parser.add_argument(
        "--retrieval-family-top-k",
        action="append",
        default=[],
        help="Per-family retrieval top-k override in 'family=int' form. Repeatable.",
    )
    parser.add_argument(
        "--blind-discriminator-family-mode",
        action="append",
        default=[],
        help="Per-family blind discriminator mode in 'family=default|off|strict' form. Repeatable.",
    )
    _add_v2_args(parser)
    _add_shared_runtime_args(parser)
    return parser


def _add_v2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--self-consistency-n",
        type=int,
        default=1,
        help="Number of self-consistency samples (per order) for the pair discriminator. 1 keeps current behavior.",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used when --self-consistency-n > 1.",
    )
    parser.add_argument(
        "--v2-wide-discriminator-gate",
        action="store_true",
        help="Widen the blind pair discriminator gate to fire on low-margin and whitening-unstable pairs.",
    )
    parser.add_argument(
        "--holistic-judge",
        action="store_true",
        help="Enable the RaR-style one-call holistic pair judge as an extra signal on low-margin / tie pairs.",
    )
    parser.add_argument(
        "--rubric-library-path",
        type=str,
        default="",
        help="Path to a compiled rubric library JSON. When set, library criteria are injected into per-example discovery.",
    )
    parser.add_argument(
        "--library-retrieval-top-k",
        type=int,
        default=0,
        help="Max library criteria to retrieve per example. Ignored when --rubric-library-path is empty.",
    )
    parser.add_argument(
        "--enable-rrd-filters",
        action="store_true",
        help="Apply the RRD misalignment and redundancy filters to merged proposals before scoring.",
    )
    parser.add_argument(
        "--rrd-redundancy-threshold",
        type=float,
        default=0.9,
        help="Jaccard threshold used by the RRD redundancy filter. Lower = more aggressive pruning.",
    )


def _build_final_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-shot final evaluation on the held-out JudgeBench validation split using a locked train-only policy.",
    )
    parser.add_argument(
        "--train-run-dir",
        type=Path,
        required=True,
        help="Path to a completed train-only development run directory.",
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help="Held-out validation JSON (default: <repo>/data/judgebench_270_generated.json).",
    )
    parser.add_argument(
        "--official-dataset",
        type=Path,
        default=None,
        help="Official JudgeBench pairwise JSONL. Defaults to the locked train run value or downloads the canonical file.",
    )
    parser.add_argument(
        "--validation-split-name",
        type=str,
        default="validation_270",
        help="Artifact directory / summary name for the held-out split (default: validation_270).",
    )
    parser.add_argument(
        "--write-detailed-outputs",
        action="store_true",
        help="Write per-example held-out artifacts and prediction reports. Off by default to avoid leakage-prone outputs.",
    )
    parser.add_argument(
        "--allow-reference-answer",
        action="store_true",
        help="Expose held-out reference answers during final evaluation. Off by default; enabling this makes the run reference-assisted rather than blind.",
    )
    parser.add_argument(
        "--retrieval-profile",
        choices=sorted(_ALLOWED_RETRIEVAL_PROFILES),
        default=None,
        help="Optional override for the locked policy's retrieval profile.",
    )
    parser.add_argument(
        "--retrieval-family-profile",
        action="append",
        default=[],
        help="Per-family retrieval override in 'family=profile' form. Repeatable.",
    )
    parser.add_argument(
        "--retrieval-family-top-k",
        action="append",
        default=[],
        help="Per-family retrieval top-k override in 'family=int' form. Repeatable.",
    )
    parser.add_argument(
        "--blind-discriminator-family-mode",
        action="append",
        default=[],
        help="Per-family blind discriminator mode in 'family=default|off|strict' form. Repeatable.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under artifacts/compiled_judgebench_final_eval_runs/.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override artifacts root (default: <repo>/artifacts/compiled_judgebench_final_eval_runs).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel JudgeBench examples to process for final evaluation (default: 4).",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse already-written detailed held-out artifacts if present.")
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=None,
        help="Optional shared cache directory for discovery / scoring JSONL caches. When set, multiple "
        "final-eval runs against the same locked policy share this cache; cache hits dramatically "
        "reduce re-roll variance from cache invalidation across phases.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    root = _repo_root()
    command = raw_args[0] if raw_args and raw_args[0] in {"train-only", "final-eval"} else "legacy-full"
    command_args = raw_args[1:] if command != "legacy-full" else raw_args

    if command == "train-only":
        parser = _build_train_only_parser()
        args = parser.parse_args(command_args)
        train_dataset = args.train_dataset or (root / "data" / "judgebench_80_human.json")
        out_root = args.out_root or (root / "artifacts" / "compiled_judgebench_train_only_runs")
        run_name = args.run_name or _default_run_name("judgebench_train_only")
        run_dir = out_root / run_name
        recursive_config = _build_recursive_config(args)
        retrieval_profile_by_family = _parse_family_overrides(
            list(args.retrieval_family_profile or []),
            value_kind="retrieval profile",
        )
        retrieval_top_k_by_family = _parse_family_overrides(
            list(args.retrieval_family_top_k or []),
            value_kind="retrieval top-k",
        )
        blind_discriminator_mode_by_family = _parse_family_overrides(
            list(args.blind_discriminator_family_mode or []),
            value_kind="blind discriminator mode",
        )
        train_reference_answer_access = bool(args.allow_train_reference_answer)
        if args.no_train_reference_answer:
            train_reference_answer_access = False
        v2_config = _v2_config_from_args(args)
        _, summary = run_judgebench_train_only_development(
            train_dataset_path=train_dataset,
            train_split_name=args.train_split_name,
            run_dir=run_dir,
            official_dataset_path=args.official_dataset,
            discovery_model_override=args.discovery_model,
            judge_model_override=args.judge_model,
            use_cache=not args.no_cache,
            max_criteria=max(1, min(args.max_criteria, 24)),
            max_pairs_per_example=max(2, args.max_pairs_per_example),
            bootstrap_iterations=max(1, args.bootstrap_iterations),
            recursive_config=recursive_config,
            covariance_ridge=max(1e-6, float(args.covariance_ridge)),
            max_workers=max(1, args.max_workers),
            fold_count=max(2, int(args.fold_count)),
            fold_shuffle_seed=args.fold_shuffle_seed,
            protocol_mode=args.protocol_mode,
            resume=args.resume,
            train_reference_answer_access=train_reference_answer_access,
            oof_reference_answer_access=args.allow_dev_reference_answer,
            write_train_fit=args.write_train_fit,
            train_fit_reference_answer_access=args.allow_train_fit_reference_answer,
            blind_scoring_profile=args.blind_scoring_profile,
            blind_budget_profile=args.blind_budget_profile,
            blind_guidance_profile=args.blind_guidance_profile,
            blind_wu_profile=args.blind_wu_profile,
            retrieval_profile=args.retrieval_profile,
            retrieval_top_k=max(1, int(args.retrieval_top_k)),
            blind_discriminator_mode_by_family=blind_discriminator_mode_by_family,
            retrieval_profile_by_family=retrieval_profile_by_family,
            retrieval_top_k_by_family=retrieval_top_k_by_family,
            v2_config=v2_config,
        )
        print(f"Wrote JudgeBench train-only artifacts to: {run_dir}")
        train_fit_summary = dict(summary.get("train_fit_summary") or {})
        train_fit_fragment = ""
        if train_fit_summary:
            train_fit_fragment = f" train_fit={float((train_fit_summary.get('wu_metrics') or {}).get('overall', 0.0)):.2f}"
        locked_alignment = dict(summary.get("locked_policy_alignment") or {})
        locked_gap_fragment = ""
        if locked_alignment.get("train_fit_available"):
            locked_gap_fragment = (
                f" locked_gap={float(locked_alignment.get('locked_train_fit_minus_oof_wu') or 0.0):.2f}"
            )
        print(
            f"{args.train_split_name}_oof={summary['oof_summary']['wu_metrics']['overall']:.2f}"
            f"{train_fit_fragment}{locked_gap_fragment} mechanism_hash={summary['mechanism_hash']}"
        )
        return

    if command == "final-eval":
        parser = _build_final_eval_parser()
        args = parser.parse_args(command_args)
        validation_dataset = args.validation_dataset or (root / "data" / "judgebench_270_generated.json")
        out_root = args.out_root or (root / "artifacts" / "compiled_judgebench_final_eval_runs")
        run_name = args.run_name or _default_run_name("judgebench_final_eval")
        run_dir = out_root / run_name
        retrieval_profile_by_family = _parse_family_overrides(
            list(args.retrieval_family_profile or []),
            value_kind="retrieval profile",
        )
        retrieval_top_k_by_family = _parse_family_overrides(
            list(args.retrieval_family_top_k or []),
            value_kind="retrieval top-k",
        )
        blind_discriminator_mode_by_family = _parse_family_overrides(
            list(args.blind_discriminator_family_mode or []),
            value_kind="blind discriminator mode",
        )
        _, summary = run_judgebench_final_evaluation(
            train_run_dir=args.train_run_dir,
            validation_dataset_path=validation_dataset,
            validation_split_name=args.validation_split_name,
            run_dir=run_dir,
            official_dataset_path=args.official_dataset,
            max_workers=max(1, args.max_workers),
            write_detailed_outputs=args.write_detailed_outputs,
            resume=args.resume,
            reference_answer_access=args.allow_reference_answer,
            retrieval_profile=args.retrieval_profile,
            retrieval_profile_by_family=retrieval_profile_by_family,
            retrieval_top_k_by_family=retrieval_top_k_by_family,
            blind_discriminator_mode_by_family=blind_discriminator_mode_by_family,
            shared_cache_dir=args.shared_cache_dir,
        )
        print(f"Wrote JudgeBench final-eval artifacts to: {run_dir}")
        diagnostic_fragments = []
        for subset_name, payload in sorted(dict(summary.get("diagnostic_subsets") or {}).items()):
            subset_summary = dict((payload or {}).get("summary", {}) or {})
            if subset_summary:
                diagnostic_fragments.append(
                    f"{subset_name}={float((subset_summary.get('wu_metrics') or {}).get('overall', 0.0)):.2f}"
                )
        diagnostics_suffix = f" {' '.join(diagnostic_fragments)}" if diagnostic_fragments else ""
        print(f"{args.validation_split_name}={summary['validation_summary']['wu_metrics']['overall']:.2f}{diagnostics_suffix}")
        return

    parser = _build_legacy_parser()
    args = parser.parse_args(command_args)
    train_dataset = args.train_dataset or (root / "data" / "judgebench_80_human.json")
    validation_dataset = args.validation_dataset or (root / "data" / "judgebench_270_generated.json")
    out_root = args.out_root or (root / "artifacts" / "compiled_judgebench_runs")
    run_name = args.run_name or _default_run_name("judgebench_eval")
    run_dir = out_root / run_name
    recursive_config = _build_recursive_config(args)
    _, summary = run_judgebench_recursive_evaluation(
        train_dataset_path=train_dataset,
        validation_dataset_path=validation_dataset,
        train_split_name=args.train_split_name,
        validation_split_name=args.validation_split_name,
        protocol_mode=args.protocol_mode,
        run_dir=run_dir,
        official_dataset_path=args.official_dataset,
        discovery_model_override=args.discovery_model,
        judge_model_override=args.judge_model,
        use_cache=not args.no_cache,
        max_criteria=max(1, min(args.max_criteria, 24)),
        max_pairs_per_example=max(2, args.max_pairs_per_example),
        bootstrap_iterations=max(1, args.bootstrap_iterations),
        refine_iterations=max(0, args.refine_iterations),
        recursive_config=recursive_config,
        covariance_ridge=max(1e-6, float(args.covariance_ridge)),
        max_workers=max(1, args.max_workers),
        resume=args.resume,
    )
    print(f"Wrote JudgeBench evaluation artifacts to: {run_dir}")
    print(
        f"{args.train_split_name}={summary['train_summary']['wu_metrics']['overall']:.2f} "
        f"{args.validation_split_name}={summary['validation_summary']['wu_metrics']['overall']:.2f} "
        f"accepted_refinements={summary['accepted_refinement_count']}"
    )


if __name__ == "__main__":
    main()
