"""
CLI entry point that runs the rubric pipeline against ``allenai/reward-bench-2``.

Usage (smoke test, ~30 items):

    python -m rubric_gen.compiled.reward_bench_2_runner \
        --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 \
        --run-name rb2_smoke \
        --items-per-subset 5

Usage (full eval, all 1865 items):

    python -m rubric_gen.compiled.reward_bench_2_runner \
        --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 \
        --run-name rb2_full \
        --shared-cache-dir artifacts/shared_cache_v35

Steps the runner performs:

1. Download ``allenai/reward-bench-2`` (cached after first call).
2. Filter / cap items per subset (smoke test or scale).
3. Expand each item to pairwise rows via :mod:`reward_bench_2_loader`.
4. Persist a ``validation_<run_name>.json`` and ``official_<run_name>.jsonl`` shaped like
   the existing JudgeBench artifacts, so :func:`run_judgebench_final_evaluation` can
   consume them unchanged.
5. Invoke the locked-policy final-eval against those files.
6. Read every per-pair artifact and aggregate into a per-subset RewardBench 2 summary
   (best-of-4 success per item + Ties weighted score). Persist to
   ``rb2_summary.json`` next to the run dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from rubric_gen.compiled.judgebench_eval import run_judgebench_final_evaluation
from rubric_gen.compiled.reward_bench_2_loader import (
    RewardBench2PairRow,
    expand_items_to_pairs,
    load_reward_bench_2,
    write_joined_dataset,
    write_official_jsonl,
)
from rubric_gen.compiled.reward_bench_2_metrics import (
    aggregate_pair_artifacts,
    load_artifacts_from_run,
)


_DEFAULT_OUT_ROOT = Path("artifacts/compiled_judgebench_final_eval_runs")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the rubric pipeline on RewardBench 2 (best-of-4 → 3 pairwise).",
    )
    parser.add_argument(
        "--train-run-dir",
        type=Path,
        required=True,
        help="Train-only run dir whose locked policy will be used (frozen).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Output run name (will create artifacts/.../<run-name>).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Final-eval root (defaults to artifacts/compiled_judgebench_final_eval_runs).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help=(
            "Optional list of subsets to keep "
            "(Factuality / 'Precise IF' / Math / Safety / Focus / Ties). "
            "Default: all six."
        ),
    )
    parser.add_argument(
        "--items-per-subset",
        type=int,
        default=None,
        help="Cap per-subset items (smoke testing).",
    )
    parser.add_argument(
        "--max-rejected",
        type=int,
        default=None,
        help="Cap rejected completions per item (Ties default 0 = all; non-Ties always 3).",
    )
    parser.add_argument(
        "--max-chosen",
        type=int,
        default=None,
        help="Cap chosen completions per item (Ties only; non-Ties always 1).",
    )
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=None,
        help="Optional shared cache dir for prompt-content cache reuse.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--retrieval-profile",
        type=str,
        default="library_v1_plus_family_v1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already-written per-pair artifacts when re-running (useful for "
        "long-running full evals).",
    )
    return parser


def _persist_dataset(
    rows: Sequence[RewardBench2PairRow],
    *,
    out_dataset_dir: Path,
    split_name: str,
) -> tuple[Path, Path]:
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    validation_path = out_dataset_dir / f"validation_{split_name}.json"
    official_path = out_dataset_dir / f"official_{split_name}.jsonl"
    write_joined_dataset(rows, validation_path)
    write_official_jsonl(rows, official_path)
    return validation_path, official_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    out_root = Path(args.out_root) if args.out_root else _DEFAULT_OUT_ROOT
    run_dir = out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[rb2] downloading allenai/reward-bench-2 ...", flush=True)
    ds = load_reward_bench_2()
    items = list(ds)
    print(f"[rb2] loaded {len(items)} items", flush=True)

    print("[rb2] expanding to pairwise rows ...", flush=True)
    rows = expand_items_to_pairs(
        items,
        split_name=args.run_name,
        max_rejected=args.max_rejected,
        max_chosen=args.max_chosen,
        subsets=args.subsets,
        items_per_subset=args.items_per_subset,
    )
    print(
        f"[rb2] expanded to {len(rows)} pairwise rows from "
        f"{len({r.rb2_item_id for r in rows})} items",
        flush=True,
    )
    if not rows:
        print("[rb2] no rows to evaluate; exiting", flush=True)
        return 1

    dataset_dir = run_dir / "rb2_dataset"
    validation_path, official_path = _persist_dataset(
        rows,
        out_dataset_dir=dataset_dir,
        split_name=args.run_name,
    )
    print(
        f"[rb2] wrote {validation_path.name} ({validation_path.stat().st_size} bytes)",
        flush=True,
    )

    print("[rb2] launching final-eval ...", flush=True)
    _, summary = run_judgebench_final_evaluation(
        train_run_dir=args.train_run_dir,
        validation_dataset_path=validation_path,
        validation_split_name=args.run_name,
        run_dir=run_dir,
        official_dataset_path=official_path,
        max_workers=max(1, int(args.max_workers)),
        write_detailed_outputs=True,
        resume=bool(args.resume),
        reference_answer_access=False,
        retrieval_profile=args.retrieval_profile,
        shared_cache_dir=args.shared_cache_dir,
    )

    print("[rb2] aggregating per-subset RewardBench 2 metrics ...", flush=True)
    artifacts = load_artifacts_from_run(run_dir)
    rb2_summary = aggregate_pair_artifacts(artifacts)
    summary_path = run_dir / "rb2_summary.json"
    summary_path.write_text(
        json.dumps(rb2_summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(f"RewardBench 2 results for {args.run_name}")
    print("=" * 70)
    for name, ss in sorted(rb2_summary.subset_summaries.items()):
        print(
            f"  {ss.subset:<12s}: items={ss.items_correct}/{ss.item_count} "
            f"({ss.accuracy_pct:>5.2f}%)  pairs={ss.pairs_correct}/{ss.pair_count} "
            f"({ss.pair_accuracy_pct:>5.2f}%)"
        )
    print()
    print(
        f"  Leaderboard avg (5-subset, no Ties): "
        f"{rb2_summary.leaderboard_average_pct:.2f}%"
    )
    if rb2_summary.ties_weighted_score is not None:
        print(
            f"  Ties weighted score (proxy):        "
            f"{rb2_summary.ties_weighted_score:.2f}%  "
            f"(acc={rb2_summary.ties_accuracy_term:.2f}, "
            f"margin={rb2_summary.ties_margin_term:.2f})"
        )
    print(f"  Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
