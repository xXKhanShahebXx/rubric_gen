"""
CLI for starter compiled-rubric local discovery (strong vs weak note pairs → atomic criteria proposals).

Run: python -m rubric_gen.compiled.discovery_runner --smoke
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from rubric_gen.compiled.discovery import run_discovery_for_examples
from rubric_gen.dataio import load_examples


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Starter compiled-rubric discovery: propose local atomic criteria from strong/weak pairs "
            "(scaffold; not recursive ontology decomposition)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset JSON with rows[] (default: data/sample_100_aci_400_agbonnet.json)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under artifacts/compiled_discovery_runs/",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override artifacts root (default: <repo>/artifacts/compiled_discovery_runs)",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Max examples to load (0 = all)")
    parser.add_argument("--source-filter", type=str, default=None, help="Substring filter on row source field")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM as provider:model (default: RUBRIC_GEN_COMPILED_JUDGE_MODEL or discover_default_judge_model).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Jsonl cache for discovery LLM responses (under run_dir/cache/).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Low-cost run: limit=1, max 1 pair per example, at most 4 criteria per pair.",
    )
    parser.add_argument(
        "--max-pairs-per-example",
        type=int,
        default=None,
        metavar="N",
        help="Cap contrast pairs per example (default: no cap).",
    )
    parser.add_argument(
        "--max-criteria",
        type=int,
        default=8,
        help="Max atomic criteria requested per pair in the LLM response (default: 8).",
    )
    parser.add_argument(
        "--task-profile",
        type=str,
        default=None,
        help="Override the task profile or use `auto` to force runtime bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=3,
        help="Maximum refinement passes when auto-bootstrapping a task profile.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    dataset = args.dataset or (root / "data" / "sample_100_aci_400_agbonnet.json")

    start = args.start
    limit = args.limit
    max_pairs: Optional[int] = args.max_pairs_per_example
    max_criteria = max(1, min(args.max_criteria, 24))

    if args.smoke:
        if limit == 0:
            limit = 1
        max_pairs = 1 if max_pairs is None else min(max_pairs, 1)
        max_criteria = min(max_criteria, 4)

    out_root = args.out_root or (root / "artifacts" / "compiled_discovery_runs")
    run_name = args.run_name or f"discovery_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_name

    examples = load_examples(dataset, start=start, limit=limit, source_filter=args.source_filter)
    if not examples:
        raise SystemExit("No examples loaded; check dataset path, --start, --limit, and --source-filter.")

    _, summary = run_discovery_for_examples(
        examples,
        run_dir=run_dir,
        model_override=args.model,
        use_cache=not args.no_cache,
        max_criteria=max_criteria,
        max_pairs_per_example=max_pairs,
        task_profile=args.task_profile,
        bootstrap_iterations=max(1, args.bootstrap_iterations),
    )

    print(f"Wrote discovery artifacts to: {run_dir}")
    print(
        f"pairs_total={summary['stats']['pairs_total']} "
        f"local_proposals_raw={summary['stats']['local_proposals_total']} "
        f"local_proposals_promoted={summary['stats']['local_proposals_promoted']} "
        f"merged_unique={summary['merged']['unique_canonical_count']}"
    )


if __name__ == "__main__":
    main()
