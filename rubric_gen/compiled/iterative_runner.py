"""
Provisional closed-loop runner: design discovery → augmented ontology → pilot evaluation (starter scaffold).

This wires local strong/weak discovery into additive ontology templates and reruns the compiled pilot.
It is not a production recursive rubric system — only an experiment harness.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rubric_gen.compiled.compiler import build_task_ontology
from rubric_gen.compiled.discovered_augmentation import build_augmented_ontology
from rubric_gen.compiled.discovery import run_discovery_for_examples
from rubric_gen.compiled.pilot_runner import _split_slices, run_pilot
from rubric_gen.compiled.profile_bootstrap import resolve_or_bootstrap_task_profile
from rubric_gen.compiled.serialize import to_json_dict, write_json
from rubric_gen.dataio import load_examples
from rubric_gen.types import ExampleRecord


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_iterative_loop(
    *,
    dataset_path: Path,
    run_name: str,
    start: int,
    limit: int,
    design_n: int,
    validation_n: int,
    pilot_n: int,
    source_filter: str | None,
    out_root: Path | None,
    support_threshold: int,
    discovery_model: str | None,
    discovery_use_cache: bool,
    discovery_max_criteria: int,
    discovery_max_pairs_per_example: int | None,
    pilot_write_csv: bool,
    judge_mode: str,
    judge_model: str | None,
    no_llm_cache: bool,
    task_profile: str | None,
    bootstrap_iterations: int,
) -> Path:
    """
    1) Load examples and split into design / validation / pilot.
    2) Run discovery on the design slice only.
    3) Build an augmented ontology from merged proposals (support threshold).
    4) Run the compiled pilot on all slices using the augmented ontology.
    """
    root = _repo_root()
    run_dir = (out_root or (root / "artifacts" / "compiled_iterative_runs")) / run_name
    discovery_dir = run_dir / "discovery"
    ontology_dir = run_dir / "ontology"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    ontology_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(dataset_path, start=start, limit=limit, source_filter=source_filter)
    if not examples:
        raise SystemExit("No examples loaded; check dataset path, --start, --limit, and --source-filter.")
    profile_resolution = resolve_or_bootstrap_task_profile(
        examples,
        explicit=task_profile,
        bootstrap_iterations=bootstrap_iterations,
    )
    resolved_task_profile = profile_resolution.profile.task_profile_id

    design_ex, val_ex, pilot_ex = _split_slices(examples, design_n, validation_n, pilot_n)
    slices: Dict[str, List[ExampleRecord]] = {
        "design": design_ex,
        "validation": val_ex,
        "pilot": pilot_ex,
    }

    _, disc_summary = run_discovery_for_examples(
        design_ex,
        run_dir=discovery_dir,
        model_override=discovery_model,
        use_cache=discovery_use_cache,
        max_criteria=discovery_max_criteria,
        max_pairs_per_example=discovery_max_pairs_per_example,
        task_profile=resolved_task_profile,
        bootstrap_iterations=bootstrap_iterations,
    )

    merged_path = discovery_dir / "summaries" / "merged_proposals.json"
    with merged_path.open("r", encoding="utf-8") as f:
        merged = json.load(f)

    base_ont = build_task_ontology(resolved_task_profile)
    write_json(ontology_dir / "base_ontology.json", base_ont)

    aug_ontology, selected = build_augmented_ontology(
        merged_proposals=merged,
        design_examples=design_ex,
        support_threshold=support_threshold,
        task_profile_id=resolved_task_profile,
    )
    write_json(ontology_dir / "augmented_ontology.json", aug_ontology)
    write_json(
        ontology_dir / "selected_discovered_templates.json",
        {
            "schema": "compiled_selected_discovered_templates_v1",
            "disclaimer": "Provisional templates from merged discovery — not human-reviewed.",
            "support_threshold": support_threshold,
            "templates": selected,
            "count": len(selected),
        },
    )

    # Nested folder name "pilot" keeps artifacts at <run_dir>/pilot/{examples,summaries,cache}/.
    run_pilot(
        dataset_path=dataset_path,
        run_name="pilot",
        start=start,
        limit=limit,
        design_n=design_n,
        validation_n=validation_n,
        pilot_n=pilot_n,
        source_filter=source_filter,
        out_root=run_dir,
        write_csv=pilot_write_csv,
        judge_mode=judge_mode,
        judge_model=judge_model,
        no_llm_cache=no_llm_cache,
        ontology=aug_ontology,
        task_profile=resolved_task_profile,
        bootstrap_iterations=bootstrap_iterations,
    )

    top_summary: Dict[str, Any] = {
        "schema": "compiled_iterative_run_summary_v1",
        "disclaimer": (
            "Provisional closed-loop scaffold — design discovery feeds additive ontology templates; "
            "not a validated recursive system."
        ),
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path.resolve()),
        "params": {
            "start": start,
            "limit": limit,
            "design_n": design_n,
            "validation_n": validation_n,
            "pilot_n": pilot_n,
            "source_filter": source_filter,
            "task_profile": resolved_task_profile,
            "bootstrap_iterations": bootstrap_iterations,
            "support_threshold": support_threshold,
            "discovery_max_criteria": discovery_max_criteria,
            "discovery_max_pairs_per_example": discovery_max_pairs_per_example,
            "judge_mode": judge_mode,
            "judge_model": judge_model,
        },
        "profile_resolution": {
            "bootstrap_used": profile_resolution.bootstrap_used,
            "iterations_run": profile_resolution.iterations_run,
            "resolved_task_profile_id": profile_resolution.profile.task_profile_id,
            "parent_profile_id": profile_resolution.profile.parent_profile_id,
            "diagnostics": profile_resolution.diagnostics,
        },
        "slices": {k: [e.example_id for e in v] for k, v in slices.items()},
        "paths": {
            "discovery": str(discovery_dir.resolve()),
            "ontology": str(ontology_dir.resolve()),
            "pilot": str((run_dir / "pilot").resolve()),
        },
        "discovery_summary_ref": str((discovery_dir / "summaries" / "run_summary.json").resolve()),
        "merged_proposals_ref": str(merged_path.resolve()),
        "pilot_summary_ref": str((run_dir / "pilot" / "summaries" / "run_summary.json").resolve()),
        "ontology_version": aug_ontology.version,
        "discovered_templates_selected": len(selected),
        "discovery_run": to_json_dict(disc_summary),
    }
    write_json(run_dir / "run_summary.json", top_summary)

    return run_dir


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Provisional iterative runner: discovery on design slice → augmented ontology → pilot "
            "(starter scaffold; not a production closed loop)."
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
        help="Run directory name under artifacts/compiled_iterative_runs/",
    )
    parser.add_argument("--out-root", type=Path, default=None, help="Override artifacts root")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Max examples to load (0 = all)")
    parser.add_argument("--source-filter", type=str, default=None)
    parser.add_argument("--design", type=int, default=100)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--pilot", type=int, default=300)
    parser.add_argument(
        "--support-threshold",
        type=int,
        default=1,
        help="Minimum merged support count for a canonical proposal to become a template (default: 1).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Low-cost run: limit=2, design=1, validation=0, pilot=1, 1 pair, ≤4 criteria.",
    )
    parser.add_argument("--discovery-model", type=str, default=None, help="provider:model for discovery LLM")
    parser.add_argument("--no-discovery-cache", action="store_true")
    parser.add_argument("--max-criteria", type=int, default=8)
    parser.add_argument("--max-pairs-per-example", type=int, default=None)
    parser.add_argument("--csv", action="store_true", help="Pilot writes summaries/candidate_evaluations.csv")
    parser.add_argument("--judge-mode", choices=("heuristic", "llm", "both"), default="heuristic")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--no-llm-cache", action="store_true")
    parser.add_argument("--task-profile", type=str, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=3)

    args = parser.parse_args(argv)

    root = _repo_root()
    dataset = args.dataset or (root / "data" / "sample_100_aci_400_agbonnet.json")

    design_n, validation_n, pilot_n = args.design, args.validation, args.pilot
    start, limit = args.start, args.limit
    max_pairs: Optional[int] = args.max_pairs_per_example
    max_criteria = max(1, min(args.max_criteria, 24))

    if args.smoke:
        limit = 2 if limit == 0 else limit
        design_n, validation_n, pilot_n = 1, 0, 1
        max_pairs = 1 if max_pairs is None else min(max_pairs, 1)
        max_criteria = min(max_criteria, 4)

    run_name = args.run_name or f"iter_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    out = run_iterative_loop(
        dataset_path=dataset,
        run_name=run_name,
        start=start,
        limit=limit,
        design_n=design_n,
        validation_n=validation_n,
        pilot_n=pilot_n,
        source_filter=args.source_filter,
        out_root=args.out_root,
        support_threshold=max(1, args.support_threshold),
        discovery_model=args.discovery_model,
        discovery_use_cache=not args.no_discovery_cache,
        discovery_max_criteria=max_criteria,
        discovery_max_pairs_per_example=max_pairs,
        pilot_write_csv=args.csv,
        judge_mode=args.judge_mode,
        judge_model=args.judge_model,
        no_llm_cache=args.no_llm_cache,
        task_profile=args.task_profile,
        bootstrap_iterations=max(1, args.bootstrap_iterations),
    )
    print(f"Wrote iterative run artifacts to: {out}")


if __name__ == "__main__":
    main()
