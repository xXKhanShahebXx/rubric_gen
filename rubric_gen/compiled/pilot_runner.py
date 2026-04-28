"""
Starter compiled-rubric pilot runner (scaffold).

End-to-end slice: load examples, compile case rubrics, build contrast candidates, run heuristic and/or
LLM judges, and write run artifacts. This is not a full discovery or adjudication system.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rubric_gen.compiled.compiler import (
    build_task_family_spec,
    build_task_ontology,
    compile_task_case_rubric,
    infer_task_family,
)
from rubric_gen.compiled.contrast_strategies import build_task_contrast_candidates, is_synthetic_candidate
from rubric_gen.compiled.heuristic_judge import evaluate_artifact_against_rubric
from rubric_gen.compiled.llm_judge import evaluate_artifact_with_llm_judge, resolve_compiled_judge_spec
from rubric_gen.compiled.profile_bootstrap import resolve_or_bootstrap_task_profile
from rubric_gen.compiled.reporting import (
    PilotRunStats,
    eval_hard_failed,
    review_queue_reasons,
)
from rubric_gen.compiled.schema import CaseEvaluationRecord, RubricOntology
from rubric_gen.compiled.serialize import to_json_dict, write_json
from rubric_gen.dataio import example_to_prompt, load_examples
from rubric_gen.llm_client import LLMRouter
from rubric_gen.storage import JsonlCache
from rubric_gen.types import ExampleRecord


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_slices(
    examples: List[ExampleRecord], design_n: int, val_n: int, pilot_n: int
) -> Tuple[List[ExampleRecord], List[ExampleRecord], List[ExampleRecord]]:
    total = design_n + val_n + pilot_n
    pool = examples[:total]
    return pool[:design_n], pool[design_n : design_n + val_n], pool[design_n + val_n :]


def _slice_name_for_example(example_id: str, slices: Dict[str, List[ExampleRecord]]) -> str:
    for name, rows in slices.items():
        if any(e.example_id == example_id for e in rows):
            return name
    return "unknown"


def _decision_to_bucket(overall: str) -> str:
    if overall == "sft_include":
        return "gold_sft"
    if overall == "repair":
        return "repair"
    if overall == "do_not_train":
        return "do_not_train"
    return overall


def run_pilot(
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
    write_csv: bool,
    judge_mode: str = "heuristic",
    judge_model: str | None = None,
    no_llm_cache: bool = False,
    ontology: RubricOntology | None = None,
    task_profile: str | None = None,
    bootstrap_iterations: int = 3,
) -> Path:
    root = _repo_root()
    examples = load_examples(dataset_path, start=start, limit=limit, source_filter=source_filter)
    design_ex, val_ex, pilot_ex = _split_slices(examples, design_n, validation_n, pilot_n)
    slices: Dict[str, List[ExampleRecord]] = {
        "design": design_ex,
        "validation": val_ex,
        "pilot": pilot_ex,
    }
    to_process = design_ex + val_ex + pilot_ex

    run_dir = (out_root or (root / "artifacts" / "compiled_runs")) / run_name
    ex_dir = run_dir / "examples"
    sum_dir = run_dir / "summaries"
    ex_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    if not to_process:
        raise SystemExit("No examples loaded; check dataset path and slicing parameters.")
    profile_resolution = resolve_or_bootstrap_task_profile(
        to_process,
        explicit=task_profile,
        bootstrap_iterations=bootstrap_iterations,
    )
    resolved_task_profile = profile_resolution.profile.task_profile_id
    ontology_resolved = ontology if ontology is not None else build_task_ontology(resolved_task_profile)
    stats = PilotRunStats()

    llm_spec = None
    router: Optional[LLMRouter] = None
    llm_cache: Optional[JsonlCache] = None
    if judge_mode in ("llm", "both"):
        llm_spec = resolve_compiled_judge_spec(judge_model)
        router = LLMRouter()
        llm_cache = JsonlCache(
            run_dir / "cache" / "compiled_llm_judge.jsonl",
            enabled=not no_llm_cache,
        )

    gold_rows: List[Dict[str, Any]] = []
    repair_rows: List[Dict[str, Any]] = []
    dnt_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    review_queue: List[Dict[str, Any]] = []

    for ex in to_process:
        stats.examples_processed += 1
        profile = profile_resolution.profile
        stats.task_profile_counts[profile.task_profile_id] = (
            stats.task_profile_counts.get(profile.task_profile_id, 0) + 1
        )
        nf_id = infer_task_family(ex, task_profile_id=profile.task_profile_id)
        stats.note_family_counts[nf_id] = stats.note_family_counts.get(nf_id, 0) + 1
        stats.task_family_counts[nf_id] = stats.task_family_counts.get(nf_id, 0) + 1

        note_family = build_task_family_spec(
            nf_id,
            ontology_resolved,
            task_profile_id=profile.task_profile_id,
        )
        case_rubric = compile_task_case_rubric(
            ex,
            ontology_resolved,
            note_family,
            task_profile_id=profile.task_profile_id,
        )
        candidates = build_task_contrast_candidates(ex, task_profile_id=profile.task_profile_id)
        task_context = example_to_prompt(ex)

        slice_name = _slice_name_for_example(ex.example_id, slices)
        per_cand: List[Dict[str, Any]] = []

        for cand in candidates:
            stats.candidates_evaluated += 1
            syn = is_synthetic_candidate(cand)
            if syn:
                stats.synthetic_candidates += 1
            else:
                stats.original_candidates += 1

            ev_h: Optional[CaseEvaluationRecord] = None
            ev_l: Optional[CaseEvaluationRecord] = None
            llm_hit = False

            if judge_mode in ("heuristic", "both"):
                ev_h = evaluate_artifact_against_rubric(
                    candidate_id=cand.candidate_id,
                    artifact_text=cand.text,
                    task_context=task_context,
                    case_rubric=case_rubric,
                )
            if judge_mode in ("llm", "both") and llm_spec and router:
                ev_l, llm_hit = evaluate_artifact_with_llm_judge(
                    candidate_id=cand.candidate_id,
                    artifact_text=cand.text,
                    task_context=task_context,
                    case_rubric=case_rubric,
                    model_spec=llm_spec,
                    router=router,
                    cache=llm_cache,
                )
                if llm_hit:
                    stats.llm_cache_hits += 1

            if judge_mode == "heuristic":
                primary = ev_h
            elif judge_mode == "llm":
                primary = ev_l
            else:
                primary = ev_h

            assert primary is not None

            gate_ev = ev_l if judge_mode == "llm" else ev_h
            if gate_ev is not None and eval_hard_failed(gate_ev):
                stats.hard_gate_failures += 1

            if judge_mode == "both" and ev_h is not None and ev_l is not None:
                disagree = (ev_h.overall_decision != ev_l.overall_decision) or (
                    eval_hard_failed(ev_h) != eval_hard_failed(ev_l)
                )
                if disagree:
                    stats.judge_disagreements += 1

            bucket_source = ev_h if judge_mode in ("heuristic", "both") else ev_l
            assert bucket_source is not None
            bucket = _decision_to_bucket(bucket_source.overall_decision)

            if not syn:
                if bucket == "gold_sft":
                    stats.decision_gold_sft += 1
                    gold_rows.append(
                        {
                            "example_id": ex.example_id,
                            "candidate_id": cand.candidate_id,
                            "source_label": cand.source_label,
                            "note_family_id": nf_id,
                            "task_profile_id": profile.task_profile_id,
                            "task_family_id": nf_id,
                            "slice": slice_name,
                            "overall_decision": bucket_source.overall_decision,
                        }
                    )
                elif bucket == "repair":
                    stats.decision_repair += 1
                    repair_rows.append(
                        {
                            "example_id": ex.example_id,
                            "candidate_id": cand.candidate_id,
                            "source_label": cand.source_label,
                            "note_family_id": nf_id,
                            "task_profile_id": profile.task_profile_id,
                            "task_family_id": nf_id,
                            "slice": slice_name,
                            "overall_decision": bucket_source.overall_decision,
                        }
                    )
                elif bucket == "do_not_train":
                    stats.decision_do_not_train += 1
                    dnt_rows.append(
                        {
                            "example_id": ex.example_id,
                            "candidate_id": cand.candidate_id,
                            "source_label": cand.source_label,
                            "note_family_id": nf_id,
                            "task_profile_id": profile.task_profile_id,
                            "task_family_id": nf_id,
                            "slice": slice_name,
                            "overall_decision": bucket_source.overall_decision,
                        }
                    )

            row_csv: Dict[str, Any] = {
                "example_id": ex.example_id,
                "candidate_id": cand.candidate_id,
                "slice": slice_name,
                "synthetic": syn,
                "note_family_id": nf_id,
                "task_profile_id": profile.task_profile_id,
                "task_family_id": nf_id,
                "source_label": cand.source_label,
                "overall_decision": primary.overall_decision,
                "decision_bucket": _decision_to_bucket(primary.overall_decision),
                "hard_gate_failed": eval_hard_failed(primary),
                "judge_mode": judge_mode,
            }
            if ev_l is not None:
                row_csv["llm_overall_decision"] = ev_l.overall_decision
                row_csv["llm_hard_gate_failed"] = eval_hard_failed(ev_l)
            if judge_mode == "both" and ev_h is not None and ev_l is not None:
                row_csv["heuristic_overall_decision"] = ev_h.overall_decision
                row_csv["judges_disagree"] = (ev_h.overall_decision != ev_l.overall_decision) or (
                    eval_hard_failed(ev_h) != eval_hard_failed(ev_l)
                )
            csv_rows.append(row_csv)

            cand_payload: Dict[str, Any] = {
                "candidate": to_json_dict(cand),
                "evaluation": to_json_dict(primary),
            }
            if judge_mode == "both" and ev_l is not None:
                cand_payload["evaluation_llm"] = to_json_dict(ev_l)
            if judge_mode == "both" and ev_h is not None and ev_l is not None:
                cand_payload["judge_comparison"] = {
                    "heuristic_overall": ev_h.overall_decision,
                    "llm_overall": ev_l.overall_decision,
                    "heuristic_hard_failed": eval_hard_failed(ev_h),
                    "llm_hard_failed": eval_hard_failed(ev_l),
                    "disagree_overall": ev_h.overall_decision != ev_l.overall_decision,
                    "disagree_hard_gate": eval_hard_failed(ev_h) != eval_hard_failed(ev_l),
                }
            per_cand.append(cand_payload)

            if not syn:
                reasons, detail = review_queue_reasons(
                    heuristic_ev=ev_h,
                    llm_ev=ev_l,
                    judge_mode=judge_mode,
                )
                if reasons:
                    review_queue.append(
                        {
                            "example_id": ex.example_id,
                            "candidate_id": cand.candidate_id,
                            "slice": slice_name,
                            "note_family_id": nf_id,
                            "task_profile_id": profile.task_profile_id,
                            "task_family_id": nf_id,
                            "source_label": cand.source_label,
                            "reasons": reasons,
                            "detail": detail,
                        }
                    )

        safe_id = ex.example_id.replace("/", "_")
        example_payload = {
            "schema": "compiled_pilot_example_v1",
            "disclaimer": "Starter pilot scaffold — not full discovery or adjudication.",
            "example_id": ex.example_id,
            "slice": slice_name,
            "note_family_id": nf_id,
            "task_profile_id": profile.task_profile_id,
            "task_family_id": nf_id,
            "case_rubric": to_json_dict(case_rubric),
            "candidates": per_cand,
            "judge_mode": judge_mode,
        }
        write_json(ex_dir / f"{safe_id}.json", example_payload)

    write_json(sum_dir / "subset_gold_sft.json", {"rows": gold_rows, "count": len(gold_rows)})
    write_json(sum_dir / "subset_repair.json", {"rows": repair_rows, "count": len(repair_rows)})
    write_json(sum_dir / "subset_do_not_train.json", {"rows": dnt_rows, "count": len(dnt_rows)})

    run_summary = {
        "schema": "compiled_pilot_run_summary_v1",
        "disclaimer": (
            "Starter pilot scaffold — not clinical-grade. "
            "Heuristic judge is non-LLM; LLM judge is a prototype single-call analytic pass."
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
            "judge_mode": judge_mode,
            "judge_model": judge_model,
            "no_llm_cache": no_llm_cache,
            "ontology_override": ontology is not None,
            "ontology_version": ontology_resolved.version if ontology_resolved else None,
        },
        "profile_resolution": {
            "bootstrap_used": profile_resolution.bootstrap_used,
            "iterations_run": profile_resolution.iterations_run,
            "resolved_task_profile_id": profile_resolution.profile.task_profile_id,
            "parent_profile_id": profile_resolution.profile.parent_profile_id,
            "diagnostics": profile_resolution.diagnostics,
        },
        "slices": {k: [e.example_id for e in v] for k, v in slices.items()},
        "stats": stats.to_dict(),
        "review_queue_count": len(review_queue),
    }
    write_json(sum_dir / "run_summary.json", run_summary)

    write_json(
        sum_dir / "review_queue.json",
        {
            "schema": "compiled_review_queue_v1",
            "disclaimer": "Starter follow-up list — LLM uncertainty, low confidence, or judge disagreement.",
            "entries": review_queue,
            "count": len(review_queue),
        },
    )

    if write_csv and csv_rows:
        csv_path = sum_dir / "candidate_evaluations.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    return run_dir


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Starter compiled-rubric pilot runner (scaffold; not full discovery).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset JSON with rows[] (default: data/sample_100_aci_400_agbonnet.json)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run directory name under artifacts/compiled_runs/")
    parser.add_argument("--out-root", type=Path, default=None, help="Override artifacts root (default: <repo>/artifacts/compiled_runs)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Max examples to load (0 = all)")
    parser.add_argument("--source-filter", type=str, default=None, help="Substring filter on row source field")
    parser.add_argument("--design", type=int, default=100, help="Design slice size (starter 100/500 workflow)")
    parser.add_argument("--validation", type=int, default=100, help="Validation slice size")
    parser.add_argument("--pilot", type=int, default=300, help="Pilot slice size")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small fast run: limit=8, design=1, validation=1, pilot=6",
    )
    parser.add_argument("--csv", action="store_true", help="Write summaries/candidate_evaluations.csv")
    parser.add_argument(
        "--judge-mode",
        choices=("heuristic", "llm", "both"),
        default="heuristic",
        help="heuristic: non-LLM only (default). llm: LLM analytic judge. both: run both and compare.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override LLM judge model as provider:model (default: RUBRIC_GEN_COMPILED_JUDGE_MODEL or discover_default_judge_model).",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Disable Jsonl cache for LLM judge responses (under run_dir/cache/).",
    )
    parser.add_argument(
        "--task-profile",
        type=str,
        default=None,
        help="Override or pin the task profile for this run; use `auto` to force bootstrap from the dataset.",
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

    design_n, validation_n, pilot_n = args.design, args.validation, args.pilot
    start, limit = args.start, args.limit
    if args.smoke:
        limit = 8 if limit == 0 else limit
        design_n, validation_n, pilot_n = 1, 1, 6

    run_name = args.run_name or f"pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    out = run_pilot(
        dataset_path=dataset,
        run_name=run_name,
        start=start,
        limit=limit,
        design_n=design_n,
        validation_n=validation_n,
        pilot_n=pilot_n,
        source_filter=args.source_filter,
        out_root=args.out_root,
        write_csv=args.csv,
        judge_mode=args.judge_mode,
        judge_model=args.judge_model,
        no_llm_cache=args.no_llm_cache,
        task_profile=args.task_profile,
        bootstrap_iterations=max(1, args.bootstrap_iterations),
    )
    print(f"Wrote compiled pilot artifacts to: {out}")


if __name__ == "__main__":
    main()
