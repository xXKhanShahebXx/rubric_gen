"""Package the per-example rubrics from a medical pair-validation run for RL training.

Walks every ``examples/*.json`` under a validation run directory (e.g.
``artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k/``)
and produces a clean, training-friendly bundle keyed by the same ``id`` used
in the slim pair JSONL (``data/medical_gpt5_b_regen_4k_rl.jsonl``).

Two output files are written by default:

  ``<out_dir>/<stem>_rubrics.jsonl``
        One row per sample (4000 rows).  Contains the prompt, both anchor
        answers, gold label, and the per-sample raw RRD rubric set plus
        the canonicalised production-bank items that match this sample.

  ``<out_dir>/<stem>_rubric_evaluations.jsonl``
        One row per (sample, rubric, candidate) tuple (~170k rows).  This
        is the per-rubric YES/NO satisfaction signal already produced by
        the GPT-4o judge during validation - useful as a reward-model
        training set or as ground-truth labels for a YES/NO classifier.

Both files preserve the row order of the source slim JSONL when the
``--align-with`` flag is supplied; otherwise rows are written in the
filesystem order of ``examples/``.

Run from the repo root:

    python scripts/package_medical_rubrics.py \
        --run-dir artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k \
        --align-with data/medical_gpt5_b_regen_4k_rl.jsonl \
        --out-dir data \
        --stem medical_gpt5_b_regen_4k

Add ``--no-evaluations`` to skip the evaluations file (recommended if you
only need rubrics-as-labels for downstream prompting).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


PRIMARY_METHOD = "rrd_whitened_uniform"
FALLBACK_METHOD = "rrd_uniform"
RUBRIC_FIELDS_KEEP = ("rubric_id", "text", "source_stage", "depth", "parent_id")
PRODUCTION_FIELDS_KEEP = (
    "production_rubric_id",
    "group_id",
    "label",
    "family",
    "canonical_text",
    "conditionality",
    "importance_tier",
    "action_taken",
    "source_member_count",
    "coverage_count",
    "discrimination_score",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Validation run directory containing examples/ subfolder.",
    )
    p.add_argument(
        "--align-with",
        type=Path,
        default=None,
        help=(
            "Optional path to the slim pair JSONL whose row order (and id "
            "field) we should preserve.  Rows in the slim file but missing "
            "from the run are skipped with a warning."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/).",
    )
    p.add_argument(
        "--stem",
        type=str,
        default="medical_gpt5_b_regen_4k",
        help="Output filename stem; suffixes _rubrics.jsonl and _rubric_evaluations.jsonl are appended.",
    )
    p.add_argument(
        "--method",
        type=str,
        default=PRIMARY_METHOD,
        choices=(PRIMARY_METHOD, FALLBACK_METHOD),
        help=(
            "Which method block to read rubrics/evaluations from.  Both methods "
            "share the same raw rubric set; whitened_uniform is the recommended "
            "default and matches the §4 audit numbers."
        ),
    )
    p.add_argument(
        "--no-evaluations",
        action="store_true",
        help="Skip the per-(sample, rubric, candidate) evaluations file.",
    )
    p.add_argument(
        "--reasoning-char-limit",
        type=int,
        default=600,
        help="Truncate per-evaluation 'reasoning' to this many characters (default: 600).",
    )
    return p


def _short_id(example_id: str) -> str:
    """Strip the source prefix to get the bare hash used in the slim JSONL.

    ``agentic_workflows__0006012-c2220aa60dd9`` -> ``0006012-c2220aa60dd9``
    """
    if "__" in example_id:
        return example_id.split("__", 1)[1]
    return example_id


def _read_align_ids(align_path: Path) -> List[str]:
    """Return the ordered list of `id` values from the slim pair JSONL."""
    ids: List[str] = []
    with align_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
    return ids


def _index_run(run_dir: Path) -> Dict[str, Path]:
    """Build {short_id: example_path} for every example file in the run."""
    index: Dict[str, Path] = {}
    for fp in sorted((run_dir / "examples").glob("*.json")):
        short = _short_id(fp.stem)
        index[short] = fp
    return index


def _slim_rubric(rub: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: rub.get(k) for k in RUBRIC_FIELDS_KEEP if k in rub}
    return out


def _slim_production_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: item.get(k) for k in PRODUCTION_FIELDS_KEEP if k in item}


def _index_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {c["candidate_id"]: c for c in candidates}


def _candidate_role(cand: Dict[str, Any]) -> str:
    """Convenience tag for downstream filtering."""
    if cand.get("origin_kind") == "pair_anchor":
        sl = cand.get("source_label") or ""
        if sl.endswith("_a"):
            return "pair_anchor_a"
        if sl.endswith("_b"):
            return "pair_anchor_b"
        return "pair_anchor"
    if cand.get("origin_kind") == "generated":
        return cand.get("source_label") or "generated"
    return cand.get("origin_kind") or "unknown"


def _build_rubric_row(
    short_id: str,
    example: Dict[str, Any],
    method_block: Dict[str, Any],
) -> Dict[str, Any]:
    rubrics = [_slim_rubric(r) for r in (method_block.get("rubrics") or [])]
    production_bank = [
        _slim_production_item(item) for item in (method_block.get("production_bank") or [])
    ]
    artifact = method_block.get("artifact") or {}
    ranking_raw = method_block.get("ranking") or []
    ranking = [
        {
            "candidate_id": r.get("candidate_id"),
            "rank": r.get("rank"),
            "score": r.get("score"),
        }
        for r in ranking_raw
    ]
    pair_label = (example.get("pair_correct_label") or "").lower()
    return {
        "id": short_id,
        "example_id": example.get("example_id"),
        "source": example.get("source"),
        "question": example.get("task_prompt") or example.get("conversation") or "",
        "reference_answer_a": example.get("pair_response_a"),
        "reference_answer_b": example.get("pair_response_b"),
        "gold_label": pair_label,
        "correct_answer": (
            "reference_answer_a" if pair_label == "a" else "reference_answer_b"
        ),
        "rubrics": rubrics,
        "rubric_count": len(rubrics),
        "production_bank": production_bank,
        "rrd_artifact": {
            "initial_rubric_count": artifact.get("initial_rubric_count"),
            "initial_seed_rubric_count": artifact.get("initial_seed_rubric_count"),
            "seed_rubric_input_count": artifact.get("seed_rubric_input_count"),
            "seed_rubric_accepted_count": artifact.get("seed_rubric_accepted_count"),
            "seed_rubric_rejected_count": artifact.get("seed_rubric_rejected_count"),
            "final_rubric_count": artifact.get("final_rubric_count"),
        },
        "ranking": ranking,
    }


def _build_evaluation_rows(
    short_id: str,
    example_id: str,
    method_block: Dict[str, Any],
    candidates_by_id: Dict[str, Dict[str, Any]],
    reasoning_char_limit: int,
) -> Iterable[Dict[str, Any]]:
    rubric_text_by_id = {
        r["rubric_id"]: r.get("text", "")
        for r in (method_block.get("rubrics") or [])
        if r.get("rubric_id")
    }
    for ev in method_block.get("evaluations") or []:
        cand_id = ev.get("candidate_id")
        cand = candidates_by_id.get(cand_id) or {}
        reasoning = ev.get("reasoning") or ""
        if reasoning and len(reasoning) > reasoning_char_limit:
            reasoning = reasoning[: reasoning_char_limit].rstrip() + "..."
        yield {
            "id": short_id,
            "example_id": example_id,
            "rubric_id": ev.get("rubric_id"),
            "rubric_text": rubric_text_by_id.get(ev.get("rubric_id"), ""),
            "candidate_id": cand_id,
            "candidate_role": _candidate_role(cand),
            "candidate_origin": cand.get("origin_kind"),
            "candidate_text": cand.get("text"),
            "satisfied": bool(ev.get("satisfied")),
            "reasoning": reasoning,
        }


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: --run-dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    examples_dir = run_dir / "examples"
    if not examples_dir.exists():
        print(f"ERROR: missing examples/ subfolder under {run_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rubrics_path = args.out_dir / f"{args.stem}_rubrics.jsonl"
    eval_path = args.out_dir / f"{args.stem}_rubric_evaluations.jsonl"

    print(f"indexing run dir: {run_dir}", flush=True)
    file_index = _index_run(run_dir)
    print(f"  found {len(file_index)} example files", flush=True)

    if args.align_with:
        ordered_ids = _read_align_ids(args.align_with)
        print(
            f"aligning to {args.align_with} -> {len(ordered_ids)} ids",
            flush=True,
        )
    else:
        ordered_ids = list(file_index.keys())

    rubrics_written = 0
    eval_rows_written = 0
    missing: List[str] = []
    fallback_used = 0

    rubrics_handle = rubrics_path.open("w", encoding="utf-8")
    eval_handle = (
        None if args.no_evaluations else eval_path.open("w", encoding="utf-8")
    )
    try:
        for i, short_id in enumerate(ordered_ids):
            fp = file_index.get(short_id)
            if fp is None:
                missing.append(short_id)
                continue
            with fp.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            example = doc.get("example") or {}
            methods = doc.get("methods") or {}
            method_block = methods.get(args.method)
            if method_block is None:
                method_block = methods.get(FALLBACK_METHOD)
                fallback_used += 1
            if method_block is None:
                missing.append(short_id)
                continue

            row = _build_rubric_row(short_id, example, method_block)
            rubrics_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rubrics_written += 1

            if eval_handle is not None:
                candidates_by_id = _index_candidates(doc.get("candidates") or [])
                for ev_row in _build_evaluation_rows(
                    short_id,
                    example.get("example_id") or short_id,
                    method_block,
                    candidates_by_id,
                    reasoning_char_limit=args.reasoning_char_limit,
                ):
                    eval_handle.write(json.dumps(ev_row, ensure_ascii=False) + "\n")
                    eval_rows_written += 1

            if (i + 1) % 500 == 0:
                print(
                    f"  processed {i + 1}/{len(ordered_ids)} "
                    f"(rubric_rows={rubrics_written}, eval_rows={eval_rows_written})",
                    flush=True,
                )
    finally:
        rubrics_handle.close()
        if eval_handle is not None:
            eval_handle.close()

    print()
    print(f"wrote {rubrics_path}  rows={rubrics_written}")
    if not args.no_evaluations:
        print(f"wrote {eval_path}  rows={eval_rows_written}")
    if missing:
        print(f"WARNING: {len(missing)} ids in --align-with had no example file")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    if fallback_used:
        print(f"NOTE: fell back to {FALLBACK_METHOD} for {fallback_used} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
