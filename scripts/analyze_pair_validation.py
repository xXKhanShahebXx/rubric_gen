"""Per-class breakdown of pair-preference accuracy for a validation run.

Reads every ``examples/*.json`` under a run directory, finds the rubric
pipeline's preferred candidate (highest score for the chosen method),
compares to the ground-truth ``pair_correct_label``, and reports:

  * Headline pair-preference accuracy (matches summary.md)
  * Always-A and Always-B baseline accuracies
  * Per-class accuracy (when the gold is A vs when the gold is B)
  * Confusion matrix
  * Per-method comparison (rrd_uniform vs rrd_whitened_uniform)

This is the JudgeBench-style audit: a high overall number isn't worth
much if it's just tracking the label prior.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


METHODS_TO_REPORT = ("rrd_uniform", "rrd_whitened_uniform")
SCORE_KEY_PRIORITY = ("whitened_uniform_score", "uniform_score", "score")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to artifacts/.../runs/<run-name>/",
    )
    return parser


def _ranking_for(method_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the candidate ranking list for a method block."""
    ranking = method_obj.get("ranking")
    if isinstance(ranking, list):
        return [r for r in ranking if isinstance(r, dict)]
    return []


def _pair_anchor_scores(
    method_obj: Dict[str, Any], example_artifact: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float]]:
    """Return (score_a, score_b) for the two pair-anchor candidates.

    Matches the post-tie-fix ``_pair_preference_outcome`` resolution: look
    up the pair_a / pair_b candidate_ids in the candidates list, then pull
    their raw ``score`` (NOT ``rank``) from the method's ``ranking`` block.
    """
    candidates = example_artifact.get("candidates", []) or []
    pair_a = next(
        (c for c in candidates if c.get("source_label") == "pair_response_a"), None
    )
    pair_b = next(
        (c for c in candidates if c.get("source_label") == "pair_response_b"), None
    )
    if not pair_a or not pair_b:
        return None, None
    by_id = {r.get("candidate_id"): r for r in _ranking_for(method_obj)}
    a_row = by_id.get(pair_a["candidate_id"])
    b_row = by_id.get(pair_b["candidate_id"])
    a_score = a_row.get("score") if a_row else None
    b_score = b_row.get("score") if b_row else None
    return (
        float(a_score) if a_score is not None else None,
        float(b_score) if b_score is not None else None,
    )


def _predicted_label(method_obj: Dict[str, Any], example_artifact: Dict[str, Any]) -> str:
    """Return 'a' / 'b' / 'tie' / '' based on raw score (higher = winner).

    Matches the post-tie-fix reporting._pair_preference_outcome semantic:
    score equality counts as a tie (= wrong in the official metric).
    """
    a, b = _pair_anchor_scores(method_obj, example_artifact)
    if a is None or b is None:
        return ""
    if a > b:
        return "a"
    if b > a:
        return "b"
    return "tie"


def main() -> int:
    args = _build_parser().parse_args()
    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    examples_dir = run_dir / "examples"
    files = sorted(examples_dir.glob("*.json"))
    print(f"Run dir       : {run_dir}")
    print(f"Examples found: {len(files)}")
    if not files:
        return 1

    gold_counts = collections.Counter()
    method_results: Dict[str, Dict[str, Any]] = {
        m: {
            "correct": 0,
            "wrong": 0,
            "tie": 0,
            "no_score": 0,
            # per-gold class
            "gold_a_correct": 0, "gold_a_wrong": 0, "gold_a_tie": 0, "gold_a_no_score": 0,
            "gold_b_correct": 0, "gold_b_wrong": 0, "gold_b_tie": 0, "gold_b_no_score": 0,
            # confusion matrix
            "pred_a_gold_a": 0, "pred_a_gold_b": 0,
            "pred_b_gold_a": 0, "pred_b_gold_b": 0,
        }
        for m in METHODS_TO_REPORT
    }

    seed_input_total = 0
    seed_accepted_total = 0
    seed_rejected_total = 0
    rows_with_no_seeds = 0
    rows_with_some_seeds = 0

    for path in files:
        ex = json.loads(path.read_text(encoding="utf-8"))
        example = ex.get("example", {})
        gold = (example.get("pair_correct_label") or "").lower()
        if gold not in {"a", "b"}:
            continue
        gold_counts[gold] += 1

        # Aggregate seed bookkeeping (rrd_uniform is canonical here).
        rrd = ex.get("methods", {}).get("rrd_uniform", {}) or {}
        artifact = rrd.get("artifact", {}) or {}
        sin = int(artifact.get("seed_rubric_input_count") or 0)
        sacc = int(artifact.get("seed_rubric_accepted_count") or 0)
        srej = int(artifact.get("seed_rubric_rejected_count") or 0)
        seed_input_total += sin
        seed_accepted_total += sacc
        seed_rejected_total += srej
        if sin > 0:
            rows_with_some_seeds += 1
        else:
            rows_with_no_seeds += 1

        for method in METHODS_TO_REPORT:
            method_obj = ex.get("methods", {}).get(method, {}) or {}
            pred = _predicted_label(method_obj, ex)
            r = method_results[method]
            if pred == "":
                r["no_score"] += 1
                r[f"gold_{gold}_no_score"] += 1
                continue
            if pred == "tie":
                r["tie"] += 1
                r[f"gold_{gold}_tie"] += 1
                continue
            # confusion
            r[f"pred_{pred}_gold_{gold}"] += 1
            if pred == gold:
                r["correct"] += 1
                r[f"gold_{gold}_correct"] += 1
            else:
                r["wrong"] += 1
                r[f"gold_{gold}_wrong"] += 1

    n = sum(gold_counts.values())
    a_n = gold_counts.get("a", 0)
    b_n = gold_counts.get("b", 0)
    print(f"Evaluable rows: {n}")
    print(f"  gold A: {a_n} ({a_n/n*100:.1f}%)")
    print(f"  gold B: {b_n} ({b_n/n*100:.1f}%)")
    print(f"Always-A baseline: {a_n}/{n} = {a_n/n*100:.2f}%")
    print(f"Always-B baseline: {b_n}/{n} = {b_n/n*100:.2f}%")

    print()
    print(f"Retrieval seed bookkeeping (across {len(files)} rows):")
    print(f"  rows w/ at least 1 seed input : {rows_with_some_seeds}")
    print(f"  rows w/ zero seed input       : {rows_with_no_seeds}")
    if rows_with_some_seeds:
        print(
            f"  avg seeds input per such row  : {seed_input_total/rows_with_some_seeds:.2f}"
        )
        print(
            f"  avg seeds accepted by RRD     : {seed_accepted_total/rows_with_some_seeds:.2f}"
        )
        print(
            f"  avg seeds rejected by RRD     : {seed_rejected_total/rows_with_some_seeds:.2f}"
        )

    for method in METHODS_TO_REPORT:
        r = method_results[method]
        evaluable = r["correct"] + r["wrong"] + r["tie"]
        decided = r["correct"] + r["wrong"]
        print()
        print(f"=== {method} ===")
        print(f"  decided (A or B picked): {decided} / {n}")
        print(f"  ties (a==b)            : {r['tie']}")
        print(f"  no_score (couldn't evaluate): {r['no_score']}")
        if decided:
            print(f"  pair-pref accuracy (over decided): {r['correct']}/{decided} = {r['correct']/decided*100:.2f}%")
        print(f"  pair-pref accuracy (over n)      : {r['correct']}/{n} = {r['correct']/n*100:.2f}%")
        if a_n:
            print(f"  recall on gold A : {r['gold_a_correct']}/{a_n} = {r['gold_a_correct']/a_n*100:.2f}%")
        if b_n:
            print(f"  recall on gold B : {r['gold_b_correct']}/{b_n} = {r['gold_b_correct']/b_n*100:.2f}%")
        print("  Confusion (pred \\ gold):")
        print(f"               gold A     gold B")
        print(f"    pred A : {r['pred_a_gold_a']:7d}    {r['pred_a_gold_b']:7d}")
        print(f"    pred B : {r['pred_b_gold_a']:7d}    {r['pred_b_gold_b']:7d}")
        # Lift over majority baseline
        majority = max(a_n, b_n) / n if n else 0
        acc = r["correct"] / n if n else 0
        print(f"  lift over always-{'B' if b_n>=a_n else 'A'} baseline: {(acc - majority)*100:+.2f} pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
