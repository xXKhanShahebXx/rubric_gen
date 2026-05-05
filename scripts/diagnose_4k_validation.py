"""Read-only diagnostics on the 4k pair-preference validation run.

Surfaces failure clusters and patterns that explain the post-tie-fix
67.23% pair-preference accuracy on rrd_whitened_uniform:

  * Per-source-family accuracy.
  * Per-Opus-verdict bucket accuracy (clean A / clean B / TIE-forced-A / parse-err).
  * Score-tie row analysis: how are the 498 ties distributed across
    source families, Opus buckets, and rubric-bank-size buckets?
  * Per-row rubric satisfaction statistics: distribution of
    (satisfied_a, satisfied_b) pairs. (0,0) rows guarantee a tie.
  * Retrieval health: zero-seed clustering, seed acceptance/rejection
    distribution.
  * 30 sample wrong predictions for qualitative inspection.

Reads only the existing artifacts under
``artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k/``
and the source label file ``data/medical_gpt5_b_regen_4k_opus_judged.jsonl``
(for Opus-verdict stratification). No LLM calls.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


DEFAULT_RUN = Path("artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k")
DEFAULT_RICH = Path("data/medical_gpt5_b_regen_4k_opus_judged.jsonl")
METHODS = ("rrd_uniform", "rrd_whitened_uniform")


def _quantiles(values: List[float], qs: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)) -> Dict[str, float]:
    if not values:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    s = sorted(values)
    n = len(s)
    out = {}
    for q in qs:
        if n == 1:
            out[f"p{int(q*100)}"] = s[0]
            continue
        idx = q * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            out[f"p{int(q*100)}"] = s[lo]
        else:
            frac = idx - lo
            out[f"p{int(q*100)}"] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def _hist(label: str, values: List[float], unit: str = "") -> None:
    if not values:
        print(f"  {label}: (empty)")
        return
    qs = _quantiles(values)
    mean = statistics.mean(values)
    print(
        f"  {label}: n={len(values):>5d}  mean={mean:6.2f}{unit}  "
        f"p10={qs['p10']:.1f}  p25={qs['p25']:.1f}  p50={qs['p50']:.1f}  "
        f"p75={qs['p75']:.1f}  p90={qs['p90']:.1f}  p99={qs['p99']:.1f}  "
        f"min={min(values):.1f}  max={max(values):.1f}"
    )


def _percent(n: int, d: int) -> str:
    return f"{(n / d * 100) if d else 0.0:5.1f}%"


def _source_id(example_id: str) -> str:
    if "__" in example_id:
        return example_id.split("__", 1)[1]
    return example_id


def _pair_anchor_scores(method_obj: Dict[str, Any], ex: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """Return (score_a, score_b, cand_id_a, cand_id_b)."""
    candidates = ex.get("candidates", []) or []
    pa = next((c for c in candidates if c.get("source_label") == "pair_response_a"), None)
    pb = next((c for c in candidates if c.get("source_label") == "pair_response_b"), None)
    if not pa or not pb:
        return None, None, None, None
    by_id = {r.get("candidate_id"): r for r in (method_obj.get("ranking") or [])}
    a_row = by_id.get(pa["candidate_id"])
    b_row = by_id.get(pb["candidate_id"])
    a_score = a_row.get("score") if a_row else None
    b_score = b_row.get("score") if b_row else None
    return (
        float(a_score) if a_score is not None else None,
        float(b_score) if b_score is not None else None,
        pa["candidate_id"], pb["candidate_id"],
    )


def _per_anchor_satisfied(method_obj: Dict[str, Any], cand_a: str, cand_b: str) -> Tuple[int, int, int]:
    """How many rubrics fired on cand_a and on cand_b? Plus total rubrics evaluated."""
    sat_a = 0
    sat_b = 0
    rubrics = set()
    for ev in method_obj.get("evaluations") or []:
        cid = ev.get("candidate_id")
        rid = ev.get("rubric_id")
        if rid:
            rubrics.add(rid)
        if not bool(ev.get("satisfied")):
            continue
        if cid == cand_a:
            sat_a += 1
        elif cid == cand_b:
            sat_b += 1
    return sat_a, sat_b, len(rubrics)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--rich", type=Path, default=DEFAULT_RICH)
    ap.add_argument("--method", type=str, default="rrd_whitened_uniform")
    ap.add_argument("--samples", type=int, default=30)
    args = ap.parse_args()

    files = sorted((args.run_dir / "examples").glob("*.json"))
    print(f"Run dir         : {args.run_dir}")
    print(f"Method          : {args.method}")
    print(f"Examples found  : {len(files)}")
    if not files:
        return 1

    # Load Opus verdicts (id -> A/B/TIE/empty)
    opus = {}
    for line in args.rich.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        opus[r["id"]] = (r.get("judge_verdict") or "").strip().upper()
    print(f"Opus verdicts   : {len(opus)} loaded from {args.rich.name}")

    # Tallies
    by_source = collections.defaultdict(lambda: collections.Counter())
    by_opus = collections.defaultdict(lambda: collections.Counter())
    by_bank_size_bucket = collections.defaultdict(lambda: collections.Counter())
    score_tie_rows = []  # (source_id, source_family, gold, opus_verdict, bank_size, sat_a, sat_b)
    pred_a_gold_b_rows = []
    pred_b_gold_a_rows = []
    bank_sizes = []
    seed_input_counts = []
    seed_accepted_counts = []
    seed_rejected_counts = []
    final_rubric_counts = []
    sat_a_dist = []
    sat_b_dist = []
    sat_a_minus_b = []
    rubric_eval_total_per_row = []
    pair_anchor_origin_kinds = collections.Counter()
    none_fired_rows = 0
    all_fired_rows = 0
    by_source_family_count = collections.Counter()
    zero_seed_by_source = collections.Counter()
    seed_count_by_source = collections.defaultdict(list)

    rand = random.Random(2026)
    sample_wrong = []  # collected first then sampled

    for path in files:
        ex = json.loads(path.read_text(encoding="utf-8"))
        example = ex.get("example", {})
        ex_id = example.get("example_id", "")
        sid = _source_id(ex_id)
        gold = (example.get("pair_correct_label") or "").strip().lower()
        if gold not in {"a", "b"}:
            continue
        family = example.get("source") or example.get("task_profile_id") or "(none)"
        opus_v = opus.get(sid, "")
        opus_bucket = opus_v if opus_v in {"A", "B", "TIE"} else "ERR"
        method_obj = ex.get("methods", {}).get(args.method, {}) or {}
        rubrics = method_obj.get("rubrics") or []
        bank_size = len(rubrics)
        artifact = method_obj.get("artifact", {}) or {}

        bank_sizes.append(bank_size)
        # bucket by bank size: small (<=4), medium (5-7), large (8+)
        size_bucket = "small (<=4)" if bank_size <= 4 else ("medium (5-7)" if bank_size <= 7 else "large (>=8)")

        seed_in = int(artifact.get("seed_rubric_input_count") or 0)
        seed_acc = int(artifact.get("seed_rubric_accepted_count") or 0)
        seed_rej = int(artifact.get("seed_rubric_rejected_count") or 0)
        final_rc = int(artifact.get("final_rubric_count") or 0)
        seed_input_counts.append(seed_in)
        seed_accepted_counts.append(seed_acc)
        seed_rejected_counts.append(seed_rej)
        final_rubric_counts.append(final_rc)
        seed_count_by_source[family].append(seed_in)
        if seed_in == 0:
            zero_seed_by_source[family] += 1
        by_source_family_count[family] += 1

        candidates = ex.get("candidates") or []
        pa = next((c for c in candidates if c.get("source_label") == "pair_response_a"), None)
        pb = next((c for c in candidates if c.get("source_label") == "pair_response_b"), None)
        if pa:
            pair_anchor_origin_kinds[pa.get("origin_kind", "")] += 1

        a_score, b_score, cand_a, cand_b = _pair_anchor_scores(method_obj, ex)
        if a_score is None or b_score is None or not cand_a or not cand_b:
            continue

        sat_a, sat_b, rubric_count = _per_anchor_satisfied(method_obj, cand_a, cand_b)
        sat_a_dist.append(sat_a)
        sat_b_dist.append(sat_b)
        sat_a_minus_b.append(sat_a - sat_b)
        rubric_eval_total_per_row.append(rubric_count)
        if sat_a == 0 and sat_b == 0:
            none_fired_rows += 1
        if rubric_count > 0 and sat_a == rubric_count and sat_b == rubric_count:
            all_fired_rows += 1

        # Predict (score-based, mirrors post-tie-fix _pair_preference_outcome)
        if a_score > b_score:
            pred = "a"
        elif b_score > a_score:
            pred = "b"
        else:
            pred = "tie"

        # Tally
        bucket_key = (gold, pred)
        by_source[family]["n"] += 1
        by_opus[opus_bucket]["n"] += 1
        by_bank_size_bucket[size_bucket]["n"] += 1
        if pred == gold:
            by_source[family]["correct"] += 1
            by_opus[opus_bucket]["correct"] += 1
            by_bank_size_bucket[size_bucket]["correct"] += 1
        elif pred == "tie":
            by_source[family]["tie"] += 1
            by_opus[opus_bucket]["tie"] += 1
            by_bank_size_bucket[size_bucket]["tie"] += 1
            score_tie_rows.append({
                "sid": sid, "family": family, "gold": gold, "opus": opus_v,
                "bank_size": bank_size, "sat_a": sat_a, "sat_b": sat_b,
            })
        else:
            by_source[family]["wrong"] += 1
            by_opus[opus_bucket]["wrong"] += 1
            by_bank_size_bucket[size_bucket]["wrong"] += 1
            if pred == "a" and gold == "b":
                pred_a_gold_b_rows.append({"sid": sid, "family": family, "opus": opus_v, "a_score": a_score, "b_score": b_score, "sat_a": sat_a, "sat_b": sat_b})
            else:
                pred_b_gold_a_rows.append({"sid": sid, "family": family, "opus": opus_v, "a_score": a_score, "b_score": b_score, "sat_a": sat_a, "sat_b": sat_b})

        # Collect for the wrong-pred sample
        if pred != gold:
            sample_wrong.append({
                "sid": sid, "family": family, "gold": gold, "pred": pred,
                "opus": opus_v, "a_score": a_score, "b_score": b_score,
                "sat_a": sat_a, "sat_b": sat_b, "bank_size": bank_size,
                "pa_text_head": (pa.get("text", "") or "")[:120] if pa else "",
                "pb_text_head": (pb.get("text", "") or "")[:120] if pb else "",
            })

    # Print results
    print()
    print("=" * 110)
    print(f"PER-SOURCE-FAMILY ACCURACY ({args.method}, score-based, ties counted as wrong)")
    print("=" * 110)
    print(f"{'family':<40} {'n':>6} {'correct':>8} {'wrong':>6} {'tie':>5} {'acc':>8} {'tie%':>7}")
    print("-" * 92)
    rows_sorted = sorted(by_source.items(), key=lambda x: -x[1]["n"])
    for family, c in rows_sorted:
        n = c["n"]; ok = c["correct"]; w = c["wrong"]; t = c["tie"]
        acc = ok / n * 100 if n else 0.0
        tie_pct = t / n * 100 if n else 0.0
        print(f"{family:<40} {n:>6} {ok:>8} {w:>6} {t:>5} {acc:>7.2f}% {tie_pct:>6.1f}%")

    print()
    print("=" * 110)
    print("PER-OPUS-VERDICT BUCKET ACCURACY")
    print("=" * 110)
    print(f"{'opus':<6} {'n':>6} {'correct':>8} {'wrong':>6} {'tie':>5} {'acc':>8} {'tie%':>7}   notes")
    for bucket in ("A", "B", "TIE", "ERR"):
        c = by_opus.get(bucket, {})
        n = c.get("n", 0); ok = c.get("correct", 0); w = c.get("wrong", 0); t = c.get("tie", 0)
        if not n: continue
        acc = ok / n * 100; tie_pct = t / n * 100
        notes = {
            "A": "Opus picked A (gold=A)",
            "B": "Opus picked B (gold=B)",
            "TIE": "Opus tie -> tie-policy=a forced gold=A",
            "ERR": "Opus parse-err -> default gold=A",
        }[bucket]
        print(f"{bucket:<6} {n:>6} {ok:>8} {w:>6} {t:>5} {acc:>7.2f}% {tie_pct:>6.1f}%   {notes}")

    print()
    print("=" * 110)
    print("PER-BANK-SIZE-BUCKET ACCURACY")
    print("=" * 110)
    print(f"{'bank_size':<14} {'n':>6} {'correct':>8} {'wrong':>6} {'tie':>5} {'acc':>8} {'tie%':>7}")
    for bucket in ("small (<=4)", "medium (5-7)", "large (>=8)"):
        c = by_bank_size_bucket.get(bucket, {})
        n = c.get("n", 0); ok = c.get("correct", 0); w = c.get("wrong", 0); t = c.get("tie", 0)
        if not n: continue
        acc = ok / n * 100; tie_pct = t / n * 100
        print(f"{bucket:<14} {n:>6} {ok:>8} {w:>6} {t:>5} {acc:>7.2f}% {tie_pct:>6.1f}%")

    print()
    print("=" * 110)
    print("PER-ROW RUBRIC SATISFACTION (DRIVERS OF SCORE-TIES)")
    print("=" * 110)
    _hist("bank size                ", bank_sizes)
    _hist("rubrics evaluated /row   ", rubric_eval_total_per_row)
    _hist("satisfied on pair_a /row ", sat_a_dist)
    _hist("satisfied on pair_b /row ", sat_b_dist)
    _hist("sat_a - sat_b (signed)   ", sat_a_minus_b)
    print(f"  (0,0) rows -- no rubric fired on either anchor : {none_fired_rows}/{len(sat_a_dist)} = {_percent(none_fired_rows, len(sat_a_dist))}")
    print(f"  (k,k) rows -- all rubrics fired on both anchors: {all_fired_rows}/{len(sat_a_dist)} = {_percent(all_fired_rows, len(sat_a_dist))}")

    print()
    print("=" * 110)
    print("SCORE-TIE BREAKDOWN")
    print("=" * 110)
    print(f"  total score-tie rows: {len(score_tie_rows)}")
    if score_tie_rows:
        tie_by_opus = collections.Counter(r["opus"] for r in score_tie_rows)
        tie_by_family = collections.Counter(r["family"] for r in score_tie_rows)
        tie_by_gold = collections.Counter(r["gold"] for r in score_tie_rows)
        tie_by_bank = collections.Counter(r["bank_size"] for r in score_tie_rows)
        sat_eq_zero = sum(1 for r in score_tie_rows if r["sat_a"] == 0 and r["sat_b"] == 0)
        sat_eq_full = sum(1 for r in score_tie_rows if r["sat_a"] == r["bank_size"] == r["sat_b"])
        sat_eq_mid = len(score_tie_rows) - sat_eq_zero - sat_eq_full
        print(f"  by gold     : {dict(tie_by_gold)}")
        print(f"  by Opus     : {dict(tie_by_opus)}")
        print(f"  by bank_size: {dict(sorted(tie_by_bank.items()))}")
        print(f"  ties with sat_a==0 and sat_b==0           : {sat_eq_zero}/{len(score_tie_rows)} = {_percent(sat_eq_zero, len(score_tie_rows))}  (no rubric fired)")
        print(f"  ties with sat_a==sat_b==bank_size (all-fire): {sat_eq_full}/{len(score_tie_rows)}")
        print(f"  ties in middle (some rubrics fired equally): {sat_eq_mid}/{len(score_tie_rows)}")
        print(f"  by source family (top 10):")
        for fam, n in tie_by_family.most_common(10):
            total_in_fam = by_source_family_count.get(fam, 0)
            print(f"    {fam:<40s} {n:>5} / {total_in_fam:>5} = {_percent(n, total_in_fam)}")

    print()
    print("=" * 110)
    print("RETRIEVAL HEALTH")
    print("=" * 110)
    rows_with_seeds = sum(1 for x in seed_input_counts if x > 0)
    rows_zero_seeds = sum(1 for x in seed_input_counts if x == 0)
    print(f"  rows w/ at least 1 seed input : {rows_with_seeds}/{len(seed_input_counts)} = {_percent(rows_with_seeds, len(seed_input_counts))}")
    print(f"  rows w/ zero seed input       : {rows_zero_seeds}/{len(seed_input_counts)} = {_percent(rows_zero_seeds, len(seed_input_counts))}")
    _hist("seeds input /row    ", [float(x) for x in seed_input_counts])
    _hist("seeds accepted /row ", [float(x) for x in seed_accepted_counts])
    _hist("seeds rejected /row ", [float(x) for x in seed_rejected_counts])
    print(f"  zero-seed rows by source family (top 10):")
    for fam, n in zero_seed_by_source.most_common(10):
        total = by_source_family_count.get(fam, 0)
        print(f"    {fam:<40s} {n:>5} / {total:>5} = {_percent(n, total)}")

    print()
    print("=" * 110)
    print(f"SAMPLE WRONG PREDICTIONS (random {args.samples} of {len(sample_wrong)})")
    print("=" * 110)
    sample = rand.sample(sample_wrong, min(args.samples, len(sample_wrong)))
    for s in sample[:args.samples]:
        is_tie = s["pred"] == "tie"
        marker = "TIE" if is_tie else f"pred={s['pred']}, gold={s['gold']}"
        print(f"  [{marker:<14}] family={s['family']:<35} opus={s['opus']:<3} bank={s['bank_size']} sat=({s['sat_a']},{s['sat_b']}) score=({s['a_score']:.3f},{s['b_score']:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
