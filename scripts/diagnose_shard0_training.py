"""Read-only diagnostics on the shard 0 training run.

Characterises the rubric library that shard 0 produced -- the input to the
medical retrieval index used during 4k validation. Looks for:

  * Bank-size, depth, source-stage, length distributions.
  * Common templated phrases (over-templating).
  * RRD outcomes (rejected, superseded, termination_rejections).
  * Per-rubric coverage on the training candidate pool (low-info detection).
  * Presence of "thorough clinical format" axis vocabulary that Opus
    rewards (the gap identified in §3 cross-experiment observation).
  * Per-example satisfaction distribution: how often does each rubric fire
    on each candidate? Are most rubrics binary-satisfied near 50%, or
    skewed toward all-satisfied / none-satisfied (low-info)?

No LLM calls. All numbers come from the existing JSON artifacts under
``artifacts/medical_rl/runs/medical_v47_train_shard0/examples/``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


DEFAULT_RUN = Path("artifacts/medical_rl/runs/medical_v47_train_shard0")

# Vocabulary that signals the "thorough clinical format" axis we expect Opus
# to reward (per §3 cross-experiment observation: regen-side wins because
# the regen prompt produces more thorough/clinician-style answers).
THOROUGHNESS_TOKENS = (
    "completeness", "complete", "covers", "comprehensive", "thorough",
    "differential", "guidelines", "guideline", "contraindication",
    "format", "structure", "structured", "concise", "padding",
    "clinical reasoning", "evidence-based",
)


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
        f"  {label}: n={len(values):>4d}  mean={mean:6.2f}{unit}  "
        f"p10={qs['p10']:.1f}  p25={qs['p25']:.1f}  p50={qs['p50']:.1f}  "
        f"p75={qs['p75']:.1f}  p90={qs['p90']:.1f}  p99={qs['p99']:.1f}  "
        f"min={min(values):.1f}  max={max(values):.1f}"
    )


def _normalize_phrase(text: str, head_tokens: int = 4) -> str:
    """Lowercase + strip punctuation + take leading N tokens for clustering."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = cleaned.split()
    return " ".join(tokens[:head_tokens])


def _percent(n: int, d: int) -> str:
    return f"{(n / d * 100) if d else 0.0:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--top-phrases", type=int, default=20)
    ap.add_argument("--method", type=str, default="rrd_uniform",
                    help="Method block to inspect (default rrd_uniform; rrd_whitened_uniform shares rubrics)")
    args = ap.parse_args()

    files = sorted((args.run_dir / "examples").glob("*.json"))
    print(f"Run dir       : {args.run_dir}")
    print(f"Method block  : {args.method}")
    print(f"Examples found: {len(files)}")
    if not files:
        return 1

    # Per-example aggregates
    bank_sizes: List[float] = []
    final_counts: List[float] = []
    initial_counts: List[float] = []
    rejected_counts: List[float] = []
    superseded_counts: List[float] = []
    term_reject_counts: List[float] = []
    rubric_lengths: List[float] = []
    depths: List[float] = []
    coverage_counts: List[float] = []
    source_stage_counter = collections.Counter()
    rubric_id_seen = set()
    unique_rubric_text_counter = collections.Counter()
    head_phrase_counter = collections.Counter()
    thoroughness_hits = collections.Counter()
    thoroughness_rubrics_per_example: List[float] = []

    # Per-rubric / per-candidate satisfaction analysis
    rubric_satisfaction_rates: List[float] = []  # one entry per rubric: % of candidates it fires on
    candidate_satisfaction_counts: List[float] = []  # one entry per candidate: # rubrics it satisfies
    eval_total = 0
    eval_yes = 0
    rubrics_zero_fire = 0
    rubrics_all_fire = 0
    rubrics_total = 0

    # Rubric counts per candidate origin (pair_anchor wouldn't exist in training, but check)
    # Training uses single-response rows from medical_rl_prompts. Candidates are: anchors + generated.
    candidate_origin_counter = collections.Counter()
    decomposition_decisions = collections.Counter()
    children_per_decomposed = []  # only when decomposition was attempted

    for path in files:
        ex = json.loads(path.read_text(encoding="utf-8"))
        method_obj = ex.get("methods", {}).get(args.method, {}) or {}
        rubrics = method_obj.get("rubrics", []) or []
        artifact = method_obj.get("artifact", {}) or {}
        evals = method_obj.get("evaluations", []) or []
        candidates = ex.get("candidates", []) or []

        bank_sizes.append(len(rubrics))
        final_counts.append(int(artifact.get("final_rubric_count") or 0))
        initial_counts.append(int(artifact.get("initial_rubric_count") or 0))
        def _count(v: Any) -> int:
            if isinstance(v, list):
                return len(v)
            if isinstance(v, (int, float)):
                return int(v)
            return 0

        rejected_counts.append(_count(artifact.get("rejected")))
        superseded_counts.append(_count(artifact.get("superseded")))
        # ``termination_rejections`` in the artifact is the *configured budget*
        # (PipelineConfig.termination_rejections, written in
        # ``rubric_gen/rrd/engine.py:881``), not the actual rejection count.
        # Reading it as a count was a bug that made every example look like it
        # hit the rejection ceiling exactly.  Track the budget separately so the
        # constant-vs-count nature is visible, and use len(artifact["rejected"])
        # for the real per-example rejection load.
        budget_field = artifact.get("termination_rejections")
        if isinstance(budget_field, (int, float)):
            term_reject_counts.append(float(budget_field))

        for c in candidates:
            candidate_origin_counter[c.get("origin_kind") or "(none)"] += 1

        thoroughness_in_example = 0
        for r in rubrics:
            text = (r.get("text") or r.get("requirement") or "").strip()
            depth = r.get("depth")
            stage = r.get("source_stage") or ""
            cov = r.get("coverage_count")
            rid = r.get("rubric_id")
            if rid:
                rubric_id_seen.add(rid)
            if text:
                rubric_lengths.append(len(text))
                unique_rubric_text_counter[text] += 1
                head_phrase_counter[_normalize_phrase(text)] += 1
                lowered = text.lower()
                hit_token = ""
                for tok in THOROUGHNESS_TOKENS:
                    if tok in lowered:
                        hit_token = tok
                        thoroughness_hits[tok] += 1
                        break
                if hit_token:
                    thoroughness_in_example += 1
            if depth is not None:
                depths.append(int(depth))
            if stage:
                source_stage_counter[stage] += 1
            if isinstance(cov, (int, float)):
                coverage_counts.append(float(cov))
            meta = r.get("metadata") or {}
            decision = (meta.get("decomposition_decision") or {}) if isinstance(meta, dict) else {}
            if decision:
                decomposition_decisions[decision.get("reason") or "(unknown)"] += 1
                children = meta.get("children") or []
                if children:
                    children_per_decomposed.append(len(children))
        thoroughness_rubrics_per_example.append(thoroughness_in_example)

        # Build (rubric_id -> list of satisfied bools) and (candidate_id -> list)
        per_rubric_fires: Dict[str, List[bool]] = collections.defaultdict(list)
        per_candidate_fires: Dict[str, List[bool]] = collections.defaultdict(list)
        for ev in evals:
            rid = ev.get("rubric_id")
            cid = ev.get("candidate_id")
            sat = bool(ev.get("satisfied"))
            eval_total += 1
            if sat:
                eval_yes += 1
            if rid:
                per_rubric_fires[rid].append(sat)
            if cid:
                per_candidate_fires[cid].append(sat)
        # rubric satisfaction rate = % of candidates this rubric fires on
        for rid, fires in per_rubric_fires.items():
            rubrics_total += 1
            rate = sum(fires) / len(fires) if fires else 0.0
            rubric_satisfaction_rates.append(rate)
            if rate == 0.0:
                rubrics_zero_fire += 1
            elif rate == 1.0:
                rubrics_all_fire += 1
        for cid, fires in per_candidate_fires.items():
            candidate_satisfaction_counts.append(sum(fires))

    print()
    print("=" * 110)
    print("BANK-SIZE & RRD OUTCOMES (per-example)")
    print("=" * 110)
    _hist("rubrics in bank        ", bank_sizes)
    _hist("initial_rubric_count   ", initial_counts)
    _hist("final_rubric_count     ", final_counts)
    _hist("rejected (per example) ", rejected_counts)
    _hist("superseded (per ex)    ", superseded_counts)
    # Renamed from "term_rejections (ex)" because the underlying field is the
    # configured rejection budget, not the per-example rejection count.  See
    # the comment in the consume loop above.
    _hist("term_rejection_budget  ", term_reject_counts)

    print()
    print("=" * 110)
    print("RUBRIC PROPERTIES (across all rubrics)")
    print("=" * 110)
    print(f"  total rubrics across all examples : {sum(int(x) for x in bank_sizes)}")
    print(f"  unique rubric_ids                  : {len(rubric_id_seen)}")
    print(f"  unique rubric texts                : {len(unique_rubric_text_counter)}")
    if unique_rubric_text_counter:
        most_common_dupe = unique_rubric_text_counter.most_common(1)[0]
        print(f"  most-duplicated text count         : {most_common_dupe[1]}  -> {most_common_dupe[0][:120]!r}")
    _hist("rubric text length (chars)", rubric_lengths, unit=" ch")
    _hist("rubric depth              ", depths)
    _hist("rubric coverage_count     ", coverage_counts)

    print()
    print(f"  source_stage distribution: {dict(source_stage_counter)}")
    print(f"  decomposition decisions  : {dict(decomposition_decisions)}")
    if children_per_decomposed:
        _hist("children per decomposed   ", children_per_decomposed)

    print()
    print("=" * 110)
    print("CANDIDATE POOL (per training row)")
    print("=" * 110)
    print(f"  origin_kind distribution: {dict(candidate_origin_counter)}")

    print()
    print("=" * 110)
    print("RUBRIC-CANDIDATE COVERAGE (LOW-INFO DETECTION)")
    print("=" * 110)
    print(f"  total (rubric,candidate) eval rows : {eval_total:,}")
    print(f"  satisfied=YES rate                  : {_percent(eval_yes, eval_total)} ({eval_yes:,}/{eval_total:,})")
    if rubric_satisfaction_rates:
        _hist("per-rubric fire-rate (% candidates)", [r * 100 for r in rubric_satisfaction_rates], unit="%")
        print(f"  rubrics that fire on ZERO candidates: {rubrics_zero_fire}/{rubrics_total} = {_percent(rubrics_zero_fire, rubrics_total)}  (useless)")
        print(f"  rubrics that fire on ALL  candidates: {rubrics_all_fire}/{rubrics_total} = {_percent(rubrics_all_fire, rubrics_total)}  (no discrimination)")
    if candidate_satisfaction_counts:
        _hist("per-candidate satisfied rubric count", candidate_satisfaction_counts)

    print()
    print("=" * 110)
    print("COMMON PHRASES (top-{0}, head 4 tokens normalised)".format(args.top_phrases))
    print("=" * 110)
    for phrase, n in head_phrase_counter.most_common(args.top_phrases):
        print(f"  {n:5d}  {phrase}")

    print()
    print("=" * 110)
    print('THOROUGHNESS / CLINICAL-FORMAT AXIS')
    print("=" * 110)
    print(
        f"  rubrics mentioning at least one thoroughness/format token: "
        f"{sum(thoroughness_hits.values())}  "
        f"({_percent(sum(thoroughness_hits.values()), sum(int(x) for x in bank_sizes))})"
    )
    for tok, n in thoroughness_hits.most_common():
        print(f"    {n:5d}  '{tok}'")
    _hist("thoroughness rubrics per example   ", thoroughness_rubrics_per_example)
    examples_without_thoroughness = sum(1 for x in thoroughness_rubrics_per_example if x == 0)
    print(
        f"  examples with ZERO thoroughness/format rubrics: "
        f"{examples_without_thoroughness}/{len(thoroughness_rubrics_per_example)} = "
        f"{_percent(examples_without_thoroughness, len(thoroughness_rubrics_per_example))}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
