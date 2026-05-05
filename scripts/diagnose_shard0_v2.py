"""Phase 1 deep-forensics on the shard 0 training run.

Companion to ``scripts/diagnose_shard0_training.py`` (v1) -- the v1 report
covered surface stats but missed the *causal* angles that drive the
4k validation ceiling.  v2 adds eight new angles:

  A. Fix v1's ``term_rejections`` constant-vs-count bug (sanity check).
  B. Per-rejection-reason histogram (overlap / conflict / misaligned /
     insufficient_decomposition_gain / duplicate / empty).  Tells us
     which acceptance filter to loosen.
  C. Per-task_profile_id rubric vocabulary -- mean bank, top head-token
     phrases, thoroughness-token rate.  Tells us if the proposer
     collapses to "the note correctly identifies ..." for non-note tasks.
  D. Per-rubric Bernoulli p(1-p) discrimination -- the full distribution,
     not just the binary "fires on 0% / 100%" check.
  E. Per-example candidate-pool failure analysis -- why mean is 5.82 vs
     target 8 (anchor / generated / synthetic breakdown by family).
  F. Decomposition arithmetic -- reconcile 947 attempts vs 36 depth=1
     survivors by counting parents that successfully superseded.
  G. Head-token cluster matrix -- 8-token rubric prefix collisions to
     quantify over-templating.
  H. Anchor-vs-generated satisfaction asymmetry -- predictor of the
     4k FM2 "B-overshoot" without peeking at the 4k artifact.

All numbers come from the JSON files under
``artifacts/medical_rl/runs/medical_v47_train_shard0/examples/``.  No LLM
calls.  Outputs both stdout (human readable) and a Markdown report at
``artifacts/diagnostics/shard0_v2_report.md``.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


DEFAULT_RUN = Path("artifacts/medical_rl/runs/medical_v47_train_shard0")
DEFAULT_REPORT = Path("artifacts/diagnostics/shard0_v2_report.md")


THOROUGHNESS_TOKENS = (
    "completeness", "complete", "covers", "comprehensive", "thorough",
    "differential", "guidelines", "guideline", "contraindication",
    "format", "structure", "structured", "concise", "padding",
    "clinical reasoning", "evidence-based",
)


# Reusable helpers (kept intentionally inline / dependency-free).


def _quantiles(values: List[float], qs: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)) -> Dict[str, float]:
    if not values:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    s = sorted(values)
    n = len(s)
    out: Dict[str, float] = {}
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


def _hist_line(label: str, values: List[float], unit: str = "") -> str:
    if not values:
        return f"  {label}: (empty)"
    qs = _quantiles(values)
    mean = statistics.mean(values)
    return (
        f"  {label}: n={len(values):>4d}  mean={mean:6.2f}{unit}  "
        f"p10={qs['p10']:.1f}  p25={qs['p25']:.1f}  p50={qs['p50']:.1f}  "
        f"p75={qs['p75']:.1f}  p90={qs['p90']:.1f}  p99={qs['p99']:.1f}  "
        f"min={min(values):.1f}  max={max(values):.1f}"
    )


def _percent(n: int, d: int) -> str:
    return f"{(n / d * 100) if d else 0.0:5.1f}%"


def _normalize_phrase(text: str, head_tokens: int = 4) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = cleaned.split()
    return " ".join(tokens[:head_tokens])


# Per-example accumulator.


class ShardAccumulator:
    """Collects all v2 stats by streaming example.json files once."""

    def __init__(self, method: str) -> None:
        self.method = method

        # Angle A — sanity: real rejection count vs the v1 budget-constant bug.
        self.real_reject_counts: List[float] = []
        self.budget_field_values: List[float] = []  # the buggy "termination_rejections" field
        self.budget_unique: collections.Counter = collections.Counter()

        # Angle B — per-rejection-reason histogram.
        self.reject_reason_counter: collections.Counter = collections.Counter()
        # Per-example reason breakdown — also exposes whether one reason
        # dominates a few outlier examples vs being spread evenly.
        self.per_example_reason_counts: List[Dict[str, int]] = []

        # Angle C — per-family rubric vocab.
        # Map family -> list of {bank_size, thoroughness_count, head_phrases}
        self.family_examples: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {
                "n": 0,
                "bank_sizes": [],
                "thoroughness_counts": [],
                "head_phrases": collections.Counter(),
                "rubric_count": 0,
                "thoroughness_rubric_count": 0,
            }
        )

        # Angle D — per-rubric discrimination (Bernoulli p(1-p)).
        self.discrimination_scores: List[float] = []
        self.fire_rates: List[float] = []
        # Bucketise: <0.05 = useless, 0.05-0.15 = weak, 0.15-0.25 = strong (max=0.25 at p=0.5).
        self.discrim_useless = 0   # p(1-p) < 0.05  → fires on >95% or <5%
        self.discrim_weak = 0      # 0.05 ≤ p(1-p) < 0.15
        self.discrim_strong = 0    # p(1-p) ≥ 0.15  → fires on 20-80%
        self.discrim_total = 0

        # Angle E — candidate pool composition by family.
        self.family_candidate_pool: Dict[str, List[Dict[str, int]]] = collections.defaultdict(list)
        self.candidate_pool_sizes: List[int] = []

        # Angle F — decomposition arithmetic.
        self.depth1_rubrics_seen = 0
        self.depth1_rubrics_with_parent_id_in_bank = 0
        self.parents_of_surviving_children: set = set()
        self.parent_to_child_count: Dict[str, int] = collections.defaultdict(int)
        self.attempted_decompositions_in_bank = 0
        self.decomposition_decision_reasons: collections.Counter = collections.Counter()

        # Angle G — head-token cluster matrix (4-token AND 8-token).
        self.head4_counter: collections.Counter = collections.Counter()
        self.head8_counter: collections.Counter = collections.Counter()
        self.rubric_total = 0

        # Angle H — anchor-vs-generated satisfaction asymmetry.
        # Per-example: mean satisfied rate on anchor candidates vs generated.
        self.per_example_anchor_minus_generated: List[float] = []
        self.per_example_anchor_mean: List[float] = []
        self.per_example_generated_mean: List[float] = []
        self.per_example_synthetic_mean: List[float] = []

    # Streaming entry point.

    def consume(self, example_artifact: Dict[str, Any]) -> None:
        ex = example_artifact.get("example", {}) or {}
        family = ex.get("task_profile_id") or "(none)"
        candidates = example_artifact.get("candidates", []) or []
        method_obj = example_artifact.get("methods", {}).get(self.method, {}) or {}
        artifact = method_obj.get("artifact", {}) or {}
        rubrics = method_obj.get("rubrics", []) or []
        evaluations = method_obj.get("evaluations", []) or []

        # --- A: real rejection count vs the buggy field. ---
        rejected_list = artifact.get("rejected") or []
        self.real_reject_counts.append(float(len(rejected_list)))
        budget_field = artifact.get("termination_rejections")
        if isinstance(budget_field, (int, float)):
            self.budget_field_values.append(float(budget_field))
            self.budget_unique[int(budget_field)] += 1

        # --- B: rejection-reason histogram. ---
        per_ex_reasons: collections.Counter = collections.Counter()
        for entry in rejected_list:
            reason = (entry.get("reason") or "(unknown)").strip().lower()
            self.reject_reason_counter[reason] += 1
            per_ex_reasons[reason] += 1
        self.per_example_reason_counts.append(dict(per_ex_reasons))

        # --- C: per-family rubric vocab. ---
        slot = self.family_examples[family]
        slot["n"] += 1
        slot["bank_sizes"].append(len(rubrics))

        # --- D: per-rubric discrimination -- needs the eval matrix per bank. ---
        per_rubric_fires: Dict[str, List[bool]] = collections.defaultdict(list)
        for ev in evaluations:
            rid = ev.get("rubric_id")
            if rid:
                per_rubric_fires[rid].append(bool(ev.get("satisfied")))

        family_thoroughness = 0
        for rubric in rubrics:
            text = (rubric.get("text") or "").strip()
            if not text:
                continue
            self.rubric_total += 1
            lowered = text.lower()
            head4 = _normalize_phrase(text, 4)
            head8 = _normalize_phrase(text, 8)
            self.head4_counter[head4] += 1
            self.head8_counter[head8] += 1
            slot["head_phrases"][head4] += 1
            slot["rubric_count"] += 1

            for tok in THOROUGHNESS_TOKENS:
                if tok in lowered:
                    family_thoroughness += 1
                    slot["thoroughness_rubric_count"] += 1
                    break

            # --- D continued: discrimination on this rubric's eval row. ---
            fires = per_rubric_fires.get(rubric.get("rubric_id") or "", [])
            if fires:
                rate = sum(fires) / len(fires)
                self.fire_rates.append(rate)
                disc = rate * (1.0 - rate)
                self.discrimination_scores.append(disc)
                self.discrim_total += 1
                if disc < 0.05:
                    self.discrim_useless += 1
                elif disc < 0.15:
                    self.discrim_weak += 1
                else:
                    self.discrim_strong += 1

            # --- F: depth-1 / parent reconciliation. ---
            depth = rubric.get("depth")
            if depth == 1:
                self.depth1_rubrics_seen += 1
                pid = rubric.get("parent_id")
                if pid:
                    self.depth1_rubrics_with_parent_id_in_bank += 1
                    # Note: parent itself has been superseded so it isn't in
                    # the bank.  We just count surviving-child parent ids.
                    self.parents_of_surviving_children.add(pid)
                    self.parent_to_child_count[pid] += 1

            meta = rubric.get("metadata") or {}
            decision = (meta.get("decomposition_decision") or {}) if isinstance(meta, dict) else {}
            if decision:
                self.attempted_decompositions_in_bank += 1
                reason = (decision.get("reason") or "(unknown)").strip().lower()
                self.decomposition_decision_reasons[reason] += 1

        slot["thoroughness_counts"].append(family_thoroughness)

        # --- E: candidate pool composition by family. ---
        origin_breakdown = {"anchor": 0, "generated": 0, "synthetic": 0, "(other)": 0}
        for c in candidates:
            kind = c.get("origin_kind") or "(other)"
            origin_breakdown[kind if kind in origin_breakdown else "(other)"] += 1
        origin_breakdown["total"] = len(candidates)
        self.candidate_pool_sizes.append(len(candidates))
        self.family_candidate_pool[family].append(origin_breakdown)

        # --- H: per-example anchor vs generated satisfaction. ---
        # Build per-candidate satisfied-rubric counts, then split by origin.
        candidate_origin: Dict[str, str] = {
            c.get("candidate_id"): (c.get("origin_kind") or "(other)") for c in candidates
        }
        per_candidate_yes: Dict[str, int] = collections.defaultdict(int)
        per_candidate_total: Dict[str, int] = collections.defaultdict(int)
        for ev in evaluations:
            cid = ev.get("candidate_id")
            if cid is None:
                continue
            per_candidate_total[cid] += 1
            if ev.get("satisfied"):
                per_candidate_yes[cid] += 1

        anchor_rates: List[float] = []
        generated_rates: List[float] = []
        synthetic_rates: List[float] = []
        for cid, total in per_candidate_total.items():
            if total == 0:
                continue
            rate = per_candidate_yes.get(cid, 0) / total
            kind = candidate_origin.get(cid, "(other)")
            if kind == "anchor":
                anchor_rates.append(rate)
            elif kind == "generated":
                generated_rates.append(rate)
            elif kind == "synthetic":
                synthetic_rates.append(rate)

        if anchor_rates:
            self.per_example_anchor_mean.append(statistics.mean(anchor_rates))
        if generated_rates:
            self.per_example_generated_mean.append(statistics.mean(generated_rates))
        if synthetic_rates:
            self.per_example_synthetic_mean.append(statistics.mean(synthetic_rates))
        if anchor_rates and generated_rates:
            self.per_example_anchor_minus_generated.append(
                statistics.mean(anchor_rates) - statistics.mean(generated_rates)
            )

    # Reporters (return list of report lines for both stdout and markdown).

    def _section(self, title: str) -> List[str]:
        return ["", "=" * 110, title, "=" * 110]

    def angle_a_lines(self) -> List[str]:
        out = self._section("ANGLE A -- term_rejections sanity (v1 bug confirmation)")
        out.append(_hist_line("real rejection count (len(artifact.rejected))", self.real_reject_counts))
        out.append(_hist_line("v1 'termination_rejections' field (BUDGET, NOT COUNT)", self.budget_field_values))
        out.append(f"  unique values for the v1 field: {dict(self.budget_unique)}")
        all_same = len(self.budget_unique) == 1
        if all_same and self.real_reject_counts:
            real_mean = statistics.mean(self.real_reject_counts)
            out.append(
                f"  ==> v1 bug CONFIRMED: every example's 'termination_rejections' = "
                f"{next(iter(self.budget_unique))} (the configured ceiling), "
                f"but the actual rejection-count mean is {real_mean:.2f}."
            )
            out.append(
                "  ==> Fix: read len(artifact['rejected']) instead.  RRD is NOT hitting "
                "the rejection ceiling; the proposer's acceptance filters do most rejection."
            )
        return out

    def angle_b_lines(self) -> List[str]:
        out = self._section("ANGLE B -- per-rejection-reason histogram")
        total = sum(self.reject_reason_counter.values())
        out.append(f"  total rejections across all examples: {total}")
        for reason, n in self.reject_reason_counter.most_common():
            out.append(f"    {n:6d}  ({n/total*100:5.1f}%)  {reason}")
        out.append("")
        out.append("  per-example dominance: how often is each reason the modal rejection?")
        modal_counter: collections.Counter = collections.Counter()
        for per_ex in self.per_example_reason_counts:
            if not per_ex:
                continue
            modal = max(per_ex.items(), key=lambda kv: kv[1])[0]
            modal_counter[modal] += 1
        nz = sum(modal_counter.values())
        for reason, n in modal_counter.most_common():
            out.append(f"    {n:6d}  ({n/nz*100:5.1f}%)  modal={reason}")
        return out

    def angle_c_lines(self, top_k: int = 12) -> List[str]:
        out = self._section("ANGLE C -- per-task_profile_id rubric vocabulary")
        out.append(f"  {'family':<35s} {'n':>5s}  {'bank_mean':>9s}  {'thor%':>7s}  {'thor/ex':>8s}")
        for family, slot in sorted(self.family_examples.items(), key=lambda kv: -kv[1]["n"]):
            n = slot["n"]
            bank_mean = statistics.mean(slot["bank_sizes"]) if slot["bank_sizes"] else 0.0
            thor_pct = (
                slot["thoroughness_rubric_count"] / slot["rubric_count"] * 100
                if slot["rubric_count"]
                else 0.0
            )
            thor_per_ex = (
                statistics.mean(slot["thoroughness_counts"]) if slot["thoroughness_counts"] else 0.0
            )
            out.append(
                f"  {family:<35s} {n:>5d}  {bank_mean:>9.2f}  {thor_pct:>6.1f}%  {thor_per_ex:>8.2f}"
            )
        out.append("")
        out.append(f"  TOP-{top_k} HEAD PHRASES PER FAMILY:")
        for family, slot in sorted(self.family_examples.items(), key=lambda kv: -kv[1]["n"]):
            if slot["n"] < 5:
                continue
            out.append(f"    [{family}] (n={slot['n']}, rubrics={slot['rubric_count']})")
            for phrase, n in slot["head_phrases"].most_common(top_k):
                out.append(f"      {n:5d}  {phrase}")
        return out

    def angle_d_lines(self) -> List[str]:
        out = self._section("ANGLE D -- per-rubric Bernoulli discrimination (p * (1-p))")
        out.append(_hist_line("per-rubric fire rate (p)", [r * 100 for r in self.fire_rates], unit="%"))
        out.append(_hist_line("per-rubric discrimination p(1-p)", self.discrimination_scores))
        if self.discrim_total:
            out.append(
                f"  buckets: useless (p(1-p)<0.05)  : {self.discrim_useless}/{self.discrim_total} = "
                f"{_percent(self.discrim_useless, self.discrim_total)}  -- fires on >95% or <5% of candidates"
            )
            out.append(
                f"  buckets: weak    (0.05<=...<0.15): {self.discrim_weak}/{self.discrim_total} = "
                f"{_percent(self.discrim_weak, self.discrim_total)}  -- modest discrimination"
            )
            out.append(
                f"  buckets: strong  (>=0.15)         : {self.discrim_strong}/{self.discrim_total} = "
                f"{_percent(self.discrim_strong, self.discrim_total)}  -- fires on 20-80% of candidates"
            )
            out.append("")
            out.append(
                "  v1 only counted 'fires on 0% / 100%' (1.0% / 10.9% from the v1 report).  "
                "v2 shows the full distribution -- the 'useless' bucket (p(1-p)<0.05) is the truer "
                "low-info ceiling because a rubric firing on 87% of candidates is also nearly useless."
            )
        return out

    def angle_e_lines(self) -> List[str]:
        out = self._section("ANGLE E -- candidate-pool composition by family")
        out.append(_hist_line("candidate pool size per example (target=8)", [float(x) for x in self.candidate_pool_sizes]))
        out.append("")
        out.append(f"  {'family':<35s} {'n':>5s}  {'pool_mean':>9s}  {'anchor':>7s}  {'gen':>5s}  {'synth':>6s}  {'<8 pools':>9s}")
        for family, pools in sorted(self.family_candidate_pool.items(), key=lambda kv: -len(kv[1])):
            n = len(pools)
            pool_total_mean = statistics.mean(p["total"] for p in pools)
            anchor_mean = statistics.mean(p["anchor"] for p in pools)
            gen_mean = statistics.mean(p["generated"] for p in pools)
            synth_mean = statistics.mean(p["synthetic"] for p in pools)
            small_pools = sum(1 for p in pools if p["total"] < 8)
            out.append(
                f"  {family:<35s} {n:>5d}  {pool_total_mean:>9.2f}  "
                f"{anchor_mean:>7.2f}  {gen_mean:>5.2f}  {synth_mean:>6.2f}  "
                f"{small_pools:>4d} ({small_pools/n*100:>3.0f}%)"
            )
        return out

    def angle_f_lines(self) -> List[str]:
        out = self._section("ANGLE F -- decomposition arithmetic")
        out.append(f"  depth=1 rubrics in final banks (across all examples): {self.depth1_rubrics_seen}")
        out.append(f"  depth=1 rubrics whose parent_id is set:                 {self.depth1_rubrics_with_parent_id_in_bank}")
        out.append(f"  unique parents that produced surviving children:        {len(self.parents_of_surviving_children)}")
        out.append(f"  attempted decompositions visible in bank (parents not superseded):")
        out.append(f"      total: {self.attempted_decompositions_in_bank}")
        for reason, n in self.decomposition_decision_reasons.most_common():
            out.append(f"      {n:5d}  reason={reason}")
        out.append("")
        out.append(
            "  Reconciliation: v1 reported {reject_children: 264, needs_two_children: 683} = 947, "
            "but never counted parents that successfully decomposed (they get superseded and removed "
            "from the bank).  v2 recovers them via depth=1 children: roughly "
            f"{len(self.parents_of_surviving_children)} successful parents produced "
            f"{self.depth1_rubrics_seen} depth=1 children, "
            "for a true decomposition-success rate of "
            + (
                f"{len(self.parents_of_surviving_children)/(len(self.parents_of_surviving_children)+self.attempted_decompositions_in_bank)*100:.1f}%"
                if (len(self.parents_of_surviving_children) + self.attempted_decompositions_in_bank) > 0
                else "n/a"
            )
            + " (vs ~3.8% reported in v1's 36/947 -- v1 was wrong because it counted attempts in the wrong direction)."
        )
        return out

    def angle_g_lines(self, top_k: int = 20) -> List[str]:
        out = self._section("ANGLE G -- head-token clustering (over-templating ceiling)")
        out.append(f"  total rubrics: {self.rubric_total}")
        out.append(f"  unique 4-token prefixes: {len(self.head4_counter)} -- collision factor "
                   f"{(self.rubric_total / max(1, len(self.head4_counter))):.2f}x")
        out.append(f"  unique 8-token prefixes: {len(self.head8_counter)} -- collision factor "
                   f"{(self.rubric_total / max(1, len(self.head8_counter))):.2f}x")
        out.append("")
        out.append(f"  TOP-{top_k} 4-token prefixes:")
        for phrase, n in self.head4_counter.most_common(top_k):
            out.append(f"    {n:5d}  ({n/self.rubric_total*100:4.1f}%)  {phrase}")
        out.append("")
        out.append(f"  TOP-{top_k} 8-token prefixes:")
        for phrase, n in self.head8_counter.most_common(top_k):
            out.append(f"    {n:5d}  ({n/self.rubric_total*100:4.1f}%)  {phrase}")
        # Concentration metric: what share of rubrics live in the top-50 4-token prefixes?
        top50 = sum(n for _, n in self.head4_counter.most_common(50))
        out.append("")
        out.append(
            f"  CONCENTRATION: top-50 4-token prefixes cover {top50}/{self.rubric_total} = "
            f"{_percent(top50, self.rubric_total)} of all rubrics."
        )
        return out

    def angle_h_lines(self) -> List[str]:
        out = self._section("ANGLE H -- anchor-vs-generated satisfaction asymmetry")
        out.append(_hist_line("per-example anchor satisfaction rate", [x * 100 for x in self.per_example_anchor_mean], unit="%"))
        out.append(_hist_line("per-example generated satisfaction rate", [x * 100 for x in self.per_example_generated_mean], unit="%"))
        out.append(_hist_line("per-example synthetic satisfaction rate", [x * 100 for x in self.per_example_synthetic_mean], unit="%"))
        out.append(_hist_line("per-example (anchor - generated)", [x * 100 for x in self.per_example_anchor_minus_generated], unit="pp"))
        if self.per_example_anchor_minus_generated:
            mean_diff = statistics.mean(self.per_example_anchor_minus_generated) * 100
            sign = "ANCHOR-FAVOURED" if mean_diff > 0 else "GENERATED-FAVOURED"
            out.append("")
            out.append(
                f"  ==> Mean (anchor - generated) = {mean_diff:+.2f} pp  --> {sign} rubric library."
            )
            if mean_diff > 5.0:
                out.append(
                    "  ==> Anchor-favoured: training rubrics over-rewarded the existing answer.  "
                    "Predicts that on 4k validation, the rubrics will pick whichever side resembles "
                    "the training-time anchor distribution (gpt-4o terse) -- i.e. wrong direction "
                    "for the gpt-5 thoroughness-favouring labels."
                )
            elif mean_diff < -5.0:
                out.append(
                    "  ==> Generated-favoured: training rubrics over-rewarded LLM-generated candidates "
                    "(typically more thorough than anchors).  Predicts the 4k FM2 'B-overshoot' "
                    "(pipeline picks B/thorough even when Opus picked A)."
                )
            else:
                out.append(
                    "  ==> Roughly balanced: the rubric library is anchor-neutral on shard 0.  "
                    "The 4k FM2 'B-overshoot' must come from the regen-prompt asymmetry (which is "
                    "applied at scoring time, not training time)."
                )
        return out

    def all_lines(self, run_dir: Path, n_examples: int) -> List[str]:
        head = [
            "Run dir       : " + str(run_dir),
            f"Method block  : {self.method}",
            f"Examples seen : {n_examples}",
        ]
        return (
            head
            + self.angle_a_lines()
            + self.angle_b_lines()
            + self.angle_c_lines()
            + self.angle_d_lines()
            + self.angle_e_lines()
            + self.angle_f_lines()
            + self.angle_g_lines()
            + self.angle_h_lines()
        )

    def markdown_summary(self, run_dir: Path, n_examples: int) -> str:
        """Compact markdown distilling the headline finding for each angle."""
        lines: List[str] = []
        lines.append("# Shard 0 v2 Forensics -- Headline Findings")
        lines.append("")
        lines.append(f"- Run dir: `{run_dir}`")
        lines.append(f"- Method block: `{self.method}`")
        lines.append(f"- Examples seen: **{n_examples}**")
        lines.append("")

        # A
        real_mean = statistics.mean(self.real_reject_counts) if self.real_reject_counts else 0.0
        budget = next(iter(self.budget_unique), None)
        lines.append("## Angle A -- v1 `term_rejections` bug")
        lines.append("")
        if budget is not None and len(self.budget_unique) == 1:
            lines.append(
                f"**v1 bug confirmed.** Every example's `termination_rejections` is the constant "
                f"**{budget}** (the configured budget), but the actual rejection-count mean is "
                f"**{real_mean:.2f}** per example.  RRD is **not** hitting its rejection ceiling -- "
                f"the proposer's acceptance filters drive most rejection."
            )
        lines.append("")

        # B
        total_rej = sum(self.reject_reason_counter.values())
        lines.append("## Angle B -- Rejection reasons")
        lines.append("")
        lines.append(f"**{total_rej} total rejections across {n_examples} examples.**")
        lines.append("")
        lines.append("| Reason | N | % |")
        lines.append("|---|---:|---:|")
        for reason, n in self.reject_reason_counter.most_common():
            lines.append(f"| `{reason}` | {n} | {n/total_rej*100:.1f}% |")
        lines.append("")
        if self.reject_reason_counter:
            top_reason, top_n = self.reject_reason_counter.most_common(1)[0]
            lines.append(
                f"**Dominant reason:** `{top_reason}` ({top_n/total_rej*100:.1f}% of rejections). "
                "If overlap dominates, lower the Jaccard threshold from 0.7 in `engine.py:64-75`. "
                "If misalignment dominates, the strong/weak bucket detection on Q&A rows is broken."
            )
        lines.append("")

        # C
        lines.append("## Angle C -- Per-family rubric vocabulary")
        lines.append("")
        lines.append("| Family | N | bank_mean | thor_pct | thor/ex |")
        lines.append("|---|---:|---:|---:|---:|")
        for family, slot in sorted(self.family_examples.items(), key=lambda kv: -kv[1]["n"]):
            n = slot["n"]
            bank_mean = statistics.mean(slot["bank_sizes"]) if slot["bank_sizes"] else 0.0
            thor_pct = (
                slot["thoroughness_rubric_count"] / slot["rubric_count"] * 100
                if slot["rubric_count"]
                else 0.0
            )
            thor_per_ex = (
                statistics.mean(slot["thoroughness_counts"]) if slot["thoroughness_counts"] else 0.0
            )
            lines.append(
                f"| `{family}` | {n} | {bank_mean:.2f} | {thor_pct:.1f}% | {thor_per_ex:.2f} |"
            )
        lines.append("")
        # Highlight worst-discriminating family from the 4k report (documentation_variants)
        # without peeking at 4k (we don't, we just compare relative rates).
        smallest_bank_family: Optional[str] = None
        smallest_bank_size = math.inf
        for family, slot in self.family_examples.items():
            if slot["n"] >= 5 and slot["bank_sizes"]:
                bank_mean = statistics.mean(slot["bank_sizes"])
                if bank_mean < smallest_bank_size:
                    smallest_bank_size = bank_mean
                    smallest_bank_family = family
        if smallest_bank_family:
            lines.append(
                f"**Smallest bank family:** `{smallest_bank_family}` "
                f"(mean {smallest_bank_size:.2f} rubrics/example) -- this family's rubrics drive "
                "the small-bank score-tie regime (bank<=4 -> 31% tie rate, per 4k report)."
            )
        lines.append("")

        # D
        lines.append("## Angle D -- Per-rubric discrimination")
        lines.append("")
        if self.discrim_total:
            lines.append(
                f"**{self.discrim_useless}/{self.discrim_total} = "
                f"{self.discrim_useless/self.discrim_total*100:.1f}% rubrics are 'useless' "
                f"(p(1-p)<0.05).** v1 only flagged 10.9% (firing on ALL candidates); v2's broader "
                "definition catches the additional rubrics that fire on, e.g., 87% of candidates."
            )
            lines.append("")
            lines.append("| Bucket | N | % |")
            lines.append("|---|---:|---:|")
            lines.append(
                f"| useless (p(1-p)<0.05) | {self.discrim_useless} | {self.discrim_useless/self.discrim_total*100:.1f}% |"
            )
            lines.append(
                f"| weak (0.05-0.15) | {self.discrim_weak} | {self.discrim_weak/self.discrim_total*100:.1f}% |"
            )
            lines.append(
                f"| strong (>=0.15) | {self.discrim_strong} | {self.discrim_strong/self.discrim_total*100:.1f}% |"
            )
        lines.append("")

        # E
        lines.append("## Angle E -- Candidate pool composition")
        lines.append("")
        if self.candidate_pool_sizes:
            mean_pool = statistics.mean(self.candidate_pool_sizes)
            small_pools = sum(1 for x in self.candidate_pool_sizes if x < 8)
            lines.append(
                f"**Mean pool size: {mean_pool:.2f} vs target 8.** "
                f"{small_pools}/{len(self.candidate_pool_sizes)} examples ({small_pools/len(self.candidate_pool_sizes)*100:.1f}%) "
                "have under-target pools, starving the whitened-uniform covariance estimate."
            )
        lines.append("")

        # F
        lines.append("## Angle F -- Decomposition arithmetic")
        lines.append("")
        successful = len(self.parents_of_surviving_children)
        attempted = self.attempted_decompositions_in_bank + successful
        rate = successful / attempted * 100 if attempted else 0.0
        lines.append(
            f"**True decomposition success rate: {successful}/{attempted} = {rate:.1f}%** "
            "(v1 reported ~3.8% because it didn't count successful parents that got superseded)."
        )
        lines.append(
            f"  Surviving depth=1 children: {self.depth1_rubrics_seen}.  "
            f"Avg children per successful parent: "
            f"{(self.depth1_rubrics_seen / successful):.2f}." if successful else
            "  No successful decompositions detected."
        )
        lines.append("")

        # G
        lines.append("## Angle G -- Head-token clustering")
        lines.append("")
        if self.rubric_total:
            top_phrase, top_n = self.head4_counter.most_common(1)[0]
            lines.append(
                f"**Top 4-token prefix:** `\"{top_phrase}\"` covers {top_n}/{self.rubric_total} = "
                f"{top_n/self.rubric_total*100:.1f}% of all rubrics.  Collision factor "
                f"{(self.rubric_total / max(1, len(self.head4_counter))):.2f}x on 4-token prefix, "
                f"{(self.rubric_total / max(1, len(self.head8_counter))):.2f}x on 8-token prefix."
            )
            top50 = sum(n for _, n in self.head4_counter.most_common(50))
            lines.append("")
            lines.append(
                f"**Concentration:** top-50 4-token prefixes account for {top50/self.rubric_total*100:.1f}% "
                "of the entire rubric library -- this is the over-templating ceiling that depresses "
                "retrieval diversity."
            )
        lines.append("")

        # H
        lines.append("## Angle H -- Anchor vs generated asymmetry")
        lines.append("")
        if self.per_example_anchor_minus_generated:
            mean_diff = statistics.mean(self.per_example_anchor_minus_generated) * 100
            anchor_mean = statistics.mean(self.per_example_anchor_mean) * 100
            gen_mean = statistics.mean(self.per_example_generated_mean) * 100
            lines.append(
                f"**Mean satisfaction rate: anchor {anchor_mean:.1f}% vs generated {gen_mean:.1f}% "
                f"(diff {mean_diff:+.1f} pp).**"
            )
            if mean_diff > 5.0:
                lines.append(
                    "==> **Anchor-favoured** library.  At validation, rubrics will reward the "
                    "side that resembles training-time anchors (gpt-4o terse) -- wrong direction "
                    "for gpt-5-favouring 4k labels."
                )
            elif mean_diff < -5.0:
                lines.append(
                    "==> **Generated-favoured** library.  Predicts the 4k FM2 'B-overshoot': "
                    "pipeline picks the more thorough side even when Opus picked the terse side."
                )
            else:
                lines.append(
                    "==> Roughly balanced library.  Means the 4k FM2 B-overshoot must come from "
                    "the regen-prompt asymmetry (applied at scoring time, not training time)."
                )
        lines.append("")

        return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--method", type=str, default="rrd_uniform")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT,
                    help="Markdown summary destination.")
    ap.add_argument("--no-report", action="store_true",
                    help="Skip writing the markdown report file.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N example files (smoke testing).")
    args = ap.parse_args()

    files = sorted((args.run_dir / "examples").glob("*.json"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"No example files found under {args.run_dir / 'examples'}", file=sys.stderr)
        return 1

    acc = ShardAccumulator(method=args.method)
    for path in files:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  warn: failed to parse {path.name}: {exc}", file=sys.stderr)
            continue
        acc.consume(artifact)

    lines = acc.all_lines(args.run_dir, len(files))
    print("\n".join(lines))

    if not args.no_report:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(acc.markdown_summary(args.run_dir, len(files)), encoding="utf-8")
        print(f"\n[wrote markdown report -> {args.report_path}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
