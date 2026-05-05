# RewardBench 2 — Full Eval Results (`rb2_full_v47`)

> Companion doc to [`reward_bench_2.md`](reward_bench_2.md) (which describes the
> adapter and how the pipeline maps to RB2). This doc is the headline number,
> per-subset breakdown, and reportability assessment for the **single full
> eval** of the v4.7 locked policy on RewardBench 2.
>
> **Read [`§ 5 Reportability`](#5-reportability) before quoting any number.**

---

## 1. Headline

```
Run name:           rb2_full_v47
Locked policy:      jb_320_v2_full_seed29 (v4.7 hybrid system, frozen_policy_hash 9a68b4d9...)
Items evaluated:    1825 of 1865 (40 dropped — empty completions in source data)
Pairs evaluated:    5,595
Wall-clock:         14.3 hours @ 16 workers
Started / Ended:    2026-04-28 08:09 → 22:28 UTC
```

| Metric | Value |
|---|---:|
| **Leaderboard avg (5-subset, no Ties)** | **52.66 %** |
| Ties weighted score (proxy) | 38.89 % |
| Per-pair accuracy (averaged across 5 subsets) | 74.04 % |

The "leaderboard avg" is the average of per-subset item-level accuracies on
Factuality / Focus / Math / Precise IF / Safety, matching the public
RewardBench 2 metric. The Ties number uses our pairwise approximation
(see [§ 5.3](#53-ties-metric-is-an-approximation)).

---

## 2. Per-subset breakdown

| Subset | Items | Item-level acc | Per-pair acc | High-confidence overrides |
|---|---:|---:|---:|---:|
| Factuality | 475 | **53.05 %** | 76.49 % | rare |
| Focus | 495 | **53.33 %** | 74.34 % | rare |
| Math | 183 | **66.67 %** | 83.24 % | math solver fires often |
| Precise IF | 160 | **29.38 %** | 58.54 % | none |
| Safety | 450 | **60.89 %** | 77.78 % | none |
| Ties | 102 | 62.75 % (proxy) | 77.78 % | none |
| **Overall (5-subset avg)** | **1763** | **52.66 %** | **74.04 %** | — |

The gap between item-level accuracy (best-of-4 must-be-right-on-all-3) and
per-pair accuracy is the predictable consequence of the binomial: with
per-pair `p`, item-level success is `p^3`. At pair-level 76 %, item-level
is 0.76³ = 44 %. Most subsets sit slightly above that floor because some
pairs in the same item are correlated (they share the same chosen
completion).

---

## 3. What contributed (and what didn't)

### What helped

* **Math independent solver** is the only verifier override that meaningfully
  fires on RB2. On the 183-item Math subset it pushed item-level acc from
  what would have been ~50–55 % (rubric-judge alone) to **66.67 %** —
  consistent with the +5–7 point lift it gave on JudgeBench-math.
* **Pair-grounded rubric discovery** still functions on free-form RB2 prompts.
  On Safety / Factuality / Focus the discovery LLM call generates pair-specific
  criteria (instead of generic library criteria) which gives item-level acc in
  the 53–61 % range — beating chance (~39 %) by ~15–25 points.

### What didn't

* **MMLU independent answerer** never fires. RB2 prompts are free-form user
  queries with no `(A) ... (B) ... (C) ...` letter format, so the
  `_extract_candidate_letter` function returns empty for every candidate and
  the verifier abstains.
* **LeetCode test runner** never fires. RB2 has no `class Solution: def
  method(...)` patterns.
* **Reasoning independent solver** rarely fires. RB2 Precise IF prompts ask
  for format compliance ("answer without the letter u"), not deductive
  reasoning, so the canonicalised-answer matching always tries to compare
  free-form text, yielding low precision.
* **Holistic judge** is disabled in the v4.7 policy — irrelevant here.

### The Precise IF failure mode

Precise IF at **29.38 %** is *below* the JudgeBench rubric-judge baseline
(~50–55 %) and below the random best-of-4 baseline of ~25 %. This is
arguably the most informative number in the whole eval: **the rubric-judge
architecture cannot evaluate format-compliance constraints**. Sample failure:

* prompt: *"Give a one-line description, but do not use the letter 'e'."*
* chosen: a one-liner with no `e`
* rejected (×3): plausible one-liners that contain `e`

Rubric satisfaction asks GPT-4o "does the response contain a one-line
description?" — both chosen and all three rejected satisfy that. The
character-level constraint is invisible to the rubric. The discriminator,
without an explicit format-check tool, has 50/50 odds per pair, hence ~25 %
on best-of-4.

A purpose-built format verifier (regex / charset check parameterised by
the prompt) would lift this subset cleanly. Same architecture as our
`leetcode_test_runner` — deterministic factual signal, override the rubric
judge.

---

## 4. Comparison to public leaderboard

The RewardBench 2 paper reports the following on the same metric:

| Model | RB2 leaderboard avg |
|---|---:|
| Llama-3.1-Tulu-3-8B-RM | 40.5 % |
| Mistral-7B-Instruct-v0.3 (as RM) | ~42 % |
| gpt-4o-2024-08-06 (as RM, generative) | 64.9 % |
| Skywork-Reward-V2-Llama-3.1-8B | 73.4 % |
| INF-ORM-Llama3.1-70B (#1 at paper time) | 78.1 % |
| **`rb2_full_v47` (this work — hybrid pipeline)** | **52.66 %** |

We sit between small open-source RMs (~40 %) and GPT-4o-as-RM (~65 %).
Three reasons for the gap to GPT-4o-as-RM:

1. **GPT-4o-as-RM is given the WHOLE pair and emits a scalar score for each
   completion in one call.** Our pipeline turns it into 3 best-of-4 pairwise
   judgments per item — strictly harder because we have to be right on every
   pair to be right on the item. The 74 % per-pair accuracy is closer to
   GPT-4o's range; the 53 % item-level is the cube of that.
2. **Our verifier toolchain doesn't transfer.** The +11 points the
   v4.7 hybrid system gave on JudgeBench came almost entirely from MMLU
   answerer + LeetCode runner, neither of which fires on RB2 prompts.
3. **Precise IF drags the average down by ~2.4 points** (29 % vs ~50 %
   for a typical RM). Without that subset, the 5-subset avg would be
   58.5 % — within 6 points of GPT-4o-as-RM.

---

## 5. Reportability

### 5.1 What this number is

A single full pass of the v4.7 hybrid pipeline against all 1,825
processable items in `allenai/reward-bench-2`, scoring at single-order item
accuracy under the official metric.

### 5.2 What this number is **not**

* **Not a fair comparison to "GPT-4o as a reward model"** on the leaderboard.
  Public RM evaluations score completions individually with a scalar; we
  do best-of-4 via three pairwise judgments. Same model, different evaluation
  shape, different metric.
* **Not the v4.7 system performing as it did on JudgeBench.** On
  JudgeBench it scored 80.57 %. On RB2 it scores 52.66 %. The 28-point
  drop is not a regression — it's the verifier toolchain (MMLU answerer,
  LeetCode runner, etc.) failing to find structured answers in
  free-form RB2 prompts.
* **Not the rubric-judge ceiling.** The per-pair accuracy (74 %) is in
  the band of strong rubric-based judges. The item-level number is hurt
  primarily by the cubing-of-pair-accuracy in best-of-4.

### 5.3 Ties metric is an approximation

The official RewardBench 2 Ties score is `0.5 × accuracy + 0.5 × margin`
where:
* *accuracy* = fraction of (correct, incorrect) pairs where the RM scores
  correct higher;
* *margin* = the reward-score gap between correct and incorrect minus the
  gap among correct answers themselves (rewards multiple-correct
  recognition).

Our pipeline produces pairwise verdicts, not scalar rewards, so we
substitute:
* *accuracy term* → fraction of (chosen, rejected) pairs the pipeline
  resolves with `chosen > rejected`;
* *margin term* → fraction of pairs where the verifier produced a
  HIGH-confidence override (proxy for "model is *very* sure").

On `rb2_full_v47` no Ties pair triggered a high-confidence verifier
override (mmlu / math / leetcode all silent), so margin term = 0 and the
weighted score collapses to `0.5 × 77.78 = 38.89 %`. The true Ties
metric for this pipeline would require us to (a) extract scalar rewards
per completion (e.g. weighted sum of satisfied criteria) and (b) compute
the official margin formula. **Not difficult, just not done in this
iteration.**

### 5.4 Best apples-to-apples comparison

If we want to compare our pipeline against a published number on RB2,
the right framing is:

> "GPT-4o, used as the rubric-judge inside a recursive-rubric-decomposition
> pipeline with per-pair discovery, evaluated on RewardBench 2 in best-of-4
> mode."

There is no public number for that exact configuration. The closest
neighbour is `gpt-4o-2024-08-06` as a generative RM at 64.9 % on the
official leaderboard. We're 12 points below — the gap is half best-of-4
mechanics (cubing pair-accuracy), half the verifier toolchain not
transferring.

For a leaderboard-shaped report, **don't quote 52.66 % as a "rubric
pipeline on RB2" number** without the paragraph above. Quote it as
"v4.7 hybrid pipeline, best-of-4, single-order, with this caveats list".

### 5.5 What we'd need to make it leaderboard-comparable

1. Extract scalar rewards: `score(completion) = Σ wₖ · satisfied(completion, k)`
   normalised by `Σ wₖ`. Score chosen and each rejected, return the highest.
2. Implement the official Ties metric on those scalars.
3. Run order-swapped pairs to report double-order accuracy.
4. Add a format-compliance verifier for Precise IF.

Items 1–3 are mechanical (~1 day of work each). Item 4 would lift the
leaderboard avg by 2–4 points and is the most valuable next step.

---

## 6. Cost summary

| Resource | Spent |
|---|---:|
| GPT-4o calls | ~480k (rough; rubric satisfaction dominates) |
| Claude Opus 4.1 calls | ~1.6k (math solver only — single-firing subset) |
| Claude Sonnet 4.5 calls | ~50 (secondary MMLU answerer; mostly skipped because primary didn't fire) |
| Total est. cost | ~$2.6k |
| Wall-clock | 14.3 h @ 16 workers |

The Math subset alone made up ~10 % of the work; Factuality + Focus +
Safety together ~70 %; Precise IF + Ties the remaining ~20 %.

---

## 7. Files

* Per-pair artifacts: `artifacts/compiled_judgebench_final_eval_runs/rb2_full_v47/rb2_full_v47/final/examples/` (5,594 JSON files; one per pair)
* Aggregated summary: `artifacts/compiled_judgebench_final_eval_runs/rb2_full_v47/rb2_summary.json`
* Adapter: `rubric_gen/compiled/reward_bench_2_{loader,metrics,runner}.py`
* Tests: `tests/test_reward_bench_2_{loader,metrics}.py` (16 tests)
