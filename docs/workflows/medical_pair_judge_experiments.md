# Medical Pair Relabel / Regen + Judge Experiments

> Inventory of every A/B + judge artifact derived from
> `data/medical_gpt41_answers_rl.jsonl`. All four runs are judged by
> `anthropic:claude-opus-4-7` using the same medical-quality rubric prompt
> (see `scripts/relabel_pair_dataset.py` for the original prompt; the regen
> runs reuse the identical judge prompt for diff-ability).

---

## 1. Master table

| # | Experiment | Side A | Side B | Regen | N | A | B | TIE | err | A% | B% | TIE% | A\|dec | B\|dec |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | phase1 (relabel only) | gpt-4o (orig) | gpt-4.1 (orig) | none | 356 | 19 | 209 | 125 | 3 | 5.3% | 58.7% | 35.1% | 8.3% | **91.7%** |
| 2 | B-regen GPT-5 (500) | gpt-4o (orig) | gpt-5 (regen) | B | 500 | 40 | 383 | 71 | 6 | 8.0% | 76.6% | 14.2% | 9.5% | **90.5%** |
| 3 | A-regen GPT-5 | gpt-5 (regen) | gpt-4.1 (orig) | A | 500 | 146 | 77 | 271 | 6 | 29.2% | 15.4% | 54.2% | **65.5%** | 34.5% |
| 4 | B-regen GPT-4o (intra-model) | gpt-4o (orig) | gpt-4o (regen) | B | 200 | 32 | 98 | 66 | 4 | 16.0% | 49.0% | 33.0% | 24.6% | **75.4%** |
| 5 | B-regen GPT-5 (4k, tie→A) | gpt-4o (orig) | gpt-5 (regen) | B | 4000 | 300 | 2947 | 720 | 33 | 7.5% | 73.7% | 18.0% | 9.2% | **90.8%** |

Column key:

- `A` / `B` / `TIE` — raw verdict counts.
- `err` — parse errors / empty verdicts (excluded from rates).
- `A%` / `B%` / `TIE%` — share of total `N`.
- `A|dec` / `B|dec` — share of decided pairs (`A + B`, excludes TIE and err).
- All judges = `anthropic:claude-opus-4-7`.

---

## 2. Per-experiment detail

### 2.1 phase1 — relabel only (no regen)

`data/medical_gpt41_answers_rl_relabeled_phase1.jsonl`

- Setup: original A (gpt-4o) vs original B (gpt-4.1); no regeneration, just
  Opus 4.7 picking a winner.
- N = **356** (subset; relabel was scoped via `--restrict-to-run-dir` to rows
  already rubric-scored at the time).
- Verdicts: A = 19 (5.3%), **B = 209 (58.7%)**, TIE = 125 (35.1%),
  parse errors = 3.
- Decided pairs (228): A 8.3% vs **B 91.7%**.
- Confidence: high 193, medium 150, low 10.
- Source script: `scripts/relabel_pair_dataset.py`.

### 2.2 B-regen GPT-5

`data/medical_gpt5_answers_rl_opus_judged.jsonl`

- Setup: original A (gpt-4o) vs **new B (gpt-5)**, judged by Opus 4.7.
- N = **500**.
- Verdicts: A = 40 (8.0%), **B = 383 (76.6%)**, TIE = 71 (14.2%),
  parse errors = 6.
- Decided pairs (423): A 9.5% vs **B 90.5%**.
- Confidence: high 304, medium 171, low 19.
- Largest quality gap of any pair → lowest TIE rate (14%).
- Source script: `scripts/regenerate_a_and_judge.py --regen-side B
  --writer-model openai:gpt-5`.

### 2.3 A-regen GPT-5

`data/medical_gpt5_a_vs_gpt41_b_opus_judged.jsonl`

- Setup: **new A (gpt-5)** vs original B (gpt-4.1), judged by Opus 4.7.
- N = **500**.
- Verdicts: **A = 146 (29.2%)**, B = 77 (15.4%), TIE = 271 (54.2%),
  parse errors = 6.
- Decided pairs (223): **A 65.5%** vs B 34.5%.
- Confidence: high 286, medium 188, low 20.
- Highest TIE rate (54%) — gpt-5 and gpt-4.1 are close in quality on this
  task.
- Source script: `scripts/regenerate_a_and_judge.py --regen-side A
  --writer-model openai:gpt-5`.

### 2.4 B-regen GPT-4o (intra-model)

`data/medical_gpt4o_b_regen_opus_judged.jsonl`

- Setup: original A (gpt-4o) vs **new B (gpt-4o)**, both same family,
  judged by Opus 4.7.
- N = **200**.
- Verdicts: A = 32 (16.0%), **B = 98 (49.0%)**, TIE = 66 (33.0%),
  parse errors = 4.
- Decided pairs (130): A 24.6% vs **B 75.4%**.
- Confidence: high 77, medium 110, low 9.
- Source script: `scripts/regenerate_a_and_judge.py --regen-side B
  --writer-model openai:gpt-4o`.

### 2.5 B-regen GPT-5, full 4k, tie→A (Opus judge)

`data/medical_gpt5_b_regen_4k_opus_judged.jsonl`
(slim 5-field version: `data/medical_gpt5_b_regen_4k_rl.jsonl`)

- Setup: original A (gpt-4o) vs **new B (gpt-5)**, judged by Opus 4.7.
  Same configuration as §2.2 but at full scale (all 4000 rows of
  `medical_gpt41_answers_rl.jsonl`) and with `--tie-policy a` so TIE
  verdicts route to `reference_answer_a` in the output `correct_answer`.
- N = **4000** (full dataset).
- Verdicts: A = 300 (7.5%), **B = 2947 (73.7%)**, TIE = 720 (18.0%),
  parse errors = 33 (0.8%).
- Decided pairs (3247): A 9.2% vs **B 90.8%** — virtually identical to the
  §2.2 500-row pilot (A 9.5% vs B 90.5%), confirming the gap is stable at
  scale.
- Confidence: high 2364 (59.1%), medium 1492 (37.3%), low 111 (2.8%),
  empty 33 (0.8%).
- **Final `correct_answer` distribution after `tie-policy=a`** (this is
  what downstream consumers of the JSONL will see):
  - `reference_answer_a` = **1053 (26.3%)** — sum of 300 A wins
    + 720 TIE → A + 33 parse-error fallbacks.
  - `reference_answer_b` = **2947 (73.7%)** — pure B wins (parse errors
    are not routed to B regardless of tie policy).
- Wall-clock: 56.7 min at workers=8 (502 regen-cache hits from the §2.2
  pilot's 500-row prefix; judge cache from the pilot did not hit because
  the prior cache file uses a different hashing scheme — see Reproduce
  section §4 for the current key formula).
- Source script: `scripts/regenerate_a_and_judge.py --regen-side B
  --writer-model openai:gpt-5 --tie-policy a --limit 4000 --workers 8`.

---

## 3. Cross-experiment observation

In **every** regen experiment (#2, #3, #4, #5) the regenerated side wins
among decided pairs:

| # | Regen side | N | Regen-side win rate (decided) |
|---|---|---:|---:|
| 2 | B (gpt-5)              |  500 | 90.5% |
| 3 | A (gpt-5)              |  500 | 65.5% |
| 4 | B (gpt-4o, intra-model)|  200 | 75.4% |
| 5 | B (gpt-5, full 4k)     | 4000 | 90.8% |

Since the regen side flips between A (#3) and B (#2, #4), this can't be pure
position bias. The most consistent explanation is **regen-prompt asymmetry**:
the explicit "expert medical assistant… clinically faithful, well-reasoned,
cover relevant facts/mechanisms/guidelines/contraindications" system prompt
in `_REGEN_SYSTEM_PROMPT` produces answers that Opus rates as more thorough
than whatever generated the original A/B.

The intra-model run (#4) is the cleanest evidence: both sides are gpt-4o, but
the regen still wins 3:1 — that delta is essentially the prompt-style
premium, not a model-quality premium.

**Practical implication for RL reward signal**: the labels in #2/#3/#4/#5
reflect "Opus prefers a thorough clinician-style answer" and not just
"model X is stronger than model Y". The phase1 file (#1, no regen) is the
only one without that confound.

---

## 4. Rubric-pipeline validation against §2.5 (4k, GPT-5 B vs GPT-4o A)

This is the JudgeBench-350-style audit: the rubric pipeline (trained on
1000 medical_rl_prompts examples in shard 0) scores each candidate against
its discovered rubric bank, and the higher-ranked candidate is the
pipeline's "preferred" answer. We measure how often that preference matches
the §2.5 Opus 4.7 label (`correct_answer`).

### 4.1 Setup

- **Source dataset:** `data/medical_gpt5_b_regen_4k_rl.jsonl` (4000 pair
  rows, slim schema: `id`, `question`, `reference_answer_a`,
  `reference_answer_b`, `correct_answer` from §2.5 with `tie-policy=a`).
- **Training source:** shard 0 of `medical_rl_prompts.jsonl` (rows 0..999),
  produced by `python -m rubric_gen.cli --preset judgebench-v47-medical
  --split train --num-shards 3 --shard-index 0 --run-name
  medical_v47_train_shard0`. Audited as up-to-date with current code: every
  in-tree change since training is additive and gated, none alters
  `--split train` output on `medical_rl_prompts.jsonl`.
- **Rubric index:** `artifacts/medical_rl/rubric_index/
  medical_v47_shard0_index.json` (built from shard 0's discovered RRD
  rubrics, embedded with `text-embedding-3-small`).
- **Validation command:**
  ```
  python -m rubric_gen.cli \
    --dataset-path data/medical_gpt5_b_regen_4k_rl.jsonl \
    --preset judgebench-v47-medical \
    --split all --num-shards 1 \
    --sample-workers 8 --max-workers 4 --resume --rubrics-only \
    --medical-rubric-index artifacts/medical_rl/rubric_index/medical_v47_shard0_index.json \
    --medical-rubric-retrieval-top-k 8 \
    --relevance-filter-strictness conservative \
    --run-name medical_v47_pair_validation_gpt5_b_4k \
    --output-dir artifacts/medical_rl_validation
  ```
- **Run dir:**
  `artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k/`
- **Wall clock:** 8.4 hours at `sample_workers=8`, `max_workers=4`.

### 4.2 Headline (from `summaries/summary_v2.md`, post-tie-fix)

The original `summary.md` reported `rrd_whitened_uniform = 73.17%` and
`rrd_uniform = 71.78%`. Those numbers benefited from a hidden alphabetical
tie-break in the ranker that silently picked `pair_a` whenever the two
anchors had identical raw scores (498 / 4000 score-tied rows for
`rrd_whitened_uniform`).

`_pair_preference_outcome` was then changed to read **raw scores**
instead of ranks and to count **score equality as wrong** (mirroring
JudgeBench's `compiled/judgebench_eval.py` behaviour, which honours a
strict `decision == "A=B"` rather than breaking it arbitrarily). The same
4000 per-example artifacts were re-aggregated via
`scripts/reaggregate_pair_validation.py`.

| Method | Pair Pref Acc | Correct / N | Δ vs old (alphabetical-tie-break) |
|---|---:|---:|---:|
| `rrd_uniform`         | **63.85%** | 2554 / 4000 | -7.95 pp |
| `rrd_whitened_uniform` | **67.23%** | 2689 / 4000 | -5.97 pp |
| `one_shot_rubrics_only` | n/a | 0 / 0 | — |

The 67.23% figure is the **apples-to-apples comparable number** to
JudgeBench: same metric definition, same single-order presentation, same
strict-tie semantics, same GPT-4o judge.

#### Old vs new headline summary

| Method | Old (alpha-tie → silent A) | New (score-tie → wrong) | Drop |
|---|---:|---:|---:|
| `rrd_uniform`         | 71.78% (2871/4000) | 63.85% (2554/4000) | -7.95 pp |
| `rrd_whitened_uniform` | 73.17% (2927/4000) | 67.23% (2689/4000) | -5.97 pp |

The 5.97 pp drop on `rrd_whitened_uniform` reflects 238 score-tied rows
where alphabetical tie-break previously matched the gold (gold was A)
+ the 260 score-tied rows where it previously didn't (gold was B); under
the new rule all 498 score-tied rows count as wrong. (For
`rrd_uniform` there are 794 score-tied rows, hence the larger drop.)

The original artifacts (`summary.md`, `method_metrics.csv`,
`summary.json`) are preserved on disk; the corrected outputs are written
side-by-side as `summary_v2.md`, `method_metrics_v2.csv`,
`summary_v2.json` so the full audit trail is intact.

### 4.3 Baseline comparison — important caveat

The §2.5 label distribution is **73.7% B / 26.3% A** because GPT-5 won
~73% of decided pairs vs GPT-4o under Opus 4.7. That makes the trivial
"always pick B" predictor a 73.67% accuracy baseline.

| Predictor | Accuracy | Lift over always-B |
|---|---:|---:|
| Always-A baseline      | 26.32% | -47.35 pp |
| Always-B baseline      | 73.67% |  0.00 pp |
| `rrd_uniform` (post-tie-fix)         | 63.85% | -9.82 pp |
| `rrd_whitened_uniform` (post-tie-fix) | 67.23% | **-6.44 pp** |

Under the strict-tie rule, the pipeline is clearly **below** always-B
(the prior is hard to beat when labels are 74/26 skewed and the
pipeline's actual signal is modest). On chance-corrected balanced
accuracy (see §4.4) the picture is still meaningful.

### 4.4 Per-class recall — actual discrimination signal (post-tie-fix)

Raw accuracy hides the class-imbalance problem. Under the new
score-tie-as-wrong rule the pipeline IS still making different predictions
per class, but the gold-A recall takes a noticeable hit because most of
the score-tied rows were previously credited as "pred A by tie-break" and
some of those genuinely matched gold A:

| Method | Recall on gold A (1053) | Recall on gold B (2947) | Balanced acc |
|---|---:|---:|---:|
| Always-B baseline      |  0.0% | 100.0% | 50.00% |
| `rrd_uniform`          | 35.3% (372/1053) | 74.0% (2182/2947) | 54.69% |
| `rrd_whitened_uniform` | 38.6% (406/1053) | 77.5% (2283/2947) | **58.01%** |

**Confusion matrix for `rrd_whitened_uniform`** (pred ∈ {A, B, tie}):

|              | gold A | gold B |
|---|---:|---:|
| pipeline picks A | **406** | 401 |
| pipeline picks B | 412 | **2283** |
| pipeline ties (`A=B` → wrong) | 235 | 263 |

Of the 4000 evaluable rows, **498 / 4000 = 12.5%** end in a strict
score-tie for `rrd_whitened_uniform` (794 / 4000 = 19.85% for
`rrd_uniform`) — these are the rows where the rubric satisfaction calls
landed on identical aggregate scores for both anchors.

**Take-aways under the strict rule:**
- Balanced accuracy of **58.0%** for `rrd_whitened_uniform` vs 50% for any
  constant predictor — still real signal, but a 11.2 pp drop from the old
  alphabetical-tie-break number (69.2%). The hidden tie-break was doing
  meaningful work for the gold-A class.
- On Opus-decided (no Opus tie/error) rows: 75.58% accuracy held under the
  old rule. Under the new strict rule, it drops to ~70%. The next
  question is whether the 498 score-ties cluster in any particular
  source-family or rubric-bank shape — that's a follow-up not yet run.
- A reasonable engineering response is to **reduce the score-tie rate**
  (e.g., add more rubrics, swap aggregation to whitened-uniform always,
  or break score-ties using a secondary signal that's not arbitrary like
  alphabetical) — anything that removes the 498-row ambiguous regime
  buys back accuracy without a hidden bias.

### 4.5 Retrieval seed bookkeeping

Across all 4000 rows:

- 3207 rows had ≥1 seed rubric retrieved + filtered through (80.2%).
- 793 rows (19.8%) had **zero seed inputs** — usually because the Sonnet
  conservative filter dropped all 8 retrieved rubrics as IRRELEVANT for
  that prompt. These rows fell back to pure RRD discovery.
- For rows with seeds: avg 3.05 seed inputs → 2.61 accepted by RRD's
  acceptance filter → 0.44 rejected.

### 4.6 Comparison context — why this is harder than JudgeBench

| Benchmark | Pair-pref accuracy | Setup |
|---|---:|---|
| JudgeBench v4.7 (350 rows), full system | ~80% | + math/code/MMLU/reasoning HP verifier overrides |
| JudgeBench v4.7 (350 rows), **rubric core only** | **69.43%** | GPT-4o judge, single-order, strict-tie semantics |
| **Medical 4k (this run), rubric core, post-tie-fix** | **67.23%** | Same metric semantics; Opus-labelled, GPT-5 B vs GPT-4o A |
| Same, balanced acc (chance-corrected)            | ~63%   | Removes the 74/26 label-prior advantage |

So at the apples-to-apples (rubric-core, strict-tie) level the medical
pipeline scores **2.20 pp below** JudgeBench's rubric core. Reasonable
given:

- Same model family on both sides (GPT-4o vs GPT-5) — both responses are
  medically substantive.
- Even Opus 4.7 (the labeler) ties on 18% of intra-model regen pairs
  (§2.4) — the discrimination task is fundamentally hard.
- The HP verifier overrides that get JudgeBench from 69% to 80% don't
  fire on free-text medical Q&A (no sympy/code-execution/MMLU oracle).

The medical pair task is fundamentally harder discrimination: even the
Opus judge (the labeler!) found ~18% of pairs ties when both candidates
were strong (see §2.4 intra-model run). The rubric pipeline, with no
explicit clinical-format prior, is asked to discriminate two valid medical
answers on substance alone.

### 4.7 Reproduce the per-class breakdown

```
python scripts/analyze_pair_validation.py \
  --run-dir artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k
```

Computes: gold-class counts, always-A/B baselines, retrieval seed stats,
per-method recall on each gold class, and a confusion matrix. The
predicted label is taken from candidate rank (lower = winner), matching
`reporting._pair_preference_outcome`.

---

## 5. Iterated validation: shard 0 failure analysis + tiebreaker cascade

After §4 surfaced the 67.23% post-tie-fix headline, we ran a focused
failure-analysis pass on the shard 0 training artifacts and the 4k
validation artifacts, picked the highest-confidence intervention from
the resulting menu, and re-scored post-hoc (no pipeline re-run).

### 5.1 Diagnostics summary

Two read-only diagnostic scripts (`scripts/diagnose_shard0_training.py`
and `scripts/diagnose_4k_validation.py`) characterised the system. The
artifacts and full reports live under
[artifacts/diagnostics/](artifacts/diagnostics/). Headline findings:

**Training (shard 0, 1000 examples, 5753 unique rubrics):**

- Mean bank size: 5.75 rubrics per example (median 6, max 9). Small
  banks → small score space → score-tie regime.
- RRD termination: every example hit the rejection ceiling
  (`term_rejections = 15`). The proposer wants more rubrics; the
  acceptance filters reject them.
- Decomposition is essentially flat: only 36 / 5753 rubrics (0.6%)
  reach depth=1. RRD's recursive machinery is not adding depth here.
- 10.9% of rubrics fire on ALL training candidates → zero ranking
  signal. 1.0% fire on zero candidates.
- 47.8% of training examples generate ZERO rubrics mentioning the
  thoroughness/clinical-format axis (covers, comprehensive,
  differential, guidelines, contraindications, etc.) — the very axis
  Opus rewards (§3 cross-experiment).

**Validation (4k):**

- Per-source-family accuracy varies wildly under the strict rule:
  `clinical_decision_support` 84%, `general_instruction_following` 67%,
  `documentation_variants` only 49% (with 28.7% tie rate).
- Score-tie rate is bank-size dependent: bank ≤ 4 → 31.4% ties,
  bank 5-7 → 14.4%, bank ≥ 8 → 6.1%.
- Mean `sat_a − sat_b = −1.31` (B satisfies 3.69 rubrics on average vs
  A's 2.38). The pipeline DOES pick up the thoroughness axis — but
  overshoots, picking B even when Opus picked A.
- Per-Opus-bucket accuracy: A=44.7%, B=77.5%, TIE-forced-to-A=36.1%,
  parse-err-forced-to-A=36.4%.
- Retrieval health: 19.8% of rows get zero seeds (Sonnet conservative
  filter drops all 8 retrieved hits). Mean seeds in = 2.44, accepted by
  RRD = 2.09.

The full analysis + ranked intervention menu is at
[artifacts/diagnostics/findings_and_fix_plan.md](artifacts/diagnostics/findings_and_fix_plan.md).

### 5.2 Chosen intervention: `uniform_then_judge` cascade

Mirrors `rubric_gen/compiled/judgebench_eval.py`'s
`uniform_breaks_strict_tie` pattern, with a final LLM fallback when
both whitened-uniform and uniform tie:

```
predicted = arg-max(rrd_whitened_uniform.score_a, score_b)
            -- score-tie? -->
predicted = arg-max(rrd_uniform.score_a, score_b)
            -- still tied? -->
predicted = direct_judge_pair(question, response_a, response_b,
                              judge=openai:gpt-4o, temperature=0.0)
```

Implemented in [rubric_gen/evaluation/pair_tiebreaker.py](rubric_gen/evaluation/pair_tiebreaker.py)
+ [scripts/rescore_with_tiebreaker.py](scripts/rescore_with_tiebreaker.py).
Caches every direct-judge call so re-runs are free. All 461 tests pass,
including 30 new tests in `tests/test_pair_tiebreaker.py`.

The fix is **post-hoc**: it consumes the existing per-example artifacts
and re-decides the score-tied subset. No pipeline re-run, no rubric
regeneration. Total cost for the 4k rescore: **~$2.45 (491 GPT-4o calls)
+ 78.7 seconds** vs the originally-budgeted $700 / 8 hours for a
pipeline re-validation.

### 5.3 Smoke (200 rows)

Same first 200 rows under both rules:

| Strategy | Acc | Recall A (45) | Recall B (155) |
|---|---:|---:|---:|
| `none` (= post-tie-fix baseline)        | 76.00% | 44.44% | 85.16% |
| `uniform_then_judge`                    | **81.00%** | **53.33%** | **89.03%** |
| Δ                                        | **+5.00 pp** | +8.89 pp | +3.87 pp |

Score-tied rows on the smoke: 16 → 5 (the 5 are GPT-4o judge ties,
counted as wrong). 11 ties resolved via judge (10 / 11 = 90.9% correct
on those).

### 5.4 Full 4k rescore

`artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k/summaries/summary_v3.md`:

| | Strict (§4.2 baseline) | Cascade (this section) | Δ |
|---|---:|---:|---:|
| **Accuracy**           | **67.23%** (2689/4000) | **72.17%** (2887/4000) | **+4.94 pp** |
| Recall on gold A (1053) | 38.56% | **45.20%** (476) | +6.64 pp |
| Recall on gold B (2947) | 77.47% | **81.81%** (2411) | +4.34 pp |
| Balanced accuracy       | 58.0%  | **63.5%**          | +5.5 pp |
| Pipeline-side ties remaining | 498 | 258 (judge said TIE) | -240 |

**Decision-policy breakdown** (where the final pick came from):

| Policy | N | Correct | Acc on policy |
|---|---:|---:|---:|
| `primary` (whitened-uniform strict winner) | 3502 | 2689 | 76.78% |
| `uniform`  (uniform broke a whitened tie) |    7 |    6 | 85.71% |
| `judge`    (GPT-4o broke both ties)        |  233 |  192 | **82.40%** |
| `judge_tied` (GPT-4o also tied → wrong)    |  258 |    0 |  0.00% |

The judge is doing real work: 82.4% correct on the rows it broke. But
it tied 258 / 491 = **52.5%** of the time it was called — these are
genuinely hard rows where even GPT-4o can't pick a winner. Lowering
this would require either a stronger judge (Opus 4.7 — but that's
circular since Opus is the labeler) or richer rubric input to the
pipeline so fewer rows reach the cascade in the first place.

### 5.5 Where we land vs the baselines

| | Accuracy on 4k | Notes |
|---|---:|---|
| Always-B trivial baseline   | 73.67% | The label distribution prior |
| `rrd_whitened_uniform` (old alphabetical-tie-break, §4.2 OLD) | 73.17% | Inflated by hidden A-bias on score-ties |
| `rrd_whitened_uniform` (strict, §4.2 NEW)                     | 67.23% | JudgeBench-comparable |
| **`rrd_whitened_uniform` + uniform_then_judge cascade (§5)**  | **72.17%** | Honest tiebreaker; +4.94 pp over strict |
| JudgeBench 350, rubric-core only                              | 69.43% | Same metric semantics |
| JudgeBench 350, full v4.7 (with HP verifiers)                 | 80.57% | HP verifiers don't transfer |

The cascade lands the medical pipeline **+2.74 pp above JudgeBench's
rubric-core baseline** under apples-to-apples semantics, and **−1.50 pp
below the always-B trivial baseline** in raw accuracy. On balanced
accuracy (chance-corrected) the gap is more meaningful: 63.5% vs the
50% chance / 50% always-B-balanced floor.

### 5.6 What the cascade does NOT fix (open questions)

- The 258 judge-tied rows (6.45% of all 4k) are still penalised — these
  are the genuinely hard pairs. Stronger labels (Opus instead of GPT-4o
  in the cascade) would be circular against the §2.5 ground truth.
- The 478 / 1000 training examples missing thoroughness rubrics, the
  10.9% useless rubrics, and the always-15 RRD termination ceiling are
  upstream issues the post-hoc fix doesn't touch. Addressing them
  needs a fresh training run (Tier 3 — extending to shards 1+2 or
  retraining shard 0 with a stronger proposer / looser RRD acceptance
  filter).
- The 19.8% zero-seed retrieval rate is similarly an upstream issue
  (loosen the Sonnet filter or expand the index).

### 5.7 Reproduce

```
# Diagnostics
python scripts/diagnose_shard0_training.py \
  --run-dir artifacts/medical_rl/runs/medical_v47_train_shard0
python scripts/diagnose_4k_validation.py \
  --run-dir artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k

# Cascade rescore (full 4k, ~$2.50, ~80 sec wall-clock at workers=8)
python scripts/rescore_with_tiebreaker.py \
  --run-dir artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k \
  --strategy uniform_then_judge --workers 8 --suffix _v3

# Strategy variants for comparison
python scripts/rescore_with_tiebreaker.py --run-dir <run> --strategy none      --suffix _strict
python scripts/rescore_with_tiebreaker.py --run-dir <run> --strategy uniform   --suffix _uniform_only
python scripts/rescore_with_tiebreaker.py --run-dir <run> --strategy judge     --suffix _judge_only
```

The original `summary.md`, `summary_v2.md`, and the new `summary_v3.md`
all live side-by-side under the run's `summaries/` directory. The
tiebreaker LLM cache is at
`artifacts/medical_rl_validation/runs/cache/pair_tiebreaker.jsonl`.

---

## 6. Reproduce the master table

The numbers above are computed directly from each JSONL by trusting the
`judge_verdict` field (no fallback to `correct_answer`, since the regen
scripts default `correct_answer` to the original label on parse errors,
which would inflate the A count).

```bash
python - <<'PY'
import collections, json
from pathlib import Path

ARTIFACTS = [
    ("phase1 (relabel)",       "data/medical_gpt41_answers_rl_relabeled_phase1.jsonl"),
    ("B-regen GPT-5 (500)",    "data/medical_gpt5_answers_rl_opus_judged.jsonl"),
    ("A-regen GPT-5",          "data/medical_gpt5_a_vs_gpt41_b_opus_judged.jsonl"),
    ("B-regen GPT-4o (intra)", "data/medical_gpt4o_b_regen_opus_judged.jsonl"),
    ("B-regen GPT-5 (4k)",     "data/medical_gpt5_b_regen_4k_opus_judged.jsonl"),
]

for label, path in ARTIFACTS:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    c = collections.Counter()
    for r in rows:
        v = str(r.get("judge_verdict") or "").strip().upper()
        c[v if v in {"A", "B", "TIE"} else ""] += 1
    n = len(rows)
    dec = c["A"] + c["B"]
    print(f"{label:<26} N={n}  A={c['A']}  B={c['B']}  TIE={c['TIE']}  err={c['']}  "
          f"A|dec={c['A']/dec*100:.1f}%  B|dec={c['B']/dec*100:.1f}%")
PY
```
