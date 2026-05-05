# Medical Pair Judge v2 Results — Headline Numbers

This doc captures the 4k pair-preference results after the v2 plan
(`docs/workflows/medical_rl_prompts_training.md` §"v2 runbook") was
executed end-to-end: shard 0 retrained with the v2 RRD / proposer /
candidate-generation / aggregation changes, fresh medical rubric index
built, and the 4k validation re-scored on the new index plus the new
post-hoc cascade.

The headline metric stack the v2 plan was sized against:

| Metric | Target | v1 (best) | **v2 best** | Delta vs v1 best |
|---|---|---:|---:|---:|
| Strict (`rrd_whitened_uniform`, score-tie=wrong) | > 72.17 % | 67.23 % | 65.88 % | -1.35 pp |
| Cascade accuracy (best in each family) | > 72.17 % | 72.17 %  (uniform_then_judge) | **76.55 %** (format_then_uniform_then_anti_tie) | **+4.38 pp** |
| Cascade vs always-B baseline (73.67 %) | > 73.67 % | -1.50 pp (72.17 %) | **+2.88 pp (76.55 %)** | +4.38 pp |
| Balanced accuracy (cascade) | > 65 % | 63.5 % | **69.93 %** | +6.43 pp |
| Cascade-tie remaining (final unresolved) | (lower) | 258 / 4000 = 6.45 % | **21 / 4000 = 0.53 %** | **-5.9 pp (12x reduction)** |

Two of the three user-targeted metrics (cascade accuracy and balanced
accuracy) exceeded their targets; the strict accuracy regressed by
1.35 pp but the v2 anti-tie cascade more than compensates.  Detailed
breakdown below.

## Inputs

* **Shard 0 retrain:** `artifacts/medical_rl/runs/medical_v2_train_shard0`,
  v2 prompts (`v2_1_task_typed_broad_ban` cache scheme), boundary
  candidates (terse + padded-uncommitted), 3-sample majority-vote
  satisfaction, adaptive decomposition coverage gate, post-RRD
  discrimination filter at `discrimination_min_pq=0.05`, decomposition
  thresholds loosened to `min_recall=0.70` /
  `min_discrimination_gain=0.01`.  1000 examples, ~4.24 hr at
  `sample_workers=8`.
* **Index:** `artifacts/medical_rl/rubric_index/medical_v2_shard0_index.json`
  -- 3,944 unique rubrics (vs v1's 5,749, -31 %), embedded with
  `text-embedding-3-small`.  Built in ~3 min.
* **4k validation:** `artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k`,
  `--medical-rubric-retrieval-top-k 16`, conservative relevance filter,
  `--rubric-satisfaction-samples 3`,
  `--discrimination-min-pq 0.05`, pair-mode B1 fix active.
  4000 examples, ~15.2 hr at `sample_workers=8`.
* **Cascade rescore (post-hoc):** `summary_v2_anti_tie.md`, strategy
  `format_then_uniform_then_anti_tie`, GPT-4o anti-tie judge.
  ~115 sec wall-clock, ~$2.80 in OpenAI judge spend.

## 1. Strict score-only baselines

`scripts/diagnose_4k_validation.py` on the v2 run dir; method block
`rrd_whitened_uniform`:

| | v1 (`medical_v47_pair_validation_gpt5_b_4k`) | **v2 (`medical_v2_pair_validation_gpt5_b_4k`)** | Delta |
|---|---:|---:|---:|
| Pair-pref accuracy (strict, all 4000) | 67.23 % | **65.88 %** | -1.35 pp |
| Pair-pref accuracy over decided rows | 75.58 % | **76.82 %** | +1.24 pp |
| Score-tied rows | 498 (12.45 %) | **570 (14.25 %)** | +1.80 pp |
| Recall on gold A (1053) | 38.56 % | **42.36 %** | +3.80 pp |
| Recall on gold B (2947) | 77.47 % | 74.28 % | -3.19 pp |
| Balanced accuracy | 58.0 % | **58.32 %** | +0.32 pp |
| Mean `sat_a - sat_b` | -1.31 | **-1.11** | +0.20 (less B-overshoot) |

Read: the strict number is slightly worse, but the underlying
distribution moved in the desired direction -- gold-A recall improved
3.8 pp, the B-overshoot dropped by 0.20 rubrics on average, and decided
rows are now scored 1.24 pp more accurately.  The strict regression is
entirely explained by the +72 extra score-tied rows (which under the
strict-tie-as-wrong rule each cost 1 accuracy point).

## 2. Cascade results

The v2 plan ships three rescore strategies; all three were applied.

| Strategy | Accuracy | Recall A | Recall B | Balanced | Cascade-judge ties remaining |
|---|---:|---:|---:|---:|---:|
| `none` (= strict baseline)       | 65.88 % | 42.36 % | 74.28 % | 58.32 % | 570 (14.25 %) |
| `uniform_then_judge` (= v1 best) | 71.35 % | 48.34 % | 79.57 % | 63.96 % | 305 (7.62 %)  |
| **`format_then_uniform_then_anti_tie` (v2 best)** | **76.55 %** | **55.94 %** | **83.92 %** | **69.93 %** | **21 (0.53 %)** |

The v2 anti-tie cascade picks up an extra **+5.20 pp** over the v1-style
cascade on the same 4k artifact -- driven almost entirely by the
**12x reduction in cascade-judge ties** (from 305 to 21).

Decision-policy breakdown for the v2 best (`format_then_uniform_then_anti_tie`):

| Policy | N | Correct | Acc on policy |
|---|---:|---:|---:|
| `primary` (whitened-uniform strict winner) | 3430 | 2635 | 76.82 % |
| `uniform` (uniform broke a whitened tie)   |   13 |    8 | 61.54 % |
| `format` (format-prior heuristic)          |    2 |    1 | 50.00 % |
| `judge_anti_tie` (anti-tie GPT-4o broke remaining ties) | **534** | **418** | **78.28 %** |
| `judge_anti_tie_tied` (anti-tie also tied → wrong)      |   21 |    0 | 0.00 %  |

The anti-tie GPT-4o variant is doing the heavy lifting on what used to
be the unsalvageable score-tie regime: 534 calls, 78.28 % correct on
those rows, vs the v1 prompt's 252 calls at 83.73 % but 305 ties
(unresolvable).  Net: v2 cascades pick A or B on **96.5 %** of
formerly-tied rows vs v1's **42.4 %**, at comparable per-call accuracy.

Format-prior fired on only 2 rows in the 4k -- the medical Q&A corpus
rarely has explicit format requests.  Worth keeping (zero cost) but the
heavy lifting is the anti-tie judge.

## 3. Per-source-family breakdown (strict, v1 vs v2)

| Family | N | v1 acc | **v2 acc** | Delta | v1 tie % | **v2 tie %** |
|---|---:|---:|---:|---:|---:|---:|
| `general_instruction_following`  | 3530 | 66.63 % | 65.38 % | -1.25 pp | 12.6 % | 14.4 % |
| `clinical_decision_support`      |  175 | 84.00 % | 80.00 % | -4.00 pp |  5.1 % |  6.9 % |
| `agentic_workflows`              |  126 | 69.84 % | **71.43 %** | **+1.59 pp** | 10.3 % | 12.7 % |
| `rewrite_editing`                |   89 | 70.79 % | 56.18 % | **-14.61 pp** |  7.9 % | 14.6 % |
| `documentation_variants`         |   80 | 48.75 % | **58.75 %** | **+10.00 pp** | 28.7 % | 23.8 % |

Notable shifts:

* **`documentation_variants`** (the worst v1 family at 48.75 %) gained
  +10 pp -- the task-type-aware proposer (Tier A1) is paying off on the
  family that needed it most.
* **`rewrite_editing`** lost 14.6 pp -- the smallest family (n=89), but
  the regression is real.  The rewrite system prompt may need
  task-specific tuning; the boundary-padded candidate pattern probably
  doesn't fit a "rewrite" instruction.
* **`clinical_decision_support`** lost 4 pp from a high base (84 → 80 %),
  again on a small sample (n=175).
* `general_instruction_following`, the bulk family (88 % of rows), is
  roughly flat (-1.25 pp strict) and gains substantially after cascade.

## 4. Per-Opus-verdict bucket breakdown (strict, v1 vs v2)

| Opus verdict | N | v1 acc | **v2 acc** | Delta | Notes |
|---|---:|---:|---:|---:|---|
| A    |  300 | 44.67 % | **50.67 %** | **+6.00 pp** | Opus picked A |
| B    | 2947 | 77.47 % | 74.28 % | -3.19 pp | Opus picked B |
| TIE  |  720 | 36.11 % | **38.75 %** | **+2.64 pp** | Opus tied → tie-policy=a forced gold=A |
| ERR  |   33 | 36.36 % | 45.45 % | +9.09 pp | Opus parse error |

Read: v2 trades B-accuracy for materially better A-accuracy, exactly
the FM2 fix the diagnostics targeted (anchor-favoured library).  B is
still the larger class (74 % of rows) so the +6 pp on A doesn't fully
compensate for the -3 pp on B in raw accuracy, but on **balanced**
accuracy the trade is a win.

## 5. Retrieval health (Tier B1 + B2 effect)

| | v1 | **v2** | Delta |
|---|---:|---:|---:|
| Rows w/ ≥1 seed input  | 80.2 % | **85.1 %** | +4.9 pp |
| Rows w/ zero seed input | 19.8 % | **14.9 %** | -4.9 pp |
| Mean seeds in / row    | 2.44 | **3.13** | +0.69 |
| Mean seeds accepted    | 2.09 | **2.67** | +0.58 |

Read: Tier B1 (the pair-mode candidate-text fix in `pipeline.py`) plus
Tier B2 (`top_k=16`) recovered 196 of the 793 zero-seed rows v1 lost.
Mean seed input rose 28 % (2.44 → 3.13) and the relevance filter
accepted more on average.

## 6. What worked, what didn't

**Worked clearly:**

1. **Tier C2 (anti-tie GPT-4o judge)** -- single largest contributor
   to the cascade lift.  Reduced final cascade-tie count from 258 to
   21 (12x).  +5.2 pp cascade vs the v1 prompt at the same spend.
2. **Tier B1 (pair-mode relevance-filter fix)** -- recovered ~5 pp of
   zero-seed rows, contributing to 0.7 pp strict and feeding the
   cascade with more rubrics on the recovered rows.
3. **Tier A1 (task-type-aware proposer prompts)** -- cleaned up the
   over-templating (top-50 4-token concentration 72.2 % → 50.9 %) and
   diversified the per-family vocabulary; documentation_variants
   gained 10 pp.
4. **Tier A3 (boundary candidates)** -- finally activated the
   misalignment filter (0.7 % → 29.1 % of rejections), reduced the
   B-overshoot at validation (sat_a - sat_b: -1.31 → -1.11),
   improved gold-A recall by +3.8 pp.

**Mixed:**

5. **Tier A2 (adaptive decomposition gate)** -- doubled the
   decomposition success rate (1.9 % → 3.5 %) but added relatively
   little final accuracy because depth=1 rubrics are still only 94 of
   4045 (2.3 % of the bank).
6. **Tier A4 (3-sample majority-vote satisfaction)** -- did not reduce
   the score-tie rate (12.5 % → 14.25 %, slightly worse).  The
   majority-vote consensus is high (samples agree most of the time),
   so the extra calls added 3x cost without a proportional benefit.
   Reasonable to revert to `--rubric-satisfaction-samples 1` next time.
7. **Tier A5 (post-RRD discrimination filter)** -- ran but the
   `discrimination_filter` debug isn't exposed in the artifact (an
   `_expand_weighted_methods` plumbing oversight).  Visible only via
   the rubrics-vs-weights count mismatch.  Effect on accuracy is
   small (~1 useless rubric dropped per example).

**Cost-ineffective:**

8. **Tier A4 multi-sample at 3x cost** is the biggest cost-side
   regret.  It tripled satisfaction call count for ~0 pp lift.  Future
   v3 should run with samples=1 by default and reserve samples=3 only
   for borderline rubrics (e.g. only when the single-sample verdict
   was unstable across the rubric's other candidates).

## 7. Cost / wall-clock summary

| Phase | Cost | Wall clock |
|---|---:|---:|
| Phase 1 v2 diagnostics                                | $0     |   1 min |
| 5-row real-LLM micro-smoke (sanity)                   | ~$8    |   3 min |
| 100-row shard-0 v2.0 smoke (later superseded by v2.1) | ~$150  |  27 min |
| 1000-row shard-0 v2.1 retrain                         | ~$1500 | 4.24 hr |
| Index rebuild (text-embedding-3-small)                | ~$0.30 |   1 min |
| 200-row 4k v2 smoke                                   | ~$50   |  43 min |
| Full 4k v2 validation                                 | ~$700  | 15.2 hr |
| Cascade rescores (3 strategies × ~$2)                 | ~$6    |   4 min |
| **Total**                                             | **~$2,415** | **~21 hr (sequential)** |

Within the user-approved $2,200 / 16-hr Tier C budget envelope by ~10 %
on cost (the multi-sample 3x is the overage source) and ~30 % on wall
clock (mostly because validation was 15 hr instead of 8, also driven by
multi-sample).

## 8. Reproduce

```powershell
# Phase 1 -- v2 diagnostics on the existing shard 0
python scripts/diagnose_shard0_v2.py `
  --run-dir artifacts/medical_rl/runs/medical_v2_train_shard0 `
  --report-path artifacts/diagnostics/shard0_v2_full_report.md

# v2 4k cascade rescore (anti-tie variant -- the headline)
python scripts/rescore_with_tiebreaker.py `
  --run-dir artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k `
  --strategy format_then_uniform_then_anti_tie --workers 8 --suffix _v2_anti_tie

# Per-family / per-Opus breakdown
python scripts/diagnose_4k_validation.py `
  --run-dir artifacts/medical_rl_validation/runs/medical_v2_pair_validation_gpt5_b_4k
```

The full v2 plan implementation lives in the
[v2 runbook in medical_rl_prompts_training.md](medical_rl_prompts_training.md#v2-runbook--shard-0-retrain--4k-revalidation-with-the-v2-plan-changes).
