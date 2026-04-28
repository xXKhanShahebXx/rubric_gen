# JudgeBench Blind 270 Winning Mechanism

## Goal
This note records the exact leak-safe mechanism selected only from blind out-of-fold evaluation on `data/judgebench_80_human.json`, then frozen once for a blind final eval on `data/judgebench_270_generated.json`.

## Selected Runs
- Strict train-only winner: `artifacts/compiled_judgebench_train_only_runs/jb_blind_oof_v7_pruned_profile`
- One-shot blind final eval: `artifacts/compiled_judgebench_final_eval_runs/jb_blind_270_v7_pruned_profile_final`
- Mechanism hash: `180b0fe6ec7caded865d0e80a6e5094089fb43ccc722cb86abf339e8d262014d`
- Frozen policy hash: `598fa651d4b6f0141ece8656337cbddb021ef5959a3138f8824cf6bf8e652549`
- Protocol mode: `generic_baseline`
- Held-out `reference_answer` access: `false`
- Retrieval on final eval: `off`

## What Won
The selected strict mechanism combined:

1. Fully blind train-only development:
   - fold-train/bootstrap did **not** use `reference_answer`
   - fold-dev / OOF scoring did **not** use `reference_answer`

2. Blind scoring profile `pruned_v1`:
   - generalize candidate-specific blind answer rows into consistency-style rows
   - prune broad / redundant rubric bundles before scoring

3. Blind budget profile `family_profile_v1` with `--max-pairs-per-example 6`:
   - keep family-specific mutation ordering and candidate budgets
   - spend more blind budget where it helped on `80` OOF, without turning retrieval on

In practice, the winner was the best tradeoff between rubric count control and family-specific blind contrast coverage. It beat the non-pruned baseline overall while reducing average rubric count from `35.26` to `16.50`.

## Strict OOF Leaderboard
Blind `80` OOF WU metrics for the completed strict candidates:

| Run | Overall | MMLU | Reasoning | Math | Code | Ties | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `jb_blind_oof_v7_pruned_profile` | **75.00** | **70.00** | 68.75 | **100.00** | **76.92** | 4 | selected strict winner |
| `jb_blind_oof_v7_baseline` | 73.75 | 67.50 | **87.50** | 90.91 | 61.54 | 0 | no pruning; very large rubric sets |
| `jb_blind_oof_v7_pruned` | 68.75 | **70.00** | 68.75 | 72.73 | 61.54 | 3 | pruning without profile budgets |
| `jb_blind_oof_v7_pruned_disc_profile` | 68.75 | 62.50 | 68.75 | 81.82 | 76.92 | 2 | blind discriminator hurt overall |

Why `jb_blind_oof_v7_pruned_profile` won:
- best overall OOF score: `75.0`
- lowest failure count among completed strict runs: `20`
- kept the rubric set compact (`16.5` avg rubrics) without collapsing code/math performance
- retrieval remained separate and was not needed to win the strict track

## Rejected And Secondary Experiments
These runs were explicitly not frozen for the blind `270` benchmark:

- `jb_blind_oof_v7_retrieval_profile`:
  retrieval-augmented secondary track, `65.0` OOF overall, so it was not competitive with the strict winner.

- `jb_blind_oof_v7_baseline_profile`:
  long-running baseline-profile ablation that was stopped at `56/80`; its partial OOF was `69.64`, so it was already out of contention.

- `jb_blind_oof_v8_pruned_profile`:
  post-selection extractor experiment; it was stopped at `67/80` with partial OOF `65.67`, making it mathematically unable to reach `75.0`.
  Its code changes were reverted before final eval so the frozen `270` run matched the actual `80` OOF winner.

## Final Blind 270 Result
The selected strict mechanism was frozen from `jb_blind_oof_v7_pruned_profile` and run exactly once on the held-out `270`:

| Run | Overall | MMLU | Reasoning | Math | Code | Ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jb_blind_270_v7_pruned_profile_final` | **66.67** | 62.28 | 59.76 | 80.00 | 82.76 | 5 |

Final eval contract:
- `reference_answer_access: false`
- `retrieval_profile: off`
- `write_detailed_outputs: false`
- one-shot freeze from the selected `80`-only strict winner

## Exact Commands
Strict train-only winner:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset data/judgebench_80_human.json \
  --protocol-mode generic_baseline \
  --train-split-name train_80 \
  --max-workers 8 \
  --max-pairs-per-example 6 \
  --run-name jb_blind_oof_v7_pruned_profile \
  --no-train-reference-answer \
  --blind-scoring-profile pruned_v1 \
  --blind-budget-profile family_profile_v1 \
  --retrieval-profile off
```

One-shot blind final eval:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner final-eval \
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_blind_oof_v7_pruned_profile \
  --validation-dataset data/judgebench_270_generated.json \
  --validation-split-name validation_270 \
  --run-name jb_blind_270_v7_pruned_profile_final \
  --max-workers 8
```

## Short Explanation
If you need a one-paragraph explanation:

> We selected the JudgeBench mechanism using only blind out-of-fold evaluation on the `80` training examples and never used the held-out `270` to choose the mechanism. The winning strict run combined blind rubric pruning/generalization (`pruned_v1`) with family/profile-specific blind mutation budgets (`family_profile_v1`) under `generic_baseline`, without retrieval. That frozen mechanism then scored `66.67` on the one-shot blind `270` final eval, with held-out `reference_answer` access still disabled.

## Main Artifacts
- `artifacts/compiled_judgebench_train_only_runs/jb_blind_oof_v7_pruned_profile/summaries/oof_summary.json`
- `artifacts/compiled_judgebench_train_only_runs/jb_blind_oof_v7_retrieval_profile/summaries/oof_summary.json`
- `artifacts/compiled_judgebench_final_eval_runs/jb_blind_270_v7_pruned_profile_final/summaries/summary.json`
- `artifacts/compiled_judgebench_final_eval_runs/jb_blind_270_v7_pruned_profile_final/validation_270/final/summaries/summary.json`
