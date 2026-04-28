# JudgeBench Train-Only Protocol

## Purpose
This workflow keeps JudgeBench mechanism development leakage-safe:

- use only the training `80` for mechanism iteration, routing bootstrap, and failure analysis
- keep `data/judgebench_270_generated.json` untouched until the mechanism is frozen
- evaluate the `270` once with aggregate-only outputs by default

The default inputs for this protocol are:

- `data/judgebench_80_human.json`
- `data/judgebench_270_generated.json`

## Protocol Modes
The JudgeBench runner now supports two protocol modes:

- `generic_baseline`
  - closest to the requested “v0” behavior
  - keeps generic routing bootstrap, recursive discovery, and uniform / whitened-uniform aggregation
  - disables JudgeBench-specific prompt nudges, family recursion overrides, exact-answer shortcuts, weight adjustments, whitening overrides, pair discriminator tie-breaks, and policy refinement heuristics
- `judgebench_tuned`
  - preserves the benchmark-specific tuned behavior for legacy comparisons

## Train-Only Development
Run balanced inner-fold development on the `80` without supplying a validation path:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset data/judgebench_80_human.json \
  --protocol-mode generic_baseline \
  --train-split-name train_80 \
  --fold-count 4 \
  --max-workers 8 \
  --run-name jb_train_only_v0
```

By default, `train-only` now keeps fold-dev / OOF scoring blind while still allowing the fold-train bootstrap side
to use the `80` references. To make the train side blind as well, pass:

```bash
--no-train-reference-answer
```

To intentionally restore reference-visible fold-dev scoring for diagnostics only, pass:

```bash
--allow-dev-reference-answer
```

Default train-only artifacts live under:

- `artifacts/compiled_judgebench_train_only_runs/<run_name>/dataset/`
- `artifacts/compiled_judgebench_train_only_runs/<run_name>/folds/`
- `artifacts/compiled_judgebench_train_only_runs/<run_name>/frozen_policy/locked_policy.json`
- `artifacts/compiled_judgebench_train_only_runs/<run_name>/summaries/oof_summary.json`
- `artifacts/compiled_judgebench_train_only_runs/<run_name>/summaries/oof_failure_analysis.json`

### What happens in train-only mode
1. Join the local `80` to the official JudgeBench pairwise file.
2. Build balanced folds by source family.
3. For each fold:
   - bootstrap a frozen policy on fold-train only, with train-side reference access controlled independently
   - score fold-dev only, blind by default
   - parallelize example processing inside each fold split
4. Aggregate out-of-fold metrics across the full `80`.
5. Build failure analysis from out-of-fold dev predictions only.
6. Freeze a locked policy on all `80`.

Fold runs use the total `--max-workers` budget as aggressively as is safe. Example processing is always parallelized;
when fold-level parallelism is enabled, the worker budget is divided across folds so the frozen-policy contexts do not
cross-contaminate held-out scoring.

## Failure Analysis Scope
The train-only workflow writes failure analysis only from out-of-fold dev predictions on the `80`.

Current summaries include:

- family-level miss clusters
- routing-profile failure clusters
- tie failures
- broad-rubric failure counts
- rubric-count buckets
- weak-candidate / mutation coverage counts overall and on failed examples
- train-side vs OOF-side reference-access bookkeeping

This is the allowed place to inspect misses and improve the mechanism.

## Final Held-Out Evaluation
After the mechanism is frozen, run the `270` once from the locked train-only policy:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner final-eval \
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_train_only_v0 \
  --validation-dataset data/judgebench_270_generated.json \
  --validation-split-name validation_270 \
  --max-workers 8 \
  --run-name jb_final_eval_v0
```

By default this command:

- uses the locked policy and mechanism hash from the train-only run
- resolves the official JudgeBench pairwise file from the locked train run or downloads the canonical file
- hides `reference_answer` during held-out evaluation and builds blind contrasts from the pair responses plus synthetic degradations
- writes aggregate held-out summaries only
- does not emit per-example held-out artifacts or prediction reports

To intentionally write detailed held-out outputs, pass:

```bash
--write-detailed-outputs
```

To intentionally restore the older reference-assisted behavior for diagnostics only, pass:

```bash
--allow-reference-answer
```

That mode is not a blind JudgeBench evaluation and should not be used for benchmark-faithful reporting.

That should not be used in the normal tuning loop.

## Leakage Guardrails
- Do not run `final-eval` until the mechanism is frozen.
- Do not inspect per-example held-out failures during development.
- Do not compare multiple candidate mechanisms on the `270`.
- Do not route `data/judgebench_270_generated.json` through the train-only command.
- Use the train-only run summary and locked policy hash as the source of truth for what was frozen.
