# JudgeBench v2 Training Runs

This document captures the three training-phase commands that go with the v2 pipeline. The v2
stack is invoked by passing the new flags to the existing `judgebench_eval_runner train-only`
command; no new runner is needed.

The sequence is:

1. Build the rubric library (once, offline).
2. Run `train_240` OOF with the full v2 stack.
3. If it clears the v2 promotion gates (`python -m rubric_gen.compiled.v2_promotion_gate_cli`),
   run `train_320` OOF.
4. If the `train_320` run also clears, spend one `blind-350` final-eval shot.

Do not skip the gate check between phases. Every phase writes
`summaries/oof_summary.json` (or `summaries/summary.json`) and the promotion-gate CLI exits with
status 2 when any gate fails.

## 0. Rebuild the rubric library

Run once, then freeze `artifacts/rubric_library/v1/library.json`:

```bash
# Deterministic seed bootstrap (no API calls; always works)
python -m rubric_gen.compiled.rubric_library_runner seed

# Production build (requires API keys and optional HF datasets)
python -m rubric_gen.compiled.rubric_library_runner build \
  --manifest path/to/library_manifest.json \
  --merge-into-existing
```

## 1. Archived replay gate (no compute)

Run the deterministic archive replay on the current best 320 base to confirm the v2 stack would
actually help before paying for a train_240 OOF:

```bash
python -m rubric_gen.compiled.archived_replay \
  --run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29 \
  --rubric-library-path artifacts/rubric_library/v1/library.json \
  --library-top-k 6
```

Exit code `0` when the process verifier would resolve at least `--min-ties-resolved` (default
`20`). Currently with the deterministic extractor this number is smaller (4 ties + 5 total
reasoning failures); use the replay output to decide whether to extend the extractor with an
LLM-backed step or to proceed to train_240.

## 2. Train-240 v2 OOF

Use `train_240_strict` with all v2 flags. The `--judge-model openai:gpt-4o-2024-05-13` override
locks the judge to GPT-4o per the v2 design constraint.

```bash
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train240_validation350/train_240_strict.json \
  --official-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train240_validation350/official_train_240_validation_350.jsonl \
  --train-split-name train_240 \
  --fold-count 5 \
  --fold-shuffle-seed 29 \
  --protocol-mode generic_baseline \
  --blind-scoring-profile pruned_disc_v1 \
  --blind-budget-profile family_profile_v2 \
  --blind-guidance-profile off \
  --blind-wu-profile stable_v1 \
  --retrieval-profile library_v1_plus_family_v1 \
  --retrieval-top-k 2 \
  --blind-discriminator-family-mode "mmlu-pro=strict" \
  --blind-discriminator-family-mode "livebench-reasoning=strict" \
  --max-criteria 10 \
  --max-pairs-per-example 6 \
  --max-workers 8 \
  --judge-model openai:gpt-4o-2024-05-13 \
  --rubric-library-path artifacts/rubric_library/v1/library.json \
  --library-retrieval-top-k 6 \
  --enable-rrd-filters \
  --rrd-redundancy-threshold 0.9 \
  --self-consistency-n 5 \
  --self-consistency-temperature 0.7 \
  --v2-wide-discriminator-gate \
  --holistic-judge \
  --run-name jb_240_v2_full_seed29
```

Then check the promotion gates:

```bash
python -m rubric_gen.compiled.v2_promotion_gate_cli \
  --run-dir artifacts/compiled_judgebench_train_only_runs/jb_240_v2_full_seed29
```

The v2 gates for 240 are: `overall ≥ 86`, each of `mmlu-pro`/`reasoning`/`math ≥ 82`,
`tie_failure_rate ≤ 5%`, `exact_answer_failure_rate ≤ 6%`,
`reasoning_verifier_trigger_rate ≥ 10%`, and
`discriminator_order_disagreement_rate ≤ 5%`. Additionally, if the external slice JSONL files
are present, `min(slice_wu) ≥ 75`.

If any gate fails, do NOT promote to 320. Instead, iterate on the failing component (library,
process verifier thresholds, self-consistency N) and re-run 240 with a different `--run-name`.

## 3. Train-320 v2 OOF

Only if the 240 run cleared all gates. Uses identical flags but with the 320 split:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/train_320_strict.json \
  --official-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/official_train_320_validation_350.jsonl \
  --train-split-name train_320 \
  --fold-count 5 \
  --fold-shuffle-seed 29 \
  --protocol-mode generic_baseline \
  --write-train-fit \
  --blind-scoring-profile pruned_disc_v1 \
  --blind-budget-profile family_profile_v2 \
  --blind-guidance-profile off \
  --blind-wu-profile stable_v1 \
  --retrieval-profile library_v1_plus_family_v1 \
  --retrieval-top-k 2 \
  --blind-discriminator-family-mode "mmlu-pro=strict" \
  --blind-discriminator-family-mode "livebench-reasoning=strict" \
  --max-criteria 10 \
  --max-pairs-per-example 6 \
  --max-workers 8 \
  --judge-model openai:gpt-4o-2024-05-13 \
  --rubric-library-path artifacts/rubric_library/v1/library.json \
  --library-retrieval-top-k 6 \
  --enable-rrd-filters \
  --rrd-redundancy-threshold 0.9 \
  --self-consistency-n 5 \
  --self-consistency-temperature 0.7 \
  --v2-wide-discriminator-gate \
  --holistic-judge \
  --run-name jb_320_v2_full_seed29
```

```bash
python -m rubric_gen.compiled.v2_promotion_gate_cli \
  --run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29
```

## 4. Blind-350 final eval

Only if the 320 v2 run cleared all gates. Do not repeat on failure. This is the only compute
step that is NOT reversible in terms of spending a blind-350 evaluation.

```bash
python -m rubric_gen.compiled.judgebench_eval_runner final-eval \
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 \
  --validation-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/validation_350.json \
  --official-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/official_train_320_validation_350.jsonl \
  --validation-split-name validation_350 \
  --write-detailed-outputs \
  --max-workers 8 \
  --retrieval-profile library_v1_plus_family_v1 \
  --run-name jb_350_blind_v2_full_seed29
```

## Rollback

The current best baseline is `jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29`. It
remains unmodified. If any v2 phase fails, reset to that baseline and iterate v2 components on a
smaller smoke run before spending another 240 / 320 / 350 budget.
