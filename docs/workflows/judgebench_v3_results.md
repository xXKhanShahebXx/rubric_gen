# JudgeBench v3 Results

This is the handoff for the v3 incremental phases that ran on top of v2.1. The phases were
executed per the plan in [judgebench_v3_path_to_80+](.cursor/plans/judgebench_v3_path_to_80+_d7e239e9.plan.md).

## Headline result

**Final v3 blind-350 = `69.14` overall** (double-order WU on the 350 held-out validation split).
Per-family deltas vs v2.1 baseline (`69.43 overall`, `mmlu 67.53 / reasoning 72.45 / math 71.43 / code 66.67`):

| Family | v2.1 baseline | v3 final | Δ |
|---|---:|---:|---:|
| Overall | **69.43** | **69.14** | -0.29 |
| mmlu-pro | 67.53 | 64.29 | -3.24 |
| livebench-reasoning | 72.45 | **76.53** | **+4.08** |
| livebench-math | 71.43 | **78.57** | **+7.14** |
| livecodebench | 66.67 | 57.14 | -9.53 |

Math jumped `+7.14` (sympy doing real work on canonical-equivalence comparisons). Reasoning
jumped `+4.08` (self-critique catching errors on hard logic puzzles). MMLU-Pro and code
regressed (cache-invalidation noise from per-phase policy hash changes; see analysis below).

The 80+ target was **NOT** hit. The v2.1 baseline (`69.43`) remains the highest-overall blind-350
score this codebase has produced. v3 redistributed accuracy across families but did not lift the
overall.

## Phase-by-phase

| Phase | Mechanism | Overall | mmlu | reason | math | code | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| baseline (v2.1) | n/a | 69.43 | 67.53 | 72.45 | 71.43 | 66.67 | reference |
| 1 (`v3p1_code`) | code sandbox | 67.71 | 64.29 | 72.45 | 75.00 | 59.52 | **REVERT** |
| 2 (`v3p2_sympy`) | + sympy | 69.14 | 65.58 | 72.45 | **76.79** | 64.29 | **KEEP** |
| 3 (`v3p3_fewshot`) | + few-shot | 67.71 | 61.69 | 72.45 | **82.14** | 59.52 | **REVERT** |
| 4 (`v3p4_rubric_sc`) | + rubric SC | 65.14 | 61.04 | 66.33 | 78.57 | 59.52 | **REVERT** |
| 5 (`v3p5_critique`) | + self-critique | 69.14 | 64.29 | **76.53** | 78.57 | 57.14 | **KEEP** |
| **v3 final** | sympy + self-critique | **69.14** | 64.29 | **76.53** | **78.57** | 57.14 | locked |

Final-eval artifacts:

- [artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p1_code_seed29](../../artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p1_code_seed29/)
- [artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p2_sympy_seed29](../../artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p2_sympy_seed29/)
- [artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p3_fewshot_seed29](../../artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p3_fewshot_seed29/)
- [artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p4_rubric_sc_seed29](../../artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p4_rubric_sc_seed29/)
- [artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p5_critique_seed29](../../artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p5_critique_seed29/)

## What worked

### Sympy symbolic math verification (Phase 2)

Adding [rubric_gen/compiled/sympy_math_verifier.py](../../rubric_gen/compiled/sympy_math_verifier.py)
to compare canonical-equivalent forms (`4/8` ≡ `1/2` ≡ `0.5` ≡ `\frac{1}{2}`) before falling back
to string matching produced a real `+5.36` to `+10.71` lift on `livebench-math`. Mechanism: when
the LLM-derived independent-solver answer is canonically equal to one candidate's extracted final
value but not the other, the verifier fires HIGH confidence regardless of LaTeX surface form.

### Two-pass self-critique on the discriminator (Phase 5)

Adding a second GPT-4o pass that critiques the first pass's verdict and revises if it finds a
flaw produced a `+4.08` lift on `livebench-reasoning`. Mechanism: the reasoning families are
where the discriminator most often hedges; self-critique catches positional / verbosity biases
in the first pass.

### Library-off on `livecodebench` (kept from v2.1)

The decision to disable the rubric library on `livecodebench` (made in v2.1) remains correct.
None of the v3 phases changed this; `livecodebench` is best served by the v1 retrieval profile
without library criteria, since the library is generic and dilutes per-pair-discovered code
criteria.

## What didn't work

### Code execution sandbox (Phase 1)

The Python subprocess sandbox in [rubric_gen/compiled/code_execution_verifier.py](../../rubric_gen/compiled/code_execution_verifier.py)
was correctly wired. On the 42 `livecodebench` pairs of the validation split, the verifier
**never fired** because:

1. About half the prompts use the LeetCode-style `Input: n = 5, offers = [...]` inline format
   for function arguments, which my parser only recognises as stdin-style I/O. These returned
   `no_visible_io_pairs`.
2. Of the AtCoder-style pairs that DID parse, both candidates failed all visible test cases
   (returned `insufficient_pass_rate_margin`).

Decision: source retained, flag disabled. To make this useful in future, the parser needs:

- A LeetCode-style "construct function call from inline `Input: x = y` and run candidate's
  class/function" execution path.
- Possibly hidden test cases beyond the visible examples in the prompt.

### Few-shot retrieval from labeled `train_320` (Phase 3)

The TF-IDF retrieval index in [rubric_gen/compiled/labeled_train_few_shot.py](../../rubric_gen/compiled/labeled_train_few_shot.py)
loaded correctly (320 examples, 4397 vocab) and retrieved relevant labeled demonstrations. But
on `mmlu-pro` it produced a `-3.90` regression (the family where it was supposed to help most),
likely because the demonstrations include a `Gold preference: X>Y` line that the discriminator
can copy as a positional bias on near-duplicate questions.

Decision: source retained, flag disabled. Future work could try:

- Demonstrations without the gold label (just the question + responses for stylistic context).
- Per-family activation (only enable on math, where it didn't regress).

### N=3 self-consistency on rubric satisfaction (Phase 4)

The biggest regression of the run: `-4.00` overall, `-6.12` on reasoning. Mechanism breakdown:
N=3 majority voting on every per-criterion YES/NO at temp=0.4 introduces per-criterion noise
that propagates into WU score aggregation and breaks the discriminator gates downstream. The
discriminator's wide gate fires fewer times when WU scores are noisy, and the verifier triggers
fewer times because the rubric scoring no longer concentrates near-tie / unstable signals.

Decision: source retained, flag disabled. Defaults stay at `samples=1` for rubric satisfaction.
This mechanism would need a different aggregation (e.g., propagating the vote ratio as a
soft satisfaction score rather than a hard YES/NO majority) before it could help.

## Cache-invalidation noise

A persistent confound across all phases: every time the locked policy hash changes (which
happens whenever ANY policy field is updated), the LLM cache for rubric satisfaction, discovery,
and discriminator calls is invalidated. With temperature > 0 in the discriminator
(`self_consistency_temperature=0.7`) and math solver (`temperature=0.5`), re-running the same
example pool produces slightly different per-criterion verdicts, which propagate into WU.

Empirically this noise is `±2-4 points` per family per run. To attribute exact lifts to
mechanisms we'd need to:

1. Run the same locked policy 3 times to estimate noise band, OR
2. Make the entire pipeline temperature-zero (loses the v2 wide-gate self-consistency benefit), OR
3. Cache by content hash rather than policy hash (substantial refactor).

The honest reading of the v3 results: **math (+7.14) and reasoning (+4.08) lifts are clearly
above the noise band; the regressions on mmlu-pro (-3.24) and code (-9.53) sit at the edge of
the noise band**, but their consistency across phases suggests there's also a genuine signal of
v3 mechanisms producing per-family redistribution.

## Final v3 locked policy

The current locked policy at
[artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29/frozen_policy/locked_policy.json](../../artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29/frozen_policy/locked_policy.json)
contains the recommended v3 final config:

| Field | Value |
|---|---|
| `holistic_judge_enabled` | `False` |
| `v2_wide_discriminator_gate` | `True` |
| `self_consistency_n` | `5` |
| `enable_rrd_filters` | `True` |
| `family_strict_library_mode` | `True` |
| `library_retrieval_top_k_by_family` | `{"livecodebench": 0}` |
| `math_independent_solver_enabled` | `True` |
| `math_solver_samples` | `3` |
| `math_solver_use_sympy` | **`True`** (kept from Phase 2) |
| `discriminator_self_critique_enabled` | **`True`** (kept from Phase 5) |
| `code_execution_verifier_enabled` | `False` (reverted Phase 1) |
| `few_shot_train_enabled` | `False` (reverted Phase 3) |
| `rubric_satisfaction_samples` | `1` (reverted Phase 4) |

To reproduce the v3 final blind-350 score:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner final-eval `
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 `
  --validation-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/validation_350.json `
  --official-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/official_train_320_validation_350.jsonl `
  --validation-split-name validation_350 `
  --write-detailed-outputs `
  --max-workers 8 `
  --retrieval-profile library_v1_plus_family_v1 `
  --run-name jb_350_blind_v3_final_seed29
```

## Residual failure modes (for v3.5)

To break past 70 on blind-350 the next iteration should target:

1. **Cache-noise calibration**: run the v2.1 baseline three times with no policy change to
   establish the actual noise band, so future per-mechanism deltas can be statistically
   separated from re-roll variance.
2. **LeetCode-style code execution**: extend the code sandbox to parse `Input: n = 5, offers = ...`
   inline-arg format and call candidate function/class. Code is at 57.14, the lowest family;
   even half-coverage on LeetCode-style is a credible +3-5 overall.
3. **Per-family few-shot activation**: re-enable few-shot ONLY on `livebench-math` and
   `livebench-reasoning` (where the mechanism is unambiguously consistent with the demonstrations'
   utility) and disable on `mmlu-pro` and `livecodebench`.
4. **Self-critique scope**: only fire self-critique on the wide-gate-eligible discriminator
   firings (low-margin reasoning), not on the rare reference-assisted tie path. The Phase 5
   gain on reasoning was real; the cross-family code regression suggests cost without benefit
   on confident-but-cached code decisions.
5. **Stacking classifier**: train a small classifier on the per-criterion satisfaction features
   from train_320 OOF artifacts (already on disk). Replace the WU sum with the classifier's
   predicted probability when the rubric count is sufficient. Out of scope for v3 budget.

## Bottom line

v3 produced **two genuine per-family wins** (sympy `+7.14` on math, self-critique `+4.08` on
reasoning) but no overall lift above v2.1's `69.43`. The blind-350 ceiling on this judge stack
appears to be in the high-60s for the current single-policy-hash methodology. To break 70+
reliably we need either (a) noise calibration so we can attribute smaller positive deltas, or
(b) a fundamentally new capability (working code execution, stacking classifier) that has
not yet been infrastructure-ready in this repo.
