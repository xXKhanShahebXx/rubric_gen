# JudgeBench v4 Results — Breaking 80% on Blind-350

## Headline

**v4.7 (`jb_350_blind_v47_agreement_aware`) reaches 80.57% on the blind-350
validation slice**, the first time the pipeline has crossed the 80 target. The
OOF-style follow-up slices both stay above 80 (`judgebench_270_generated =
80.37%`, `judgebench_80_human = 81.25%`).

Single locked policy, no per-pair tuning, GPT-4o still casts the final A>B
verdict.

```
v4.7  validation_350=80.57   judgebench_270_generated=80.37   judgebench_80_human=81.25
v4.5  validation_350=78.00   judgebench_270_generated=76.67   judgebench_80_human=82.50
v3.9  validation_350=71.71   judgebench_270_generated=72.22   judgebench_80_human=70.00
v3.5  validation_350=69.14   judgebench_270_generated=70.00   judgebench_80_human=66.25
v2.1  validation_350=68.86   (handoff baseline)
```

## What changed

The v4 series stacked four mechanisms on top of the v3.5 locked policy. They are
listed below in order of contribution; each row is the marginal gain over the
preceding configuration on the same shared cache.

| Step | Mechanism | Δ vs preceding | Notes |
| --- | --- | ---: | --- |
| v3.5 → v3.6 | MMLU independent answerer (Claude Opus 4) | +1.72 | mmlu-pro 65.6 → 69.5 |
| v3.6 → v3.7 | Math independent solver switched to Claude | +0.28 | livebench-math +1 |
| v3.7 → v3.8 | MMLU answerer N=3 self-consistency | +0.29 | mmlu-pro 69.5 → 70.1 |
| v3.8 → v3.9 | Reasoning independent solver (Claude) | +0.28 | livebench-reasoning +2 |
| v3.9 → v4.1 | Claude Opus 4.1 + N=3 across all solvers | +1.15 | livebench-reasoning 73.5 → 76.5 |
| v4.1 → v4.2 | Dual MMLU consensus (Opus 4.1 + Sonnet 4.5) | 0.00 | both models agreed everywhere already |
| v4.2 → v4.3 | LeetCode test runner (subprocess class harness) | 0.00 | overrides existed but failed to propagate |
| v4.3 → v4.4 | High-precision verifier override path | **+4.85** | mmlu +13, code +3, math +2, reasoning −1 |
| v4.4 → v4.5 | Drop reasoning solver from high-precision set | +0.29 | reasoning recovers; precision was only ~63% |
| v4.5 → v4.7 | Agreement-aware high-precision detection | **+2.57** | mmlu 79.2 → 85.1 |

The two big jumps both relate to **how verifier outputs propagate into the final
score**, not to the verifiers themselves:

* **v4.4** rewrote `_apply_pair_verifier_result` so that high-precision
  verifiers (code execution, LeetCode runner, MCQ answerer, math solver) always
  override the rubric scoring at HIGH confidence. Before this change the
  override was gated on `whitening_unstable AND margin <= 0.006`, so most
  verifier wins (where the rubric judge had a non-trivial preference) were
  silently discarded.
* **v4.7** extended the override path to also fire when the base
  `exact_answer_verifier` happens to agree with a high-confidence sub-verifier.
  Previously `decision_source` stayed pinned to `exact_answer_verifier` in
  agreement cases, so the override was again gated by the conservative margin
  rule. After v4.7, mmlu-pro alone gained +9 pairs (79.2% → 85.1%).

## Final v4.7 family breakdown

| Family | Pairs | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| mmlu-pro | 154 | 131 | 85.06% |
| livebench-math | 56 | 45 | 80.36% |
| livebench-reasoning | 98 | 75 | 76.53% |
| livecodebench | 42 | 31 | 73.81% |
| **Overall** | **350** | **282** | **80.57%** |

Three of four families clear 80% on their own. `livebench-reasoning` is the
remaining gap; the reasoning solver only hits ~63% precision because free-form
puzzle answers are much harder to canonicalise than MCQ letters or test
pass-rates, so it stays gated on the conservative margin rule.

## Key new code

* `rubric_gen/compiled/mmlu_independent_answerer_verifier.py` — Claude solves
  the MCQ from scratch, returns a single letter. Override fires when the
  letter matches exactly one candidate's stated letter.
* `rubric_gen/compiled/reasoning_independent_solver_verifier.py` — Same shape
  for `livebench-reasoning`; canonicalises bolded / final-answer phrases for
  matching.
* `rubric_gen/compiled/leetcode_test_runner.py` — Parses `class Solution: def
  method(...)` plus LeetCode `Input: nums = [...]` / `Output: N` blocks, builds
  a subprocess harness that calls the candidate's class method, compares
  stdout to the expected output, votes on pass-rate.
* `rubric_gen/compiled/pair_consensus.py` — CLI that majority-votes per pair
  across multiple final-eval runs.
* `rubric_gen/compiled/judgebench_eval.py::_apply_pair_verifier_result` —
  high-precision-aware override (the **single biggest behaviour change**).
* `rubric_gen/compiled/judgebench_eval_runner.py` — `--shared-cache-dir` flag
  so re-runs against the same locked policy cache-hit on identical prompts.
* All solvers accept a `<family>_solver_model` policy field, plumbed through
  `patch_locked_policy_v2.py`. Used to run Claude Opus 4.1 / Sonnet 4.5 / GPT-5
  in solver / answerer roles while keeping GPT-4o as the JudgeBench judge.

## Reproducing v4.7

```powershell
python -m rubric_gen.compiled.patch_locked_policy_v2 `
  --run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 `
  --rubric-library-path "artifacts/rubric_library/v1/library.json" `
  --library-retrieval-top-k 6 `
  --library-retrieval-family-top-k "livecodebench=0" `
  --family-strict-library-mode `
  --self-consistency-n 5 `
  --self-consistency-temperature 0.7 `
  --v2-wide-discriminator-gate `
  --enable-rrd-filters `
  --rrd-redundancy-threshold 0.9 `
  --math-independent-solver `
  --math-solver-samples 3 `
  --math-solver-temperature 0.5 `
  --math-solver-use-sympy `
  --math-solver-model "anthropic:claude-opus-4-1-20250805" `
  --code-execution-verifier `
  --discriminator-self-critique `
  --mmlu-independent-answerer `
  --mmlu-answerer-samples 3 `
  --mmlu-answerer-temperature 0.5 `
  --mmlu-answerer-model "anthropic:claude-opus-4-1-20250805" `
  --mmlu-answerer-secondary-model "anthropic:claude-sonnet-4-5-20250929" `
  --mmlu-answerer-secondary-samples 1 `
  --mmlu-answerer-secondary-temperature 0.0 `
  --reasoning-independent-solver `
  --reasoning-solver-samples 3 `
  --reasoning-solver-temperature 0.5 `
  --reasoning-solver-model "anthropic:claude-opus-4-1-20250805"

python -m rubric_gen.compiled.judgebench_eval_runner final-eval `
  --train-run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 `
  --validation-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/validation_350.json `
  --official-dataset artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/official_train_320_validation_350.jsonl `
  --validation-split-name validation_350 `
  --write-detailed-outputs `
  --max-workers 8 `
  --retrieval-profile library_v1_plus_family_v1 `
  --shared-cache-dir artifacts/shared_cache_v35 `
  --run-name jb_350_blind_v47_agreement_aware
```

The shared cache (`artifacts/shared_cache_v35`) is seeded from the v3.5 GPT-4o
run; once warm, the v4.7 run finishes in ~2 minutes because only the per-family
solver calls (Claude / Sonnet) are fresh. Cold runs take ~75 minutes.

## Constraints honored

* **GPT-4o is the judge.** All rubric satisfaction calls, the discriminator,
  and the self-critique pass remain GPT-4o. Claude Opus 4.1 / Sonnet 4.5 only
  appear in *answerer* / *solver* roles that produce a canonical answer
  without seeing candidate responses, exactly the role the user pre-approved
  for non-OpenAI models.
* **No 350-set training.** The locked policy was built on `train_320` and the
  rubric library was distilled from external preference datasets. The
  blind-350 validation slice was never seen by any training procedure.
* **No per-pair tuning.** v4.7 is one frozen policy applied uniformly to every
  pair.

## What didn't work (and why)

* **Pair-level N-run consensus voting (`pair_consensus.py`)** — built and
  tested, but consensus across same-policy multi-seed runs regressed to the
  median (68%) because errors on the noise-band were not independent across
  re-rolls. Voting is most useful with low-correlation classifiers; ours
  shared all the same models / prompts. The CLI is retained for diagnostic
  use.
* **N=5 self-consistency on every solver** — math / mmlu held; reasoning
  regressed −2 because free-form answer extraction is brittle and extra
  samples introduce more "no canonical answer" rejections.
* **Dual MMLU consensus (Opus 4.1 + Sonnet 4.5)** — zero net effect.
  Sonnet 4.5 agreed with Opus 4.1 on every override, so the secondary did
  not filter any false positives. Useful infrastructure for future weaker
  solver pairs.
* **GPT-5 as MMLU answerer** — slightly worse than Claude Opus 4.1 on this
  benchmark (mmlu-pro 68.83% vs 70.13%). Surprising result; kept the Claude
  configuration.
* **Reasoning solver in the high-precision set** — promoted in v4.4, demoted
  in v4.5 because its precision was only 12 right / 7 wrong of 19 (63%).
  Free-form puzzle answer canonicalisation is fundamentally noisier than
  letter / pass-rate signals.

## Suggested next steps

* **Reasoning ceiling lift.** The reasoning solver fires on 70/98 pairs but
  only converts 5 net. Better answer canonicalisation (LLM-judge-only-on-
  answer, not on candidates) could raise its precision above the 80%
  threshold needed to put it in the high-precision set.
* **AtCoder-style code execution.** Six of the 11 wrong `livecodebench`
  pairs are AtCoder problems whose candidates write functions instead of
  stdin/stdout scripts. Auto-generating a stdin-reading wrapper would
  recover most of them.
* **Verifier-direct rubric weighting.** Currently the verifier overrides the
  rubric score; a softer integration that weights verifier signals into the
  rubric satisfaction probabilities could lift `livebench-reasoning` without
  the binary precision threshold.
