# JudgeBench 350 Recovery Handoff

## Status At Handoff
This note is the handoff for the JudgeBench `350` recovery / architecture-reset work as of `2026-04-24`.

Current state:

- The blind-350 architecture-reset plan was implemented.
- The follow-up `320` OOF-first sweep program was completed.
- No `320` candidate cleared the spend gates for another blind `350` shot.
- A targeted `livebench-reasoning` rescue patch was implemented and tested on `train_240`.
- That `train_240` patch run did **not** improve enough to justify promotion to a new `320` run.
- There is **no current spendable candidate** for a new blind `350` evaluation.

If someone picks this up next, the current best base to continue from is still:

- `artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29`

That is the strongest all-around `320` OOF run even though it is still below the blind spend gates.

## Goal And Constraints
User goal:

- get blind `350` validation into the `80s`
- do not spend blind `350` shots casually
- be willing to make architecture changes if needed

Working assumptions and constraints used in this recovery:

- compute was not the limiting factor
- the development loop should remain leak-safe
- `320` OOF-first sweeps are the main pre-spend filter
- blind `350` should only be run if the new candidate clearly clears stronger family-floor and failure-rate gates

Practical spend gate used during this recovery:

- overall WU `>= 86`
- `mmlu-pro` WU `>= 82`
- `livebench-reasoning` WU `>= 82`
- exact-answer failure rate `<= 7%`
- tie failure rate `<= 5%`

## Read These First
Existing docs worth reading before changing anything:

- `docs/workflows/judgebench_train_only_protocol.md`
- `docs/workflows/judgebench_blind_270_winning_mechanism.md`

Those two docs explain the train-only / freeze / blind-eval contract and the earlier blind-270 mechanism that originally looked promising before the blind-350 generalization gap became the real problem.

## Important Data Splits And Artifact Roots
Key datasets used in the recent recovery loop:

- `artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/train_320_strict.json`
- `artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/official_train_320_validation_350.jsonl`
- `artifacts/generated_judgebench_splits/strict_disjoint_v1_train240_validation350/train_240_strict.json`
- `artifacts/generated_judgebench_splits/strict_disjoint_v1_train240_validation350/official_train_240_validation_350.jsonl`

Main artifact roots:

- `artifacts/compiled_judgebench_train_only_runs/`
- `artifacts/compiled_judgebench_final_eval_runs/`

For each train-only run, the useful files are:

- `.../summaries/oof_summary.json`
- `.../summaries/oof_failure_analysis.json`
- `.../folds/fold_*/dev/examples/*.json`
- `.../train_fit/examples/*.json` when `--write-train-fit` was enabled
- `.../frozen_policy/locked_policy.json`

## Code Changes Completed
The blind-350 architecture-reset plan was completed across the following files.

### `rubric_gen/compiled/judgebench_eval.py`
This is the main file that changed.

Implemented:

- blind-parity behavior so prompts can hide references without losing internal answer-key / task-typing features
- private answer-key preservation for blind scoring
- deterministic pair verifier integration
- calibrated pair discriminator routing with neutral labels and explicit order swaps
- additional calibration metrics written into summaries
- latest targeted reasoning rescue patch:
  - `_reasoning_small_margin_discriminator_candidate()`
  - widened discriminator eligibility for non-exact-answer, code-like `livebench-reasoning` cases with small margins

### `rubric_gen/compiled/judgebench_eval_runner.py`
Updated CLI semantics so train-side reference access defaults are aligned with blind parity:

- `--allow-train-reference-answer` is now the explicit opt-in flag
- train-only development defaults to blind-parity bootstrap instead of reference-visible bootstrap

### `rubric_gen/compiled/judgebench_selection_audit.py`
Extended to track and gate on new calibration metrics and transport reliability features.

Important caveat:

- this tool currently does **not** work cleanly on the current `oof_summary.json` schema written by recent OOF runs
- it failed on `compiled_judgebench_split_summary_v1`
- manual audit was used instead

### `rubric_gen/compiled/discovery.py`
Updated pair discriminator prompts to use:

- neutral candidate IDs (`X`, `Y`)
- order-swapped evaluation
- reduced presentation bias

### `rubric_gen/compiled/judgebench_verifiers.py`
New module added for deterministic verifier logic:

- exact-answer checks
- option / value consistency
- format checks
- marker conflict detection
- family-specific confidence scoring

### Tests Added / Updated

- `tests/test_judgebench_eval.py`
- `tests/test_judgebench_selection_audit.py`
- `tests/test_judgebench_verifiers.py`

At the end of this handoff, the recent `judgebench_eval` test module was passing:

- `44 passed`

## What The Architecture Reset Was Trying To Fix
The core blind-350 problem was not raw train-only fit. The main failure mode was transport:

- train / OOF mechanisms looked better than blind `350`
- reasoning and exact-answer families were fragile
- too many ties and exact-answer misses survived selection
- the verifier and discriminator layers were not sufficiently active on the failure slices that mattered

The reset addressed four main areas:

1. blind-parity core
2. verifier layer
3. calibrated uncertainty routing
4. audit calibration / gates

All four of those plan items were completed.

## Post-Reset `320` OOF Sweep Leaderboard
These were the key `320` OOF-first runs after the architecture reset.

| Run | Overall | MMLU | Reasoning | Math | Code | Failure Rate | Tie Fails | Exact Fails | Main Takeaway |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29` | **85.625** | **82.5** | 77.5 | **83.75** | 98.75 | 0.14375 | 31 | 24 | best all-around base; still fails reasoning and failure-rate gates |
| `jb_320_archreset_blindparity_seedv1_strictdisc_seed29` | 84.375 | 76.25 | **85.0** | 78.75 | 97.5 | 0.15625 | 25 | 35 | best reasoning donor; hurts `mmlu-pro` too much |
| `jb_320_archreset_blindparity_qv1_reasonseed_defaultdisc_seed29` | 83.125 | 78.75 | 76.25 | 77.5 | **100.0** | 0.16875 | 32 | 32 | hybrid did not beat best base |
| `jb_320_archreset_blindparity_qv1_reasonseed_strictdisc_seed29` | 81.875 | 80.0 | 76.25 | 73.75 | 97.5 | 0.18125 | 37 | 34 | weaker than defaultdisc hybrid |
| `jb_320_archreset_blindparity_seedv1_defaultdisc_seed29` | 81.5625 | 75.0 | 77.5 | 76.25 | 97.5 | 0.184375 | 30 | 37 | seed retrieval without strict disc was not enough |
| `jb_320_archreset_blindparity_focusk1_defaultdisc_seed29` | 81.25 | 80.0 | 80.0 | 67.5 | 97.5 | 0.1875 | 37 | 30 | too weak on math / overall |
| `jb_320_archreset_blindparity_focusk1_strictdisc_seed29` | 80.9375 | 78.75 | **85.0** | 63.75 | 96.25 | 0.190625 | 38 | 35 | reasoning went up, math collapsed |

### Key Conclusion From The `320` Sweeps
No run cleared the spend gates.

The best run was:

- `jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29`

Why it was still not spendable:

- `livebench-reasoning` only `77.5`
- tie failures `31 / 320` (`9.69%`)
- exact-answer failures `24 / 320` (`7.5%`)

That run came closest because it held:

- strong `mmlu-pro`
- strong `livebench-math`
- strong `livecodebench`
- the best overall WU

But it still missed the blind-350 floor on reasoning and reliability.

## Best-Base Calibration Details
Calibration details for the strongest `320` base:

- run: `jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29`
- discriminator usage rate: `0.028125`
- discriminator order disagreement rate: `0.0`
- `livebench-reasoning` verifier trigger rate: `0.0`
- low-confidence bucket accuracy: `0.850163`

This matters because it implies:

- the discriminator itself is not obviously unstable
- the verifier is basically not helping on reasoning
- the reasoning slice still has transport headroom because the resolution layers are underused

The strongest reasoning donor run was:

- `jb_320_archreset_blindparity_seedv1_strictdisc_seed29`

Useful details:

- reasoning `85.0`
- overall `84.375`
- `mmlu-pro` only `76.25`
- `livebench-reasoning` verifier trigger rate still `0.0`

So the donor improved reasoning mostly through retrieval / rubric behavior, not verifier activation.

## What Worked
These are the things that genuinely helped or were worth keeping.

### 1. Blind-Parity Architecture Reset
This was worth doing and should be kept.

Why:

- it made train-only development more honest
- it decoupled prompt blindness from internal answer-key features
- it created a cleaner base for future transport fixes

### 2. `320` OOF-First Gating Before Blind `350`
This was the correct operating discipline and saved blind-350 shots.

Why:

- multiple attractive candidates still failed clear spend gates
- if we had trusted them early, we would likely have spent more blind `350` runs on non-transportable mechanisms

### 3. `family_question_v1` Retrieval As The Best All-Around Base
Among the post-reset `320` runs, the best base was still:

- `retrieval_v1 + strictdisc`

Why it worked:

- strongest overall score
- preserved `mmlu-pro` and math better than seed-style retrieval
- gave the cleanest base to patch rather than replacing the whole mechanism

### 4. `family_question_seed_v1` As A Reasoning Donor
Seed-style retrieval was useful as a donor even though it was not the best full policy.

Why:

- it consistently pushed `livebench-reasoning` upward
- it revealed that reasoning gains were possible, just not with acceptable collateral damage to `mmlu-pro`

### 5. Neutral-Label, Order-Swapped Pair Discriminator
This change looks worth keeping.

Evidence:

- order disagreement stayed at `0.0` on the checked `320` and `240` runs
- when the discriminator fired on the reasoning slice, it often looked directionally useful

## What Did Not Work
These are the things that either failed outright or should be treated cautiously.

### 1. The Current `judgebench_selection_audit.py` Flow On New OOF Summaries
This is currently broken for the current OOF output shape.

Observed failure:

- it expected a different summary schema
- it failed on `compiled_judgebench_split_summary_v1`

Practical consequence:

- candidate comparison had to be done manually from `oof_summary.json` and `oof_failure_analysis.json`

### 2. `focusk1` Variants
They were not competitive enough.

Pattern:

- reasoning sometimes improved
- math degraded too much
- overall score stayed too low

### 3. `qv1_reasonseed` Hybrids
They did not beat the strongest base.

Pattern:

- some localized improvements
- not enough overall lift
- still too many failure-rate issues

### 4. The Family-Wide Reasoning Small-Margin Route Widening
This was the latest targeted patch and it did **not** work as a promotable fix.

What was done:

- diffed `retrieval_v1_strictdisc` against `seedv1_strictdisc` on the `livebench-reasoning` slice
- noticed that many retrieval-base misses were no-signal ties or tiny-margin flips in `person_right_*` profiles
- widened blind discriminator routing for non-exact-answer, code-like reasoning tasks with small margins

Supporting observation before the run:

- archived replay suggested `18` extra reasoning examples would become discriminator-eligible
- `6` previously missed errors appeared to be covered by that widened route

Actual fast-harness result:

- `jb_240_reasoningroute_retrieval_v1_strictdisc_seed29`
- overall `79.58333333333333`
- `mmlu-pro` `78.33333333333333`
- `livebench-reasoning` `71.66666666666667`
- `livebench-math` `68.33333333333333`
- `livecodebench` `100.0`
- failure count `49`
- failure rate `0.20416666666666666`
- tie failures `31`
- exact-answer failures `29`
- discriminator usage rate `0.058333`
- `livebench-reasoning` verifier trigger rate still `0.0`

Conclusion:

- widening the route family-wide was too blunt
- it increased discriminator usage but did not transport into a better candidate
- do **not** promote this patch as-is

### 5. Trusting Terminal Footer Summaries
This caused confusion and should not be used as source of truth.

Observed inconsistency:

- the terminal footer printed `train_240_oof=82.50`
- the saved `oof_summary.json` on disk showed `79.58333333333333`

Actionable rule:

- always trust `oof_summary.json` and `oof_failure_analysis.json`
- never trust the terminal footer alone

## Operational Issues And Lessons

### Silent Background Runs
Some long-running train-only jobs produced almost no terminal output while still progressing.

Reliable monitoring methods:

- count `folds/fold_*/dev/examples/*.json`
- wait for `summaries/oof_summary.json`
- inspect final `oof_failure_analysis.json`

Do **not** assume a quiet terminal means a hung run.

### Stale / Misleading Process Information
Terminal metadata sometimes showed a shell still running long after the useful work was already done or after the child process was gone.

Practical rule:

- if the artifact directory is progressing, trust the artifacts over the shell metadata

### Summary Schema Drift
This is worth fixing because it slows iteration.

Recommended fix:

- make `judgebench_selection_audit.py` accept current `compiled_judgebench_split_summary_v1` OOF summaries directly

## Current Best Base To Continue From
Continue from:

- `artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29`

Why this is still the base:

- highest `320` OOF overall
- only run with both strong `mmlu-pro` and strong math
- better all-around transport than the reasoning-favoring donor runs

Use as donor / comparison:

- `artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_seedv1_strictdisc_seed29`

Why:

- reasoning `85.0`
- good source of reasoning-specific retrieval / profile patterns
- but not acceptable as a full replacement policy because `mmlu-pro` fell too much

## What The Failure Slices Are Saying Now
The current blockers are still:

1. `livebench-reasoning`
2. tie failures
3. exact-answer failures
4. more recently, `livebench-math` if the mechanism is perturbed too aggressively

For the best `320` base, the failure clusters still showed:

- reasoning cluster: `{'A=B': 9, 'A>B': 4, 'B>A': 5}`
- `mmlu-pro` cluster: `{'A=B': 10, 'A>B': 3, 'B>A': 1}`
- math cluster: `{'A=B': 11, 'B>A': 2}`

That pattern says:

- ties are a major part of the remaining error
- the reasoning problem is not only wrong-direction decisions; it is also unresolved / under-resolved cases
- math is relatively strong in the best base, so future patches should avoid disturbing it unless there is a clear payoff

## What Could Work Next
These are the next things most worth trying.

### 1. Revert To The Best `320` Base Before New Experiments
Do not treat the `train_240` reasoning-route patch as the new default.

Recommended base:

- `retrieval_v1_strictdisc_seed29`

### 2. Make Reasoning Fixes Profile-Specific, Not Family-Wide
The failed patch widened routing across non-exact-answer code-like reasoning too broadly.

Better direction:

- focus specifically on the `person_right_left_*` and `person_right_have_*` reasoning profiles
- patch only the profiles that dominated the retrieval-base misses

### 3. Make The Reasoning Verifier Actually Trigger
This is a major missing piece.

Current evidence:

- `verifier_trigger_rate_by_family['livebench-reasoning'] == 0.0` on the checked `320` leaders
- still `0.0` on the failed `240` rescue run

Better direction:

- add reasoning-state / assignment-completeness / contradiction signals
- reward completed consistent final mappings
- penalize `UNKNOWN`, missing slots, parity contradictions, unsupported final conclusions
- use these as precise verifier signals rather than just routing more cases to the discriminator

### 4. Use Archived Example Replay Before Spending A Full `240` Or `320`
The archived reasoning diff and replay step was helpful and should be reused.

Why:

- it is cheap
- it can test routing logic on the exact old failure IDs
- it avoids expensive runs for obviously bad routing ideas

### 5. Fix Tie / Exact-Answer Reliability On The Best Base
Even the best `320` run was still blocked by tie and exact-answer rates.

Practical targets:

- reduce tie failures from `31` to `<= 16`
- reduce exact-answer failures from `24` to `<= 22`, ideally lower

Note:

- future work should try to reduce these without harming `livebench-math`

### 6. Keep `seedv1_strictdisc` As A Donor, Not As The New Base
Good use:

- compare reasoning-only failures against `retrieval_v1_strictdisc`
- lift retrieval / rubric / prompt ideas from it into the reasoning slice

Bad use:

- replacing the whole mechanism with `seedv1_strictdisc`

### 7. Patch The Audit Tool
This is not a modeling improvement, but it is worth doing because it will speed up future work:

- update `judgebench_selection_audit.py` to support current OOF summary schema directly
- then re-encode the stronger spend gates there

## Suggested Next Experimental Order
If starting fresh from this handoff, the best order is:

1. use `jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29` as the base
2. diff its reasoning failures against `jb_320_archreset_blindparity_seedv1_strictdisc_seed29`
3. patch only the dominant reasoning profiles, not the whole family
4. add reasoning-verifier signals that can actually trigger
5. replay the new routing / verifier logic on archived failure examples first
6. if replay still looks good, run one new `train_240` OOF harness
7. only if `train_240` looks materially better, run one new `320` OOF candidate
8. do **not** spend blind `350` unless the new `320` candidate clears the gates

## Runs And Artifacts To Keep Handy
Best all-around `320` base:

- `artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29/summaries/oof_summary.json`
- `artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29/summaries/oof_failure_analysis.json`

Best reasoning donor:

- `artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_seedv1_strictdisc_seed29/summaries/oof_summary.json`
- `artifacts/compiled_judgebench_train_only_runs/jb_320_archreset_blindparity_seedv1_strictdisc_seed29/summaries/oof_failure_analysis.json`

Latest failed reasoning rescue:

- `artifacts/compiled_judgebench_train_only_runs/jb_240_reasoningroute_retrieval_v1_strictdisc_seed29/summaries/oof_summary.json`
- `artifacts/compiled_judgebench_train_only_runs/jb_240_reasoningroute_retrieval_v1_strictdisc_seed29/summaries/oof_failure_analysis.json`

## Commands Worth Reusing
Best `320` base command:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset "artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/train_320_strict.json" \
  --official-dataset "artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/official_train_320_validation_350.jsonl" \
  --train-split-name train_320 \
  --fold-count 5 \
  --fold-shuffle-seed 29 \
  --protocol-mode generic_baseline \
  --write-train-fit \
  --blind-scoring-profile pruned_disc_v1 \
  --blind-budget-profile family_profile_v2 \
  --blind-guidance-profile off \
  --blind-wu-profile stable_v1 \
  --retrieval-profile family_question_v1 \
  --retrieval-top-k 2 \
  --blind-discriminator-family-mode "mmlu-pro=strict" \
  --blind-discriminator-family-mode "livebench-reasoning=strict" \
  --max-criteria 8 \
  --max-pairs-per-example 6 \
  --max-workers 8 \
  --run-name jb_320_archreset_blindparity_retrieval_v1_strictdisc_seed29
```

Reasoning donor command:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset "artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/train_320_strict.json" \
  --official-dataset "artifacts/generated_judgebench_splits/strict_disjoint_v1_train320_validation350/official_train_320_validation_350.jsonl" \
  --train-split-name train_320 \
  --fold-count 5 \
  --fold-shuffle-seed 29 \
  --protocol-mode generic_baseline \
  --write-train-fit \
  --blind-scoring-profile pruned_disc_v1 \
  --blind-budget-profile family_profile_v2 \
  --blind-guidance-profile off \
  --blind-wu-profile stable_v1 \
  --retrieval-profile family_question_seed_v1 \
  --retrieval-top-k 2 \
  --blind-discriminator-family-mode "mmlu-pro=strict" \
  --blind-discriminator-family-mode "livebench-reasoning=strict" \
  --max-criteria 8 \
  --max-pairs-per-example 6 \
  --max-workers 8 \
  --run-name jb_320_archreset_blindparity_seedv1_strictdisc_seed29
```

Latest failed reasoning-route harness command:

```bash
python -m rubric_gen.compiled.judgebench_eval_runner train-only \
  --train-dataset "artifacts/generated_judgebench_splits/strict_disjoint_v1_train240_validation350/train_240_strict.json" \
  --official-dataset "artifacts/generated_judgebench_splits/strict_disjoint_v1_train240_validation350/official_train_240_validation_350.jsonl" \
  --train-split-name train_240 \
  --fold-count 5 \
  --fold-shuffle-seed 29 \
  --protocol-mode generic_baseline \
  --blind-scoring-profile pruned_disc_v1 \
  --blind-budget-profile family_profile_v2 \
  --blind-guidance-profile off \
  --blind-wu-profile stable_v1 \
  --retrieval-profile family_question_v1 \
  --retrieval-top-k 2 \
  --retrieval-family-profile "livebench-reasoning=family_question_seed_v1" \
  --retrieval-family-top-k "livebench-reasoning=2" \
  --blind-discriminator-family-mode "mmlu-pro=strict" \
  --blind-discriminator-family-mode "livebench-reasoning=strict" \
  --max-criteria 8 \
  --max-pairs-per-example 6 \
  --max-workers 8 \
  --run-name jb_240_reasoningroute_retrieval_v1_strictdisc_seed29
```

## Bottom Line
The work so far established a better blind-parity architecture and identified the real current state:

- the best current base is `retrieval_v1_strictdisc_seed29`
- the most useful donor is `seedv1_strictdisc_seed29`
- the remaining bottleneck is still `livebench-reasoning` plus tie / exact-answer reliability
- the quick reasoning-route widening patch did not hold up on `train_240`

The next person should not start from scratch. They should start from the best `320` base, patch the reasoning slice more surgically, and avoid spending blind `350` until a new `320` OOF run actually clears the gates.
