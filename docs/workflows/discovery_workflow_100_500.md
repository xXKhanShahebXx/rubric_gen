# Discovery Workflow For The 100/500 Pilot

## Purpose
This runbook turns the compiled rubric design into an executable workflow for a `100`-example design set
and a `500`-example pilot set.

The immediate goal is:

- discover high-value failure modes
- compile a reliable rubric system
- validate transfer on a broader pilot
- produce a curated SFT subset

The immediate goal is not:

- full-scale RL
- broad model benchmarking across many note families before the evaluator stabilizes

## Dataset Split
### 100-Sample Design Set
Use only for rubric design and validation.

- `60` discovery examples
- `20` adjudication examples
- `20` locked validation examples

### 500-Sample Pilot Set
Use only after freezing a design-set rubric version.

- `400` working pilot examples
- `100` locked pilot generalization examples

## Output Artifacts
This workflow should produce:

- `rubric_ontology_v0_x`
- `note_family_spec_v0_x`
- `case_rubric` instances for design and pilot examples
- `case_evaluation_record` outputs
- `adjudication_log_v0_x`
- a `gold SFT subset`
- a `repair subset`
- a `do_not_train subset`

## Phase 1: Determine Note-Family Boundaries
Before generating rubrics, determine whether the dataset needs one note family or several.

Questions to answer:

- Do the notes share the same section structure?
- Do they allow the same level of inference?
- Are the same omissions and safety failures equally serious across all examples?

Decision rule:

- If the answers are mostly yes, use one note family initially.
- If not, split the dataset into note-family bundles before rubric induction.

Do not force one rubric family across materially different note styles.

## Phase 2: Build Discovery Contrast Sets
The discovery slice should not only contain the reference note.

For each of the `60` discovery examples, build a contrast set containing:

- the reference note
- `2-4` model-generated notes from strong or varied prompts
- `1-2` intentionally incomplete or terse notes
- `1-2` controlled note mutations

This produces a local comparison set rich enough to induce discriminative criteria.

## Phase 3: Mutation Harness
Because the design set is small, synthetic negatives are required.

Each mutation should target one dominant failure mode.

Recommended mutation types:

- omit a clinically important fact
- flip a negation
- hallucinate a medication, diagnosis, allergy, or measurement
- distort chronology or duration
- move content into the wrong section
- turn uncertainty into certainty
- add unsupported plan language
- merge clinician and patient statements incorrectly

Mutation rules:

- keep one mutation focused on one failure mode whenever possible
- preserve fluency so the evaluator must detect semantic failure, not obvious corruption
- label each mutation with an error family from the shared taxonomy

## Phase 4: Local Criterion Induction
Use each contrast set to induce local atomic criteria.

Procedure:

1. Compare the reference note against weak and mutated notes.
2. Ask what specific property makes one note clinically better.
3. Draft an atomic criterion capturing that property.
4. Verify whether the criterion actually separates the notes in that case.
5. Refine or discard the criterion if it is too vague or too broad.

Keep only criteria that are:

- atomic
- self-contained
- evidence-seeking
- mostly binary
- single-failure-mode focused

## Phase 5: Recursive Decomposition
When a local criterion is too broad, decompose it.

Examples:

- `captures relevant history` may decompose into symptom duration, relevant negatives, medication context,
  and chronology.
- `is clinically faithful` may decompose into no hallucination, no contradiction, and no unsupported inference.

Use decomposition when:

- the criterion is satisfied by most notes in the contrast set
- different failure modes are being conflated
- the judge cannot apply the criterion consistently

## Phase 6: Compress Into A Reusable Ontology
After local induction, compress the resulting pool into:

- reusable dimensions
- subdimensions
- criterion templates
- error taxonomy entries

Compression rules:

- merge near-duplicate criteria
- keep one canonical template for each meaningfully distinct failure mode
- separate catastrophic checks from soft-quality checks
- preserve note-family-specific variants only when they are truly family dependent

This is the point where `RubricOntology` and `NoteFamilySpec` should be frozen for the first validation pass.

## Phase 7: Compile Case Rubrics
For each case in the design and pilot sets, compile a `CaseRubric`.

The compiler should:

- instantiate only criteria supported by the dialogue or policy
- attach evidence anchors whenever possible
- split checks into `hard_gates` and `soft_checks`
- mark irrelevant criteria as not applicable instead of forcing a verdict

Avoid compiling every template into every case rubric. Over-instantiation will make the evaluator noisy.

## Phase 8: Judge Calibration
Before trusting scores, calibrate the judge bundle.

Calibration loop:

1. score the same examples multiple times
2. inspect unstable criteria
3. clarify wording or add examples for unstable criteria
4. rerun the same slice

Only add ensemble judging when a criterion remains unstable after wording and evidence-anchor improvements.

## Phase 9: Adjudication Loop
Use the `20` adjudication examples to resolve ambiguous policy questions.

Only escalate cases where:

- the same criterion gives unstable verdicts
- strong human reviewers disagree on policy
- the ontology or note-family spec does not specify the answer clearly
- the hard-gate decision changes depending on interpretation

Resolved adjudications should update:

- the ontology
- the note-family spec
- the judge bundle wording

Prefer updating the smallest artifact that fixes the ambiguity.

## Phase 10: Locked Validation On The Design Set
After each major ontology or judge-bundle revision, re-run the locked `20` validation examples.

Promotion check:

- did hard-gate detection improve on mutation cases?
- did repeated scoring become more stable?
- did the evaluator separate stronger notes from weaker notes better than before?

Do not accept a revision that only memorizes the `60` discovery examples.

## Phase 11: Freeze And Roll Out To The 500 Pilot
Once the design-set rubric system is stable, freeze:

- ontology version
- note-family spec version
- judge bundle version

Then run the `400` working pilot examples and hold back the locked `100`.

Pilot analysis should include:

- note-family slices
- dimension-level score distributions
- hard-gate trigger rates
- most common error families
- disagreement rates by criterion

## Phase 12: Build Training Subsets
Classify pilot outputs into three buckets.

### `gold SFT subset`
Criteria:

- passes all hard gates
- exceeds the soft-score threshold
- no unresolved ambiguity

### `repair subset`
Criteria:

- passes hard gates
- below target soft-score threshold
- useful for rewrite or targeted improvement

### `do_not_train subset`
Criteria:

- fails hard gates
- highly ambiguous
- inconsistent or unsafe reference behavior

## Promotion Gates
Promote the system from design to pilot-ready only if all are true:

- mutation failures are consistently caught by the intended hard gates
- repeated scoring is stable enough for the same inputs
- the locked design validation slice improves rather than regresses
- the locked pilot generalization slice shows acceptable transfer

## What To Log During Every Pass
For every scoring pass, retain:

- rubric version
- judge bundle version
- criterion-level verdicts
- rationales
- evidence anchors
- hard-gate counts
- dimension scores
- final training bucket

This logging is required if the pilot is later used to support SFT filtering or RL reward design.

## Recommended Review Rhythm
Use a narrow loop rather than a giant batch loop.

- After every `10-15` discovery examples, inspect whether new criteria are still being found.
- After every major ontology revision, rerun the locked validation slice.
- After every adjudication batch, update only the necessary artifact and re-evaluate.

This keeps the evaluator grounded and prevents silent drift.
