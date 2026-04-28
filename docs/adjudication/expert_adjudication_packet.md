# Expert Adjudication Packet

## Purpose
This document defines how to present ambiguous medical dialogue-to-note cases to a clinician or domain
expert for resolution.

The goal is not to ask experts to re-score everything. The goal is to ask only the smallest number of
high-value questions required to improve the rubric system.

## When To Escalate A Case
Escalate a case only if at least one of the following is true:

- a hard-gate verdict changes depending on interpretation
- the same criterion is unstable across repeated judge runs
- two strong reviewers disagree on whether the note is acceptable
- the ontology or note-family spec does not specify the answer clearly
- a mutation exposes a policy boundary the current rubric cannot express

Do not escalate cases that are merely stylistic unless style affects safety, completeness, or usability in a
clinically meaningful way.

## Adjudication Packet Contents
Each packet should contain:

- `case_id`
- `note_family`
- `rubric_version`
- `criterion_ids_under_review`
- `ambiguity_type`
- `dialogue_excerpt`
- `reference_note_excerpt`
- `candidate_note_excerpt`
- `current_judge_outputs`
- `specific_question`
- `decision_options`
- `space_for_rationale`

## Packet Template
Use the following structure for each escalation.

### Header
- `Case ID`
- `Note family`
- `Rubric version`
- `Adjudication ID`
- `Owner`
- `Review deadline`

### Context
- `Why this case was escalated`
- `What rubric criterion or policy is unclear`
- `Whether the ambiguity affects a hard gate or only a soft score`

### Evidence
- `Dialogue excerpt`
- `Reference note excerpt`
- `Candidate note excerpt`
- `Relevant evidence anchors`

### Current system behavior
- `Current criterion verdict(s)`
- `Current rationale(s)`
- `Observed disagreement or instability`

### Decision question
The question should be specific, policy-shaped, and answerable.

Good:

- `May the note state diagnosis X as confirmed when the dialogue only supports suspicion or differential consideration?`
- `Is negative symptom Y mandatory when symptom Z is documented in this note family?`
- `Must medication dose changes appear in the plan section, or is any clinically clear placement acceptable?`

Bad:

- `Is this note good?`
- `Which note do you prefer?`
- `How would you rewrite this?`

### Decision options
Each packet should provide a small answer set.

Recommended option types:

- `allowed`
- `not_allowed`
- `allowed_only_if_condition`
- `not_required_but_preferred`
- `required_for_this_note_family`
- `insufficient_context`

### Required expert response
Ask the expert to provide:

- one selected decision
- a short rationale
- any condition or boundary that changes the answer
- whether the result should update the ontology, note-family spec, judge bundle, or only this case

## Suggested Ambiguity Categories
Use one of these categories unless a new one is clearly needed:

- `inference_policy`
- `required_negative`
- `section_placement`
- `plan_support`
- `chronology`
- `speaker_attribution`
- `compression`
- `severity_threshold`
- `other`

## Example Packet
### Header
- `Case ID`: `case_001`
- `Note family`: `soap_note`
- `Rubric version`: `v0_1`
- `Adjudication ID`: `adj_case_001_inference_v0_1`

### Context
- `Why escalated`: the evaluator disagrees on whether the note may assert pneumonia.
- `Impact`: hard-gate ambiguity on unsupported inference.

### Evidence
- `Dialogue excerpt`: `They said they might do a chest x-ray, but nobody told me I have pneumonia.`
- `Reference note excerpt`: `Assessment: cough, fever; rule out pneumonia.`
- `Candidate note excerpt`: `Assessment: community-acquired pneumonia.`

### Current system behavior
- `Criterion under review`: `no_unsupported_diagnosis`
- `Observed outputs`: one judge marked `UNMET`, one marked `MET` under a broader inference rule.

### Decision question
- `May the note state community-acquired pneumonia as a confirmed diagnosis in this case?`

### Decision options
- `not_allowed`
- `allowed_only_if_condition`
- `insufficient_context`

## How To Use Expert Answers
Translate the answer into the smallest possible system change.

If the answer clarifies a general rule:

- update `RubricOntology`

If the answer is specific to a note family:

- update `NoteFamilySpec`

If the answer indicates the judge prompt was too vague:

- update the `judge_bundle`

If the answer applies only to one unusual example:

- update only the `CaseRubric` or mark the example as `needs_review`

## Adjudication Writing Rules
- Prefer policy statements over free-form advice.
- Preserve the expert's boundary conditions.
- Convert subjective language into operational wording where possible.
- Record whether the resolution changes a hard gate.
- Record whether the resolution should affect future mutations and reporting.

## Batch Review Guidance
Do not send experts a random batch of cases.

Prioritize cases by:

1. hard-gate impact
2. judge instability
3. frequency of the ambiguity across examples
4. risk of contaminating SFT data if left unresolved

Batching guidance:

- `5-10` cases per review batch is usually enough
- avoid mixing many ambiguity types in one batch when possible
- group repeated policy questions together so one answer can update many cases

## Recommended Output Format
Each resolved packet should become an `AdjudicationRecord` with:

- `adjudication_id`
- `status`
- `ambiguity_type`
- `question`
- `resolution`
- `rationale`
- `affects`
- `decided_by`
- `decided_at`

## Promotion Rule
If an ambiguity repeatedly changes hard-gate outcomes, do not continue expanding the pilot until it has
been resolved and translated back into the rubric system.
