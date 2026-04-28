# Compiled Rubric System

## Purpose
This document defines the `compiled rubric system` for medical dialogue-to-note quality evaluation.

The current repository already contains an `RRD` pipeline for rubric discovery and note evaluation. This
spec adds a second, more structured architecture for cases where the evaluator must behave like a
`rubric compiler + validator` rather than a single generic judge prompt.

The design is optimized for:

- clinical faithfulness over generic writing quality
- high-signal evaluation from a small `100 + 500` pilot regime
- reusable artifacts that can later support SFT curation and reward modeling
- explicit handling of catastrophic failures, ambiguity, and note-family differences

## Design Goals
- Separate `what quality means` from `how a rubric is instantiated`.
- Separate `catastrophic correctness failures` from `soft quality trade-offs`.
- Allow one evaluator family to support multiple note schemas without collapsing them into one noisy rubric.
- Keep the system auditable by preserving evidence anchors, rationales, and human adjudications.
- Make each artifact versionable so ontology changes, policy changes, and scoring changes do not get mixed together.

## Non-Goals
- This spec does not define the final compiler implementation.
- This spec does not require RL to be used immediately.
- This spec does not replace the current RRD pipeline; it adds a sibling system that can reuse RRD-inspired
  discovery methods while enforcing stronger structure.

## System Layers
The compiled rubric stack has four layers.

### Layer 0: Documentation Contract
This layer defines the task policy for a note family.

It captures:

- note family or schema name
- allowed level of inference
- section requirements
- section ordering rules when relevant
- stylistic constraints that actually matter
- rules for representing uncertainty
- family-specific safety expectations

### Layer 1: Universal Clinical Note Checks
These checks apply to all note families.

Examples:

- no hallucinated medications, diagnoses, allergies, or measurements
- no contradictions with the dialogue
- no unsupported diagnostic certainty
- no negation or laterality flips
- clear distinction between observed facts and inferred conclusions

### Layer 2: Note-Family Template
This layer contains reusable bundles for a note family.

Examples:

- SOAP-like notes
- HPI-centered notes
- assessment and plan style notes
- specialty-specific documentation styles

This layer may collapse into a single bundle if the dataset is homogeneous.

### Layer 3: Case-Instantiated Atomic Checklist
This layer is compiled from the concrete dialogue, reference note, note-family spec, and rubric ontology.

Examples:

- `mentions chest pain duration`
- `documents denial of fever`
- `does not add smoking history if absent`
- `places medication changes in the plan section`

## Core Artifacts
The system is intentionally split into independent versioned artifacts.

### `rubric_ontology_vX`
Defines:

- quality dimensions
- subdimensions
- criterion templates
- severity tiers
- error taxonomy

This is the main source of truth for what evaluation concepts exist.

### `note_family_spec_vX`
Defines:

- note-family schema
- section contract
- inference policy
- family-specific constraints
- template subsets allowed for that family

### `rubric_compiler_vX`
Defines:

- how ontology templates are instantiated into case-specific checks
- when a criterion is skipped
- how evidence anchors are attached
- which checks become hard gates versus soft checks

### `judge_bundle_vX`
Defines:

- criterion prompt format
- verdict vocabulary
- scoring conventions
- calibration examples
- ensemble or retry policy if any

### `adjudication_log_vX`
Defines:

- open ambiguities
- resolved policy questions
- rationale for the resolution
- which artifacts must change as a result

This artifact must stay separate from the ontology so policy updates remain auditable.

## Scoring Architecture
The evaluator must use two stages instead of a single undifferentiated weighted sum.

### Stage 1: Hard Gates
Hard gates exist for catastrophic failures that should block an example from high-quality SFT selection.

Typical hard-gate failures:

- hallucinated core facts
- contradiction with the dialogue
- unsupported diagnosis or treatment claims
- wrong negation
- wrong laterality
- serious attribution mistakes

### Stage 2: Soft Score
Only gate-passing examples receive a soft score.

Soft scoring should operate on dimensions such as:

- completeness
- chronology
- section fidelity
- organization
- readability
- uncertainty calibration

Soft scores should be normalized by dimension and then aggregated with severity-aware weights.

## Severity Tiers
The default severity tiers are:

- `catastrophic`
- `essential`
- `important`
- `optional`

Recommended behavior:

- `catastrophic`: a failed criterion is a hard gate
- `essential`: heavily weighted and often required for strong SFT examples
- `important`: strong preference, usually not exclusionary on its own
- `optional`: useful but should not dominate selection

## Verdict Vocabulary
The default criterion verdict vocabulary is:

- `MET`
- `UNMET`
- `NOT_APPLICABLE`
- `CANNOT_ASSESS`

Additional labels may be used for analysis, but these four should remain the common interface across
compiled rubrics.

## Evidence Anchors
Every instantiated case criterion should carry `evidence anchors` whenever possible.

Evidence anchors should identify the support for the criterion from one of:

- dialogue text
- reference note text
- note-family specification
- rubric ontology
- adjudication policy

The goal is to keep each criterion grounded and reduce judge hallucination during scoring.

## Error Taxonomy
The system should preserve a reusable error taxonomy because small pilots need more than naturally
occurring failures.

Suggested top-level error families:

- omission
- hallucination
- contradiction
- negation flip
- laterality error
- chronology distortion
- section misplacement
- unsupported inference
- certainty inflation
- attribution error

This taxonomy should be shared across:

- mutation design
- rubric discovery
- adjudication
- reporting

## Compiler Contract
The rubric compiler consumes:

- dialogue
- reference note
- note-family spec
- rubric ontology
- optional adjudication overrides

The compiler emits:

- hard-gate criteria
- soft-score criteria
- evidence anchors
- applicability decisions
- version metadata linking the case rubric back to its source artifacts
- optional per-criterion `eval_kind` / `judge_hints` for heuristic or programmatic judges (starter scaffold in `rubric_gen.compiled`)

## Recommended Data Products Per Example
Each scored example should produce a structured record with:

- `case_id`
- `note_family`
- `rubric_version`
- `hard_gate_results`
- `dimension_scores`
- `criterion_results`
- `evidence_anchors`
- `rationales`
- `overall_decision`

This record should be reusable for:

- SFT curation
- rewrite-target generation
- failure analysis
- later reward experiments

## Small-Data Rollout
The intended rollout for this spec is:

- `100` examples for design and validation
- `500` examples for pilot transfer and SFT-quality curation

Recommended split:

- `60` design examples for failure discovery and local rubric induction
- `20` adjudication examples for policy questions and ambiguity resolution
- `20` locked validation examples
- `400` pilot working examples
- `100` pilot generalization examples

## Promotion Gates
Do not treat the system as pilot-ready until all of the following are true:

- hard-gate criteria reliably catch synthetic mutation failures
- repeated scoring gives stable verdicts on the same examples
- the locked validation slice improves in discrimination, not only fit to the design slice
- the pilot generalization slice transfers acceptably beyond the initial `100`

## Relationship To Existing RRD Code
This system should reuse strong ideas from the existing RRD-inspired work in the repository:

- recursive decomposition for broad criteria
- redundancy pruning
- correlation-aware aggregation
- contrastive discovery using multiple candidate notes

However, compiled rubrics add stronger structure than a flat rubric bank by introducing:

- note-family-aware templates
- hard-gate versus soft-score separation
- evidence anchors
- versioned adjudication

## Machine-Readable Schemas
The canonical JSON schemas that back this spec live under:

- `docs/spec/schemas/rubric_ontology.schema.json`
- `docs/spec/schemas/note_family_spec.schema.json`
- `docs/spec/schemas/case_rubric.schema.json`
- `docs/spec/schemas/case_evaluation_record.schema.json`
- `docs/spec/schemas/adjudication_record.schema.json`
