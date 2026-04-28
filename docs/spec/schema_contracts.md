# Compiled Rubric Schema Contracts

## Purpose
This document defines the concrete artifact contracts for the compiled rubric system in a way that is
stable enough to implement against.

The five core artifacts are:

1. `RubricOntology`
2. `NoteFamilySpec`
3. `CaseRubric`
4. `CaseEvaluationRecord`
5. `AdjudicationRecord`

## Conventions
### Versioning
- Every top-level artifact must carry a `version`.
- When possible, include stable IDs such as `ontology_id`, `note_family_id`, or `rubric_id`.
- Derived artifacts should record the source artifact versions they were compiled from.

### Identifiers
- IDs should be short, stable, and machine-friendly.
- Prefer lowercase snake case for enums and IDs.
- Keep human-readable labels separate from IDs.

### Common Enums
Severity tiers:

- `catastrophic`
- `essential`
- `important`
- `optional`

Verdict types:

- `binary`
- `ordinal`
- `nominal`

Criterion verdicts:

- `MET`
- `UNMET`
- `NOT_APPLICABLE`
- `CANNOT_ASSESS`

Overall decisions:

- `sft_include`
- `repair`
- `do_not_train`
- `needs_review`

## Artifact 1: `RubricOntology`
### Role
Defines the reusable quality ontology for clinical documentation.

### Required fields
- `ontology_id`
- `version`
- `severity_tiers`
- `dimensions`
- `criterion_templates`
- `error_taxonomy`

### Field contract
- `ontology_id`: stable identifier for the ontology bundle.
- `version`: ontology version string.
- `severity_tiers`: ordered list of supported severity tiers.
- `dimensions`: top-level quality dimensions such as `dialogue_faithfulness` or `clinical_completeness`.
- `criterion_templates`: reusable criterion definitions that can later be compiled into case-specific checks.
- `error_taxonomy`: reusable error labels shared by mutation design, scoring, and reporting.

### Suggested shape
```json
{
  "ontology_id": "clinical_note_core",
  "version": "v0_1",
  "severity_tiers": [
    "catastrophic",
    "essential",
    "important",
    "optional"
  ],
  "dimensions": [
    {
      "dimension_id": "dialogue_faithfulness",
      "label": "Dialogue Faithfulness",
      "description": "The note must remain grounded in the dialogue.",
      "hard_gate_eligible": true,
      "subdimensions": [
        {
          "subdimension_id": "hallucination",
          "label": "Hallucination"
        },
        {
          "subdimension_id": "contradiction",
          "label": "Contradiction"
        }
      ]
    }
  ],
  "criterion_templates": [
    {
      "template_id": "no_unsupported_diagnosis",
      "dimension_id": "dialogue_faithfulness",
      "subdimension_id": "hallucination",
      "label": "No Unsupported Diagnosis Claims",
      "description": "Do not state a diagnosis unless supported by the dialogue or policy.",
      "severity_tier": "catastrophic",
      "default_verdict_type": "binary",
      "evidence_policy": "dialogue_or_reference",
      "hard_gate_default": true,
      "typical_failure_codes": [
        "unsupported_inference",
        "certainty_inflation"
      ]
    }
  ],
  "error_taxonomy": [
    {
      "error_code": "unsupported_inference",
      "label": "Unsupported Inference",
      "description": "The note adds a conclusion not grounded in the evidence.",
      "severity_tier": "catastrophic"
    }
  ]
}
```

## Artifact 2: `NoteFamilySpec`
### Role
Defines a note-family contract that sits between the global ontology and case-specific compiled rubrics.

### Required fields
- `note_family_id`
- `version`
- `label`
- `documentation_contract`
- `section_specs`

### Field contract
- `note_family_id`: stable note-family identifier.
- `version`: version string for the family contract.
- `label`: human-readable family name.
- `documentation_contract`: task policy for inference, uncertainty, and required sections.
- `section_specs`: explicit section definitions for the note family.
- `family_template_ids`: optional subset of ontology templates allowed or emphasized for this family.
- `hard_gate_error_codes`: error codes that always count as hard failures for this family.

### Suggested shape
```json
{
  "note_family_id": "soap_note",
  "version": "v0_1",
  "label": "SOAP Note",
  "documentation_contract": {
    "inference_policy": "strict_grounded",
    "uncertainty_policy": "explicit_only",
    "required_section_ids": [
      "subjective",
      "objective",
      "assessment",
      "plan"
    ],
    "optional_section_ids": [
      "medications"
    ],
    "style_rules": [
      "Prefer concise clinical prose.",
      "Use section headers."
    ]
  },
  "section_specs": [
    {
      "section_id": "subjective",
      "label": "Subjective",
      "required": true,
      "allowed_content": [
        "history",
        "symptoms",
        "patient_reported_context"
      ]
    }
  ],
  "family_template_ids": [
    "no_unsupported_diagnosis"
  ],
  "hard_gate_error_codes": [
    "unsupported_inference",
    "negation_flip"
  ]
}
```

## Artifact 3: `CaseRubric`
### Role
A compiled, case-specific rubric generated from the dialogue, reference note, note-family spec, and ontology.

### Required fields
- `rubric_id`
- `version`
- `example_id`
- `note_family_id`
- `ontology_version`
- `note_family_version`
- `hard_gates`
- `soft_checks`
- `aggregation`

### Field contract
- `hard_gates`: catastrophic criteria that block SFT inclusion when failed.
- `soft_checks`: remaining case-specific analytic criteria.
- `aggregation`: soft-score aggregation policy.
- `evidence_anchors`: support snippets or policy references attached to each criterion.

### Optional fields on compiled criteria (starter scaffold)
Implementations may attach lightweight machine metadata to help a non-LLM judge route checks:

- `eval_kind`: short string naming the evaluation strategy (for example `section_header_coverage`, `symptom_detail_bundle`).
- `judge_hints`: a small JSON-compatible bag of parameters (regex lists, keyword lists, section ids) produced at compile time from the case.

These fields are additive; validators should ignore unknown keys.

### Evidence anchor contract
Each evidence anchor should contain:

- `source`
- `quote`

Optional fields:

- `speaker`
- `section_id`
- `start_char`
- `end_char`

### Suggested shape
```json
{
  "rubric_id": "case_001_rubric_v0_1",
  "version": "v0_1",
  "example_id": "case_001",
  "note_family_id": "soap_note",
  "ontology_version": "v0_1",
  "note_family_version": "v0_1",
  "hard_gates": [
    {
      "criterion_id": "case_001_no_unsupported_diagnosis",
      "template_id": "no_unsupported_diagnosis",
      "dimension_id": "dialogue_faithfulness",
      "subdimension_id": "hallucination",
      "label": "No Unsupported Diagnosis Claims",
      "requirement": "Do not diagnose pneumonia unless the dialogue supports it.",
      "severity_tier": "catastrophic",
      "verdict_type": "binary",
      "evidence_anchors": [
        {
          "source": "dialogue",
          "quote": "Patient reports cough but no diagnosis was confirmed."
        }
      ],
      "compile_rationale": "This case includes symptoms that could tempt unsupported diagnostic certainty."
    }
  ],
  "soft_checks": [
    {
      "criterion_id": "case_001_mentions_cough_duration",
      "dimension_id": "clinical_completeness",
      "label": "Mentions Cough Duration",
      "requirement": "Document how long the cough has been present.",
      "severity_tier": "essential",
      "verdict_type": "binary",
      "evidence_anchors": [
        {
          "source": "dialogue",
          "quote": "I've had this cough for three days."
        }
      ]
    }
  ],
  "aggregation": {
    "hard_gate_policy": "all_must_pass",
    "minimum_soft_score": 0.85,
    "soft_dimension_weights": {
      "clinical_completeness": 0.4,
      "section_fidelity": 0.2,
      "uncertainty_calibration": 0.2,
      "readability": 0.2
    }
  }
}
```

## Artifact 4: `CaseEvaluationRecord`
### Role
Stores the judge output for one candidate note evaluated against one compiled case rubric.

### Required fields
- `evaluation_id`
- `rubric_id`
- `example_id`
- `candidate_id`
- `note_family_id`
- `rubric_version`
- `hard_gate_results`
- `soft_results`
- `dimension_scores`
- `overall_decision`

### Field contract
- `hard_gate_results`: verdicts for catastrophic criteria.
- `soft_results`: verdicts for remaining analytic criteria.
- `dimension_scores`: normalized dimension-level scores.
- `overall_decision`: whether the evaluated note should be included for SFT, routed to repair, blocked, or reviewed.
- `judge_metadata`: optional evaluator provenance such as model, prompt version, and run mode.

### Criterion result contract
Each criterion result should contain:

- `criterion_id`
- `verdict`
- `rationale`

Optional fields:

- `score_value`
- `confidence`
- `error_codes`
- `evidence_used`

### Suggested shape
```json
{
  "evaluation_id": "case_001_candidate_a_eval_v0_1",
  "rubric_id": "case_001_rubric_v0_1",
  "example_id": "case_001",
  "candidate_id": "candidate_a",
  "note_family_id": "soap_note",
  "rubric_version": "v0_1",
  "hard_gate_results": [
    {
      "criterion_id": "case_001_no_unsupported_diagnosis",
      "verdict": "MET",
      "score_value": 1.0,
      "rationale": "The note summarizes symptoms but does not assert an unsupported diagnosis."
    }
  ],
  "soft_results": [
    {
      "criterion_id": "case_001_mentions_cough_duration",
      "verdict": "UNMET",
      "score_value": 0.0,
      "rationale": "The note omits the three-day duration of cough."
    }
  ],
  "dimension_scores": [
    {
      "dimension_id": "clinical_completeness",
      "earned_score": 0.0,
      "max_score": 1.0,
      "normalized_score": 0.0,
      "criterion_count": 1
    }
  ],
  "overall_decision": "repair",
  "judge_metadata": {
    "mode": "pointwise_analytic",
    "prompt_version": "judge_bundle_v0_1"
  }
}
```

## Artifact 5: `AdjudicationRecord`
### Role
Stores a human resolution for an ambiguity that the ontology, note-family spec, or judge bundle cannot
resolve reliably on its own.

### Required fields
- `adjudication_id`
- `version`
- `status`
- `ambiguity_type`
- `question`

### Conditionally required for resolved records
- `resolution`
- `rationale`
- `decided_by`
- `decided_at`

### Field contract
- `status`: one of `open`, `resolved`, or `deprecated`.
- `ambiguity_type`: the policy class of the ambiguity.
- `question`: the clinician-facing question.
- `resolution`: the decision that should be applied.
- `affects`: which artifact(s) should change because of the decision.
- `evidence_anchors`: dialogue or note excerpts that motivated the question.

### Suggested ambiguity types
- `inference_policy`
- `required_negative`
- `section_placement`
- `plan_support`
- `compression`
- `chronology`
- `other`

### Suggested shape
```json
{
  "adjudication_id": "adj_case_001_inference_v0_1",
  "version": "v0_1",
  "status": "resolved",
  "ambiguity_type": "inference_policy",
  "example_id": "case_001",
  "note_family_id": "soap_note",
  "question": "May the note state community-acquired pneumonia when only cough and fever are discussed but no diagnosis is confirmed?",
  "resolution": "No. The note may describe symptoms and differential context but must not assert pneumonia as a confirmed diagnosis.",
  "rationale": "The dialogue does not confirm the diagnosis and the note policy is strict grounded documentation.",
  "affects": [
    {
      "artifact_type": "note_family_spec",
      "artifact_id": "soap_note",
      "change_type": "clarify"
    },
    {
      "artifact_type": "rubric_ontology",
      "artifact_id": "clinical_note_core",
      "change_type": "add_rule"
    }
  ],
  "evidence_anchors": [
    {
      "source": "dialogue",
      "quote": "They said they might do a chest x-ray, but nobody told me I have pneumonia."
    }
  ],
  "decided_by": "clinician_01",
  "decided_at": "2026-04-05T17:30:00Z"
}
```

## Recommended Artifact Relationships
- `RubricOntology` should be referenced by `CaseRubric`.
- `NoteFamilySpec` should be referenced by `CaseRubric`.
- `CaseRubric` should be referenced by `CaseEvaluationRecord`.
- `AdjudicationRecord` should be allowed to update the ontology, note-family spec, judge bundle, or a single
  case rubric.

## Minimum Viable Implementation Order
If these contracts are implemented incrementally, the recommended order is:

1. `RubricOntology`
2. `NoteFamilySpec`
3. `CaseRubric`
4. `CaseEvaluationRecord`
5. `AdjudicationRecord`
