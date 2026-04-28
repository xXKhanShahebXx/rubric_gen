"""Dataclass models for compiled-rubric artifacts (see docs/spec/schema_contracts.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --- Shared / nested types ---


@dataclass
class EvidenceAnchor:
    source: str
    quote: str
    speaker: Optional[str] = None
    section_id: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


@dataclass
class SubdimensionSpec:
    subdimension_id: str
    label: str


@dataclass
class DimensionSpec:
    dimension_id: str
    label: str
    description: str
    hard_gate_eligible: bool
    subdimensions: List[SubdimensionSpec] = field(default_factory=list)


@dataclass
class CriterionTemplate:
    template_id: str
    dimension_id: str
    label: str
    description: str
    severity_tier: str
    default_verdict_type: str
    evidence_policy: str
    hard_gate_default: bool
    typical_failure_codes: List[str] = field(default_factory=list)
    subdimension_id: Optional[str] = None
    # Provisional templates merged from local discovery (starter closed-loop scaffold).
    provisional_discovered: bool = False
    # If set, only instantiate for these note_family_ids; None or empty = all families.
    note_family_scope: Optional[List[str]] = None
    task_family_scope: Optional[List[str]] = None
    task_profile_id: Optional[str] = None
    artifact_label: str = "artifact"


@dataclass
class ErrorTaxonomyEntry:
    error_code: str
    label: str
    description: str
    severity_tier: str


@dataclass
class RubricOntology:
    ontology_id: str
    version: str
    severity_tiers: List[str]
    dimensions: List[DimensionSpec]
    criterion_templates: List[CriterionTemplate]
    error_taxonomy: List[ErrorTaxonomyEntry]


@dataclass
class DocumentationContract:
    inference_policy: str
    uncertainty_policy: str
    required_section_ids: List[str]
    optional_section_ids: List[str] = field(default_factory=list)
    style_rules: List[str] = field(default_factory=list)


@dataclass
class SectionSpec:
    section_id: str
    label: str
    required: bool
    allowed_content: List[str] = field(default_factory=list)


@dataclass
class NoteFamilySpec:
    note_family_id: str
    version: str
    label: str
    documentation_contract: DocumentationContract
    section_specs: List[SectionSpec]
    family_template_ids: List[str] = field(default_factory=list)
    hard_gate_error_codes: List[str] = field(default_factory=list)
    task_profile_id: Optional[str] = None
    task_family_id: Optional[str] = None
    artifact_label: str = "note"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_family_id:
            self.task_family_id = self.note_family_id


@dataclass
class CompiledCriterion:
    criterion_id: str
    dimension_id: str
    label: str
    requirement: str
    severity_tier: str
    verdict_type: str
    evidence_anchors: List[EvidenceAnchor] = field(default_factory=list)
    template_id: Optional[str] = None
    subdimension_id: Optional[str] = None
    compile_rationale: Optional[str] = None
    # Starter scaffold: tells the heuristic judge how to evaluate this compiled row (non-LLM dispatch).
    eval_kind: Optional[str] = None
    judge_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationPolicy:
    hard_gate_policy: str
    minimum_soft_score: float
    soft_dimension_weights: Dict[str, float]


@dataclass
class CaseRubric:
    rubric_id: str
    version: str
    example_id: str
    note_family_id: str
    ontology_version: str
    note_family_version: str
    hard_gates: List[CompiledCriterion]
    soft_checks: List[CompiledCriterion]
    aggregation: AggregationPolicy
    task_profile_id: Optional[str] = None
    task_family_id: Optional[str] = None
    artifact_label: str = "note"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_family_id:
            self.task_family_id = self.note_family_id


@dataclass
class CriterionResult:
    criterion_id: str
    verdict: str
    rationale: str
    score_value: Optional[float] = None
    confidence: Optional[float] = None
    error_codes: List[str] = field(default_factory=list)
    evidence_used: List[EvidenceAnchor] = field(default_factory=list)


@dataclass
class DimensionScore:
    dimension_id: str
    earned_score: float
    max_score: float
    normalized_score: float
    criterion_count: int


@dataclass
class CaseEvaluationRecord:
    evaluation_id: str
    rubric_id: str
    example_id: str
    candidate_id: str
    note_family_id: str
    rubric_version: str
    hard_gate_results: List[CriterionResult]
    soft_results: List[CriterionResult]
    dimension_scores: List[DimensionScore]
    overall_decision: str
    judge_metadata: Dict[str, Any] = field(default_factory=dict)
    task_profile_id: Optional[str] = None
    task_family_id: Optional[str] = None
    artifact_label: str = "note"

    def __post_init__(self) -> None:
        if not self.task_family_id:
            self.task_family_id = self.note_family_id


@dataclass
class AffectEntry:
    artifact_type: str
    artifact_id: str
    change_type: str


@dataclass
class AdjudicationRecord:
    adjudication_id: str
    version: str
    status: str
    ambiguity_type: str
    question: str
    example_id: Optional[str] = None
    note_family_id: Optional[str] = None
    task_family_id: Optional[str] = None
    resolution: Optional[str] = None
    rationale: Optional[str] = None
    affects: List[AffectEntry] = field(default_factory=list)
    evidence_anchors: List[EvidenceAnchor] = field(default_factory=list)
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.task_family_id:
            self.task_family_id = self.note_family_id


TaskFamilySpec = NoteFamilySpec
