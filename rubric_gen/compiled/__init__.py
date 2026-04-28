"""Starter executable layer for the compiled rubric system (schemas, compiler scaffold, heuristic demo)."""

from rubric_gen.compiled.schema import (
    AdjudicationRecord,
    AffectEntry,
    AggregationPolicy,
    CaseEvaluationRecord,
    CaseRubric,
    CompiledCriterion,
    CriterionResult,
    CriterionTemplate,
    DimensionScore,
    DimensionSpec,
    DocumentationContract,
    ErrorTaxonomyEntry,
    EvidenceAnchor,
    NoteFamilySpec,
    TaskFamilySpec,
    RubricOntology,
    SectionSpec,
    SubdimensionSpec,
)
from rubric_gen.compiled.llm_judge import evaluate_note_with_llm_judge, resolve_compiled_judge_spec
from rubric_gen.compiled.serialize import to_json_dict, write_json

__all__ = [
    "AdjudicationRecord",
    "AffectEntry",
    "AggregationPolicy",
    "CaseEvaluationRecord",
    "CaseRubric",
    "CompiledCriterion",
    "CriterionResult",
    "CriterionTemplate",
    "DimensionScore",
    "DimensionSpec",
    "DocumentationContract",
    "ErrorTaxonomyEntry",
    "EvidenceAnchor",
    "NoteFamilySpec",
    "TaskFamilySpec",
    "RubricOntology",
    "SectionSpec",
    "SubdimensionSpec",
    "to_json_dict",
    "write_json",
    "evaluate_note_with_llm_judge",
    "resolve_compiled_judge_spec",
]
