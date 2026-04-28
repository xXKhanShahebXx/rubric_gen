from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from rubric_gen.types import ExampleRecord


@dataclass(frozen=True)
class TaskProfile:
    task_profile_id: str
    label: str
    artifact_label: str
    artifact_kind: str
    default_task_prompt: str
    default_task_family_id: str
    contrast_strategy_id: str
    strong_source_priority: Tuple[str, ...]
    discovery_context: str
    discovery_dimensions: Tuple[str, ...]
    parent_profile_id: Optional[str] = None
    built_in: bool = True
    feature_tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


_PROFILE_REGISTRY: Dict[str, TaskProfile] = {
    "note_documentation": TaskProfile(
        task_profile_id="note_documentation",
        label="Medical note documentation",
        artifact_label="note",
        artifact_kind="note",
        default_task_prompt="Write a clinically faithful medical note from the source transcript.",
        default_task_family_id="general_clinical_note",
        contrast_strategy_id="note_documentation",
        strong_source_priority=("reference_note", "reference_artifact", "augmented_note", "augmented_artifact", "note_truncated", "artifact_truncated"),
        discovery_context="clinical dialogue-to-note quality",
        discovery_dimensions=(
            "structure",
            "symptom_detail",
            "pertinent_negatives",
            "study_results",
            "follow_up_specificity",
            "return_precautions",
            "medication_management",
            "intervention_plan",
            "testing_plan",
            "diagnostic_reasoning",
            "certainty_language",
        ),
    ),
    "documentation_variants": TaskProfile(
        task_profile_id="documentation_variants",
        label="Documentation variants",
        artifact_label="document",
        artifact_kind="document",
        default_task_prompt="Produce the requested documentation artifact from the provided context.",
        default_task_family_id="generic_document",
        contrast_strategy_id="documentation_variants",
        strong_source_priority=("reference_artifact", "reference_note", "augmented_artifact", "augmented_note", "artifact_truncated", "note_truncated"),
        discovery_context="documentation artifact quality",
        discovery_dimensions=(
            "structure",
            "source_grounding",
            "content_coverage",
            "action_items",
            "follow_up",
            "reasoning_support",
        ),
    ),
    "rewrite_editing": TaskProfile(
        task_profile_id="rewrite_editing",
        label="Rewrite and editing",
        artifact_label="rewrite",
        artifact_kind="artifact",
        default_task_prompt="Rewrite the provided text while preserving requested constraints and source meaning.",
        default_task_family_id="rewrite_transform",
        contrast_strategy_id="rewrite_editing",
        strong_source_priority=("reference_artifact", "augmented_artifact", "artifact_truncated"),
        discovery_context="rewrite/edit quality",
        discovery_dimensions=(
            "instruction_adherence",
            "meaning_preservation",
            "style_transformation",
            "format_compliance",
            "anti_fabrication",
        ),
    ),
    "clinical_decision_support": TaskProfile(
        task_profile_id="clinical_decision_support",
        label="Clinical decision support",
        artifact_label="recommendation",
        artifact_kind="response",
        default_task_prompt="Generate a grounded clinical decision-support response from the provided context.",
        default_task_family_id="recommendation_plan",
        contrast_strategy_id="clinical_decision_support",
        strong_source_priority=("reference_artifact", "augmented_artifact", "artifact_truncated"),
        discovery_context="clinical decision-support quality",
        discovery_dimensions=(
            "context_grounding",
            "recommendation_quality",
            "reasoning",
            "safety",
            "next_steps",
            "follow_up",
        ),
    ),
    "general_instruction_following": TaskProfile(
        task_profile_id="general_instruction_following",
        label="General instruction following",
        artifact_label="response",
        artifact_kind="response",
        default_task_prompt="Follow the task instructions using the provided context.",
        default_task_family_id="context_bound_response",
        contrast_strategy_id="general_instruction_following",
        strong_source_priority=("reference_artifact", "augmented_artifact", "artifact_truncated"),
        discovery_context="instruction-following response quality",
        discovery_dimensions=(
            "instruction_adherence",
            "grounding",
            "completeness",
            "format_communication",
            "safety",
        ),
    ),
    "agentic_workflows": TaskProfile(
        task_profile_id="agentic_workflows",
        label="Agentic workflows",
        artifact_label="workflow output",
        artifact_kind="workflow_output",
        default_task_prompt="Produce the requested workflow output grounded in the observed steps and results.",
        default_task_family_id="tool_grounded_run",
        contrast_strategy_id="agentic_workflows",
        strong_source_priority=("reference_artifact", "augmented_artifact", "artifact_truncated"),
        discovery_context="agentic workflow output quality",
        discovery_dimensions=(
            "task_completion",
            "tool_result_grounding",
            "verification",
            "failure_handling",
            "final_response_quality",
        ),
    ),
}

_DYNAMIC_PROFILE_REGISTRY: Dict[str, TaskProfile] = {}

_PROFILE_INFERENCE_ORDER: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("note_documentation", ("healthcare scribe", "medical note", "soap note", "clinical note", "transcript")),
    (
        "documentation_variants",
        ("discharge", "progress note", "pre-op", "preop", "summary", "handoff", "patient message", "alert"),
    ),
    ("rewrite_editing", ("rewrite", "rephrase", "edit", "grammar", "proofread", "tone", "style", "shorten", "expand")),
    (
        "clinical_decision_support",
        ("differential", "recommendation", "next step", "management plan", "triage", "workup", "assessment"),
    ),
    (
        "agentic_workflows",
        ("workflow", "tool result", "observation", "retry", "verification", "failed step", "agent"),
    ),
)


def _has_profile_cue(normalized_text: str, cue: str) -> bool:
    needle = " ".join((cue or "").lower().split())
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def list_task_profiles() -> List[TaskProfile]:
    return list(task_profile_registry().values())


def get_task_profile(task_profile_id: str | None) -> TaskProfile:
    profile_id = (task_profile_id or "note_documentation").strip() or "note_documentation"
    if profile_id in _PROFILE_REGISTRY:
        return _PROFILE_REGISTRY[profile_id]
    if profile_id in _DYNAMIC_PROFILE_REGISTRY:
        return _DYNAMIC_PROFILE_REGISTRY[profile_id]
    return _PROFILE_REGISTRY["general_instruction_following"]


def infer_task_profile_id_from_text(blob: str) -> str:
    normalized = " ".join((blob or "").lower().split())
    for profile_id, cues in _PROFILE_INFERENCE_ORDER:
        if any(_has_profile_cue(normalized, cue) for cue in cues):
            return profile_id
    return "general_instruction_following"


def infer_task_profile_id(example: ExampleRecord, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if example.task_profile_id and example.task_profile_id.strip():
        return example.task_profile_id.strip()
    blob = " ".join(
        part
        for part in (
            example.task_prompt,
            example.conversation,
            example.reference_artifact,
            example.augmented_artifact,
            example.task_family_id,
            str(example.metadata.get("task_profile_id", "")),
        )
        if part
    )
    return infer_task_profile_id_from_text(blob)


def resolve_task_profile(example: ExampleRecord, explicit: str | None = None) -> TaskProfile:
    return get_task_profile(infer_task_profile_id(example, explicit=explicit))


def task_profile_registry() -> Mapping[str, TaskProfile]:
    merged = dict(_PROFILE_REGISTRY)
    merged.update(_DYNAMIC_PROFILE_REGISTRY)
    return merged


def register_task_profile(profile: TaskProfile) -> TaskProfile:
    _DYNAMIC_PROFILE_REGISTRY[profile.task_profile_id] = profile
    return profile


def clear_dynamic_task_profiles() -> None:
    _DYNAMIC_PROFILE_REGISTRY.clear()


def task_profile_archetype_id(task_profile_id: str | None) -> str:
    profile = get_task_profile(task_profile_id)
    return profile.parent_profile_id or profile.task_profile_id
