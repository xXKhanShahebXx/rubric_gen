from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    max_tokens: int = 2048


@dataclass
class ExampleRecord:
    example_id: str
    source: str
    source_id: str
    dataset_subset: str
    conversation: str
    task_prompt: str
    reference_note: str = ""
    augmented_note: str = ""
    note_truncated: str = ""
    structured_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_profile_id: str = "note_documentation"
    task_family_id: str = ""
    artifact_kind: str = "note"
    reference_artifact: str = ""
    augmented_artifact: str = ""
    artifact_truncated: str = ""

    def __post_init__(self) -> None:
        if not self.reference_artifact and self.reference_note:
            self.reference_artifact = self.reference_note
        if not self.reference_note and self.reference_artifact:
            self.reference_note = self.reference_artifact

        if not self.augmented_artifact and self.augmented_note:
            self.augmented_artifact = self.augmented_note
        if not self.augmented_note and self.augmented_artifact:
            self.augmented_note = self.augmented_artifact

        if not self.artifact_truncated and self.note_truncated:
            self.artifact_truncated = self.note_truncated
        if not self.note_truncated and self.artifact_truncated:
            self.note_truncated = self.artifact_truncated

        if not self.artifact_kind:
            self.artifact_kind = "artifact"

    def artifact_fields(self) -> List[Tuple[str, str]]:
        return [
            ("reference_artifact", self.reference_artifact),
            ("augmented_artifact", self.augmented_artifact),
            ("artifact_truncated", self.artifact_truncated),
        ]

    @property
    def strongest_reference_artifact(self) -> str:
        for _, value in self.artifact_fields():
            if value.strip():
                return value
        return ""


@dataclass
class CandidateNote:
    candidate_id: str
    example_id: str
    text: str
    source_label: str
    quality_bucket: str
    origin_kind: str
    model_alias: Optional[str] = None
    provider: Optional[str] = None
    prompt_style: Optional[str] = None
    temperature: float = 0.0
    parent_candidate_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifact_kind: str = "artifact"
    task_profile_id: str = ""
    task_family_id: str = ""


@dataclass
class RubricCriterion:
    rubric_id: str
    text: str
    source_stage: str
    depth: int
    round_index: int
    parent_id: Optional[str] = None
    coverage_count: int = 0
    accepted: bool = True
    rejection_reason: Optional[str] = None
    filter_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RubricEvaluation:
    rubric_id: str
    candidate_id: str
    satisfied: bool
    reasoning: str = ""
    raw_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateScore:
    candidate_id: str
    method: str
    score: float
    rank: int
    satisfied_count: int
    rubric_count: int
    quality_bucket: str
    source_label: str
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMTextResponse:
    text: str
    raw_text: str
    latency_s: float
    model_alias: str
    provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)


CandidateArtifact = CandidateNote
