"""
HealthBench-based external evaluation for the compiled rubric discovery pipeline.

The evaluator does four things:
1. Routes HealthBench examples into task-profile buckets while preserving the old note slice for regression.
2. Runs local strong-vs-weak rubric discovery on ideal-vs-alt completions for each selected example.
3. Compares generated local criteria to HealthBench physician criteria at the example level.
4. Samples disagreements for a separate adjudication pass so wording/granularity mismatches do not get
   over-counted as true misses.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.discovery import (
    RecursiveDiscoveryConfig,
    discover_pair_criteria,
    merge_proposal_entries,
)
from rubric_gen.compiled.gold_refinement import (
    aggregate_granularity_reports,
    apply_calibration_hints_to_generated_row,
    build_prompt_calibration_guidance,
    calibration_enabled_profiles,
    classify_granularity_gaps,
    derive_calibration_hints,
    refine_generated_rows,
)
from rubric_gen.compiled.gold_standards import (
    GoldAlignmentArtifact,
    GoldCriterion,
    build_alignment_lookup,
    gold_criteria_as_rows,
)
from rubric_gen.compiled.llm_judge import resolve_compiled_judge_spec
from rubric_gen.compiled.serialize import write_json
from rubric_gen.compiled.task_profiles import get_task_profile, infer_task_profile_id_from_text
from rubric_gen.config import discover_default_comparison_judge_model, parse_model_spec
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec


FILTER_VERSION = "healthbench_filter_v1"
ROUTING_VERSION = "healthbench_routing_v1"
ALIGNMENT_PROMPT_VERSION = "healthbench_alignment_v1"
ADJUDICATION_PROMPT_VERSION = "healthbench_adjudication_v1"

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_SECTION_HEADER_RE = re.compile(
    r"(?m)^\s*(?:\*{0,2})?"
    r"(subjective|objective|assessment(?:\s+and\s+plan)?|plan|chief complaint|history of present illness|"
    r"hpi|pmh|past medical history|medical history|social history|family history|review of systems|ros|"
    r"physical exam(?:ination)?|vital signs|diagnosis|orders|follow-up|follow up|instructions)"
    r"\s*(?:[:\-]|\*\*:)",
    re.IGNORECASE,
)

_PRIMARY_NOTE_TASK_CUES = (
    "clinical note",
    "progress note",
    "outpatient progress note",
    "inpatient note",
    "pre-op note",
    "preoperative note",
    "opd note",
    "medical report",
    "draft an outpatient progress note",
    "draft a preoperative note",
    "draft an inpatient note",
    "drafting a preoperative note",
    "provides an outpatient progress note",
    "draft a clinical note",
    "rewrite the entire note",
    "note includes",
    "assessment and plan",
    "a&p",
    "check please",
)

_SECONDARY_DOCUMENTATION_CUES = (
    "ehr alert",
    "text alert",
    "patient counseling summary",
    "mychart message",
    "handout",
    "slide deck",
    "content to highlight",
    "documentation task",
)

_DOCUMENTATION_VARIANT_CUES = (
    "progress note",
    "discharge",
    "discharge summary",
    "pre-op note",
    "preoperative note",
    "medical report",
    "mychart message",
    "patient message",
    "ehr alert",
    "text alert",
    "summary",
    "handout",
    "slide deck",
    "report",
)

_REWRITE_TASK_CUES = (
    "rewrite",
    "rephrase",
    "edit",
    "proofread",
    "grammar",
    "tone",
    "style",
    "bullet points",
    "json",
)

_CARE_GUIDANCE_CUES = (
    "what should i do",
    "what do i do",
    "what could it be",
    "any ideas what it could be",
    "can i just wait it out",
    "can i wait",
    "should i worry",
    "should i see",
    "should i go",
    "is it normal",
    "what does it mean",
    "how long",
    "what treatment",
    "what are the options",
    "how do we proceed",
    "cuanto tiempo",
    "cuánto tiempo",
    "que debo hacer",
    "qué debo hacer",
)

_DECISION_SUPPORT_CUES = (
    "recommend",
    "recommendation",
    "differential",
    "next step",
    "next steps",
    "triage",
    "management",
    "workup",
    "assessment",
    "interventions",
    "evidence-based",
    "guideline",
    "guidelines",
    "diagnosis",
    "treatment",
)

_AGENTIC_TASK_CUES = (
    "workflow",
    "tool result",
    "search results",
    "use the tool",
    "run the tool",
    "call the tool",
    "observation",
    "retry",
    "error log",
    "failed step",
    "step 1",
    "step 2",
    "multi-step",
    "multi step",
)

_DECISION_SUPPORT_THEME_CUES = (
    "context_seeking",
    "emergency_referrals",
    "global_health",
    "health_data_tasks",
    "complex_responses",
)

_PROVIDER_CUES = (
    "i'm a pulmonologist",
    "i am a pulmonologist",
    "i am an outpatient cardiologist",
    "i'm an outpatient cardiologist",
    "i'm a doctor",
    "i am a doctor",
    "i'm a nurse",
    "i am a nurse",
    "i'm an informatics nurse",
    "i am an informatics nurse",
    "health professional",
    "clinician",
    "doctor",
    "nurse practitioner",
    "resident",
    "outpatient cardiologist",
    "pulmonologist",
)

_NEGATIVE_POLARITY_CUES = (
    "fails to",
    "does not",
    "doesn't",
    "do not",
    "must not",
    "should not",
    "avoid",
    "avoids",
    "buries",
    "unsupported",
    "inaccurate",
    "incorrect",
    "without explanation",
    "assume",
    "assumes",
    "invent",
    "overly technical",
    "unsafe",
    "not given",
    "lack",
    "omits",
    "omission",
)

_FAMILY_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "when",
    "must",
    "should",
    "would",
    "include",
    "including",
    "document",
    "documents",
    "note",
    "response",
    "patient",
    "clinical",
    "medical",
    "health",
    "care",
    "task",
    "same",
    "given",
    "based",
    "there",
    "their",
    "about",
    "into",
    "than",
    "them",
    "they",
    "your",
    "have",
    "has",
    "been",
    "being",
    "under",
    "only",
    "just",
    "more",
    "less",
    "very",
}


@dataclass
class HealthBenchCompletion:
    completion_id: str
    text: str
    source: str


@dataclass
class HealthBenchExpertCriterion:
    criterion: str
    points: int
    tags: List[str] = field(default_factory=list)


@dataclass
class HealthBenchExample:
    prompt_id: str
    dialogue: str
    reference_answer: str
    completions: List[HealthBenchCompletion]
    expert_rubrics: List[HealthBenchExpertCriterion]
    is_multi_turn: bool
    n_turns: int
    themes: List[str] = field(default_factory=list)
    example_tags: List[str] = field(default_factory=list)

    def ideal_completion(self) -> Optional[HealthBenchCompletion]:
        for completion in self.completions:
            if completion.completion_id == "ideal" and completion.text.strip():
                return completion
        return None


@dataclass
class SubsetDecision:
    prompt_id: str
    selected: bool
    category: str
    reasons: List[str] = field(default_factory=list)
    dialogue_note_hits: List[str] = field(default_factory=list)
    expert_note_hits: List[str] = field(default_factory=list)
    secondary_hits: List[str] = field(default_factory=list)
    provider_hits: List[str] = field(default_factory=list)
    dialogue_header_count: int = 0
    ideal_header_count: int = 0


@dataclass
class HealthBenchRoutingDecision:
    prompt_id: str
    selected: bool
    category: str
    task_profile_id: str
    task_family_id: str
    artifact_kind: str
    route_confidence: str
    reasons: List[str] = field(default_factory=list)
    note_regression_selected: bool = False
    note_regression_category: str = "excluded"


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _phrase_hits(text: str, phrases: Sequence[str]) -> List[str]:
    normalized = _normalize_text(text)
    hits: List[str] = []
    for phrase in phrases:
        if _has_phrase(normalized, phrase) and phrase not in hits:
            hits.append(phrase)
    return hits


def _has_phrase(normalized_text: str, phrase: str) -> bool:
    needle = _normalize_text(phrase)
    if not needle:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _has_any_phrase(normalized_text: str, phrases: Sequence[str]) -> bool:
    return any(_has_phrase(normalized_text, phrase) for phrase in phrases)


def _note_header_count(text: str) -> int:
    labels = {match.group(1).lower() for match in _SECTION_HEADER_RE.finditer(text or "")}
    return len(labels)


def _criterion_level(tags: Sequence[str]) -> str:
    for tag in tags:
        if tag == "level:example":
            return "example"
        if tag == "level:cluster":
            return "cluster"
    return "unknown"


def infer_criterion_polarity(text: str, *, points: Optional[int] = None) -> str:
    if points is not None and points < 0:
        return "negative"
    normalized = _normalize_text(text)
    if any(cue in normalized for cue in _NEGATIVE_POLARITY_CUES):
        return "negative"
    return "positive"


def classify_task_compatibility(example: HealthBenchExample) -> SubsetDecision:
    dialogue_hits = _phrase_hits(example.dialogue, _PRIMARY_NOTE_TASK_CUES)
    expert_hits = _phrase_hits(
        "\n".join(criterion.criterion for criterion in example.expert_rubrics),
        _PRIMARY_NOTE_TASK_CUES,
    )
    secondary_hits = _phrase_hits(
        f"{example.dialogue}\n" + "\n".join(criterion.criterion for criterion in example.expert_rubrics),
        _SECONDARY_DOCUMENTATION_CUES,
    )
    provider_hits = _phrase_hits(
        f"{example.dialogue}\n" + "\n".join(example.example_tags),
        _PROVIDER_CUES,
    )
    if any(tag == "physician_agreed_category:health-professional" for tag in example.example_tags):
        provider_hits = list(dict.fromkeys([*provider_hits, "health-professional-tag"]))

    dialogue_header_count = _note_header_count(example.dialogue)
    ideal_header_count = _note_header_count(example.ideal_completion().text if example.ideal_completion() else "")

    reasons: List[str] = []
    category = "excluded"
    selected = False

    if ideal_header_count >= 2:
        reasons.append("ideal_completion_has_multiple_note_headers")
    if dialogue_header_count >= 2:
        reasons.append("dialogue_contains_embedded_note_sections")
    if dialogue_hits:
        reasons.append("dialogue_requests_note_like_output")
    if expert_hits:
        reasons.append("expert_rubrics_reference_note_task")
    if provider_hits:
        reasons.append("provider_or_health_professional_context")
    if secondary_hits:
        reasons.append("documentation_adjacent_secondary_task")

    if ideal_header_count >= 2 or dialogue_header_count >= 2 or dialogue_hits:
        category = "primary_note_task"
        selected = True
    elif expert_hits and provider_hits:
        category = "primary_note_task"
        selected = True
    elif secondary_hits and (provider_hits or ideal_header_count >= 1):
        category = "secondary_documentation_task"
        selected = False

    return SubsetDecision(
        prompt_id=example.prompt_id,
        selected=selected,
        category=category,
        reasons=reasons,
        dialogue_note_hits=dialogue_hits,
        expert_note_hits=expert_hits,
        secondary_hits=secondary_hits,
        provider_hits=provider_hits,
        dialogue_header_count=dialogue_header_count,
        ideal_header_count=ideal_header_count,
    )


def load_healthbench_dataset(dataset_path: Path) -> Tuple[Dict[str, Any], List[HealthBenchExample]]:
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    examples: List[HealthBenchExample] = []
    for row in payload.get("examples", []):
        completions = [
            HealthBenchCompletion(
                completion_id=str(item.get("id", "")).strip(),
                text=str(item.get("text", "")).strip(),
                source=str(item.get("source", "")).strip(),
            )
            for item in row.get("completions", [])
            if isinstance(item, dict)
        ]
        expert_rubrics = [
            HealthBenchExpertCriterion(
                criterion=str(item.get("criterion", "")).strip(),
                points=int(item.get("points", 0) or 0),
                tags=[str(tag) for tag in item.get("tags", []) if isinstance(tag, str)],
            )
            for item in row.get("expert_rubrics", [])
            if isinstance(item, dict)
        ]
        examples.append(
            HealthBenchExample(
                prompt_id=str(row.get("prompt_id", "")).strip(),
                dialogue=str(row.get("dialogue", "")).strip(),
                reference_answer=str(row.get("reference_answer", "")).strip(),
                completions=completions,
                expert_rubrics=expert_rubrics,
                is_multi_turn=bool(row.get("is_multi_turn", False)),
                n_turns=int(row.get("n_turns", 0) or 0),
                themes=[str(x) for x in row.get("themes", []) if isinstance(x, str)],
                example_tags=[str(x) for x in row.get("example_tags", []) if isinstance(x, str)],
            )
        )

    metadata = {k: v for k, v in payload.items() if k != "examples"}
    return metadata, examples


def select_healthbench_subset(
    examples: Sequence[HealthBenchExample],
) -> Tuple[List[HealthBenchExample], List[SubsetDecision], Dict[str, Any]]:
    decisions = [classify_task_compatibility(example) for example in examples]
    selected_ids = {decision.prompt_id for decision in decisions if decision.selected}
    selected = [example for example in examples if example.prompt_id in selected_ids]

    counts = {
        "primary_note_task": sum(1 for d in decisions if d.category == "primary_note_task"),
        "secondary_documentation_task": sum(
            1 for d in decisions if d.category == "secondary_documentation_task"
        ),
        "excluded": sum(1 for d in decisions if d.category == "excluded"),
        "selected": sum(1 for d in decisions if d.selected),
    }
    summary = {
        "schema": "healthbench_subset_summary_v1",
        "filter_version": FILTER_VERSION,
        "counts": counts,
        "selected_prompt_ids": [example.prompt_id for example in selected],
    }
    return selected, decisions, summary


def _ideal_or_reference_text(example: HealthBenchExample) -> str:
    ideal = example.ideal_completion()
    if ideal is not None and ideal.text.strip():
        return ideal.text
    return example.reference_answer


def _routing_blob(example: HealthBenchExample) -> str:
    return "\n".join(
        [
            example.dialogue,
            _ideal_or_reference_text(example),
            "\n".join(criterion.criterion for criterion in example.expert_rubrics),
            "\n".join(example.themes),
            "\n".join(example.example_tags),
        ]
    )


def _routing_task_blob(example: HealthBenchExample) -> str:
    return "\n".join(
        [
            example.dialogue,
            "\n".join(example.themes),
            "\n".join(example.example_tags),
        ]
    )


def _route_task_family(task_profile_id: str, normalized: str) -> str:
    if task_profile_id == "documentation_variants":
        if _has_any_phrase(normalized, ("discharge", "discharge summary")):
            return "discharge_summary"
        if _has_any_phrase(normalized, ("pre-op note", "preoperative note")):
            return "preop_note"
        if _has_any_phrase(normalized, ("progress note", "outpatient progress note")):
            return "progress_note"
        if _has_any_phrase(normalized, ("alert", "ehr alert", "text alert")):
            return "alert_message"
        if _has_any_phrase(normalized, ("mychart message", "patient message")):
            return "patient_message"
        if _has_any_phrase(normalized, ("summary", "report", "handout", "slide deck")):
            return "clinical_summary"
        return "generic_document"
    if task_profile_id == "rewrite_editing":
        if _has_any_phrase(normalized, ("grammar", "proofread")):
            return "grammar_cleanup"
        if _has_any_phrase(normalized, ("tone", "style", "formal", "casual")):
            return "style_transfer"
        if _has_any_phrase(normalized, ("json", "table", "bullet", "bullet points", "format")):
            return "structured_transform"
        return "rewrite_transform"
    if task_profile_id == "clinical_decision_support":
        if _has_any_phrase(normalized, ("differential",)):
            return "differential_diagnosis"
        if _has_any_phrase(normalized, ("triage", "urgent", "emergency")):
            return "next_step_triage"
        if _has_any_phrase(normalized, ("recommend", "management", "plan", "intervention")):
            return "recommendation_plan"
        return "evidence_based_answer"
    if task_profile_id == "agentic_workflows":
        if _has_any_phrase(normalized, ("retry", "blocked", "failure", "error")):
            return "failure_recovery"
        if _has_any_phrase(normalized, ("tool", "query", "search", "result", "observation")):
            return "tool_grounded_run"
        if _has_any_phrase(normalized, ("step 1", "step 2", "workflow", "sequence")):
            return "multi_step_execution"
        return "workflow_summary"
    if task_profile_id == "note_documentation":
        return "general_clinical_note"
    if _has_any_phrase(normalized, ("json", "table", "bullet", "bullet points", "format")):
        return "format_constrained_response"
    if normalized:
        return "context_bound_response"
    return "open_instruction_response"


def route_healthbench_task(example: HealthBenchExample) -> HealthBenchRoutingDecision:
    note_decision = classify_task_compatibility(example)
    reference_text = _ideal_or_reference_text(example)
    task_blob = _routing_task_blob(example)
    full_blob = _routing_blob(example)
    normalized_task = _normalize_text(task_blob)
    normalized_full = _normalize_text(full_blob)
    reasons: List[str] = []
    route_confidence = "medium"

    if not reference_text.strip():
        return HealthBenchRoutingDecision(
            prompt_id=example.prompt_id,
            selected=False,
            category="unsupported",
            task_profile_id="general_instruction_following",
            task_family_id="open_instruction_response",
            artifact_kind="response",
            route_confidence="low",
            reasons=["missing_reference_completion"],
            note_regression_selected=note_decision.selected,
            note_regression_category=note_decision.category,
        )

    if note_decision.category == "secondary_documentation_task":
        task_profile_id = "documentation_variants"
        reasons.append("secondary_documentation_slice_match")
        route_confidence = "high"
    elif note_decision.selected:
        task_profile_id = "note_documentation"
        reasons.append("note_regression_slice_match")
        route_confidence = "high"
    elif _has_any_phrase(
        normalized_task,
        ("ehr alert", "text alert", "mychart message", "patient message", "handout", "slide deck"),
    ):
        task_profile_id = "documentation_variants"
        reasons.append("documentation_variant_cues")
        route_confidence = "high"
    elif _has_any_phrase(normalized_task, _DOCUMENTATION_VARIANT_CUES) and not _has_any_phrase(
        normalized_task, ("subjective", "objective", "assessment and plan", "soap")
    ):
        task_profile_id = "documentation_variants"
        reasons.append("documentation_variant_cues")
        route_confidence = "high"
    elif _has_any_phrase(normalized_task, _REWRITE_TASK_CUES):
        task_profile_id = "rewrite_editing"
        reasons.append("rewrite_task_cues")
        route_confidence = "high"
    elif _has_any_phrase(normalized_task, _AGENTIC_TASK_CUES):
        task_profile_id = "agentic_workflows"
        reasons.append("agentic_task_cues")
        route_confidence = "high"
    elif _has_any_phrase(normalized_task, _CARE_GUIDANCE_CUES) or _has_any_phrase(
        normalized_task, _DECISION_SUPPORT_CUES
    ) or _has_any_phrase(normalized_task, _DECISION_SUPPORT_THEME_CUES) or _has_any_phrase(
        normalized_full, _DECISION_SUPPORT_CUES
    ):
        task_profile_id = "clinical_decision_support"
        reasons.append("clinical_decision_support_cues")
        route_confidence = "high"
    else:
        inferred = infer_task_profile_id_from_text(task_blob)
        task_profile_id = inferred or "general_instruction_following"
        if task_profile_id == "general_instruction_following":
            reasons.append("general_instruction_following_fallback")
            route_confidence = "low"
        else:
            reasons.append("task_text_inference_match")

    task_family_id = _route_task_family(task_profile_id, normalized_full)
    profile = get_task_profile(task_profile_id)
    return HealthBenchRoutingDecision(
        prompt_id=example.prompt_id,
        selected=True,
        category=task_profile_id,
        task_profile_id=task_profile_id,
        task_family_id=task_family_id,
        artifact_kind=profile.artifact_kind,
        route_confidence=route_confidence,
        reasons=reasons,
        note_regression_selected=note_decision.selected,
        note_regression_category=note_decision.category,
    )


def route_healthbench_examples(
    examples: Sequence[HealthBenchExample],
) -> Tuple[List[HealthBenchExample], List[HealthBenchRoutingDecision], Dict[str, Any]]:
    decisions = [route_healthbench_task(example) for example in examples]
    selected_ids = {decision.prompt_id for decision in decisions if decision.selected}
    selected = [example for example in examples if example.prompt_id in selected_ids]
    category_counts: Dict[str, int] = {}
    task_profile_counts: Dict[str, int] = {}
    confidence_counts: Dict[str, int] = {}
    for decision in decisions:
        category_counts[decision.category] = category_counts.get(decision.category, 0) + 1
        task_profile_counts[decision.task_profile_id] = (
            task_profile_counts.get(decision.task_profile_id, 0) + 1
        )
        confidence_counts[decision.route_confidence] = (
            confidence_counts.get(decision.route_confidence, 0) + 1
        )
    summary = {
        "schema": "healthbench_routing_summary_v1",
        "routing_version": ROUTING_VERSION,
        "counts": {
            "selected": sum(1 for decision in decisions if decision.selected),
            "excluded": sum(1 for decision in decisions if not decision.selected),
            "note_regression_selected": sum(
                1 for decision in decisions if decision.note_regression_selected
            ),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "task_profile_counts": dict(sorted(task_profile_counts.items())),
        "route_confidence_counts": dict(sorted(confidence_counts.items())),
        "selected_prompt_ids": [example.prompt_id for example in selected],
        "note_regression_prompt_ids": [
            decision.prompt_id for decision in decisions if decision.note_regression_selected
        ],
    }
    return selected, decisions, summary


def build_healthbench_candidates(
    example: HealthBenchExample,
    *,
    example_id: str,
    task_profile_id: str,
    task_family_id: str,
    artifact_kind: str,
) -> Tuple[Optional[CandidateNote], List[CandidateNote]]:
    ideal = example.ideal_completion()
    strong: Optional[CandidateNote] = None
    if ideal is not None:
        strong = CandidateNote(
            candidate_id=f"{example_id}__{ideal.completion_id}",
            example_id=example_id,
            text=ideal.text,
            source_label="ideal_completion",
            quality_bucket="reference",
            origin_kind="healthbench_completion",
            metadata={"completion_id": ideal.completion_id, "completion_source": ideal.source},
            artifact_kind=artifact_kind,
            task_profile_id=task_profile_id,
            task_family_id=task_family_id,
        )
    elif example.reference_answer.strip():
        strong = CandidateNote(
            candidate_id=f"{example_id}__reference_answer",
            example_id=example_id,
            text=example.reference_answer,
            source_label="reference_answer",
            quality_bucket="reference",
            origin_kind="healthbench_reference",
            metadata={"completion_source": "reference_answer"},
            artifact_kind=artifact_kind,
            task_profile_id=task_profile_id,
            task_family_id=task_family_id,
        )

    weak: List[CandidateNote] = []
    for completion in example.completions:
        if strong and completion.completion_id == strong.metadata.get("completion_id"):
            continue
        if completion.completion_id == "ideal":
            continue
        if not completion.text.strip():
            continue
        weak.append(
            CandidateNote(
                candidate_id=f"{example_id}__{completion.completion_id}",
                example_id=example_id,
                text=completion.text,
                source_label=f"healthbench_alt:{completion.completion_id}",
                quality_bucket="contrast_alt",
                origin_kind="healthbench_completion",
                metadata={"completion_id": completion.completion_id, "completion_source": completion.source},
                artifact_kind=artifact_kind,
                task_profile_id=task_profile_id,
                task_family_id=task_family_id,
            )
        )
    return strong, weak


def healthbench_to_example_record(
    example: HealthBenchExample,
    *,
    routing_decision: HealthBenchRoutingDecision,
) -> ExampleRecord:
    strong = example.ideal_completion()
    reference_text = strong.text if strong is not None else example.reference_answer
    profile = get_task_profile(routing_decision.task_profile_id)
    reference_note = reference_text if routing_decision.task_profile_id == "note_documentation" else ""
    return ExampleRecord(
        example_id=f"healthbench__{example.prompt_id}",
        source="healthbench",
        source_id=example.prompt_id,
        dataset_subset=routing_decision.category,
        conversation=example.dialogue,
        task_prompt=profile.default_task_prompt,
        reference_note=reference_note,
        reference_artifact=reference_text,
        task_profile_id=routing_decision.task_profile_id,
        task_family_id=routing_decision.task_family_id,
        artifact_kind=routing_decision.artifact_kind,
        metadata={
            "healthbench_prompt_id": example.prompt_id,
            "themes": list(example.themes),
            "example_tags": list(example.example_tags),
            "note_regression_selected": routing_decision.note_regression_selected,
            "note_regression_category": routing_decision.note_regression_category,
        },
    )


def map_shared_rubric_family(
    text: str,
    *,
    dimension: str = "",
    tags: Optional[Sequence[str]] = None,
) -> str:
    normalized = _normalize_text(f"{dimension} {text} {' '.join(tags or [])}")

    dim_normalized = _normalize_text(dimension)
    generic_dim_fallback: Optional[str] = None
    if dim_normalized in {"format_compliance", "format_communication", "documentation_structure"}:
        generic_dim_fallback = "structure_documentation"
    elif dim_normalized in {"meaning_preservation", "content_coverage", "completeness"}:
        generic_dim_fallback = "content_coverage"
    elif dim_normalized in {"instruction_adherence", "task_completion", "final_response_quality"}:
        generic_dim_fallback = "content_coverage"
    elif dim_normalized in {
        "source_grounding",
        "context_grounding",
        "grounding",
        "tool_result_grounding",
        "anti_fabrication",
    }:
        generic_dim_fallback = "context_grounding"
    elif dim_normalized in {"action_items", "recommendation_quality"}:
        generic_dim_fallback = "next_steps_recommendation"
    elif dim_normalized in {"verification"}:
        return "verification"
    elif dim_normalized in {"failure_handling"}:
        return "failure_handling"
    elif dim_normalized in {"safety"}:
        return "factual_accuracy_safety"
    elif dim_normalized in {"reasoning_support", "reasoning"}:
        return "diagnostic_reasoning"
    elif dim_normalized in {"structure", "documentation_structure"}:
        return "structure_documentation"
    elif dim_normalized in {"symptom_detail", "pertinent_negatives", "study_results"}:
        return "symptom_exam_detail"
    elif dim_normalized in {"follow_up_specificity"}:
        return "follow_up"
    elif dim_normalized in {"return_precautions"}:
        return "return_precautions_escalation"
    elif dim_normalized in {"medication_management", "treatment_grounding"}:
        return "medication_plan"
    elif dim_normalized in {"testing_plan"}:
        return "testing_workup"
    elif dim_normalized in {"intervention_plan"}:
        return "intervention_plan"
    elif dim_normalized in {"diagnostic_reasoning"}:
        return "diagnostic_reasoning"
    elif dim_normalized in {"certainty_language"}:
        return "certainty_language"

    if _has_any_phrase(
        normalized,
        (
            "follow the user's instructions",
            "instruction adherence",
            "instruction following",
            "follow the instruction",
            "requested transformation",
            "requested format",
            "constraint following",
        ),
    ):
        return "instruction_adherence"
    if _has_any_phrase(
        normalized,
        (
            "preserve meaning",
            "preserves meaning",
            "retain the original meaning",
            "content coverage",
            "completeness",
            "requested content",
            "salient content",
        ),
    ):
        return "content_coverage"
    if _has_any_phrase(
        normalized,
        (
            "bullet points",
            "bullet point",
            "json",
            "table",
            "section headings",
            "section headers",
            "organized into",
            "structured formatting",
            "clear headings",
        ),
    ):
        return "structure_documentation"
    if _has_any_phrase(
        normalized,
        (
            "workflow step",
            "task completion",
            "completion status",
            "final deliverable",
            "final response",
        ),
    ):
        return "content_coverage"
    if _has_any_phrase(
        normalized,
        (
            "verification",
            "double-check",
            "double check",
            "validated",
            "validated result",
            "confirmed",
            "confirmed the output",
        ),
    ):
        return "verification"
    if _has_any_phrase(
        normalized,
        (
            "failure handling",
            "retry",
            "fallback",
            "blocked",
            "error recovery",
            "recovered from failure",
        ),
    ):
        return "failure_handling"
    if _has_any_phrase(
        normalized,
        (
            "next steps",
            "recommended next steps",
            "recommendation quality",
            "management recommendation",
            "action items",
        ),
    ):
        return "next_steps_recommendation"
    if _has_any_phrase(
        normalized,
        (
            "seek context",
            "request additional information",
            "requests additional information",
            "missing pieces",
            "leave a blank",
            "leaves a blank",
            "only the given information",
            "only given information",
            "does not seek context",
            "assum",
            "invent",
            "not given by the user",
            "unsupported details",
            "context awareness",
            "asks about the specific surgical procedure",
            "requests a specific task",
            "asks for guidance on what the user wants",
            "clarifies whether the surgical evaluation",
            "clarify what the user would like assistance with",
            "unsupported additions",
            "tool result grounding",
            "grounded in the available context",
        ),
    ):
        return "context_grounding"
    if _has_any_phrase(
        normalized,
        (
            "health professional",
            "plain language",
            "jargon",
            "technical terms",
            "clinical terminology",
            "communication quality",
            "understandable",
            "too simple",
            "overly technical",
            "tailored for hps",
            "tailored for hp",
        ),
    ):
        return "communication_audience"
    if _has_any_phrase(
        normalized,
        (
            "return precautions",
            "seek urgent",
            "seek emergency",
            "urgent medical assessment",
            "call their obstetrician immediately",
            "visit an emergency department",
            "warrant urgent medical assessment",
            "escalation",
            "go to the hospital right away",
            "red flag",
        ),
    ):
        return "return_precautions_escalation"
    if _has_any_phrase(
        normalized,
        (
            "consult a healthcare professional",
            "consult a doctor",
            "consult a healthcare provider",
            "see a doctor",
            "see a healthcare professional",
            "speak with a provider",
            "speak with your provider",
            "medical follow-up",
            "medical follow up",
            "proper diagnosis and treatment",
            "follow personalized medical advice",
        ),
    ):
        return "content_coverage"
    if _has_any_phrase(
        normalized,
        (
            "follow-up",
            "follow up",
            "reassess",
            "recheck",
            "monitor response",
            "next steps",
            "opd note",
            "follow-through",
            "follow-up interval",
            "follow up interval",
            "surveillance",
            "guideline",
            "guidelines",
            "fleischner",
        ),
    ):
        return "follow_up"
    if _has_any_phrase(
        normalized,
        (
            "medication",
            "antibiotic",
            "antibiotics",
            "dose",
            "dosing",
            "schedule",
            "regimen",
            "ceftriaxone",
            "ciprofloxacin",
            "levofloxacin",
            "hydroxyurea",
            "ibuprofen",
            "continue",
            "start",
            "stop",
            "refill",
        ),
    ):
        return "medication_plan"
    if _has_any_phrase(
        normalized,
        (
            "urine culture",
            "lab",
            "labs",
            "imaging",
            "workup",
            "diagnostic test",
            "diagnostic tests",
            "blood cultures",
            "ecg",
            "ekg",
            "cbc",
            "bmp",
            "pet scan",
            "pet imaging",
            "pet/ct",
            "pet ct",
            "mri",
            "ct",
            "schedule follow-up lab tests",
            "obtain",
            "repeat",
        ),
    ):
        return "testing_workup"
    if _has_any_phrase(
        normalized,
        (
            "referral",
            "consult",
            "consultation",
            "procedure",
            "surgery",
            "surgical",
            "anesthesia",
            "device",
            "intervention",
            "exchange transfusion",
        ),
    ):
        return "intervention_plan"
    if _has_any_phrase(
        normalized,
        (
            "reasoning",
            "rationale",
            "based on",
            "because",
            "why",
            "risk-benefit",
            "risk benefit",
            "assessment",
            "diagnosis-evidence",
            "supporting evidence",
            "medical decision making",
        ),
    ):
        return "diagnostic_reasoning"
    if re.search(r"(?<![a-z0-9])\d+(?:\.\d+)?\s*(?:mm|mg|mcg|ml/min|days?|weeks?|months?|years?)(?![a-z0-9])", normalized):
        if _has_any_phrase(normalized, ("follow-up", "follow up", "interval", "surveillance", "timeline")):
            return "follow_up"
        if _has_any_phrase(normalized, ("medication", "antibiotic", "dose", "dosing", "schedule", "ibuprofen", "acetaminophen")):
            return "medication_plan"
        return "factual_accuracy_safety"
    if _has_any_phrase(
        normalized,
        (
            "uncertainty",
            "certainty",
            "definitive",
            "guarantee",
            "promise",
            "overly definitive",
            "unsupported",
            "100% safe",
            "100% effective",
        ),
    ):
        return "certainty_language"
    if _has_any_phrase(
        normalized,
        (
            "structure",
            "section headings",
            "section headers",
            "clinical note",
            "progress note",
            "preoperative note",
            "pre-op evaluation",
            "outpatient progress note",
            "inpatient note",
            "medical report",
            "chief complaint",
            "hpi",
            "history of present illness",
            "pmh",
            "past medical history",
            "social history",
            "family history",
            "review of symptoms",
            "review of systems",
            "physical examination",
            "physical exam",
            "assessment and plan",
            "assessment",
            "plan.",
            "plan based on the provided information",
        ),
    ):
        return "structure_documentation"
    if _has_any_phrase(
        normalized,
        (
            "symptom",
            "fever",
            "vital signs",
            "physical exam",
            "history",
            "complaint",
            "swelling",
            "crackles",
            "retractions",
            "oxygen saturation",
        ),
    ):
        return "symptom_exam_detail"
    if _has_any_phrase(
        normalized,
        (
            "axis:accuracy",
            "accurate",
            "medical consensus",
            "safe response",
            "harm",
            "unsafe",
            "inaccuracy",
            "incorrect",
            "guideline",
            "guidelines",
        ),
    ):
        return "factual_accuracy_safety"
    if generic_dim_fallback is not None:
        return generic_dim_fallback
    return "other"


def map_expert_rubric_family(criterion: HealthBenchExpertCriterion) -> str:
    return map_shared_rubric_family(criterion.criterion, tags=criterion.tags)


def map_generated_criterion_family(row: Mapping[str, Any]) -> str:
    return map_shared_rubric_family(
        f"{row.get('label', '')} {row.get('requirement', '')}",
        dimension=str(row.get("dimension", "")),
    )


def _content_tokens(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for token in _WORD_RE.findall((part or "").lower()):
            if len(token) < 4 or token in _FAMILY_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _heuristic_alignment(
    expert_rows: Sequence[Dict[str, Any]],
    generated_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    expert_matches: List[Dict[str, Any]] = []
    generated_links: Dict[int, List[int]] = {i: [] for i in range(len(generated_rows))}

    for expert_idx, expert in enumerate(expert_rows):
        best_generated: Optional[int] = None
        best_score = 0.0
        expert_tokens = _content_tokens(expert["criterion"])
        for generated_idx, generated in enumerate(generated_rows):
            generated_tokens = _content_tokens(
                str(generated.get("label", "")),
                str(generated.get("requirement", "")),
            )
            union = expert_tokens | generated_tokens
            overlap = len(expert_tokens & generated_tokens) / max(1, len(union))
            family_bonus = (
                0.2 if expert["family"] == generated["family"] and expert["family"] != "other" else 0.0
            )
            score = overlap + family_bonus
            if score > best_score:
                best_score = score
                best_generated = generated_idx

        match_label = "none"
        if best_generated is not None and best_score >= 0.55:
            match_label = "direct"
            generated_links[best_generated].append(expert_idx)
        elif best_generated is not None and best_score >= 0.30:
            match_label = "partial"
            generated_links[best_generated].append(expert_idx)

        expert_matches.append(
            {
                "expert_index": expert_idx,
                "best_generated_index": best_generated if match_label != "none" else None,
                "match_label": match_label,
                "reason": f"heuristic_score={best_score:.2f}",
            }
        )

    generated_assessments: List[Dict[str, Any]] = []
    expert_families = {row["family"] for row in expert_rows}
    for generated_idx, generated in enumerate(generated_rows):
        matched = generated_links.get(generated_idx, [])
        if matched:
            precision_label = "aligned"
        elif generated["family"] in expert_families and generated["family"] != "other":
            precision_label = "broader_but_valid"
        elif generated["family"] == "other":
            precision_label = "off_target"
        else:
            precision_label = "valid_extra"
        generated_assessments.append(
            {
                "generated_index": generated_idx,
                "matched_expert_indices": matched,
                "precision_label": precision_label,
                "reason": "heuristic_fallback",
            }
        )

    return {
        "expert_matches": expert_matches,
        "generated_assessments": generated_assessments,
        "fallback": True,
        "parse_error": None,
    }


_ALIGNMENT_SYSTEM_PROMPT = """You compare physician-written expert rubric criteria against rubric criteria
generated by another system for the SAME task.

Rules:
- Match criteria by clinical intent, not wording overlap.
- direct: the generated criterion captures essentially the same evaluator check as the expert criterion.
- partial: the generated criterion overlaps meaningfully but is broader, narrower, or only covers part of it.
- none: the generated criterion does not meaningfully cover the expert criterion.
- Be conservative: do not match generic communication boilerplate to a specific clinical requirement unless the
  generated criterion would truly allow an evaluator to score that requirement.
- For generated_assessments:
  - aligned: clearly matches one or more expert criteria.
  - broader_but_valid: clinically valid and related, but broader than the expert wording.
  - valid_extra: reasonable additional criterion not really represented in the expert list.
  - off_target: not well supported for the task or not appropriate to compare as a rubric item.

Return a single JSON object with the exact schema requested. No markdown fences."""


def _truncate_chars(text: str, limit: int) -> str:
    clean = text or ""
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "\n…[truncated]"


def _build_alignment_prompt(
    *,
    dialogue: str,
    expert_rows: Sequence[Dict[str, Any]],
    generated_rows: Sequence[Dict[str, Any]],
) -> str:
    return (
        "DIALOGUE / TASK:\n"
        f"{_truncate_chars(dialogue, 6000)}\n\n"
        "PHYSICIAN EXPERT RUBRICS (example-level only):\n"
        f"{json.dumps(list(expert_rows), ensure_ascii=False, indent=2)}\n\n"
        "GENERATED LOCAL RUBRIC CRITERIA:\n"
        f"{json.dumps(list(generated_rows), ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "expert_matches": [\n'
        "    {\n"
        '      "expert_index": 0,\n'
        '      "best_generated_index": 0,\n'
        '      "match_label": "direct" | "partial" | "none",\n'
        '      "reason": "<short reason>"\n'
        "    }\n"
        "  ],\n"
        '  "generated_assessments": [\n'
        "    {\n"
        '      "generated_index": 0,\n'
        '      "matched_expert_indices": [0],\n'
        '      "precision_label": "aligned" | "broader_but_valid" | "valid_extra" | "off_target",\n'
        '      "reason": "<short reason>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Include exactly {len(expert_rows)} expert_matches rows and exactly {len(generated_rows)} "
        "generated_assessments rows."
    )


def _parse_alignment_payload(
    raw_text: str,
    *,
    expert_count: int,
    generated_count: int,
) -> Dict[str, Any]:
    obj = extract_json_object(raw_text)
    if not obj:
        raise ValueError("Alignment response did not contain parseable JSON.")
    expert_matches = obj.get("expert_matches")
    generated_assessments = obj.get("generated_assessments")
    if not isinstance(expert_matches, list) or not isinstance(generated_assessments, list):
        raise ValueError("Alignment JSON must contain expert_matches and generated_assessments arrays.")

    by_expert: Dict[int, Dict[str, Any]] = {}
    for row in expert_matches:
        if not isinstance(row, dict):
            continue
        idx = row.get("expert_index")
        if isinstance(idx, int):
            by_expert[idx] = row
    missing_experts = [idx for idx in range(expert_count) if idx not in by_expert]
    if missing_experts:
        raise ValueError(f"Alignment JSON missing expert indexes: {missing_experts[:8]}")

    by_generated: Dict[int, Dict[str, Any]] = {}
    for row in generated_assessments:
        if not isinstance(row, dict):
            continue
        idx = row.get("generated_index")
        if isinstance(idx, int):
            by_generated[idx] = row
    missing_generated = [idx for idx in range(generated_count) if idx not in by_generated]
    if missing_generated:
        raise ValueError(f"Alignment JSON missing generated indexes: {missing_generated[:8]}")

    normalized_expert_rows: List[Dict[str, Any]] = []
    for idx in range(expert_count):
        row = dict(by_expert[idx])
        label = str(row.get("match_label", "")).strip().lower()
        if label not in {"direct", "partial", "none"}:
            label = "none"
        best = row.get("best_generated_index")
        if not isinstance(best, int) or not (0 <= best < generated_count):
            best = None
        if label == "none":
            best = None
        normalized_expert_rows.append(
            {
                "expert_index": idx,
                "best_generated_index": best,
                "match_label": label,
                "reason": str(row.get("reason", "")).strip(),
            }
        )

    normalized_generated_rows: List[Dict[str, Any]] = []
    for idx in range(generated_count):
        row = dict(by_generated[idx])
        label = str(row.get("precision_label", "")).strip().lower()
        if label not in {"aligned", "broader_but_valid", "valid_extra", "off_target"}:
            label = "off_target"
        matched = row.get("matched_expert_indices")
        matched_indices = [
            int(value)
            for value in matched
            if isinstance(value, int) and 0 <= int(value) < expert_count
        ] if isinstance(matched, list) else []
        normalized_generated_rows.append(
            {
                "generated_index": idx,
                "matched_expert_indices": matched_indices,
                "precision_label": label,
                "reason": str(row.get("reason", "")).strip(),
            }
        )

    return {
        "expert_matches": normalized_expert_rows,
        "generated_assessments": normalized_generated_rows,
        "fallback": False,
        "parse_error": None,
    }


def _resolve_alignment_spec(explicit_model: Optional[str]) -> Optional[ModelSpec]:
    if explicit_model and explicit_model.strip():
        return parse_model_spec(explicit_model.strip(), default_alias="healthbench-alignment")
    return discover_default_comparison_judge_model()


def _load_calibration_hints(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    policy_path: Optional[Path] = None
    if candidate.is_dir():
        hints_path = candidate / "calibration_hints.json"
        policy_path = candidate / "calibration_profile_policy.json"
    else:
        hints_path = candidate
        policy_path = candidate.with_name("calibration_profile_policy.json")
    if not hints_path.exists():
        return None
    with hints_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if policy_path is not None and policy_path.exists():
            with policy_path.open("r", encoding="utf-8") as handle:
                policy_payload = json.load(handle)
            if isinstance(policy_payload, dict):
                payload = dict(payload)
                payload["eligibility"] = policy_payload
        return payload
    return None


def align_generated_to_expert(
    *,
    dialogue: str,
    expert_rows: Sequence[Dict[str, Any]],
    generated_rows: Sequence[Dict[str, Any]],
    model_spec: Optional[ModelSpec],
    router: Optional[LLMRouter],
    cache: Optional[JsonlCache],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    if not expert_rows:
        generated_assessments = [
            {
                "generated_index": idx,
                "matched_expert_indices": [],
                "precision_label": "valid_extra",
                "reason": "no_expert_rubrics_for_example",
            }
            for idx in range(len(generated_rows))
        ]
        return (
            {
                "expert_matches": [],
                "generated_assessments": generated_assessments,
                "fallback": True,
                "parse_error": None,
            },
            False,
            None,
        )

    if not generated_rows:
        return (
            {
                "expert_matches": [
                    {
                        "expert_index": idx,
                        "best_generated_index": None,
                        "match_label": "none",
                        "reason": "no_generated_criteria",
                    }
                    for idx in range(len(expert_rows))
                ],
                "generated_assessments": [],
                "fallback": True,
                "parse_error": None,
            },
            False,
            None,
        )

    if model_spec is None or router is None:
        return _heuristic_alignment(expert_rows, generated_rows), False, None

    cache_hit = False
    raw_text = ""
    payload_for_key = {
        "prompt_version": ALIGNMENT_PROMPT_VERSION,
        "model": f"{model_spec.provider}:{model_spec.model}",
        "dialogue_hash": stable_hash(dialogue),
        "expert_hash": stable_hash(list(expert_rows)),
        "generated_hash": stable_hash(list(generated_rows)),
    }
    cache_key = make_cache_key(ALIGNMENT_PROMPT_VERSION, payload_for_key)
    if cache and cache.enabled:
        cache.load()
        hit = cache.get(cache_key)
        if hit and isinstance(hit.get("raw_response"), str):
            raw_text = hit["raw_response"]
            cache_hit = True

    if not raw_text:
        user_prompt = _build_alignment_prompt(
            dialogue=dialogue,
            expert_rows=expert_rows,
            generated_rows=generated_rows,
        )
        resp = router.generate(
            model_spec,
            system_prompt=_ALIGNMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=3000,
        )
        raw_text = resp.raw_text or resp.text
        if cache and cache.enabled:
            cache.set(cache_key, {"raw_response": raw_text})

    try:
        parsed = _parse_alignment_payload(
            raw_text,
            expert_count=len(expert_rows),
            generated_count=len(generated_rows),
        )
        return parsed, cache_hit, None
    except ValueError as exc:
        fallback = _heuristic_alignment(expert_rows, generated_rows)
        fallback["parse_error"] = str(exc)
        return fallback, cache_hit, str(exc)


def healthbench_gold_criteria(example: HealthBenchExample) -> List[GoldCriterion]:
    rows: List[GoldCriterion] = []
    for idx, criterion in enumerate(example.expert_rubrics):
        if _criterion_level(criterion.tags) != "example":
            continue
        rows.append(
            GoldCriterion(
                criterion_id=f"{example.prompt_id}:expert:{idx}",
                criterion=criterion.criterion,
                points=criterion.points,
                tags=list(criterion.tags),
                polarity=infer_criterion_polarity(criterion.criterion, points=criterion.points),
                family=map_expert_rubric_family(criterion),
                metadata={
                    "expert_index": idx,
                    "level": _criterion_level(criterion.tags),
                },
            )
        )
    return rows


def _expert_rows_for_alignment(example: HealthBenchExample) -> List[Dict[str, Any]]:
    rows = gold_criteria_as_rows(healthbench_gold_criteria(example))
    for row in rows:
        metadata = row.get("metadata", {})
        row["expert_index"] = int(metadata.get("expert_index", row.get("expert_index", 0)))
        row["level"] = str(metadata.get("level", "example"))
    return rows


def _annotate_generated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_profile_id: str,
    calibration_hints: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        requirement = str(row.get("requirement", "")).strip()
        label = str(row.get("label", "")).strip()
        dimension = str(row.get("dimension", "")).strip()
        annotated_row = {
            "generated_index": idx,
            "dimension": dimension,
            "label": label,
            "requirement": requirement,
            "severity_tier": str(row.get("severity_tier", "")).strip(),
            "count": int(row.get("count", 0) or 0),
            "pair_ids": [str(x) for x in row.get("pair_ids", []) if isinstance(x, str)],
            "example_ids": [str(x) for x in row.get("example_ids", []) if isinstance(x, str)],
            "family": map_generated_criterion_family(row),
            "polarity": infer_criterion_polarity(f"{label} {requirement}"),
        }
        for key in (
            "refinement_origin",
            "gold_anchor_expert_index",
            "gold_gap_type",
            "pre_refinement_generated_index",
            "family_pre_calibration",
            "calibration_family_override",
            "criterion_ids",
            "parent_criterion_ids",
            "root_pair_ids",
            "recursion_depths",
            "recursion_reasons",
            "decomposition_sources",
            "criterion_id",
            "parent_criterion_id",
            "root_pair_id",
            "recursion_depth",
            "recursion_reason",
            "decomposition_source",
        ):
            if key in row:
                annotated_row[key] = row.get(key)
        annotated.append(
            apply_calibration_hints_to_generated_row(
                annotated_row,
                task_profile_id=task_profile_id,
                calibration_hints=calibration_hints,
            )
        )
    return annotated


def _compute_example_metrics(
    *,
    expert_rows: Sequence[Dict[str, Any]],
    generated_rows: Sequence[Dict[str, Any]],
    alignment: Mapping[str, Any],
) -> Dict[str, Any]:
    matched_expert_weight = 0
    total_expert_weight = 0
    matched_expert_count = 0
    direct_match_count = 0
    partial_match_count = 0
    polarity_match_count = 0
    polarity_total = 0

    generated_by_index = {row["generated_index"]: row for row in generated_rows}
    for expert_row in expert_rows:
        total_expert_weight += abs(int(expert_row["points"]))
    for row in alignment.get("expert_matches", []):
        if not isinstance(row, dict):
            continue
        expert_index = row.get("expert_index")
        if not isinstance(expert_index, int) or not (0 <= expert_index < len(expert_rows)):
            continue
        expert_row = expert_rows[expert_index]
        match_label = str(row.get("match_label", "none")).lower()
        if match_label not in {"direct", "partial"}:
            continue
        matched_expert_count += 1
        matched_expert_weight += abs(int(expert_row["points"]))
        if match_label == "direct":
            direct_match_count += 1
        else:
            partial_match_count += 1
        generated_index = row.get("best_generated_index")
        if isinstance(generated_index, int) and generated_index in generated_by_index:
            polarity_total += 1
            if generated_by_index[generated_index]["polarity"] == expert_row["polarity"]:
                polarity_match_count += 1

    aligned_generated_count = 0
    broader_generated_count = 0
    valid_extra_count = 0
    off_target_count = 0
    for row in alignment.get("generated_assessments", []):
        if not isinstance(row, dict):
            continue
        label = str(row.get("precision_label", "")).lower()
        if label == "aligned":
            aligned_generated_count += 1
        elif label == "broader_but_valid":
            broader_generated_count += 1
        elif label == "valid_extra":
            valid_extra_count += 1
        else:
            off_target_count += 1

    generated_total = len(generated_rows)
    expert_total = len(expert_rows)
    return {
        "expert_criteria_total": expert_total,
        "expert_criteria_matched": matched_expert_count,
        "expert_direct_matches": direct_match_count,
        "expert_partial_matches": partial_match_count,
        "weighted_recall": (matched_expert_weight / total_expert_weight) if total_expert_weight else 0.0,
        "precision": (
            (aligned_generated_count + broader_generated_count) / generated_total if generated_total else 0.0
        ),
        "generated_criteria_total": generated_total,
        "generated_aligned": aligned_generated_count,
        "generated_broader_but_valid": broader_generated_count,
        "generated_valid_extra": valid_extra_count,
        "generated_off_target": off_target_count,
        "polarity_accuracy": (polarity_match_count / polarity_total) if polarity_total else 0.0,
        "expert_recall": (matched_expert_count / expert_total) if expert_total else 0.0,
    }


def _family_coverage_rows(
    *,
    expert_rows: Sequence[Dict[str, Any]],
    generated_rows: Sequence[Dict[str, Any]],
    alignment: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}

    def ensure(family: str) -> Dict[str, Any]:
        if family not in summary:
            summary[family] = {
                "expert_count": 0,
                "expert_weight": 0,
                "matched_expert_count": 0,
                "matched_expert_weight": 0,
                "generated_count": 0,
            }
        return summary[family]

    expert_lookup = {row["expert_index"]: row for row in expert_rows}
    for expert in expert_rows:
        bucket = ensure(expert["family"])
        bucket["expert_count"] += 1
        bucket["expert_weight"] += abs(int(expert["points"]))
    for generated in generated_rows:
        ensure(generated["family"])["generated_count"] += 1
    for row in alignment.get("expert_matches", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("match_label", "")).lower() not in {"direct", "partial"}:
            continue
        expert_idx = row.get("expert_index")
        if not isinstance(expert_idx, int) or expert_idx not in expert_lookup:
            continue
        expert = expert_lookup[expert_idx]
        bucket = ensure(expert["family"])
        bucket["matched_expert_count"] += 1
        bucket["matched_expert_weight"] += abs(int(expert["points"]))
    return summary


@dataclass
class HealthBenchGoldProvider:
    provider_id: str = "healthbench"

    def gold_rows_for_example(self, example: HealthBenchExample) -> List[Dict[str, Any]]:
        return _expert_rows_for_alignment(example)

    def compare_generated_rows(
        self,
        *,
        example: HealthBenchExample,
        generated_rows: Sequence[Dict[str, Any]],
        model_spec: Optional[ModelSpec],
        router: Optional[LLMRouter],
        cache: Optional[JsonlCache],
        use_heuristic_only: bool = False,
    ) -> GoldAlignmentArtifact:
        gold_rows = self.gold_rows_for_example(example)
        alignment, cache_hit, parse_error = align_generated_to_expert(
            dialogue=example.dialogue,
            expert_rows=gold_rows,
            generated_rows=generated_rows,
            model_spec=None if use_heuristic_only else model_spec,
            router=None if use_heuristic_only else router,
            cache=None if use_heuristic_only else cache,
        )
        metrics = _compute_example_metrics(
            expert_rows=gold_rows,
            generated_rows=generated_rows,
            alignment=alignment,
        )
        family_summary = _family_coverage_rows(
            expert_rows=gold_rows,
            generated_rows=generated_rows,
            alignment=alignment,
        )
        return GoldAlignmentArtifact(
            provider_id=self.provider_id,
            gold_rows=gold_rows,
            alignment=alignment,
            metrics=metrics,
            family_summary=family_summary,
            alignment_lookup=build_alignment_lookup(alignment),
            cache_hit=cache_hit,
            parse_error=parse_error,
            metadata={"use_heuristic_only": use_heuristic_only},
        )


def _match_rank(label: str) -> int:
    return {"none": 0, "partial": 1, "direct": 2}.get(str(label).strip().lower(), 0)


def _precision_rank(label: str) -> int:
    return {
        "off_target": 0,
        "valid_extra": 1,
        "broader_but_valid": 2,
        "aligned": 3,
    }.get(str(label).strip().lower(), 0)


def _preserve_pre_refinement_matches(
    *,
    pre_alignment: Mapping[str, Any],
    post_alignment: Mapping[str, Any],
    post_generated_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = {
        "expert_matches": [dict(row) for row in post_alignment.get("expert_matches", []) if isinstance(row, Mapping)],
        "generated_assessments": [
            dict(row) for row in post_alignment.get("generated_assessments", []) if isinstance(row, Mapping)
        ],
        "fallback": bool(post_alignment.get("fallback", False)),
        "parse_error": post_alignment.get("parse_error"),
    }
    pre_to_post: Dict[int, int] = {}
    for row in post_generated_rows:
        try:
            pre_idx = row.get("pre_refinement_generated_index")
            post_idx = row.get("generated_index")
            if isinstance(pre_idx, int) and isinstance(post_idx, int):
                pre_to_post[pre_idx] = post_idx
        except AttributeError:
            continue

    expert_lookup = {
        int(row["expert_index"]): row
        for row in merged["expert_matches"]
        if isinstance(row.get("expert_index"), int)
    }
    generated_lookup = {
        int(row["generated_index"]): row
        for row in merged["generated_assessments"]
        if isinstance(row.get("generated_index"), int)
    }

    for row in pre_alignment.get("expert_matches", []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("match_label", "")).strip().lower() not in {"direct", "partial"}:
            continue
        expert_index = row.get("expert_index")
        old_generated_index = row.get("best_generated_index")
        if not isinstance(expert_index, int) or not isinstance(old_generated_index, int):
            continue
        new_generated_index = pre_to_post.get(old_generated_index)
        if new_generated_index is None:
            continue
        current = expert_lookup.get(expert_index)
        if current is None:
            current = {
                "expert_index": expert_index,
                "best_generated_index": new_generated_index,
                "match_label": row.get("match_label", "partial"),
                "reason": "preserved_pre_refinement_match",
            }
            merged["expert_matches"].append(current)
            expert_lookup[expert_index] = current
            continue
        if _match_rank(current.get("match_label", "none")) < _match_rank(row.get("match_label", "none")):
            current["best_generated_index"] = new_generated_index
            current["match_label"] = row.get("match_label", "partial")
            current["reason"] = "preserved_pre_refinement_match"

    for row in pre_alignment.get("generated_assessments", []):
        if not isinstance(row, Mapping):
            continue
        old_generated_index = row.get("generated_index")
        if not isinstance(old_generated_index, int):
            continue
        new_generated_index = pre_to_post.get(old_generated_index)
        if new_generated_index is None:
            continue
        matched_indices = [
            int(item)
            for item in (row.get("matched_expert_indices") or [])
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
        ]
        current = generated_lookup.get(new_generated_index)
        if current is None:
            current = {
                "generated_index": new_generated_index,
                "matched_expert_indices": matched_indices,
                "precision_label": row.get("precision_label", "broader_but_valid"),
                "reason": "preserved_pre_refinement_assessment",
            }
            merged["generated_assessments"].append(current)
            generated_lookup[new_generated_index] = current
            continue
        if _precision_rank(current.get("precision_label", "off_target")) < _precision_rank(
            row.get("precision_label", "off_target")
        ):
            current["precision_label"] = row.get("precision_label", "broader_but_valid")
            current["reason"] = "preserved_pre_refinement_assessment"
        current_matches = [
            int(item)
            for item in (current.get("matched_expert_indices") or [])
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
        ]
        if matched_indices:
            current["matched_expert_indices"] = sorted(set(current_matches) | set(matched_indices))

    merged["expert_matches"] = sorted(
        merged["expert_matches"],
        key=lambda row: int(row.get("expert_index", -1)),
    )
    merged["generated_assessments"] = sorted(
        merged["generated_assessments"],
        key=lambda row: int(row.get("generated_index", -1)),
    )
    return merged


def _granularity_gap_payload(raw_gap: Any) -> Dict[str, Any]:
    if isinstance(raw_gap, Mapping):
        return dict(raw_gap)
    if hasattr(raw_gap, "__dataclass_fields__"):
        return asdict(raw_gap)
    return {}


def _has_high_priority_gold_gaps(
    granularity_report: Mapping[str, Any],
    *,
    gap_types: Sequence[str] = ("missing_gold_criterion", "family_mismatch"),
    min_priority: int = 7,
) -> bool:
    target_gap_types = {str(item) for item in gap_types}
    for raw_gap in granularity_report.get("gaps", []):
        gap = _granularity_gap_payload(raw_gap)
        if not gap:
            continue
        if str(gap.get("source", "")) != "gold":
            continue
        if str(gap.get("gap_type", "")) not in target_gap_types:
            continue
        try:
            if int(gap.get("priority", 0) or 0) >= min_priority:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _post_refine_realign_reasons(
    *,
    recursive_structure_changed: bool,
    granularity_report: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    if recursive_structure_changed:
        reasons.append("recursive_structure_changed")
    if _has_high_priority_gold_gaps(granularity_report):
        reasons.append("persistent_high_priority_gold_gaps")
    return reasons


_ADJUDICATION_SYSTEM_PROMPT = """You review disagreement items between physician-written rubric criteria and
pipeline-generated criteria for the same task.

Choose the single best category.

For expert_miss items:
- true_miss: the pipeline genuinely failed to cover the expert criterion.
- wording_equivalent: the pipeline effectively covered it, but the earlier match missed the wording overlap.
- granularity_difference: overlap exists, but one side is broader or narrower.
- task_mismatch: the expert criterion is not really comparable to a note/documentation rubric for this task.
- generic_boilerplate: the expert criterion is generic boilerplate rather than a task-specific miss.

For generated_extra items:
- valid_extra: the generated criterion is clinically reasonable and useful, even though it is not in the expert list.
- wording_equivalent: it is already represented in the expert list, but wording obscured the match.
- granularity_difference: it is related to an expert criterion but broader or narrower.
- off_target: it is not appropriate or not well supported for the task.

Return a single JSON object with the exact schema requested. No markdown fences."""


def _build_adjudication_prompt(
    *,
    item: Mapping[str, Any],
    dialogue: str,
    ideal_text: str,
    generated_rows: Sequence[Dict[str, Any]],
    expert_rows: Sequence[Dict[str, Any]],
) -> str:
    item_type = str(item.get("item_type", ""))
    if item_type == "expert_miss":
        options = ["true_miss", "wording_equivalent", "granularity_difference", "task_mismatch", "generic_boilerplate"]
    else:
        options = ["valid_extra", "wording_equivalent", "granularity_difference", "off_target"]
    return (
        f"ITEM TYPE: {item_type}\n\n"
        "DIALOGUE / TASK:\n"
        f"{_truncate_chars(dialogue, 5000)}\n\n"
        "IDEAL COMPLETION:\n"
        f"{_truncate_chars(ideal_text, 4500)}\n\n"
        "FOCUS ITEM:\n"
        f"{json.dumps(dict(item), ensure_ascii=False, indent=2)}\n\n"
        "EXPERT RUBRICS (example-level):\n"
        f"{json.dumps(list(expert_rows), ensure_ascii=False, indent=2)}\n\n"
        "GENERATED CRITERIA:\n"
        f"{json.dumps(list(generated_rows), ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "category": "<one of the allowed labels>",\n'
        '  "covered_counterpart_indices": [0],\n'
        '  "reason": "<short reason>"\n'
        "}\n"
        f"Allowed labels for this item: {options}"
    )


def _heuristic_adjudication(item: Mapping[str, Any], expert_rows: Sequence[Dict[str, Any]], generated_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if item.get("item_type") == "expert_miss":
        family = str(item.get("family", "other"))
        if family in {row["family"] for row in generated_rows} and family != "other":
            category = "granularity_difference"
        elif family in {"communication_audience", "factual_accuracy_safety"}:
            category = "generic_boilerplate"
        else:
            category = "true_miss"
    else:
        family = str(item.get("family", "other"))
        if family in {row["family"] for row in expert_rows} and family != "other":
            category = "granularity_difference"
        elif family == "other":
            category = "off_target"
        else:
            category = "valid_extra"
    return {
        "category": category,
        "covered_counterpart_indices": [],
        "reason": "heuristic_fallback",
        "fallback": True,
        "parse_error": None,
    }


def _parse_adjudication_payload(raw_text: str, *, allowed_labels: Sequence[str]) -> Dict[str, Any]:
    obj = extract_json_object(raw_text)
    if not obj:
        raise ValueError("Adjudication response did not contain parseable JSON.")
    category = str(obj.get("category", "")).strip()
    if category not in allowed_labels:
        raise ValueError(f"Adjudication category must be one of: {allowed_labels}")
    counterparts = obj.get("covered_counterpart_indices")
    covered = [int(value) for value in counterparts if isinstance(value, int)] if isinstance(counterparts, list) else []
    return {
        "category": category,
        "covered_counterpart_indices": covered,
        "reason": str(obj.get("reason", "")).strip(),
        "fallback": False,
        "parse_error": None,
    }


def adjudicate_disagreement(
    *,
    item: Mapping[str, Any],
    dialogue: str,
    ideal_text: str,
    expert_rows: Sequence[Dict[str, Any]],
    generated_rows: Sequence[Dict[str, Any]],
    model_spec: Optional[ModelSpec],
    router: Optional[LLMRouter],
    cache: Optional[JsonlCache],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    if item.get("item_type") == "expert_miss":
        allowed = ("true_miss", "wording_equivalent", "granularity_difference", "task_mismatch", "generic_boilerplate")
    else:
        allowed = ("valid_extra", "wording_equivalent", "granularity_difference", "off_target")

    if model_spec is None or router is None:
        return _heuristic_adjudication(item, expert_rows, generated_rows), False, None

    cache_hit = False
    raw_text = ""
    payload_for_key = {
        "prompt_version": ADJUDICATION_PROMPT_VERSION,
        "model": f"{model_spec.provider}:{model_spec.model}",
        "item_hash": stable_hash(dict(item)),
        "dialogue_hash": stable_hash(dialogue),
        "ideal_hash": stable_hash(ideal_text),
        "expert_hash": stable_hash(list(expert_rows)),
        "generated_hash": stable_hash(list(generated_rows)),
    }
    cache_key = make_cache_key(ADJUDICATION_PROMPT_VERSION, payload_for_key)
    if cache and cache.enabled:
        cache.load()
        hit = cache.get(cache_key)
        if hit and isinstance(hit.get("raw_response"), str):
            raw_text = hit["raw_response"]
            cache_hit = True

    if not raw_text:
        prompt = _build_adjudication_prompt(
            item=item,
            dialogue=dialogue,
            ideal_text=ideal_text,
            generated_rows=generated_rows,
            expert_rows=expert_rows,
        )
        resp = router.generate(
            model_spec,
            system_prompt=_ADJUDICATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=1200,
        )
        raw_text = resp.raw_text or resp.text
        if cache and cache.enabled:
            cache.set(cache_key, {"raw_response": raw_text})

    try:
        parsed = _parse_adjudication_payload(raw_text, allowed_labels=allowed)
        return parsed, cache_hit, None
    except ValueError as exc:
        fallback = _heuristic_adjudication(item, expert_rows, generated_rows)
        fallback["parse_error"] = str(exc)
        return fallback, cache_hit, str(exc)


def _select_disagreement_items(
    comparison_examples: Sequence[Dict[str, Any]],
    *,
    sample_size: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for example_result in comparison_examples:
        prompt_id = str(example_result.get("prompt_id", ""))
        for row in example_result.get("expert_rows", []):
            match = example_result.get("alignment_lookup", {}).get(("expert", row["expert_index"]))
            if not isinstance(match, dict):
                continue
            if match.get("match_label") == "none":
                items.append(
                    {
                        "prompt_id": prompt_id,
                        "item_type": "expert_miss",
                        "expert_index": row["expert_index"],
                        "criterion": row["criterion"],
                        "points": row["points"],
                        "family": row["family"],
                        "priority": abs(int(row["points"])),
                    }
                )
        for row in example_result.get("generated_rows", []):
            assessment = example_result.get("alignment_lookup", {}).get(("generated", row["generated_index"]))
            if not isinstance(assessment, dict):
                continue
            if assessment.get("precision_label") in {"valid_extra", "off_target"}:
                items.append(
                    {
                        "prompt_id": prompt_id,
                        "item_type": "generated_extra",
                        "generated_index": row["generated_index"],
                        "label": row["label"],
                        "requirement": row["requirement"],
                        "family": row["family"],
                        "priority": 1 if assessment.get("precision_label") == "off_target" else 0,
                    }
                )

    ordered = sorted(
        items,
        key=lambda item: (
            0 if item["item_type"] == "expert_miss" else 1,
            -int(item.get("priority", 0) or 0),
            item.get("prompt_id", ""),
        ),
    )
    selected: List[Dict[str, Any]] = []
    per_prompt: Dict[str, int] = {}
    for item in ordered:
        prompt_id = str(item.get("prompt_id", ""))
        if per_prompt.get(prompt_id, 0) >= 2:
            continue
        selected.append(item)
        per_prompt[prompt_id] = per_prompt.get(prompt_id, 0) + 1
        if len(selected) >= sample_size:
            break
    return selected


def _aggregate_family_summary(per_example_rows: Sequence[Mapping[str, Dict[str, Any]]]) -> Dict[str, Any]:
    totals: Dict[str, Dict[str, Any]] = {}
    for summary in per_example_rows:
        for family, row in summary.items():
            bucket = totals.setdefault(
                family,
                {
                    "expert_count": 0,
                    "expert_weight": 0,
                    "matched_expert_count": 0,
                    "matched_expert_weight": 0,
                    "generated_count": 0,
                },
            )
            for key in bucket:
                bucket[key] += int(row.get(key, 0) or 0)
    for family, row in totals.items():
        expert_weight = int(row["expert_weight"])
        expert_count = int(row["expert_count"])
        row["weighted_recall"] = (row["matched_expert_weight"] / expert_weight) if expert_weight else 0.0
        row["expert_recall"] = (row["matched_expert_count"] / expert_count) if expert_count else 0.0
    return totals


def _aggregate_metrics_by_task_profile(
    example_results: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in example_results:
        profile_id = str(row.get("task_profile_id", "unknown"))
        grouped.setdefault(profile_id, []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for profile_id, rows in grouped.items():
        out[profile_id] = _aggregate_overall_metrics(rows)
    return out


def _aggregate_family_summary_by_task_profile(
    example_results: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Dict[str, Any]]]] = {}
    for row in example_results:
        profile_id = str(row.get("task_profile_id", "unknown"))
        grouped.setdefault(profile_id, []).append(row.get("family_summary", {}))
    return {
        profile_id: _aggregate_family_summary(rows)
        for profile_id, rows in grouped.items()
    }


def _aggregate_overall_metrics(example_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {
        "examples_scored": len(example_results),
        "expert_criteria_total": 0,
        "expert_criteria_matched": 0,
        "expert_direct_matches": 0,
        "expert_partial_matches": 0,
        "generated_criteria_total": 0,
        "generated_aligned": 0,
        "generated_broader_but_valid": 0,
        "generated_valid_extra": 0,
        "generated_off_target": 0,
        "weighted_recall_numerator": 0.0,
        "weighted_recall_denominator": 0.0,
        "polarity_accuracy_numerator": 0.0,
        "polarity_accuracy_denominator": 0.0,
    }
    for example in example_results:
        metrics = example["metrics"]
        totals["expert_criteria_total"] += int(metrics["expert_criteria_total"])
        totals["expert_criteria_matched"] += int(metrics["expert_criteria_matched"])
        totals["expert_direct_matches"] += int(metrics["expert_direct_matches"])
        totals["expert_partial_matches"] += int(metrics["expert_partial_matches"])
        totals["generated_criteria_total"] += int(metrics["generated_criteria_total"])
        totals["generated_aligned"] += int(metrics["generated_aligned"])
        totals["generated_broader_but_valid"] += int(metrics["generated_broader_but_valid"])
        totals["generated_valid_extra"] += int(metrics["generated_valid_extra"])
        totals["generated_off_target"] += int(metrics["generated_off_target"])
        expert_rows = example["expert_rows"]
        alignment = example["alignment"]
        expert_weight_total = sum(abs(int(row["points"])) for row in expert_rows)
        matched_weight = 0
        polarity_total = 0
        polarity_matches = 0
        generated_lookup = {row["generated_index"]: row for row in example["generated_rows"]}
        for match in alignment["expert_matches"]:
            if match["match_label"] not in {"direct", "partial"}:
                continue
            expert = expert_rows[match["expert_index"]]
            matched_weight += abs(int(expert["points"]))
            generated_index = match["best_generated_index"]
            if isinstance(generated_index, int) and generated_index in generated_lookup:
                polarity_total += 1
                if generated_lookup[generated_index]["polarity"] == expert["polarity"]:
                    polarity_matches += 1
        totals["weighted_recall_numerator"] += matched_weight
        totals["weighted_recall_denominator"] += expert_weight_total
        totals["polarity_accuracy_numerator"] += polarity_matches
        totals["polarity_accuracy_denominator"] += polarity_total

    generated_total = totals["generated_criteria_total"]
    expert_total = totals["expert_criteria_total"]
    return {
        "examples_scored": totals["examples_scored"],
        "expert_criteria_total": expert_total,
        "expert_criteria_matched": totals["expert_criteria_matched"],
        "expert_recall": (totals["expert_criteria_matched"] / expert_total) if expert_total else 0.0,
        "expert_direct_matches": totals["expert_direct_matches"],
        "expert_partial_matches": totals["expert_partial_matches"],
        "generated_criteria_total": generated_total,
        "generated_precision": (
            (totals["generated_aligned"] + totals["generated_broader_but_valid"]) / generated_total
            if generated_total
            else 0.0
        ),
        "generated_aligned": totals["generated_aligned"],
        "generated_broader_but_valid": totals["generated_broader_but_valid"],
        "generated_valid_extra": totals["generated_valid_extra"],
        "generated_off_target": totals["generated_off_target"],
        "weighted_recall": (
            totals["weighted_recall_numerator"] / totals["weighted_recall_denominator"]
            if totals["weighted_recall_denominator"]
            else 0.0
        ),
        "polarity_accuracy": (
            totals["polarity_accuracy_numerator"] / totals["polarity_accuracy_denominator"]
            if totals["polarity_accuracy_denominator"]
            else 0.0
        ),
    }


def _metric_deltas(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, float]:
    keys = (
        "weighted_recall",
        "expert_recall",
        "generated_precision",
        "expert_direct_matches",
        "generated_off_target",
        "polarity_accuracy",
    )
    out: Dict[str, float] = {}
    for key in keys:
        try:
            out[key] = float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def run_healthbench_evaluation(
    *,
    dataset_path: Path,
    run_dir: Path,
    start: int,
    limit: int,
    discovery_model_override: Optional[str],
    alignment_model_override: Optional[str],
    adjudication_model_override: Optional[str],
    use_cache: bool,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    disagreement_sample_size: int,
    gold_provider: str = "healthbench",
    refine_iterations: int = 1,
    apply_calibration: Optional[Path] = None,
    emit_calibration_hints: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    dataset_path = Path(dataset_path)
    run_dir = Path(run_dir)
    if gold_provider != "healthbench":
        raise ValueError(f"Unsupported gold provider: {gold_provider}")
    provider = HealthBenchGoldProvider()
    calibration_input_path = Path(apply_calibration) if apply_calibration is not None else None
    calibration_hints = _load_calibration_hints(calibration_input_path)

    subset_dir = run_dir / "subset"
    routing_dir = run_dir / "routing"
    discovery_dir = run_dir / "discovery"
    discovery_examples_dir = discovery_dir / "examples"
    comparison_dir = run_dir / "comparison"
    comparison_examples_dir = comparison_dir / "examples"
    summaries_dir = run_dir / "summaries"
    refinement_dir = run_dir / "refinement"
    refinement_examples_dir = refinement_dir / "examples"
    calibration_dir = run_dir / "calibration"
    cache_dir = run_dir / "cache"

    for directory in (
        subset_dir,
        routing_dir,
        discovery_dir,
        discovery_examples_dir,
        comparison_dir,
        comparison_examples_dir,
        summaries_dir,
        refinement_dir,
        refinement_examples_dir,
        calibration_dir,
        cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    metadata, examples = load_healthbench_dataset(dataset_path)
    sliced_examples = list(examples[start:])
    if limit > 0:
        sliced_examples = sliced_examples[:limit]

    note_selected_examples, note_decisions, note_subset_summary = select_healthbench_subset(sliced_examples)
    write_json(subset_dir / "summary.json", note_subset_summary)
    write_json(subset_dir / "decisions.json", {"schema": "healthbench_subset_decisions_v1", "decisions": note_decisions})
    write_json(
        subset_dir / "selected_examples.json",
        {
            "schema": "healthbench_selected_examples_v1",
            "selected_prompt_ids": [example.prompt_id for example in note_selected_examples],
        },
    )

    selected_examples, routing_decisions, routing_summary = route_healthbench_examples(sliced_examples)
    write_json(routing_dir / "summary.json", routing_summary)
    write_json(
        routing_dir / "decisions.json",
        {"schema": "healthbench_routing_decisions_v1", "decisions": routing_decisions},
    )
    write_json(
        routing_dir / "selected_examples.json",
        {
            "schema": "healthbench_routed_examples_v1",
            "selected_prompt_ids": [example.prompt_id for example in selected_examples],
        },
    )

    if not selected_examples:
        raise ValueError("HealthBench routing selected no runnable examples.")

    decision_by_prompt = {decision.prompt_id: decision for decision in routing_decisions}
    discovery_spec = resolve_compiled_judge_spec(discovery_model_override)
    alignment_spec = _resolve_alignment_spec(alignment_model_override)
    adjudication_spec = _resolve_alignment_spec(adjudication_model_override)
    router = LLMRouter()

    discovery_cache = JsonlCache(cache_dir / "healthbench_discovery.jsonl", enabled=use_cache)
    alignment_cache = JsonlCache(cache_dir / "healthbench_alignment.jsonl", enabled=use_cache)
    adjudication_cache = JsonlCache(cache_dir / "healthbench_adjudication.jsonl", enabled=use_cache)
    recursive_config = RecursiveDiscoveryConfig()

    discovery_stats = {
        "examples_selected": len(selected_examples),
        "task_profile_counts": dict(routing_summary.get("task_profile_counts", {})),
        "note_regression_selected": int(note_subset_summary["counts"]["selected"]),
        "pairs_total": 0,
        "pairs_succeeded": 0,
        "pairs_failed_parse": 0,
        "cache_hits": 0,
        "local_proposals_total": 0,
        "local_proposals_promoted": 0,
        "local_proposals_rejected_grounding": 0,
        "generated_canonical_total": 0,
        "generated_refined_total": 0,
        "calibration_guidance_examples": 0,
        "recursive_calls": 0,
        "recursive_cache_hits": 0,
        "recursive_parse_failures": 0,
        "recursive_parents_considered": 0,
        "recursive_parents_expanded": 0,
        "recursive_children_raw_total": 0,
        "recursive_children_promoted": 0,
        "recursive_children_rejected_grounding": 0,
        "examples_with_recursive_change": 0,
    }
    alignment_stats = {"cache_hits": 0, "parse_fallbacks": 0}
    refinement_stats = {
        "gold_provider": provider.provider_id,
        "refine_iterations_requested": max(0, refine_iterations),
        "examples_considered": 0,
        "examples_changed": 0,
        "iterations_accepted": 0,
        "rows_added": 0,
        "rows_dropped": 0,
        "calibration_input_path": str(calibration_input_path) if calibration_input_path else None,
        "calibration_guidance_loaded": bool(calibration_hints),
        "heuristic_post_rescore_examples": 0,
        "post_refine_realign_examples": 0,
        "post_refine_realign_cache_hits": 0,
        "post_refine_realign_parse_fallbacks": 0,
    }
    pre_comparison_examples: List[Dict[str, Any]] = []
    comparison_examples: List[Dict[str, Any]] = []
    pre_family_rows: List[Dict[str, Dict[str, Any]]] = []
    family_rows: List[Dict[str, Dict[str, Any]]] = []
    pre_granularity_reports: List[Dict[str, Any]] = []
    post_granularity_reports: List[Dict[str, Any]] = []

    for example in selected_examples:
        decision = decision_by_prompt[example.prompt_id]
        profile = get_task_profile(decision.task_profile_id)
        calibration_guidance = build_prompt_calibration_guidance(
            calibration_hints,
            task_profile_id=decision.task_profile_id,
        )
        if calibration_guidance.strip():
            discovery_stats["calibration_guidance_examples"] += 1
        example_record = healthbench_to_example_record(example, routing_decision=decision)
        strong, weak_candidates = build_healthbench_candidates(
            example,
            example_id=example_record.example_id,
            task_profile_id=decision.task_profile_id,
            task_family_id=decision.task_family_id,
            artifact_kind=decision.artifact_kind,
        )
        if strong is None or not weak_candidates:
            continue
        if max_pairs_per_example is not None:
            weak_candidates = weak_candidates[: max_pairs_per_example]

        example_local_rows: List[Dict[str, Any]] = []
        pairs_payload: List[Dict[str, Any]] = []
        example_recursion = {
            "recursive_calls": 0,
            "recursive_cache_hits": 0,
            "recursive_parse_failures": 0,
            "recursive_parents_considered": 0,
            "recursive_parents_expanded": 0,
            "recursive_children_raw_total": 0,
            "recursive_children_promoted": 0,
            "recursive_children_rejected_grounding": 0,
            "pairs_with_recursive_change": 0,
        }
        example_recursive_changed = False
        for weak in weak_candidates:
            discovery_stats["pairs_total"] += 1
            pair_result = discover_pair_criteria(
                example=example_record,
                strong=strong,
                weak=weak,
                model_spec=discovery_spec,
                router=router,
                cache=discovery_cache,
                max_criteria=max_criteria,
                task_profile_id=decision.task_profile_id,
                artifact_label=profile.artifact_label,
                calibration_guidance=calibration_guidance,
                recursive_config=recursive_config,
            )
            if pair_result["cache"] == "hit":
                discovery_stats["cache_hits"] += 1
            if pair_result["parse_error"]:
                discovery_stats["pairs_failed_parse"] += 1
            else:
                discovery_stats["pairs_succeeded"] += 1
            discovery_stats["local_proposals_total"] += int(pair_result["raw_proposals_total"])
            discovery_stats["local_proposals_promoted"] += int(pair_result["promoted_proposals_total"])
            discovery_stats["local_proposals_rejected_grounding"] = discovery_stats.get(
                "local_proposals_rejected_grounding", 0
            ) + int(pair_result["rejected_proposals_total"])
            recursion_stats = dict(pair_result.get("recursion", {}))
            for key in (
                "recursive_calls",
                "recursive_cache_hits",
                "recursive_parse_failures",
                "recursive_parents_considered",
                "recursive_parents_expanded",
                "recursive_children_raw_total",
                "recursive_children_promoted",
                "recursive_children_rejected_grounding",
            ):
                value = int(recursion_stats.get(key, 0) or 0)
                discovery_stats[key] += value
                example_recursion[key] += value
            if bool(recursion_stats.get("changed_structure")):
                example_recursive_changed = True
                example_recursion["pairs_with_recursive_change"] += 1
            example_local_rows.extend(pair_result["proposals"])
            pairs_payload.append(
                {
                    "pair_id": pair_result["pair_id"],
                    "strong_candidate_id": strong.candidate_id,
                    "weak_candidate_id": weak.candidate_id,
                    "strong_source_label": strong.source_label,
                    "weak_source_label": weak.source_label,
                    "cache": pair_result["cache"],
                    "parse_error": pair_result["parse_error"],
                    "grounding": pair_result["grounding"],
                    "raw_proposals": pair_result["raw_proposals"],
                    "promoted_root_proposals": pair_result["promoted_root_proposals"],
                    "rejected_root_proposals": pair_result["rejected_root_proposals"],
                    "proposals": pair_result["proposals"],
                    "rejected_proposals": pair_result["rejected_proposals"],
                    "recursive_steps": pair_result["recursive_steps"],
                    "recursion": pair_result["recursion"],
                }
            )
        if example_recursive_changed:
            discovery_stats["examples_with_recursive_change"] += 1

        merged = merge_proposal_entries(example_local_rows)
        pre_generated_rows = _annotate_generated_rows(
            merged["canonical_proposals"],
            task_profile_id=decision.task_profile_id,
            calibration_hints=calibration_hints,
        )
        discovery_stats["generated_canonical_total"] += len(pre_generated_rows)
        discovery_artifact = {
            "schema": "compiled_healthbench_discovery_example_v2",
            "prompt_id": example.prompt_id,
            "routing_decision": decision,
            "calibration_guidance": calibration_guidance or None,
            "recursive_discovery": {
                **example_recursion,
                "changed_structure": example_recursive_changed,
            },
            "strong_anchor": {
                "candidate_id": strong.candidate_id,
                "source_label": strong.source_label,
            },
            "pairs": pairs_payload,
            "generated_canonical_criteria": pre_generated_rows,
            "merged_local_summary": {
                "unique_canonical_count": merged["unique_canonical_count"],
                "total_local_proposals": merged["total_local_proposals"],
            },
        }
        write_json(discovery_examples_dir / f"{example.prompt_id}.json", discovery_artifact)

        pre_result = provider.compare_generated_rows(
            example=example,
            generated_rows=pre_generated_rows,
            model_spec=alignment_spec,
            router=router if alignment_spec is not None else None,
            cache=alignment_cache,
        )
        if pre_result.cache_hit:
            alignment_stats["cache_hits"] += 1
        if pre_result.parse_error:
            alignment_stats["parse_fallbacks"] += 1

        pre_granularity = classify_granularity_gaps(
            provider_id=provider.provider_id,
            prompt_id=example.prompt_id,
            task_profile_id=decision.task_profile_id,
            gold_rows=pre_result.gold_rows,
            generated_rows=pre_generated_rows,
            alignment=pre_result.alignment,
        )
        pre_granularity_reports.append(pre_granularity)
        pre_family_rows.append(pre_result.family_summary)

        working_generated_rows = pre_generated_rows
        working_result = pre_result
        working_granularity = pre_granularity
        refinement_outputs: List[Dict[str, Any]] = []
        accepted_actions: List[Any] = []
        post_alignment_source = "pre_refinement_alignment"
        post_realign_reasons: List[str] = []
        if refine_iterations > 0:
            refinement_stats["examples_considered"] += 1
        for _ in range(max(0, refine_iterations)):
            refinement_output = refine_generated_rows(
                prompt_id=example.prompt_id,
                task_profile_id=decision.task_profile_id,
                gold_rows=working_result.gold_rows,
                generated_rows=working_generated_rows,
                alignment=working_result.alignment,
                granularity_report=working_granularity,
                calibration_hints=calibration_hints,
            )
            refinement_outputs.append(refinement_output)
            if not bool(refinement_output.get("changed")):
                break
            candidate_generated_rows = _annotate_generated_rows(
                refinement_output["generated_rows"],
                task_profile_id=decision.task_profile_id,
                calibration_hints=calibration_hints,
            )
            working_generated_rows = candidate_generated_rows
            working_result = provider.compare_generated_rows(
                example=example,
                generated_rows=working_generated_rows,
                model_spec=None,
                router=None,
                cache=None,
                use_heuristic_only=True,
            )
            preserved_alignment = _preserve_pre_refinement_matches(
                pre_alignment=pre_result.alignment,
                post_alignment=working_result.alignment,
                post_generated_rows=working_generated_rows,
            )
            working_result = GoldAlignmentArtifact(
                provider_id=working_result.provider_id,
                gold_rows=working_result.gold_rows,
                alignment=preserved_alignment,
                metrics=_compute_example_metrics(
                    expert_rows=working_result.gold_rows,
                    generated_rows=working_generated_rows,
                    alignment=preserved_alignment,
                ),
                family_summary=_family_coverage_rows(
                    expert_rows=working_result.gold_rows,
                    generated_rows=working_generated_rows,
                    alignment=preserved_alignment,
                ),
                alignment_lookup=build_alignment_lookup(preserved_alignment),
                cache_hit=working_result.cache_hit,
                parse_error=working_result.parse_error,
                metadata={**working_result.metadata, "preserved_pre_refinement_matches": True},
            )
            working_granularity = classify_granularity_gaps(
                provider_id=provider.provider_id,
                prompt_id=example.prompt_id,
                task_profile_id=decision.task_profile_id,
                gold_rows=working_result.gold_rows,
                generated_rows=working_generated_rows,
                alignment=working_result.alignment,
            )
            refinement_stats["iterations_accepted"] += 1
            accepted_actions.extend(refinement_output.get("actions", []))

        if accepted_actions:
            refinement_stats["examples_changed"] += 1
            refinement_stats["heuristic_post_rescore_examples"] += 1
            post_alignment_source = "heuristic_post_refinement"
            refinement_stats["rows_added"] += sum(
                1 for action in accepted_actions if getattr(action, "kind", "") == "add_gold_aligned_row"
            )
            refinement_stats["rows_dropped"] += sum(
                1 for action in accepted_actions if getattr(action, "kind", "") == "drop_generated_row"
            )
            post_realign_reasons = _post_refine_realign_reasons(
                recursive_structure_changed=example_recursive_changed,
                granularity_report=working_granularity,
            )
            if alignment_spec is not None and post_realign_reasons:
                realigned_result = provider.compare_generated_rows(
                    example=example,
                    generated_rows=working_generated_rows,
                    model_spec=alignment_spec,
                    router=router,
                    cache=alignment_cache,
                )
                refinement_stats["post_refine_realign_examples"] += 1
                if realigned_result.cache_hit:
                    refinement_stats["post_refine_realign_cache_hits"] += 1
                if realigned_result.parse_error:
                    refinement_stats["post_refine_realign_parse_fallbacks"] += 1
                preserved_alignment = _preserve_pre_refinement_matches(
                    pre_alignment=pre_result.alignment,
                    post_alignment=realigned_result.alignment,
                    post_generated_rows=working_generated_rows,
                )
                working_result = GoldAlignmentArtifact(
                    provider_id=realigned_result.provider_id,
                    gold_rows=realigned_result.gold_rows,
                    alignment=preserved_alignment,
                    metrics=_compute_example_metrics(
                        expert_rows=realigned_result.gold_rows,
                        generated_rows=working_generated_rows,
                        alignment=preserved_alignment,
                    ),
                    family_summary=_family_coverage_rows(
                        expert_rows=realigned_result.gold_rows,
                        generated_rows=working_generated_rows,
                        alignment=preserved_alignment,
                    ),
                    alignment_lookup=build_alignment_lookup(preserved_alignment),
                    cache_hit=realigned_result.cache_hit,
                    parse_error=realigned_result.parse_error,
                    metadata={
                        **realigned_result.metadata,
                        "preserved_pre_refinement_matches": True,
                        "post_refine_realign": True,
                        "post_realign_reasons": list(post_realign_reasons),
                    },
                )
                working_granularity = classify_granularity_gaps(
                    provider_id=provider.provider_id,
                    prompt_id=example.prompt_id,
                    task_profile_id=decision.task_profile_id,
                    gold_rows=working_result.gold_rows,
                    generated_rows=working_generated_rows,
                    alignment=working_result.alignment,
                )
                post_alignment_source = "full_post_refinement_realign"

        post_result = working_result
        post_generated_rows = working_generated_rows
        post_granularity = working_granularity
        discovery_stats["generated_refined_total"] += len(post_generated_rows)
        family_rows.append(post_result.family_summary)
        post_granularity_reports.append(post_granularity)

        refinement_artifact = {
            "schema": "compiled_healthbench_refinement_example_v1",
            "prompt_id": example.prompt_id,
            "routing_decision": decision,
            "recursive_discovery": {
                **example_recursion,
                "changed_structure": example_recursive_changed,
            },
            "pre_refinement": {
                "expert_rows": pre_result.gold_rows,
                "generated_rows": pre_generated_rows,
                "alignment": pre_result.alignment,
                "metrics": pre_result.metrics,
                "family_summary": pre_result.family_summary,
                "granularity_report": pre_granularity,
                "alignment_source": "pre_refinement_alignment",
                "alignment_metadata": dict(pre_result.metadata),
            },
            "post_refinement": {
                "expert_rows": post_result.gold_rows,
                "generated_rows": post_generated_rows,
                "alignment": post_result.alignment,
                "metrics": post_result.metrics,
                "family_summary": post_result.family_summary,
                "granularity_report": post_granularity,
                "alignment_source": post_alignment_source,
                "alignment_metadata": dict(post_result.metadata),
                "post_realign_reasons": list(post_realign_reasons),
            },
            "iterations": refinement_outputs,
        }
        write_json(refinement_examples_dir / f"{example.prompt_id}.json", refinement_artifact)
        write_json(comparison_examples_dir / f"{example.prompt_id}.json", refinement_artifact)

        pre_comparison_examples.append(
            {
                "prompt_id": example.prompt_id,
                "dialogue": example.dialogue,
                "ideal_text": strong.text,
                "task_profile_id": decision.task_profile_id,
                "task_family_id": decision.task_family_id,
                "note_regression_selected": decision.note_regression_selected,
                "expert_rows": pre_result.gold_rows,
                "generated_rows": pre_generated_rows,
                "alignment": pre_result.alignment,
                "alignment_lookup": pre_result.alignment_lookup,
                "metrics": pre_result.metrics,
                "family_summary": pre_result.family_summary,
                "recursive_discovery": {
                    **example_recursion,
                    "changed_structure": example_recursive_changed,
                },
            }
        )

        comparison_examples.append(
            {
                "prompt_id": example.prompt_id,
                "dialogue": example.dialogue,
                "ideal_text": strong.text,
                "task_profile_id": decision.task_profile_id,
                "task_family_id": decision.task_family_id,
                "note_regression_selected": decision.note_regression_selected,
                "expert_rows": post_result.gold_rows,
                "generated_rows": post_generated_rows,
                "alignment": post_result.alignment,
                "alignment_lookup": post_result.alignment_lookup,
                "metrics": post_result.metrics,
                "family_summary": post_result.family_summary,
                "recursive_discovery": {
                    **example_recursion,
                    "changed_structure": example_recursive_changed,
                },
                "post_alignment_source": post_alignment_source,
                "post_alignment_metadata": dict(post_result.metadata),
                "post_realign_reasons": list(post_realign_reasons),
                "pre_refinement_metrics": pre_result.metrics,
                "pre_refinement_generated_rows": pre_generated_rows,
                "pre_refinement_alignment": pre_result.alignment,
                "pre_refinement_family_summary": pre_result.family_summary,
                "pre_refinement_granularity": pre_granularity,
                "granularity": post_granularity,
                "refinement_actions": accepted_actions,
            }
        )

    disagreement_items = _select_disagreement_items(
        comparison_examples,
        sample_size=max(0, disagreement_sample_size),
    )
    adjudications: List[Dict[str, Any]] = []
    adjudication_category_counts: Dict[str, int] = {}
    by_prompt = {row["prompt_id"]: row for row in comparison_examples}
    adjudication_cache_hits = 0
    adjudication_parse_fallbacks = 0
    for item in disagreement_items:
        prompt_id = str(item.get("prompt_id", ""))
        example_row = by_prompt.get(prompt_id)
        if example_row is None:
            continue
        result, hit, parse_error = adjudicate_disagreement(
            item=item,
            dialogue=example_row["dialogue"],
            ideal_text=example_row["ideal_text"],
            expert_rows=example_row["expert_rows"],
            generated_rows=example_row["generated_rows"],
            model_spec=adjudication_spec,
            router=router if adjudication_spec is not None else None,
            cache=adjudication_cache,
        )
        if hit:
            adjudication_cache_hits += 1
        if parse_error:
            adjudication_parse_fallbacks += 1
        adjudications.append(
            {
                "prompt_id": prompt_id,
                "item": item,
                "adjudication": result,
            }
        )
        category = str(result.get("category", "unresolved"))
        adjudication_category_counts[category] = adjudication_category_counts.get(category, 0) + 1

    pre_alignment_metrics = _aggregate_overall_metrics(pre_comparison_examples)
    post_alignment_metrics = _aggregate_overall_metrics(comparison_examples)
    pre_note_regression_metrics = _aggregate_overall_metrics(
        [row for row in pre_comparison_examples if bool(row.get("note_regression_selected"))]
    )
    post_note_regression_metrics = _aggregate_overall_metrics(
        [row for row in comparison_examples if bool(row.get("note_regression_selected"))]
    )
    granularity_report = {
        "schema": "compiled_healthbench_granularity_report_v1",
        "pre_refinement": aggregate_granularity_reports(pre_granularity_reports),
        "post_refinement": aggregate_granularity_reports(post_granularity_reports),
    }
    calibration_hints_payload = derive_calibration_hints(
        pre_granularity_reports,
        provider_id=provider.provider_id,
        existing_hints=calibration_hints,
    )
    calibration_hints_path = calibration_dir / "calibration_hints.json"
    if emit_calibration_hints:
        write_json(calibration_hints_path, calibration_hints_payload)
    pre_post_summary = {
        "schema": "compiled_healthbench_pre_post_summary_v1",
        "pre_refinement": pre_alignment_metrics,
        "post_refinement": post_alignment_metrics,
        "delta": _metric_deltas(pre_alignment_metrics, post_alignment_metrics),
        "note_regression_pre_refinement": pre_note_regression_metrics,
        "note_regression_post_refinement": post_note_regression_metrics,
        "note_regression_delta": _metric_deltas(pre_note_regression_metrics, post_note_regression_metrics),
    }

    family_coverage = {
        "schema": "compiled_healthbench_family_coverage_v2",
        "pre_refinement": {
            "families": _aggregate_family_summary(pre_family_rows),
            "by_task_profile": _aggregate_family_summary_by_task_profile(pre_comparison_examples),
        },
        "post_refinement": {
            "families": _aggregate_family_summary(family_rows),
            "by_task_profile": _aggregate_family_summary_by_task_profile(comparison_examples),
        },
    }
    disagreement_summary = {
        "schema": "compiled_healthbench_disagreement_sample_v1",
        "sample_size": len(adjudications),
        "category_counts": adjudication_category_counts,
        "items": adjudications,
    }
    write_json(summaries_dir / "family_coverage.json", family_coverage)
    write_json(summaries_dir / "disagreement_sample.json", disagreement_summary)
    write_json(run_dir / "granularity_report.json", granularity_report)
    write_json(run_dir / "pre_post_comparison.json", pre_post_summary)

    run_summary = {
        "schema": "compiled_healthbench_eval_summary_v2",
        "dataset_path": str(dataset_path),
        "params": {
            "start": start,
            "limit": limit,
            "max_pairs_per_example": max_pairs_per_example,
            "max_criteria": max_criteria,
            "disagreement_sample_size": disagreement_sample_size,
            "cache_enabled": use_cache,
            "gold_provider": gold_provider,
            "refine_iterations": refine_iterations,
            "apply_calibration": str(calibration_input_path) if calibration_input_path else None,
            "emit_calibration_hints": emit_calibration_hints,
            "recursive_config": {
                "enabled": recursive_config.enabled,
                "max_depth": recursive_config.max_depth,
                "max_recursive_parents_per_pair": recursive_config.max_recursive_parents_per_pair,
                "max_children_per_parent": recursive_config.max_children_per_parent,
                "max_recursive_calls_per_pair": recursive_config.max_recursive_calls_per_pair,
            },
        },
        "dataset_metadata": metadata,
        "routing": routing_summary,
        "subset": note_subset_summary,
        "discovery": discovery_stats,
        "pre_refinement_alignment": {
            **pre_alignment_metrics,
            **alignment_stats,
        },
        "alignment": {
            **post_alignment_metrics,
            **alignment_stats,
        },
        "pre_refinement_alignment_by_task_profile": _aggregate_metrics_by_task_profile(pre_comparison_examples),
        "alignment_by_task_profile": _aggregate_metrics_by_task_profile(comparison_examples),
        "pre_refinement_note_regression_alignment": pre_note_regression_metrics,
        "note_regression_alignment": post_note_regression_metrics,
        "refinement": {
            **refinement_stats,
            "pre_to_post_delta": _metric_deltas(pre_alignment_metrics, post_alignment_metrics),
            "note_regression_delta": _metric_deltas(pre_note_regression_metrics, post_note_regression_metrics),
        },
        "granularity": {
            "pre_refinement": granularity_report["pre_refinement"]["gap_counts"],
            "post_refinement": granularity_report["post_refinement"]["gap_counts"],
        },
        "calibration": {
            "provider_id": provider.provider_id,
            "input_loaded": bool(calibration_hints),
            "input_path": str(calibration_input_path) if calibration_input_path else None,
            "eligible_profiles": calibration_enabled_profiles(calibration_hints),
            "hints_emitted": emit_calibration_hints,
            "hints_path": str(calibration_hints_path) if emit_calibration_hints else None,
        },
        "adjudication": {
            "sample_size": len(adjudications),
            "cache_hits": adjudication_cache_hits,
            "parse_fallbacks": adjudication_parse_fallbacks,
            "category_counts": adjudication_category_counts,
        },
        "paths": {
            "routing": str(routing_dir),
            "subset": str(subset_dir),
            "discovery": str(discovery_dir),
            "comparison": str(comparison_dir),
            "refinement": str(refinement_dir),
            "summaries": str(summaries_dir),
            "granularity_report": str(run_dir / "granularity_report.json"),
            "pre_post_comparison": str(run_dir / "pre_post_comparison.json"),
            "calibration_hints": str(calibration_hints_path) if emit_calibration_hints else None,
        },
    }
    write_json(run_dir / "run_summary.json", run_summary)
    return run_dir, run_summary


__all__ = [
    "FILTER_VERSION",
    "ROUTING_VERSION",
    "ALIGNMENT_PROMPT_VERSION",
    "ADJUDICATION_PROMPT_VERSION",
    "HealthBenchCompletion",
    "HealthBenchExpertCriterion",
    "HealthBenchExample",
    "SubsetDecision",
    "HealthBenchRoutingDecision",
    "classify_task_compatibility",
    "route_healthbench_task",
    "route_healthbench_examples",
    "load_healthbench_dataset",
    "select_healthbench_subset",
    "infer_criterion_polarity",
    "map_expert_rubric_family",
    "map_generated_criterion_family",
    "run_healthbench_evaluation",
]
