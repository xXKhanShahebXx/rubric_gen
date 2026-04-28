"""
Starter local rubric discovery from strong-vs-weak note pairs (compiled rubric MVP scaffold).

Uses contrast candidates from ``mutations`` and one LLM call per pair to propose small sets of
atomic criteria, with an optional shallow recursive decomposition pass for broad promoted criteria.
Does not update the compiled ontology automatically.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.contrast_strategies import (
    build_task_contrast_candidates,
    is_synthetic_candidate,
    mutation_grounding_profiles_for_profile,
)
from rubric_gen.compiled.llm_judge import resolve_compiled_judge_spec
from rubric_gen.compiled.profile_bootstrap import resolve_or_bootstrap_task_profile
from rubric_gen.compiled.serialize import write_json
from rubric_gen.compiled.task_profiles import get_task_profile
from rubric_gen.dataio import example_to_prompt
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec

DISCOVERY_PROMPT_VERSION = "compiled_discovery_v9"
DISCOVERY_DECOMPOSITION_PROMPT_VERSION = "compiled_discovery_decomposition_v1"


@dataclass(frozen=True)
class RecursiveDiscoveryConfig:
    enabled: bool = True
    max_depth: int = 1
    max_recursive_parents_per_pair: int = 2
    max_children_per_parent: int = 3
    max_recursive_calls_per_pair: int = 2

_SYSTEM_PROMPT = """You are helping design evaluation criteria for clinical dialogue-to-note quality.
You compare a stronger reference note to a weaker alternative for the SAME encounter.

Rules:
- Ground differences only in the transcript and the two notes; do not invent patient facts.
- Only propose criteria supported by a visible difference between the stronger and weaker note.
- If the weaker note came from a narrow synthetic mutation, stay tightly scoped to that observed mutation.
- Propose SMALL, ATOMIC criteria: each criterion checks one clear documentation behavior.
- Prefer binary-leaning checks (easy to verify yes/no from the note) over vague rubric language.
- Each criterion must be self-contained (readable without the other note).
- Use dimensions that match documentation quality (e.g. structure, symptom_detail, pertinent_negatives,
  study_results, follow_up_specificity, return_precautions, medication_management,
  intervention_plan, testing_plan, diagnostic_reasoning, certainty_language) —
  pick the best fit per criterion.
- severity_tier is a short label such as: hard_gate, high, medium, low.

Return a single JSON object as specified in the user message. No markdown fences."""


def _system_prompt_for_profile(task_profile_id: str) -> str:
    profile = get_task_profile(task_profile_id)
    archetype_id = profile.parent_profile_id or profile.task_profile_id
    if archetype_id == "note_documentation":
        return _SYSTEM_PROMPT
    quality = profile.discovery_context or "instruction-following response quality"
    artifact_label = profile.artifact_label or "response"
    dimensions = ", ".join(profile.discovery_dimensions or ("instruction_adherence", "grounding", "completeness"))
    lowered_quality = quality.lower()
    extra_rules: List[str] = []
    if any(token in lowered_quality for token in ("logic-puzzle", "multiple-choice", "mathematical", "exact-answer")):
        extra_rules.append(
            "- Prefer criteria that distinguish the correct final answer or conclusion, not only wrapper formatting."
        )
    if any(token in lowered_quality for token in ("logic-puzzle", "constraint-following", "constraint")):
        extra_rules.append("- Prefer criteria for clue consistency, contradiction avoidance, and satisfying stated constraints.")
    if any(token in lowered_quality for token in ("mathematical", "multiple-choice", "exact-answer")):
        extra_rules.append("- Prefer criteria for the exact required option/value and the requested final-answer syntax.")
    if "code-solution" in lowered_quality:
        extra_rules.append("- Prefer executable behavior, I/O, and constraint correctness over implementation style.")
        extra_rules.append(
            "- Do not propose criteria that merely prefer one implementation technique over another "
            "(for example explicit zero checks, XOR vs arithmetic, or a particular pruning strategy) unless "
            "that technique is required by the prompt or clearly changes behavior under the stated constraints."
        )
    extra_rules_text = "\n".join(extra_rules)
    if extra_rules_text:
        extra_rules_text = f"\n{extra_rules_text}"
    return f"""You are helping design evaluation criteria for {quality}.
You compare a stronger reference {artifact_label} to a weaker alternative for the SAME task instance.

Rules:
- Ground differences only in the task context and the two artifacts; do not invent facts.
- Only propose criteria supported by a visible difference between the stronger and weaker artifact.
- If the weaker artifact came from a narrow synthetic mutation, stay tightly scoped to that observed mutation.
- Propose SMALL, ATOMIC criteria: each criterion checks one clear behavior.
- Prefer binary-leaning checks (easy to verify yes/no from a single artifact) over vague rubric language.
- Each criterion must be self-contained (readable without the other artifact).
- Use dimensions that match this task family ({dimensions}) and pick the best fit per criterion.
- severity_tier is a short label such as: hard_gate, high, medium, low.
{extra_rules_text}

Return a single JSON object as specified in the user message. No markdown fences."""

_WORD_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_STRIP_SYMPTOM_ALLOWED_PHRASES = (
    "pain",
    "pain characteristics",
    "pain quality",
    "location",
    "quadrant",
    "severity",
    "duration",
    "timing",
    "trigger",
    "meal",
    "postprandial",
    "fever",
    "fevers",
    "afebrile",
    "temperature",
    "constitutional",
)
_STRIP_SYMPTOM_BLOCKED_PHRASES = (
    "nausea",
    "vomiting",
    "vomit",
    "denies vomiting",
    "pertinent negative",
    "all vital signs",
    "all relevant vital signs",
    "oxygen saturation",
    "heart rate",
    "respiratory rate",
    "blood pressure",
    "bowel sounds",
    "peritoneal",
    "murphy",
    "guarding",
    "rebound",
    "physical exam",
    "exam finding",
    "exam findings",
    "abdominal exam",
)
_DROP_STUDY_ALLOWED_PHRASES = (
    "ultrasound",
    "imaging",
    "study",
    "studies",
    "test",
    "tests",
    "result",
    "results",
    "finding",
    "findings",
    "reviewed",
    "outside facility",
    "gallstones",
    "gallbladder",
    "common bile duct",
    "bile duct",
)
_DROP_STUDY_BLOCKED_PHRASES = (
    "patient education",
    "education",
    "counseling",
    "surgery",
    "surgical",
    "laparoscopic",
    "preoperative",
    "intraoperative",
    "postoperative",
    "diabetes",
    "follow up",
    "follow-up",
    "instruction",
    "instructions",
    "medical treatment",
)
_DROP_FOLLOWUP_ALLOWED_PHRASES = (
    "follow-up",
    "follow up",
    "follow-up timing",
    "follow up timing",
    "follow-up interval",
    "follow-up appointment",
    "return to clinic",
    "recheck",
    "reevaluation",
    "follow with",
    "appointment",
    "surveillance",
    "outpatient follow-up",
)
_DROP_FOLLOWUP_BLOCKED_PHRASES = (
    "return precautions",
    "return if",
    "return sooner",
    "seek care",
    "worsening symptoms",
    "call the office if",
    "call if",
    "red flags",
    "urgent care",
    "medication dose",
    "medication dosing",
    "anticoagulant",
    "antibiotic",
    "ultrasound",
    "imaging",
    "procedure",
    "surgery",
    "testing",
    "lab order",
    "patient education",
    "counseling",
    "understands and agrees",
    "agreement",
    "monitoring",
    "daily weight",
    "blood pressure",
    "patient portal",
    "provider will be in touch",
    "pending test results",
    "test results",
    "follow up on test results",
    "follow-up on test results",
)
_DROP_RETURN_ALLOWED_PHRASES = (
    "return precautions",
    "return if",
    "return sooner",
    "seek care",
    "seek medical attention",
    "call the office if",
    "call if",
    "worsening symptoms",
    "red flags",
    "urgent care",
    "emergency room",
    "call 911",
)
_DROP_RETURN_BLOCKED_PHRASES = (
    "follow-up timing",
    "follow up timing",
    "follow-up appointment",
    "follow up appointment",
    "recheck",
    "surveillance",
    "monitoring",
    "medication",
    "procedure",
    "testing",
    "lab order",
)
_DROP_MEDICATION_ALLOWED_PHRASES = (
    "medication",
    "medications",
    "dose",
    "dosage",
    "dosing",
    "frequency",
    "refill",
    "prescription",
    "drug",
    "continue",
    "start",
    "stop",
    "increase",
    "decrease",
    "adjust",
    "inhaler",
    "injection",
    "anticoagulant",
    "antibiotic",
    "lisinopril",
    "metformin",
    "tylenol",
    "albuterol",
    "lasix",
)
_DROP_MEDICATION_BLOCKED_PHRASES = (
    "follow-up",
    "follow up",
    "return precautions",
    "return if",
    "procedure",
    "surgery",
    "referral",
    "physical therapy",
    "testing",
    "lab",
    "imaging",
    "patient education",
    "counseling",
    "understands and agrees",
    "agreement",
    "monitoring",
    "daily weight",
    "blood pressure",
    "patient portal",
    "endocrinologist",
    "primary care provider",
)
_DROP_PROCEDURE_ALLOWED_PHRASES = (
    "procedure",
    "surgery",
    "surgical",
    "intervention",
    "referral",
    "physical therapy",
    "brace",
    "splint",
    "crutches",
    "aircast",
    "device",
    "egd",
    "endoscopy",
    "ablation",
    "cholecystectomy",
    "laparoscopic",
)
_DROP_PROCEDURE_BLOCKED_PHRASES = (
    "follow-up",
    "follow up",
    "return precautions",
    "return if",
    "medication",
    "dose",
    "dosage",
    "testing",
    "lab",
    "imaging",
    "patient education",
    "counseling",
    "understands and agrees",
    "agreement",
    "discussed with the patient",
    "discussion",
    "questions answered",
    "patient questions",
    "activity restriction",
    "activity restrictions",
    "hiking",
    "hospital stay",
    "daily weight",
    "weight monitoring",
    "scale",
    "blood pressure cuff",
    "blood pressure monitor",
    "dietary",
    "high-fat",
    "high fat",
    "fatty foods",
    "mri",
    "ct",
    "ultrasound",
    "echocardiogram",
    "spirometry",
    "cbc",
    "covid",
)
_DROP_TESTING_ALLOWED_PHRASES = (
    "order",
    "ordered",
    "ordering",
    "repeat",
    "obtain",
    "schedule",
    "testing",
    "test",
    "tests",
    "workup",
    "lab",
    "labs",
    "imaging",
    "ultrasound",
    "mri",
    "ct",
    "cbc",
    "bmp",
    "cmp",
    "a1c",
    "echocardiogram",
    "urinalysis",
    "pregnancy test",
    "covid test",
)
_DROP_TESTING_BLOCKED_PHRASES = (
    "findings",
    "result",
    "results",
    "outside facility",
    "gallstones",
    "follow-up",
    "follow up",
    "return precautions",
    "medication",
    "procedure",
    "surgery",
)
_DROP_TREATMENT_ALLOWED_PHRASES = (
    "treatment",
    "treatment plan",
    "management plan",
    "medication",
    "medications",
    "therapy",
    "therapeutic",
    "supportive care",
    "pain control",
    "anticoagulant",
    "antibiotic",
    "dose",
    "dosing",
    "procedure",
    "surgery",
    "surgical",
    "intervention",
    "diet",
    "dietary",
    "avoid high-fat",
    "avoid high fat",
    "fatty foods",
)
_DROP_TREATMENT_BLOCKED_PHRASES = (
    "follow-up timing",
    "follow up timing",
    "follow-up appointment",
    "follow up appointment",
    "follow-up plan",
    "follow up plan",
    "follow-up with",
    "follow up with",
    "return precautions",
    "return to clinic",
    "outpatient follow-up",
    "ultrasound findings",
    "imaging results",
    "patient education",
    "counseling",
    "understands and agrees",
    "agreement",
    "because",
    "due to",
    "rationale",
)
_DROP_REASONING_ALLOWED_PHRASES = (
    "reasoning",
    "rationale",
    "because",
    "due to",
    "consistent with",
    "suggestive",
    "supported by",
    "supports",
    "based on",
    "correlation",
    "correlate",
    "justify",
    "justified",
    "working diagnosis",
    "impression",
    "risk",
    "excluded because",
    "decision-making",
)
_DROP_REASONING_BLOCKED_PHRASES = (
    "patient education",
    "counseling",
    "follow-up timing",
    "follow up timing",
    "return precautions",
    "call the clinic if",
    "call if symptoms worsen",
    "worsening symptoms",
    "seek urgent care",
    "return sooner",
    "medication dosing",
    "dose adjustment",
)

_DISCOVERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "better",
    "by",
    "clear",
    "clearly",
    "detail",
    "details",
    "document",
    "documentation",
    "documented",
    "encounter",
    "for",
    "from",
    "here",
    "in",
    "include",
    "included",
    "includes",
    "is",
    "it",
    "main",
    "matters",
    "more",
    "must",
    "note",
    "patient",
    "present",
    "quality",
    "should",
    "shows",
    "specific",
    "states",
    "statement",
    "that",
    "the",
    "their",
    "there",
    "this",
    "thorough",
    "to",
    "visit",
    "with",
}
_GENERIC_RECURSION_DIMENSIONS = {
    "",
    "content_coverage",
    "completeness",
    "final_response_quality",
    "format_communication",
    "grounding",
    "instruction_adherence",
    "reasoning_support",
    "source_grounding",
    "task_completion",
}
_BROAD_RECURSION_HINTS = (
    "content coverage",
    "completeness",
    "overall quality",
    "main next steps",
    "general quality",
    "requested details",
    "requested format",
    "key details",
    "all relevant",
    "main instructions",
)

_MUTATION_GROUNDING_PROFILES: Dict[str, Dict[str, Any]] = {
    "flatten_structure": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker note mainly loses section/header structure rather than clinical content.",
        "keywords": (
            "chief complaint",
            "review of systems",
            "assessment and plan",
            "section heading",
            "section headings",
            "section header",
            "section headers",
            "soap",
            "subjective",
            "objective",
            "documentation structure",
        ),
    },
    "strip_symptom_detail_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops encounter-specific symptom characterization such as pain "
            "location/quality, timing, triggers, or fever/temperature detail. Do not propose criteria "
            "that are really about unrelated pertinent negatives or granular physical-exam minutiae."
        ),
        "keywords": (
            "pain",
            "pain characteristics",
            "pain quality",
            "duration",
            "timing",
            "trigger",
            "meal",
            "postprandial",
            "location",
            "quadrant",
            "fever",
            "fevers",
            "afebrile",
            "temperature",
            "constitutional",
        ),
    },
    "drop_vomiting_negative": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker note mainly drops the pertinent negative about vomiting.",
        "keywords": (
            "vomiting",
            "vomit",
            "denies vomiting",
            "pertinent negative",
            "pertinent negatives",
            "ros negative",
            "review of systems",
        ),
    },
    "drop_study_mentions": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops studies, imaging/lab findings, or reasoning explicitly tied to "
            "those findings. Do not propose broader surgery, education, follow-up, or comorbidity-plan "
            "criteria unless they are directly about the missing study/result content."
        ),
        "keywords": (
            "ultrasound",
            "imaging",
            "study mention",
            "study mentions",
            "test mention",
            "test mentions",
            "study finding",
            "study findings",
            "reviewed result",
            "outside facility",
            "test result",
            "test results",
            "lab",
            "labs",
            "results",
        ),
    },
    "drop_followup_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops scheduled follow-up timing or planned reevaluation. Do not "
            "propose return-precaution, home-monitoring, patient-agreement, test-result callback, "
            "medication, procedure, or testing-plan criteria unless they are explicitly about the "
            "scheduled follow-up plan."
        ),
        "keywords": (
            "follow-up",
            "follow up",
            "follow-up timing",
            "recheck",
            "outpatient",
            "follow with",
            "surveillance",
            "appointment",
        ),
    },
    "drop_return_precaution_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops return precautions, warning signs, or escalation guidance about "
            "when to call, return, or seek urgent care. Do not propose routine follow-up timing, medication, "
            "procedure, or testing-plan criteria unless they are explicitly framed as return precautions."
        ),
        "keywords": (
            "return precautions",
            "return if",
            "return sooner",
            "seek care",
            "seek medical attention",
            "call the office if",
            "worsening symptoms",
            "red flags",
            "urgent care",
            "call 911",
        ),
    },
    "drop_medication_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops medication details such as names, dose changes, refill plans, "
            "continuation/discontinuation instructions, or dosing/frequency guidance. Do not propose "
            "disease-monitoring, patient-agreement, procedure, testing, follow-up, or general counseling "
            "criteria unless they are explicitly about the medication regimen itself."
        ),
        "keywords": (
            "medication",
            "medications",
            "dose",
            "dosage",
            "dosing",
            "refill",
            "prescription",
            "continue",
            "start",
            "stop",
            "increase",
            "decrease",
            "inhaler",
        ),
    },
    "drop_procedure_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops procedure, referral, device, or intervention recommendations. "
            "Do not propose diagnostic-test orders, postoperative lifestyle restrictions, patient "
            "discussion/agreement boilerplate, medication-only, routine follow-up, or return-precaution "
            "criteria unless they are inseparable from the missing intervention recommendation."
        ),
        "keywords": (
            "procedure",
            "surgery",
            "surgical",
            "intervention",
            "referral",
            "physical therapy",
            "brace",
            "splint",
            "crutches",
            "aircast",
            "device",
            "egd",
            "endoscopy",
        ),
    },
    "drop_testing_plan_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops planned diagnostic workup such as ordered labs, imaging, repeat "
            "testing, or pending-study plans. Do not propose criteria about already-reviewed findings, "
            "medications, procedures, or follow-up instructions unless they are explicitly about the planned "
            "testing strategy."
        ),
        "keywords": (
            "testing",
            "test",
            "tests",
            "ordered",
            "repeat",
            "obtain",
            "schedule",
            "labs",
            "imaging",
            "workup",
            "ultrasound",
            "mri",
            "cbc",
            "a1c",
        ),
    },
    "drop_treatment_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "Legacy broad mutation: the weaker note mainly drops treatment or management details such as "
            "medications, procedures, therapies, or diet-related recommendations. Prefer newer narrower "
            "mutation families for medication, intervention, and testing-specific criteria."
        ),
        "keywords": (
            "treatment",
            "management plan",
            "medication",
            "medications",
            "therapy",
            "supportive care",
            "anticoagulant",
            "antibiotic",
            "dose",
            "procedure",
            "surgery",
            "surgical",
            "intervention",
            "diet",
            "dietary",
        ),
    },
    "drop_assessment_reasoning_lines": {
        "delta_mode": "strong_only",
        "prompt_hint": (
            "The weaker note mainly drops reasoning that links symptoms, exam findings, or study results "
            "to the assessment or explains why a management choice was made. Do not propose section-only "
            "or generic plan-presence criteria unless they clearly express that reasoning link."
        ),
        "keywords": (
            "reasoning",
            "medical reasoning",
            "rationale",
            "because",
            "due to",
            "consistent with",
            "supported by",
            "based on",
            "working diagnosis",
            "impression",
            "risk",
            "correlation",
            "decision-making",
        ),
    },
    "inflate_certainty": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker note mainly adds overconfident or unsupported certainty language.",
        "keywords": (
            "certainty",
            "diagnostic certainty",
            "unsupported",
            "pathognomonic",
            "definitive",
            "cautious language",
            "hedged",
            "overconfident",
        ),
    },
}


def _max_dialogue_chars() -> int:
    return int(
        os.getenv(
            "RUBRIC_GEN_COMPILED_DISCOVERY_MAX_DIALOGUE_CHARS",
            os.getenv("RUBRIC_GEN_COMPILED_JUDGE_MAX_DIALOGUE_CHARS", "14000"),
        )
    )


def _max_note_chars() -> int:
    return int(
        os.getenv(
            "RUBRIC_GEN_COMPILED_DISCOVERY_MAX_NOTE_CHARS",
            os.getenv("RUBRIC_GEN_COMPILED_JUDGE_MAX_NOTE_CHARS", "12000"),
        )
    )


def _truncate(text: str, limit: int) -> str:
    t = text or ""
    if len(t) <= limit:
        return t
    return t[:limit] + "\n…[truncated]"


def select_strong_candidate(
    candidates: List[CandidateNote],
    source_priority: Optional[Tuple[str, ...]] = None,
) -> Optional[CandidateNote]:
    """Prefer reference-style originals before weaker variants or truncations."""
    originals = [c for c in candidates if not is_synthetic_candidate(c)]
    priority = source_priority or (
        "reference_note",
        "reference_artifact",
        "augmented_note",
        "augmented_artifact",
        "note_truncated",
        "artifact_truncated",
    )
    for label in priority:
        for c in originals:
            if c.source_label == label and c.text.strip():
                return c
    return originals[0] if originals else None


def select_weak_candidates(
    candidates: List[CandidateNote],
    strong: CandidateNote,
    source_priority: Optional[Tuple[str, ...]] = None,
) -> List[CandidateNote]:
    """Prefer synthetic mutations; otherwise later / 'weaker' original fields after the strong note."""
    synthetics = [c for c in candidates if is_synthetic_candidate(c) and c.text.strip()]
    if synthetics:
        return synthetics

    originals = [c for c in candidates if not is_synthetic_candidate(c) and c.text.strip()]
    order = source_priority or (
        "reference_note",
        "reference_artifact",
        "augmented_note",
        "augmented_artifact",
        "note_truncated",
        "artifact_truncated",
    )
    rank = {lbl: i for i, lbl in enumerate(order)}
    sorted_orig = sorted(originals, key=lambda c: rank.get(c.source_label, 99))
    try:
        idx = next(i for i, c in enumerate(sorted_orig) if c.candidate_id == strong.candidate_id)
    except StopIteration:
        return []
    return sorted_orig[idx + 1 :]


def pair_id(strong: CandidateNote, weak: CandidateNote) -> str:
    return f"{strong.candidate_id}__vs__{weak.candidate_id}"


def _candidate_mutation_id(candidate: CandidateNote) -> Optional[str]:
    mid = candidate.metadata.get("mutation_id")
    if isinstance(mid, str) and mid.strip():
        return mid.strip()
    label = (candidate.source_label or "").strip()
    if label.startswith("synthetic_mutation:"):
        mid = label.split(":", 1)[1].strip()
        return mid or None
    return None


def _normalize_match_text(text: str) -> str:
    tokens = _WORD_RE.findall((text or "").lower().replace("_", " "))
    return " ".join(tokens)


def _trim_preview(text: str, limit: int = 180) -> str:
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _line_delta(strong_text: str, weak_text: str) -> Tuple[List[str], List[str]]:
    def normalized_lines(text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for raw in (text or "").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            out.append((_normalize_match_text(stripped), stripped))
        return out

    strong_lines = normalized_lines(strong_text)
    weak_lines = normalized_lines(weak_text)
    weak_norms = {norm for norm, _ in weak_lines if norm}
    strong_norms = {norm for norm, _ in strong_lines if norm}

    strong_only = [raw for norm, raw in strong_lines if norm and norm not in weak_norms]
    weak_only = [raw for norm, raw in weak_lines if norm and norm not in strong_norms]
    return strong_only, weak_only


def _proposal_blob(row: Dict[str, Any]) -> str:
    return _normalize_match_text(
        " ".join(
            str(row.get(key, ""))
            for key in ("dimension", "label", "requirement", "rationale")
        )
    )


def _proposal_salient_terms(row: Dict[str, Any]) -> List[str]:
    tokens = [
        tok
        for tok in _WORD_RE.findall(
            " ".join(
                str(row.get(key, "")).lower().replace("_", " ")
                for key in ("dimension", "label", "requirement")
            )
        )
        if len(tok) >= 4 and tok not in _DISCOVERY_STOPWORDS
    ]

    phrases: List[str] = []
    for width in (3, 2):
        for i in range(len(tokens) - width + 1):
            phrase = " ".join(tokens[i : i + width])
            if phrase not in phrases:
                phrases.append(phrase)
    for tok in tokens:
        if len(tok) >= 7 and tok not in phrases:
            phrases.append(tok)
    return phrases[:18]


def _normalized_phrase_hits(text: str, phrases: Tuple[str, ...]) -> List[str]:
    normalized = _normalize_match_text(text)
    return [phrase for phrase in phrases if _normalize_match_text(phrase) in normalized]


def _gate_allowed_blocked_phrases(
    *,
    proposal_blob: str,
    delta_hits: List[str],
    profile_hits: List[str],
    allowed_phrases: Tuple[str, ...],
    blocked_phrases: Tuple[str, ...],
) -> Tuple[bool, Dict[str, Any]]:
    allowed_hits = _normalized_phrase_hits(proposal_blob, allowed_phrases)
    blocked_hits = _normalized_phrase_hits(proposal_blob, blocked_phrases)
    keep = bool(delta_hits or profile_hits) and bool(allowed_hits) and not blocked_hits
    return keep, {
        "allowed_phrase_hits": allowed_hits,
        "blocked_phrase_hits": blocked_hits,
    }


def _apply_mutation_specific_gate(
    *,
    mutation_id: str,
    proposal_blob: str,
    delta_hits: List[str],
    profile_hits: List[str],
    task_profile_id: str = "note_documentation",
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    if task_profile_id != "note_documentation":
        active_profile = profile or {}
        allowed = tuple(str(value) for value in active_profile.get("allowed_phrases", ()) if str(value).strip())
        blocked = tuple(str(value) for value in active_profile.get("blocked_phrases", ()) if str(value).strip())
        if allowed or blocked:
            return _gate_allowed_blocked_phrases(
                proposal_blob=proposal_blob,
                delta_hits=delta_hits,
                profile_hits=profile_hits,
                allowed_phrases=allowed or tuple(profile_hits),
                blocked_phrases=blocked,
            )
        return bool(delta_hits or profile_hits), {}

    if mutation_id == "drop_study_mentions":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_STUDY_ALLOWED_PHRASES,
            blocked_phrases=_DROP_STUDY_BLOCKED_PHRASES,
        )
    if mutation_id == "drop_followup_lines":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_FOLLOWUP_ALLOWED_PHRASES,
            blocked_phrases=_DROP_FOLLOWUP_BLOCKED_PHRASES,
        )
    if mutation_id == "drop_return_precaution_lines":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_RETURN_ALLOWED_PHRASES,
            blocked_phrases=_DROP_RETURN_BLOCKED_PHRASES,
        )
    if mutation_id == "drop_medication_lines":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_MEDICATION_ALLOWED_PHRASES,
            blocked_phrases=_DROP_MEDICATION_BLOCKED_PHRASES,
        )
    if mutation_id == "drop_procedure_lines":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_PROCEDURE_ALLOWED_PHRASES,
            blocked_phrases=_DROP_PROCEDURE_BLOCKED_PHRASES,
        )
    if mutation_id == "drop_testing_plan_lines":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_TESTING_ALLOWED_PHRASES,
            blocked_phrases=_DROP_TESTING_BLOCKED_PHRASES,
        )
    if mutation_id == "drop_treatment_lines":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_TREATMENT_ALLOWED_PHRASES,
            blocked_phrases=_DROP_TREATMENT_BLOCKED_PHRASES,
        )
    if mutation_id == "drop_assessment_reasoning_lines":
        return _gate_allowed_blocked_phrases(
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            allowed_phrases=_DROP_REASONING_ALLOWED_PHRASES,
            blocked_phrases=_DROP_REASONING_BLOCKED_PHRASES,
        )
    if mutation_id != "strip_symptom_detail_lines":
        return bool(delta_hits or profile_hits), {}

    return _gate_allowed_blocked_phrases(
        proposal_blob=proposal_blob,
        delta_hits=delta_hits,
        profile_hits=profile_hits,
        allowed_phrases=_STRIP_SYMPTOM_ALLOWED_PHRASES,
        blocked_phrases=_STRIP_SYMPTOM_BLOCKED_PHRASES,
    )


def filter_pair_grounded_proposals(
    *,
    strong: CandidateNote,
    weak: CandidateNote,
    proposals: List[Dict[str, Any]],
    task_profile_id: str = "note_documentation",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministic post-filter for discovery proposals.

    For synthetic weak notes, keep only proposals that align with the mutation profile
    and/or match lexical cues in the pair delta. This prevents off-target criteria from
    entering the merged proposal pool while preserving raw suggestions for inspection.
    """
    mutation_id = _candidate_mutation_id(weak)
    grounding_profiles = (
        _MUTATION_GROUNDING_PROFILES
        if task_profile_id == "note_documentation"
        else mutation_grounding_profiles_for_profile(task_profile_id)
    )
    profile = grounding_profiles.get(mutation_id or "")
    if not proposals:
        return [], [], {"mode": "none", "weak_mutation_id": mutation_id}
    if not mutation_id or profile is None:
        return list(proposals), [], {
            "mode": "unfiltered",
            "weak_mutation_id": mutation_id,
            "task_profile_id": task_profile_id,
            "accepted_count": len(proposals),
            "rejected_count": 0,
        }

    strong_only_lines, weak_only_lines = _line_delta(strong.text, weak.text)
    strong_only_blob = _normalize_match_text("\n".join(strong_only_lines))
    weak_only_blob = _normalize_match_text("\n".join(weak_only_lines))
    delta_mode = str(profile.get("delta_mode", "strong_only"))
    grounding_blob = strong_only_blob if delta_mode == "strong_only" else weak_only_blob
    profile_keywords = tuple(profile.get("keywords", ()))

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for row in proposals:
        proposal_blob = _proposal_blob(row)
        salient_terms = _proposal_salient_terms(row)
        delta_hits = [term for term in salient_terms if term and term in grounding_blob]
        profile_hits = [
            kw for kw in profile_keywords if _normalize_match_text(str(kw)) in proposal_blob
        ]
        keep, mutation_gate = _apply_mutation_specific_gate(
            mutation_id=mutation_id,
            proposal_blob=proposal_blob,
            delta_hits=delta_hits,
            profile_hits=profile_hits,
            task_profile_id=task_profile_id,
            profile=dict(profile),
        )
        annotated = {
            **row,
            "promotion_eligible": keep,
            "grounding": {
                "mode": "synthetic_mutation_delta",
                "weak_mutation_id": mutation_id,
                "task_profile_id": task_profile_id,
                "delta_mode": delta_mode,
                "delta_term_hits": delta_hits,
                "profile_keyword_hits": profile_hits,
                **mutation_gate,
            },
        }
        if keep:
            accepted.append(annotated)
        else:
            annotated["grounding_rejection_reason"] = (
                f"proposal_not_aligned_with_{mutation_id}_pair_delta"
            )
            rejected.append(annotated)

    return accepted, rejected, {
        "mode": "synthetic_mutation_delta",
        "weak_mutation_id": mutation_id,
        "task_profile_id": task_profile_id,
        "delta_mode": delta_mode,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "strong_only_line_count": len(strong_only_lines),
        "weak_only_line_count": len(weak_only_lines),
        "strong_only_preview": [_trim_preview(line) for line in strong_only_lines[:6]],
        "weak_only_preview": [_trim_preview(line) for line in weak_only_lines[:6]],
    }


def _proposal_word_count(row: Mapping[str, Any]) -> int:
    return len(
        _WORD_RE.findall(
            " ".join(str(row.get(key, "")) for key in ("label", "requirement"))
        )
    )


def _proposal_compound_marker_count(row: Mapping[str, Any]) -> int:
    requirement = str(row.get("requirement", "")).lower()
    return requirement.count(" and ") + requirement.count(";") + requirement.count(", ")


def _decomposition_reason_for_proposal(
    row: Mapping[str, Any],
    *,
    calibration_guidance: str = "",
) -> Optional[str]:
    dimension = _normalize_key_part(str(row.get("dimension", "")))
    proposal_blob = _normalize_key_part(
        " ".join(str(row.get(key, "")) for key in ("dimension", "label", "requirement"))
    )
    if dimension in _GENERIC_RECURSION_DIMENSIONS and calibration_guidance.strip():
        return "calibration_guided_generic_parent"
    if dimension in _GENERIC_RECURSION_DIMENSIONS and any(
        hint in proposal_blob for hint in _BROAD_RECURSION_HINTS
    ):
        return "generic_dimension_broad_requirement"
    if any(hint in proposal_blob for hint in _BROAD_RECURSION_HINTS):
        return "broad_label_or_requirement"
    if _proposal_compound_marker_count(row) >= 2:
        return "compound_requirement"
    if _proposal_word_count(row) >= 18:
        return "long_requirement"
    if dimension in _GENERIC_RECURSION_DIMENSIONS and _proposal_word_count(row) >= 12:
        return "generic_dimension"
    return None


def _criterion_id_for_row(
    row: Mapping[str, Any],
    *,
    root_pair_id: str,
    recursion_depth: int,
    parent_criterion_id: Optional[str],
) -> str:
    digest = stable_hash(
        {
            "root_pair_id": root_pair_id,
            "recursion_depth": recursion_depth,
            "parent_criterion_id": parent_criterion_id or "",
            "proposal_index": row.get("proposal_index"),
            "dimension": str(row.get("dimension", "")),
            "label": str(row.get("label", "")),
            "requirement": str(row.get("requirement", "")),
        }
    )[:12]
    return f"{root_pair_id}__criterion_{recursion_depth}_{digest}"


def _apply_recursive_provenance(
    rows: Sequence[Mapping[str, Any]],
    *,
    root_pair_id: str,
    recursion_depth: int,
    parent_criterion_id: Optional[str],
    recursion_reason: str,
    decomposition_source: str,
) -> List[Dict[str, Any]]:
    annotated_rows: List[Dict[str, Any]] = []
    for row in rows:
        annotated = dict(row)
        annotated["root_pair_id"] = root_pair_id
        annotated["recursion_depth"] = recursion_depth
        annotated["parent_criterion_id"] = parent_criterion_id
        annotated["recursion_reason"] = recursion_reason
        annotated["decomposition_source"] = decomposition_source
        annotated["criterion_id"] = _criterion_id_for_row(
            annotated,
            root_pair_id=root_pair_id,
            recursion_depth=recursion_depth,
            parent_criterion_id=parent_criterion_id,
        )
        annotated_rows.append(annotated)
    return annotated_rows


def _decomposition_system_prompt_for_profile(task_profile_id: str) -> str:
    return (
        f"{_system_prompt_for_profile(task_profile_id)}\n\n"
        "When given a parent criterion, only decompose it into narrower child criteria if the"
        " same strong-vs-weak artifact difference clearly supports those children. Prefer small,"
        " non-overlapping children over paraphrasing the parent."
    )


def _pair_discriminator_system_prompt_for_profile(task_profile_id: str) -> str:
    return (
        f"{_system_prompt_for_profile(task_profile_id)}\n\n"
        "You are breaking a scoring tie between candidate artifacts for the same task instance.\n"
        "Choose a directional preference only when you can point to a concrete task-relevant behavior,"
        " constraint, or final-answer difference. If they are genuinely equivalent on the task-relevant"
        " evidence, you may return A=B."
    )


def build_pair_discriminator_prompts(
    *,
    example: ExampleRecord,
    candidate_a: CandidateNote,
    candidate_b: CandidateNote,
    task_profile_id: str,
    artifact_label: str = "response",
    calibration_guidance: str = "",
    candidate_a_id: str = "X",
    candidate_b_id: str = "Y",
    few_shot_block: str = "",
) -> Tuple[str, str]:
    dlim, nlim = _max_dialogue_chars(), _max_note_chars()
    context = _truncate(example.task_prompt or example.conversation, dlim)
    reference = _truncate(example.reference_artifact, nlim)
    text_a = _truncate(candidate_a.text, nlim)
    text_b = _truncate(candidate_b.text, nlim)
    reference_block = ""
    if reference:
        reference_block = f"\nREFERENCE {artifact_label.upper()}:\n{reference}\n"
    calibration_hint = ""
    if calibration_guidance.strip():
        calibration_hint = f"\n{calibration_guidance.strip()}\n"
    system_prompt = _pair_discriminator_system_prompt_for_profile(task_profile_id)
    if few_shot_block.strip():
        system_prompt = f"{system_prompt}\n\n{few_shot_block.strip()}"
    user_prompt = f"""TASK CONTEXT:
{context}
{reference_block}{calibration_hint}

ID {candidate_a_id} (source: {candidate_a.source_label}):
{text_a}

ID {candidate_b_id} (source: {candidate_b.source_label}):
{text_b}

Task: the current rubric pass could not distinguish ID {candidate_a_id} from ID {candidate_b_id}.
Identify the most concrete task-relevant difference between them, if one exists, and choose the better candidate.
Prefer behavioral correctness, satisfying stated constraints, and the final requested deliverable over implementation
style, formatting polish, or preferring one implementation technique when both behave the same.

Return JSON with this exact shape:
{{
  "decision": "<{candidate_a_id}>{candidate_b_id}|{candidate_b_id}>{candidate_a_id}|{candidate_a_id}={candidate_b_id}>",
  "distinguishing_behavior": "<one sentence naming the key task-relevant difference, or 'none'>",
  "confidence": "<high|medium|low>"
}}
Only choose {candidate_a_id}={candidate_b_id} if there is no concrete task-relevant difference."""
    return system_prompt, user_prompt


def _build_user_prompt(
    *,
    dialogue: str,
    strong_note: str,
    weak_note: str,
    strong_label: str,
    weak_label: str,
    weak_mutation_id: Optional[str],
    max_criteria: int,
    task_profile_id: str = "note_documentation",
    artifact_label: str = "note",
    calibration_guidance: str = "",
) -> str:
    dlim, nlim = _max_dialogue_chars(), _max_note_chars()
    d = _truncate(dialogue, dlim)
    s = _truncate(strong_note, nlim)
    w = _truncate(weak_note, nlim)
    grounding_profiles = (
        _MUTATION_GROUNDING_PROFILES
        if task_profile_id == "note_documentation"
        else mutation_grounding_profiles_for_profile(task_profile_id)
    )
    profile = grounding_profiles.get(weak_mutation_id or "")
    grounding_hint = ""
    if profile:
        grounding_hint = (
            f"\nPAIR GROUNDING HINT: weaker artifact synthetic mutation = {weak_mutation_id}."
            f" {profile.get('prompt_hint', '')}\n"
        )
    calibration_hint = ""
    if calibration_guidance.strip():
        calibration_hint = f"\n{calibration_guidance.strip()}\n"
    context_label = "DIALOGUE (encounter context)" if task_profile_id == "note_documentation" else "TASK CONTEXT"
    stronger_label = "STRONGER NOTE" if task_profile_id == "note_documentation" else f"STRONGER {artifact_label.upper()}"
    weaker_label = "WEAKER NOTE" if task_profile_id == "note_documentation" else f"WEAKER {artifact_label.upper()}"
    return f"""{context_label}:
{d}

{stronger_label} (preferred; source: {strong_label}):
{s}

{weaker_label} (contrast; source: {weak_label}):
{w}
{grounding_hint}{calibration_hint}

Task: propose up to {max_criteria} local atomic rubric criteria that explain why the stronger {artifact_label}
is better for THIS task instance. Each criterion should be checkable on a single {artifact_label} in isolation.

Return JSON with this exact shape:
{{
  "proposals": [
    {{
      "dimension": "<short_snake_or_label>",
      "label": "<short human label>",
      "requirement": "<one sentence, atomic, binary-leaning>",
      "severity_tier": "<hard_gate|high|medium|low>",
      "rationale": "<one sentence: why this criterion matters here>"
    }}
  ]
}}
Use at least 1 and at most {max_criteria} proposals. Empty array only if the two artifacts are effectively
equivalent for the relevant task quality (rare)."""


def _build_decomposition_user_prompt(
    *,
    dialogue: str,
    strong_note: str,
    weak_note: str,
    strong_label: str,
    weak_label: str,
    weak_mutation_id: Optional[str],
    parent_row: Mapping[str, Any],
    max_children: int,
    task_profile_id: str = "note_documentation",
    artifact_label: str = "note",
    calibration_guidance: str = "",
) -> str:
    dlim, nlim = _max_dialogue_chars(), _max_note_chars()
    d = _truncate(dialogue, dlim)
    s = _truncate(strong_note, nlim)
    w = _truncate(weak_note, nlim)
    grounding_profiles = (
        _MUTATION_GROUNDING_PROFILES
        if task_profile_id == "note_documentation"
        else mutation_grounding_profiles_for_profile(task_profile_id)
    )
    profile = grounding_profiles.get(weak_mutation_id or "")
    grounding_hint = ""
    if profile:
        grounding_hint = (
            f"\nPAIR GROUNDING HINT: weaker artifact synthetic mutation = {weak_mutation_id}."
            f" {profile.get('prompt_hint', '')}\n"
        )
    calibration_hint = ""
    if calibration_guidance.strip():
        calibration_hint = f"\n{calibration_guidance.strip()}\n"
    context_label = "DIALOGUE (encounter context)" if task_profile_id == "note_documentation" else "TASK CONTEXT"
    stronger_label = "STRONGER NOTE" if task_profile_id == "note_documentation" else f"STRONGER {artifact_label.upper()}"
    weaker_label = "WEAKER NOTE" if task_profile_id == "note_documentation" else f"WEAKER {artifact_label.upper()}"
    parent_payload = {
        "dimension": str(parent_row.get("dimension", "")),
        "label": str(parent_row.get("label", "")),
        "requirement": str(parent_row.get("requirement", "")),
        "severity_tier": str(parent_row.get("severity_tier", "")),
        "criterion_id": str(parent_row.get("criterion_id", "")),
    }
    return f"""{context_label}:
{d}

{stronger_label} (preferred; source: {strong_label}):
{s}

{weaker_label} (contrast; source: {weak_label}):
{w}
{grounding_hint}{calibration_hint}

PARENT CRITERION TO DECOMPOSE:
{json.dumps(parent_payload, ensure_ascii=False, indent=2)}

Task: if the parent criterion is too broad, break it into up to {max_children} narrower child criteria that are each
individually grounded in the visible difference between the stronger and weaker {artifact_label}. Child criteria should
be smaller than the parent, should not simply restate the parent, and should avoid overlap where possible.

If the parent is already atomic for this pair, return an empty array.

Return JSON with this exact shape:
{{
  "proposals": [
    {{
      "dimension": "<short_snake_or_label>",
      "label": "<short human label>",
      "requirement": "<one sentence, atomic, binary-leaning>",
      "severity_tier": "<hard_gate|high|medium|low>",
      "rationale": "<one sentence: why this child criterion matters here>"
    }}
  ]
}}
Use at most {max_children} child proposals."""


def _parse_proposals(raw: str, *, max_criteria: int) -> List[Dict[str, Any]]:
    obj = extract_json_object(raw)
    if not obj:
        raise ValueError("LLM response did not contain parseable JSON object.")
    proposals = obj.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("JSON must contain a 'proposals' array.")
    out: List[Dict[str, Any]] = []
    for row in proposals[:max_criteria]:
        if not isinstance(row, dict):
            continue
        req = str(row.get("requirement", "")).strip()
        if not req:
            continue
        out.append(
            {
                "dimension": str(row.get("dimension", "")).strip(),
                "label": str(row.get("label", "")).strip(),
                "requirement": req,
                "severity_tier": str(row.get("severity_tier", "")).strip(),
                "rationale": str(row.get("rationale", row.get("why", ""))).strip(),
            }
        )
    return out


def _run_discovery_prompt(
    *,
    prompt_version: str,
    payload_for_key: Mapping[str, Any],
    user_prompt: str,
    system_prompt: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    max_criteria: int,
    cache_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    cache_key = make_cache_key(prompt_version, dict(payload_for_key))
    cache_hit = False
    raw_text = ""
    if cache and cache.enabled:
        cache.load()
        hit = cache.get(cache_key)
        if hit and isinstance(hit.get("raw_response"), str):
            raw_text = hit["raw_response"]
            cache_hit = True

    if not raw_text:
        resp = router.generate(
            model_spec,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        raw_text = resp.raw_text or resp.text
        if cache and cache.enabled:
            record = {"raw_response": raw_text}
            if cache_metadata:
                record.update(dict(cache_metadata))
            cache.set(cache_key, record)

    err: Optional[str] = None
    try:
        proposals = _parse_proposals(raw_text, max_criteria=max_criteria)
    except ValueError as exc:
        err = str(exc)
        proposals = []
    return proposals, cache_hit, err


def _enrich_pair_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    example: ExampleRecord,
    strong: CandidateNote,
    weak: CandidateNote,
    task_profile_id: str,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    pid = pair_id(strong, weak)
    enriched: List[Dict[str, Any]] = []
    for i, row in enumerate(proposals):
        payload = {
            **dict(row),
            "example_id": example.example_id,
            "pair_id": pid,
            "strong_candidate_id": strong.candidate_id,
            "weak_candidate_id": weak.candidate_id,
            "proposal_index": i,
            "task_profile_id": task_profile_id,
        }
        if extra_fields:
            payload.update(dict(extra_fields))
        enriched.append(payload)
    return enriched


def propose_criteria_for_pair(
    *,
    example: ExampleRecord,
    strong: CandidateNote,
    weak: CandidateNote,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    max_criteria: int,
    task_profile_id: str = "note_documentation",
    artifact_label: str = "note",
    calibration_guidance: str = "",
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """
    One LLM call per pair. Returns (enriched proposals, cache_hit, error_message).
    """
    pid = pair_id(strong, weak)
    payload_for_key = {
        "prompt_version": DISCOVERY_PROMPT_VERSION,
        "model": f"{model_spec.provider}:{model_spec.model}",
        "dialogue_hash": stable_hash(example_to_prompt(example)),
        "strong_id": strong.candidate_id,
        "weak_id": weak.candidate_id,
        "strong_text_hash": stable_hash(strong.text),
        "weak_text_hash": stable_hash(weak.text),
        "task_profile_id": task_profile_id,
        "calibration_guidance_hash": stable_hash(calibration_guidance.strip()) if calibration_guidance.strip() else "",
    }
    proposals, cache_hit, err = _run_discovery_prompt(
        prompt_version=DISCOVERY_PROMPT_VERSION,
        payload_for_key=payload_for_key,
        user_prompt=_build_user_prompt(
            dialogue=example_to_prompt(example),
            strong_note=strong.text,
            weak_note=weak.text,
            strong_label=strong.source_label,
            weak_label=weak.source_label,
            weak_mutation_id=_candidate_mutation_id(weak),
            max_criteria=max_criteria,
            task_profile_id=task_profile_id,
            artifact_label=artifact_label,
            calibration_guidance=calibration_guidance,
        ),
        system_prompt=_system_prompt_for_profile(task_profile_id),
        model_spec=model_spec,
        router=router,
        cache=cache,
        max_criteria=max_criteria,
        cache_metadata={
            "example_id": example.example_id,
            "pair_id": pid,
        },
    )
    return (
        _enrich_pair_proposals(
            proposals,
            example=example,
            strong=strong,
            weak=weak,
            task_profile_id=task_profile_id,
        ),
        cache_hit,
        err,
    )


def _decompose_criterion_for_pair(
    *,
    example: ExampleRecord,
    strong: CandidateNote,
    weak: CandidateNote,
    parent_row: Mapping[str, Any],
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    max_children: int,
    task_profile_id: str,
    artifact_label: str,
    calibration_guidance: str = "",
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    pid = pair_id(strong, weak)
    payload_for_key = {
        "prompt_version": DISCOVERY_DECOMPOSITION_PROMPT_VERSION,
        "model": f"{model_spec.provider}:{model_spec.model}",
        "dialogue_hash": stable_hash(example_to_prompt(example)),
        "strong_id": strong.candidate_id,
        "weak_id": weak.candidate_id,
        "strong_text_hash": stable_hash(strong.text),
        "weak_text_hash": stable_hash(weak.text),
        "task_profile_id": task_profile_id,
        "parent_criterion_id": str(parent_row.get("criterion_id", "")),
        "parent_label": str(parent_row.get("label", "")),
        "parent_requirement_hash": stable_hash(str(parent_row.get("requirement", ""))),
        "parent_depth": int(parent_row.get("recursion_depth", 0) or 0),
        "calibration_guidance_hash": stable_hash(calibration_guidance.strip()) if calibration_guidance.strip() else "",
    }
    proposals, cache_hit, err = _run_discovery_prompt(
        prompt_version=DISCOVERY_DECOMPOSITION_PROMPT_VERSION,
        payload_for_key=payload_for_key,
        user_prompt=_build_decomposition_user_prompt(
            dialogue=example_to_prompt(example),
            strong_note=strong.text,
            weak_note=weak.text,
            strong_label=strong.source_label,
            weak_label=weak.source_label,
            weak_mutation_id=_candidate_mutation_id(weak),
            parent_row=parent_row,
            max_children=max_children,
            task_profile_id=task_profile_id,
            artifact_label=artifact_label,
            calibration_guidance=calibration_guidance,
        ),
        system_prompt=_decomposition_system_prompt_for_profile(task_profile_id),
        model_spec=model_spec,
        router=router,
        cache=cache,
        max_criteria=max_children,
        cache_metadata={
            "example_id": example.example_id,
            "pair_id": pid,
            "parent_criterion_id": str(parent_row.get("criterion_id", "")),
        },
    )
    return (
        _enrich_pair_proposals(
            proposals,
            example=example,
            strong=strong,
            weak=weak,
            task_profile_id=task_profile_id,
            extra_fields={"decomposition_parent_criterion_id": parent_row.get("criterion_id")},
        ),
        cache_hit,
        err,
    )


def _expand_recursive_proposal(
    *,
    parent_row: Mapping[str, Any],
    example: ExampleRecord,
    strong: CandidateNote,
    weak: CandidateNote,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    task_profile_id: str,
    artifact_label: str,
    calibration_guidance: str,
    recursive_config: RecursiveDiscoveryConfig,
    state: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    state["recursive_parents_considered"] += 1
    current_depth = int(parent_row.get("recursion_depth", 0) or 0)
    recursion_reason = _decomposition_reason_for_proposal(
        parent_row,
        calibration_guidance=calibration_guidance,
    )
    if (
        not recursive_config.enabled
        or current_depth >= recursive_config.max_depth
        or not recursion_reason
        or state["recursive_calls"] >= recursive_config.max_recursive_calls_per_pair
        or state["recursive_parents_expanded"] >= recursive_config.max_recursive_parents_per_pair
    ):
        return [dict(parent_row)], [], []

    state["recursive_calls"] += 1
    state["recursive_parents_expanded"] += 1
    raw_children, cache_hit, parse_error = _decompose_criterion_for_pair(
        example=example,
        strong=strong,
        weak=weak,
        parent_row=parent_row,
        model_spec=model_spec,
        router=router,
        cache=cache,
        max_children=recursive_config.max_children_per_parent,
        task_profile_id=task_profile_id,
        artifact_label=artifact_label,
        calibration_guidance=calibration_guidance,
    )
    if cache_hit:
        state["recursive_cache_hits"] += 1
    if parse_error:
        state["recursive_parse_failures"] += 1
    state["recursive_children_raw_total"] += len(raw_children)

    annotated_children = _apply_recursive_provenance(
        raw_children,
        root_pair_id=str(parent_row.get("root_pair_id") or parent_row.get("pair_id") or pair_id(strong, weak)),
        recursion_depth=current_depth + 1,
        parent_criterion_id=str(parent_row.get("criterion_id", "")).strip() or None,
        recursion_reason=recursion_reason,
        decomposition_source="criterion_decomposition",
    )
    promoted_children, rejected_children, grounding = filter_pair_grounded_proposals(
        strong=strong,
        weak=weak,
        proposals=annotated_children,
        task_profile_id=task_profile_id,
    )
    state["recursive_children_promoted"] += len(promoted_children)
    state["recursive_children_rejected_grounding"] += len(rejected_children)
    recursive_step = {
        "parent_criterion_id": parent_row.get("criterion_id"),
        "parent_label": str(parent_row.get("label", "")),
        "parent_requirement": str(parent_row.get("requirement", "")),
        "recursion_depth": current_depth + 1,
        "recursion_reason": recursion_reason,
        "cache": "hit" if cache_hit else "miss",
        "parse_error": parse_error,
        "grounding": grounding,
        "raw_child_proposals": annotated_children,
        "promoted_child_proposals": promoted_children,
        "rejected_child_proposals": rejected_children,
        "changed_structure": bool(promoted_children),
    }
    rejected_all = list(rejected_children)
    if not promoted_children:
        return [dict(parent_row)], [recursive_step], rejected_all

    if current_depth + 1 >= recursive_config.max_depth:
        return promoted_children, [recursive_step], rejected_all

    leaves: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = [recursive_step]
    for child in promoted_children:
        child_leaves, child_steps, child_rejected = _expand_recursive_proposal(
            parent_row=child,
            example=example,
            strong=strong,
            weak=weak,
            model_spec=model_spec,
            router=router,
            cache=cache,
            task_profile_id=task_profile_id,
            artifact_label=artifact_label,
            calibration_guidance=calibration_guidance,
            recursive_config=recursive_config,
            state=state,
        )
        leaves.extend(child_leaves)
        steps.extend(child_steps)
        rejected_all.extend(child_rejected)
    return leaves, steps, rejected_all


def discover_pair_criteria(
    *,
    example: ExampleRecord,
    strong: CandidateNote,
    weak: CandidateNote,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    max_criteria: int,
    task_profile_id: str = "note_documentation",
    artifact_label: str = "note",
    calibration_guidance: str = "",
    recursive_config: Optional[RecursiveDiscoveryConfig] = None,
) -> Dict[str, Any]:
    pid = pair_id(strong, weak)
    recursive_config = recursive_config or RecursiveDiscoveryConfig()
    raw_root, root_cache_hit, root_parse_error = propose_criteria_for_pair(
        example=example,
        strong=strong,
        weak=weak,
        model_spec=model_spec,
        router=router,
        cache=cache,
        max_criteria=max_criteria,
        task_profile_id=task_profile_id,
        artifact_label=artifact_label,
        calibration_guidance=calibration_guidance,
    )
    raw_root = _apply_recursive_provenance(
        raw_root,
        root_pair_id=pid,
        recursion_depth=0,
        parent_criterion_id=None,
        recursion_reason="",
        decomposition_source="root_pair_discovery",
    )
    promoted_root, rejected_root, root_grounding = filter_pair_grounded_proposals(
        strong=strong,
        weak=weak,
        proposals=raw_root,
        task_profile_id=task_profile_id,
    )

    recursion_state = {
        "recursive_calls": 0,
        "recursive_cache_hits": 0,
        "recursive_parse_failures": 0,
        "recursive_parents_considered": 0,
        "recursive_parents_expanded": 0,
        "recursive_children_raw_total": 0,
        "recursive_children_promoted": 0,
        "recursive_children_rejected_grounding": 0,
    }
    final_proposals: List[Dict[str, Any]] = []
    recursive_steps: List[Dict[str, Any]] = []
    all_rejected: List[Dict[str, Any]] = list(rejected_root)
    for row in promoted_root:
        leaves, steps, rejected_children = _expand_recursive_proposal(
            parent_row=row,
            example=example,
            strong=strong,
            weak=weak,
            model_spec=model_spec,
            router=router,
            cache=cache,
            task_profile_id=task_profile_id,
            artifact_label=artifact_label,
            calibration_guidance=calibration_guidance,
            recursive_config=recursive_config,
            state=recursion_state,
        )
        final_proposals.extend(leaves)
        recursive_steps.extend(steps)
        all_rejected.extend(rejected_children)

    changed_structure = any(bool(step.get("changed_structure")) for step in recursive_steps)
    return {
        "pair_id": pid,
        "raw_proposals": raw_root,
        "promoted_root_proposals": promoted_root,
        "rejected_root_proposals": rejected_root,
        "proposals": final_proposals,
        "rejected_proposals": all_rejected,
        "grounding": root_grounding,
        "cache": "hit" if root_cache_hit else "miss",
        "parse_error": root_parse_error,
        "recursive_steps": recursive_steps,
        "recursion": {
            **recursion_state,
            "changed_structure": changed_structure,
        },
        "raw_proposals_total": len(raw_root) + recursion_state["recursive_children_raw_total"],
        "promoted_proposals_total": len(final_proposals),
        "rejected_proposals_total": len(all_rejected),
    }

def _normalize_key_part(s: str) -> str:
    t = (s or "").strip().lower()
    return _WS_RE.sub(" ", t)


def merge_proposal_entries(local_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Global merge / dedup: group by normalized dimension + severity + label + requirement text.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for row in local_rows:
        dim = _normalize_key_part(str(row.get("dimension", "")))
        sev = _normalize_key_part(str(row.get("severity_tier", "")))
        lab = _normalize_key_part(str(row.get("label", "")))
        req = _normalize_key_part(str(row.get("requirement", "")))
        key = f"{dim}||{sev}||{lab}||{req}"
        if key not in buckets:
            buckets[key] = {
                "merge_key": key,
                "dimension": row.get("dimension", ""),
                "label": row.get("label", ""),
                "requirement": row.get("requirement", ""),
                "severity_tier": row.get("severity_tier", ""),
                "count": 0,
                "example_ids": [],
                "pair_ids": [],
                "criterion_ids": [],
                "parent_criterion_ids": [],
                "root_pair_ids": [],
                "recursion_depths": [],
                "recursion_reasons": [],
                "decomposition_sources": [],
            }
            order.append(key)
        b = buckets[key]
        b["count"] += 1
        ex = row.get("example_id")
        if isinstance(ex, str) and ex and ex not in b["example_ids"]:
            b["example_ids"].append(ex)
        pid = row.get("pair_id")
        if isinstance(pid, str) and pid and pid not in b["pair_ids"]:
            b["pair_ids"].append(pid)
        criterion_id = row.get("criterion_id")
        if isinstance(criterion_id, str) and criterion_id and criterion_id not in b["criterion_ids"]:
            b["criterion_ids"].append(criterion_id)
        parent_id = row.get("parent_criterion_id")
        if isinstance(parent_id, str) and parent_id and parent_id not in b["parent_criterion_ids"]:
            b["parent_criterion_ids"].append(parent_id)
        root_pair_id = row.get("root_pair_id")
        if isinstance(root_pair_id, str) and root_pair_id and root_pair_id not in b["root_pair_ids"]:
            b["root_pair_ids"].append(root_pair_id)
        recursion_depth = row.get("recursion_depth")
        if isinstance(recursion_depth, int) and recursion_depth not in b["recursion_depths"]:
            b["recursion_depths"].append(recursion_depth)
        recursion_reason = row.get("recursion_reason")
        if isinstance(recursion_reason, str) and recursion_reason and recursion_reason not in b["recursion_reasons"]:
            b["recursion_reasons"].append(recursion_reason)
        decomposition_source = row.get("decomposition_source")
        if (
            isinstance(decomposition_source, str)
            and decomposition_source
            and decomposition_source not in b["decomposition_sources"]
        ):
            b["decomposition_sources"].append(decomposition_source)

    canonical = [buckets[k] for k in order]
    return {
        "schema": "compiled_discovery_merge_v1",
        "disclaimer": "Starter merge — simple dedup of promoted recursive leaves; not ontology integration.",
        "canonical_proposals": canonical,
        "unique_canonical_count": len(canonical),
        "total_local_proposals": len(local_rows),
    }


def merge_proposal_entries_with_rrd_filters(
    local_rows: List[Dict[str, Any]],
    *,
    pair_contexts: Optional[Sequence[Any]] = None,
    redundancy_threshold: float = 0.9,
    enable_misalignment: bool = True,
    enable_redundancy: bool = True,
) -> Dict[str, Any]:
    """
    RRD-aware variant of :func:`merge_proposal_entries`.

    The base merger dedupes canonical proposals lexically. This variant additionally runs the
    misalignment and redundancy filters from
    :mod:`rubric_gen.compiled.rrd_filters` on the merged rows so only direction-consistent,
    non-correlated rows reach downstream scoring.
    """
    from rubric_gen.compiled.rrd_filters import apply_rrd_filters

    merged = merge_proposal_entries(local_rows)
    canonical = list(merged.get("canonical_proposals", []) or [])
    pre_filter_count = len(canonical)
    filtered, stats = apply_rrd_filters(
        canonical,
        pair_contexts=tuple(pair_contexts or ()),
        redundancy_threshold=redundancy_threshold,
        enable_misalignment=enable_misalignment,
        enable_redundancy=enable_redundancy,
    )
    merged["canonical_proposals"] = filtered
    merged["unique_canonical_count"] = len(filtered)
    merged["rrd_filters"] = {
        "schema": "compiled_discovery_rrd_filters_v1",
        "pre_filter_count": pre_filter_count,
        "post_filter_count": len(filtered),
        "enable_misalignment": bool(enable_misalignment),
        "enable_redundancy": bool(enable_redundancy),
        "redundancy_threshold": float(redundancy_threshold),
        "stats": stats.to_dict(),
    }
    return merged


def run_discovery_for_examples(
    examples: List[ExampleRecord],
    *,
    run_dir: Path,
    model_override: Optional[str],
    use_cache: bool,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    task_profile: Optional[str] = None,
    bootstrap_iterations: int = 3,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Run discovery for loaded examples; write per-example JSON and return (run_dir, aggregate stats dict).
    """
    run_dir = Path(run_dir)
    ex_dir = run_dir / "examples"
    sum_dir = run_dir / "summaries"
    cache_dir = run_dir / "cache"
    ex_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    spec = resolve_compiled_judge_spec(model_override)
    router = LLMRouter()
    cache = JsonlCache(cache_dir / "compiled_discovery.jsonl", enabled=use_cache)
    profile_resolution = resolve_or_bootstrap_task_profile(
        examples,
        explicit=task_profile,
        bootstrap_iterations=bootstrap_iterations,
    )
    resolved_task_profile = profile_resolution.profile.task_profile_id
    recursive_config = RecursiveDiscoveryConfig()

    all_local: List[Dict[str, Any]] = []
    stats = {
        "examples_total": len(examples),
        "examples_with_pairs": 0,
        "pairs_total": 0,
        "pairs_succeeded": 0,
        "pairs_failed_parse": 0,
        "cache_hits": 0,
        "local_proposals_total": 0,
        "local_proposals_promoted": 0,
        "local_proposals_rejected_grounding": 0,
        "recursive_calls": 0,
        "recursive_cache_hits": 0,
        "recursive_parse_failures": 0,
        "recursive_parents_considered": 0,
        "recursive_parents_expanded": 0,
        "recursive_children_raw_total": 0,
        "recursive_children_promoted": 0,
        "recursive_children_rejected_grounding": 0,
        "pairs_with_recursive_change": 0,
        "task_profile_id": resolved_task_profile,
        "profile_bootstrap_used": profile_resolution.bootstrap_used,
    }

    for ex in examples:
        profile = profile_resolution.profile
        candidates = build_task_contrast_candidates(ex, task_profile_id=profile.task_profile_id)
        strong = select_strong_candidate(candidates, source_priority=profile.strong_source_priority)
        weak_list = (
            select_weak_candidates(candidates, strong, source_priority=profile.strong_source_priority)
            if strong
            else []
        )
        if strong and max_pairs_per_example is not None:
            weak_list = weak_list[: max_pairs_per_example]

        pairs_payload: List[Dict[str, Any]] = []
        if not strong or not weak_list:
            safe = ex.example_id.replace("/", "_")
            write_json(
                ex_dir / f"{safe}.json",
                {
                    "schema": "compiled_discovery_example_v1",
                    "disclaimer": (
                        "Starter discovery scaffold — shallow recursive decomposition enabled. "
                        "Skipped: no strong/weak pair available."
                    ),
                    "example_id": ex.example_id,
                    "task_profile_id": profile.task_profile_id,
                    "pairs": [],
                    "skip_reason": "no_strong_weak_pair",
                },
            )
            continue

        stats["examples_with_pairs"] += 1
        for weak in weak_list:
            stats["pairs_total"] += 1
            pair_result = discover_pair_criteria(
                example=ex,
                strong=strong,
                weak=weak,
                model_spec=spec,
                router=router,
                cache=cache,
                max_criteria=max_criteria,
                task_profile_id=profile.task_profile_id,
                artifact_label=profile.artifact_label,
                recursive_config=recursive_config,
            )
            if pair_result["cache"] == "hit":
                stats["cache_hits"] += 1
            if pair_result["parse_error"]:
                stats["pairs_failed_parse"] += 1
            else:
                stats["pairs_succeeded"] += 1
            stats["local_proposals_total"] += int(pair_result["raw_proposals_total"])
            stats["local_proposals_promoted"] += int(pair_result["promoted_proposals_total"])
            stats["local_proposals_rejected_grounding"] += int(pair_result["rejected_proposals_total"])
            recursion_stats = dict(pair_result.get("recursion", {}))
            stats["recursive_calls"] += int(recursion_stats.get("recursive_calls", 0) or 0)
            stats["recursive_cache_hits"] += int(recursion_stats.get("recursive_cache_hits", 0) or 0)
            stats["recursive_parse_failures"] += int(recursion_stats.get("recursive_parse_failures", 0) or 0)
            stats["recursive_parents_considered"] += int(recursion_stats.get("recursive_parents_considered", 0) or 0)
            stats["recursive_parents_expanded"] += int(recursion_stats.get("recursive_parents_expanded", 0) or 0)
            stats["recursive_children_raw_total"] += int(recursion_stats.get("recursive_children_raw_total", 0) or 0)
            stats["recursive_children_promoted"] += int(recursion_stats.get("recursive_children_promoted", 0) or 0)
            stats["recursive_children_rejected_grounding"] += int(
                recursion_stats.get("recursive_children_rejected_grounding", 0) or 0
            )
            if bool(recursion_stats.get("changed_structure")):
                stats["pairs_with_recursive_change"] += 1
            all_local.extend(pair_result["proposals"])

            pairs_payload.append(
                {
                    "pair_id": pair_result["pair_id"],
                    "strong_candidate_id": strong.candidate_id,
                    "weak_candidate_id": weak.candidate_id,
                    "strong_source_label": strong.source_label,
                    "weak_source_label": weak.source_label,
                    "task_profile_id": profile.task_profile_id,
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

        safe = ex.example_id.replace("/", "_")
        write_json(
            ex_dir / f"{safe}.json",
            {
                "schema": "compiled_discovery_example_v1",
                "disclaimer": "Starter discovery scaffold — shallow recursive decomposition enabled.",
                "example_id": ex.example_id,
                "task_profile_id": profile.task_profile_id,
                "strong_anchor": {
                    "candidate_id": strong.candidate_id,
                    "source_label": strong.source_label,
                },
                "pairs": pairs_payload,
            },
        )

    merged = merge_proposal_entries(all_local)
    write_json(sum_dir / "merged_proposals.json", merged)

    run_summary = {
        "schema": "compiled_discovery_run_summary_v1",
        "disclaimer": (
            "Starter local rubric discovery — proposes criteria from strong/weak pairs with shallow recursive "
            "decomposition; does not update compiled ontology."
        ),
        "model": f"{spec.provider}:{spec.model}",
        "prompt_version": DISCOVERY_PROMPT_VERSION,
        "params": {
            "max_criteria": max_criteria,
            "max_pairs_per_example": max_pairs_per_example,
            "cache_enabled": use_cache,
            "task_profile": resolved_task_profile,
            "bootstrap_iterations": bootstrap_iterations,
            "recursive_config": {
                "enabled": recursive_config.enabled,
                "max_depth": recursive_config.max_depth,
                "max_recursive_parents_per_pair": recursive_config.max_recursive_parents_per_pair,
                "max_children_per_parent": recursive_config.max_children_per_parent,
                "max_recursive_calls_per_pair": recursive_config.max_recursive_calls_per_pair,
            },
        },
        "profile_resolution": {
            "bootstrap_used": profile_resolution.bootstrap_used,
            "iterations_run": profile_resolution.iterations_run,
            "resolved_task_profile_id": profile_resolution.profile.task_profile_id,
            "parent_profile_id": profile_resolution.profile.parent_profile_id,
            "diagnostics": profile_resolution.diagnostics,
        },
        "stats": stats,
        "merged": {
            "unique_canonical_count": merged["unique_canonical_count"],
            "total_local_proposals": merged["total_local_proposals"],
            "raw_local_proposals": stats["local_proposals_total"],
            "promoted_local_proposals": stats["local_proposals_promoted"],
            "rejected_local_proposals": stats["local_proposals_rejected_grounding"],
        },
    }
    write_json(sum_dir / "run_summary.json", run_summary)

    return run_dir, run_summary
