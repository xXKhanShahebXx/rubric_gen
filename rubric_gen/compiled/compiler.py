"""
Starter compiled-rubric compiler (scaffold only).

Uses a small ontology of reusable criterion templates, instantiates them per case when
dialogue / reference / structured_summary evidence supports them, and attaches evidence anchors.

This is intentionally heuristic — not a clinical-grade rules engine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rubric_gen.dataio import example_to_prompt, strongest_anchor_text
from rubric_gen.compiled.task_profiles import get_task_profile, resolve_task_profile, task_profile_archetype_id
from rubric_gen.types import ExampleRecord

from rubric_gen.compiled.schema import (
    AggregationPolicy,
    CaseRubric,
    CompiledCriterion,
    CriterionTemplate,
    DimensionSpec,
    DocumentationContract,
    ErrorTaxonomyEntry,
    EvidenceAnchor,
    NoteFamilySpec,
    RubricOntology,
    SectionSpec,
    SubdimensionSpec,
)

STARTER_ONTOLOGY_ID = "clinical_note_core"
STARTER_ONTOLOGY_VERSION = "v0_4"
STARTER_NOTE_FAMILY_VERSION = "v0_4"
STARTER_RUBRIC_VERSION = "v0_4"


def _snippet(text: str, needle: str, width: int = 220) -> Optional[str]:
    if not text or not needle:
        return None
    lower = text.lower()
    key = needle.lower()
    idx = lower.find(key)
    if idx < 0:
        return None
    start = max(0, idx - width // 2)
    end = min(len(text), start + width)
    snippet = text[start:end].strip()
    return snippet if snippet else None


def _discovered_requirement_text(template: CriterionTemplate) -> str:
    """Extract atomic requirement text embedded in provisional discovered template descriptions."""
    desc = template.description or ""
    m = re.search(r"\(support=\d+\)\s*(.+)$", desc, re.DOTALL)
    if m:
        return m.group(1).strip()
    return desc.strip()


def _first_matching_snippet(conversation: str, needles: Sequence[str]) -> Optional[Tuple[str, str]]:
    for needle in needles:
        s = _snippet(conversation, needle)
        if s:
            return needle, s
    return None


def _patient_text(conversation: str) -> str:
    """Roughly isolate patient turns for symptom / negative detection."""
    chunks: List[str] = []
    for line in conversation.splitlines():
        low = line.lower()
        if "[patient]" in low or "patient:" in low:
            chunks.append(line)
    return "\n".join(chunks) if chunks else conversation


def infer_note_family(example: ExampleRecord) -> str:
    """Coarse note-family id from layout cues in reference / augmented text."""
    aug = example.augmented_note or ""
    ref = example.reference_note or ""
    blob = f"{aug}\n{ref}"
    if re.search(r"(?m)^\s*S:\s*", aug) and re.search(r"(?m)^\s*O:\s*", aug):
        return "soap_note"
    if "CHIEF COMPLAINT" in ref.upper() and "ASSESSMENT" in ref.upper():
        return "structured_clinical_note"
    if re.search(r"(?m)^\s*S:\s*", blob) and re.search(r"(?m)^\s*O:\s*", blob):
        return "soap_note"
    return "general_clinical_note"


@dataclass
class _CaseSignals:
    """Lightweight per-case cues used to decide which templates to compile."""

    meal_pattern: bool
    duration_cue: bool
    severity_cue: bool
    location_cue: bool
    denies_vomiting: bool
    fever_discussed: bool
    imaging_or_procedure: bool
    lab_mentioned: bool
    cc_snippet: Optional[str]
    cc_terms: List[str]
    visit_motivation: Optional[str]


_IMAGING_TERMS = (
    "ultrasound",
    "sonogram",
    "ct scan",
    " ct ",
    "mri",
    "x-ray",
    " xray",
    "imaging",
    "echo",
    "ekg",
    "ecg",
)

_LAB_TERMS = (
    "lab",
    "labs",
    "blood work",
    "cbc",
    "bmp",
    "cmp",
    "culture",
    "troponin",
    "a1c",
    "hemoglobin",
)


def _visit_motivation_from_summary(structured_summary: Optional[Dict[str, Any]]) -> Optional[str]:
    if not structured_summary:
        return None
    vm = structured_summary.get("visit motivation")
    if isinstance(vm, str) and vm.strip() and vm.strip().lower() not in ("none",):
        return vm.strip()
    return None


def _collect_case_signals(example: ExampleRecord) -> _CaseSignals:
    conv = example.conversation or ""
    cl = conv.lower()
    patient_blob = _patient_text(conv).lower()

    meal_pattern = any(
        k in cl for k in ("after i eat", "after meals", "meal", "eating", "postprandial", "food")
    )
    duration_cue = bool(re.search(r"\b(weeks?|months?|days?|hours?|several)\b", cl))
    severity_cue = bool(re.search(r"\b(scale|eight|nine|ten|\b\d/\b|out of ten|severe)\b", cl))
    location_cue = any(
        k in cl for k in ("right", "left", "ribs", "quadrant", "belly", "chest", "abdomen", "epigastric")
    )
    denies_vomiting = ("haven't thrown up" in patient_blob) or ("have n't thrown up" in patient_blob)
    fever_discussed = "fever" in cl
    imaging_or_procedure = any(t in cl for t in _IMAGING_TERMS)
    lab_mentioned = any(t in cl for t in _LAB_TERMS)

    cc_terms: List[str] = []
    match = re.search(
        r"(?:for|about|with)\s+(?:this\s+)?(.{8,120}?)(?:\.|,|they|he did|she did|i've|i have|that)",
        cl,
        re.IGNORECASE,
    )
    cc_snippet = None
    if match:
        frag = match.group(1).strip()
        if len(frag) > 6:
            cc_snippet = _snippet(conv, frag[:40]) or frag[:200]
            cc_terms = [w for w in re.split(r"\W+", frag.lower()) if len(w) > 3][:8]

    visit_motivation = _visit_motivation_from_summary(example.structured_summary)

    return _CaseSignals(
        meal_pattern=meal_pattern,
        duration_cue=duration_cue,
        severity_cue=severity_cue,
        location_cue=location_cue,
        denies_vomiting=denies_vomiting,
        fever_discussed=fever_discussed,
        imaging_or_procedure=imaging_or_procedure,
        lab_mentioned=lab_mentioned,
        cc_snippet=cc_snippet,
        cc_terms=cc_terms,
        visit_motivation=visit_motivation,
    )


def build_starter_ontology() -> RubricOntology:
    return RubricOntology(
        ontology_id=STARTER_ONTOLOGY_ID,
        version=STARTER_ONTOLOGY_VERSION,
        severity_tiers=["catastrophic", "essential", "important", "optional"],
        dimensions=[
            DimensionSpec(
                dimension_id="dialogue_faithfulness",
                label="Dialogue Faithfulness",
                description="The note must remain grounded in the dialogue.",
                hard_gate_eligible=True,
                subdimensions=[
                    SubdimensionSpec("hallucination", "Hallucination"),
                    SubdimensionSpec("contradiction", "Contradiction"),
                ],
            ),
            DimensionSpec(
                dimension_id="clinical_completeness",
                label="Clinical Completeness",
                description="The note should capture salient clinical content from the encounter.",
                hard_gate_eligible=False,
                subdimensions=[
                    SubdimensionSpec("symptom_detail", "Symptom detail"),
                    SubdimensionSpec("ros_negatives", "Pertinent negatives"),
                    SubdimensionSpec("study_results", "Study results"),
                ],
            ),
            DimensionSpec(
                dimension_id="documentation_structure",
                label="Documentation Structure",
                description="The note should respect the expected section scaffold for its note family.",
                hard_gate_eligible=False,
                subdimensions=[
                    SubdimensionSpec("section_presence", "Section presence"),
                ],
            ),
            DimensionSpec(
                dimension_id="management_plan",
                label="Management Plan",
                description=(
                    "The note should capture grounded medications, interventions, planned testing, "
                    "follow-up timing, and return guidance from the encounter."
                ),
                hard_gate_eligible=False,
                subdimensions=[
                    SubdimensionSpec("follow_up_specificity", "Follow-up specificity"),
                    SubdimensionSpec("return_precautions", "Return precautions"),
                    SubdimensionSpec("medication_management", "Medication management"),
                    SubdimensionSpec("intervention_plan", "Intervention plan"),
                    SubdimensionSpec("testing_plan", "Testing plan"),
                    SubdimensionSpec("treatment_grounding", "Treatment grounding"),
                ],
            ),
            DimensionSpec(
                dimension_id="diagnostic_reasoning",
                label="Diagnostic Reasoning",
                description=(
                    "The note should connect the assessment and selected plan to supporting symptoms, "
                    "exam findings, or reviewed results."
                ),
                hard_gate_eligible=False,
                subdimensions=[
                    SubdimensionSpec("assessment_linkage", "Assessment linkage"),
                ],
            ),
        ],
        criterion_templates=[
            CriterionTemplate(
                template_id="no_unsupported_diagnosis",
                dimension_id="dialogue_faithfulness",
                subdimension_id="hallucination",
                label="No Unsupported High-Risk Inferences",
                description=(
                    "Do not introduce catastrophic-risk conditions or unsupported specifics absent from "
                    "the encounter narrative."
                ),
                severity_tier="catastrophic",
                default_verdict_type="binary",
                evidence_policy="dialogue_or_reference",
                hard_gate_default=True,
                typical_failure_codes=["unsupported_inference", "certainty_inflation"],
            ),
            CriterionTemplate(
                template_id="no_unsupported_certainty",
                dimension_id="dialogue_faithfulness",
                subdimension_id="hallucination",
                label="No Unsupported Diagnostic Certainty",
                description=(
                    "Avoid definitive diagnostic language for entities not clearly supported by dialogue, "
                    "exam, or reviewed results as discussed."
                ),
                severity_tier="catastrophic",
                default_verdict_type="binary",
                evidence_policy="dialogue_or_reference",
                hard_gate_default=True,
                typical_failure_codes=["certainty_inflation"],
            ),
            CriterionTemplate(
                template_id="required_sections_note_family",
                dimension_id="documentation_structure",
                subdimension_id="section_presence",
                label="Required Sections for Note Family",
                description="Include the section headings expected for this note family when applicable.",
                severity_tier="essential",
                default_verdict_type="binary",
                evidence_policy="note_layout",
                hard_gate_default=False,
                typical_failure_codes=["section_missing"],
            ),
            CriterionTemplate(
                template_id="chief_complaint_symptom_presence",
                dimension_id="clinical_completeness",
                subdimension_id="symptom_detail",
                label="Chief Complaint / Presenting Symptom Alignment",
                description="Reflect the presenting problem described in the visit when one is extractable.",
                severity_tier="essential",
                default_verdict_type="binary",
                evidence_policy="dialogue_only",
                hard_gate_default=False,
                typical_failure_codes=["omission"],
            ),
            CriterionTemplate(
                template_id="symptom_detail_attributes",
                dimension_id="clinical_completeness",
                subdimension_id="symptom_detail",
                label="Salient Symptom Attributes",
                description="Capture location, severity, duration, or trigger/pattern when discussed.",
                severity_tier="essential",
                default_verdict_type="binary",
                evidence_policy="dialogue_only",
                hard_gate_default=False,
                typical_failure_codes=["omission"],
            ),
            CriterionTemplate(
                template_id="associated_symptom_or_negative",
                dimension_id="clinical_completeness",
                subdimension_id="ros_negatives",
                label="Associated Symptom or Relevant Negative",
                description="When the dialogue documents an associated symptom or a pertinent negative, mirror it.",
                severity_tier="important",
                default_verdict_type="binary",
                evidence_policy="dialogue_only",
                hard_gate_default=False,
                typical_failure_codes=["omission", "negation_flip"],
            ),
            CriterionTemplate(
                template_id="discussed_study_or_result_presence",
                dimension_id="clinical_completeness",
                subdimension_id="study_results",
                label="Discussed Study or Result Presence",
                description="When imaging or labs are reviewed in the visit, document them in the note.",
                severity_tier="important",
                default_verdict_type="binary",
                evidence_policy="dialogue_or_reference",
                hard_gate_default=False,
                typical_failure_codes=["omission"],
            ),
        ],
        error_taxonomy=[
            ErrorTaxonomyEntry(
                error_code="unsupported_inference",
                label="Unsupported Inference",
                description="The note adds a conclusion not grounded in the evidence.",
                severity_tier="catastrophic",
            ),
            ErrorTaxonomyEntry(
                error_code="certainty_inflation",
                label="Certainty Inflation",
                description="Definitive language without matching encounter support.",
                severity_tier="catastrophic",
            ),
            ErrorTaxonomyEntry(
                error_code="omission",
                label="Omission",
                description="Clinically relevant dialogue content missing from the note.",
                severity_tier="important",
            ),
            ErrorTaxonomyEntry(
                error_code="section_missing",
                label="Section Missing",
                description="Expected documentation section not clearly present.",
                severity_tier="essential",
            ),
            ErrorTaxonomyEntry(
                error_code="negation_flip",
                label="Negation Flip",
                description="Pertinent negative or denial misrepresented.",
                severity_tier="important",
            ),
        ],
    )


def build_note_family_spec(note_family_id: str, ontology: RubricOntology) -> NoteFamilySpec:
    if note_family_id == "soap_note":
        contract = DocumentationContract(
            inference_policy="strict_grounded",
            uncertainty_policy="explicit_only",
            required_section_ids=["subjective", "objective", "assessment", "plan"],
            optional_section_ids=["medications"],
            style_rules=["Prefer concise clinical prose.", "Use section headers."],
        )
        sections = [
            SectionSpec(
                section_id="subjective",
                label="Subjective",
                required=True,
                allowed_content=["history", "symptoms", "patient_reported_context"],
            ),
            SectionSpec(
                section_id="objective",
                label="Objective",
                required=True,
                allowed_content=["vitals", "exam", "studies"],
            ),
            SectionSpec(
                section_id="assessment",
                label="Assessment",
                required=True,
                allowed_content=["diagnoses", "impression"],
            ),
            SectionSpec(
                section_id="plan",
                label="Plan",
                required=True,
                allowed_content=["follow_up", "procedures", "medications"],
            ),
        ]
        label = "SOAP Note"
    elif note_family_id == "structured_clinical_note":
        contract = DocumentationContract(
            inference_policy="strict_grounded",
            uncertainty_policy="explicit_only",
            required_section_ids=["chief_complaint", "assessment_plan"],
            optional_section_ids=["history", "physical_exam", "results"],
            style_rules=[
                "Use clear section headers.",
                "Keep assessment and plan aligned with the visit narrative.",
            ],
        )
        sections = [
            SectionSpec("chief_complaint", "Chief complaint", True, ["symptoms", "reason_for_visit"]),
            SectionSpec("assessment_plan", "Assessment and plan", True, ["diagnoses", "next_steps"]),
        ]
        label = "Structured clinical note"
    else:
        contract = DocumentationContract(
            inference_policy="strict_grounded",
            uncertainty_policy="explicit_only",
            required_section_ids=["encounter_summary"],
            optional_section_ids=[],
            style_rules=["Ground statements in the dialogue.", "Avoid adding unsupported specifics."],
        )
        sections = [
            SectionSpec("encounter_summary", "Encounter summary", True, ["history", "exam", "plan"]),
        ]
        label = "General clinical note"

    family_ids = [
        t.template_id
        for t in ontology.criterion_templates
        if _template_applies_to_note_family(t, note_family_id)
    ]

    return NoteFamilySpec(
        note_family_id=note_family_id,
        version=STARTER_NOTE_FAMILY_VERSION,
        label=label,
        documentation_contract=contract,
        section_specs=sections,
        family_template_ids=family_ids,
        hard_gate_error_codes=["unsupported_inference", "certainty_inflation"],
        task_profile_id="note_documentation",
        task_family_id=note_family_id,
        artifact_label="note",
    )


def _template_applies_to_note_family(t: CriterionTemplate, note_family_id: str) -> bool:
    """Discovered templates may be scoped to one note family; base templates apply everywhere."""
    scope = t.task_family_scope or t.note_family_scope
    if not scope:
        return True
    return note_family_id in scope


def _section_header_patterns(note_family_id: str) -> List[Dict[str, Any]]:
    """Regex patterns matched against lines of the candidate note (starter heuristic)."""
    if note_family_id == "soap_note":
        return [
            {
                "section_id": "subjective",
                "label": "Subjective / S",
                "patterns": [r"(?m)^\s*S:\s*", r"(?mi)^\s*\*{0,2}\s*subjective"],
            },
            {
                "section_id": "objective",
                "label": "Objective / O",
                "patterns": [r"(?m)^\s*O:\s*", r"(?mi)^\s*\*{0,2}\s*objective"],
            },
            {
                "section_id": "assessment",
                "label": "Assessment / A",
                "patterns": [r"(?m)^\s*A:\s*", r"(?mi)^\s*\*{0,2}\s*assessment"],
            },
            {
                "section_id": "plan",
                "label": "Plan / P",
                "patterns": [r"(?m)^\s*P:\s*", r"(?mi)^\s*\*{0,2}\s*plan"],
            },
        ]
    if note_family_id == "structured_clinical_note":
        return [
            {
                "section_id": "chief_complaint",
                "label": "Chief complaint",
                "patterns": [r"(?mi)chief\s+complaint"],
            },
            {
                "section_id": "assessment_plan",
                "label": "Assessment and plan",
                "patterns": [r"(?mi)assessment\s+and\s+plan", r"(?mi)^\s*assessment\b"],
            },
        ]
    return [
        {
            "section_id": "encounter_summary",
            "label": "Encounter summary",
            "patterns": [r"(?mi)summary", r"(?mi)encounter", r"(?mi)clinical\s+note"],
        },
    ]


def compile_case_rubric(
    example: ExampleRecord,
    ontology: RubricOntology,
    note_family: NoteFamilySpec,
) -> CaseRubric:
    """Compile a case rubric: multiple templates, gated by case signals, with dialogue anchors."""
    ex_id = example.example_id
    conv = example.conversation or ""
    signals = _collect_case_signals(example)

    hard_gates: List[CompiledCriterion] = []
    soft_checks: List[CompiledCriterion] = []

    diag_anchor = _first_matching_snippet(conv, ["gallstones", "cholecystitis", "ultrasound", "pain in my belly"])
    diag_quote = (diag_anchor[1] if diag_anchor else None) or _snippet(conv, "pain") or (conv[:220] + "…" if len(conv) > 220 else conv)

    hard_gates.append(
        CompiledCriterion(
            criterion_id=f"{ex_id}_no_unsupported_diagnosis",
            template_id="no_unsupported_diagnosis",
            dimension_id="dialogue_faithfulness",
            subdimension_id="hallucination",
            label="No Unsupported High-Risk Inferences",
            requirement=(
                "Do not introduce high-risk conditions or critical specifics that never appear in the "
                "dialogue or reviewed encounter content."
            ),
            severity_tier="catastrophic",
            verdict_type="binary",
            evidence_anchors=[EvidenceAnchor(source="dialogue", quote=diag_quote or "")],
            compile_rationale="Universal starter gate; anchored to salient dialogue evidence for this case.",
            eval_kind="hard_gate_unsupported_bundle",
            judge_hints={"check": "high_risk_terms_in_note_vs_dialogue"},
        )
    )

    # Second hard gate: certainty inflation — when dialogue uses probabilistic / belief language.
    _hedge = re.compile(
        r"\b(believe|think|likely|possible|might|maybe|suspect|concern|could be|appears)\b",
        re.IGNORECASE,
    )
    if _hedge.search(conv):
        cert_quote = _snippet(conv, "believe") or _snippet(conv, "might") or _snippet(conv, "possible") or diag_quote
        hard_gates.append(
            CompiledCriterion(
                criterion_id=f"{ex_id}_no_unsupported_certainty",
                template_id="no_unsupported_certainty",
                dimension_id="dialogue_faithfulness",
                subdimension_id="hallucination",
                label="No Unsupported Diagnostic Certainty",
                requirement=(
                    "Do not state diagnoses or complications as definitively established when the encounter "
                    "language is probabilistic or hedged, unless exam or reviewed results justify certainty."
                ),
                severity_tier="catastrophic",
                verdict_type="binary",
                evidence_anchors=[EvidenceAnchor(source="dialogue", quote=cert_quote or "")],
                compile_rationale="Dialogue contains hedging language; flag definitive chart language that overshoots evidence.",
                eval_kind="hard_gate_certainty_vs_dialogue",
                judge_hints={"hedge_detected_in_dialogue": True},
            )
        )

    # Required sections (soft): always compile for transparency; judge may mark N/A for sparse notes.
    sec_patterns = _section_header_patterns(note_family.note_family_id)
    soft_checks.append(
        CompiledCriterion(
            criterion_id=f"{ex_id}_required_sections",
            template_id="required_sections_note_family",
            dimension_id="documentation_structure",
            subdimension_id="section_presence",
            label=f"Required Sections ({note_family.label})",
            requirement=(
                f"For note family `{note_family.note_family_id}`, include clear section coverage for: "
                f"{', '.join(note_family.documentation_contract.required_section_ids)}."
            ),
            severity_tier="essential",
            verdict_type="binary",
            evidence_anchors=[
                EvidenceAnchor(
                    source="note_family_spec",
                    quote=json.dumps([s["section_id"] for s in sec_patterns]),
                )
            ],
            compile_rationale="Instantiated from note-family required sections (header patterns are heuristic).",
            eval_kind="section_header_coverage",
            judge_hints={"sections": sec_patterns, "note_family_id": note_family.note_family_id},
        )
    )

    # Chief complaint alignment
    cc_terms = list(signals.cc_terms)
    if signals.visit_motivation:
        vm = signals.visit_motivation.lower()
        cc_terms.extend([w for w in re.split(r"\W+", vm) if len(w) > 3])
    cc_terms = list(dict.fromkeys([t for t in cc_terms if t]))[:6]
    if cc_terms or signals.cc_snippet:
        cq = signals.cc_snippet or _snippet(conv, cc_terms[0]) if cc_terms else diag_quote
        soft_checks.append(
            CompiledCriterion(
                criterion_id=f"{ex_id}_chief_complaint_alignment",
                template_id="chief_complaint_symptom_presence",
                dimension_id="clinical_completeness",
                subdimension_id="symptom_detail",
                label="Chief Complaint / Presenting Symptom Alignment",
                requirement="Document the presenting problem using terms consistent with the patient narrative.",
                severity_tier="essential",
                verdict_type="binary",
                evidence_anchors=[EvidenceAnchor(source="dialogue", quote=cq or "")],
                compile_rationale="Compiled because a chief-complaint phrase or structured visit motivation was available.",
                eval_kind="dialogue_anchor_terms_in_note",
                judge_hints={"terms": cc_terms or ["pain", "abdominal"], "min_terms_matched": 1},
            )
        )

    # Symptom detail bundle — compile sub-checks only when signals fire.
    detail_hints: Dict[str, Any] = {"patterns": []}
    if signals.meal_pattern:
        mq = (
            _snippet(conv, "after i eat")
            or _snippet(conv, "eating")
            or _snippet(conv, "meal")
            or diag_quote
        )
        detail_hints["patterns"].append(
            {
                "id": "meal_related",
                "label": "Meal-related pattern",
                "dialogue_detected": True,
                "note_keywords": [
                    "after eating",
                    "after meals",
                    "postprandial",
                    "meal",
                    "meals",
                    "eating",
                    "related to eating",
                    "with meals",
                    "high fat",
                    "fatty",
                    "diet",
                    "food",
                    "foods",
                ],
                "anchor_quote": mq,
            }
        )
    if signals.location_cue and signals.severity_cue:
        detail_hints["patterns"].append(
            {
                "id": "location_severity",
                "label": "Location and severity",
                "dialogue_detected": True,
                "note_keywords": ["right", "rib", "quadrant", "8", "severe", "scale"],
                "require_min_keyword_hits": 2,
                "anchor_quote": _snippet(conv, "ribs") or _snippet(conv, "eight") or diag_quote,
            }
        )
    if signals.duration_cue:
        detail_hints["patterns"].append(
            {
                "id": "duration",
                "label": "Duration or time course",
                "dialogue_detected": True,
                "note_keywords": ["week", "month", "day", "several"],
                "require_min_keyword_hits": 1,
                "anchor_quote": _snippet(conv, "several weeks") or _snippet(conv, "weeks") or diag_quote,
            }
        )

    if detail_hints["patterns"]:
        anchors: List[EvidenceAnchor] = []
        for p in detail_hints["patterns"]:
            q = p.get("anchor_quote") or diag_quote or ""
            if q:
                anchors.append(EvidenceAnchor(source="dialogue", quote=q))
        if not anchors:
            anchors = [EvidenceAnchor(source="dialogue", quote=diag_quote or "")]
        soft_checks.append(
            CompiledCriterion(
                criterion_id=f"{ex_id}_symptom_detail_attributes",
                template_id="symptom_detail_attributes",
                dimension_id="clinical_completeness",
                subdimension_id="symptom_detail",
                label="Salient Symptom Attributes Discussed",
                requirement="When timing, location, severity, or triggers are discussed, reflect them in the note.",
                severity_tier="essential",
                verdict_type="binary",
                evidence_anchors=anchors[:4],
                compile_rationale="Compiled from case signals (meal pattern, duration, location/severity cues).",
                eval_kind="symptom_detail_bundle",
                judge_hints=detail_hints,
            )
        )

    # Pertinent negative: vomiting denial
    if signals.denies_vomiting:
        vq = _snippet(conv, "thrown up") or _snippet(conv, "vomit")
        soft_checks.append(
            CompiledCriterion(
                criterion_id=f"{ex_id}_pertinent_negative_vomiting",
                template_id="associated_symptom_or_negative",
                dimension_id="clinical_completeness",
                subdimension_id="ros_negatives",
                label="Document Vomiting Denial (when discussed)",
                requirement="If the patient denies vomiting, document that pertinent negative.",
                severity_tier="important",
                verdict_type="binary",
                evidence_anchors=[EvidenceAnchor(source="dialogue", quote=vq or "")],
                compile_rationale="Patient denied vomiting in dialogue; mirror as a relevant negative.",
                eval_kind="relevant_negative_mirror",
                judge_hints={
                    "dialogue_markers": ["haven't thrown up", "have n't thrown up", "denies vomiting", "no vomiting"],
                    "note_markers": ["denies vomiting", "deny vomiting", "no vomiting", "denied vomiting", "has not vomited"],
                },
            )
        )

    # Fever associated symptom
    if signals.fever_discussed:
        fq = _snippet(conv, "fever")
        soft_checks.append(
            CompiledCriterion(
                criterion_id=f"{ex_id}_associated_symptom_fever",
                template_id="associated_symptom_or_negative",
                dimension_id="clinical_completeness",
                subdimension_id="ros_negatives",
                label="Document Fever Symptom (when discussed)",
                requirement="If fevers are discussed, document fever / low-grade fever status in the note.",
                severity_tier="important",
                verdict_type="binary",
                evidence_anchors=[EvidenceAnchor(source="dialogue", quote=fq or "")],
                compile_rationale="Fever mentioned in dialogue; should appear in ROS or HPI.",
                eval_kind="anchor_terms_presence",
                judge_hints={
                    "terms": ["fever", "febrile", "afebrile", "temperature"],
                    "min_terms_matched": 1,
                },
            )
        )

    # Imaging / labs discussed
    study_terms: List[str] = []
    if signals.imaging_or_procedure:
        for t in _IMAGING_TERMS:
            if t.strip() in conv.lower():
                study_terms.append(t.strip())
    if signals.lab_mentioned:
        for t in _LAB_TERMS:
            if t in conv.lower():
                study_terms.append(t)
    study_terms = list(dict.fromkeys(study_terms))[:5]

    if study_terms:
        sq = _snippet(conv, study_terms[0]) or diag_quote
        soft_checks.append(
            CompiledCriterion(
                criterion_id=f"{ex_id}_discussed_studies_results",
                template_id="discussed_study_or_result_presence",
                dimension_id="clinical_completeness",
                subdimension_id="study_results",
                label="Discussed Imaging or Labs in Note",
                requirement="When imaging or labs are discussed in the visit, ensure the note references them.",
                severity_tier="important",
                verdict_type="binary",
                evidence_anchors=[EvidenceAnchor(source="dialogue", quote=sq or "")],
                compile_rationale="Encounter discusses imaging and/or labs; compile a documentation presence check.",
                eval_kind="discussed_study_mentioned_in_note",
                judge_hints={"study_terms": study_terms},
            )
        )

    # Provisional discovered templates (additive soft checks; LLM judge is the primary path).
    for tmpl in ontology.criterion_templates:
        if not tmpl.provisional_discovered:
            continue
        sc = tmpl.task_family_scope or tmpl.note_family_scope
        if sc and note_family.note_family_id not in sc:
            continue
        req_text = _discovered_requirement_text(tmpl)
        if not req_text:
            continue
        needle = req_text.split()[:8]
        anchor_q = None
        for w in needle:
            if len(w) > 3:
                anchor_q = _snippet(conv, w)
                if anchor_q:
                    break
        anchor_q = anchor_q or _snippet(conv, req_text[:48]) or (req_text[:320] + ("…" if len(req_text) > 320 else ""))

        soft_checks.append(
            CompiledCriterion(
                criterion_id=f"{ex_id}_{tmpl.template_id}",
                template_id=tmpl.template_id,
                dimension_id=tmpl.dimension_id,
                subdimension_id=tmpl.subdimension_id,
                label=tmpl.label,
                requirement=req_text,
                severity_tier=tmpl.severity_tier,
                verdict_type="binary",
                evidence_anchors=[
                    EvidenceAnchor(source="discovery_proposal", quote=anchor_q or ""),
                ],
                compile_rationale=(
                    "Provisional discovered template from local strong/weak merge; "
                    "heuristic path defaults to pass — prefer LLM analytic judge."
                ),
                eval_kind="discovered_provisional_soft",
                judge_hints={
                    "provisional_discovered": True,
                    "note_family_scope": sc,
                    "task_family_scope": sc,
                },
            )
        )

    aggregation = AggregationPolicy(
        hard_gate_policy="all_must_pass",
        minimum_soft_score=0.75,
        soft_dimension_weights={
            "clinical_completeness": 0.30,
            "documentation_structure": 0.15,
            "dialogue_faithfulness": 0.20,
            "management_plan": 0.20,
            "diagnostic_reasoning": 0.15,
        },
    )

    return CaseRubric(
        rubric_id=f"{ex_id}_rubric_{STARTER_RUBRIC_VERSION}",
        version=STARTER_RUBRIC_VERSION,
        example_id=ex_id,
        note_family_id=note_family.note_family_id,
        ontology_version=ontology.version,
        note_family_version=note_family.version,
        hard_gates=hard_gates,
        soft_checks=soft_checks,
        aggregation=aggregation,
        task_profile_id="note_documentation",
        task_family_id=note_family.note_family_id,
        artifact_label="note",
    )


_GENERIC_PROFILE_VERSION = "v0_1"
_GENERIC_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")
_GENERIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}
_GENERIC_REASONING_MARKERS = (
    "because",
    "due to",
    "based on",
    "supported by",
    "evidence",
    "rationale",
    "reasoning",
)
_GENERIC_ACTION_MARKERS = (
    "plan",
    "recommend",
    "recommendation",
    "next step",
    "follow-up",
    "follow up",
    "schedule",
    "return if",
    "call if",
    "continue",
    "start",
    "stop",
)
_GENERIC_TOOL_MARKERS = (
    "tool",
    "search",
    "result",
    "observation",
    "stdout",
    "stderr",
    "query",
)
_GENERIC_VERIFICATION_MARKERS = (
    "verify",
    "verified",
    "checked",
    "confirmed",
    "validation",
)
_GENERIC_FAILURE_MARKERS = (
    "failed",
    "failure",
    "unable",
    "blocked",
    "retry",
    "fallback",
    "error",
)
_GENERIC_UNSUPPORTED_MARKERS = (
    "additional unsupported claim",
    "definitively proven",
    "definitively correct",
    "requires no further uncertainty handling",
)
_GENERIC_INSTRUCTION_MARKERS = (
    "json",
    "bullet",
    "bullets",
    "table",
    "heading",
    "headings",
    "concise",
    "formal",
    "preserve",
    "keep",
    "maintain",
)

_GENERIC_PROFILE_DIMENSIONS: Dict[str, List[DimensionSpec]] = {
    "documentation_variants": [
        DimensionSpec("source_grounding", "Source Grounding", "The document should stay grounded in the provided source material.", True, [SubdimensionSpec("unsupported_addition", "Unsupported additions")]),
        DimensionSpec("content_coverage", "Content Coverage", "The document should preserve salient content from the source and task.", False, [SubdimensionSpec("salient_content", "Salient content")]),
        DimensionSpec("documentation_structure", "Documentation Structure", "The document should respect the requested scaffold or expected sections.", False, [SubdimensionSpec("section_presence", "Section presence")]),
        DimensionSpec("action_items", "Action Items", "The document should preserve requested plans, next steps, or follow-up actions.", False, [SubdimensionSpec("next_steps", "Next steps")]),
        DimensionSpec("reasoning_support", "Reasoning Support", "When the document explains or justifies content, that support should be preserved.", False, [SubdimensionSpec("supporting_rationale", "Supporting rationale")]),
    ],
    "rewrite_editing": [
        DimensionSpec("instruction_adherence", "Instruction Adherence", "The rewrite should follow the requested editing transformation.", False, [SubdimensionSpec("requested_transform", "Requested transform")]),
        DimensionSpec("meaning_preservation", "Meaning Preservation", "The rewrite should preserve the source meaning unless instructed otherwise.", True, [SubdimensionSpec("content_preservation", "Content preservation")]),
        DimensionSpec("style_transformation", "Style Transformation", "The rewrite should reflect the requested style, tone, or clarity improvements.", False, [SubdimensionSpec("style_change", "Style change")]),
        DimensionSpec("format_compliance", "Format Compliance", "The rewrite should preserve the requested structure or output format.", False, [SubdimensionSpec("output_format", "Output format")]),
        DimensionSpec("anti_fabrication", "Anti-Fabrication", "The rewrite should not add unsupported content.", True, [SubdimensionSpec("unsupported_addition", "Unsupported additions")]),
    ],
    "clinical_decision_support": [
        DimensionSpec("context_grounding", "Context Grounding", "Recommendations should be grounded in the provided evidence.", True, [SubdimensionSpec("unsupported_claim", "Unsupported claim")]),
        DimensionSpec("recommendation_quality", "Recommendation Quality", "The response should provide the requested management recommendation or next step.", False, [SubdimensionSpec("next_step", "Next step")]),
        DimensionSpec("reasoning", "Reasoning", "The response should explain why the recommendation fits the evidence.", False, [SubdimensionSpec("supporting_reasoning", "Supporting reasoning")]),
        DimensionSpec("safety", "Safety", "The response should preserve uncertainty, caveats, or escalation guidance when relevant.", True, [SubdimensionSpec("risk_handling", "Risk handling")]),
        DimensionSpec("follow_up", "Follow-up", "The response should retain monitoring, follow-up, or reevaluation guidance when present.", False, [SubdimensionSpec("follow_up_plan", "Follow-up plan")]),
    ],
    "general_instruction_following": [
        DimensionSpec("instruction_adherence", "Instruction Adherence", "The response should follow the user's explicit instructions.", False, [SubdimensionSpec("constraint_following", "Constraint following")]),
        DimensionSpec("grounding", "Grounding", "The response should stay grounded in the provided context.", True, [SubdimensionSpec("unsupported_addition", "Unsupported additions")]),
        DimensionSpec("completeness", "Completeness", "The response should address the requested content.", False, [SubdimensionSpec("requested_content", "Requested content")]),
        DimensionSpec("format_communication", "Format And Communication", "The response should respect requested format and remain clear.", False, [SubdimensionSpec("output_format", "Output format")]),
        DimensionSpec("safety", "Safety", "The response should preserve relevant caution or uncertainty.", False, [SubdimensionSpec("uncertainty_handling", "Uncertainty handling")]),
    ],
    "agentic_workflows": [
        DimensionSpec("task_completion", "Task Completion", "The workflow output should reflect the requested steps and completion status.", False, [SubdimensionSpec("step_coverage", "Step coverage")]),
        DimensionSpec("tool_result_grounding", "Tool Result Grounding", "Conclusions should be grounded in the observed tool results.", True, [SubdimensionSpec("observed_results", "Observed results")]),
        DimensionSpec("verification", "Verification", "The workflow should preserve validation or double-checking steps when present.", False, [SubdimensionSpec("verification_step", "Verification step")]),
        DimensionSpec("failure_handling", "Failure Handling", "The workflow should retain blockers, retries, or fallback handling.", False, [SubdimensionSpec("error_recovery", "Error recovery")]),
        DimensionSpec("final_response_quality", "Final Response Quality", "The workflow should preserve the final deliverable or conclusion.", False, [SubdimensionSpec("deliverable", "Deliverable")]),
    ],
}

_GENERIC_PROFILE_TEMPLATE_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "documentation_variants": [
        {"template_id": "no_unsupported_additions", "dimension_id": "source_grounding", "subdimension_id": "unsupported_addition", "label": "No Unsupported Additions", "description": "Do not add document content that is unsupported by the provided source.", "severity_tier": "catastrophic", "hard_gate_default": True},
        {"template_id": "required_sections_task_family", "dimension_id": "documentation_structure", "subdimension_id": "section_presence", "label": "Required Task-Family Sections", "description": "Include the expected sections for this documentation family when applicable.", "severity_tier": "essential", "hard_gate_default": False},
        {"template_id": "salient_content_coverage", "dimension_id": "content_coverage", "subdimension_id": "salient_content", "label": "Salient Content Coverage", "description": "Preserve the main content conveyed by the stronger artifact.", "severity_tier": "important", "hard_gate_default": False},
        {"template_id": "action_items_preserved", "dimension_id": "action_items", "subdimension_id": "next_steps", "label": "Action Items Preserved", "description": "Retain plans, next steps, or follow-up actions that appear in the stronger artifact.", "severity_tier": "important", "hard_gate_default": False},
        {"template_id": "reasoning_support_preserved", "dimension_id": "reasoning_support", "subdimension_id": "supporting_rationale", "label": "Reasoning Support Preserved", "description": "Retain explicit rationale or supporting evidence when it appears in the stronger artifact.", "severity_tier": "important", "hard_gate_default": False},
    ],
    "rewrite_editing": [
        {"template_id": "no_unsupported_additions", "dimension_id": "anti_fabrication", "subdimension_id": "unsupported_addition", "label": "No Unsupported Additions", "description": "Do not introduce new unsupported content during rewriting.", "severity_tier": "catastrophic", "hard_gate_default": True},
        {"template_id": "preserve_core_meaning", "dimension_id": "meaning_preservation", "subdimension_id": "content_preservation", "label": "Preserve Core Meaning", "description": "Maintain the original meaning unless the instruction explicitly requests a semantic change.", "severity_tier": "essential", "hard_gate_default": False},
        {"template_id": "requested_transformation_applied", "dimension_id": "instruction_adherence", "subdimension_id": "requested_transform", "label": "Requested Transformation Applied", "description": "Apply the rewrite, editing, or tone-change instruction.", "severity_tier": "essential", "hard_gate_default": False},
        {"template_id": "requested_format_preserved", "dimension_id": "format_compliance", "subdimension_id": "output_format", "label": "Requested Format Preserved", "description": "Respect requested headings, bullets, or formatting constraints.", "severity_tier": "important", "hard_gate_default": False},
    ],
    "clinical_decision_support": [
        {"template_id": "no_unsupported_additions", "dimension_id": "context_grounding", "subdimension_id": "unsupported_claim", "label": "No Unsupported Clinical Claims", "description": "Do not add unsupported conclusions beyond the provided evidence.", "severity_tier": "catastrophic", "hard_gate_default": True},
        {"template_id": "recommendation_present", "dimension_id": "recommendation_quality", "subdimension_id": "next_step", "label": "Recommendation Present", "description": "State the requested recommendation or next step when the task expects one.", "severity_tier": "essential", "hard_gate_default": False},
        {"template_id": "reasoning_present", "dimension_id": "reasoning", "subdimension_id": "supporting_reasoning", "label": "Reasoning Present", "description": "Explain why the recommendation follows from the evidence when the stronger artifact does so.", "severity_tier": "important", "hard_gate_default": False},
        {"template_id": "followup_present", "dimension_id": "follow_up", "subdimension_id": "follow_up_plan", "label": "Follow-up Guidance Present", "description": "Preserve follow-up or reevaluation guidance when present in the stronger artifact.", "severity_tier": "important", "hard_gate_default": False},
        {"template_id": "safety_caveats_present", "dimension_id": "safety", "subdimension_id": "risk_handling", "label": "Safety Caveats Preserved", "description": "Preserve cautions, uncertainty, or escalation guidance when present.", "severity_tier": "essential", "hard_gate_default": False},
    ],
    "general_instruction_following": [
        {"template_id": "no_unsupported_additions", "dimension_id": "grounding", "subdimension_id": "unsupported_addition", "label": "No Unsupported Additions", "description": "Do not add unsupported content beyond the given context.", "severity_tier": "catastrophic", "hard_gate_default": True},
        {"template_id": "instruction_followed", "dimension_id": "instruction_adherence", "subdimension_id": "constraint_following", "label": "Instruction Followed", "description": "Follow the explicit task instruction and constraints.", "severity_tier": "essential", "hard_gate_default": False},
        {"template_id": "requested_content_present", "dimension_id": "completeness", "subdimension_id": "requested_content", "label": "Requested Content Present", "description": "Address the requested content completely.", "severity_tier": "important", "hard_gate_default": False},
        {"template_id": "requested_format_present", "dimension_id": "format_communication", "subdimension_id": "output_format", "label": "Requested Format Present", "description": "Respect requested formatting and communication structure.", "severity_tier": "important", "hard_gate_default": False},
    ],
    "agentic_workflows": [
        {"template_id": "no_unsupported_additions", "dimension_id": "tool_result_grounding", "subdimension_id": "observed_results", "label": "No Unsupported Conclusions", "description": "Do not claim workflow outcomes unsupported by the observed steps or tool results.", "severity_tier": "catastrophic", "hard_gate_default": True},
        {"template_id": "workflow_steps_present", "dimension_id": "task_completion", "subdimension_id": "step_coverage", "label": "Workflow Steps Present", "description": "Preserve the main workflow steps or completion markers from the stronger artifact.", "severity_tier": "essential", "hard_gate_default": False},
        {"template_id": "tool_results_present", "dimension_id": "tool_result_grounding", "subdimension_id": "observed_results", "label": "Tool Results Present", "description": "Retain observed tool outputs or evidence that support the conclusion.", "severity_tier": "essential", "hard_gate_default": False},
        {"template_id": "verification_present", "dimension_id": "verification", "subdimension_id": "verification_step", "label": "Verification Present", "description": "Retain validation or double-checking steps when present.", "severity_tier": "important", "hard_gate_default": False},
        {"template_id": "failure_handling_present", "dimension_id": "failure_handling", "subdimension_id": "error_recovery", "label": "Failure Handling Present", "description": "Retain retries, blockers, or fallback handling when present.", "severity_tier": "important", "hard_gate_default": False},
        {"template_id": "final_deliverable_present", "dimension_id": "final_response_quality", "subdimension_id": "deliverable", "label": "Final Deliverable Present", "description": "Retain the final answer, deliverable, or completion summary.", "severity_tier": "important", "hard_gate_default": False},
    ],
}

_GENERIC_PROFILE_FAMILY_SPECS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "documentation_variants": {
        "generic_document": {"label": "Generic document", "required_sections": [], "optional_sections": [], "style_rules": ["Keep the artifact grounded in the source context."], "section_patterns": []},
        "progress_note": {"label": "Progress note", "required_sections": ["summary", "assessment", "plan"], "optional_sections": ["follow_up"], "style_rules": ["Use concise documentation-style structure."], "section_patterns": [{"section_id": "summary", "label": "Summary", "patterns": [r"(?mi)^summary\b", r"(?mi)^interval history\b"]}, {"section_id": "assessment", "label": "Assessment", "patterns": [r"(?mi)^assessment\b", r"(?mi)^impression\b"]}, {"section_id": "plan", "label": "Plan", "patterns": [r"(?mi)^plan\b", r"(?mi)^recommendations?\b"]}]},
        "discharge_summary": {"label": "Discharge summary", "required_sections": ["hospital_course", "disposition", "follow_up"], "optional_sections": ["medications"], "style_rules": ["Make disposition and follow-up explicit."], "section_patterns": [{"section_id": "hospital_course", "label": "Hospital course", "patterns": [r"(?mi)^hospital course\b", r"(?mi)^course\b"]}, {"section_id": "disposition", "label": "Disposition", "patterns": [r"(?mi)^disposition\b", r"(?mi)^discharge\b"]}, {"section_id": "follow_up", "label": "Follow-up", "patterns": [r"(?mi)^follow[\s-]*up\b", r"(?mi)^return precautions?\b"]}]},
        "preop_note": {"label": "Pre-op note", "required_sections": ["procedure", "risk_assessment", "plan"], "optional_sections": ["instructions"], "style_rules": ["Make planned procedure and risk framing explicit."], "section_patterns": [{"section_id": "procedure", "label": "Procedure", "patterns": [r"(?mi)^procedure\b", r"(?mi)^planned procedure\b"]}, {"section_id": "risk_assessment", "label": "Risk assessment", "patterns": [r"(?mi)^risk\b", r"(?mi)^assessment\b"]}, {"section_id": "plan", "label": "Plan", "patterns": [r"(?mi)^plan\b"]}]},
        "patient_message": {"label": "Patient message", "required_sections": [], "optional_sections": [], "style_rules": ["Keep the message concise and grounded."], "section_patterns": []},
        "alert_message": {"label": "Alert or escalation message", "required_sections": [], "optional_sections": [], "style_rules": ["Highlight the alert condition and next step clearly."], "section_patterns": []},
        "clinical_summary": {"label": "Clinical summary", "required_sections": [], "optional_sections": [], "style_rules": ["Summarize the key points without adding unsupported detail."], "section_patterns": []},
    },
    "rewrite_editing": {
        "rewrite_transform": {"label": "Rewrite transform", "required_sections": [], "optional_sections": [], "style_rules": ["Honor the requested rewrite while preserving meaning."], "section_patterns": []},
        "grammar_cleanup": {"label": "Grammar cleanup", "required_sections": [], "optional_sections": [], "style_rules": ["Improve grammar and readability without changing meaning."], "section_patterns": []},
        "structured_transform": {"label": "Structured transform", "required_sections": [], "optional_sections": [], "style_rules": ["Respect requested structure or output formatting."], "section_patterns": []},
        "style_transfer": {"label": "Style transfer", "required_sections": [], "optional_sections": [], "style_rules": ["Apply the requested tone or style change."], "section_patterns": []},
    },
    "clinical_decision_support": {
        "recommendation_plan": {"label": "Recommendation plan", "required_sections": [], "optional_sections": [], "style_rules": ["Keep recommendations grounded and actionable."], "section_patterns": []},
        "differential_diagnosis": {"label": "Differential diagnosis", "required_sections": [], "optional_sections": [], "style_rules": ["Ground the differential in the provided evidence."], "section_patterns": []},
        "next_step_triage": {"label": "Next-step triage", "required_sections": [], "optional_sections": [], "style_rules": ["Emphasize safety and escalation triggers."], "section_patterns": []},
        "evidence_based_answer": {"label": "Evidence-based answer", "required_sections": [], "optional_sections": [], "style_rules": ["Tie the answer to the presented evidence."], "section_patterns": []},
    },
    "general_instruction_following": {
        "context_bound_response": {"label": "Context-bound response", "required_sections": [], "optional_sections": [], "style_rules": ["Stay grounded in the provided context."], "section_patterns": []},
        "format_constrained_response": {"label": "Format-constrained response", "required_sections": [], "optional_sections": [], "style_rules": ["Respect the requested output format."], "section_patterns": []},
        "open_instruction_response": {"label": "Open instruction response", "required_sections": [], "optional_sections": [], "style_rules": ["Follow the requested instruction clearly and directly."], "section_patterns": []},
    },
    "agentic_workflows": {
        "tool_grounded_run": {"label": "Tool-grounded run", "required_sections": [], "optional_sections": [], "style_rules": ["Ground claims in observed tool results."], "section_patterns": []},
        "multi_step_execution": {"label": "Multi-step execution", "required_sections": [], "optional_sections": [], "style_rules": ["Retain step ordering and completion state."], "section_patterns": []},
        "failure_recovery": {"label": "Failure recovery", "required_sections": [], "optional_sections": [], "style_rules": ["Make retries, blockers, or fallbacks explicit."], "section_patterns": []},
        "workflow_summary": {"label": "Workflow summary", "required_sections": [], "optional_sections": [], "style_rules": ["Summarize final status and deliverable clearly."], "section_patterns": []},
    },
}


def _build_generic_ontology(task_profile_id: str) -> RubricOntology:
    profile = get_task_profile(task_profile_id)
    archetype_id = task_profile_archetype_id(task_profile_id)
    dims = list(_GENERIC_PROFILE_DIMENSIONS.get(archetype_id, []))
    template_specs = _GENERIC_PROFILE_TEMPLATE_SPECS.get(archetype_id, [])
    templates = [
        CriterionTemplate(
            template_id=spec["template_id"],
            dimension_id=spec["dimension_id"],
            subdimension_id=spec.get("subdimension_id"),
            label=spec["label"],
            description=spec["description"],
            severity_tier=spec["severity_tier"],
            default_verdict_type="binary",
            evidence_policy="task_context_or_reference",
            hard_gate_default=bool(spec.get("hard_gate_default")),
            typical_failure_codes=["unsupported_inference"] if spec.get("hard_gate_default") else ["omission"],
            task_profile_id=profile.task_profile_id,
            artifact_label=profile.artifact_label,
        )
        for spec in template_specs
    ]
    return RubricOntology(
        ontology_id=f"{profile.task_profile_id}_core",
        version=_GENERIC_PROFILE_VERSION,
        severity_tiers=["catastrophic", "essential", "important", "optional"],
        dimensions=dims,
        criterion_templates=templates,
        error_taxonomy=[
            ErrorTaxonomyEntry("unsupported_inference", "Unsupported Inference", "The artifact adds unsupported content relative to the task context.", "catastrophic"),
            ErrorTaxonomyEntry("omission", "Omission", "The artifact omits salient requested content.", "important"),
            ErrorTaxonomyEntry("section_missing", "Section Missing", "A required task-family section is missing or unclear.", "essential"),
            ErrorTaxonomyEntry("format_violation", "Format Violation", "The artifact misses a requested output format or scaffold.", "important"),
        ],
    )


def build_task_ontology(task_profile_id: Optional[str] = None) -> RubricOntology:
    profile = get_task_profile(task_profile_id or "note_documentation")
    archetype_id = task_profile_archetype_id(profile.task_profile_id)
    if archetype_id == "note_documentation":
        return build_starter_ontology()
    return _build_generic_ontology(profile.task_profile_id)


def _template_applies_to_task_family(template: CriterionTemplate, task_family_id: str) -> bool:
    scope = template.task_family_scope or template.note_family_scope
    if not scope:
        return True
    return task_family_id in scope


def _family_spec_for_profile(task_profile_id: str, task_family_id: str) -> Dict[str, Any]:
    profile = get_task_profile(task_profile_id)
    archetype_id = task_profile_archetype_id(task_profile_id)
    families = _GENERIC_PROFILE_FAMILY_SPECS.get(archetype_id, {})
    archetype_profile = get_task_profile(archetype_id)
    return families.get(
        task_family_id,
        families.get(
            profile.default_task_family_id,
            families.get(archetype_profile.default_task_family_id, {}),
        ),
    )


def infer_task_family(example: ExampleRecord, task_profile_id: Optional[str] = None) -> str:
    profile = resolve_task_profile(example, explicit=task_profile_id)
    archetype_id = task_profile_archetype_id(profile.task_profile_id)
    if archetype_id == "note_documentation":
        return infer_note_family(example)
    if example.task_family_id:
        return example.task_family_id

    blob = " ".join(part for part in (example.task_prompt, example.conversation, strongest_anchor_text(example)) if part).lower()
    if archetype_id == "documentation_variants":
        if "discharge" in blob:
            return "discharge_summary"
        if "progress note" in blob:
            return "progress_note"
        if "pre-op" in blob or "preop" in blob:
            return "preop_note"
        if "alert" in blob or "escalation" in blob:
            return "alert_message"
        if "message" in blob or "portal" in blob:
            return "patient_message"
        if "summary" in blob:
            return "clinical_summary"
        return "generic_document"
    if archetype_id == "rewrite_editing":
        if "grammar" in blob or "proofread" in blob:
            return "grammar_cleanup"
        if any(token in blob for token in ("json", "bullet", "bullets", "table", "format")):
            return "structured_transform"
        if any(token in blob for token in ("tone", "style", "formal", "casual")):
            return "style_transfer"
        return "rewrite_transform"
    if archetype_id == "clinical_decision_support":
        if "differential" in blob:
            return "differential_diagnosis"
        if "triage" in blob or "urgent" in blob:
            return "next_step_triage"
        if "recommend" in blob or "plan" in blob or "management" in blob:
            return "recommendation_plan"
        return "evidence_based_answer"
    if archetype_id == "agentic_workflows":
        if "failure" in blob or "retry" in blob or "blocked" in blob:
            return "failure_recovery"
        if "step" in blob or "workflow" in blob:
            return "multi_step_execution"
        if "tool" in blob or "result" in blob or "observation" in blob:
            return "tool_grounded_run"
        return "workflow_summary"
    if any(token in blob for token in ("json", "bullet", "bullets", "table", "format")):
        return "format_constrained_response"
    if example.conversation.strip():
        return "context_bound_response"
    return "open_instruction_response"


def build_task_family_spec(
    task_family_id: str,
    ontology: RubricOntology,
    *,
    task_profile_id: Optional[str] = None,
) -> NoteFamilySpec:
    profile = get_task_profile(task_profile_id or "note_documentation")
    archetype_id = task_profile_archetype_id(profile.task_profile_id)
    if archetype_id == "note_documentation":
        return build_note_family_spec(task_family_id, ontology)

    raw = _family_spec_for_profile(profile.task_profile_id, task_family_id)
    sections = [
        SectionSpec(section_id=entry["section_id"], label=entry["label"], required=True, allowed_content=[])
        for entry in raw.get("section_patterns", [])
    ]
    family_ids = [
        tmpl.template_id
        for tmpl in ontology.criterion_templates
        if _template_applies_to_task_family(tmpl, task_family_id)
    ]
    return NoteFamilySpec(
        note_family_id=task_family_id,
        version=_GENERIC_PROFILE_VERSION,
        label=raw.get("label") or profile.label,
        documentation_contract=DocumentationContract(
            inference_policy="strict_grounded",
            uncertainty_policy="explicit_only",
            required_section_ids=list(raw.get("required_sections", [])),
            optional_section_ids=list(raw.get("optional_sections", [])),
            style_rules=list(raw.get("style_rules", [])),
        ),
        section_specs=sections,
        family_template_ids=family_ids,
        hard_gate_error_codes=["unsupported_inference"],
        task_profile_id=profile.task_profile_id,
        task_family_id=task_family_id,
        artifact_label=profile.artifact_label,
        metadata={"section_patterns": list(raw.get("section_patterns", []))},
    )


def _extract_salient_terms(text: str, *, limit: int = 8) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for token in _GENERIC_WORD_RE.findall((text or "").lower()):
        if len(token) < 4 or token in _GENERIC_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _markers_in_text(text: str, markers: Sequence[str]) -> List[str]:
    lower = (text or "").lower()
    return [marker for marker in markers if marker in lower]


def _make_compiled(
    *,
    example_id: str,
    suffix: str,
    template_id: str,
    dimension_id: str,
    subdimension_id: Optional[str],
    label: str,
    requirement: str,
    severity_tier: str,
    evidence_quote: str,
    eval_kind: str,
    judge_hints: Dict[str, Any],
) -> CompiledCriterion:
    return CompiledCriterion(
        criterion_id=f"{example_id}_{suffix}",
        template_id=template_id,
        dimension_id=dimension_id,
        subdimension_id=subdimension_id,
        label=label,
        requirement=requirement,
        severity_tier=severity_tier,
        verdict_type="binary",
        evidence_anchors=[EvidenceAnchor(source="task_context", quote=evidence_quote)],
        eval_kind=eval_kind,
        judge_hints=judge_hints,
    )


def compile_task_case_rubric(
    example: ExampleRecord,
    ontology: RubricOntology,
    task_family: NoteFamilySpec,
    *,
    task_profile_id: Optional[str] = None,
) -> CaseRubric:
    profile = resolve_task_profile(example, explicit=task_profile_id or task_family.task_profile_id)
    archetype_id = task_profile_archetype_id(profile.task_profile_id)
    if archetype_id == "note_documentation":
        return compile_case_rubric(example, ontology, task_family)

    example_id = example.example_id
    task_context = example_to_prompt(example)
    anchor = strongest_anchor_text(example)
    family_id = task_family.task_family_id or task_family.note_family_id
    hard_gates: List[CompiledCriterion] = []
    soft_checks: List[CompiledCriterion] = []

    hard_gates.append(
        _make_compiled(
            example_id=example_id,
            suffix="no_unsupported_additions",
            template_id="no_unsupported_additions",
            dimension_id=next((d.dimension_id for d in ontology.dimensions if "ground" in d.dimension_id or "fabrication" in d.dimension_id or "tool_result_grounding" == d.dimension_id), ontology.dimensions[0].dimension_id if ontology.dimensions else "grounding"),
            subdimension_id=None,
            label="No Unsupported Additions",
            requirement=f"Do not add unsupported claims or overconfident assertions to the {profile.artifact_label}.",
            severity_tier="catastrophic",
            evidence_quote=task_context[:320],
            eval_kind="generic_unsupported_assertions",
            judge_hints={"unsupported_markers": list(_GENERIC_UNSUPPORTED_MARKERS)},
        )
    )

    section_patterns = list(task_family.metadata.get("section_patterns", []))
    if section_patterns:
        soft_checks.append(
            _make_compiled(
                example_id=example_id,
                suffix="required_sections",
                template_id="required_sections_task_family",
                dimension_id="documentation_structure" if any(d.dimension_id == "documentation_structure" for d in ontology.dimensions) else "format_compliance",
                subdimension_id="section_presence",
                label=f"Required Sections ({task_family.label})",
                requirement=f"Include the expected scaffold for task family `{family_id}` when applicable.",
                severity_tier="essential",
                evidence_quote=json.dumps([entry.get("section_id") for entry in section_patterns]),
                eval_kind="section_header_coverage",
                judge_hints={"sections": section_patterns, "note_family_id": family_id},
            )
        )

    instruction_terms = _markers_in_text(example.task_prompt, _GENERIC_INSTRUCTION_MARKERS)
    if instruction_terms:
        soft_checks.append(
            _make_compiled(
                example_id=example_id,
                suffix="instruction_following",
                template_id="instruction_followed",
                dimension_id="instruction_adherence",
                subdimension_id=None,
                label="Instruction Following",
                requirement=f"The {profile.artifact_label} should follow explicit output constraints from the task instruction.",
                severity_tier="essential",
                evidence_quote=example.task_prompt[:320],
                eval_kind="artifact_marker_presence",
                judge_hints={"markers": instruction_terms, "min_markers_matched": 1},
            )
        )

    salient_terms = _extract_salient_terms(anchor)
    if salient_terms:
        dim_id = "meaning_preservation" if archetype_id == "rewrite_editing" else "content_coverage"
        if archetype_id == "general_instruction_following":
            dim_id = "completeness"
        if archetype_id == "agentic_workflows":
            dim_id = "task_completion"
        soft_checks.append(
            _make_compiled(
                example_id=example_id,
                suffix="salient_content",
                template_id="salient_content_coverage" if archetype_id == "documentation_variants" else "preserve_core_meaning",
                dimension_id=dim_id,
                subdimension_id=None,
                label="Salient Content Preserved",
                requirement=f"Retain the salient content from the stronger {profile.artifact_label} when it is relevant to the task.",
                severity_tier="important",
                evidence_quote=anchor[:320],
                eval_kind="anchor_terms_presence",
                judge_hints={"terms": salient_terms[:6], "min_terms_matched": 2 if len(salient_terms) >= 4 else 1},
            )
        )

    action_terms = _markers_in_text(anchor, _GENERIC_ACTION_MARKERS)
    if action_terms:
        dim_id = "action_items" if archetype_id == "documentation_variants" else "recommendation_quality"
        if archetype_id == "clinical_decision_support":
            dim_id = "follow_up" if any(term in {"follow-up", "follow up"} for term in action_terms) else "recommendation_quality"
        soft_checks.append(
            _make_compiled(
                example_id=example_id,
                suffix="action_items",
                template_id="action_items_preserved",
                dimension_id=dim_id,
                subdimension_id=None,
                label="Action Items Preserved",
                requirement=f"Retain the key plans, recommendations, or next steps present in the stronger {profile.artifact_label}.",
                severity_tier="important",
                evidence_quote=anchor[:320],
                eval_kind="artifact_marker_presence",
                judge_hints={"markers": action_terms[:6], "min_markers_matched": 1},
            )
        )

    reasoning_terms = _markers_in_text(anchor, _GENERIC_REASONING_MARKERS)
    if reasoning_terms and archetype_id in {"documentation_variants", "clinical_decision_support"}:
        soft_checks.append(
            _make_compiled(
                example_id=example_id,
                suffix="reasoning_support",
                template_id="reasoning_present",
                dimension_id="reasoning_support" if archetype_id == "documentation_variants" else "reasoning",
                subdimension_id=None,
                label="Reasoning Support Preserved",
                requirement=f"Retain explicit reasoning or evidence links when the stronger {profile.artifact_label} includes them.",
                severity_tier="important",
                evidence_quote=anchor[:320],
                eval_kind="artifact_marker_presence",
                judge_hints={"markers": reasoning_terms[:6], "min_markers_matched": 1},
            )
        )

    if archetype_id == "agentic_workflows":
        tool_terms = _markers_in_text(anchor + "\n" + task_context, _GENERIC_TOOL_MARKERS)
        if tool_terms:
            soft_checks.append(
                _make_compiled(
                    example_id=example_id,
                    suffix="tool_results",
                    template_id="tool_results_present",
                    dimension_id="tool_result_grounding",
                    subdimension_id="observed_results",
                    label="Tool Results Preserved",
                    requirement="Preserve observed tool results or evidence that support the workflow conclusion.",
                    severity_tier="essential",
                    evidence_quote=anchor[:320],
                    eval_kind="artifact_marker_presence",
                    judge_hints={"markers": tool_terms[:6], "min_markers_matched": 1},
                )
            )
        verification_terms = _markers_in_text(anchor + "\n" + task_context, _GENERIC_VERIFICATION_MARKERS)
        if verification_terms:
            soft_checks.append(
                _make_compiled(
                    example_id=example_id,
                    suffix="verification",
                    template_id="verification_present",
                    dimension_id="verification",
                    subdimension_id="verification_step",
                    label="Verification Preserved",
                    requirement="Retain validation or double-checking steps when present.",
                    severity_tier="important",
                    evidence_quote=anchor[:320],
                    eval_kind="artifact_marker_presence",
                    judge_hints={"markers": verification_terms[:6], "min_markers_matched": 1},
                )
            )
        failure_terms = _markers_in_text(anchor + "\n" + task_context, _GENERIC_FAILURE_MARKERS)
        if failure_terms:
            soft_checks.append(
                _make_compiled(
                    example_id=example_id,
                    suffix="failure_handling",
                    template_id="failure_handling_present",
                    dimension_id="failure_handling",
                    subdimension_id="error_recovery",
                    label="Failure Handling Preserved",
                    requirement="Retain blockers, retries, or fallback handling when present.",
                    severity_tier="important",
                    evidence_quote=anchor[:320],
                    eval_kind="artifact_marker_presence",
                    judge_hints={"markers": failure_terms[:6], "min_markers_matched": 1},
                )
            )

    for tmpl in ontology.criterion_templates:
        if not tmpl.provisional_discovered:
            continue
        scope = tmpl.task_family_scope or tmpl.note_family_scope
        if scope and family_id not in scope:
            continue
        req_text = _discovered_requirement_text(tmpl)
        if not req_text:
            continue
        soft_checks.append(
            CompiledCriterion(
                criterion_id=f"{example_id}_{tmpl.template_id}",
                template_id=tmpl.template_id,
                dimension_id=tmpl.dimension_id,
                subdimension_id=tmpl.subdimension_id,
                label=tmpl.label,
                requirement=req_text,
                severity_tier=tmpl.severity_tier,
                verdict_type="binary",
                evidence_anchors=[EvidenceAnchor(source="discovery_proposal", quote=req_text[:320])],
                compile_rationale="Provisional discovered template promoted through the generic task-profile compiler.",
                eval_kind="discovered_provisional_soft",
                judge_hints={"provisional_discovered": True, "task_family_scope": scope},
            )
        )

    present_dims = {criterion.dimension_id for criterion in hard_gates + soft_checks}
    weights = {dim: round(1.0 / max(1, len(present_dims)), 4) for dim in sorted(present_dims)}
    aggregation = AggregationPolicy(
        hard_gate_policy="all_must_pass",
        minimum_soft_score=0.7,
        soft_dimension_weights=weights,
    )
    return CaseRubric(
        rubric_id=f"{example_id}_rubric_{profile.task_profile_id}_{_GENERIC_PROFILE_VERSION}",
        version=_GENERIC_PROFILE_VERSION,
        example_id=example_id,
        note_family_id=family_id,
        ontology_version=ontology.version,
        note_family_version=task_family.version,
        hard_gates=hard_gates,
        soft_checks=soft_checks,
        aggregation=aggregation,
        task_profile_id=profile.task_profile_id,
        task_family_id=family_id,
        artifact_label=profile.artifact_label,
        metadata={"task_prompt": example.task_prompt[:500]},
    )
