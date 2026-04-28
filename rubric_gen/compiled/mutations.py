"""
Deterministic contrast candidates for compiled-rubric pilot runs.

Builds a small set of CandidateNote rows per example: real note fields plus simple
mutations of the strongest available anchor text. Mutations are intentionally shallow
and aligned with starter rubric families (structure, symptom detail, negatives,
studies, follow-up timing, return precautions, medication planning, intervention
planning, testing plans, diagnostic reasoning, certainty).
"""

from __future__ import annotations

import re
from typing import Callable, List, Sequence, Tuple

from rubric_gen.dataio import strongest_anchor_text
from rubric_gen.types import CandidateNote, ExampleRecord

# Fixed mutation order for reproducibility across runs.
MUTATION_IDS: Tuple[str, ...] = (
    "flatten_structure",
    "strip_symptom_detail_lines",
    "drop_vomiting_negative",
    "drop_study_mentions",
    "drop_followup_lines",
    "drop_return_precaution_lines",
    "drop_medication_lines",
    "drop_procedure_lines",
    "drop_testing_plan_lines",
    "drop_assessment_reasoning_lines",
    "inflate_certainty",
)

_SYMPTOM_DETAIL_PATTERNS = re.compile(
    r"(after\s+(i\s+)?eat|postprandial|meal|meals|eating|"
    r"\b8\s*/\s*10\b|scale|ribs|quadrant|several\s+weeks|several\s+months|"
    r"low-?grade\s+fever|afebrile)",
    re.IGNORECASE,
)

_VOMIT_PATTERNS = re.compile(
    r"(denies\s+vomit|deny\s+vomit|no\s+vomit|has\s+not\s+vomit|n't\s+thrown\s+up)",
    re.IGNORECASE,
)

_STUDY_PATTERNS = re.compile(
    r"\b(ultrasound|sonogram|ct\s+scan|\bct\b|mri|x-?ray|imaging|"
    r"\blabs?\b|lab\s+work|blood\s+work|cbc|bmp|cmp|d-?dimer|troponin|culture)\b",
    re.IGNORECASE,
)

_FOLLOWUP_PATTERNS = re.compile(
    r"\b(follow[\s-]*up|followed\s+up|follow\s+with|recheck|re-?evaluat(?:e|ion)|"
    r"appointment|return\s+to\s+clinic(?!\s+if)|return\s+visit|"
    r"outpatient\s+(?:basis|follow[\s-]*up|surveillance)|"
    r"surveillance|repeat\s+act)\b",
    re.IGNORECASE,
)

_RETURN_PRECAUTION_PATTERNS = re.compile(
    r"\b(return\s+if|return\s+sooner|seek\s+(?:care|medical attention)|"
    r"go\s+to\s+(?:the\s+)?(?:er|ed|emergency(?:\s+room)?(?:\s+department)?)|"
    r"call\s+(?:the\s+)?office\s+if|call\s+if|call\s+911|worsen(?:ing)?|"
    r"red\s+flags?|urgent\s+care)\b",
    re.IGNORECASE,
)

_MEDICATION_PATTERNS = re.compile(
    r"(\b(medication|medications|drug|drugs|prescrib(?:e|ed)|prescription|refill|"
    r"dose|dosage|dosing|frequency|tablet|capsule|inhaler|injection|infusion|topical)\b|"
    r"\b(ibuprofen|acetaminophen|tylenol|phenytoin|colchicine|enoxaparin|heparin|"
    r"warfarin|aspirin|lisinopril|metformin|doxycycline|albuterol|flovent|wixela|"
    r"alvesco|zyrtec|epipen|lasix|sumatriptan|methotrexate|metrocream|naltrexone|"
    r"protonix|excedrin|motrin|hydrochlorothiazide|digoxin|coricidin)\b|"
    r"\b(start(?:ed)?|stop(?:ped)?|continue(?:d)?|resume(?:d)?|increase(?:d)?|"
    r"decrease(?:d)?|adjust(?:ed)?)\b.{0,80}\b("
    r"medication|medications|dose|dosage|dosing|tablet|capsule|inhaler|injection|"
    r"anticoagulant|antibiotic|ibuprofen|acetaminophen|tylenol|phenytoin|colchicine|"
    r"enoxaparin|heparin|warfarin|aspirin|lisinopril|metformin|doxycycline|albuterol|"
    r"flovent|wixela|alvesco|zyrtec|epipen|lasix|sumatriptan|methotrexate|metrocream|"
    r"naltrexone|protonix|excedrin|motrin|hydrochlorothiazide|digoxin|coricidin)\b)",
    re.IGNORECASE,
)

_PROCEDURE_PATTERNS = re.compile(
    r"\b(procedure|procedures|surgery|surgical|operative|operation|intervention|"
    r"referral|referred|refer|physical therapy|occupational therapy|brace|splint|"
    r"crutches|aircast|sling|device|egd|endoscopy|ablation|cholecystectomy|"
    r"laparoscopic)\b",
    re.IGNORECASE,
)

_TESTING_PLAN_PATTERNS = re.compile(
    r"(\b(order(?:ed|ing)?|repeat|obtain|schedule|pending)\b.*\b("
    r"test(?:ing)?|workup|labs?|imaging|ultrasound|mri|ct|x-?ray|cbc|bmp|cmp|"
    r"a1c|hemoglobin\s+a1c|echocardiogram|echo|urinalysis|pregnancy\s+test|"
    r"covid(?:-19)?\s+test)\b|"
    r"\b(test(?:ing)?|workup|labs?|imaging|ultrasound|mri|ct|x-?ray|cbc|bmp|cmp|"
    r"a1c|hemoglobin\s+a1c|echocardiogram|echo|urinalysis|pregnancy\s+test|"
    r"covid(?:-19)?\s+test)\b.*\b(order(?:ed|ing)?|repeat|obtain|schedule|pending)\b)",
    re.IGNORECASE,
)

_TREATMENT_PATTERNS = re.compile(
    r"(\b(received|receive|started|start|continue|continued|stop|stopped|prescrib(?:e|ed)|"
    r"medications?|therapy|therapeutic|treatment|supportive\s+care|anticoagulant|antibiotic|"
    r"analgesic|pain\s+control|ibuprofen|acetaminophen|phenytoin|colchicine|enoxaparin|"
    r"heparin|warfarin|aspirin|procedure|intervention|surgery|surgical|operation|operative|"
    r"laparoscopic|cholecystectomy|trapping|occlusion|dietary|fatty\s+foods?)\b|"
    r"\bavoid\s+(?:high[-\s]*fat|fatty)\b)",
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r"(medical\s+reasoning|\bbecause\b|\bdue\s+to\b|consistent\s+with|suggestive\s+of|suggesting|"
    r"supported\s+by|supports|based\s+on|in\s+light\s+of|which\s+seemed\s+to\s+be\s+the\s+cause|"
    r"\btherefore\b|\bthus\b|\bhence\b|risk\s+of|rationale|correlat(?:e|es|ed|ion)|"
    r"working\s+diagnosis|likely\s+secondary\s+to|favo(?:u)?rs?|explains?|excluded\s+because|"
    r"posed\s+a\s+risk|decided\s+to\s+perform|decision\s+was\s+made)",
    re.IGNORECASE,
)

_REASONING_HEADER_PATTERNS = re.compile(
    r"^\s*(assessment|impression|medical reasoning)\s*[:\-]",
    re.IGNORECASE,
)


def _flatten_structure(text: str) -> str:
    """Remove common section headers / SOAP line prefixes (starter heuristic)."""
    if not text.strip():
        return text
    t = text
    # Drop standalone SOAP header lines and bold markdown section labels.
    t = re.sub(r"(?m)^\s*\*{0,2}\s*(S|O|A|P)\s*:\s*\*{0,2}\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?m)^\s*(S|O|A|P)\s*:\s*", "", t)
    for label in (
        "CHIEF COMPLAINT",
        "REVIEW OF SYSTEMS",
        "ASSESSMENT AND PLAN",
        "MEDICAL HISTORY",
        "PHYSICAL EXAM",
    ):
        t = re.sub(rf"(?m)^\s*{re.escape(label)}\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _drop_matching_lines(text: str, should_drop: Callable[[str], bool]) -> str:
    if not text.strip():
        return text
    kept: List[str] = []
    dropped_any = False
    for line in text.splitlines():
        if should_drop(line):
            dropped_any = True
            continue
        kept.append(line)
    if not dropped_any:
        return text
    out = "\n".join(kept)
    return out if out.strip() else text


def _strip_symptom_detail_lines(text: str) -> str:
    """Remove lines likely carrying symptom-detail checks (meal pattern, severity, duration, etc.)."""
    return _drop_matching_lines(text, lambda line: bool(_SYMPTOM_DETAIL_PATTERNS.search(line)))


def _drop_vomiting_negative(text: str) -> str:
    """Remove lines documenting vomiting denial / pertinent negative."""
    return _drop_matching_lines(text, lambda line: bool(_VOMIT_PATTERNS.search(line)))


def _drop_study_mentions(text: str) -> str:
    """Remove lines mentioning imaging or labs (when present)."""
    return _drop_matching_lines(text, lambda line: bool(_STUDY_PATTERNS.search(line)))


def _drop_followup_lines(text: str) -> str:
    """Remove lines documenting follow-up timing, reevaluation, or outpatient monitoring."""
    return _drop_matching_lines(text, lambda line: bool(_FOLLOWUP_PATTERNS.search(line)))


def _drop_return_precaution_lines(text: str) -> str:
    """Remove lines documenting red flags, return-if guidance, or escalation precautions."""
    return _drop_matching_lines(text, lambda line: bool(_RETURN_PRECAUTION_PATTERNS.search(line)))


def _drop_medication_lines(text: str) -> str:
    """Remove lines documenting medication names, changes, continuation, or dosing."""
    return _drop_matching_lines(text, lambda line: bool(_MEDICATION_PATTERNS.search(line)))


def _drop_procedure_lines(text: str) -> str:
    """Remove lines documenting procedures, referrals, devices, or intervention plans."""
    return _drop_matching_lines(text, lambda line: bool(_PROCEDURE_PATTERNS.search(line)))


def _drop_testing_plan_lines(text: str) -> str:
    """Remove lines documenting ordered, repeated, or pending diagnostic workup."""
    return _drop_matching_lines(text, lambda line: bool(_TESTING_PLAN_PATTERNS.search(line)))


def _drop_treatment_lines(text: str) -> str:
    """Legacy broad treatment mutation kept for compatibility with old artifacts/tests."""
    return _drop_matching_lines(text, lambda line: bool(_TREATMENT_PATTERNS.search(line)))


def _drop_assessment_reasoning_lines(text: str) -> str:
    """Remove lines that explicitly justify the assessment or explain plan selection."""
    return _drop_matching_lines(
        text,
        lambda line: bool(
            _REASONING_HEADER_PATTERNS.search(line) or _REASONING_PATTERNS.search(line)
        ),
    )


def _inflate_certainty(text: str) -> str:
    """Append strong claim language to exercise certainty hard-gate heuristics."""
    suffix = (
        "\n\nAddendum: Findings are pathognomonic for the working diagnosis per clinical assessment."
    )
    return (text.rstrip() + suffix).strip()


_MUTATION_FUNCS: dict[str, Callable[[str], str]] = {
    "flatten_structure": _flatten_structure,
    "strip_symptom_detail_lines": _strip_symptom_detail_lines,
    "drop_vomiting_negative": _drop_vomiting_negative,
    "drop_study_mentions": _drop_study_mentions,
    "drop_followup_lines": _drop_followup_lines,
    "drop_return_precaution_lines": _drop_return_precaution_lines,
    "drop_medication_lines": _drop_medication_lines,
    "drop_procedure_lines": _drop_procedure_lines,
    "drop_testing_plan_lines": _drop_testing_plan_lines,
    "drop_treatment_lines": _drop_treatment_lines,
    "drop_assessment_reasoning_lines": _drop_assessment_reasoning_lines,
    "inflate_certainty": _inflate_certainty,
}


def _original_candidate(
    example: ExampleRecord,
    *,
    field_name: str,
    text: str,
) -> CandidateNote:
    safe = example.example_id.replace("/", "_")
    return CandidateNote(
        candidate_id=f"{safe}__{field_name}",
        example_id=example.example_id,
        text=text,
        source_label=field_name,
        quality_bucket="dataset_note",
        origin_kind="original",
        metadata={"synthetic": False, "note_field": field_name},
        artifact_kind="note",
        task_profile_id="note_documentation",
        task_family_id=example.task_family_id,
    )


def _synthetic_candidate(
    example: ExampleRecord,
    *,
    mutation_id: str,
    text: str,
    anchor_preview: str,
) -> CandidateNote:
    safe = example.example_id.replace("/", "_")
    return CandidateNote(
        candidate_id=f"{safe}__mut__{mutation_id}",
        example_id=example.example_id,
        text=text,
        source_label=f"synthetic_mutation:{mutation_id}",
        quality_bucket="synthetic_contrast",
        origin_kind="synthetic_mutation",
        metadata={
            "synthetic": True,
            "mutation_id": mutation_id,
            "anchor_char_len": len(anchor_preview),
        },
        artifact_kind="note",
        task_profile_id="note_documentation",
        task_family_id=example.task_family_id,
    )


def build_contrast_candidates(example: ExampleRecord, mutation_ids: Sequence[str] | None = None) -> List[CandidateNote]:
    """
    Original notes from the example (non-empty fields) plus deterministic mutations of the anchor note.

    Anchor priority matches ``strongest_anchor_text``: reference, then augmented, then truncated.
    """
    mids = tuple(mutation_ids) if mutation_ids is not None else MUTATION_IDS
    out: List[CandidateNote] = []

    if example.reference_note.strip():
        out.append(_original_candidate(example, field_name="reference_note", text=example.reference_note))
    if example.augmented_note.strip():
        out.append(_original_candidate(example, field_name="augmented_note", text=example.augmented_note))
    if example.note_truncated.strip():
        out.append(_original_candidate(example, field_name="note_truncated", text=example.note_truncated))

    anchor = strongest_anchor_text(example)
    if not anchor.strip():
        return out

    seen_text: set[str] = set()
    for c in out:
        seen_text.add(c.text.strip())

    for mid in mids:
        fn = _MUTATION_FUNCS.get(mid)
        if fn is None:
            continue
        mutated = fn(anchor)
        if not mutated.strip():
            continue
        # Skip uninformative duplicates (mutation matched anchor).
        if mutated.strip() == anchor.strip():
            continue
        if mutated.strip() in seen_text:
            continue
        seen_text.add(mutated.strip())
        out.append(
            _synthetic_candidate(
                example,
                mutation_id=mid,
                text=mutated,
                anchor_preview=anchor[:200],
            )
        )

    return out


def is_synthetic_candidate(c: CandidateNote) -> bool:
    return bool(c.metadata.get("synthetic")) or c.origin_kind == "synthetic_mutation"
