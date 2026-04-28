"""
Provisional ontology augmentation from merged local discovery outputs (starter scaffold).

Converts `merged_proposals.json` canonical rows into additive `CriterionTemplate` entries with stable ids.
This is not semantic validation or clinical sign-off — a closed-loop experiment hook only.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from rubric_gen.compiled.compiler import (
    build_starter_ontology,
    build_task_ontology,
    infer_note_family,
    infer_task_family,
)
from rubric_gen.compiled.task_profiles import get_task_profile, task_profile_archetype_id
from rubric_gen.compiled.schema import CriterionTemplate, RubricOntology
from rubric_gen.types import ExampleRecord

_WS_RE = re.compile(r"\s+")
_MUTATION_ID_RE = re.compile(r"__mut__([a-z0-9_]+)")
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "hard_gate": 3}
_SECTION_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("CHIEF COMPLAINT", ("chief complaint",)),
    ("MEDICAL HISTORY", ("medical history", "past medical history")),
    ("FAMILY HISTORY", ("family history",)),
    ("REVIEW OF SYSTEMS", ("review of systems", "ros")),
    ("VITALS", ("vitals",)),
    ("PHYSICAL EXAM", ("physical exam",)),
    ("RESULTS", ("results", "imaging", "labs", "studies")),
    ("ASSESSMENT AND PLAN", ("assessment and plan",)),
    ("ASSESSMENT", ("assessment",)),
    ("PLAN", ("plan",)),
    ("INSTRUCTIONS", ("instructions",)),
    ("FOLLOW-UP", ("follow-up", "follow up")),
    ("SUBJECTIVE", ("subjective",)),
    ("OBJECTIVE", ("objective",)),
)
_SYMPTOM_FACET_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "pain characteristics",
        (
            "pain",
            "location",
            "quadrant",
            "severity",
            "duration",
            "trigger",
            "meal",
            "postprandial",
            "nature of the abdominal pain",
        ),
    ),
    (
        "fever or temperature status",
        (
            "fever",
            "fevers",
            "afebrile",
            "temperature",
            "constitutional",
        ),
    ),
    (
        "associated gastrointestinal symptoms",
        (
            "nausea",
            "vomiting",
            "gastrointestinal symptoms",
        ),
    ),
    (
        "key abdominal exam findings",
        (
            "abdominal exam",
            "guarding",
            "murphy",
            "rebound",
            "peritoneal",
            "bowel sounds",
            "exam findings",
        ),
    ),
)
_CERTAINTY_HINTS: Tuple[str, ...] = (
    "certainty",
    "uncertainty",
    "pathognomonic",
    "definitive",
    "absolute",
    "overconfident",
    "balanced language",
    "diagnostic uncertainty",
    "clinical uncertainty",
)
_FOLLOWUP_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "follow-up timing",
        (
            "follow-up",
            "follow up",
            "follow-up timing",
            "follow-up interval",
            "appointment",
            "recheck",
            "return to clinic",
        ),
    ),
    (
        "planned reassessment",
        (
            "surveillance",
            "follow with",
            "reevaluation",
            "reassessment",
        ),
    ),
)
_RETURN_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "return precautions",
        (
            "return precautions",
            "return if",
            "return sooner",
            "call the office if",
            "call if",
            "worsening symptoms",
            "red flags",
            "urgent care",
            "seek care",
            "call 911",
        ),
    ),
)
_MEDICATION_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "medication details",
        (
            "medication",
            "medications",
            "dose",
            "dosage",
            "dosing",
            "frequency",
            "refill",
            "prescription",
            "inhaler",
            "injection",
            "continue",
            "start",
            "stop",
            "increase",
            "decrease",
        ),
    ),
)
_INTERVENTION_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "procedure or intervention plan",
        (
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
        ),
    ),
)
_TESTING_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "planned diagnostic workup",
        (
            "order",
            "ordered",
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
            "cmp",
            "bmp",
            "a1c",
            "echocardiogram",
            "urinalysis",
            "pregnancy test",
            "covid test",
        ),
    ),
)
_TREATMENT_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "medication details",
        (
            "medication",
            "medications",
            "dose",
            "dosing",
            "anticoagulant",
            "antibiotic",
            "supportive care",
            "pain control",
        ),
    ),
    (
        "procedure or surgery plan",
        (
            "procedure",
            "intervention",
            "surgery",
            "surgical",
            "operative",
        ),
    ),
    (
        "dietary management",
        (
            "diet",
            "dietary",
            "high-fat",
            "high fat",
            "fatty foods",
        ),
    ),
)
_REASONING_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "diagnosis-evidence linkage",
        (
            "reasoning",
            "rationale",
            "because",
            "due to",
            "consistent with",
            "supported by",
            "based on",
            "correlation",
            "justify",
        ),
    ),
    (
        "plan-selection rationale",
        (
            "decision-making",
            "excluded because",
            "risk",
            "working diagnosis",
            "impression",
        ),
    ),
)


def map_discovery_severity_to_ontology_tier(raw: str) -> str:
    """Map discovery labels (hard_gate|high|medium|low) onto starter ontology severity tiers."""
    s = _WS_RE.sub(" ", (raw or "").strip().lower())
    if not s:
        return "optional"
    if "hard_gate" in s or s in ("hard", "gate"):
        return "catastrophic"
    if s in ("high",):
        return "essential"
    if s in ("medium",):
        return "important"
    if s in ("low",):
        return "optional"
    if "high" in s:
        return "essential"
    if "medium" in s:
        return "important"
    if "low" in s:
        return "optional"
    return "optional"


def cap_provisional_discovered_tier(tier: str) -> str:
    """
    Provisional discovered templates stay soft checks, so never let a single local discovery
    promote into catastrophic severity. Cap that case at essential until human review exists.
    """
    normalized = _WS_RE.sub(" ", (tier or "").strip().lower())
    if normalized == "catastrophic":
        return "essential"
    return normalized or "optional"


def map_discovery_dimension_to_ontology(dim_raw: str) -> Tuple[str, str]:
    """Best-effort mapping from free-form discovery dimension strings to starter dimension/subdimension ids."""
    d = _WS_RE.sub(" ", (dim_raw or "").replace("_", " ").strip().lower())
    if any(
        x in d
        for x in (
            "hallucinat",
            "faithful",
            "contradict",
            "unsupported",
            "certainty",
            "grounded in dialogue",
            "grounded in evidence",
        )
    ):
        return "dialogue_faithfulness", "hallucination"
    if any(x in d for x in ("structure", "section", "soap", "header", "scaffold")):
        return "documentation_structure", "section_presence"
    if any(
        x in d
        for x in (
            "return precaution",
            "return if",
            "return sooner",
            "seek care",
            "red flag",
            "urgent care",
        )
    ):
        return "management_plan", "return_precautions"
    if any(
        x in d
        for x in (
            "follow up",
            "follow-up",
            "return to clinic",
            "outpatient",
            "recheck",
            "surveillance",
        )
    ):
        return "management_plan", "follow_up_specificity"
    if any(
        x in d
        for x in (
            "medication",
            "medications",
            "dose",
            "dosage",
            "dosing",
            "refill",
            "prescription",
            "inhaler",
            "injection",
        )
    ):
        return "management_plan", "medication_management"
    if any(
        x in d
        for x in (
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
            "endoscopy",
            "egd",
            "ablation",
        )
    ):
        return "management_plan", "intervention_plan"
    if any(
        x in d
        for x in (
            "testing plan",
            "planned testing",
            "diagnostic workup",
            "ordered test",
            "ordered imaging",
            "ordered lab",
            "testing",
            "test order",
            "workup",
        )
    ):
        return "management_plan", "testing_plan"
    if any(
        x in d
        for x in (
            "reason",
            "rationale",
            "impression",
            "correlation",
            "evidence",
            "justify",
            "supported",
            "based on",
        )
    ):
        return "diagnostic_reasoning", "assessment_linkage"
    if any(
        x in d
        for x in (
            "treatment grounding",
            "treatment",
            "management",
            "medication",
            "therapy",
            "procedure",
            "surgery",
            "operative",
            "diet",
        )
    ):
        return "management_plan", "treatment_grounding"
    if any(x in d for x in ("test", "study", "result", "imaging", "lab")):
        return "clinical_completeness", "study_results"
    return "clinical_completeness", "symptom_detail"


def stable_discovered_template_id(merge_key: str) -> str:
    h = hashlib.sha256((merge_key or "").encode("utf-8")).hexdigest()[:16]
    return f"discovered__{h}"


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _unique_strs(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _discovery_severity_rank(raw: str) -> int:
    return _SEVERITY_RANK.get(_normalize_text(raw), 0)


def _cap_broadened_supportive_severity(raw: str) -> str:
    """
    Broad consolidations should stay supportive rather than duplicating stronger base checks.
    """
    sev = _normalize_text(raw)
    if _discovery_severity_rank(sev) > _SEVERITY_RANK["medium"]:
        return "medium"
    return sev or "medium"


def _extract_section_mentions(rows: List[Mapping[str, Any]]) -> List[str]:
    blob = _normalize_text(
        " ".join(
            f"{row.get('label', '')} {row.get('requirement', '')}"
            for row in rows
        )
    )
    mentions: List[str] = []
    for label, cues in _SECTION_HINTS:
        if any(cue in blob for cue in cues):
            mentions.append(label)
    if "ASSESSMENT AND PLAN" in mentions:
        mentions = [m for m in mentions if m not in {"ASSESSMENT", "PLAN"}]
    return mentions


def _format_section_examples(section_mentions: List[str], limit: int = 4) -> str:
    if not section_mentions:
        return ""
    shown = section_mentions[:limit]
    if len(section_mentions) > limit:
        return ", ".join(shown) + ", and related note-family sections"
    return ", ".join(shown)


def _extract_pair_mutation_ids(row: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for pair_id in row.get("pair_ids") or []:
        if not isinstance(pair_id, str):
            continue
        for match in _MUTATION_ID_RE.finditer(pair_id):
            out.append(match.group(1))
    return _unique_strs(out)


def _looks_like_section_scaffold_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    if not any(
        token in text
        for token in (
            "section",
            "sections",
            "subsection",
            "subsections",
            "header",
            "headers",
            "subheading",
            "subheadings",
            "scaffold",
        )
    ):
        return False
    if any(token in text for token in ("header", "headers", "subheading", "subheadings", "scaffold")):
        return True
    if any(
        phrase in text
        for phrase in (
            "distinct section",
            "distinct sections",
            "distinct subsection",
            "distinct subsections",
            "dedicated section",
            "dedicated sections",
            "dedicated subsection",
            "dedicated subsections",
            "separate section",
            "separate sections",
            "separate subsection",
            "separate subsections",
            "clearly delineated section",
            "clearly delineated sections",
            "clearly delineated subsection",
            "clearly delineated subsections",
            "section presence",
            "section headers",
            "clear section",
            "clear sections",
            "assessment and plan subsections",
        )
    ):
        return True
    return False


def _extract_symptom_facets(rows: List[Mapping[str, Any]]) -> List[str]:
    blob = _normalize_text(
        " ".join(
            f"{row.get('label', '')} {row.get('requirement', '')}"
            for row in rows
        )
    )
    facets: List[str] = []
    for label, cues in _SYMPTOM_FACET_HINTS:
        if any(cue in blob for cue in cues):
            facets.append(label)
    return facets


def _format_symptom_facets(facets: List[str], limit: int = 4) -> str:
    if not facets:
        return ""
    shown = facets[:limit]
    if len(facets) > limit:
        return ", ".join(shown) + ", and related encounter-specific detail"
    return ", ".join(shown)


def _extract_named_facets(
    rows: List[Mapping[str, Any]],
    hint_sets: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> List[str]:
    blob = _normalize_text(
        " ".join(
            f"{row.get('label', '')} {row.get('requirement', '')}"
            for row in rows
        )
    )
    facets: List[str] = []
    for label, cues in hint_sets:
        if any(cue in blob for cue in cues):
            facets.append(label)
    return facets


def _format_named_facets(facets: List[str], *, tail: str, limit: int = 4) -> str:
    if not facets:
        return ""
    shown = facets[:limit]
    if len(facets) > limit:
        return ", ".join(shown) + ", " + tail
    return ", ".join(shown)


def _looks_like_symptom_detail_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    if any(
        token in text
        for token in (
            "symptom",
            "symptoms",
            "pain",
            "fever",
            "afebrile",
            "temperature",
            "nausea",
            "vomiting",
            "constitutional",
            "vital signs",
            "vitals",
            "abdominal exam",
            "murphy",
            "guarding",
            "rebound",
            "peritoneal",
            "bowel sounds",
            "review of systems",
            "quadrant",
            "location",
            "severity",
            "duration",
            "trigger",
        )
    ):
        return True
    dim_id, sub_id = map_discovery_dimension_to_ontology(str(row.get("dimension", "")))
    return dim_id == "clinical_completeness" and sub_id == "symptom_detail"


def _looks_like_certainty_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    return any(hint in text for hint in _CERTAINTY_HINTS)


def _looks_like_followup_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    if any(
        token in text
        for token in (
            "return precautions",
            "return if",
            "return sooner",
            "call if",
            "seek care",
            "red flags",
            "urgent care",
        )
    ):
        return False
    if any(
        token in text
        for token in (
            "understands and agrees",
            "agreement",
            "patient portal",
            "daily weight",
            "blood pressure",
            "provider will be in touch",
            "monitoring",
        )
    ):
        return False
    return any(
        token in text
        for token in (
            "follow-up",
            "follow up",
            "return to clinic",
            "outpatient",
            "recheck",
            "appointment",
            "surveillance",
            "reevaluation",
            "reassessment",
            "follow with",
        )
    )


def _looks_like_return_precaution_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    return any(
        token in text
        for token in (
            "return precautions",
            "return if",
            "return sooner",
            "call if",
            "call the office if",
            "seek care",
            "worsening symptoms",
            "red flags",
            "urgent care",
            "call 911",
        )
    )


def _looks_like_medication_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    if any(
        token in text
        for token in (
            "understands and agrees",
            "agreement",
            "monitoring",
            "daily weight",
            "blood pressure",
            "patient portal",
            "endocrinologist",
            "primary care provider",
        )
    ):
        return False
    return any(
        token in text
        for token in (
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
            "inhaler",
            "injection",
        )
    )


def _looks_like_intervention_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    if any(
        token in text
        for token in (
            "questions answered",
            "patient questions",
            "discussed with the patient",
            "discussion",
            "understands and agrees",
            "agreement",
            "activity restriction",
            "activity restrictions",
            "hiking",
            "hospital stay",
            "daily weight",
            "weight monitoring",
            "scale",
            "blood pressure cuff",
            "blood pressure monitor",
            "mri",
            "ct",
            "ultrasound",
            "echocardiogram",
            "spirometry",
            "cbc",
            "covid",
            "dietary",
            "high fat",
            "fatty foods",
        )
    ):
        return False
    return any(
        token in text
        for token in (
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
        )
    )


def _looks_like_testing_plan_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    return any(
        token in text
        for token in (
            "ordered",
            "order",
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
    )


def _looks_like_treatment_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    return any(
        token in text
        for token in (
            "treatment",
            "management plan",
            "medication",
            "medications",
            "therapy",
            "supportive care",
            "pain control",
            "dose",
            "dosing",
            "procedure",
            "surgery",
            "surgical",
            "intervention",
            "diet",
            "dietary",
            "fatty foods",
        )
    )


def _looks_like_reasoning_row(row: Mapping[str, Any]) -> bool:
    text = _normalize_text(
        f"{row.get('dimension', '')} {row.get('label', '')} {row.get('requirement', '')}"
    )
    if any(
        token in text
        for token in (
            "call the clinic if",
            "call if symptoms worsen",
            "worsening symptoms",
            "seek urgent care",
            "return sooner",
            "patient instructions",
        )
    ):
        return False
    return any(
        token in text
        for token in (
            "reasoning",
            "rationale",
            "because",
            "due to",
            "consistent with",
            "supported by",
            "supports",
            "based on",
            "correlation",
            "justify",
            "justified",
            "working diagnosis",
            "impression",
            "risk",
            "decision-making",
        )
    )


def _example_id_to_note_family(examples: List[ExampleRecord]) -> Dict[str, str]:
    return {ex.example_id: infer_note_family(ex) for ex in examples}


def _example_id_to_task_family(
    examples: List[ExampleRecord],
    *,
    task_profile_id: str,
) -> Dict[str, str]:
    return {
        ex.example_id: infer_task_family(ex, task_profile_id=task_profile_id)
        for ex in examples
    }


def compute_note_family_scope_for_proposal(
    example_ids: List[str],
    example_id_to_nf: Mapping[str, str],
) -> Optional[List[str]]:
    """
    If all supporting examples share one note family, scope templates to it.
    If mixed or unknown, return None (instantiate for any note family).
    """
    nfs: List[str] = []
    for eid in example_ids:
        nf = example_id_to_nf.get(eid)
        if nf:
            nfs.append(nf)
    if not nfs:
        return None
    uniq = sorted(set(nfs))
    if len(uniq) == 1:
        return uniq
    return None


def compute_task_family_scope_for_proposal(
    example_ids: List[str],
    example_id_to_family: Mapping[str, str],
) -> Optional[List[str]]:
    return compute_note_family_scope_for_proposal(example_ids, example_id_to_family)


def _structure_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    dim_id, sub_id = map_discovery_dimension_to_ontology(str(row.get("dimension", "")))
    looks_like_structure = dim_id == "documentation_structure" and sub_id == "section_presence"
    if not looks_like_structure and not _looks_like_section_scaffold_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_structure_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    section_mentions = _extract_section_mentions(rows)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    sections_text = _format_section_examples(section_mentions)
    requirement = "The note includes distinct section headers that preserve the expected note-family scaffold."
    if sections_text:
        requirement = (
            "The note includes distinct section headers that preserve the expected note-family scaffold "
            f"(e.g., {sections_text})."
        )
    merge_key = "consolidated::documentation_structure::section_presence"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if section_mentions:
        merge_key += "::sections=" + ",".join(section_mentions)

    return {
        "merge_key": merge_key,
        "dimension": "structure",
        "label": "Expected note-family section scaffold present",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "section_presence_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _symptom_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["strip_symptom_detail_lines"]:
        return None
    if not _looks_like_symptom_detail_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_symptom_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_symptom_facets(rows)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_symptom_facets(facets)
    requirement = (
        "The note preserves salient encounter-specific symptom and relevant exam detail "
        "needed to characterize the presentation."
    )
    if facets_text:
        requirement = (
            "The note preserves salient encounter-specific symptom and relevant exam detail "
            f"needed to characterize the presentation (e.g., {facets_text})."
        )
    merge_key = "consolidated::clinical_completeness::symptom_detail::mutation=strip_symptom_detail_lines"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "symptom_detail",
        "label": "Salient encounter symptom detail preserved",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "symptom_detail_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _certainty_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["inflate_certainty"]:
        return None
    if not _looks_like_certainty_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_certainty_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    requirement = (
        "The note avoids unsupported or overly definitive diagnostic language and reflects "
        "appropriate clinical uncertainty when the evidence remains limited."
    )
    merge_key = "consolidated::certainty_language::inflate_certainty"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)

    return {
        "merge_key": merge_key,
        "dimension": "certainty_language",
        "label": "Appropriate diagnostic certainty language",
        "requirement": requirement,
        "severity_tier": _normalize_text(strongest_raw_sev) or "medium",
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "certainty_language_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _followup_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["drop_followup_lines"]:
        return None
    if not _looks_like_followup_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_followup_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_named_facets(rows, _FOLLOWUP_HINTS)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_named_facets(facets, tail="and related follow-up timing detail")
    requirement = (
        "The note preserves encounter-specific scheduled follow-up timing or planned reassessment when "
        "discussed."
    )
    if facets_text:
        requirement = (
            "The note preserves encounter-specific scheduled follow-up timing or planned reassessment "
            f"when discussed (e.g., {facets_text})."
        )
    merge_key = "consolidated::management_plan::follow_up_specificity::mutation=drop_followup_lines"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "follow_up_specificity",
        "label": "Specific follow-up timing documented",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "follow_up_timing_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _return_precaution_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["drop_return_precaution_lines"]:
        return None
    if not _looks_like_return_precaution_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_return_precaution_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_named_facets(rows, _RETURN_HINTS)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_named_facets(facets, tail="and related escalation guidance")
    requirement = (
        "The note preserves explicit return precautions or escalation guidance when discussed."
    )
    if facets_text:
        requirement = (
            "The note preserves explicit return precautions or escalation guidance when discussed "
            f"(e.g., {facets_text})."
        )
    merge_key = "consolidated::management_plan::return_precautions::mutation=drop_return_precaution_lines"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "return_precautions",
        "label": "Clear return precautions documented",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "return_precautions_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _medication_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["drop_medication_lines"]:
        return None
    if not _looks_like_medication_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_medication_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_named_facets(rows, _MEDICATION_HINTS)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_named_facets(facets, tail="and related medication guidance")
    requirement = (
        "The note preserves medication names, adjustments, continuation/discontinuation, or dosing "
        "guidance when discussed."
    )
    if facets_text:
        requirement = (
            "The note preserves medication names, adjustments, continuation/discontinuation, or dosing "
            f"guidance when discussed (e.g., {facets_text})."
        )
    merge_key = "consolidated::management_plan::medication_management::mutation=drop_medication_lines"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "medication_management",
        "label": "Grounded medication plan documented",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "medication_management_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _intervention_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["drop_procedure_lines"]:
        return None
    if not _looks_like_intervention_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_intervention_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_named_facets(rows, _INTERVENTION_HINTS)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_named_facets(facets, tail="and related intervention detail")
    requirement = (
        "The note preserves procedure, referral, device, or intervention recommendations when discussed."
    )
    if facets_text:
        requirement = (
            "The note preserves procedure, referral, device, or intervention recommendations when discussed "
            f"(e.g., {facets_text})."
        )
    merge_key = "consolidated::management_plan::intervention_plan::mutation=drop_procedure_lines"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "intervention_plan",
        "label": "Grounded intervention plan documented",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "intervention_plan_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _testing_plan_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["drop_testing_plan_lines"]:
        return None
    if not _looks_like_testing_plan_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_testing_plan_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_named_facets(rows, _TESTING_HINTS)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_named_facets(facets, tail="and related planned workup")
    requirement = (
        "The note preserves planned diagnostic tests, repeat studies, or workup orders when discussed."
    )
    if facets_text:
        requirement = (
            "The note preserves planned diagnostic tests, repeat studies, or workup orders when discussed "
            f"(e.g., {facets_text})."
        )
    merge_key = "consolidated::management_plan::testing_plan::mutation=drop_testing_plan_lines"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "testing_plan",
        "label": "Planned diagnostic testing documented",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "testing_plan_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _treatment_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["drop_treatment_lines"]:
        return None
    if not _looks_like_treatment_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_treatment_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_named_facets(rows, _TREATMENT_HINTS)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_named_facets(facets, tail="and related management detail")
    requirement = (
        "The note preserves grounded treatment, medication, procedure, or dietary management "
        "details when discussed."
    )
    if facets_text:
        requirement = (
            "The note preserves grounded treatment, medication, procedure, or dietary management "
            f"details when discussed (e.g., {facets_text})."
        )
    merge_key = "consolidated::management_plan::treatment_grounding::mutation=drop_treatment_lines"
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "treatment_grounding",
        "label": "Grounded treatment plan documented",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "treatment_grounding_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def _reasoning_group_key(
    row: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
) -> Optional[Tuple[str, ...]]:
    mutation_ids = _extract_pair_mutation_ids(row)
    if mutation_ids != ["drop_assessment_reasoning_lines"]:
        return None
    if not _looks_like_reasoning_row(row):
        return None
    ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
    scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)
    return tuple(scope or ["*"])


def _consolidate_reasoning_rows(
    rows: List[Mapping[str, Any]],
    *,
    scope_key: Tuple[str, ...],
) -> Dict[str, Any]:
    facets = _extract_named_facets(rows, _REASONING_HINTS)
    merged_examples = _unique_strs(
        [str(x) for row in rows for x in (row.get("example_ids") or []) if isinstance(x, str)]
    )
    merged_pairs = _unique_strs(
        [str(x) for row in rows for x in (row.get("pair_ids") or []) if isinstance(x, str)]
    )
    strongest_raw_sev = max(
        (str(row.get("severity_tier", "")) for row in rows),
        key=_discovery_severity_rank,
        default="medium",
    )
    consolidated_sev = _cap_broadened_supportive_severity(strongest_raw_sev)
    facets_text = _format_named_facets(facets, tail="and related assessment rationale")
    requirement = (
        "The note links the assessment or selected management plan to the supporting symptoms, "
        "exam findings, or study results discussed in the encounter."
    )
    if facets_text:
        requirement = (
            "The note links the assessment or selected management plan to the supporting symptoms, "
            f"exam findings, or study results discussed in the encounter (e.g., {facets_text})."
        )
    merge_key = (
        "consolidated::diagnostic_reasoning::assessment_linkage::"
        "mutation=drop_assessment_reasoning_lines"
    )
    if scope_key:
        merge_key += "::scope=" + ",".join(scope_key)
    if facets:
        merge_key += "::facets=" + ",".join(facets)

    return {
        "merge_key": merge_key,
        "dimension": "diagnostic_reasoning",
        "label": "Assessment linked to supporting evidence",
        "requirement": requirement,
        "severity_tier": consolidated_sev,
        "count": len(merged_pairs) or len(merged_examples) or max(int(row.get("count") or 0) for row in rows),
        "example_ids": merged_examples,
        "pair_ids": merged_pairs,
        "consolidation_kind": "diagnostic_reasoning_union",
        "consolidated_from_merge_keys": [
            str(row.get("merge_key", "")).strip()
            for row in rows
            if str(row.get("merge_key", "")).strip()
        ],
    }


def consolidate_promotable_rows(
    canonical: List[Mapping[str, Any]],
    *,
    example_id_to_nf: Mapping[str, str],
) -> List[Mapping[str, Any]]:
    """
    Collapse overlapping discovery rows into broader promotable rows when they clearly refer
    to the same scaffold/completeness concept for one note family.
    """
    structure_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    symptom_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    certainty_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    followup_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    return_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    medication_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    intervention_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    testing_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    treatment_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    reasoning_grouped: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for row in canonical:
        structure_key = _structure_group_key(row, example_id_to_nf=example_id_to_nf)
        if structure_key is not None:
            structure_grouped.setdefault(structure_key, []).append(row)
            continue
        symptom_key = _symptom_group_key(row, example_id_to_nf=example_id_to_nf)
        if symptom_key is not None:
            symptom_grouped.setdefault(symptom_key, []).append(row)
            continue
        certainty_key = _certainty_group_key(row, example_id_to_nf=example_id_to_nf)
        if certainty_key is not None:
            certainty_grouped.setdefault(certainty_key, []).append(row)
            continue
        followup_key = _followup_group_key(row, example_id_to_nf=example_id_to_nf)
        if followup_key is not None:
            followup_grouped.setdefault(followup_key, []).append(row)
            continue
        return_key = _return_precaution_group_key(row, example_id_to_nf=example_id_to_nf)
        if return_key is not None:
            return_grouped.setdefault(return_key, []).append(row)
            continue
        medication_key = _medication_group_key(row, example_id_to_nf=example_id_to_nf)
        if medication_key is not None:
            medication_grouped.setdefault(medication_key, []).append(row)
            continue
        intervention_key = _intervention_group_key(row, example_id_to_nf=example_id_to_nf)
        if intervention_key is not None:
            intervention_grouped.setdefault(intervention_key, []).append(row)
            continue
        testing_key = _testing_plan_group_key(row, example_id_to_nf=example_id_to_nf)
        if testing_key is not None:
            testing_grouped.setdefault(testing_key, []).append(row)
            continue
        treatment_key = _treatment_group_key(row, example_id_to_nf=example_id_to_nf)
        if treatment_key is not None:
            treatment_grouped.setdefault(treatment_key, []).append(row)
            continue
        reasoning_key = _reasoning_group_key(row, example_id_to_nf=example_id_to_nf)
        if reasoning_key is not None:
            reasoning_grouped.setdefault(reasoning_key, []).append(row)

    emitted_structure: set[Tuple[str, ...]] = set()
    emitted_symptom: set[Tuple[str, ...]] = set()
    emitted_certainty: set[Tuple[str, ...]] = set()
    emitted_followup: set[Tuple[str, ...]] = set()
    emitted_return: set[Tuple[str, ...]] = set()
    emitted_medication: set[Tuple[str, ...]] = set()
    emitted_intervention: set[Tuple[str, ...]] = set()
    emitted_testing: set[Tuple[str, ...]] = set()
    emitted_treatment: set[Tuple[str, ...]] = set()
    emitted_reasoning: set[Tuple[str, ...]] = set()
    out: List[Mapping[str, Any]] = []
    for row in canonical:
        structure_key = _structure_group_key(row, example_id_to_nf=example_id_to_nf)
        if structure_key is not None:
            if structure_key in emitted_structure:
                continue
            members = structure_grouped.get(structure_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_structure_rows(members, scope_key=structure_key))
            emitted_structure.add(structure_key)
            continue
        symptom_key = _symptom_group_key(row, example_id_to_nf=example_id_to_nf)
        if symptom_key is not None:
            if symptom_key in emitted_symptom:
                continue
            members = symptom_grouped.get(symptom_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_symptom_rows(members, scope_key=symptom_key))
            emitted_symptom.add(symptom_key)
            continue
        certainty_key = _certainty_group_key(row, example_id_to_nf=example_id_to_nf)
        if certainty_key is not None:
            if certainty_key in emitted_certainty:
                continue
            members = certainty_grouped.get(certainty_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_certainty_rows(members, scope_key=certainty_key))
            emitted_certainty.add(certainty_key)
            continue
        followup_key = _followup_group_key(row, example_id_to_nf=example_id_to_nf)
        if followup_key is not None:
            if followup_key in emitted_followup:
                continue
            members = followup_grouped.get(followup_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_followup_rows(members, scope_key=followup_key))
            emitted_followup.add(followup_key)
            continue
        return_key = _return_precaution_group_key(row, example_id_to_nf=example_id_to_nf)
        if return_key is not None:
            if return_key in emitted_return:
                continue
            members = return_grouped.get(return_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_return_precaution_rows(members, scope_key=return_key))
            emitted_return.add(return_key)
            continue
        medication_key = _medication_group_key(row, example_id_to_nf=example_id_to_nf)
        if medication_key is not None:
            if medication_key in emitted_medication:
                continue
            members = medication_grouped.get(medication_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_medication_rows(members, scope_key=medication_key))
            emitted_medication.add(medication_key)
            continue
        intervention_key = _intervention_group_key(row, example_id_to_nf=example_id_to_nf)
        if intervention_key is not None:
            if intervention_key in emitted_intervention:
                continue
            members = intervention_grouped.get(intervention_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_intervention_rows(members, scope_key=intervention_key))
            emitted_intervention.add(intervention_key)
            continue
        testing_key = _testing_plan_group_key(row, example_id_to_nf=example_id_to_nf)
        if testing_key is not None:
            if testing_key in emitted_testing:
                continue
            members = testing_grouped.get(testing_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_testing_plan_rows(members, scope_key=testing_key))
            emitted_testing.add(testing_key)
            continue
        treatment_key = _treatment_group_key(row, example_id_to_nf=example_id_to_nf)
        if treatment_key is not None:
            if treatment_key in emitted_treatment:
                continue
            members = treatment_grouped.get(treatment_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_treatment_rows(members, scope_key=treatment_key))
            emitted_treatment.add(treatment_key)
            continue
        reasoning_key = _reasoning_group_key(row, example_id_to_nf=example_id_to_nf)
        if reasoning_key is not None:
            if reasoning_key in emitted_reasoning:
                continue
            members = reasoning_grouped.get(reasoning_key, [])
            if len(members) <= 1:
                out.append(row)
            else:
                out.append(_consolidate_reasoning_rows(members, scope_key=reasoning_key))
            emitted_reasoning.add(reasoning_key)
            continue
        out.append(row)
    return out


def criterion_templates_from_merged_proposals(
    merged: Mapping[str, Any],
    *,
    example_id_to_nf: Mapping[str, str],
    support_threshold: int,
) -> Tuple[List[CriterionTemplate], List[Dict[str, Any]]]:
    """
    Select canonical proposals at or above support_threshold and build provisional templates.

    Returns (templates, selected_rows) where selected_rows are JSON-serializable records for artifacts.
    """
    canonical = merged.get("canonical_proposals") or []
    if not isinstance(canonical, list):
        canonical = []
    canonical = list(consolidate_promotable_rows(canonical, example_id_to_nf=example_id_to_nf))

    templates: List[CriterionTemplate] = []
    selected_rows: List[Dict[str, Any]] = []

    for row in canonical:
        if not isinstance(row, dict):
            continue
        count = int(row.get("count") or 0)
        if count < support_threshold:
            continue
        req = str(row.get("requirement", "")).strip()
        if not req:
            continue
        merge_key = str(row.get("merge_key", "")).strip()
        if not merge_key:
            continue

        dim_id, sub_id = map_discovery_dimension_to_ontology(str(row.get("dimension", "")))
        sev_disc = str(row.get("severity_tier", ""))
        mapped_tier = map_discovery_severity_to_ontology_tier(sev_disc)
        tier = cap_provisional_discovered_tier(mapped_tier)
        label = str(row.get("label", "")).strip() or "Discovered criterion"
        ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
        scope = compute_note_family_scope_for_proposal(ex_ids, example_id_to_nf)

        # Starter policy: discovered checks compile as soft checks; severity tier still informs LLM weighting.
        hard_gate_default = False

        tmpl_id = stable_discovered_template_id(merge_key)
        desc = (
            "[Provisional discovered — not human-reviewed] "
            f"(support={count}) {req}"
        )

        tmpl = CriterionTemplate(
            template_id=tmpl_id,
            dimension_id=dim_id,
            subdimension_id=sub_id,
            label=f"{label} (provisional discovered)",
            description=desc,
            severity_tier=tier,
            default_verdict_type="binary",
            evidence_policy="dialogue_or_reference",
            hard_gate_default=hard_gate_default,
            typical_failure_codes=["omission"],
            provisional_discovered=True,
            note_family_scope=scope,
        )
        templates.append(tmpl)

        selected_rows.append(
            {
                "template_id": tmpl_id,
                "merge_key": merge_key,
                "support_count": count,
                "dimension": row.get("dimension", ""),
                "label": label,
                "requirement": req,
                "severity_tier_discovery": sev_disc,
                "severity_tier_ontology_raw": mapped_tier,
                "severity_tier_ontology": tier,
                "severity_tier_capped": tier != mapped_tier,
                "note_family_scope": scope,
                "example_ids": ex_ids,
                "consolidation_kind": row.get("consolidation_kind"),
                "consolidated_from_merge_keys": row.get("consolidated_from_merge_keys") or [],
            }
        )

    return templates, selected_rows


def map_discovery_dimension_to_task_ontology(
    dim_raw: str,
    *,
    task_profile_id: str,
) -> Tuple[str, str]:
    archetype_id = task_profile_archetype_id(task_profile_id)
    if archetype_id == "note_documentation":
        return map_discovery_dimension_to_ontology(dim_raw)

    d = _WS_RE.sub(" ", (dim_raw or "").replace("_", " ").strip().lower())
    if archetype_id == "documentation_variants":
        if any(token in d for token in ("structure", "section", "header", "scaffold", "format")):
            return "documentation_structure", "section_presence"
        if any(token in d for token in ("ground", "unsupported", "hallucinat", "fabricat")):
            return "source_grounding", "unsupported_addition"
        if any(token in d for token in ("follow up", "follow-up", "action", "next step", "recommend")):
            return "action_items", "next_steps"
        if any(token in d for token in ("reason", "rationale", "evidence", "because", "supported")):
            return "reasoning_support", "supporting_rationale"
        return "content_coverage", "salient_content"
    if archetype_id == "rewrite_editing":
        if any(token in d for token in ("format", "json", "bullet", "heading", "structure")):
            return "format_compliance", "output_format"
        if any(token in d for token in ("tone", "style", "grammar", "clarity", "rewrite")):
            return "style_transformation", "style_change"
        if any(token in d for token in ("unsupported", "fabricat", "hallucinat", "new fact")):
            return "anti_fabrication", "unsupported_addition"
        if any(token in d for token in ("instruction", "constraint", "request", "transform")):
            return "instruction_adherence", "requested_transform"
        return "meaning_preservation", "content_preservation"
    if archetype_id == "clinical_decision_support":
        if any(token in d for token in ("unsupported", "ground", "hallucinat", "fabricat")):
            return "context_grounding", "unsupported_claim"
        if any(token in d for token in ("safety", "warning", "caveat", "uncertainty", "risk")):
            return "safety", "risk_handling"
        if any(token in d for token in ("follow up", "follow-up", "monitor", "reevaluat")):
            return "follow_up", "follow_up_plan"
        if any(token in d for token in ("reason", "rationale", "evidence", "because", "supported")):
            return "reasoning", "supporting_reasoning"
        return "recommendation_quality", "next_step"
    if archetype_id == "agentic_workflows":
        if any(token in d for token in ("tool", "result", "observation", "ground")):
            return "tool_result_grounding", "observed_results"
        if any(token in d for token in ("verify", "validation", "check", "confirmed")):
            return "verification", "verification_step"
        if any(token in d for token in ("failure", "retry", "fallback", "error", "blocked")):
            return "failure_handling", "error_recovery"
        if any(token in d for token in ("final", "deliverable", "summary", "answer")):
            return "final_response_quality", "deliverable"
        return "task_completion", "step_coverage"
    if any(token in d for token in ("format", "json", "bullet", "heading", "table")):
        return "format_communication", "output_format"
    if any(token in d for token in ("unsupported", "ground", "fabricat", "hallucinat")):
        return "grounding", "unsupported_addition"
    if any(token in d for token in ("instruction", "constraint", "request", "follow")):
        return "instruction_adherence", "constraint_following"
    if any(token in d for token in ("safety", "uncertainty", "risk", "caution")):
        return "safety", "uncertainty_handling"
    return "completeness", "requested_content"


def criterion_templates_from_merged_proposals_generic(
    merged: Mapping[str, Any],
    *,
    example_id_to_family: Mapping[str, str],
    support_threshold: int,
    task_profile_id: str,
) -> Tuple[List[CriterionTemplate], List[Dict[str, Any]]]:
    canonical = merged.get("canonical_proposals") or []
    if not isinstance(canonical, list):
        canonical = []

    templates: List[CriterionTemplate] = []
    selected_rows: List[Dict[str, Any]] = []
    for row in canonical:
        if not isinstance(row, dict):
            continue
        count = int(row.get("count") or 0)
        if count < support_threshold:
            continue
        req = str(row.get("requirement", "")).strip()
        merge_key = str(row.get("merge_key", "")).strip()
        if not req or not merge_key:
            continue

        dim_id, sub_id = map_discovery_dimension_to_task_ontology(
            str(row.get("dimension", "")),
            task_profile_id=task_profile_id,
        )
        sev_disc = str(row.get("severity_tier", ""))
        mapped_tier = map_discovery_severity_to_ontology_tier(sev_disc)
        tier = cap_provisional_discovered_tier(mapped_tier)
        label = str(row.get("label", "")).strip() or "Discovered criterion"
        ex_ids = [str(x) for x in (row.get("example_ids") or []) if isinstance(x, str)]
        scope = compute_task_family_scope_for_proposal(ex_ids, example_id_to_family)
        tmpl_id = stable_discovered_template_id(merge_key)
        tmpl = CriterionTemplate(
            template_id=tmpl_id,
            dimension_id=dim_id,
            subdimension_id=sub_id,
            label=f"{label} (provisional discovered)",
            description=f"[Provisional discovered — not human-reviewed] (support={count}) {req}",
            severity_tier=tier,
            default_verdict_type="binary",
            evidence_policy="task_context_or_reference",
            hard_gate_default=False,
            typical_failure_codes=["omission"],
            provisional_discovered=True,
            task_family_scope=scope,
            task_profile_id=task_profile_id,
            artifact_label=get_task_profile(task_profile_id).artifact_label,
        )
        templates.append(tmpl)
        selected_rows.append(
            {
                "template_id": tmpl_id,
                "merge_key": merge_key,
                "support_count": count,
                "dimension": row.get("dimension", ""),
                "label": label,
                "requirement": req,
                "severity_tier_discovery": sev_disc,
                "severity_tier_ontology_raw": mapped_tier,
                "severity_tier_ontology": tier,
                "severity_tier_capped": tier != mapped_tier,
                "task_family_scope": scope,
                "example_ids": ex_ids,
                "consolidation_kind": row.get("consolidation_kind"),
                "consolidated_from_merge_keys": row.get("consolidated_from_merge_keys") or [],
            }
        )
    return templates, selected_rows


def build_augmented_ontology(
    *,
    merged_proposals: Mapping[str, Any],
    design_examples: List[ExampleRecord],
    support_threshold: int,
    task_profile_id: str = "note_documentation",
) -> Tuple[RubricOntology, List[Dict[str, Any]]]:
    """
    Starter ontology = base starter + provisional discovered templates (additive).
    Bumps ontology version string to reflect augmentation.
    """
    archetype_id = task_profile_archetype_id(task_profile_id)
    if archetype_id == "note_documentation":
        base = build_starter_ontology()
        id_to_nf = _example_id_to_note_family(design_examples)
        discovered, selected = criterion_templates_from_merged_proposals(
            merged_proposals,
            example_id_to_nf=id_to_nf,
            support_threshold=support_threshold,
        )
    else:
        base = build_task_ontology(task_profile_id)
        id_to_family = _example_id_to_task_family(
            design_examples,
            task_profile_id=task_profile_id,
        )
        discovered, selected = criterion_templates_from_merged_proposals_generic(
            merged_proposals,
            example_id_to_family=id_to_family,
            support_threshold=support_threshold,
            task_profile_id=task_profile_id,
        )

    aug_version = f"{base.version}_aug_disc_{len(discovered)}"
    return (
        RubricOntology(
            ontology_id=base.ontology_id,
            version=aug_version,
            severity_tiers=list(base.severity_tiers),
            dimensions=list(base.dimensions),
            criterion_templates=list(base.criterion_templates) + discovered,
            error_taxonomy=list(base.error_taxonomy),
        ),
        selected,
    )
