from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class CompressionRule:
    category_id: str
    label: str
    canonical_text: str
    family: str
    priority: int
    keywords: Sequence[str]
    anti_keywords: Sequence[str] = ()
    min_hits: int = 1


COMPRESSION_RULES: List[CompressionRule] = [
    CompressionRule(
        category_id="unsupported_facts",
        label="Unsupported Facts",
        canonical_text=(
            "The note does not introduce unsupported diagnoses, medications, test results, "
            "follow-up instructions, or exam findings."
        ),
        family="safety_and_grounding",
        priority=10,
        keywords=("unsupported", "does not introduce", "avoid"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="visit_reason",
        label="Visit Reason",
        canonical_text="The note states the visit reason or chief complaint supported by the transcript.",
        family="visit_framing",
        priority=20,
        keywords=("chief complaint", "visit reason", "presenting", "follow-up after", "status post", "recent anterior stemi"),
        anti_keywords=("plan", "monitor"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="medication_adjustment",
        label="Medication Adjustments",
        canonical_text="The note records the medication plan accurately, including additions, continuations, and dose changes discussed in the transcript.",
        family="hypertension_management",
        priority=30,
        keywords=("lisinopril", "hydrochlorothiazide", "lasix", "aldactone", "aspirin", "brilinta", "lipitor", "toprol", "continue", "add", "diuretic"),
        anti_keywords=("pillbox", "missed doses", "app use"),
        min_hits=2,
    ),
    CompressionRule(
        category_id="hypertension_rationale",
        label="Hypertension Rationale",
        canonical_text=(
            "The note states that blood pressure is uncontrolled and that this is why treatment is being adjusted."
        ),
        family="hypertension_management",
        priority=40,
        keywords=("uncontrolled", "out of control", "requires medication changes", "medication adjustment"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="bp_monitoring",
        label="Blood Pressure Monitoring",
        canonical_text=(
            "The note records the blood pressure monitoring plan, including frequency, duration, "
            "and symptomatic/asymptomatic context when stated."
        ),
        family="hypertension_management",
        priority=50,
        keywords=("monitor blood pressure", "twice daily", "symptomatic", "asymptomatic", "blood pressure readings"),
        anti_keywords=("follow-up", "return visit"),
        min_hits=2,
    ),
    CompressionRule(
        category_id="followup",
        label="Follow-Up Plan",
        canonical_text="The note records the follow-up interval and purpose discussed in the transcript.",
        family="hypertension_management",
        priority=60,
        keywords=("follow-up appointment", "follow-up visit", "return visit", "return in", "2 weeks", "two weeks", "reassessment"),
        anti_keywords=("twice daily", "monitor"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="bp_symptom_cluster",
        label="BP Symptom Cluster",
        canonical_text=(
            "The note captures the blood-pressure-related symptom cluster, including lightheadedness, "
            "headache, and dizziness characterization when discussed."
        ),
        family="symptoms",
        priority=70,
        keywords=("lightheaded", "headache", "fuzzy", "vertigo", "unsteadiness", "dizziness"),
        anti_keywords=("chest heaviness", "respiratory difficulty"),
        min_hits=2,
    ),
    CompressionRule(
        category_id="chest_symptoms",
        label="Chest Symptom Qualifiers",
        canonical_text="The note records the chest symptom qualifiers exactly as discussed, without overstating severity.",
        family="symptoms",
        priority=80,
        keywords=("chest heaviness", "chest pain", "respiratory difficulty", "sustained chest pain"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="physical_exam",
        label="Physical Exam Findings",
        canonical_text="The note records the key physical exam findings discussed in the encounter.",
        family="exam_and_results",
        priority=90,
        keywords=("murmur", "rales", "wheezes", "rhonchi", "air movement", "edema", "physical exam"),
        min_hits=2,
    ),
    CompressionRule(
        category_id="diagnostic_results",
        label="Diagnostic Results",
        canonical_text="The note records the important diagnostic or test results discussed in the encounter.",
        family="exam_and_results",
        priority=95,
        keywords=("ekg", "echocardiogram", "ejection fraction", "ef", "mitral regurgitation", "normal sinus rhythm", "ca-125"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="lab_plan",
        label="Laboratory Plan",
        canonical_text="The note records the ordered laboratory plan completely and accurately.",
        family="exam_and_results",
        priority=100,
        keywords=("cholesterol panel", "cbc", "comprehensive metabolic panel", "urinalysis", "laboratory tests"),
        min_hits=2,
    ),
    CompressionRule(
        category_id="assessment_and_diagnoses",
        label="Assessment and Diagnoses",
        canonical_text="The note records the key diagnoses or assessment statements supported by the transcript.",
        family="assessment_and_plan",
        priority=105,
        keywords=("assessment includes", "diagnoses", "coronary artery disease", "heart failure", "hypertension", "diabetes", "mitral regurgitation"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="hydration_counseling",
        label="Hydration Counseling",
        canonical_text=(
            "The note records hydration counseling, including baseline intake guidance and any "
            "increase-with-heat-or-activity instruction."
        ),
        family="counseling",
        priority=110,
        keywords=("hydration", "water", "glasses", "fluid intake", "warm weather", "activity"),
        anti_keywords=("alcohol", "drinks per week"),
        min_hits=2,
    ),
    CompressionRule(
        category_id="lifestyle_diet_counseling",
        label="Lifestyle and Diet Counseling",
        canonical_text="The note records diet, exercise, or lifestyle counseling that is part of the care plan.",
        family="counseling",
        priority=115,
        keywords=("salt restriction", "nutrition counseling", "cardiac rehabilitation", "pepperoni", "fries", "walking the dog", "physical activity"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="alcohol_counseling",
        label="Alcohol Counseling",
        canonical_text="The note records alcohol-limit counseling when it is part of the plan.",
        family="counseling",
        priority=120,
        keywords=("alcohol", "drinks per week", "beer"),
        anti_keywords=("hydration", "water", "glasses"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="medication_adherence",
        label="Medication Adherence",
        canonical_text=(
            "The note records medication adherence context, including aids used and missed-dose frequency when discussed."
        ),
        family="adherence_and_context",
        priority=130,
        keywords=("pillbox", "app use", "app", "missed doses", "adherence"),
        anti_keywords=("lisinopril", "hydrochlorothiazide"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="osteoarthritis_management",
        label="Osteoarthritis Management",
        canonical_text="The note records osteoarthritis management and activity guidance discussed in the encounter.",
        family="secondary_conditions",
        priority=140,
        keywords=("osteoarthritis", "tylenol", "golf", "activity modification"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="history_and_context",
        label="History and Context",
        canonical_text="The note records the key historical or prior-treatment context needed to understand the current case.",
        family="history_and_context",
        priority=145,
        keywords=("road traffic accident", "surgical history", "cystectomies", "hysterectomy", "reconstruction", "prior gynecologic surgeries"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="procedure_details",
        label="Procedure Details",
        canonical_text="The note records the key procedural or intraoperative details discussed in the transcript.",
        family="procedure_and_intervention",
        priority=146,
        keywords=("dental implants", "implant", "harmonic scalpel", "dissection", "intraoperative", "uterine vessels", "ureter", "basal tissue bed", "adherent"),
        min_hits=2,
    ),
    CompressionRule(
        category_id="imaging_and_diagnostics",
        label="Imaging and Diagnostics",
        canonical_text="The note records the important imaging, measurement, or diagnostic findings discussed in the case.",
        family="procedure_and_intervention",
        priority=147,
        keywords=("ultrasound", "ct pelvis", "cbct", "ca-125", "tissue thickness", "multilocular septated mass"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="treatment_rationale",
        label="Treatment Rationale",
        canonical_text="The note states why the selected treatment or device choice was appropriate for this case.",
        family="procedure_and_intervention",
        priority=148,
        keywords=("rationale", "indication", "necessary", "required", "operative laparoscopy", "healing abutments"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="monitoring_and_followup",
        label="Monitoring and Follow-Up",
        canonical_text="The note records close monitoring, follow-up, or patient-reporting instructions when they are part of the plan.",
        family="procedure_and_intervention",
        priority=149,
        keywords=("close monitoring", "monitoring", "follow-up of the healing process", "report any new or worsening symptoms", "follow-up appointment", "postoperative instructions"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="shared_decision_making",
        label="Shared Decision-Making",
        canonical_text="The note records patient counseling, options discussion, or informed consent when discussed.",
        family="procedure_and_intervention",
        priority=150,
        keywords=("informed consent", "counseling process", "shared decision-making", "counselled on options", "surgical options"),
        min_hits=1,
    ),
    CompressionRule(
        category_id="note_structure",
        label="Note Structure",
        canonical_text="The note uses clear clinical structure and section organization.",
        family="documentation_quality",
        priority=160,
        keywords=("section", "organized", "structure", "headers", "soap"),
        min_hits=1,
    ),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_keyword(text: str, keyword: str) -> bool:
    normalized = _normalize(text)
    escaped = re.escape(keyword.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, normalized) is not None


def _score_rule_match(text: str, rule: CompressionRule) -> tuple[int, int]:
    positive_hits = 0
    score = 0
    for keyword in rule.keywords:
        if _contains_keyword(text, keyword):
            positive_hits += 1
            score += 2
    for anti_keyword in rule.anti_keywords:
        if _contains_keyword(text, anti_keyword):
            score -= 2
    return positive_hits, score


def _representative_score(rule: CompressionRule, rubric: Dict[str, object]) -> float:
    text = str(rubric.get("text", ""))
    source_stage = str(rubric.get("source_stage", ""))
    depth = int(rubric.get("depth", 0))
    coverage_count = int(rubric.get("coverage_count", 0) or 0)
    word_count = max(1, len(text.split()))

    positive_hits, raw_score = _score_rule_match(text, rule)
    score = float(raw_score + positive_hits)
    if source_stage == "initial":
        score += 2.5
    if depth == 0:
        score += 1.0
    score += min(coverage_count, 8) * 0.1
    score -= word_count / 200.0
    return score


def compress_rubric_bank(rubrics: Sequence[Dict[str, object]]) -> Dict[str, object]:
    compressed_bank: List[Dict[str, object]] = []
    unmapped: List[Dict[str, object]] = []
    family_counts: Dict[str, int] = {}

    for rule in sorted(COMPRESSION_RULES, key=lambda item: item.priority):
        matched = []
        for rubric in rubrics:
            positive_hits, score = _score_rule_match(str(rubric.get("text", "")), rule)
            if positive_hits >= rule.min_hits and score > 0:
                matched.append(rubric)
        if not matched:
            continue

        representative = max(matched, key=lambda rubric: _representative_score(rule, rubric))
        family_counts[rule.family] = family_counts.get(rule.family, 0) + 1
        compressed_bank.append(
            {
                "category_id": rule.category_id,
                "label": rule.label,
                "family": rule.family,
                "canonical_text": rule.canonical_text,
                "representative_rubric_id": representative.get("rubric_id"),
                "representative_text": representative.get("text"),
                "member_count": len(matched),
                "member_rubric_ids": [rubric.get("rubric_id") for rubric in matched],
                "member_texts": [rubric.get("text") for rubric in matched],
                "source_stages": sorted({str(rubric.get("source_stage", "")) for rubric in matched}),
                "depths": sorted({int(rubric.get("depth", 0) or 0) for rubric in matched}),
                "max_coverage_count": max(int(rubric.get("coverage_count", 0) or 0) for rubric in matched),
            }
        )

    mapped_ids = {
        rubric_id
        for entry in compressed_bank
        for rubric_id in entry["member_rubric_ids"]
        if rubric_id is not None
    }
    for rubric in rubrics:
        if rubric.get("rubric_id") not in mapped_ids:
            unmapped.append(
                {
                    "rubric_id": rubric.get("rubric_id"),
                    "text": rubric.get("text"),
                    "source_stage": rubric.get("source_stage"),
                    "depth": rubric.get("depth"),
                }
            )

    return {
        "compressed_bank": compressed_bank,
        "family_counts": family_counts,
        "unmapped_rubrics": unmapped,
        "raw_rubric_count": len(rubrics),
        "compressed_rubric_count": len(compressed_bank),
    }
