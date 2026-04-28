from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence


PRODUCTION_GROUPS: Dict[str, Dict[str, object]] = {
    "unsupported_facts": {
        "label": "Unsupported Facts",
        "family": "safety_and_grounding",
        "canonical_text": (
            "If the encounter did not establish a diagnosis, medication, test result, "
            "exam finding, or follow-up instruction, the note must not invent it."
        ),
        "conditionality": "do_not_invent",
        "importance_tier": "critical",
        "categories": {"unsupported_facts"},
    },
    "visit_reason": {
        "label": "Visit Reason",
        "family": "visit_framing",
        "canonical_text": "The note states the visit reason or chief complaint supported by the encounter.",
        "conditionality": "if_relevant",
        "importance_tier": "major",
        "categories": {"visit_reason"},
    },
    "medication_plan": {
        "label": "Medication Plan",
        "family": "assessment_and_plan",
        "canonical_text": (
            "If medication changes or continuations were discussed, the note records them accurately "
            "and clearly distinguishes new changes from existing therapy."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {"medication_adjustment", "hypertension_rationale"},
    },
    "monitoring_followup": {
        "label": "Monitoring and Follow-Up",
        "family": "assessment_and_plan",
        "canonical_text": (
            "If monitoring instructions or follow-up plans were discussed, the note records their timing, "
            "purpose, and key details accurately."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {"bp_monitoring", "followup", "monitoring_and_followup"},
    },
    "symptom_characterization": {
        "label": "Symptom Characterization",
        "family": "symptoms",
        "canonical_text": (
            "The note accurately captures the clinically relevant symptoms and their important qualifiers "
            "without overstating or omitting key details."
        ),
        "conditionality": "if_relevant",
        "importance_tier": "major",
        "categories": {"bp_symptom_cluster", "chest_symptoms"},
    },
    "physical_exam": {
        "label": "Physical Exam Findings",
        "family": "exam_and_results",
        "canonical_text": "If physical exam findings were discussed, the note records the important findings accurately.",
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {"physical_exam"},
    },
    "diagnostics_and_tests": {
        "label": "Diagnostics and Tests",
        "family": "exam_and_results",
        "canonical_text": (
            "If diagnostic findings, ordered tests, or imaging were discussed, the note records them accurately "
            "and associates them with the right problem when relevant."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {"diagnostic_results", "lab_plan", "imaging_and_diagnostics"},
    },
    "assessment_and_diagnoses": {
        "label": "Assessment and Diagnoses",
        "family": "assessment_and_plan",
        "canonical_text": (
            "If an assessment or diagnosis was established in the encounter, the note records it accurately "
            "and does not invent one when it was not established."
        ),
        "conditionality": "if_established",
        "importance_tier": "critical",
        "categories": {"assessment_and_diagnoses"},
    },
    "counseling_and_lifestyle": {
        "label": "Counseling and Lifestyle",
        "family": "counseling",
        "canonical_text": (
            "If counseling, lifestyle guidance, hydration advice, alcohol limits, or exercise recommendations "
            "were discussed, the note records them accurately."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {
            "hydration_counseling",
            "alcohol_counseling",
            "lifestyle_diet_counseling",
        },
    },
    "medication_adherence": {
        "label": "Medication Adherence",
        "family": "adherence_and_context",
        "canonical_text": (
            "If adherence behavior or adherence aids were discussed, the note records them accurately without "
            "turning minor context into unsupported facts."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {"medication_adherence"},
    },
    "secondary_condition_management": {
        "label": "Secondary Condition Management",
        "family": "secondary_conditions",
        "canonical_text": (
            "If secondary chronic problems were discussed, the note records their current status and management accurately."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "minor",
        "categories": {"osteoarthritis_management"},
    },
    "history_and_context": {
        "label": "History and Context",
        "family": "history_and_context",
        "canonical_text": (
            "The note records the historical or prior-treatment context that is necessary to understand the current case."
        ),
        "conditionality": "if_relevant",
        "importance_tier": "major",
        "categories": {"history_and_context"},
    },
    "procedure_details": {
        "label": "Procedure Details",
        "family": "procedure_and_intervention",
        "canonical_text": (
            "If a procedure or intraoperative details were discussed, the note records the key procedural facts accurately."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {"procedure_details"},
    },
    "treatment_decision": {
        "label": "Treatment Decision",
        "family": "procedure_and_intervention",
        "canonical_text": (
            "If the encounter included treatment selection, rationale, or shared decision-making, the note records that "
            "decision and its rationale accurately."
        ),
        "conditionality": "if_discussed",
        "importance_tier": "major",
        "categories": {"treatment_rationale", "shared_decision_making"},
    },
    "note_structure": {
        "label": "Note Structure",
        "family": "documentation_quality",
        "canonical_text": "The note uses clear clinical structure and section organization appropriate for the task.",
        "conditionality": "absolute",
        "importance_tier": "minor",
        "categories": {"note_structure"},
    },
}


def _category_to_group() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group_id, spec in PRODUCTION_GROUPS.items():
        for category_id in spec["categories"]:
            mapping[str(category_id)] = group_id
    return mapping


CATEGORY_TO_GROUP = _category_to_group()


def _satisfaction_map(evaluations: Sequence[Dict[str, object]]) -> Dict[str, set[str]]:
    satisfied: Dict[str, set[str]] = {}
    for evaluation in evaluations:
        rubric_id = str(evaluation.get("rubric_id", ""))
        candidate_id = str(evaluation.get("candidate_id", ""))
        if not rubric_id or not candidate_id:
            continue
        if bool(evaluation.get("satisfied")):
            satisfied.setdefault(rubric_id, set()).add(candidate_id)
    return satisfied


def _behavior_stats(
    member_rubric_ids: Sequence[str],
    satisfied_by_rubric: Dict[str, set[str]],
    candidates: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    candidate_count = max(1, len(candidates))
    union_ids: set[str] = set()
    for rubric_id in member_rubric_ids:
        union_ids |= satisfied_by_rubric.get(rubric_id, set())

    strong = [
        candidate["candidate_id"]
        for candidate in candidates
        if candidate.get("quality_bucket") in {"gold_like", "strong_anchor", "frontier_generated", "open_generated"}
    ]
    weak = [
        candidate["candidate_id"]
        for candidate in candidates
        if candidate.get("quality_bucket") == "synthetically_degraded"
    ]
    p = len(union_ids) / candidate_count
    strong_rate = (len(union_ids & set(strong)) / max(1, len(strong))) if strong else 0.0
    weak_rate = (len(union_ids & set(weak)) / max(1, len(weak))) if weak else 0.0
    return {
        "coverage_count": float(len(union_ids)),
        "coverage_ratio": p,
        "discrimination_score": p * (1.0 - p),
        "strong_rate": strong_rate,
        "weak_rate": weak_rate,
    }


def build_production_bank(
    rubrics: Sequence[Dict[str, object]],
    evaluations: Sequence[Dict[str, object]],
    compressed_bank: Sequence[Dict[str, object]],
    candidates: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    rubric_lookup = {str(rubric.get("rubric_id")): rubric for rubric in rubrics}
    satisfied_by_rubric = _satisfaction_map(evaluations)

    grouped_entries: Dict[str, List[Dict[str, object]]] = {}
    for entry in compressed_bank:
        category_id = str(entry.get("category_id", ""))
        group_id = CATEGORY_TO_GROUP.get(category_id, category_id)
        grouped_entries.setdefault(group_id, []).append(entry)

    production_bank: List[Dict[str, object]] = []
    family_counts: Counter[str] = Counter()
    dropped_groups: List[str] = []
    merged_groups = 0
    rewritten_groups = 0

    for index, (group_id, entries) in enumerate(grouped_entries.items()):
        spec = PRODUCTION_GROUPS.get(group_id)
        if spec is None:
            continue

        member_rubric_ids: List[str] = []
        member_texts: List[str] = []
        representative_entry = max(entries, key=lambda item: (int(item.get("member_count", 0)), int(item.get("max_coverage_count", 0))))
        for entry in entries:
            for rubric_id in entry.get("member_rubric_ids", []):
                if rubric_id not in member_rubric_ids:
                    member_rubric_ids.append(str(rubric_id))
            for text in entry.get("member_texts", []):
                if text not in member_texts:
                    member_texts.append(str(text))

        if not member_rubric_ids:
            dropped_groups.append(group_id)
            continue

        behavior = _behavior_stats(member_rubric_ids, satisfied_by_rubric, candidates)
        action_taken = "merged" if len(entries) > 1 or len(member_rubric_ids) > 1 else "kept"
        if action_taken == "merged":
            merged_groups += 1
        if str(representative_entry.get("representative_text", "")) != str(spec["canonical_text"]):
            rewritten_groups += 1

        production_entry = {
            "production_rubric_id": f"production_{index}",
            "group_id": group_id,
            "label": spec["label"],
            "family": spec["family"],
            "canonical_text": spec["canonical_text"],
            "text": spec["canonical_text"],
            "conditionality": spec["conditionality"],
            "importance_tier": spec["importance_tier"],
            "action_taken": action_taken,
            "source_category_ids": [str(entry.get("category_id", "")) for entry in entries],
            "source_raw_rubric_ids": member_rubric_ids,
            "source_member_count": len(member_rubric_ids),
            "representative_rubric_id": representative_entry.get("representative_rubric_id"),
            "representative_text": representative_entry.get("representative_text"),
            "member_texts": member_texts,
            **behavior,
        }
        production_bank.append(production_entry)
        family_counts[spec["family"]] += 1

    production_bank.sort(
        key=lambda entry: (
            {"critical": 0, "major": 1, "minor": 2}.get(str(entry["importance_tier"]), 3),
            str(entry["family"]),
            str(entry["label"]),
        )
    )

    return {
        "production_bank": production_bank,
        "production_bank_summary": {
            "raw_rubric_count": len(rubrics),
            "compressed_rubric_count": len(compressed_bank),
            "production_bank_count": len(production_bank),
            "merged_group_count": merged_groups,
            "rewritten_group_count": rewritten_groups,
            "dropped_groups": dropped_groups,
            "family_counts": dict(family_counts),
        },
    }
