"""
Gold-driven rubric refinement and calibration hint utilities.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

_GENERIC_FAMILIES = {
    "",
    "other",
    "content_coverage",
    "instruction_adherence",
    "task_completion",
    "final_response_quality",
}
_GENERIC_DIMENSIONS = {
    "",
    "content_coverage",
    "completeness",
    "format_compliance",
    "format_communication",
    "grounding",
    "instruction_adherence",
    "meaning_preservation",
    "source_grounding",
    "style_transformation",
    "task_completion",
    "tool_result_grounding",
}
_FAMILY_TO_DIMENSION = {
    "communication_audience": "communication_quality",
    "certainty_language": "certainty_language",
    "context_grounding": "context_grounding",
    "diagnostic_reasoning": "diagnostic_reasoning",
    "factual_accuracy_safety": "factual_accuracy_safety",
    "follow_up": "follow_up_specificity",
    "intervention_plan": "intervention_plan",
    "medication_plan": "medication_management",
    "return_precautions_escalation": "return_precautions",
    "structure_documentation": "structure",
    "symptom_exam_detail": "symptom_detail",
    "testing_workup": "testing_plan",
}
_FAMILY_LABELS = {
    "certainty_language": "Calibrated certainty language",
    "communication_audience": "Audience-appropriate communication",
    "context_grounding": "Grounded in provided context",
    "diagnostic_reasoning": "Explicit diagnostic reasoning",
    "factual_accuracy_safety": "Clinically safe factual content",
    "follow_up": "Specific follow-up guidance",
    "instruction_adherence": "Instruction followed",
    "intervention_plan": "Intervention plan preserved",
    "medication_plan": "Medication plan preserved",
    "return_precautions_escalation": "Return precautions included",
    "structure_documentation": "Requested structure preserved",
    "symptom_exam_detail": "Symptom and exam detail preserved",
    "task_completion": "Task completed correctly",
    "testing_workup": "Testing and workup preserved",
}
_FAMILY_PROMPT_NUDGES = {
    "certainty_language": "Keep certainty and uncertainty wording as a dedicated criterion instead of folding it into general quality.",
    "communication_audience": "Keep audience targeting and plain-language adaptation as a dedicated criterion when the task specifies an audience.",
    "context_grounding": "Use a dedicated grounding criterion when the stronger artifact stays within the provided context and the weaker artifact invents or assumes details.",
    "diagnostic_reasoning": "Keep diagnostic reasoning or recommendation rationale as its own criterion rather than hiding it inside general completeness.",
    "factual_accuracy_safety": "Use a dedicated safety or factual-accuracy criterion when incorrect claims, unsafe advice, or unsupported certainty matter.",
    "follow_up": "Keep follow-up timing, surveillance intervals, and next-step monitoring as dedicated criteria instead of broad completeness checks.",
    "intervention_plan": "Keep referrals, procedures, surgeries, or escalation actions as dedicated intervention criteria when they are discussed.",
    "medication_plan": "Keep medication names, dose changes, contraindications, and regimen details as dedicated criteria rather than broad treatment checks.",
    "return_precautions_escalation": "Keep return precautions and escalation triggers as their own criterion when the task includes warning signs or emergency guidance.",
    "structure_documentation": "Keep required structure or sectioning as a dedicated criterion when the task calls for a specific document shape.",
    "symptom_exam_detail": "Keep symptom, history, vitals, or exam-detail capture as separate criteria instead of generic content coverage.",
    "testing_workup": "Keep tests, imaging, repeat studies, and workup recommendations as dedicated criteria rather than broad management checks.",
}
_PROMPT_NUDGE_TO_FAMILY = {text: family for family, text in _FAMILY_PROMPT_NUDGES.items()}
_GOLD_GAP_PRIORITY = {
    "family_mismatch": 0,
    "too_coarse": 1,
    "missing_gold_criterion": 2,
}


@dataclass
class GranularityGap:
    prompt_id: str
    task_profile_id: str
    source: str
    gap_type: str
    family: str
    priority: int
    reason: str
    expert_index: Optional[int] = None
    generated_index: Optional[int] = None
    gold_criterion: str = ""
    gold_family: str = ""
    generated_label: str = ""
    generated_requirement: str = ""
    generated_dimension: str = ""


@dataclass
class RefinementAction:
    kind: str
    prompt_id: str
    task_profile_id: str
    reason: str
    family: str = ""
    expert_index: Optional[int] = None
    generated_index: Optional[int] = None
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _tokens(*parts: str) -> set[str]:
    out: set[str] = set()
    for part in parts:
        for token in _WORD_RE.findall((part or "").lower()):
            if len(token) >= 4:
                out.add(token)
    return out


def _gap_dict(gap: Any) -> Dict[str, Any]:
    if isinstance(gap, GranularityGap):
        return {
            "prompt_id": gap.prompt_id,
            "task_profile_id": gap.task_profile_id,
            "source": gap.source,
            "gap_type": gap.gap_type,
            "family": gap.family,
            "priority": gap.priority,
            "reason": gap.reason,
            "expert_index": gap.expert_index,
            "generated_index": gap.generated_index,
            "gold_criterion": gap.gold_criterion,
            "gold_family": gap.gold_family,
            "generated_label": gap.generated_label,
            "generated_requirement": gap.generated_requirement,
            "generated_dimension": gap.generated_dimension,
        }
    if isinstance(gap, Mapping):
        return dict(gap)
    return {}


def _family_dimension(family: str) -> str:
    normalized = _norm(family)
    return _FAMILY_TO_DIMENSION.get(normalized, normalized or "gold_standard")


def _label_from_family_or_text(family: str, criterion: str) -> str:
    normalized_family = _norm(family)
    if normalized_family in _FAMILY_LABELS:
        return _FAMILY_LABELS[normalized_family]
    prefix = criterion.strip().split(".")[0].split(";")[0].strip()
    prefix = re.sub(
        r"^(?:the|a|an)\s+(?:response|note|document|artifact)\s+(?:must|should|needs to|should be)\s+",
        "",
        prefix,
        flags=re.IGNORECASE,
    )
    words = prefix.split()
    if not words:
        return "Gold-standard criterion"
    short = " ".join(words[:6]).strip()
    return short[:1].upper() + short[1:]


def _severity_from_points(points: Any) -> str:
    try:
        magnitude = abs(int(points))
    except (TypeError, ValueError):
        magnitude = 0
    if magnitude >= 10:
        return "hard_gate"
    if magnitude >= 7:
        return "high"
    if magnitude >= 4:
        return "medium"
    return "low"


def _best_generated_overlap(
    gold_row: Mapping[str, Any],
    generated_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[int], float]:
    best_index: Optional[int] = None
    best_score = 0.0
    gold_family = _norm(str(gold_row.get("family", "")))
    gold_tokens = _tokens(str(gold_row.get("criterion", "")))
    for idx, row in enumerate(generated_rows):
        row_tokens = _tokens(str(row.get("label", "")), str(row.get("requirement", "")))
        union = gold_tokens | row_tokens
        overlap = len(gold_tokens & row_tokens) / max(1, len(union))
        family_bonus = 0.2 if _norm(str(row.get("family", ""))) == gold_family and gold_family else 0.0
        score = overlap + family_bonus
        if score > best_score:
            best_index = idx
            best_score = score
    return best_index, best_score


def _calibration_policy(task_profile_id: str) -> Dict[str, Any]:
    normalized = _norm(task_profile_id)
    if normalized == "note_documentation":
        return {
            "allow_prompt_guidance": False,
            "allow_dimension_family_bias": False,
            "max_prompt_nudges": 0,
            "min_missing_for_prompt_nudge": 999,
            "min_missing_for_dimension_bias": 999,
        }
    return {
        "allow_prompt_guidance": True,
        "allow_dimension_family_bias": True,
        "max_prompt_nudges": 3,
        "min_missing_for_prompt_nudge": 4,
        "min_missing_for_dimension_bias": 6,
    }


def calibration_enabled_profiles(calibration_hints: Optional[Mapping[str, Any]]) -> List[str]:
    if not calibration_hints:
        return []
    eligibility = calibration_hints.get("eligibility")
    if not isinstance(eligibility, Mapping):
        return []
    return sorted(
        {
            _norm(str(profile))
            for profile in (eligibility.get("enabled_profiles") or [])
            if str(profile).strip()
        }
    )


def calibration_profile_is_enabled(
    calibration_hints: Optional[Mapping[str, Any]],
    *,
    task_profile_id: str,
) -> bool:
    normalized = _norm(task_profile_id)
    if not normalized:
        return False
    return normalized in set(calibration_enabled_profiles(calibration_hints))


def build_prompt_calibration_guidance(
    calibration_hints: Optional[Mapping[str, Any]],
    *,
    task_profile_id: str,
) -> str:
    if not calibration_hints:
        return ""
    if not calibration_profile_is_enabled(calibration_hints, task_profile_id=task_profile_id):
        return ""
    policy = _calibration_policy(task_profile_id)
    if not policy["allow_prompt_guidance"]:
        return ""
    profile_hints = calibration_hints.get("by_task_profile", {}).get(task_profile_id, {})
    missing_counts = {
        _norm(str(key)): int(value)
        for key, value in (profile_hints.get("observed_missing_family_counts", {}) or {}).items()
        if str(key).strip()
    }
    nudges: List[str] = []
    for item in profile_hints.get("prompt_nudges", []):
        text = str(item).strip()
        if not text:
            continue
        family = _PROMPT_NUDGE_TO_FAMILY.get(text, "")
        if family:
            if family in _GENERIC_FAMILIES:
                continue
            if missing_counts.get(_norm(family), 0) < int(policy["min_missing_for_prompt_nudge"]):
                continue
        nudges.append(text)
    if not nudges:
        return ""
    bullets = "\n".join(f"- {item}" for item in nudges[: int(policy["max_prompt_nudges"])])
    return f"PRIOR GOLD CALIBRATION HINTS:\n{bullets}\n"


def apply_calibration_hints_to_generated_row(
    row: Mapping[str, Any],
    *,
    task_profile_id: str,
    calibration_hints: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    out = dict(row)
    if not calibration_hints:
        return out
    if not calibration_profile_is_enabled(calibration_hints, task_profile_id=task_profile_id):
        return out
    policy = _calibration_policy(task_profile_id)
    if not policy["allow_dimension_family_bias"]:
        return out
    profile_hints = calibration_hints.get("by_task_profile", {}).get(task_profile_id, {})
    missing_counts = {
        _norm(str(key)): int(value)
        for key, value in (profile_hints.get("observed_missing_family_counts", {}) or {}).items()
        if str(key).strip()
    }
    dimension_bias = {
        _norm(str(key)): _norm(str(value))
        for key, value in (profile_hints.get("dimension_family_bias", {}) or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if not dimension_bias:
        return out
    current_family = _norm(str(out.get("family", "")))
    dimension = _norm(str(out.get("dimension", "")))
    override_family = dimension_bias.get(dimension)
    if (
        override_family
        and override_family not in _GENERIC_FAMILIES
        and missing_counts.get(override_family, 0) >= int(policy["min_missing_for_dimension_bias"])
        and current_family in _GENERIC_FAMILIES
    ):
        out["family_pre_calibration"] = out.get("family", "")
        out["family"] = override_family
        out["calibration_family_override"] = override_family
    return out


def classify_granularity_gaps(
    *,
    provider_id: str,
    prompt_id: str,
    task_profile_id: str,
    gold_rows: Sequence[Mapping[str, Any]],
    generated_rows: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
) -> Dict[str, Any]:
    generated_lookup = {
        int(row.get("generated_index", idx)): dict(row)
        for idx, row in enumerate(generated_rows)
        if isinstance(row, Mapping)
    }
    expert_link_counts: Counter[int] = Counter()
    assessment_lookup: Dict[int, Mapping[str, Any]] = {}
    for row in alignment.get("expert_matches", []):
        try:
            if row.get("match_label") in {"direct", "partial"}:
                expert_link_counts[int(row["best_generated_index"])] += 1
        except (KeyError, TypeError, ValueError):
            continue
    for row in alignment.get("generated_assessments", []):
        try:
            assessment_lookup[int(row["generated_index"])] = row
        except (KeyError, TypeError, ValueError):
            continue

    gold_family_set = {_norm(str(row.get("family", ""))) for row in gold_rows}
    gaps: List[GranularityGap] = []

    for raw_match in alignment.get("expert_matches", []):
        try:
            expert_index = int(raw_match["expert_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if expert_index < 0 or expert_index >= len(gold_rows):
            continue
        match_label = str(raw_match.get("match_label", "none")).strip().lower()
        if match_label == "direct":
            continue
        gold_row = gold_rows[expert_index]
        gold_family = _norm(str(gold_row.get("family", "")))
        generated_index: Optional[int] = None
        generated_row: Optional[Mapping[str, Any]] = None
        try:
            candidate = raw_match.get("best_generated_index")
            if candidate is not None:
                generated_index = int(candidate)
                generated_row = generated_lookup.get(generated_index)
        except (TypeError, ValueError):
            generated_index = None
            generated_row = None

        overlap_index, overlap_score = _best_generated_overlap(gold_row, generated_rows)
        if generated_row is None and overlap_index is not None:
            generated_index = overlap_index
            generated_row = generated_lookup.get(overlap_index)

        generated_family = _norm(str((generated_row or {}).get("family", "")))
        gap_type = "missing_gold_criterion"
        if match_label == "partial":
            if generated_row is not None and (
                generated_family == gold_family
                or generated_family in _GENERIC_FAMILIES
                or expert_link_counts.get(generated_index or -1, 0) > 1
            ):
                gap_type = "too_coarse"
            elif generated_row is not None:
                gap_type = "family_mismatch"
        elif generated_row is not None and overlap_score >= 0.18:
            if generated_family == gold_family or generated_family in _GENERIC_FAMILIES:
                gap_type = "too_coarse"
            else:
                gap_type = "family_mismatch"

        gaps.append(
            GranularityGap(
                prompt_id=prompt_id,
                task_profile_id=task_profile_id,
                source="gold",
                gap_type=gap_type,
                family=gold_family,
                gold_family=gold_family,
                priority=abs(int(gold_row.get("points", 0) or 0)),
                reason=str(raw_match.get("reason", "")),
                expert_index=expert_index,
                generated_index=generated_index,
                gold_criterion=str(gold_row.get("criterion", "")),
                generated_label=str((generated_row or {}).get("label", "")),
                generated_requirement=str((generated_row or {}).get("requirement", "")),
                generated_dimension=str((generated_row or {}).get("dimension", "")),
            )
        )

    for raw_assessment in alignment.get("generated_assessments", []):
        try:
            generated_index = int(raw_assessment["generated_index"])
        except (KeyError, TypeError, ValueError):
            continue
        precision_label = str(raw_assessment.get("precision_label", "")).strip().lower()
        if precision_label == "aligned":
            continue
        generated_row = generated_lookup.get(generated_index, {})
        generated_family = _norm(str(generated_row.get("family", "")))
        matched_expert_indices = [
            int(item)
            for item in (raw_assessment.get("matched_expert_indices") or [])
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit())
        ]
        gold_family = ""
        if matched_expert_indices:
            idx = matched_expert_indices[0]
            if 0 <= idx < len(gold_rows):
                gold_family = _norm(str(gold_rows[idx].get("family", "")))

        if precision_label == "broader_but_valid":
            gap_type = "too_coarse"
        elif precision_label == "valid_extra":
            gap_type = "too_granular" if generated_family in _GENERIC_FAMILIES or generated_family in gold_family_set else "valid_extra"
        else:
            gap_type = "family_mismatch" if generated_family and generated_family != "other" else "off_target"

        gaps.append(
            GranularityGap(
                prompt_id=prompt_id,
                task_profile_id=task_profile_id,
                source="generated",
                gap_type=gap_type,
                family=generated_family,
                gold_family=gold_family,
                priority=1,
                reason=str(raw_assessment.get("reason", "")),
                generated_index=generated_index,
                generated_label=str(generated_row.get("label", "")),
                generated_requirement=str(generated_row.get("requirement", "")),
                generated_dimension=str(generated_row.get("dimension", "")),
            )
        )

    gap_counts: Counter[str] = Counter()
    family_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for gap in gaps:
        gap_counts[gap.gap_type] += 1
        family_key = gap.family or gap.gold_family or "other"
        family_counts[family_key][gap.gap_type] += 1

    return {
        "schema": "compiled_gold_granularity_report_v1",
        "provider_id": provider_id,
        "prompt_id": prompt_id,
        "task_profile_id": task_profile_id,
        "gap_counts": dict(gap_counts),
        "family_gap_counts": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(family_counts.items())
        },
        "gaps": gaps,
    }


def refine_generated_rows(
    *,
    prompt_id: str,
    task_profile_id: str,
    gold_rows: Sequence[Mapping[str, Any]],
    generated_rows: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    granularity_report: Mapping[str, Any],
    calibration_hints: Optional[Mapping[str, Any]],
    max_new_rows: int = 6,
) -> Dict[str, Any]:
    profile_hints = (calibration_hints or {}).get("by_task_profile", {}).get(task_profile_id, {})
    drop_families = {_norm(str(item)) for item in profile_hints.get("drop_families", []) if str(item).strip()}
    assessment_lookup: Dict[int, Mapping[str, Any]] = {}
    for row in alignment.get("generated_assessments", []):
        try:
            assessment_lookup[int(row["generated_index"])] = row
        except (KeyError, TypeError, ValueError):
            continue

    prepared_rows: List[Dict[str, Any]] = [
        apply_calibration_hints_to_generated_row(
            row,
            task_profile_id=task_profile_id,
            calibration_hints=calibration_hints,
        )
        for row in generated_rows
    ]
    generated_lookup = {
        int(row.get("generated_index", idx)): row
        for idx, row in enumerate(prepared_rows)
    }

    kept_rows: List[Dict[str, Any]] = []
    actions: List[RefinementAction] = []
    gold_family_set = {_norm(str(row.get("family", ""))) for row in gold_rows}

    for idx, row in generated_lookup.items():
        assessment = assessment_lookup.get(idx, {})
        precision_label = str(assessment.get("precision_label", "")).strip().lower()
        family = _norm(str(row.get("family", "")))
        should_drop = False
        if precision_label == "off_target" and (family not in gold_family_set or family in drop_families or family == "other"):
            should_drop = True
        elif precision_label == "valid_extra" and family in drop_families:
            should_drop = True
        if should_drop:
            actions.append(
                RefinementAction(
                    kind="drop_generated_row",
                    prompt_id=prompt_id,
                    task_profile_id=task_profile_id,
                    generated_index=idx,
                    family=family,
                    reason=str(assessment.get("reason", "gold refinement drop")),
                    before=dict(row),
                )
            )
            continue
        kept_rows.append(dict(row))

    existing_requirements = {_norm(str(row.get("requirement", ""))) for row in kept_rows}
    candidate_gold_gaps = [
        _gap_dict(raw_gap)
        for raw_gap in granularity_report.get("gaps", [])
        if _gap_dict(raw_gap).get("source") == "gold"
        and _gap_dict(raw_gap).get("gap_type") in {"too_coarse", "family_mismatch", "missing_gold_criterion"}
    ]
    candidate_gold_gaps.sort(
        key=lambda gap: (
            _GOLD_GAP_PRIORITY.get(str(gap.get("gap_type", "")), 99),
            -(int(gap.get("priority", 0) or 0)),
            str(gap.get("family", "")),
            str(gap.get("gold_criterion", "")),
        )
    )

    additions = 0
    for gap in candidate_gold_gaps:
        if additions >= max_new_rows:
            break
        expert_index = gap.get("expert_index")
        if not isinstance(expert_index, int) or expert_index < 0 or expert_index >= len(gold_rows):
            continue
        gold_row = gold_rows[expert_index]
        requirement = str(gold_row.get("criterion", "")).strip()
        if not requirement:
            continue
        normalized_requirement = _norm(requirement)
        if normalized_requirement in existing_requirements:
            continue
        family = _norm(str(gold_row.get("family", "")))
        refined_row = {
            "generated_index": -1,
            "dimension": _family_dimension(family),
            "label": _label_from_family_or_text(family, requirement),
            "requirement": requirement,
            "severity_tier": _severity_from_points(gold_row.get("points")),
            "count": 1,
            "example_ids": [],
            "pair_ids": [],
            "family": family,
            "polarity": str(gold_row.get("polarity", "")),
            "refinement_origin": "gold_refinement",
            "gold_anchor_expert_index": expert_index,
            "gold_gap_type": str(gap.get("gap_type", "")),
        }
        kept_rows.append(refined_row)
        existing_requirements.add(normalized_requirement)
        additions += 1
        actions.append(
            RefinementAction(
                kind="add_gold_aligned_row",
                prompt_id=prompt_id,
                task_profile_id=task_profile_id,
                expert_index=expert_index,
                family=family,
                reason=str(gap.get("gap_type", "gold gap")),
                after=dict(refined_row),
            )
        )

    final_rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str, str]] = set()
    for row in kept_rows:
        family = _norm(str(row.get("family", "")))
        key = (_norm(str(row.get("label", ""))), _norm(str(row.get("requirement", ""))), family)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        finalized = dict(row)
        finalized["pre_refinement_generated_index"] = finalized.get("generated_index")
        finalized["generated_index"] = len(final_rows)
        final_rows.append(finalized)

    return {
        "schema": "compiled_gold_refinement_result_v1",
        "prompt_id": prompt_id,
        "task_profile_id": task_profile_id,
        "changed": bool(actions),
        "generated_rows": final_rows,
        "actions": actions,
        "summary": {
            "rows_before": len(generated_rows),
            "rows_after": len(final_rows),
            "rows_added": sum(1 for action in actions if action.kind == "add_gold_aligned_row"),
            "rows_dropped": sum(1 for action in actions if action.kind == "drop_generated_row"),
        },
    }


def aggregate_granularity_reports(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gap_counts: Counter[str] = Counter()
    by_task_profile: Dict[str, Counter[str]] = defaultdict(Counter)
    by_family: Dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        task_profile_id = str(report.get("task_profile_id", "unknown"))
        for raw_gap in report.get("gaps", []):
            gap = _gap_dict(raw_gap)
            gap_type = str(gap.get("gap_type", "unknown"))
            family = str(gap.get("family") or gap.get("gold_family") or "other")
            gap_counts[gap_type] += 1
            by_task_profile[task_profile_id][gap_type] += 1
            by_family[family][gap_type] += 1
    return {
        "schema": "compiled_gold_granularity_summary_v1",
        "report_count": len(reports),
        "gap_counts": dict(sorted(gap_counts.items())),
        "by_task_profile": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(by_task_profile.items())
        },
        "by_family": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(by_family.items())
        },
    }


def derive_calibration_hints(
    reports: Sequence[Mapping[str, Any]],
    *,
    provider_id: str,
    existing_hints: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    missing_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    extra_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    bias_counts: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    global_gap_counts: Counter[str] = Counter()

    for report in reports:
        task_profile_id = str(report.get("task_profile_id", "unknown"))
        for raw_gap in report.get("gaps", []):
            gap = _gap_dict(raw_gap)
            gap_type = str(gap.get("gap_type", "unknown"))
            global_gap_counts[gap_type] += 1
            family = _norm(str(gap.get("family") or gap.get("gold_family") or ""))
            gold_family = _norm(str(gap.get("gold_family", "")))
            generated_dimension = _norm(str(gap.get("generated_dimension", "")))
            if gap.get("source") == "gold" and gap_type in {"too_coarse", "family_mismatch", "missing_gold_criterion"} and family:
                missing_counts[task_profile_id][family] += 1
            if gap.get("source") == "generated" and gap_type in {"too_granular", "family_mismatch", "off_target"} and family:
                extra_counts[task_profile_id][family] += 1
            if generated_dimension and gold_family and generated_dimension in _GENERIC_DIMENSIONS and gold_family not in _GENERIC_FAMILIES:
                bias_counts[task_profile_id][generated_dimension][gold_family] += 1

    by_task_profile: Dict[str, Dict[str, Any]] = {}
    profiles = set(missing_counts) | set(extra_counts) | set((existing_hints or {}).get("by_task_profile", {}))
    for task_profile_id in sorted(profiles):
        existing_profile = (existing_hints or {}).get("by_task_profile", {}).get(task_profile_id, {})
        dimension_family_bias = dict(existing_profile.get("dimension_family_bias", {}))
        for dimension, counter in bias_counts.get(task_profile_id, {}).items():
            if not counter:
                continue
            family, count = counter.most_common(1)[0]
            if count >= 2:
                dimension_family_bias[dimension] = family

        split_bias_families = set(existing_profile.get("split_bias_families", []))
        merge_bias_families = set(existing_profile.get("merge_bias_families", []))
        drop_families = set(existing_profile.get("drop_families", []))
        prompt_nudges = set(existing_profile.get("prompt_nudges", []))

        for family, count in missing_counts.get(task_profile_id, {}).items():
            if count >= 2:
                split_bias_families.add(family)
                nudge = _FAMILY_PROMPT_NUDGES.get(family)
                if nudge:
                    prompt_nudges.add(nudge)
        for family, count in extra_counts.get(task_profile_id, {}).items():
            if count >= 2:
                merge_bias_families.add(family)
            if family in {"other", "content_coverage"} and count >= 2:
                drop_families.add(family)

        if not (dimension_family_bias or split_bias_families or merge_bias_families or drop_families or prompt_nudges):
            continue
        by_task_profile[task_profile_id] = {
            "dimension_family_bias": dict(sorted(dimension_family_bias.items())),
            "split_bias_families": sorted(str(item) for item in split_bias_families if str(item).strip()),
            "merge_bias_families": sorted(str(item) for item in merge_bias_families if str(item).strip()),
            "drop_families": sorted(str(item) for item in drop_families if str(item).strip()),
            "prompt_nudges": sorted(str(item) for item in prompt_nudges if str(item).strip()),
            "observed_missing_family_counts": dict(sorted(missing_counts.get(task_profile_id, {}).items())),
            "observed_extra_family_counts": dict(sorted(extra_counts.get(task_profile_id, {}).items())),
        }

    return {
        "schema": "compiled_gold_calibration_hints_v1",
        "provider_id": provider_id,
        "summary": {
            "reports_analyzed": len(reports),
            "gap_counts": dict(sorted(global_gap_counts.items())),
        },
        "by_task_profile": by_task_profile,
    }


def derive_calibration_profile_policy(
    *,
    baseline_summary: Mapping[str, Any],
    apply_summary: Mapping[str, Any],
    min_examples: int = 5,
    protected_profiles: Sequence[str] = ("note_documentation",),
) -> Dict[str, Any]:
    baseline_profiles = baseline_summary.get("pre_refinement_alignment_by_task_profile", {}) or {}
    apply_profiles = apply_summary.get("pre_refinement_alignment_by_task_profile", {}) or {}
    protected = {_norm(str(profile)) for profile in protected_profiles if str(profile).strip()}
    profile_ids = sorted(
        {
            _norm(str(profile))
            for profile in list(baseline_profiles.keys()) + list(apply_profiles.keys())
            if str(profile).strip()
        }
    )
    metrics = ("weighted_recall", "expert_recall", "generated_precision", "generated_off_target")
    enabled_profiles: List[str] = []
    by_task_profile: Dict[str, Dict[str, Any]] = {}

    for profile_id in profile_ids:
        baseline = baseline_profiles.get(profile_id, {}) or {}
        applied = apply_profiles.get(profile_id, {}) or {}
        examples = int(applied.get("examples_scored", baseline.get("examples_scored", 0)) or 0)
        delta = {
            key: float(applied.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0)
            for key in metrics
        }
        eligible = (
            examples >= min_examples
            and profile_id not in protected
            and delta["weighted_recall"] >= 0.0
            and delta["expert_recall"] >= 0.0
            and delta["generated_precision"] >= 0.0
            and delta["generated_off_target"] <= 0.0
        )
        if eligible:
            enabled_profiles.append(profile_id)
        if profile_id in protected:
            reason = "protected_profile"
        elif examples < min_examples:
            reason = "insufficient_heldout_examples"
        elif eligible:
            reason = "heldout_improvement"
        else:
            failures: List[str] = []
            if delta["weighted_recall"] < 0.0:
                failures.append("weighted_recall_regressed")
            if delta["expert_recall"] < 0.0:
                failures.append("expert_recall_regressed")
            if delta["generated_precision"] < 0.0:
                failures.append("precision_regressed")
            if delta["generated_off_target"] > 0.0:
                failures.append("off_target_increased")
            reason = ",".join(failures) if failures else "not_eligible"
        by_task_profile[profile_id] = {
            "eligible": eligible,
            "reason": reason,
            "examples_scored": examples,
            "baseline": {key: baseline.get(key, 0.0) for key in ("examples_scored", *metrics)},
            "apply": {key: applied.get(key, 0.0) for key in ("examples_scored", *metrics)},
            "delta": delta,
        }

    return {
        "schema": "compiled_calibration_profile_policy_v1",
        "comparison_scope": "pre_refinement_alignment_by_task_profile",
        "min_examples": int(min_examples),
        "protected_profiles": sorted(protected),
        "enabled_profiles": sorted(enabled_profiles),
        "source_runs": {
            "baseline_run_summary": str(baseline_summary.get("paths", {}).get("summary_source", "")),
            "apply_run_summary": str(apply_summary.get("paths", {}).get("summary_source", "")),
        },
        "by_task_profile": by_task_profile,
    }


__all__ = [
    "GranularityGap",
    "RefinementAction",
    "aggregate_granularity_reports",
    "calibration_enabled_profiles",
    "calibration_profile_is_enabled",
    "apply_calibration_hints_to_generated_row",
    "build_prompt_calibration_guidance",
    "classify_granularity_gaps",
    "derive_calibration_profile_policy",
    "derive_calibration_hints",
    "refine_generated_rows",
]
