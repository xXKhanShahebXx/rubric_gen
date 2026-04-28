"""
Heuristic judge (starter scaffold): evaluates each compiled criterion using eval_kind + judge_hints.

Non-LLM, intentionally shallow — demonstrates per-criterion dispatch, not clinical adequacy.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from rubric_gen.compiled.schema import (
    CaseEvaluationRecord,
    CaseRubric,
    CompiledCriterion,
    CriterionResult,
    DimensionScore,
    EvidenceAnchor,
)

# High-risk terms that must not appear in the note unless also in the dialogue (starter list).
_HIGH_RISK_TERMS = [
    "malignancy",
    "cancer",
    "metastasis",
    "pulmonary embolism",
    "acute mi",
    "myocardial infarction",
    "bacterial meningitis",
]

# Phrases implying stronger certainty than typical conversational hedging (starter).
_STRONG_CLAIM_MARKERS = [
    "pathognomonic",
    "definitively cured",
    "stage iv",
    "stage 4",
    "metastatic disease",
]

_EVAL_KIND_DISPATCH: Dict[str, str] = {
    "no_unsupported_diagnosis": "hard_gate_unsupported_bundle",
    "no_unsupported_certainty": "hard_gate_certainty_vs_dialogue",
    "required_sections_note_family": "section_header_coverage",
    "chief_complaint_symptom_presence": "dialogue_anchor_terms_in_note",
    "symptom_detail_attributes": "symptom_detail_bundle",
    "associated_symptom_or_negative": "relevant_negative_mirror",
    "discussed_study_or_result_presence": "discussed_study_mentioned_in_note",
}


def _resolve_eval_kind(crit: CompiledCriterion) -> str:
    if crit.eval_kind:
        return crit.eval_kind
    if crit.template_id and crit.template_id in _EVAL_KIND_DISPATCH:
        return _EVAL_KIND_DISPATCH[crit.template_id]
    return "generic_pass"


def _hard_unsupported_bundle(note: str, dialogue: str) -> Tuple[str, str, float, List[str]]:
    n = note.lower()
    d = dialogue.lower()
    for term in _HIGH_RISK_TERMS:
        if term in n and term not in d:
            return (
                "UNMET",
                f"Note mentions '{term}' which is absent from the dialogue (starter high-risk list).",
                0.0,
                ["unsupported_inference"],
            )
    return (
        "MET",
        "No high-risk unsupported terms detected relative to dialogue (starter heuristic).",
        1.0,
        [],
    )


def _hard_certainty_vs_dialogue(note: str, dialogue: str) -> Tuple[str, str, float, List[str]]:
    """Flag a few strong claim markers in the note that never appear in the dialogue."""
    n = note.lower()
    d = dialogue.lower()
    for phrase in _STRONG_CLAIM_MARKERS:
        if phrase in n and phrase not in d:
            return (
                "UNMET",
                f"Note contains strong language ('{phrase}') not grounded in the dialogue transcript.",
                0.0,
                ["certainty_inflation"],
            )
    return (
        "MET",
        "No obvious certainty-inflation phrases detected beyond dialogue support (starter heuristic).",
        1.0,
        [],
    )


# When the case was inferred as SOAP from reference layout, candidates may still use narrative
# headings (HPI, PE, etc.). These alternates are additive; they do not change the compiler output.
_SOAP_SECTION_ALTERNATES: Dict[str, List[str]] = {
    "subjective": [
        r"(?mi)\bhpi\b",
        r"(?mi)history\s+of\s+present",
        r"(?mi)chief\s+complaint",
        r"(?mi)\binterval\s+history\b",
    ],
    "objective": [
        r"(?mi)\bphysical\s+exam(ination)?\b",
        r"(?mi)\bvitals?\b",
        r"(?mi)\bexam\b",
        r"(?mi)\blabs?\b",
    ],
    "assessment": [
        r"(?mi)\bimpression\b",
        r"(?mi)\bdiagnos(e|is)\b",
        r"(?mi)\bclinical\s+assessment\b",
    ],
    "plan": [
        r"(?mi)\bfollow[\s-]*up\b",
        r"(?mi)\bdisposition\b",
        r"(?mi)\brecommendations?\b",
    ],
}


def _section_header_coverage(note: str, crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    sections: List[Dict[str, Any]] = hints.get("sections") or []
    if not sections:
        return "MET", "No section patterns compiled; default MET.", 1.0, []

    nf = hints.get("note_family_id") or ""
    matched = 0
    missing_labels: List[str] = []
    for sec in sections:
        label = sec.get("label") or sec.get("section_id", "section")
        sid = str(sec.get("section_id") or "")
        patterns = list(sec.get("patterns") or [])
        if nf == "soap_note" and sid in _SOAP_SECTION_ALTERNATES:
            patterns = patterns + _SOAP_SECTION_ALTERNATES[sid]
        ok = False
        for p in patterns:
            try:
                if re.search(p, note, re.MULTILINE):
                    ok = True
                    break
            except re.error:
                continue
        if ok:
            matched += 1
        else:
            missing_labels.append(str(label))

    if matched == len(sections):
        return (
            "MET",
            f"All {len(sections)} expected section header patterns found (heuristic).",
            1.0,
            [],
        )
    return (
        "UNMET",
        "Missing or unclear section headers for: "
        + ", ".join(missing_labels[:6])
        + ("…" if len(missing_labels) > 6 else ""),
        max(0.0, matched / max(1, len(sections))),
        ["section_missing"],
    )


def _dialogue_anchor_terms_in_note(
    note: str, dialogue: str, crit: CompiledCriterion
) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    terms: List[str] = hints.get("terms") or []
    min_matched = int(hints.get("min_terms_matched") or 1)
    nl = note.lower()
    dl = dialogue.lower()

    grounded = [t.lower() for t in terms if len(t) > 2 and t.lower() in dl]
    if not grounded:
        return (
            "MET",
            "No grounded chief-complaint terms extracted; criterion not strongly triggered (starter).",
            1.0,
            [],
        )

    hits_in_note = sum(1 for t in grounded if t in nl)
    if hits_in_note >= min_matched:
        return (
            "MET",
            f"Note reflects at least {min_matched} anchored presenting-term(s) from the dialogue.",
            1.0,
            [],
        )
    return (
        "UNMET",
        f"Presenting terms discussed in dialogue ({grounded[:4]}) are not clearly reflected in the note.",
        0.0,
        ["omission"],
    )


def _symptom_detail_bundle(note: str, crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    patterns: List[Dict[str, Any]] = hints.get("patterns") or []
    if not patterns:
        return "MET", "No symptom-detail patterns active; default MET.", 1.0, []

    nl = note.lower()
    failures: List[str] = []
    for p in patterns:
        label = p.get("label", "detail")
        kws = [k.lower() for k in p.get("note_keywords", [])]
        req = int(p.get("require_min_keyword_hits") or 1)
        hits = sum(1 for k in kws if k in nl)
        if hits < req:
            failures.append(label)

    if not failures:
        return (
            "MET",
            "Symptom attributes implied by dialogue appear represented in the note (keyword heuristic).",
            1.0,
            [],
        )
    return (
        "UNMET",
        "Insufficient keyword coverage for: " + ", ".join(failures[:4]),
        0.0,
        ["omission"],
    )


def _relevant_negative_mirror(note: str, dialogue: str, crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    d_markers = [m.lower() for m in hints.get("dialogue_markers") or []]
    n_markers = [m.lower() for m in hints.get("note_markers") or []]
    dl = dialogue.lower()
    nl = note.lower()

    if not d_markers or not n_markers:
        return "MET", "Negative mirror hints missing; default MET.", 1.0, []

    dialogue_hit = any(m in dl for m in d_markers)
    if not dialogue_hit:
        return "MET", "Dialogue does not match negative markers; criterion relaxed.", 1.0, []

    note_hit = any(m in nl for m in n_markers)
    if note_hit:
        return "MET", "Pertinent negative appears reflected in the note (starter heuristic).", 1.0, []
    return (
        "UNMET",
        "Dialogue documents a pertinent negative, but the note does not clearly mirror it.",
        0.0,
        ["omission"],
    )


def _anchor_terms_presence(note: str, dialogue: str, crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    terms = [t.lower() for t in hints.get("terms") or []]
    min_matched = int(hints.get("min_terms_matched") or 1)
    nl = note.lower()
    dl = dialogue.lower()

    grounded = [t for t in terms if t in dl]
    if not grounded:
        return "MET", "Anchor terms not grounded in dialogue; default MET.", 1.0, []

    hits = sum(1 for t in grounded if t in nl)
    if hits >= min_matched:
        return "MET", "Associated symptom language appears in the note.", 1.0, []
    return (
        "UNMET",
        "Associated symptom discussed in dialogue is not clearly documented in the note.",
        0.0,
        ["omission"],
    )


def _discussed_study_mentioned_in_note(note: str, dialogue: str, crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    study_terms = [t.lower() for t in hints.get("study_terms") or []]
    nl = note.lower()
    dl = dialogue.lower()

    missing: List[str] = []
    for t in study_terms:
        if t in dl and t not in nl:
            missing.append(t)

    if not missing:
        return (
            "MET",
            "Discussed study/lab terms from the dialogue appear in the note (starter heuristic).",
            1.0,
            [],
        )
    return (
        "UNMET",
        f"Discussed study/lab term(s) missing from note: {', '.join(missing[:5])}",
        0.0,
        ["omission"],
    )


def _artifact_marker_presence(note: str, crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    markers = [str(marker).lower() for marker in hints.get("markers") or [] if str(marker).strip()]
    min_markers = int(hints.get("min_markers_matched") or 1)
    if not markers:
        return "MET", "No generic markers configured; default MET.", 1.0, []
    lower = note.lower()
    hits = sum(1 for marker in markers if marker in lower)
    if hits >= min_markers:
        return "MET", "Required task markers appear present in the artifact.", 1.0, []
    return (
        "UNMET",
        "Expected task markers are missing from the artifact.",
        max(0.0, hits / max(1, min_markers)),
        ["omission"],
    )


def _generic_unsupported_assertions(note: str, dialogue: str, crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    hints = crit.judge_hints or {}
    markers = [str(marker).lower() for marker in hints.get("unsupported_markers") or [] if str(marker).strip()]
    lower_note = note.lower()
    lower_context = dialogue.lower()
    for marker in markers:
        if marker in lower_note and marker not in lower_context:
            return (
                "UNMET",
                f"Artifact contains unsupported assertion marker '{marker}' outside the task context.",
                0.0,
                ["unsupported_inference"],
            )
    return "MET", "No configured unsupported-assertion markers were triggered.", 1.0, []


def _generic_pass(crit: CompiledCriterion) -> Tuple[str, str, float, List[str]]:
    return "MET", f"No evaluator for criterion; default MET (scaffold). template_id={crit.template_id!r}", 1.0, []


def _evaluate_one(
    crit: CompiledCriterion,
    note_text: str,
    dialogue: str,
) -> CriterionResult:
    kind = _resolve_eval_kind(crit)
    verdict: str
    rationale: str
    score: float
    codes: List[str]

    if kind == "hard_gate_unsupported_bundle":
        verdict, rationale, score, codes = _hard_unsupported_bundle(note_text, dialogue)
    elif kind == "hard_gate_certainty_vs_dialogue":
        verdict, rationale, score, codes = _hard_certainty_vs_dialogue(note_text, dialogue)
    elif kind == "section_header_coverage":
        verdict, rationale, score, codes = _section_header_coverage(note_text, crit)
    elif kind == "dialogue_anchor_terms_in_note":
        verdict, rationale, score, codes = _dialogue_anchor_terms_in_note(note_text, dialogue, crit)
    elif kind == "symptom_detail_bundle":
        verdict, rationale, score, codes = _symptom_detail_bundle(note_text, crit)
    elif kind == "relevant_negative_mirror":
        verdict, rationale, score, codes = _relevant_negative_mirror(note_text, dialogue, crit)
    elif kind == "anchor_terms_presence":
        verdict, rationale, score, codes = _anchor_terms_presence(note_text, dialogue, crit)
    elif kind == "discussed_study_mentioned_in_note":
        verdict, rationale, score, codes = _discussed_study_mentioned_in_note(note_text, dialogue, crit)
    elif kind == "artifact_marker_presence":
        verdict, rationale, score, codes = _artifact_marker_presence(note_text, crit)
    elif kind == "generic_unsupported_assertions":
        verdict, rationale, score, codes = _generic_unsupported_assertions(note_text, dialogue, crit)
    elif kind == "discovered_provisional_soft":
        verdict, rationale, score, codes = _generic_pass(crit)
    else:
        verdict, rationale, score, codes = _generic_pass(crit)

    ev: List[EvidenceAnchor] = []
    if crit.evidence_anchors:
        ev = crit.evidence_anchors[:2]

    return CriterionResult(
        criterion_id=crit.criterion_id,
        verdict=verdict,
        rationale=f"[{kind}] {rationale}",
        score_value=score,
        error_codes=codes,
        evidence_used=ev,
    )


def dimension_scores_from_results(
    case_rubric: CaseRubric,
    hard_results: List[CriterionResult],
    soft_results: List[CriterionResult],
) -> List[DimensionScore]:
    by_dim: Dict[str, List[float]] = {}
    crit_ids = {c.criterion_id: c.dimension_id for c in case_rubric.hard_gates + case_rubric.soft_checks}

    for r in hard_results + soft_results:
        dim = crit_ids.get(r.criterion_id, "unknown")
        by_dim.setdefault(dim, []).append(float(r.score_value or 0.0))

    out: List[DimensionScore] = []
    for dim_id, scores in sorted(by_dim.items()):
        earned = sum(scores)
        mx = float(len(scores))
        norm = earned / mx if mx else 1.0
        out.append(
            DimensionScore(
                dimension_id=dim_id,
                earned_score=earned,
                max_score=mx,
                normalized_score=norm,
                criterion_count=len(scores),
            )
        )
    return out


def aggregate_overall_decision(
    case_rubric: CaseRubric,
    hard_results: List[CriterionResult],
    soft_results: List[CriterionResult],
) -> str:
    """Combine per-criterion outcomes into sft_include / repair / do_not_train (starter policy)."""
    if any(r.verdict == "UNMET" for r in hard_results):
        return "do_not_train"
    if any(r.verdict == "CANNOT_ASSESS" for r in hard_results):
        return "repair"
    soft_scores: List[float] = []
    for r in soft_results:
        if r.score_value is not None:
            soft_scores.append(float(r.score_value))
        elif r.verdict == "MET":
            soft_scores.append(1.0)
        elif r.verdict == "UNMET":
            soft_scores.append(0.0)
        else:
            soft_scores.append(0.5)
    soft_mean = sum(soft_scores) / len(soft_scores) if soft_scores else 1.0
    min_soft = case_rubric.aggregation.minimum_soft_score
    if soft_mean < min_soft:
        return "repair"
    return "sft_include"


def evaluate_note_against_rubric(
    *,
    candidate_id: str,
    note_text: str,
    dialogue: str,
    case_rubric: CaseRubric,
    evaluation_suffix: str = "eval_v0_2",
) -> CaseEvaluationRecord:
    hard_results = [_evaluate_one(c, note_text, dialogue) for c in case_rubric.hard_gates]
    soft_results = [_evaluate_one(c, note_text, dialogue) for c in case_rubric.soft_checks]

    dim_scores = dimension_scores_from_results(case_rubric, hard_results, soft_results)
    overall = aggregate_overall_decision(case_rubric, hard_results, soft_results)

    return CaseEvaluationRecord(
        evaluation_id=f"{case_rubric.example_id}_{candidate_id}_{evaluation_suffix}",
        rubric_id=case_rubric.rubric_id,
        example_id=case_rubric.example_id,
        candidate_id=candidate_id,
        note_family_id=case_rubric.note_family_id,
        rubric_version=case_rubric.version,
        hard_gate_results=hard_results,
        soft_results=soft_results,
        dimension_scores=dim_scores,
        overall_decision=overall,
        judge_metadata={
            "mode": "heuristic_per_criterion",
            "prompt_version": "none",
            "scaffold": "compiled_rubric_gen.v0_2",
            "eval_kinds": [_resolve_eval_kind(c) for c in case_rubric.hard_gates + case_rubric.soft_checks],
        },
        task_profile_id=case_rubric.task_profile_id,
        task_family_id=case_rubric.task_family_id,
        artifact_label=case_rubric.artifact_label,
    )


def evaluate_artifact_against_rubric(
    *,
    candidate_id: str,
    artifact_text: str,
    task_context: str,
    case_rubric: CaseRubric,
    evaluation_suffix: str = "eval_v0_2",
) -> CaseEvaluationRecord:
    return evaluate_note_against_rubric(
        candidate_id=candidate_id,
        note_text=artifact_text,
        dialogue=task_context,
        case_rubric=case_rubric,
        evaluation_suffix=evaluation_suffix,
    )
