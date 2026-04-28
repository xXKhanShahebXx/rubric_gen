"""
Reasoning process verifier for ``livebench-reasoning`` JudgeBench examples.

The existing :mod:`rubric_gen.compiled.judgebench_verifiers` module is anchored on exact-answer /
option-consistency signals. On ``livebench-reasoning`` it never triggers
(``verifier_trigger_rate_by_family['livebench-reasoning'] == 0.0`` on the best 320 OOF run),
because logic-puzzle solutions rarely surface a single clean reference answer to match.

This module adds a process-level verifier that extracts structured reasoning state from each
response and scores them on:

- **assignment_completeness**: were all slots filled with committed values?
- **internal_consistency**: do committed values satisfy stated exclusivity / parity constraints?
- **conclusion_grounded**: does the final answer follow from the assignment?
- **contradiction_avoidance**: are there lingering "actually wait" reversals?
- **format_ok**: is the final answer in the requested syntax?

The extractor is lightweight: rather than an LLM call per response, we use deterministic regex
parsing of the response text. This keeps the verifier cheap and cache-free while still giving a
signal on the dominant failure clusters (``person_right_*`` puzzles, XOR clue reasoning, logic
state tracking). A caller can plug in an LLM-backed extractor via ``ReasoningProcessVerifier``'s
``extractor`` argument for production use; the default is the deterministic parser.

Score blending: the module returns a :class:`ReasoningProcessVerifierOutcome` compatible with
``JudgeBenchVerifierOutcome`` (duck-typed via ``asdict``) so it slots into the existing
``_apply_pair_verifier_result`` override machinery.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


_ENTITY_RE = re.compile(r"\b([A-Z][a-z]{1,15})\b")
_SLOT_RE = re.compile(
    r"(?:position|seat|spot|slot|house|chair|order)\s*[:#]?\s*(\d+)",
    re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z][a-z]{1,15})\s*(?:is|sits|lives|stands|owns|has|prefers|holds)\s+(?:at|in|on)?\s*([A-Za-z0-9_\-\s]{1,30})",
)
_UNKNOWN_MARKERS = (
    "unknown",
    "unclear",
    "cannot determine",
    "cannot be determined",
    "not enough information",
    "insufficient information",
    "?",
)
_REVERSAL_MARKERS = (
    "actually wait",
    "actually, wait",
    "wait, actually",
    "oh wait",
    "let me reconsider",
    "on second thought",
    "i was wrong",
    "scratch that",
    "on reflection",
)
_FINAL_LINE_MARKERS = (
    "final answer",
    "answer:",
    "conclusion:",
    "therefore,",
    "so the answer",
)


@dataclass(frozen=True)
class ReasoningStateExtract:
    response_label: str
    extracted_text: str
    entities: Tuple[str, ...]
    slot_mentions: Tuple[str, ...]
    assignments: Tuple[Tuple[str, str], ...]
    unknown_markers: int
    reversal_markers: int
    final_answer_line: str
    format_ok: bool
    raw_length: int


@dataclass(frozen=True)
class ReasoningProcessVerifierCandidateScore:
    response_label: str
    assignment_completeness: float
    internal_consistency: float
    conclusion_grounded: float
    contradiction_avoidance: float
    format_ok: float
    composite: float
    extract: ReasoningStateExtract


@dataclass(frozen=True)
class ReasoningProcessVerifierOutcome:
    source_family: str
    triggered: bool
    recommended_decision: str
    confidence: str
    reason: str
    margin: float
    composite_a: float
    composite_b: float
    candidate_scores: Dict[str, Dict[str, Any]]
    extractor_kind: str = "deterministic_v1"
    verifier_schema: str = "reasoning_process_verifier_v1"


@dataclass
class ReasoningProcessVerifierConfig:
    trigger_margin: float = 0.04
    min_assignment_coverage: float = 0.25
    medium_confidence_margin: float = 0.06
    high_confidence_margin: float = 0.15
    format_weight: float = 0.1
    completeness_weight: float = 0.3
    consistency_weight: float = 0.25
    conclusion_weight: float = 0.25
    contradiction_weight: float = 0.1


ExtractorFn = Callable[[str, str, Mapping[str, Any]], ReasoningStateExtract]


def _final_answer_line(text: str) -> Tuple[str, bool]:
    if not text:
        return "", False
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in _FINAL_LINE_MARKERS):
            return line, True
    return lines[-1] if lines else "", False


def _unique_tuple(seq: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def deterministic_extractor(
    response_label: str,
    response_text: str,
    prompt_features: Mapping[str, Any],
) -> ReasoningStateExtract:
    text = response_text or ""
    lowered = text.lower()
    entities = _unique_tuple(_ENTITY_RE.findall(text))
    slot_mentions = _unique_tuple(_SLOT_RE.findall(text))
    assignments_raw = _ASSIGNMENT_RE.findall(text)
    assignments: List[Tuple[str, str]] = []
    for ent, slot in assignments_raw:
        slot_clean = re.sub(r"\s+", " ", slot or "").strip()
        slot_clean = slot_clean.rstrip(".,;:")
        if not slot_clean:
            continue
        assignments.append((ent, slot_clean))
    unknown_markers = sum(1 for marker in _UNKNOWN_MARKERS if marker in lowered)
    reversal_markers = sum(1 for marker in _REVERSAL_MARKERS if marker in lowered)
    final_line, has_final_marker = _final_answer_line(text)
    requested_answer_mode = str(prompt_features.get("requested_answer_mode", "") or "")
    format_ok = has_final_marker
    if requested_answer_mode and requested_answer_mode == "single_word":
        format_ok = format_ok or bool(re.match(r"^[A-Za-z][A-Za-z0-9\-_]*$", final_line.strip().rstrip(".")))
    return ReasoningStateExtract(
        response_label=response_label,
        extracted_text=text,
        entities=entities,
        slot_mentions=slot_mentions,
        assignments=tuple(assignments),
        unknown_markers=int(unknown_markers),
        reversal_markers=int(reversal_markers),
        final_answer_line=final_line,
        format_ok=bool(format_ok),
        raw_length=len(text),
    )


def _score_assignment_completeness(extract: ReasoningStateExtract) -> float:
    if extract.unknown_markers > 0:
        penalty = min(0.5, 0.2 * extract.unknown_markers)
        base = 0.6 - penalty
        return max(0.0, base)
    total_entities = len(extract.entities) or 0
    if total_entities == 0:
        if extract.assignments:
            return 0.6
        return 0.3
    assigned_entities = {ent for ent, _ in extract.assignments}
    assigned = len(assigned_entities & set(extract.entities))
    coverage = assigned / max(1, total_entities)
    return min(1.0, 0.3 + 0.7 * coverage)


def _score_internal_consistency(extract: ReasoningStateExtract) -> float:
    if not extract.assignments:
        return 0.5
    slot_to_entities: Dict[str, List[str]] = {}
    entity_to_slots: Dict[str, List[str]] = {}
    for ent, slot in extract.assignments:
        slot_norm = slot.lower()
        slot_to_entities.setdefault(slot_norm, []).append(ent)
        entity_to_slots.setdefault(ent, []).append(slot_norm)
    conflict_penalty = 0.0
    for entities in slot_to_entities.values():
        if len(set(entities)) > 1:
            conflict_penalty += 0.25
    for slots in entity_to_slots.values():
        if len(set(slots)) > 1:
            conflict_penalty += 0.15
    score = 1.0 - min(1.0, conflict_penalty)
    return max(0.0, score)


def _score_conclusion_grounded(extract: ReasoningStateExtract) -> float:
    final = extract.final_answer_line.lower()
    if not final:
        return 0.3
    if any(marker in final for marker in _UNKNOWN_MARKERS):
        return 0.1
    if not extract.assignments:
        return 0.55
    mentioned = any(ent.lower() in final for ent, _ in extract.assignments)
    return 0.9 if mentioned else 0.5


def _score_contradiction_avoidance(extract: ReasoningStateExtract) -> float:
    if extract.reversal_markers == 0:
        return 1.0
    if extract.reversal_markers == 1:
        return 0.75
    if extract.reversal_markers == 2:
        return 0.5
    return max(0.1, 0.5 - 0.1 * (extract.reversal_markers - 2))


def _score_format_ok(extract: ReasoningStateExtract) -> float:
    return 1.0 if extract.format_ok else 0.5


def score_response(
    extract: ReasoningStateExtract,
    *,
    config: ReasoningProcessVerifierConfig,
) -> ReasoningProcessVerifierCandidateScore:
    completeness = _score_assignment_completeness(extract)
    consistency = _score_internal_consistency(extract)
    conclusion = _score_conclusion_grounded(extract)
    contradiction = _score_contradiction_avoidance(extract)
    format_ok = _score_format_ok(extract)
    composite = (
        config.completeness_weight * completeness
        + config.consistency_weight * consistency
        + config.conclusion_weight * conclusion
        + config.contradiction_weight * contradiction
        + config.format_weight * format_ok
    )
    return ReasoningProcessVerifierCandidateScore(
        response_label=extract.response_label,
        assignment_completeness=round(completeness, 6),
        internal_consistency=round(consistency, 6),
        conclusion_grounded=round(conclusion, 6),
        contradiction_avoidance=round(contradiction, 6),
        format_ok=round(format_ok, 6),
        composite=round(composite, 6),
        extract=extract,
    )


def _candidate_score_payload(score: ReasoningProcessVerifierCandidateScore) -> Dict[str, Any]:
    extract_payload = asdict(score.extract)
    extract_payload["assignments"] = [list(pair) for pair in score.extract.assignments]
    return {
        "response_label": score.response_label,
        "assignment_completeness": score.assignment_completeness,
        "internal_consistency": score.internal_consistency,
        "conclusion_grounded": score.conclusion_grounded,
        "contradiction_avoidance": score.contradiction_avoidance,
        "format_ok": score.format_ok,
        "composite": score.composite,
        "extract": extract_payload,
    }


class ReasoningProcessVerifier:
    """Pair-level reasoning process verifier."""

    def __init__(
        self,
        *,
        config: Optional[ReasoningProcessVerifierConfig] = None,
        extractor: Optional[ExtractorFn] = None,
    ) -> None:
        self.config = config or ReasoningProcessVerifierConfig()
        self.extractor = extractor or deterministic_extractor

    def applies_to(self, source_family: str) -> bool:
        return source_family == "livebench-reasoning"

    def evaluate(
        self,
        *,
        source_family: str,
        response_a: str,
        response_b: str,
        prompt_features: Optional[Mapping[str, Any]] = None,
    ) -> ReasoningProcessVerifierOutcome:
        features = dict(prompt_features or {})
        extract_a = self.extractor("A", response_a, features)
        extract_b = self.extractor("B", response_b, features)
        score_a = score_response(extract_a, config=self.config)
        score_b = score_response(extract_b, config=self.config)
        margin = round(score_a.composite - score_b.composite, 6)
        abs_margin = abs(margin)
        triggered = abs_margin >= self.config.trigger_margin
        if abs_margin >= self.config.high_confidence_margin:
            confidence = "high"
        elif abs_margin >= self.config.medium_confidence_margin:
            confidence = "medium"
        elif abs_margin > 0:
            confidence = "low"
        else:
            confidence = ""
        recommended_decision = ""
        if triggered and confidence in {"medium", "high"}:
            recommended_decision = "A>B" if margin > 0 else "B>A"
        reason = "reasoning_process_insufficient_margin"
        if triggered and confidence in {"medium", "high"}:
            if score_a.assignment_completeness != score_b.assignment_completeness:
                better_label = "a" if score_a.assignment_completeness > score_b.assignment_completeness else "b"
                reason = f"{better_label}_assignment_completeness"
            elif score_a.internal_consistency != score_b.internal_consistency:
                better_label = "a" if score_a.internal_consistency > score_b.internal_consistency else "b"
                reason = f"{better_label}_internal_consistency"
            elif score_a.conclusion_grounded != score_b.conclusion_grounded:
                better_label = "a" if score_a.conclusion_grounded > score_b.conclusion_grounded else "b"
                reason = f"{better_label}_conclusion_grounded"
            else:
                better_label = "a" if margin > 0 else "b"
                reason = f"{better_label}_reasoning_process_margin"
        candidate_scores = {
            "A": _candidate_score_payload(score_a),
            "B": _candidate_score_payload(score_b),
        }
        return ReasoningProcessVerifierOutcome(
            source_family=source_family,
            triggered=bool(triggered),
            recommended_decision=recommended_decision,
            confidence=confidence,
            reason=reason,
            margin=abs_margin,
            composite_a=score_a.composite,
            composite_b=score_b.composite,
            candidate_scores=candidate_scores,
        )
