"""
RRD-style misalignment and redundancy filters for discovered rubric proposals.

These are the two filters from *Rethinking Rubric Generation for Improving LLM Judge and Reward
Modeling for Open-ended Tasks* (Park et al., 2602.05125v1) that drove most of the reported +17.7pt
JudgeBench-GPT-4o gain. They were not previously applied to the canonical proposal pool in this
repo; we now apply them after ``merge_proposal_entries`` and before scoring.

Misalignment filter
-------------------
A proposal is misaligned when the criterion text, read literally, would more plausibly favor the
weaker (rejected) side than the stronger (preferred) side of the discovery pair. In an LLM-
assisted build a GPT-4o call answers this. In the deterministic fallback used here and in tests
we rely on a lexical-overlap heuristic that matches the default builder verifier: a proposal is
aligned when the requirement text overlaps more with the *strong* side than with the *weak* side
of at least one of its originating pairs. Proposals without any lexical overlap with either side
are treated as misaligned (they describe behavior not observed in the pair).

Redundancy filter
-----------------
Near-duplicate requirements inflate whitening covariance estimates and add noise to scoring. The
filter greedily keeps the strongest member of each cluster (``hard_gate`` > ``high`` > ``medium``
> ``low``, tiebreaker count) where two rows share a cluster if the Jaccard similarity of their
requirement/label tokens exceeds ``redundancy_threshold``.

Both filters preserve the canonical-proposal shape so downstream code (``_prepare_rows_for_scoring``,
``canonical_rows_to_rubrics``) continues to work unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[a-z0-9]+")

_SEVERITY_RANK: Mapping[str, int] = {
    "hard_gate": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(str(text).lower())


def _token_set(text: str) -> set:
    return set(_tokens(text))


def _requirement_blob(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("label", "") or ""),
        str(row.get("requirement", "") or ""),
        str(row.get("dimension", "") or ""),
    ]
    return " ".join(p for p in parts if p)


def _severity(row: Mapping[str, Any]) -> int:
    value = str(row.get("severity_tier", "") or "").strip().lower()
    return _SEVERITY_RANK.get(value, 2)


@dataclass(frozen=True)
class PairContext:
    """
    Context for a single discovery pair used to measure misalignment.

    ``strong_text`` is the reference / preferred response; ``weak_text`` is the synthetic or
    alternative response. ``pair_id`` matches the ``pair_ids`` list on canonical proposals so the
    filter can locate the right pair for each row.
    """

    pair_id: str
    strong_text: str
    weak_text: str


@dataclass
class FilterStats:
    seen: int = 0
    dropped_misaligned: int = 0
    dropped_redundant: int = 0
    dropped_no_pair_context: int = 0
    cluster_count: int = 0
    cluster_assignments: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seen": self.seen,
            "dropped_misaligned": self.dropped_misaligned,
            "dropped_redundant": self.dropped_redundant,
            "dropped_no_pair_context": self.dropped_no_pair_context,
            "cluster_count": self.cluster_count,
            "cluster_assignments": dict(self.cluster_assignments),
        }


MisalignmentEvaluator = Callable[[Mapping[str, Any], Sequence[PairContext]], bool]


def default_misalignment_evaluator(
    row: Mapping[str, Any],
    pair_contexts: Sequence[PairContext],
) -> bool:
    """
    Deterministic, network-free misalignment check. Returns ``True`` when the proposal is aligned
    (i.e. should be kept). A proposal is aligned if, on at least one originating pair, the
    requirement text shares more tokens with the strong side than with the weak side.
    """
    if not pair_contexts:
        return True
    req_tokens = _token_set(_requirement_blob(row))
    if not req_tokens:
        return False
    pair_ids = list(row.get("pair_ids") or [])
    relevant = [ctx for ctx in pair_contexts if ctx.pair_id in pair_ids] or list(pair_contexts)
    if not relevant:
        return True
    best_margin = -1
    for ctx in relevant:
        strong_tokens = _token_set(ctx.strong_text)
        weak_tokens = _token_set(ctx.weak_text)
        strong_overlap = len(req_tokens & strong_tokens)
        weak_overlap = len(req_tokens & weak_tokens)
        margin = strong_overlap - weak_overlap
        if margin > best_margin:
            best_margin = margin
        if strong_overlap > 0 and margin >= 0:
            return True
    return False


def apply_misalignment_filter(
    rows: Sequence[Mapping[str, Any]],
    pair_contexts: Sequence[PairContext],
    *,
    evaluator: Optional[MisalignmentEvaluator] = None,
    stats: Optional[FilterStats] = None,
) -> List[Dict[str, Any]]:
    """
    Keep only direction-aligned rows. Mutates ``stats`` if provided.
    """
    evaluator_fn = evaluator or default_misalignment_evaluator
    stats = stats or FilterStats()
    kept: List[Dict[str, Any]] = []
    for row in rows:
        stats.seen += 1
        row_dict = dict(row)
        pair_ids = row_dict.get("pair_ids") or []
        if not pair_ids and not pair_contexts:
            stats.dropped_no_pair_context += 1
        if evaluator_fn(row_dict, pair_contexts):
            kept.append(row_dict)
        else:
            stats.dropped_misaligned += 1
    return kept


def _jaccard(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b) or 1
    return inter / union


def apply_redundancy_filter(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.9,
    stats: Optional[FilterStats] = None,
) -> List[Dict[str, Any]]:
    """
    Greedy redundancy collapse. Preserves the input order among non-redundant rows.
    """
    stats = stats or FilterStats()
    kept: List[Dict[str, Any]] = []
    kept_token_sets: List[set] = []
    cluster_reps: List[str] = []
    for row in rows:
        row_dict = dict(row)
        req_tokens = _token_set(_requirement_blob(row_dict))
        if not req_tokens:
            kept.append(row_dict)
            kept_token_sets.append(req_tokens)
            cluster_reps.append(row_dict.get("merge_key", "") or row_dict.get("label", ""))
            continue
        merged_index = -1
        for idx, kept_tokens in enumerate(kept_token_sets):
            if _jaccard(req_tokens, kept_tokens) >= threshold:
                merged_index = idx
                break
        if merged_index == -1:
            kept.append(row_dict)
            kept_token_sets.append(req_tokens)
            cluster_reps.append(row_dict.get("merge_key", "") or row_dict.get("label", ""))
            continue
        existing = kept[merged_index]
        candidate_sev = _severity(row_dict)
        existing_sev = _severity(existing)
        if candidate_sev < existing_sev:
            row_dict["count"] = int(existing.get("count", 0) or 0) + int(row_dict.get("count", 0) or 0)
            kept[merged_index] = row_dict
            kept_token_sets[merged_index] = req_tokens
            cluster_reps[merged_index] = row_dict.get("merge_key", "") or row_dict.get("label", "")
            stats.dropped_redundant += 1
        elif candidate_sev == existing_sev:
            existing_count = int(existing.get("count", 0) or 0)
            candidate_count = int(row_dict.get("count", 0) or 0)
            if candidate_count > existing_count:
                row_dict["count"] = existing_count + candidate_count
                kept[merged_index] = row_dict
                kept_token_sets[merged_index] = req_tokens
                cluster_reps[merged_index] = row_dict.get("merge_key", "") or row_dict.get("label", "")
            else:
                existing["count"] = existing_count + candidate_count
                kept[merged_index] = existing
            stats.dropped_redundant += 1
        else:
            existing["count"] = int(existing.get("count", 0) or 0) + int(row_dict.get("count", 0) or 0)
            kept[merged_index] = existing
            stats.dropped_redundant += 1
    stats.cluster_count = len(kept)
    stats.cluster_assignments = {
        rep: str(kept[idx].get("merge_key", "") or "")
        for idx, rep in enumerate(cluster_reps)
        if rep
    }
    return kept


def apply_rrd_filters(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_contexts: Sequence[PairContext] = (),
    misalignment_evaluator: Optional[MisalignmentEvaluator] = None,
    redundancy_threshold: float = 0.9,
    enable_misalignment: bool = True,
    enable_redundancy: bool = True,
) -> Tuple[List[Dict[str, Any]], FilterStats]:
    """
    Convenience wrapper that runs misalignment then redundancy and returns the filtered rows plus
    aggregate statistics for artifact writing.
    """
    stats = FilterStats()
    working: List[Dict[str, Any]] = [dict(r) for r in rows]
    if enable_misalignment:
        working = apply_misalignment_filter(
            working,
            pair_contexts,
            evaluator=misalignment_evaluator,
            stats=stats,
        )
    if enable_redundancy:
        working = apply_redundancy_filter(
            working,
            threshold=redundancy_threshold,
            stats=stats,
        )
    return working, stats


def build_pair_contexts(
    pair_payloads: Sequence[Mapping[str, Any]],
) -> List[PairContext]:
    """
    Derive pair contexts from a list of pair payloads written by ``_process_judgebench_example``.
    Each payload has ``pair_id``, ``strong_candidate_id``, ``weak_candidate_id``, and -- via the
    surrounding scoring artifact -- access to candidate text. This helper accepts a flat shape::

        [{"pair_id": "...", "strong_text": "...", "weak_text": "..."}, ...]

    Callers in ``judgebench_eval`` adapt their pair payloads to this shape.
    """
    contexts: List[PairContext] = []
    for payload in pair_payloads or []:
        pair_id = str(payload.get("pair_id", "") or "")
        strong_text = str(payload.get("strong_text", "") or "")
        weak_text = str(payload.get("weak_text", "") or "")
        if not pair_id:
            continue
        contexts.append(PairContext(pair_id=pair_id, strong_text=strong_text, weak_text=weak_text))
    return contexts
