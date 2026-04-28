from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rubric_gen.config import PipelineConfig
from rubric_gen.llm_client import LLMRouter, parse_yes_no
from rubric_gen.rrd.prompts import (
    CONFLICT_SYSTEM,
    DECOMPOSITION_SYSTEM,
    INITIAL_RUBRIC_SYSTEM,
    INITIAL_RUBRIC_SYSTEM_PROMPT_ONLY,
    OVERLAP_SYSTEM,
    SATISFACTION_SYSTEM,
    SATISFACTION_SYSTEM_RESPONSE_ONLY,
    render_conflict_prompt,
    render_decomposition_prompt,
    render_initial_rubric_prompt,
    render_initial_rubric_prompt_without_responses,
    render_overlap_prompt,
    render_satisfaction_prompt,
    render_satisfaction_prompt_response_only,
)
from rubric_gen.rrd.weighting import (
    compute_uniform_weights,
    compute_whitened_uniform_weights,
    score_candidates,
)
from rubric_gen.storage import JsonlCache, make_cache_key
from rubric_gen.types import CandidateNote, ExampleRecord, RubricCriterion, RubricEvaluation


DEFAULT_HEURISTIC_RUBRICS = [
    "The note states the chief complaint or visit reason supported by the transcript.",
    "The note captures the clinically important symptoms and relevant negatives discussed in the transcript.",
    "The note records medications and dose or frequency details only when they are supported by the transcript.",
    "The note separates patient-reported history from clinician exam findings or observed results.",
    "The note includes the clinician assessment or main diagnoses only when they are supported by the transcript.",
    "The note includes follow-up, orders, referrals, or counseling that were explicitly discussed.",
    "The note avoids unsupported facts, diagnoses, medications, and test results.",
    "The note uses a clear clinical structure with sectioned or problem-oriented organization.",
]


def normalize_rubric_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_rubrics(raw_text: str) -> List[str]:
    matches = re.findall(r"<RUBRIC>\s*(.*?)\s*</RUBRIC>", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return [normalize_rubric_text(match) for match in matches if normalize_rubric_text(match)]

    lines = []
    for line in raw_text.splitlines():
        candidate = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if len(candidate) >= 12:
            lines.append(normalize_rubric_text(candidate))
    return lines


def _simple_overlap(existing_rubrics: Sequence[str], new_rubric: str) -> bool:
    normalized_new = set(normalize_rubric_text(new_rubric).lower().split())
    for existing in existing_rubrics:
        normalized_existing = set(normalize_rubric_text(existing).lower().split())
        if normalize_rubric_text(existing).lower() == normalize_rubric_text(new_rubric).lower():
            return True
        if not normalized_new or not normalized_existing:
            continue
        overlap = len(normalized_new & normalized_existing) / max(1, len(normalized_new | normalized_existing))
        if overlap >= 0.7:
            return True
    return False


_POSITIVE_PREFIXES = (
    "note includes",
    "the note includes",
    "note documents",
    "the note documents",
    "note specifies",
    "the note specifies",
    "note contains",
    "the note contains",
    "note records",
    "the note records",
    "note preserves",
    "the note preserves",
    "note states",
    "the note states",
    "note orders",
    "the note orders",
    "note addresses",
    "the note addresses",
)

_NEGATIVE_PREFIXES = (
    "note avoids",
    "the note avoids",
    "note does not",
    "the note does not",
    "note should not",
    "the note should not",
    "note must not",
    "the note must not",
    "note excludes",
    "the note excludes",
)

_RUBRIC_STOPWORDS = {
    "the",
    "note",
    "patient",
    "and",
    "with",
    "that",
    "from",
    "this",
    "into",
    "only",
    "when",
    "they",
    "were",
    "their",
    "does",
    "doesnt",
    "should",
    "must",
    "have",
    "has",
    "been",
    "being",
    "part",
    "plan",
    "plans",
    "consistent",
    "transcript",
    "discussion",
    "described",
    "documented",
}


def _requirement_polarity(text: str) -> str:
    lowered = normalize_rubric_text(text).lower()
    for prefix in _NEGATIVE_PREFIXES:
        if lowered.startswith(prefix):
            return "negative"
    for prefix in _POSITIVE_PREFIXES:
        if lowered.startswith(prefix):
            return "positive"
    return "neutral"


def _content_tokens(text: str) -> set[str]:
    lowered = normalize_rubric_text(text).lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return {
        token
        for token in lowered.split()
        if len(token) > 2 and token not in _RUBRIC_STOPWORDS
    }


def _simple_conflict(existing_rubrics: Sequence[str], new_rubric: str) -> bool:
    polarity_new = _requirement_polarity(new_rubric)
    tokens_new = _content_tokens(new_rubric)
    if polarity_new == "neutral" or not tokens_new:
        return False
    for existing in existing_rubrics:
        polarity_existing = _requirement_polarity(existing)
        if polarity_existing == "neutral" or polarity_existing == polarity_new:
            continue
        tokens_existing = _content_tokens(existing)
        if not tokens_existing:
            continue
        overlap = len(tokens_new & tokens_existing) / max(1, min(len(tokens_new), len(tokens_existing)))
        if overlap >= 0.6:
            return True
    return False


def _heuristic_decomposition(rubric_text: str) -> List[str]:
    lowered = rubric_text.lower()
    if "symptoms" in lowered or "negatives" in lowered:
        return [
            "The note captures the main presenting symptoms or complaints that were discussed.",
            "The note preserves clinically relevant negatives or denials that matter for the assessment.",
        ]
    if "medications" in lowered or "dose" in lowered:
        return [
            "The note names the medications that were discussed in the transcript.",
            "The note preserves dose, frequency, or route details when they were stated in the transcript.",
        ]
    if "assessment" in lowered or "diagnoses" in lowered:
        return [
            "The note includes the clinician assessment or diagnoses that were explicitly discussed.",
            "The note links the assessment or diagnoses to supporting findings from the transcript.",
        ]
    if "follow-up" in lowered or "orders" in lowered or "referrals" in lowered or "counseling" in lowered:
        return [
            "The note records follow-up actions, referrals, or ordered next steps that were discussed.",
            "The note preserves counseling or adherence guidance that was explicitly discussed.",
        ]
    if "unsupported" in lowered or "avoid" in lowered:
        return [
            "The note does not introduce unsupported diagnoses, medications, or procedures.",
            "The note does not introduce unsupported results, follow-up instructions, or safety claims.",
        ]
    if "structure" in lowered or "organization" in lowered:
        return [
            "The note uses clear section headers or problem-oriented organization.",
            "The note places findings and plans in appropriate clinical sections.",
        ]
    return [
        "The note captures the key patient history that matters for this encounter.",
        "The note captures the clinician assessment or plan that matters for this encounter.",
    ]


def _heuristic_satisfaction(candidate: CandidateNote, rubric_text: str) -> bool:
    lowered_rubric = rubric_text.lower()
    lowered_note = candidate.text.lower()
    if "chief complaint" in lowered_rubric or "visit reason" in lowered_rubric:
        return "chief complaint" in lowered_note or "history of present illness" in lowered_note or "cc:" in lowered_note
    if "medication" in lowered_rubric:
        return bool(re.search(r"\bmg\b|medication|aspirin|lisinopril|atorvastatin|metformin", lowered_note))
    if "follow-up" in lowered_rubric or "referral" in lowered_rubric or "next step" in lowered_rubric:
        return "follow up" in lowered_note or "follow-up" in lowered_note or "return" in lowered_note or "referral" in lowered_note
    if "unsupported" in lowered_rubric:
        return candidate.quality_bucket not in {"synthetically_degraded", "note_truncated"}
    if "structure" in lowered_rubric or "section" in lowered_rubric:
        return any(
            marker in lowered_note
            for marker in ["assessment", "plan", "history", "review of systems", "objective", "subjective", "chief complaint"]
        )
    return candidate.quality_bucket in {"gold_like", "strong_anchor", "frontier_generated"}


def _candidate_set_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _binary_discrimination_score(coverage_count: int, total_candidates: int) -> float:
    if total_candidates <= 0:
        return 0.0
    probability = coverage_count / total_candidates
    return probability * (1.0 - probability)


class RRDEngine:
    def __init__(
        self,
        config: PipelineConfig,
        router: Optional[LLMRouter],
        proposal_cache: JsonlCache,
        filter_cache: JsonlCache,
        satisfaction_cache: JsonlCache,
    ):
        self.config = config
        self.router = router
        self.proposal_cache = proposal_cache
        self.filter_cache = filter_cache
        self.satisfaction_cache = satisfaction_cache

    def propose_initial_rubrics(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        include_responses: bool = True,
    ) -> List[str]:
        if self.config.dry_run or self.router is None or self.config.rubric_proposer is None:
            return DEFAULT_HEURISTIC_RUBRICS[: self.config.max_initial_rubrics]

        prompt = (
            render_initial_rubric_prompt(example, candidates, self.config.max_initial_rubrics)
            if include_responses
            else render_initial_rubric_prompt_without_responses(example, self.config.max_initial_rubrics)
        )
        cache_key = make_cache_key(
            "initial_rubrics" if include_responses else "initial_rubrics_prompt_only",
            {
                "example_id": example.example_id,
                "prompt": prompt,
                "model": self.config.rubric_proposer.model,
            },
        )
        cached = self.proposal_cache.get(cache_key)
        if cached is None:
            try:
                response = self.router.generate(
                    self.config.rubric_proposer,
                    system_prompt=INITIAL_RUBRIC_SYSTEM if include_responses else INITIAL_RUBRIC_SYSTEM_PROMPT_ONLY,
                    user_prompt=prompt,
                )
                cached = self.proposal_cache.set(
                    cache_key,
                    {
                        "raw_text": response.raw_text,
                        "rubrics": extract_rubrics(response.text),
                    },
                )
            except Exception:
                return DEFAULT_HEURISTIC_RUBRICS[: self.config.max_initial_rubrics]
        rubrics = [normalize_rubric_text(text) for text in cached.get("rubrics", [])]
        return [rubric for rubric in rubrics if rubric][: self.config.max_initial_rubrics]

    def decompose_rubric(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        rubric_text: str,
        other_rubrics: Sequence[str],
    ) -> List[str]:
        if self.config.dry_run or self.router is None or self.config.rubric_proposer is None:
            return _heuristic_decomposition(rubric_text)

        prompt = render_decomposition_prompt(example, candidates, rubric_text, other_rubrics)
        cache_key = make_cache_key(
            "decompose_rubric",
            {
                "example_id": example.example_id,
                "rubric": rubric_text,
                "other_rubrics": list(other_rubrics),
                "prompt": prompt,
                "model": self.config.rubric_proposer.model,
            },
        )
        cached = self.proposal_cache.get(cache_key)
        if cached is None:
            try:
                response = self.router.generate(
                    self.config.rubric_proposer,
                    system_prompt=DECOMPOSITION_SYSTEM,
                    user_prompt=prompt,
                )
                cached = self.proposal_cache.set(
                    cache_key,
                    {
                        "raw_text": response.raw_text,
                        "rubrics": extract_rubrics(response.text),
                    },
                )
            except Exception:
                return _heuristic_decomposition(rubric_text)
        rubrics = [normalize_rubric_text(text) for text in cached.get("rubrics", []) if normalize_rubric_text(text)]
        return rubrics[:2]

    def _run_filter(self, stage: str, system_prompt: str, user_prompt: str) -> bool:
        if self.config.dry_run or self.router is None or self.config.rubric_judge is None:
            return False
        cache_key = make_cache_key(
            stage,
            {
                "user_prompt": user_prompt,
                "model": self.config.rubric_judge.model,
            },
        )
        cached = self.filter_cache.get(cache_key)
        if cached is None:
            try:
                response = self.router.generate(
                    self.config.rubric_judge,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                verdict, _ = parse_yes_no(response.text)
                cached = self.filter_cache.set(
                    cache_key,
                    {
                        "raw_text": response.raw_text,
                        "verdict": bool(verdict),
                    },
                )
            except Exception:
                return False
        return bool(cached.get("verdict"))

    def overlaps_existing(self, existing_rubrics: Sequence[str], new_rubric: str) -> bool:
        if not existing_rubrics:
            return False
        if _simple_overlap(existing_rubrics, new_rubric):
            return True
        return self._run_filter(
            stage="overlap_filter",
            system_prompt=OVERLAP_SYSTEM,
            user_prompt=render_overlap_prompt(existing_rubrics, new_rubric),
        )

    def conflicts_existing(self, existing_rubrics: Sequence[str], new_rubric: str) -> bool:
        if not existing_rubrics:
            return False
        if _simple_conflict(existing_rubrics, new_rubric):
            return True
        return self._run_filter(
            stage="conflict_filter",
            system_prompt=CONFLICT_SYSTEM,
            user_prompt=render_conflict_prompt(existing_rubrics, new_rubric),
        )

    def evaluate_rubric_on_candidate(
        self,
        example: ExampleRecord,
        candidate: CandidateNote,
        rubric: RubricCriterion,
    ) -> RubricEvaluation:
        cache_key = make_cache_key(
            "rubric_satisfaction",
            {
                "example_id": example.example_id,
                "candidate_id": candidate.candidate_id,
                "rubric": rubric.text,
                "model": self.config.rubric_judge.model if self.config.rubric_judge else "heuristic",
            },
        )
        cached = self.satisfaction_cache.get(cache_key)
        if cached is not None:
            return RubricEvaluation(**cached["evaluation"])

        if self.config.dry_run or self.router is None or self.config.rubric_judge is None:
            satisfied = _heuristic_satisfaction(candidate, rubric.text)
            evaluation = RubricEvaluation(
                rubric_id=rubric.rubric_id,
                candidate_id=candidate.candidate_id,
                satisfied=satisfied,
                reasoning="heuristic fallback",
                raw_response="heuristic fallback",
            )
        else:
            try:
                response = self.router.generate(
                    self.config.rubric_judge,
                    system_prompt=(
                        SATISFACTION_SYSTEM_RESPONSE_ONLY
                        if self.config.paper_response_only_judging
                        else SATISFACTION_SYSTEM
                    ),
                    user_prompt=(
                        render_satisfaction_prompt_response_only(candidate, rubric.text)
                        if self.config.paper_response_only_judging
                        else render_satisfaction_prompt(example, candidate, rubric.text)
                    ),
                )
                verdict, reasoning = parse_yes_no(response.text)
                evaluation = RubricEvaluation(
                    rubric_id=rubric.rubric_id,
                    candidate_id=candidate.candidate_id,
                    satisfied=bool(verdict),
                    reasoning=reasoning,
                    raw_response=response.raw_text,
                )
            except Exception:
                evaluation = RubricEvaluation(
                    rubric_id=rubric.rubric_id,
                    candidate_id=candidate.candidate_id,
                    satisfied=_heuristic_satisfaction(candidate, rubric.text),
                    reasoning="heuristic fallback after provider failure",
                    raw_response="heuristic fallback after provider failure",
                )

        self.satisfaction_cache.set(cache_key, {"evaluation": asdict(evaluation)})
        return evaluation

    def evaluate_rubric_set(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        rubrics: Sequence[RubricCriterion],
    ) -> List[RubricEvaluation]:
        evaluations: List[RubricEvaluation] = []
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            for rubric in rubrics:
                for candidate in candidates:
                    futures.append(
                        pool.submit(self.evaluate_rubric_on_candidate, example, candidate, rubric)
                    )
            for future in as_completed(futures):
                evaluations.append(future.result())
        return evaluations

    def _coverage_count(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        rubric: RubricCriterion,
    ) -> int:
        evaluations = self.evaluate_rubric_set(example, candidates, [rubric])
        return sum(1 for evaluation in evaluations if evaluation.satisfied)

    def _satisfied_candidate_ids(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        rubrics: Sequence[RubricCriterion],
    ) -> Dict[str, set[str]]:
        evaluations = self.evaluate_rubric_set(example, candidates, rubrics)
        satisfied: Dict[str, set[str]] = {rubric.rubric_id: set() for rubric in rubrics}
        for evaluation in evaluations:
            if evaluation.satisfied:
                satisfied.setdefault(evaluation.rubric_id, set()).add(evaluation.candidate_id)
        return satisfied

    def _misaligned(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        rubric: RubricCriterion,
    ) -> Tuple[bool, Dict[str, float]]:
        strong = [
            candidate
            for candidate in candidates
            if candidate.quality_bucket in {"gold_like", "strong_anchor"}
        ]
        weak = [
            candidate
            for candidate in candidates
            if candidate.quality_bucket == "synthetically_degraded"
        ]
        if not strong or not weak:
            return False, {"strong_mean": 0.0, "weak_mean": 0.0}

        evaluations = self.evaluate_rubric_set(example, list(strong) + list(weak), [rubric])
        by_candidate = {evaluation.candidate_id: float(evaluation.satisfied) for evaluation in evaluations}
        strong_mean = sum(by_candidate.get(candidate.candidate_id, 0.0) for candidate in strong) / len(strong)
        weak_mean = sum(by_candidate.get(candidate.candidate_id, 0.0) for candidate in weak) / len(weak)
        return weak_mean > (strong_mean + 0.05), {
            "strong_mean": strong_mean,
            "weak_mean": weak_mean,
        }

    def _accept_candidate_rubric(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        existing_active_rubrics: Sequence[RubricCriterion],
        rubric_text: str,
        source_stage: str,
        depth: int,
        round_index: int,
        parent_id: Optional[str] = None,
    ) -> Tuple[Optional[RubricCriterion], Optional[str]]:
        normalized_text = normalize_rubric_text(rubric_text)
        existing_texts = [rubric.text for rubric in existing_active_rubrics]
        if not normalized_text:
            return None, "empty"
        if normalize_rubric_text(normalized_text).lower() in {
            normalize_rubric_text(existing).lower() for existing in existing_texts
        }:
            return None, "duplicate"
        if self.overlaps_existing(existing_texts, normalized_text):
            return None, "overlap"
        if self.conflicts_existing(existing_texts, normalized_text):
            return None, "conflict"

        rubric = RubricCriterion(
            rubric_id=f"{example.example_id}__rubric_{round_index}_{len(existing_active_rubrics)}_{depth}",
            text=normalized_text,
            source_stage=source_stage,
            depth=depth,
            round_index=round_index,
            parent_id=parent_id,
        )
        misaligned, stats = self._misaligned(example, candidates, rubric)
        rubric.metadata["misalignment_stats"] = stats
        if misaligned:
            rubric.accepted = False
            rubric.rejection_reason = "misaligned"
            return None, "misaligned"
        return rubric, None

    def score_rubric_set(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        rubrics: Sequence[RubricCriterion],
        method_prefix: str,
        evaluation_candidates: Sequence[CandidateNote] | None = None,
    ) -> Dict[str, object]:
        rubric_list = list(rubrics)
        scoring_candidates = list(evaluation_candidates or candidates)
        evaluations = self.evaluate_rubric_set(example, scoring_candidates, rubric_list)
        uniform_weights = compute_uniform_weights(rubric_list)
        wu_weights, wu_debug = compute_whitened_uniform_weights(
            rubric_list,
            scoring_candidates,
            evaluations,
            ridge=self.config.covariance_ridge,
        )
        return {
            "rubrics": [asdict(rubric) for rubric in rubric_list],
            "evaluations": [asdict(evaluation) for evaluation in evaluations],
            "uniform": {
                "weights": uniform_weights,
                "ranking": [
                    asdict(score)
                    for score in score_candidates(
                        f"{method_prefix}_uniform",
                        rubric_list,
                        scoring_candidates,
                        evaluations,
                        uniform_weights,
                    )
                ],
            },
            "whitened_uniform": {
                "weights": wu_weights,
                "debug": wu_debug,
                "ranking": [
                    asdict(score)
                    for score in score_candidates(
                        f"{method_prefix}_whitened_uniform",
                        rubric_list,
                        scoring_candidates,
                        evaluations,
                        wu_weights,
                    )
                ],
            },
        }

    def _evaluate_decomposition_children(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        parent: RubricCriterion,
        children: Sequence[RubricCriterion],
    ) -> Tuple[bool, bool, Dict[str, object]]:
        if len(children) < 2:
            return (
                False,
                False,
                {
                    "reason": "needs_two_children",
                    "accepted_children": len(children),
                },
            )

        satisfied = self._satisfied_candidate_ids(example, candidates, [parent, *children])
        parent_ids = satisfied.get(parent.rubric_id, set())
        child_sets = [satisfied.get(child.rubric_id, set()) for child in children]
        child_union = set().union(*child_sets)
        total_candidates = len(candidates)

        for child, child_ids in zip(children, child_sets):
            child.coverage_count = len(child_ids)

        parent_coverage = len(parent_ids)
        parent_discrimination = _binary_discrimination_score(parent_coverage, total_candidates)
        child_coverages = [len(child_ids) for child_ids in child_sets]
        child_discriminations = [
            _binary_discrimination_score(child_coverage, total_candidates)
            for child_coverage in child_coverages
        ]
        coverage_recall = (
            len(parent_ids & child_union) / max(1, parent_coverage)
            if parent_coverage
            else 0.0
        )
        extra_ratio = (
            len(child_union - parent_ids) / max(1, len(child_union))
            if child_union
            else 0.0
        )
        pair_overlap = _candidate_set_overlap(child_sets[0], child_sets[1])
        children_narrower = (
            bool(parent_ids)
            and all(child_coverage < parent_coverage for child_coverage in child_coverages)
        )
        children_informative = all(0 < child_coverage < total_candidates for child_coverage in child_coverages)
        discrimination_gain = max(child_discriminations, default=0.0) - parent_discrimination

        retain_children = (
            children_informative
            and children_narrower
            and coverage_recall >= self.config.decomposition_min_recall
            and extra_ratio <= self.config.decomposition_max_extra_ratio
            and pair_overlap <= self.config.decomposition_max_pair_overlap
            and discrimination_gain >= self.config.decomposition_min_discrimination_gain
        )
        supersede_parent = retain_children
        return (
            retain_children,
            supersede_parent,
            {
                "reason": "retain_children" if retain_children else "reject_children",
                "parent_coverage": parent_coverage,
                "child_coverages": child_coverages,
                "child_union_coverage": len(child_union),
                "accepted_children": len(children),
                "coverage_recall": coverage_recall,
                "extra_ratio": extra_ratio,
                "pair_overlap": pair_overlap,
                "parent_discrimination": parent_discrimination,
                "child_discriminations": child_discriminations,
                "discrimination_gain": discrimination_gain,
                "children_narrower": children_narrower,
                "children_informative": children_informative,
                "retain_children": retain_children,
                "supersede_parent": supersede_parent,
            },
        )

    def run_rrd(
        self,
        example: ExampleRecord,
        candidates: Sequence[CandidateNote],
        evaluation_candidates: Sequence[CandidateNote] | None = None,
    ) -> Dict[str, object]:
        active_rubrics: List[RubricCriterion] = []
        rejected: List[Dict[str, str]] = []
        superseded: List[str] = []
        queue: List[RubricCriterion] = []
        round_index = 0

        for rubric_text in self.propose_initial_rubrics(example, candidates, include_responses=True):
            rubric, reason = self._accept_candidate_rubric(
                example=example,
                candidates=candidates,
                existing_active_rubrics=active_rubrics,
                rubric_text=rubric_text,
                source_stage="initial",
                depth=0,
                round_index=round_index,
            )
            round_index += 1
            if rubric is None:
                rejected.append({"rubric": rubric_text, "reason": reason or "rejected"})
                continue
            active_rubrics.append(rubric)
            queue.append(rubric)

        final_rubrics: List[RubricCriterion] = []
        rejected_count = len(rejected)

        while queue and rejected_count <= self.config.termination_rejections:
            current = queue.pop(0)
            current.coverage_count = self._coverage_count(example, candidates, current)

            if (
                current.coverage_count > self.config.decomposition_threshold
                and current.depth < self.config.max_decomposition_depth
                and len(active_rubrics) < self.config.max_final_rubrics
            ):
                other_rubrics = [rubric.text for rubric in active_rubrics if rubric.rubric_id != current.rubric_id]
                proposed_children = self.decompose_rubric(example, candidates, current.text, other_rubrics)
                accepted_children: List[RubricCriterion] = []

                for child_text in proposed_children:
                    child, reason = self._accept_candidate_rubric(
                        example=example,
                        candidates=candidates,
                        existing_active_rubrics=[
                            rubric
                            for rubric in active_rubrics
                            if rubric.rubric_id != current.rubric_id
                        ]
                        + accepted_children,
                        rubric_text=child_text,
                        source_stage="decomposition",
                        depth=current.depth + 1,
                        round_index=round_index,
                        parent_id=current.rubric_id,
                    )
                    round_index += 1
                    if child is None:
                        rejected.append({"rubric": child_text, "reason": reason or "rejected"})
                        rejected_count += 1
                        continue
                    accepted_children.append(child)

                if accepted_children:
                    retain_children, should_supersede, supersede_stats = self._evaluate_decomposition_children(
                        example,
                        candidates,
                        current,
                        accepted_children,
                    )
                    current.metadata["children"] = [child.rubric_id for child in accepted_children]
                    current.metadata["decomposition_decision"] = supersede_stats
                    if not retain_children:
                        for child in accepted_children:
                            child.accepted = False
                            child.rejection_reason = "insufficient_decomposition_gain"
                            rejected.append(
                                {
                                    "rubric": child.text,
                                    "reason": "insufficient_decomposition_gain",
                                }
                            )
                            rejected_count += 1
                    else:
                        active_rubrics.extend(accepted_children)
                        queue.extend(accepted_children)
                    if should_supersede:
                        active_rubrics[:] = [
                            rubric for rubric in active_rubrics if rubric.rubric_id != current.rubric_id
                        ]
                        current.accepted = False
                        current.rejection_reason = "superseded_by_decomposition"
                        superseded.append(current.rubric_id)
                        continue

            final_rubrics.append(current)

        final_rubrics = [rubric for rubric in final_rubrics if rubric.rubric_id not in superseded]
        scored = self.score_rubric_set(
            example,
            candidates,
            final_rubrics,
            method_prefix="rrd",
            evaluation_candidates=evaluation_candidates or candidates,
        )
        scored["rrd_artifact"] = {
            "initial_rubric_count": len([rubric for rubric in active_rubrics if rubric.source_stage == "initial"]),
            "final_rubric_count": len(final_rubrics),
            "rejected": rejected,
            "superseded": superseded,
            "termination_rejections": self.config.termination_rejections,
        }
        return scored
