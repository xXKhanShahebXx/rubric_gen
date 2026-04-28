from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from rubric_gen.types import CandidateNote, CandidateScore, RubricCriterion, RubricEvaluation


def compute_uniform_weights(rubrics: Iterable[RubricCriterion]) -> Dict[str, float]:
    rubric_list = list(rubrics)
    if not rubric_list:
        return {}
    weight = 1.0 / len(rubric_list)
    return {rubric.rubric_id: weight for rubric in rubric_list}


def _evaluation_matrix(
    rubrics: List[RubricCriterion],
    candidates: List[CandidateNote],
    evaluations: Iterable[RubricEvaluation],
) -> np.ndarray:
    rubric_index = {rubric.rubric_id: idx for idx, rubric in enumerate(rubrics)}
    candidate_index = {candidate.candidate_id: idx for idx, candidate in enumerate(candidates)}
    matrix = np.zeros((len(candidates), len(rubrics)), dtype=float)
    for evaluation in evaluations:
        row = candidate_index.get(evaluation.candidate_id)
        col = rubric_index.get(evaluation.rubric_id)
        if row is None or col is None:
            continue
        matrix[row, col] = 1.0 if evaluation.satisfied else 0.0
    return matrix


def compute_whitened_uniform_weights(
    rubrics: List[RubricCriterion],
    candidates: List[CandidateNote],
    evaluations: Iterable[RubricEvaluation],
    ridge: float = 1e-3,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    if not rubrics:
        return {}, {"raw_weights": [], "clipped_weights": [], "eigenvalues": []}
    if len(rubrics) == 1:
        return {rubrics[0].rubric_id: 1.0}, {
            "raw_weights": [1.0],
            "clipped_weights": [1.0],
            "eigenvalues": [1.0],
        }

    matrix = _evaluation_matrix(rubrics, candidates, evaluations)
    covariance = np.cov(matrix, rowvar=False, bias=True)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]], dtype=float)
    covariance = covariance + (ridge * np.eye(covariance.shape[0]))

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, ridge, None)
    inv_sqrt = eigenvectors @ np.diag(np.power(eigenvalues, -0.5)) @ eigenvectors.T
    raw_weights = inv_sqrt @ np.ones(len(rubrics))
    clipped = np.clip(raw_weights, 0.0, None)
    if float(clipped.sum()) <= 0.0:
        clipped = np.ones(len(rubrics), dtype=float)
    normalized = clipped / clipped.sum()

    return (
        {rubric.rubric_id: float(normalized[idx]) for idx, rubric in enumerate(rubrics)},
        {
            "raw_weights": raw_weights.tolist(),
            "clipped_weights": normalized.tolist(),
            "eigenvalues": eigenvalues.tolist(),
        },
    )


def score_candidates(
    method: str,
    rubrics: List[RubricCriterion],
    candidates: List[CandidateNote],
    evaluations: Iterable[RubricEvaluation],
    weights: Dict[str, float],
) -> List[CandidateScore]:
    candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
    satisfied_counts = {candidate.candidate_id: 0 for candidate in candidates}
    totals = {candidate.candidate_id: 0.0 for candidate in candidates}

    for evaluation in evaluations:
        if evaluation.rubric_id not in weights:
            continue
        if evaluation.satisfied:
            satisfied_counts[evaluation.candidate_id] += 1
            totals[evaluation.candidate_id] += weights[evaluation.rubric_id]

    ranked_ids = sorted(
        totals,
        key=lambda candidate_id: (
            -totals[candidate_id],
            -satisfied_counts[candidate_id],
            candidate_id,
        ),
    )

    ranked: List[CandidateScore] = []
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        candidate = candidate_lookup[candidate_id]
        ranked.append(
            CandidateScore(
                candidate_id=candidate_id,
                method=method,
                score=float(totals[candidate_id]),
                rank=rank,
                satisfied_count=satisfied_counts[candidate_id],
                rubric_count=len(rubrics),
                quality_bucket=candidate.quality_bucket,
                source_label=candidate.source_label,
            )
        )
    return ranked
