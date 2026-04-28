"""
Generic gold-rubric comparison abstractions.

These interfaces let benchmark-specific corpora expose normalized gold criteria and a
provider-backed comparison step without baking that logic into one evaluator module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


@dataclass
class GoldCriterion:
    criterion_id: str
    criterion: str
    points: int
    polarity: str
    family: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldAlignmentArtifact:
    provider_id: str
    gold_rows: List[Dict[str, Any]]
    alignment: Dict[str, Any]
    metrics: Dict[str, Any]
    family_summary: Dict[str, Dict[str, Any]]
    alignment_lookup: Dict[Tuple[str, int], Dict[str, Any]]
    cache_hit: bool = False
    parse_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GoldRubricProvider(Protocol):
    provider_id: str

    def gold_rows_for_example(self, example: Any) -> List[Dict[str, Any]]:
        """Return normalized gold criteria for one example."""

    def compare_generated_rows(
        self,
        *,
        example: Any,
        generated_rows: Sequence[Dict[str, Any]],
        model_spec: Any,
        router: Any,
        cache: Any,
        use_heuristic_only: bool = False,
    ) -> GoldAlignmentArtifact:
        """Compare generated rows against the provider's gold criteria."""


def build_alignment_lookup(alignment: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in alignment.get("expert_matches", []):
        try:
            lookup[("expert", int(row["expert_index"]))] = dict(row)
        except (KeyError, TypeError, ValueError):
            continue
    for row in alignment.get("generated_assessments", []):
        try:
            lookup[("generated", int(row["generated_index"]))] = dict(row)
        except (KeyError, TypeError, ValueError):
            continue
    return lookup


def gold_criteria_as_rows(criteria: Sequence[GoldCriterion]) -> List[Dict[str, Any]]:
    return [
        {
            "criterion_id": row.criterion_id,
            "criterion": row.criterion,
            "points": row.points,
            "polarity": row.polarity,
            "family": row.family,
            "tags": list(row.tags),
            "metadata": dict(row.metadata),
        }
        for row in criteria
    ]


__all__ = [
    "GoldCriterion",
    "GoldAlignmentArtifact",
    "GoldRubricProvider",
    "build_alignment_lookup",
    "gold_criteria_as_rows",
]
