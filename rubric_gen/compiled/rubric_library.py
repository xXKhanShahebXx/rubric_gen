"""
External rubric library for JudgeBench v2.

Stores a frozen set of portable criteria distilled offline from external preference sources
(HelpSteer3, UltraFeedback, PPE, synthetic multi-model pairs). The library is used at inference
time by the ``library_v1`` retrieval profile to seed per-pair discovery with direction-validated
criteria that are not fit to any JudgeBench train-split profile bootstrap.

The library is a JSON file with the following top-level schema::

    {
      "schema": "compiled_judgebench_rubric_library_v1",
      "version": "v1",
      "built_at": "...",
      "build_metadata": {...},
      "criteria": [ RubricLibraryCriterion, ... ]
    }

Each ``RubricLibraryCriterion`` is designed to be mergeable into the existing
``merge_proposal_entries`` canonical-proposal shape so downstream scoring is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


RUBRIC_LIBRARY_SCHEMA = "compiled_judgebench_rubric_library_v1"
RUBRIC_LIBRARY_CRITERION_SCHEMA = "compiled_judgebench_rubric_library_criterion_v1"

_ALLOWED_FAMILY_TAGS: Set[str] = {
    "mmlu-pro",
    "livebench-reasoning",
    "livebench-math",
    "livecodebench",
    "generic",
}

_ALLOWED_SEVERITY_TIERS: Set[str] = {"hard_gate", "high", "medium", "low"}


@dataclass(frozen=True)
class RubricLibraryCriterion:
    criterion_id: str
    dimension: str
    label: str
    requirement: str
    severity_tier: str
    applicable_families: Tuple[str, ...]
    source_tag: str
    focus_kind: str = ""
    verification_notes: str = ""
    direction_evidence: int = 0
    redundancy_cluster_id: str = ""

    def matches_family(self, source_family: str) -> bool:
        if not self.applicable_families:
            return True
        if "generic" in self.applicable_families:
            return True
        return source_family in self.applicable_families

    def to_canonical_row(
        self,
        *,
        example_id: str,
        pair_id: str,
    ) -> Dict[str, Any]:
        """
        Convert a library criterion into a canonical-proposal row that ``merge_proposal_entries``
        and ``_prepare_rows_for_scoring`` can consume unchanged.
        """
        row = {
            "merge_key": f"library::{self.criterion_id}",
            "dimension": self.dimension,
            "label": self.label,
            "requirement": self.requirement,
            "severity_tier": self.severity_tier,
            "count": 1,
            "example_ids": [example_id] if example_id else [],
            "pair_ids": [pair_id] if pair_id else [],
            "criterion_ids": [self.criterion_id],
            "parent_criterion_ids": [],
            "root_pair_ids": [pair_id] if pair_id else [],
            "recursion_depths": [],
            "recursion_reasons": ["rubric_library"],
            "decomposition_sources": [f"rubric_library::{self.source_tag}"],
            "rubric_library_source": self.source_tag,
            "rubric_library_focus_kind": self.focus_kind,
            "rubric_library_criterion_id": self.criterion_id,
        }
        return row


@dataclass
class RubricLibrary:
    version: str
    criteria: List[RubricLibraryCriterion]
    build_metadata: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def criterion_count(self) -> int:
        return len(self.criteria)

    def filter_by_family(
        self,
        source_family: str,
        *,
        limit: Optional[int] = None,
        strict: bool = False,
    ) -> List[RubricLibraryCriterion]:
        """
        Return library criteria applicable to ``source_family``.

        When ``strict`` is True, only return criteria whose ``applicable_families`` literally
        contains ``source_family`` -- generic criteria are excluded. This is the "per-family
        library" mode used to prevent generic noise from polluting family-specific rubric
        scoring (e.g., generic criteria leak into ``livecodebench`` and regress accuracy).
        """
        if strict:
            matched = [c for c in self.criteria if source_family in c.applicable_families]
        else:
            matched = [c for c in self.criteria if c.matches_family(source_family)]
        matched.sort(
            key=lambda c: (
                0 if c.severity_tier == "hard_gate" else 1,
                0 if source_family in c.applicable_families else 1,
                -int(c.direction_evidence),
                c.criterion_id,
            )
        )
        if limit is not None and limit >= 0:
            matched = matched[: int(limit)]
        return matched

    def by_focus_kinds(
        self,
        source_family: str,
        focus_kinds: Sequence[str],
        *,
        limit: Optional[int] = None,
    ) -> List[RubricLibraryCriterion]:
        normalized = {str(fk).strip().lower() for fk in focus_kinds if str(fk).strip()}
        candidates = [
            c
            for c in self.criteria
            if c.matches_family(source_family) and (not normalized or c.focus_kind.lower() in normalized)
        ]
        if not candidates:
            return self.filter_by_family(source_family, limit=limit)
        candidates.sort(
            key=lambda c: (
                0 if c.severity_tier == "hard_gate" else 1,
                0 if source_family in c.applicable_families else 1,
                -int(c.direction_evidence),
                c.criterion_id,
            )
        )
        if limit is not None and limit >= 0:
            candidates = candidates[: int(limit)]
        return candidates

    def summarize(self) -> Dict[str, Any]:
        by_family: Dict[str, int] = {}
        by_dimension: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for c in self.criteria:
            for fam in c.applicable_families or ("generic",):
                by_family[fam] = by_family.get(fam, 0) + 1
            by_dimension[c.dimension] = by_dimension.get(c.dimension, 0) + 1
            by_severity[c.severity_tier] = by_severity.get(c.severity_tier, 0) + 1
        return {
            "version": self.version,
            "criterion_count": self.criterion_count,
            "by_family": by_family,
            "by_dimension": by_dimension,
            "by_severity": by_severity,
            "build_metadata": dict(self.build_metadata or {}),
        }


def _normalize_family_tags(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ("generic",)
    if isinstance(raw, str):
        parts = [t.strip() for t in raw.split(",") if t.strip()]
    elif isinstance(raw, Iterable):
        parts = [str(t).strip() for t in raw if str(t).strip()]
    else:
        parts = []
    filtered = tuple(p for p in parts if p in _ALLOWED_FAMILY_TAGS)
    return filtered or ("generic",)


def _normalize_severity(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value not in _ALLOWED_SEVERITY_TIERS:
        return "medium"
    return value


def _criterion_from_mapping(row: Mapping[str, Any]) -> RubricLibraryCriterion:
    criterion_id = str(row.get("criterion_id", "") or "").strip()
    if not criterion_id:
        raise ValueError("Rubric library criterion is missing required 'criterion_id'.")
    return RubricLibraryCriterion(
        criterion_id=criterion_id,
        dimension=str(row.get("dimension", "") or "").strip(),
        label=str(row.get("label", "") or "").strip(),
        requirement=str(row.get("requirement", "") or "").strip(),
        severity_tier=_normalize_severity(row.get("severity_tier")),
        applicable_families=_normalize_family_tags(row.get("applicable_families")),
        source_tag=str(row.get("source_tag", "") or "seed").strip(),
        focus_kind=str(row.get("focus_kind", "") or "").strip(),
        verification_notes=str(row.get("verification_notes", "") or "").strip(),
        direction_evidence=int(row.get("direction_evidence", 0) or 0),
        redundancy_cluster_id=str(row.get("redundancy_cluster_id", "") or "").strip(),
    )


def load_rubric_library(path: Path) -> RubricLibrary:
    """Load a rubric library from disk. Fails loudly on schema mismatches."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Rubric library at {path} is not a JSON object.")
    schema = str(payload.get("schema", "") or "")
    if schema != RUBRIC_LIBRARY_SCHEMA:
        raise ValueError(
            f"Rubric library at {path} has schema={schema!r}, expected {RUBRIC_LIBRARY_SCHEMA!r}."
        )
    version = str(payload.get("version", "v1") or "v1").strip()
    rows = payload.get("criteria", [])
    if not isinstance(rows, list):
        raise ValueError(f"Rubric library at {path} is missing a 'criteria' array.")
    criteria = [_criterion_from_mapping(row) for row in rows if isinstance(row, Mapping)]
    metadata = dict(payload.get("build_metadata", {}) or {})
    return RubricLibrary(
        version=version,
        criteria=criteria,
        build_metadata=metadata,
        path=path,
    )


def save_rubric_library(library: RubricLibrary, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RUBRIC_LIBRARY_SCHEMA,
        "version": library.version,
        "criterion_count": library.criterion_count,
        "build_metadata": dict(library.build_metadata or {}),
        "criteria": [
            {
                "schema": RUBRIC_LIBRARY_CRITERION_SCHEMA,
                "criterion_id": c.criterion_id,
                "dimension": c.dimension,
                "label": c.label,
                "requirement": c.requirement,
                "severity_tier": c.severity_tier,
                "applicable_families": list(c.applicable_families),
                "source_tag": c.source_tag,
                "focus_kind": c.focus_kind,
                "verification_notes": c.verification_notes,
                "direction_evidence": int(c.direction_evidence),
                "redundancy_cluster_id": c.redundancy_cluster_id,
            }
            for c in library.criteria
        ],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


_DEFAULT_LIBRARY_RELATIVE = Path("artifacts") / "rubric_library" / "v1" / "library.json"


def default_library_path(repo_root: Path) -> Path:
    return Path(repo_root) / _DEFAULT_LIBRARY_RELATIVE


def maybe_load_default_library(repo_root: Path) -> Optional[RubricLibrary]:
    """Return the default v1 library if it exists; ``None`` otherwise (never raises on missing file)."""
    path = default_library_path(repo_root)
    if not path.exists():
        return None
    try:
        return load_rubric_library(path)
    except Exception:
        return None
