"""
Offline builder for the JudgeBench v2 rubric library.

Given a manifest of external preference-pair sources (HelpSteer3, UltraFeedback, PPE, synthetic
multi-model pairs), this module runs the OpenJudge-style verify-refine loop to distill a compact
set of direction-validated, non-redundant criteria and writes them to
``artifacts/rubric_library/v1/library.json``.

The builder never reads JudgeBench 350 validation samples. Manifests must point at external data.

Design
------
The builder is structured so the per-pair discovery step is pluggable. In production runs we use
``rubric_gen.compiled.discovery.discover_pair_criteria`` with a multi-model proposer ensemble
(GPT-4o, Claude, Gemini). For tests and offline smoke runs we ship a ``_HandCuratedProposer`` that
returns deterministic rubric rows derived from the source pair tags — this is what populates the
seed library when no API keys are available.

Even the seed library goes through the verify-refine + coding-rate redundancy filter, so the
downstream contract is identical regardless of proposer.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.rubric_library import (
    RUBRIC_LIBRARY_SCHEMA,
    RubricLibrary,
    RubricLibraryCriterion,
    load_rubric_library,
    save_rubric_library,
)


_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ExternalPreferencePair:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    source: str
    source_family: str
    focus_kind: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposedCriterion:
    dimension: str
    label: str
    requirement: str
    severity_tier: str
    focus_kind: str
    source_tag: str

    def fingerprint(self) -> str:
        blob = "|".join((self.dimension, self.label, self.requirement, self.severity_tier)).lower()
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


ProposerFn = Callable[[ExternalPreferencePair], List[ProposedCriterion]]
VerifierFn = Callable[[ExternalPreferencePair, ProposedCriterion], Tuple[bool, int]]


def _stable_pair_id(pair: ExternalPreferencePair) -> str:
    if pair.pair_id:
        return pair.pair_id
    blob = f"{pair.source}::{pair.prompt[:200]}::{pair.chosen[:200]}::{pair.rejected[:200]}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _text_overlap(a: str, b: str) -> float:
    tokens_a = set(_tokens(a))
    tokens_b = set(_tokens(b))
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    denom = max(len(tokens_a), len(tokens_b))
    return inter / denom if denom else 0.0


def _default_verify(
    pair: ExternalPreferencePair,
    criterion: ProposedCriterion,
) -> Tuple[bool, int]:
    """
    Deterministic verifier used when no LLM proposer is configured.

    A criterion is accepted when the chosen side has at least as much surface-level coverage of
    the requirement text as the rejected side. This is intentionally conservative — the point
    of the offline builder is to drop criteria that fail *direction* on training data, not to be
    a state-of-the-art judge. Real distillation runs swap this for a GPT-4o verify-refine call.
    """
    requirement_tokens = set(_tokens(criterion.requirement))
    if not requirement_tokens:
        return False, 0
    chosen_tokens = set(_tokens(pair.chosen))
    rejected_tokens = set(_tokens(pair.rejected))
    chosen_score = len(requirement_tokens & chosen_tokens)
    rejected_score = len(requirement_tokens & rejected_tokens)
    aligned = chosen_score >= rejected_score and chosen_score > 0
    evidence = chosen_score - rejected_score
    return aligned, evidence


@dataclass
class BuilderConfig:
    target_total: int = 60
    per_family_target: int = 14
    redundancy_threshold: float = 0.82
    min_direction_evidence: int = 1
    max_per_dimension: int = 6
    require_verification: bool = True


@dataclass
class BuildResult:
    library: RubricLibrary
    accepted_count: int
    rejected_misaligned: int
    rejected_redundant: int
    proposals_seen: int
    per_family_counts: Dict[str, int]
    per_dimension_counts: Dict[str, int]


def _coding_rate_rank(
    accepted: List[Tuple[RubricLibraryCriterion, List[int]]],
    threshold: float,
) -> List[Tuple[RubricLibraryCriterion, List[int]]]:
    """
    Collapse near-duplicate criteria using a coding-rate style redundancy filter: greedily keep
    the strongest candidate per cluster, where two candidates share a cluster if their
    requirement-token Jaccard similarity exceeds ``threshold``.
    """
    kept: List[Tuple[RubricLibraryCriterion, List[int]]] = []
    for candidate, satisfaction in accepted:
        requirement_tokens = set(_tokens(candidate.requirement))
        is_redundant = False
        for kept_candidate, _ in kept:
            kept_tokens = set(_tokens(kept_candidate.requirement))
            inter = len(requirement_tokens & kept_tokens)
            union = len(requirement_tokens | kept_tokens) or 1
            if inter / union >= threshold:
                if candidate.severity_tier == "hard_gate" and kept_candidate.severity_tier != "hard_gate":
                    kept[:] = [
                        (c, s) for c, s in kept if c.criterion_id != kept_candidate.criterion_id
                    ]
                    kept.append((candidate, satisfaction))
                    is_redundant = True
                    break
                is_redundant = True
                break
        if not is_redundant:
            kept.append((candidate, satisfaction))
    return kept


def _assign_cluster_ids(kept: List[Tuple[RubricLibraryCriterion, List[int]]]) -> List[RubricLibraryCriterion]:
    result: List[RubricLibraryCriterion] = []
    for idx, (c, _) in enumerate(kept):
        result.append(
            RubricLibraryCriterion(
                criterion_id=c.criterion_id,
                dimension=c.dimension,
                label=c.label,
                requirement=c.requirement,
                severity_tier=c.severity_tier,
                applicable_families=c.applicable_families,
                source_tag=c.source_tag,
                focus_kind=c.focus_kind,
                verification_notes=c.verification_notes,
                direction_evidence=c.direction_evidence,
                redundancy_cluster_id=f"cluster_{idx:03d}",
            )
        )
    return result


def distill_library(
    pairs: Sequence[ExternalPreferencePair],
    *,
    proposer: ProposerFn,
    verifier: Optional[VerifierFn] = None,
    config: Optional[BuilderConfig] = None,
) -> BuildResult:
    """
    Core distillation loop: for each pair, proposer yields candidates, verifier drops misaligned
    ones, then a coding-rate redundancy filter collapses near-duplicates and a budget is applied.
    """
    cfg = config or BuilderConfig()
    verifier_fn = verifier or _default_verify
    per_family_counts: Dict[str, int] = {}
    per_dimension_counts: Dict[str, int] = {}
    proposals_seen = 0
    rejected_misaligned = 0

    accepted: List[Tuple[RubricLibraryCriterion, List[int]]] = []
    seen_fingerprints: Dict[str, Tuple[RubricLibraryCriterion, List[int]]] = {}

    for pair_index, pair in enumerate(pairs):
        pair_id = _stable_pair_id(pair)
        for proposed in proposer(pair):
            proposals_seen += 1
            if cfg.require_verification:
                aligned, evidence = verifier_fn(pair, proposed)
                if not aligned or evidence < cfg.min_direction_evidence:
                    rejected_misaligned += 1
                    continue
            else:
                aligned, evidence = True, cfg.min_direction_evidence
            fingerprint = proposed.fingerprint()
            criterion_id = f"lib_{fingerprint}"
            applicable_families: Tuple[str, ...]
            if pair.source_family:
                applicable_families = (pair.source_family,)
            else:
                applicable_families = ("generic",)
            if criterion_id in seen_fingerprints:
                existing, satisfaction = seen_fingerprints[criterion_id]
                merged_families = tuple(
                    sorted(set(existing.applicable_families) | set(applicable_families))
                )
                merged = RubricLibraryCriterion(
                    criterion_id=existing.criterion_id,
                    dimension=existing.dimension,
                    label=existing.label,
                    requirement=existing.requirement,
                    severity_tier=existing.severity_tier,
                    applicable_families=merged_families,
                    source_tag=existing.source_tag,
                    focus_kind=existing.focus_kind or proposed.focus_kind,
                    verification_notes=existing.verification_notes,
                    direction_evidence=existing.direction_evidence + max(0, int(evidence)),
                    redundancy_cluster_id=existing.redundancy_cluster_id,
                )
                satisfaction.append(pair_index)
                seen_fingerprints[criterion_id] = (merged, satisfaction)
                continue
            criterion = RubricLibraryCriterion(
                criterion_id=criterion_id,
                dimension=proposed.dimension,
                label=proposed.label,
                requirement=proposed.requirement,
                severity_tier=proposed.severity_tier,
                applicable_families=applicable_families,
                source_tag=proposed.source_tag,
                focus_kind=proposed.focus_kind,
                verification_notes=pair.source,
                direction_evidence=max(0, int(evidence)),
                redundancy_cluster_id="",
            )
            seen_fingerprints[criterion_id] = (criterion, [pair_index])

    accepted = list(seen_fingerprints.values())

    kept = _coding_rate_rank(accepted, threshold=cfg.redundancy_threshold)
    rejected_redundant = len(accepted) - len(kept)

    kept.sort(
        key=lambda item: (
            0 if item[0].severity_tier == "hard_gate" else 1,
            -int(item[0].direction_evidence),
            item[0].dimension,
            item[0].criterion_id,
        )
    )

    budgeted: List[Tuple[RubricLibraryCriterion, List[int]]] = []
    dim_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    for criterion, satisfaction in kept:
        if dim_counts.get(criterion.dimension, 0) >= cfg.max_per_dimension:
            continue
        fam_ok = True
        for fam in criterion.applicable_families:
            if family_counts.get(fam, 0) >= cfg.per_family_target and fam != "generic":
                fam_ok = False
                break
        if not fam_ok:
            continue
        if len(budgeted) >= cfg.target_total:
            break
        budgeted.append((criterion, satisfaction))
        dim_counts[criterion.dimension] = dim_counts.get(criterion.dimension, 0) + 1
        for fam in criterion.applicable_families:
            family_counts[fam] = family_counts.get(fam, 0) + 1

    final_criteria = _assign_cluster_ids(budgeted)
    per_family_counts.update(family_counts)
    per_dimension_counts.update(dim_counts)

    built_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    library = RubricLibrary(
        version="v1",
        criteria=final_criteria,
        build_metadata={
            "built_at": built_at,
            "proposals_seen": proposals_seen,
            "rejected_misaligned": rejected_misaligned,
            "rejected_redundant": rejected_redundant,
            "pair_count": len(pairs),
            "config": {
                "target_total": cfg.target_total,
                "per_family_target": cfg.per_family_target,
                "redundancy_threshold": cfg.redundancy_threshold,
                "min_direction_evidence": cfg.min_direction_evidence,
                "max_per_dimension": cfg.max_per_dimension,
                "require_verification": cfg.require_verification,
            },
        },
    )

    return BuildResult(
        library=library,
        accepted_count=len(final_criteria),
        rejected_misaligned=rejected_misaligned,
        rejected_redundant=rejected_redundant,
        proposals_seen=proposals_seen,
        per_family_counts=per_family_counts,
        per_dimension_counts=per_dimension_counts,
    )


def build_library_from_manifest(
    manifest: Mapping[str, Any],
    *,
    proposer: ProposerFn,
    verifier: Optional[VerifierFn] = None,
    config: Optional[BuilderConfig] = None,
    loader_registry: Optional[Mapping[str, Callable[[Mapping[str, Any]], Iterable[ExternalPreferencePair]]]] = None,
) -> BuildResult:
    """
    Manifest-driven builder. The manifest is a mapping with an optional ``sources`` array; each
    entry names a source kind and arguments, loaded via ``loader_registry`` (the caller supplies
    HelpSteer3 / UltraFeedback / PPE loaders in production). In this module's tests we supply an
    inline list of pairs for determinism.
    """
    registry = dict(loader_registry or {})
    pairs: List[ExternalPreferencePair] = []
    inline_pairs = list(manifest.get("inline_pairs", []) or [])
    for row in inline_pairs:
        if isinstance(row, ExternalPreferencePair):
            pairs.append(row)
            continue
        if not isinstance(row, Mapping):
            continue
        pairs.append(
            ExternalPreferencePair(
                pair_id=str(row.get("pair_id", "") or ""),
                prompt=str(row.get("prompt", "") or ""),
                chosen=str(row.get("chosen", "") or ""),
                rejected=str(row.get("rejected", "") or ""),
                source=str(row.get("source", "") or "inline"),
                source_family=str(row.get("source_family", "") or "generic"),
                focus_kind=str(row.get("focus_kind", "") or ""),
                metadata=dict(row.get("metadata", {}) or {}),
            )
        )
    for source in manifest.get("sources", []) or []:
        if not isinstance(source, Mapping):
            continue
        kind = str(source.get("kind", "") or "").strip()
        loader = registry.get(kind)
        if loader is None:
            continue
        for pair in loader(source):
            pairs.append(pair)

    return distill_library(pairs, proposer=proposer, verifier=verifier, config=config)


def write_library(library: RubricLibrary, path: Path) -> Path:
    return save_rubric_library(library, path)


def read_library(path: Path) -> RubricLibrary:
    return load_rubric_library(path)


def merge_library_with_existing(
    existing: Optional[RubricLibrary],
    new: RubricLibrary,
    *,
    prefer_new: bool = False,
) -> RubricLibrary:
    """
    Simple stable merge: dedupe by ``criterion_id``. When ``prefer_new`` is ``True`` the new
    library wins on conflicts. This is used by iterative re-builds that append criteria to a
    previously frozen v1 library.
    """
    if not existing:
        return new
    keyed: Dict[str, RubricLibraryCriterion] = {c.criterion_id: c for c in existing.criteria}
    for c in new.criteria:
        if c.criterion_id in keyed and not prefer_new:
            continue
        keyed[c.criterion_id] = c
    merged = list(keyed.values())
    metadata = dict(existing.build_metadata or {})
    metadata["merged_with"] = dict(new.build_metadata or {})
    return RubricLibrary(
        version=max(existing.version, new.version),
        criteria=merged,
        build_metadata=metadata,
        path=existing.path,
    )
