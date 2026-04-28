"""
External preference slices used as a blind-350 *proxy* during train-only development.

The train-only OOF score has a large (~20-point) generalisation gap to the blind-350 validation
set in this repo. To reduce the risk of spending a blind-350 shot on a non-transportable
candidate, every OOF run additionally scores the locked policy on two *external* held-out slices
that are disjoint from both JudgeBench splits:

- ``helpsteer3_val``: 400 HelpSteer3-Preference pairs (prompt, chosen, rejected), fixed seed.
- ``ppe_mmlu_gpqa_val``: 400 PPE pairs from MMLU-Pro + GPQA subsets.

The slices are stored as JSONL under ``artifacts/external_eval_sets/`` and are loaded by
:func:`load_external_slice`. The scoring routine here re-uses the existing GPT-4o pair judge via
:mod:`rubric_gen.compiled.holistic_judge.run_holistic_pair_judge`, treating the locked rubric
library as the rubric input. This gives a decision per pair that we compare against the
preference label to produce a WU-style score.

The slices are optional: if the JSONL files are missing, :func:`score_external_slices` returns an
empty summary so training harnesses fail open.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from rubric_gen.compiled.rubric_library import RubricLibrary, maybe_load_default_library


EXTERNAL_EVAL_SET_DIR = Path("artifacts") / "external_eval_sets"

_SUPPORTED_FAMILIES = ("generic", "mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench")


@dataclass(frozen=True)
class ExternalSlicePair:
    pair_id: str
    prompt: str
    response_a: str
    response_b: str
    label: str
    source_family: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalSliceScoring:
    slice_name: str
    pair_count: int
    correct_count: int
    tie_count: int
    wu_score: float
    by_family: Dict[str, Dict[str, Any]]
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "schema": "compiled_judgebench_external_slice_scoring_v1",
            "pair_count": int(self.pair_count),
            "correct_count": int(self.correct_count),
            "tie_count": int(self.tie_count),
            "wu_score": float(self.wu_score),
            "by_family": dict(self.by_family),
            "failures": list(self.failures),
        }


def default_slice_path(slice_name: str, *, repo_root: Path) -> Path:
    return Path(repo_root) / EXTERNAL_EVAL_SET_DIR / f"{slice_name}.jsonl"


def load_external_slice(path: Path) -> List[ExternalSlicePair]:
    """Load a slice JSONL. Missing files return an empty list (fail open)."""
    path = Path(path)
    if not path.exists():
        return []
    pairs: List[ExternalSlicePair] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, Mapping):
                continue
            prompt = str(row.get("prompt", "") or "")
            response_a = str(row.get("response_a", row.get("chosen", "")) or "")
            response_b = str(row.get("response_b", row.get("rejected", "")) or "")
            label = str(row.get("label", "A>B") or "A>B").strip()
            if label not in {"A>B", "B>A", "A=B"}:
                label = "A>B"
            source_family = str(row.get("source_family", "generic") or "generic").strip().lower()
            if source_family not in _SUPPORTED_FAMILIES:
                source_family = "generic"
            pairs.append(
                ExternalSlicePair(
                    pair_id=str(row.get("pair_id", f"{path.stem}_{idx}") or f"{path.stem}_{idx}"),
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                    label=label,
                    source_family=source_family,
                    source=str(row.get("source", path.stem) or path.stem),
                    metadata=dict(row.get("metadata", {}) or {}),
                )
            )
    return pairs


def _family_accuracy(pairs: Sequence[ExternalSlicePair], decisions: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    counts: Dict[str, Dict[str, int]] = {}
    for pair, decision in zip(pairs, decisions):
        family = pair.source_family
        bucket = counts.setdefault(family, {"pair_count": 0, "correct_count": 0, "tie_count": 0})
        bucket["pair_count"] += 1
        if decision == pair.label and decision != "A=B":
            bucket["correct_count"] += 1
        if decision == "A=B":
            bucket["tie_count"] += 1
    result: Dict[str, Dict[str, Any]] = {}
    for family, bucket in counts.items():
        pair_count = bucket["pair_count"] or 1
        result[family] = {
            "pair_count": bucket["pair_count"],
            "correct_count": bucket["correct_count"],
            "tie_count": bucket["tie_count"],
            "wu_score": round(100.0 * bucket["correct_count"] / pair_count, 4),
        }
    return result


PairJudgeFn = Callable[[ExternalSlicePair, RubricLibrary], str]


def _default_fallback_judge(pair: ExternalSlicePair, library: RubricLibrary) -> str:
    """
    Deterministic fallback judge used when no LLM judge is configured. Returns a decision by
    counting which response has more keyword overlap with the family's library criteria. This is
    only intended for smoke tests; real runs should pass a real LLM-backed judge.
    """
    if not pair.response_a and not pair.response_b:
        return "A=B"
    criteria = library.filter_by_family(pair.source_family, limit=6)
    terms: List[str] = []
    for c in criteria:
        terms.extend(token.strip(".,;:").lower() for token in (c.label + " " + c.requirement).split() if token)
    if not terms:
        terms = [token.strip(".,;:").lower() for token in pair.prompt.split() if token]
    response_a_lower = pair.response_a.lower()
    response_b_lower = pair.response_b.lower()
    score_a = sum(1 for term in terms if term and term in response_a_lower)
    score_b = sum(1 for term in terms if term and term in response_b_lower)
    if score_a > score_b:
        return "A>B"
    if score_b > score_a:
        return "B>A"
    return "A=B"


def score_external_slice(
    pairs: Sequence[ExternalSlicePair],
    *,
    library: RubricLibrary,
    slice_name: str,
    pair_judge: Optional[PairJudgeFn] = None,
    max_failures_tracked: int = 40,
) -> ExternalSliceScoring:
    judge_fn = pair_judge or _default_fallback_judge
    decisions: List[str] = []
    failures: List[Dict[str, Any]] = []
    correct = 0
    tie = 0
    for pair in pairs:
        try:
            decision = judge_fn(pair, library)
        except Exception:
            decision = ""
        if decision not in {"A>B", "B>A", "A=B"}:
            decision = "A=B"
        decisions.append(decision)
        if decision == "A=B":
            tie += 1
        if decision == pair.label and decision != "A=B":
            correct += 1
        elif decision != pair.label:
            if len(failures) < max_failures_tracked:
                failures.append(
                    {
                        "pair_id": pair.pair_id,
                        "source_family": pair.source_family,
                        "label": pair.label,
                        "decision": decision,
                        "source": pair.source,
                    }
                )
    pair_count = len(pairs) or 1
    wu_score = round(100.0 * correct / pair_count, 4)
    return ExternalSliceScoring(
        slice_name=slice_name,
        pair_count=len(pairs),
        correct_count=correct,
        tie_count=tie,
        wu_score=wu_score,
        by_family=_family_accuracy(pairs, decisions),
        failures=failures,
    )


def score_external_slices(
    *,
    repo_root: Path,
    slice_names: Sequence[str] = ("helpsteer3_val", "ppe_mmlu_gpqa_val"),
    library: Optional[RubricLibrary] = None,
    pair_judge: Optional[PairJudgeFn] = None,
) -> Dict[str, Dict[str, Any]]:
    if library is None:
        library = maybe_load_default_library(repo_root)
    if library is None:
        return {
            "available": False,
            "reason": "rubric_library_missing",
            "slices": {},
        }
    slices: Dict[str, Dict[str, Any]] = {}
    any_loaded = False
    for slice_name in slice_names:
        path = default_slice_path(slice_name, repo_root=repo_root)
        pairs = load_external_slice(path)
        if not pairs:
            slices[slice_name] = {
                "available": False,
                "path": str(path),
                "reason": "slice_file_missing_or_empty",
            }
            continue
        any_loaded = True
        scoring = score_external_slice(
            pairs,
            library=library,
            slice_name=slice_name,
            pair_judge=pair_judge,
        )
        slices[slice_name] = {
            "available": True,
            "path": str(path),
            **scoring.to_dict(),
        }
    return {
        "schema": "compiled_judgebench_external_slice_summary_v1",
        "available": any_loaded,
        "slices": slices,
    }


def write_seed_slice_if_missing(
    slice_name: str,
    *,
    repo_root: Path,
    overwrite: bool = False,
    seed: int = 29,
) -> Path:
    """
    Write a tiny deterministic seed slice so tests can run end-to-end. Production slices should be
    generated by an offline tool that downloads HelpSteer3/PPE and writes JSONL pairs; this helper
    only exists to give the pipeline something non-empty in CI.
    """
    path = default_slice_path(slice_name, repo_root=repo_root)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    families = ("generic", "mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench")
    rows: List[Dict[str, Any]] = []
    for idx in range(20):
        family = families[idx % len(families)]
        chosen = f"Strong response {idx} for {family}: includes final answer and supporting detail."
        rejected = f"Weak response {idx} for {family}: leaves the final answer unclear."
        label = rng.choice(["A>B", "B>A"])
        if label == "A>B":
            response_a, response_b = chosen, rejected
        else:
            response_a, response_b = rejected, chosen
        rows.append(
            {
                "pair_id": f"{slice_name}_{idx:03d}",
                "prompt": f"Example prompt {idx} for {family}.",
                "response_a": response_a,
                "response_b": response_b,
                "label": label,
                "source_family": family,
                "source": slice_name,
            }
        )
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path
