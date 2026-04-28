"""
Per-pair N-sample voting on final blind-350 decisions.

The diagnostic on the v3 runs identified ~104 noise-band pairs that flip correct/wrong status
across runs that differ only in re-rolled discriminator / math-solver samples (with cache
invalidated by the per-run cache directory). Many of these are genuinely close to the WU
decision boundary and benefit from majority voting across multiple full pipeline runs.

This module computes per-pair majority votes across M run directories and reports the
consensus accuracy. To use it:

1. Run ``judgebench_eval_runner final-eval`` M times with the same locked policy and
   ``--shared-cache-dir`` so that prompt-content cache hits eliminate spurious re-rolls but
   the temperature>0 samples still vary.
2. Pass each ``--run-name`` to this CLI.
3. The output ``pair_consensus.json`` contains per-pair majority decisions and the consensus
   accuracy (single-order, since the codebase's ``compute_double_order_accuracy`` uses
   ``decision_reversed = flip(decision)``).

The vote is a simple plurality on the final ``decision`` field of each per-example artifact's
``scoring.whitened_uniform.result``. Ties default to ``A=B`` (counted as a failure when the
gold is directional).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


_FAMILIES = ("mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench")


@dataclass
class PairConsensusEntry:
    pair_id: str
    source_family: str
    label: str
    votes: Dict[str, int]
    consensus_decision: str
    is_consensus_correct: bool
    runs_seen: int


@dataclass
class PairConsensusSummary:
    run_dirs: List[str]
    total_pairs: int
    correct_count: int
    by_family: Dict[str, Dict[str, int]]
    pairs_with_unanimous_votes: int
    pairs_with_split_votes: int
    pairs_with_tie_votes: int
    entries: List[PairConsensusEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        accuracy = (100.0 * self.correct_count / self.total_pairs) if self.total_pairs else 0.0
        return {
            "schema": "compiled_judgebench_pair_consensus_v1",
            "run_dirs": list(self.run_dirs),
            "total_pairs": int(self.total_pairs),
            "correct_count": int(self.correct_count),
            "consensus_accuracy_pct": round(accuracy, 4),
            "by_family": {
                fam: {
                    "pair_count": int(payload.get("pair_count", 0)),
                    "correct_count": int(payload.get("correct_count", 0)),
                    "accuracy_pct": round(
                        100.0
                        * payload.get("correct_count", 0)
                        / max(1, payload.get("pair_count", 1)),
                        4,
                    ),
                }
                for fam, payload in self.by_family.items()
            },
            "pairs_with_unanimous_votes": int(self.pairs_with_unanimous_votes),
            "pairs_with_split_votes": int(self.pairs_with_split_votes),
            "pairs_with_tie_votes": int(self.pairs_with_tie_votes),
            "entries": [asdict(e) for e in self.entries],
        }


def _load_run_artifacts(run_dir: Path) -> Dict[str, Mapping[str, Any]]:
    """Return a mapping of pair_id -> artifact for the per-example JSON files in ``run_dir``."""
    examples_dir = None
    for candidate in run_dir.rglob("examples"):
        if candidate.is_dir():
            examples_dir = candidate
            break
    if examples_dir is None:
        return {}
    out: Dict[str, Mapping[str, Any]] = {}
    for path in sorted(examples_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair_id = str(((payload.get("pair", {}) or {}).get("pair_id") or "")).strip()
        if pair_id:
            out[pair_id] = payload
    return out


def _decision_of(artifact: Mapping[str, Any]) -> str:
    return str(
        (((artifact.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {})
        .get("decision", "")
    ).strip()


def compute_pair_consensus(run_dirs: List[Path]) -> PairConsensusSummary:
    if not run_dirs:
        return PairConsensusSummary(
            run_dirs=[],
            total_pairs=0,
            correct_count=0,
            by_family={},
            pairs_with_unanimous_votes=0,
            pairs_with_split_votes=0,
            pairs_with_tie_votes=0,
        )
    artifacts_per_run: List[Dict[str, Mapping[str, Any]]] = [
        _load_run_artifacts(Path(d)) for d in run_dirs
    ]
    pair_ids: set = set()
    for run in artifacts_per_run:
        pair_ids.update(run.keys())

    entries: List[PairConsensusEntry] = []
    by_family: Dict[str, Dict[str, int]] = {}
    correct_total = 0
    unanimous = 0
    split = 0
    tied = 0
    for pair_id in sorted(pair_ids):
        votes: Counter = Counter()
        label = ""
        source_family = ""
        runs_seen = 0
        for run in artifacts_per_run:
            artifact = run.get(pair_id)
            if not artifact:
                continue
            runs_seen += 1
            label = label or str(((artifact.get("pair", {}) or {}).get("label") or ""))
            source_family = source_family or str(
                ((artifact.get("pair", {}) or {}).get("source_family") or "")
            )
            decision = _decision_of(artifact)
            if decision in {"A>B", "B>A", "A=B"}:
                votes[decision] += 1
        if not votes:
            continue
        sorted_votes = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        winner, winner_count = sorted_votes[0]
        runner_up_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
        if winner_count == runs_seen:
            unanimous += 1
        elif winner_count > runner_up_count:
            split += 1
        else:
            tied += 1
            winner = "A=B"
        is_correct = winner == label
        entries.append(
            PairConsensusEntry(
                pair_id=pair_id,
                source_family=source_family,
                label=label,
                votes=dict(votes),
                consensus_decision=winner,
                is_consensus_correct=bool(is_correct),
                runs_seen=runs_seen,
            )
        )
        if is_correct:
            correct_total += 1
        family_bucket = by_family.setdefault(
            source_family,
            {"pair_count": 0, "correct_count": 0},
        )
        family_bucket["pair_count"] += 1
        if is_correct:
            family_bucket["correct_count"] += 1

    return PairConsensusSummary(
        run_dirs=[str(d) for d in run_dirs],
        total_pairs=len(entries),
        correct_count=correct_total,
        by_family=by_family,
        pairs_with_unanimous_votes=unanimous,
        pairs_with_split_votes=split,
        pairs_with_tie_votes=tied,
        entries=entries,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-pair majority-vote consensus across multiple JudgeBench final-eval runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="A run directory containing validation_<split>/final/examples/*.json artifacts. Repeat.",
    )
    parser.add_argument("--out-path", type=Path, default=None)
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw)
    summary = compute_pair_consensus(args.run_dir)
    out_path = args.out_path or (Path(args.run_dir[0]).parent / f"{Path(args.run_dir[0]).name}_consensus.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    payload = summary.to_dict()
    headline = {
        "run_count": len(args.run_dir),
        "total_pairs": payload["total_pairs"],
        "consensus_accuracy_pct": payload["consensus_accuracy_pct"],
        "by_family": payload["by_family"],
        "pairs_with_unanimous_votes": payload["pairs_with_unanimous_votes"],
        "pairs_with_split_votes": payload["pairs_with_split_votes"],
        "pairs_with_tie_votes": payload["pairs_with_tie_votes"],
        "out_path": str(out_path),
    }
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
