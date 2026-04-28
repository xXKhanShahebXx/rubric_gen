from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rubric_gen.compiled.pair_consensus import compute_pair_consensus


def _write_run(tmp: Path, run_name: str, decisions: dict) -> Path:
    """Create a fake run dir with artifacts in validation_350/final/examples/."""
    run_dir = tmp / run_name
    examples_dir = run_dir / "validation_350" / "final" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for pair_id, (decision, label, source_family) in decisions.items():
        artifact = {
            "pair": {"pair_id": pair_id, "label": label, "source_family": source_family},
            "scoring": {"whitened_uniform": {"result": {"decision": decision}}},
        }
        (examples_dir / f"{pair_id}.json").write_text(json.dumps(artifact), encoding="utf-8")
    return run_dir


class ComputePairConsensusTests(unittest.TestCase):
    def test_unanimous_correct_consensus(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run1 = _write_run(tmp_path, "r1", {"p1": ("A>B", "A>B", "mmlu-pro")})
            run2 = _write_run(tmp_path, "r2", {"p1": ("A>B", "A>B", "mmlu-pro")})
            run3 = _write_run(tmp_path, "r3", {"p1": ("A>B", "A>B", "mmlu-pro")})
            summary = compute_pair_consensus([run1, run2, run3])
            self.assertEqual(summary.total_pairs, 1)
            self.assertEqual(summary.correct_count, 1)
            self.assertEqual(summary.pairs_with_unanimous_votes, 1)
            self.assertEqual(summary.entries[0].consensus_decision, "A>B")
            self.assertTrue(summary.entries[0].is_consensus_correct)

    def test_majority_recovers_from_one_wrong_run(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run1 = _write_run(tmp_path, "r1", {"p1": ("A>B", "A>B", "mmlu-pro")})
            run2 = _write_run(tmp_path, "r2", {"p1": ("B>A", "A>B", "mmlu-pro")})
            run3 = _write_run(tmp_path, "r3", {"p1": ("A>B", "A>B", "mmlu-pro")})
            summary = compute_pair_consensus([run1, run2, run3])
            self.assertEqual(summary.correct_count, 1)
            self.assertEqual(summary.pairs_with_split_votes, 1)
            self.assertEqual(summary.entries[0].consensus_decision, "A>B")

    def test_tied_votes_default_to_a_eq_b(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run1 = _write_run(tmp_path, "r1", {"p1": ("A>B", "A>B", "mmlu-pro")})
            run2 = _write_run(tmp_path, "r2", {"p1": ("B>A", "A>B", "mmlu-pro")})
            summary = compute_pair_consensus([run1, run2])
            self.assertEqual(summary.pairs_with_tie_votes, 1)
            self.assertEqual(summary.entries[0].consensus_decision, "A=B")
            self.assertFalse(summary.entries[0].is_consensus_correct)

    def test_per_family_aggregation(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run1 = _write_run(
                tmp_path,
                "r1",
                {
                    "p_math": ("A>B", "A>B", "livebench-math"),
                    "p_reason": ("A>B", "B>A", "livebench-reasoning"),
                },
            )
            run2 = _write_run(
                tmp_path,
                "r2",
                {
                    "p_math": ("A>B", "A>B", "livebench-math"),
                    "p_reason": ("A>B", "B>A", "livebench-reasoning"),
                },
            )
            summary = compute_pair_consensus([run1, run2])
            self.assertEqual(summary.by_family["livebench-math"]["correct_count"], 1)
            self.assertEqual(summary.by_family["livebench-reasoning"]["correct_count"], 0)

    def test_handles_missing_artifact_in_some_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run1 = _write_run(tmp_path, "r1", {"p1": ("A>B", "A>B", "mmlu-pro")})
            run2 = _write_run(tmp_path, "r2", {})
            summary = compute_pair_consensus([run1, run2])
            self.assertEqual(summary.total_pairs, 1)
            self.assertEqual(summary.entries[0].runs_seen, 1)


if __name__ == "__main__":
    unittest.main()
