from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List


def _load_rescore_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rescore_pair_artifacts.py"
    spec = importlib.util.spec_from_file_location("rescore_pair_artifacts", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_rescore = _load_rescore_module()


def _make_pair_artifact(
    *,
    source_id: str,
    pair_label: str,
    a_rank: int,
    b_rank: int,
) -> Dict[str, Any]:
    """Synthesize a per-example artifact with one method's pair ranking."""
    candidates = [
        {
            "candidate_id": f"med__{source_id}__pair_a",
            "source_label": "pair_response_a",
            "quality_bucket": "pair_anchor",
            "metadata": {"pair_correct_label": pair_label},
        },
        {
            "candidate_id": f"med__{source_id}__pair_b",
            "source_label": "pair_response_b",
            "quality_bucket": "pair_anchor",
            "metadata": {"pair_correct_label": pair_label},
        },
    ]
    methods = {
        "rrd_uniform": {
            "ranking": [
                {
                    "candidate_id": f"med__{source_id}__pair_a",
                    "rank": a_rank,
                    "score": float(10 - a_rank),
                },
                {
                    "candidate_id": f"med__{source_id}__pair_b",
                    "rank": b_rank,
                    "score": float(10 - b_rank),
                },
            ],
            "rubrics": [],
        }
    }
    return {
        "example": {
            "example_id": f"med__{source_id}",
            "source_id": source_id,
            "pair_correct_label": pair_label,
        },
        "candidates": candidates,
        "methods": methods,
    }


def _write_relabeled_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# _override_pair_label
# ---------------------------------------------------------------------------


class OverridePairLabelTests(unittest.TestCase):
    def test_override_replaces_label_on_pair_anchor_candidates(self) -> None:
        artifact = _make_pair_artifact(source_id="0006000-x", pair_label="a", a_rank=1, b_rank=2)
        out = _rescore._override_pair_label(artifact, "b")
        labels = [c["metadata"]["pair_correct_label"] for c in out["candidates"]]
        self.assertEqual(labels, ["b", "b"])
        # Mirrors onto the example block too.
        self.assertEqual(out["example"]["pair_correct_label"], "b")
        # Original artifact is unchanged.
        self.assertEqual(artifact["candidates"][0]["metadata"]["pair_correct_label"], "a")

    def test_override_with_empty_label_clears_pair_metric(self) -> None:
        from rubric_gen.evaluation.reporting import aggregate_method_metrics

        artifact = _make_pair_artifact(source_id="x", pair_label="a", a_rank=1, b_rank=2)
        cleared = _rescore._override_pair_label(artifact, "")
        rows = aggregate_method_metrics([cleared])
        self.assertEqual(rows[0]["pair_preference_evaluable"], 0)


# ---------------------------------------------------------------------------
# Aggregator response to label override
# ---------------------------------------------------------------------------


class RescoreFlowTests(unittest.TestCase):
    def test_relabel_flips_accuracy(self) -> None:
        from rubric_gen.evaluation.reporting import aggregate_method_metrics

        # Three artifacts, all with original label "a" and pair_a ranking #1
        # (so the pipeline picks A on every one). Original accuracy = 1.0.
        artifacts = [
            _make_pair_artifact(source_id=f"x{i}", pair_label="a", a_rank=1, b_rank=2)
            for i in range(3)
        ]
        before = aggregate_method_metrics(artifacts)
        self.assertEqual(before[0]["pair_preference_correct"], 3)
        self.assertEqual(before[0]["pair_preference_evaluable"], 3)
        self.assertAlmostEqual(before[0]["pair_preference_accuracy"], 1.0)

        # Flip every label to "b" via the override path. The pipeline still
        # picks A every time, so accuracy now drops to 0.
        relabeled = [_rescore._override_pair_label(art, "b") for art in artifacts]
        after = aggregate_method_metrics(relabeled)
        self.assertEqual(after[0]["pair_preference_correct"], 0)
        self.assertEqual(after[0]["pair_preference_evaluable"], 3)
        self.assertAlmostEqual(after[0]["pair_preference_accuracy"], 0.0)

    def test_partial_relabel_some_correct_some_wrong(self) -> None:
        from rubric_gen.evaluation.reporting import aggregate_method_metrics

        # Pipeline always picks A. Original labels all "a" (accuracy 1.0).
        artifacts = [
            _make_pair_artifact(source_id=f"x{i}", pair_label="a", a_rank=1, b_rank=2)
            for i in range(4)
        ]
        # Relabel two of them to "b". Pipeline still picks A on all four.
        # New accuracy = 2/4 = 0.5.
        relabeled = []
        for i, art in enumerate(artifacts):
            new_label = "b" if i < 2 else "a"
            relabeled.append(_rescore._override_pair_label(art, new_label))
        rows = aggregate_method_metrics(relabeled)
        self.assertAlmostEqual(rows[0]["pair_preference_accuracy"], 0.5)
        self.assertEqual(rows[0]["pair_preference_correct"], 2)
        self.assertEqual(rows[0]["pair_preference_evaluable"], 4)


# ---------------------------------------------------------------------------
# _load_relabeled_index
# ---------------------------------------------------------------------------


class LoadRelabeledIndexTests(unittest.TestCase):
    def test_loads_id_to_row_and_id_to_label(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "relabel.jsonl"
            _write_relabeled_jsonl(
                path,
                [
                    {"id": "0006000-a", "judge_verdict": "A", "correct_answer": "reference_answer_a"},
                    {"id": "0006001-b", "judge_verdict": "B", "correct_answer": "reference_answer_b"},
                    {"id": "0006002-tie", "judge_verdict": "TIE", "correct_answer": "reference_answer_a"},
                    {"id": "0006003-bad", "judge_verdict": "JUNK"},
                ],
            )
            by_id, label_by_id = _rescore._load_relabeled_index(path)

        self.assertEqual(set(by_id.keys()), {"0006000-a", "0006001-b", "0006002-tie", "0006003-bad"})
        self.assertEqual(label_by_id["0006000-a"], "a")
        self.assertEqual(label_by_id["0006001-b"], "b")
        self.assertEqual(label_by_id["0006002-tie"], "")  # TIE -> empty
        self.assertEqual(label_by_id["0006003-bad"], "")  # unknown verdict -> empty


if __name__ == "__main__":
    unittest.main()
