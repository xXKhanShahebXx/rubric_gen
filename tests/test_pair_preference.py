from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from rubric_gen.candidate_generation import _anchor_candidates
from rubric_gen.dataio import (
    _apply_split_and_shard,
    _normalize_pair_label,
    load_examples,
)
from rubric_gen.evaluation.reporting import (
    _pair_preference_outcome,
    aggregate_method_metrics,
)
from rubric_gen.types import ExampleRecord


# ---------------------------------------------------------------------------
# _normalize_pair_label
# ---------------------------------------------------------------------------


class NormalizePairLabelTests(unittest.TestCase):
    def test_explicit_a_aliases(self) -> None:
        for value in [
            "a",
            "A",
            "  a  ",
            "reference_answer_a",
            "Reference_Answer_A",
            "response_a",
            "answer_a",
            "candidate_a",
            "pair_response_a",
        ]:
            self.assertEqual(_normalize_pair_label(value), "a", value)

    def test_explicit_b_aliases(self) -> None:
        for value in [
            "b",
            "B",
            "reference_answer_b",
            "response_b",
            "answer_b",
            "candidate_b",
            "pair_response_b",
        ]:
            self.assertEqual(_normalize_pair_label(value), "b", value)

    def test_integer_fallback(self) -> None:
        self.assertEqual(_normalize_pair_label(0), "a")
        self.assertEqual(_normalize_pair_label(1), "b")
        self.assertEqual(_normalize_pair_label(2), "")
        self.assertEqual(_normalize_pair_label(-1), "")

    def test_bool_does_not_resolve(self) -> None:
        # bool is a subclass of int but the semantics ("True/False" -> A/B) are
        # ambiguous, so we should not silently convert.
        self.assertEqual(_normalize_pair_label(True), "")
        self.assertEqual(_normalize_pair_label(False), "")

    def test_empty_or_unknown_returns_empty(self) -> None:
        self.assertEqual(_normalize_pair_label(None), "")
        self.assertEqual(_normalize_pair_label(""), "")
        self.assertEqual(_normalize_pair_label("c"), "")
        self.assertEqual(_normalize_pair_label("tie"), "")


# ---------------------------------------------------------------------------
# load_examples on a pair JSONL
# ---------------------------------------------------------------------------


class PairLoaderTests(unittest.TestCase):
    def _write_pair_jsonl(self, tmp: Path, rows: List[Dict[str, Any]]) -> Path:
        path = tmp / "pair.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def test_pair_fields_populated_when_present(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_pair_jsonl(
                Path(tmp),
                [
                    {
                        "id": "0006000-test",
                        "question": "What is the recommended treatment for vocal cord palsy?",
                        "reference_answer_a": "Voice therapy first, then surgery if needed.",
                        "reference_answer_b": "Always immediate thyroplasty.",
                        "correct_answer": "reference_answer_a",
                    }
                ],
            )
            examples = load_examples(path)
        self.assertEqual(len(examples), 1)
        ex = examples[0]
        self.assertTrue(ex.has_pair_candidates)
        self.assertEqual(ex.pair_response_a, "Voice therapy first, then surgery if needed.")
        self.assertEqual(ex.pair_response_b, "Always immediate thyroplasty.")
        self.assertEqual(ex.pair_correct_label, "a")

    def test_pair_fields_empty_on_single_response_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_pair_jsonl(
                Path(tmp),
                [
                    {
                        "id": "0005000-test",
                        "Question": "What is the capital of France?",
                        "response": "Paris.",
                        "source": "medical_o1_subset_b",
                    }
                ],
            )
            examples = load_examples(path)
        self.assertEqual(len(examples), 1)
        ex = examples[0]
        self.assertFalse(ex.has_pair_candidates)
        self.assertEqual(ex.pair_response_a, "")
        self.assertEqual(ex.pair_response_b, "")
        self.assertEqual(ex.pair_correct_label, "")

    def test_unknown_label_leaves_pair_label_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            path = self._write_pair_jsonl(
                Path(tmp),
                [
                    {
                        "id": "0006001-test",
                        "question": "Q?",
                        "reference_answer_a": "A response.",
                        "reference_answer_b": "B response.",
                        "correct_answer": "neither",
                    }
                ],
            )
            examples = load_examples(path)
        ex = examples[0]
        # Pair fields populated but label is unrecognised -> empty.
        self.assertTrue(ex.has_pair_candidates)
        self.assertEqual(ex.pair_correct_label, "")


# ---------------------------------------------------------------------------
# _anchor_candidates: pair anchors fire and skip the legacy anchors
# ---------------------------------------------------------------------------


class PairAnchorsTests(unittest.TestCase):
    def _make_pair_example(self) -> ExampleRecord:
        return ExampleRecord(
            example_id="ex_pair",
            source="medical_gpt41_answers_rl",
            source_id="src_1",
            dataset_subset="",
            conversation="",
            task_prompt="prompt",
            task_profile_id="general_instruction_following",
            pair_response_a="Response A text.",
            pair_response_b="Response B text.",
            pair_correct_label="a",
        )

    def test_pair_anchors_produced_with_pair_anchor_bucket(self) -> None:
        anchors = _anchor_candidates(self._make_pair_example())
        self.assertEqual(len(anchors), 2)
        self.assertEqual([a.source_label for a in anchors], ["pair_response_a", "pair_response_b"])
        self.assertTrue(all(a.quality_bucket == "pair_anchor" for a in anchors))
        self.assertTrue(all(a.origin_kind == "pair_anchor" for a in anchors))
        # Each anchor carries the label so the reporting layer can read it
        # straight off the candidate without walking back to the example.
        for a in anchors:
            self.assertEqual(a.metadata.get("pair_correct_label"), "a")

    def test_pair_anchors_skip_legacy_gold_and_augmented_anchors(self) -> None:
        ex = self._make_pair_example()
        # Pretend the loader also populated the legacy anchors (e.g. because a
        # stale --reference-field response was passed). The pair branch must win.
        ex.reference_note = "Legacy gold."
        ex.augmented_note = "Legacy augmented."
        ex.reference_artifact = "Legacy gold."
        ex.augmented_artifact = "Legacy augmented."

        anchors = _anchor_candidates(ex)
        self.assertEqual(len(anchors), 2)
        self.assertEqual({a.source_label for a in anchors}, {"pair_response_a", "pair_response_b"})

    def test_legacy_anchors_unchanged_when_no_pair(self) -> None:
        ex = ExampleRecord(
            example_id="ex_legacy",
            source="medical_o1_subset_b",
            source_id="src_2",
            dataset_subset="",
            conversation="",
            task_prompt="prompt",
            task_profile_id="general_instruction_following",
            reference_note="The gold answer.",
        )
        anchors = _anchor_candidates(ex)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].source_label, "reference_note")
        self.assertEqual(anchors[0].quality_bucket, "gold_like")


# ---------------------------------------------------------------------------
# _pair_preference_outcome
# ---------------------------------------------------------------------------


def _example_artifact_with_pair(label: str) -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "candidate_id": "ex_1__pair_a",
                "source_label": "pair_response_a",
                "quality_bucket": "pair_anchor",
                "metadata": {"pair_correct_label": label},
            },
            {
                "candidate_id": "ex_1__pair_b",
                "source_label": "pair_response_b",
                "quality_bucket": "pair_anchor",
                "metadata": {"pair_correct_label": label},
            },
            {
                "candidate_id": "ex_1__generated_0",
                "source_label": "generated_direct",
                "quality_bucket": "frontier_generated",
                "metadata": {},
            },
        ]
    }


def _ranking(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return rows


class PairPreferenceOutcomeTests(unittest.TestCase):
    def test_returns_false_none_when_pair_anchors_missing(self) -> None:
        artifact = {"candidates": [{"candidate_id": "x", "source_label": "reference_note"}]}
        evaluable, correct = _pair_preference_outcome(artifact, _ranking([]))
        self.assertFalse(evaluable)
        self.assertIsNone(correct)

    def test_returns_false_none_when_label_invalid(self) -> None:
        artifact = _example_artifact_with_pair(label="")
        ranking = _ranking([
            {"candidate_id": "ex_1__pair_a", "rank": 1, "score": 0.9},
            {"candidate_id": "ex_1__pair_b", "rank": 2, "score": 0.4},
        ])
        evaluable, correct = _pair_preference_outcome(artifact, ranking)
        self.assertFalse(evaluable)
        self.assertIsNone(correct)

    def test_pipeline_correct_when_higher_score_matches_label_a(self) -> None:
        artifact = _example_artifact_with_pair(label="a")
        ranking = _ranking([
            {"candidate_id": "ex_1__pair_a", "rank": 1, "score": 0.9},
            {"candidate_id": "ex_1__pair_b", "rank": 2, "score": 0.4},
        ])
        evaluable, correct = _pair_preference_outcome(artifact, ranking)
        self.assertTrue(evaluable)
        self.assertTrue(correct)

    def test_pipeline_correct_when_higher_score_matches_label_b(self) -> None:
        artifact = _example_artifact_with_pair(label="b")
        ranking = _ranking([
            {"candidate_id": "ex_1__pair_a", "rank": 3, "score": 0.2},
            {"candidate_id": "ex_1__pair_b", "rank": 1, "score": 0.8},
        ])
        evaluable, correct = _pair_preference_outcome(artifact, ranking)
        self.assertTrue(evaluable)
        self.assertTrue(correct)

    def test_pipeline_wrong_when_higher_score_misses_label(self) -> None:
        artifact = _example_artifact_with_pair(label="b")
        ranking = _ranking([
            {"candidate_id": "ex_1__pair_a", "rank": 1, "score": 0.9},
            {"candidate_id": "ex_1__pair_b", "rank": 2, "score": 0.4},
        ])
        evaluable, correct = _pair_preference_outcome(artifact, ranking)
        self.assertTrue(evaluable)
        self.assertFalse(correct)

    def test_pipeline_tie_counted_as_wrong(self) -> None:
        artifact = _example_artifact_with_pair(label="a")
        ranking = _ranking([
            {"candidate_id": "ex_1__pair_a", "rank": 1, "score": 0.5},
            {"candidate_id": "ex_1__pair_b", "rank": 1, "score": 0.5},
        ])
        evaluable, correct = _pair_preference_outcome(artifact, ranking)
        self.assertTrue(evaluable)
        self.assertFalse(correct)

    def test_returns_false_none_when_anchor_missing_from_ranking(self) -> None:
        artifact = _example_artifact_with_pair(label="a")
        ranking = _ranking([
            {"candidate_id": "ex_1__pair_a", "rank": 1, "score": 0.9},
        ])
        evaluable, correct = _pair_preference_outcome(artifact, ranking)
        self.assertFalse(evaluable)
        self.assertIsNone(correct)


# ---------------------------------------------------------------------------
# aggregate_method_metrics: pair_preference rolls up across examples
# ---------------------------------------------------------------------------


def _build_example_artifact(label: str, a_rank: int, b_rank: int) -> Dict[str, Any]:
    return {
        "example": {"example_id": "ex"},
        "candidates": _example_artifact_with_pair(label)["candidates"],
        "methods": {
            "rrd_uniform": {
                "ranking": [
                    {"candidate_id": "ex_1__pair_a", "rank": a_rank, "score": float(10 - a_rank)},
                    {"candidate_id": "ex_1__pair_b", "rank": b_rank, "score": float(10 - b_rank)},
                ],
                "rubrics": [],
            }
        },
    }


class AggregateMethodMetricsPairTests(unittest.TestCase):
    def test_aggregator_computes_pair_preference_accuracy(self) -> None:
        artifacts = [
            _build_example_artifact(label="a", a_rank=1, b_rank=2),  # correct
            _build_example_artifact(label="b", a_rank=1, b_rank=2),  # wrong
            _build_example_artifact(label="a", a_rank=2, b_rank=1),  # wrong
            _build_example_artifact(label="b", a_rank=2, b_rank=1),  # correct
        ]
        rows = aggregate_method_metrics(artifacts)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["method"], "rrd_uniform")
        self.assertEqual(row["pair_preference_evaluable"], 4)
        self.assertEqual(row["pair_preference_correct"], 2)
        self.assertAlmostEqual(row["pair_preference_accuracy"], 0.5)

    def test_aggregator_pair_preference_zero_when_no_pair_examples(self) -> None:
        # Non-pair example: candidates lack pair anchors.
        artifact = {
            "example": {"example_id": "ex"},
            "candidates": [
                {
                    "candidate_id": "x",
                    "source_label": "reference_note",
                    "quality_bucket": "gold_like",
                    "metadata": {},
                }
            ],
            "methods": {
                "rrd_uniform": {
                    "ranking": [{"candidate_id": "x", "rank": 1, "score": 1.0}],
                    "rubrics": [],
                }
            },
        }
        rows = aggregate_method_metrics([artifact])
        self.assertEqual(rows[0]["pair_preference_evaluable"], 0)
        self.assertEqual(rows[0]["pair_preference_accuracy"], 0.0)


# ---------------------------------------------------------------------------
# Sharding under --split all
# ---------------------------------------------------------------------------


class SplitAllShardingTests(unittest.TestCase):
    def _rows(self, n: int) -> List[Dict[str, Any]]:
        return [{"id": f"row_{i}"} for i in range(n)]

    def test_split_all_no_shards_returns_all_rows(self) -> None:
        rows = self._rows(12)
        out = _apply_split_and_shard(
            rows,
            split="all",
            train_size=0,
            val_size=0,
            num_shards=1,
            shard_index=0,
        )
        self.assertEqual(out, rows)

    def test_split_all_evenly_divides_rows_across_shards(self) -> None:
        rows = self._rows(12)
        shards = [
            _apply_split_and_shard(
                rows,
                split="all",
                train_size=0,
                val_size=0,
                num_shards=4,
                shard_index=i,
            )
            for i in range(4)
        ]
        self.assertEqual([len(s) for s in shards], [3, 3, 3, 3])
        # Concatenating shards in order reproduces the original rows.
        flat: List[Dict[str, Any]] = []
        for s in shards:
            flat.extend(s)
        self.assertEqual(flat, rows)

    def test_split_all_indivisible_count_raises(self) -> None:
        rows = self._rows(13)
        with self.assertRaises(ValueError) as cm:
            _apply_split_and_shard(
                rows,
                split="all",
                train_size=0,
                val_size=0,
                num_shards=4,
                shard_index=0,
            )
        self.assertIn("evenly divisible", str(cm.exception))

    def test_split_all_invalid_shard_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            _apply_split_and_shard(
                self._rows(12),
                split="all",
                train_size=0,
                val_size=0,
                num_shards=4,
                shard_index=4,
            )

    def test_split_train_sharding_still_works(self) -> None:
        # Regression: phase 5's refactor preserved the existing train sharding.
        rows = self._rows(12)
        out = _apply_split_and_shard(
            rows,
            split="train",
            train_size=9,
            val_size=0,
            num_shards=3,
            shard_index=1,
        )
        self.assertEqual(len(out), 3)
        self.assertEqual([r["id"] for r in out], ["row_3", "row_4", "row_5"])

    def test_split_val_unchanged_by_phase5(self) -> None:
        rows = self._rows(12)
        out = _apply_split_and_shard(
            rows,
            split="val",
            train_size=8,
            val_size=4,
            num_shards=1,
            shard_index=0,
        )
        self.assertEqual([r["id"] for r in out], ["row_8", "row_9", "row_10", "row_11"])


if __name__ == "__main__":
    unittest.main()
