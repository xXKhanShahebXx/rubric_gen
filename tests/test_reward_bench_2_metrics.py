from __future__ import annotations

import unittest

from rubric_gen.compiled.reward_bench_2_metrics import aggregate_pair_artifacts


def _make_artifact(
    *,
    pair_id: str,
    item_id: str,
    subset: str,
    decision: str,
    decision_source: str = "discriminator",
    confidence: str = "low",
) -> dict:
    return {
        "pair": {
            "pair_id": pair_id,
            "label": "A>B",
            "metadata": {
                "reward_bench_2": {
                    "item_id": item_id,
                    "subset": subset,
                    "rejected_index": 0,
                    "chosen_index": 0,
                    "num_correct": 1,
                    "num_incorrect": 3,
                    "total_completions": 4,
                }
            },
        },
        "scoring": {
            "whitened_uniform": {"result": {"decision": decision}},
            "pair_verifier": {
                "decision_source": decision_source,
                "confidence": confidence,
            },
        },
    }


class AggregateBestOf4Tests(unittest.TestCase):
    def test_all_three_correct_marks_item_correct(self) -> None:
        artifacts = [
            _make_artifact(pair_id=f"p{i}", item_id="x", subset="Factuality", decision="A>B")
            for i in range(3)
        ]
        summary = aggregate_pair_artifacts(artifacts)
        self.assertIn("Factuality", summary.subset_summaries)
        self.assertEqual(summary.subset_summaries["Factuality"].items_correct, 1)
        self.assertEqual(summary.subset_summaries["Factuality"].pairs_correct, 3)

    def test_two_correct_one_wrong_is_item_wrong(self) -> None:
        artifacts = [
            _make_artifact(pair_id="p0", item_id="x", subset="Factuality", decision="A>B"),
            _make_artifact(pair_id="p1", item_id="x", subset="Factuality", decision="A>B"),
            _make_artifact(pair_id="p2", item_id="x", subset="Factuality", decision="B>A"),
        ]
        summary = aggregate_pair_artifacts(artifacts)
        self.assertEqual(summary.subset_summaries["Factuality"].items_correct, 0)
        self.assertEqual(summary.subset_summaries["Factuality"].pairs_correct, 2)

    def test_multiple_items_aggregated(self) -> None:
        artifacts = []
        for item in ("a", "b", "c"):
            for i in range(3):
                artifacts.append(
                    _make_artifact(
                        pair_id=f"{item}_p{i}",
                        item_id=item,
                        subset="Factuality",
                        decision="A>B",
                    )
                )
        # one item gets a wrong pair
        artifacts[0] = _make_artifact(
            pair_id="a_p0", item_id="a", subset="Factuality", decision="B>A"
        )
        summary = aggregate_pair_artifacts(artifacts)
        ss = summary.subset_summaries["Factuality"]
        self.assertEqual(ss.item_count, 3)
        self.assertEqual(ss.items_correct, 2)

    def test_leaderboard_average_excludes_ties(self) -> None:
        artifacts = []
        for subset in ("Factuality", "Math"):
            for i in range(3):
                artifacts.append(
                    _make_artifact(
                        pair_id=f"{subset}_p{i}",
                        item_id=subset,
                        subset=subset,
                        decision="A>B",
                    )
                )
        artifacts.append(
            _make_artifact(pair_id="t_p0", item_id="t1", subset="Ties", decision="B>A")
        )
        summary = aggregate_pair_artifacts(artifacts)
        # Both non-Ties subsets at 100%, Ties at 0% -> leaderboard average = 100
        self.assertAlmostEqual(summary.leaderboard_average_pct, 100.0)

    def test_high_precision_pairs_counted(self) -> None:
        artifacts = [
            _make_artifact(
                pair_id="p0",
                item_id="x",
                subset="Factuality",
                decision="A>B",
                decision_source="mmlu_independent_answerer",
                confidence="high",
            ),
            _make_artifact(
                pair_id="p1", item_id="x", subset="Factuality", decision="A>B"
            ),
            _make_artifact(
                pair_id="p2", item_id="x", subset="Factuality", decision="A>B"
            ),
        ]
        summary = aggregate_pair_artifacts(artifacts)
        per_item = summary.per_item[0]
        self.assertEqual(per_item.pairs_high_confidence, 1)


class TiesScoreTests(unittest.TestCase):
    def test_ties_weighted_score_with_some_high_confidence(self) -> None:
        artifacts = []
        for i in range(4):
            artifacts.append(
                _make_artifact(
                    pair_id=f"t_p{i}",
                    item_id="t1",
                    subset="Ties",
                    decision="A>B",
                    decision_source="mmlu_independent_answerer" if i < 2 else "discriminator",
                    confidence="high" if i < 2 else "low",
                )
            )
        summary = aggregate_pair_artifacts(artifacts)
        self.assertIsNotNone(summary.ties_weighted_score)
        self.assertAlmostEqual(summary.ties_accuracy_term, 100.0)
        self.assertAlmostEqual(summary.ties_margin_term, 50.0)
        self.assertAlmostEqual(summary.ties_weighted_score, 75.0)


if __name__ == "__main__":
    unittest.main()
