from __future__ import annotations

import unittest

from rubric_gen.compiled.labeled_train_few_shot import (
    LabeledTrainExample,
    LabeledTrainFewShotIndex,
    build_few_shot_block,
    format_demonstrations,
)


def _make_examples() -> list[LabeledTrainExample]:
    return [
        LabeledTrainExample(
            pair_id="p_math_1",
            source="livebench-math/algebra",
            source_family="livebench-math",
            question="Solve the quadratic equation x^2 + 5x + 6 = 0.",
            response_a="x = -2 and x = -3.",
            response_b="x = 2 and x = 3.",
            label="A>B",
        ),
        LabeledTrainExample(
            pair_id="p_math_2",
            source="livebench-math/calculus",
            source_family="livebench-math",
            question="Find the derivative of x^3.",
            response_a="3x^2",
            response_b="x^2",
            label="A>B",
        ),
        LabeledTrainExample(
            pair_id="p_reason_1",
            source="livebench-reasoning/logic",
            source_family="livebench-reasoning",
            question="Three friends are seated in a row.",
            response_a="Alice is in the middle.",
            response_b="Bob is in the middle.",
            label="B>A",
        ),
    ]


class LabeledTrainFewShotIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = LabeledTrainFewShotIndex.from_examples(_make_examples())

    def test_top_k_returns_most_similar_for_quadratic_query(self) -> None:
        results = self.index.top_k("Solve x^2 + 7x + 12 = 0.", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][1].pair_id, "p_math_1")
        self.assertGreater(results[0][0], results[1][0])

    def test_top_k_filters_by_family(self) -> None:
        results = self.index.top_k(
            "Three friends sat at a table",
            top_k=2,
            same_family="livebench-reasoning",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1].pair_id, "p_reason_1")

    def test_empty_query_returns_empty(self) -> None:
        self.assertEqual(self.index.top_k("", top_k=2), [])

    def test_no_matches_returns_empty_list(self) -> None:
        results = self.index.top_k(
            "completely unrelated topic words zzz qqq",
            top_k=2,
            same_family="livecodebench",
        )
        self.assertEqual(results, [])


class FormatDemonstrationsTests(unittest.TestCase):
    def test_block_includes_label_and_truncates(self) -> None:
        examples = _make_examples()
        scored = [(0.95, examples[0])]
        block = format_demonstrations(scored)
        self.assertIn("Demonstration 1", block)
        self.assertIn("livebench-math", block)
        self.assertIn("Gold preference: X>Y", block)

    def test_total_cap_truncates_demonstrations(self) -> None:
        examples = _make_examples()
        scored = [(0.95, examples[0]), (0.9, examples[1]), (0.85, examples[2])]
        block = format_demonstrations(
            scored,
            max_chars_per_pair=1000,
            max_chars_total=300,
        )
        self.assertLessEqual(len(block), 800)


class BuildFewShotBlockTests(unittest.TestCase):
    def test_no_index_returns_empty_string(self) -> None:
        self.assertEqual(build_few_shot_block(index=None, query="x"), "")

    def test_zero_top_k_returns_empty_string(self) -> None:
        index = LabeledTrainFewShotIndex.from_examples(_make_examples())
        self.assertEqual(build_few_shot_block(index=index, query="x", top_k=0), "")

    def test_normal_call_returns_demonstrations(self) -> None:
        index = LabeledTrainFewShotIndex.from_examples(_make_examples())
        block = build_few_shot_block(
            index=index,
            query="Solve quadratic",
            top_k=1,
            same_family="livebench-math",
        )
        self.assertIn("Demonstration 1", block)
        self.assertIn("livebench-math", block)


if __name__ == "__main__":
    unittest.main()
