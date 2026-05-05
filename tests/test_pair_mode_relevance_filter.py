"""Tests for the pair-mode relevance-filter candidate-text fix (Tier B1, v2 plan).

The bug: ``_retrieve_seed_rubrics_for_example`` previously passed only
``example.augmented_artifact or example.reference_artifact`` to the relevance
filter.  For pair-only datasets (e.g. ``medical_gpt5_b_regen_4k_rl.jsonl``)
the loader does not populate either field for ``reference_answer_a/b``, so
the filter saw an empty ``[""]`` string and dropped almost every retrieved
rubric as IRRELEVANT (19.8% zero-seed rate at the original 4k run).

The fix lives in ``RubricPipeline._collect_filter_candidate_texts``: when
``example.has_pair_candidates`` is true, both pair anchors are passed so the
filter sees real text.
"""

from __future__ import annotations

import unittest

from rubric_gen.pipeline import RubricPipeline
from rubric_gen.types import ExampleRecord


def _example(
    *,
    pair_a: str = "",
    pair_b: str = "",
    reference_artifact: str = "",
    augmented_artifact: str = "",
) -> ExampleRecord:
    return ExampleRecord(
        example_id="x",
        source="medical_o1",
        source_id="x",
        dataset_subset="",
        conversation="",
        task_prompt="Q?",
        reference_artifact=reference_artifact,
        augmented_artifact=augmented_artifact,
        pair_response_a=pair_a,
        pair_response_b=pair_b,
    )


class CollectFilterCandidateTextsTests(unittest.TestCase):
    def test_pair_mode_emits_both_anchors(self) -> None:
        ex = _example(pair_a="The answer is A.", pair_b="The answer is B.")
        out = RubricPipeline._collect_filter_candidate_texts(ex)
        self.assertEqual(out, ["The answer is A.", "The answer is B."])

    def test_pair_mode_skips_empty_b(self) -> None:
        # Half-populated pair (only A): falls back to single artifact path.
        ex = _example(pair_a="The answer is A.", pair_b="")
        # has_pair_candidates is False (B empty), so legacy behaviour.
        out = RubricPipeline._collect_filter_candidate_texts(ex)
        self.assertEqual(out, [""])

    def test_legacy_single_response_uses_augmented_artifact(self) -> None:
        ex = _example(augmented_artifact="A clinical note.")
        out = RubricPipeline._collect_filter_candidate_texts(ex)
        self.assertEqual(out, ["A clinical note."])

    def test_legacy_single_response_falls_back_to_reference_artifact(self) -> None:
        ex = _example(reference_artifact="A reference note.")
        out = RubricPipeline._collect_filter_candidate_texts(ex)
        self.assertEqual(out, ["A reference note."])

    def test_no_text_anywhere_returns_empty_string(self) -> None:
        ex = _example()
        out = RubricPipeline._collect_filter_candidate_texts(ex)
        self.assertEqual(out, [""])

    def test_pair_mode_strips_whitespace(self) -> None:
        ex = _example(pair_a="  alpha  \n", pair_b="\nbeta\t")
        out = RubricPipeline._collect_filter_candidate_texts(ex)
        self.assertEqual(out, ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
