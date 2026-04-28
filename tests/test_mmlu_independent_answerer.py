from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rubric_gen.compiled.mmlu_independent_answerer_verifier import (
    _extract_candidate_letter,
    _parse_solver_letter,
    evaluate_mmlu_independent_answerer,
    run_independent_answerer,
)
from rubric_gen.types import ModelSpec


def _make_model() -> ModelSpec:
    return ModelSpec(
        alias="solver",
        provider="openai",
        model="gpt-4o-2024-05-13",
        api_key_env="K",
    )


class ParseSolverLetterTests(unittest.TestCase):
    def test_final_answer_letter(self) -> None:
        self.assertEqual(_parse_solver_letter("FINAL_ANSWER: C"), "C")

    def test_final_answer_with_space(self) -> None:
        self.assertEqual(_parse_solver_letter("FINAL ANSWER: H"), "H")

    def test_lowercase_normalized(self) -> None:
        self.assertEqual(_parse_solver_letter("FINAL_ANSWER: c"), "C")

    def test_no_letter_returns_empty(self) -> None:
        self.assertEqual(_parse_solver_letter("Some text without final answer."), "")

    def test_unknown_returns_empty(self) -> None:
        self.assertEqual(_parse_solver_letter("FINAL_ANSWER: UNKNOWN"), "")


class ExtractCandidateLetterTests(unittest.TestCase):
    def test_uses_final_answer_marker(self) -> None:
        self.assertEqual(
            _extract_candidate_letter("Reasoning.\nFINAL_ANSWER: B"), "B"
        )

    def test_uses_repeated_letter_format(self) -> None:
        text = "After analysis, the answer is GGGGG"
        self.assertEqual(_extract_candidate_letter(text), "G")

    def test_uses_final_answer_phrase(self) -> None:
        text = "Therefore, the answer is (D)."
        self.assertEqual(_extract_candidate_letter(text), "D")

    def test_falls_back_to_last_parenthesized_letter(self) -> None:
        text = "Considered (A) but rejected, then (C) but rejected, finally (E) is correct."
        self.assertEqual(_extract_candidate_letter(text), "E")


class RunIndependentAnswererTests(unittest.TestCase):
    def test_returns_letter(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="Step 1.\nFINAL_ANSWER: C",
            text="Step 1.\nFINAL_ANSWER: C",
            metadata={},
        )
        result = run_independent_answerer(
            question="What is correct?",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertEqual(result.parsed_letter, "C")
        self.assertEqual(router.generate.call_count, 1)

    def test_majority_vote_picks_winner(self) -> None:
        router = MagicMock()
        responses = [
            SimpleNamespace(raw_text="FINAL_ANSWER: C", text="FINAL_ANSWER: C", metadata={}),
            SimpleNamespace(raw_text="FINAL_ANSWER: B", text="FINAL_ANSWER: B", metadata={}),
            SimpleNamespace(raw_text="FINAL_ANSWER: C", text="FINAL_ANSWER: C", metadata={}),
        ]
        router.generate.side_effect = responses
        result = run_independent_answerer(
            question="Q",
            model_spec=_make_model(),
            router=router,
            cache=None,
            samples=3,
            temperature=0.5,
        )
        self.assertEqual(result.parsed_letter, "C")


class EvaluateMMLUTests(unittest.TestCase):
    def test_a_matches_solver_high_confidence(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="FINAL_ANSWER: D", text="FINAL_ANSWER: D", metadata={}
        )
        outcome = evaluate_mmlu_independent_answerer(
            question="Q",
            response_a="Final answer: D",
            response_b="Final answer: B",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "A>B")
        self.assertEqual(outcome.confidence, "high")

    def test_b_matches_solver(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="FINAL_ANSWER: B", text="FINAL_ANSWER: B", metadata={}
        )
        outcome = evaluate_mmlu_independent_answerer(
            question="Q",
            response_a="Final answer: D",
            response_b="Final answer: B",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertEqual(outcome.recommended_decision, "B>A")

    def test_neither_matches_does_not_trigger(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="FINAL_ANSWER: A", text="FINAL_ANSWER: A", metadata={}
        )
        outcome = evaluate_mmlu_independent_answerer(
            question="Q",
            response_a="Final answer: D",
            response_b="Final answer: B",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.reason, "neither_matches_solver")

    def test_dual_consensus_disagree_does_not_trigger(self) -> None:
        router = MagicMock()
        responses = [
            SimpleNamespace(raw_text="FINAL_ANSWER: D", text="FINAL_ANSWER: D", metadata={}),
            SimpleNamespace(raw_text="FINAL_ANSWER: B", text="FINAL_ANSWER: B", metadata={}),
        ]
        router.generate.side_effect = responses
        secondary_spec = ModelSpec(
            alias="secondary", provider="anthropic", model="claude-sonnet-4-5", api_key_env="K"
        )
        outcome = evaluate_mmlu_independent_answerer(
            question="Q",
            response_a="Final answer: D",
            response_b="Final answer: B",
            model_spec=_make_model(),
            router=router,
            cache=None,
            secondary_model_spec=secondary_spec,
        )
        self.assertFalse(outcome.triggered)
        self.assertIn("primary_secondary_disagree", outcome.reason)

    def test_dual_consensus_agree_triggers(self) -> None:
        router = MagicMock()
        responses = [
            SimpleNamespace(raw_text="FINAL_ANSWER: D", text="FINAL_ANSWER: D", metadata={}),
            SimpleNamespace(raw_text="FINAL_ANSWER: D", text="FINAL_ANSWER: D", metadata={}),
        ]
        router.generate.side_effect = responses
        secondary_spec = ModelSpec(
            alias="secondary", provider="anthropic", model="claude-sonnet-4-5", api_key_env="K"
        )
        outcome = evaluate_mmlu_independent_answerer(
            question="Q",
            response_a="Final answer: D",
            response_b="Final answer: B",
            model_spec=_make_model(),
            router=router,
            cache=None,
            secondary_model_spec=secondary_spec,
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "A>B")

    def test_solver_unknown_does_not_trigger_initial(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="FINAL_ANSWER: UNKNOWN",
            text="FINAL_ANSWER: UNKNOWN",
            metadata={},
        )
        outcome = evaluate_mmlu_independent_answerer(
            question="Q",
            response_a="Final answer: D",
            response_b="Final answer: B",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertFalse(outcome.triggered)


if __name__ == "__main__":
    unittest.main()
