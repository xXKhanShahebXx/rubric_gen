from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

from rubric_gen.compiled.math_independent_solver_verifier import (
    _canonicalize_answer,
    _extract_candidate_final_value,
    _parse_final_answer_line,
    evaluate_math_independent_solver,
    run_independent_solver,
)
from rubric_gen.types import ModelSpec


def _make_model() -> ModelSpec:
    return ModelSpec(
        alias="solver",
        provider="openai",
        model="gpt-4o-2024-05-13",
        api_key_env="TEST_API_KEY",
    )


class CanonicalizeAnswerTests(unittest.TestCase):
    def test_simple_integer(self) -> None:
        self.assertEqual(_canonicalize_answer("42"), "42")

    def test_negative_integer(self) -> None:
        self.assertEqual(_canonicalize_answer("-7"), "-7")

    def test_float_normalizes_trailing_zero(self) -> None:
        self.assertEqual(_canonicalize_answer("5.0"), "5")
        self.assertEqual(_canonicalize_answer("5"), "5")

    def test_decimal_preserves_real_decimals(self) -> None:
        self.assertEqual(_canonicalize_answer("3.14"), "3.14")

    def test_fraction_in_lowest_terms(self) -> None:
        self.assertEqual(_canonicalize_answer("4/8"), "1/2")
        self.assertEqual(_canonicalize_answer("3/4"), "3/4")

    def test_repeated_letter_preserved(self) -> None:
        self.assertEqual(_canonicalize_answer("AAAAA"), "aaaaa")

    def test_option_letter(self) -> None:
        self.assertEqual(_canonicalize_answer("C"), "c")

    def test_text_with_embedded_number(self) -> None:
        self.assertEqual(_canonicalize_answer("the answer is 17"), "17")

    def test_empty_input(self) -> None:
        self.assertEqual(_canonicalize_answer(""), "")
        self.assertEqual(_canonicalize_answer("   "), "")

    def test_dollar_signs_stripped(self) -> None:
        self.assertEqual(_canonicalize_answer("$5050$"), "5050")


class ParseFinalAnswerLineTests(unittest.TestCase):
    def test_explicit_marker_with_colon(self) -> None:
        self.assertEqual(_parse_final_answer_line("Some work.\nFINAL_ANSWER: 42"), "42")

    def test_explicit_marker_with_underscore(self) -> None:
        self.assertEqual(_parse_final_answer_line("FINAL ANSWER: 17\n"), "17")

    def test_falls_back_to_last_line(self) -> None:
        self.assertEqual(_parse_final_answer_line("step 1\nstep 2\n42"), "42")


class ExtractCandidateFinalValueTests(unittest.TestCase):
    def test_uses_final_answer_marker(self) -> None:
        text = "Solution: 30 + 12 = 42.\nFinal answer: 42"
        self.assertEqual(_extract_candidate_final_value(text), "42")

    def test_uses_the_answer_is_pattern(self) -> None:
        text = "Working through the problem.\nThe answer is 17."
        self.assertEqual(_extract_candidate_final_value(text), "17")

    def test_falls_back_to_last_numeric_token(self) -> None:
        text = "step 1: 10\nstep 2: 20\nstep 3: 30"
        self.assertEqual(_extract_candidate_final_value(text), "30")

    def test_prefers_boxed_latex_over_text(self) -> None:
        text = "After computation, the answer is provided as $\\boxed{42}$."
        self.assertEqual(_extract_candidate_final_value(text), "42")

    def test_returns_last_boxed_when_multiple(self) -> None:
        text = "First try: $\\boxed{10}$. Actually, $\\boxed{42}$."
        self.assertEqual(_extract_candidate_final_value(text), "42")

    def test_finds_repeated_letter_in_freeform_text(self) -> None:
        text = "Repeat the answer five times as requested: ddddd"
        self.assertEqual(_extract_candidate_final_value(text), "ddddd")


class CanonicalizeAnswerEdgeTests(unittest.TestCase):
    def test_strips_leading_instruction_prefix(self) -> None:
        self.assertEqual(_canonicalize_answer("the answer is 42"), "42")
        self.assertEqual(_canonicalize_answer("therefore 17"), "17")
        self.assertEqual(_canonicalize_answer("repeat the answer five times as requested: ddddd"), "ddddd")

    def test_canonicalizes_boxed_latex(self) -> None:
        self.assertEqual(_canonicalize_answer("$\\boxed{42}$"), "42")
        self.assertEqual(_canonicalize_answer("\\boxed{3/4}"), "3/4")

    def test_finds_repeated_letter_within_phrase(self) -> None:
        self.assertEqual(_canonicalize_answer("the final answer is aaaaa repeated"), "aaaaa")


class RunIndependentSolverTests(unittest.TestCase):
    def test_returns_canonicalized_answer(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="Step 1: 1+1.\nStep 2: 2+0.\nFINAL_ANSWER: 2",
            text="Step 1: 1+1.\nStep 2: 2+0.\nFINAL_ANSWER: 2",
            metadata={},
        )
        result = run_independent_solver(
            question="What is 1 + 1?",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertEqual(result.parsed_answer, "2")
        self.assertEqual(result.canonical_answer, "2")
        self.assertEqual(result.parse_error, "")
        self.assertEqual(router.generate.call_count, 1)

    def test_solver_unknown_yields_parse_error(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="I am not sure.\nFINAL_ANSWER: UNKNOWN",
            text="I am not sure.\nFINAL_ANSWER: UNKNOWN",
            metadata={},
        )
        result = run_independent_solver(
            question="hard problem",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertEqual(result.parse_error, "solver_returned_unknown")


class EvaluateMathIndependentSolverTests(unittest.TestCase):
    def _router_returning(self, raw: str) -> MagicMock:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(raw_text=raw, text=raw, metadata={})
        return router

    def test_a_matches_solver_high_confidence(self) -> None:
        router = self._router_returning(
            "Step 1: combine.\nFINAL_ANSWER: 42"
        )
        outcome = evaluate_math_independent_solver(
            question="What is 6 * 7?",
            response_a="The product is 6 * 7.\nFinal answer: 42",
            response_b="The product is 6 * 7.\nFinal answer: 36",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "A>B")
        self.assertEqual(outcome.confidence, "high")
        self.assertEqual(outcome.solver_canonical_answer, "42")
        self.assertTrue(outcome.a_match)
        self.assertFalse(outcome.b_match)

    def test_b_matches_solver_high_confidence(self) -> None:
        router = self._router_returning("FINAL_ANSWER: 36")
        outcome = evaluate_math_independent_solver(
            question="problem",
            response_a="Final answer: 42",
            response_b="Final answer: 36",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "B>A")
        self.assertEqual(outcome.confidence, "high")

    def test_neither_matches_solver_does_not_trigger(self) -> None:
        router = self._router_returning("FINAL_ANSWER: 100")
        outcome = evaluate_math_independent_solver(
            question="problem",
            response_a="Final answer: 42",
            response_b="Final answer: 36",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "")
        self.assertEqual(outcome.reason, "neither_matches_solver")

    def test_both_match_solver_does_not_trigger(self) -> None:
        router = self._router_returning("FINAL_ANSWER: 42")
        outcome = evaluate_math_independent_solver(
            question="problem",
            response_a="Final answer: 42",
            response_b="Final answer: 42.0",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.reason, "both_match_solver")

    def test_solver_unknown_does_not_trigger(self) -> None:
        router = self._router_returning("FINAL_ANSWER: UNKNOWN")
        outcome = evaluate_math_independent_solver(
            question="problem",
            response_a="Final answer: 42",
            response_b="Final answer: 36",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "")


if __name__ == "__main__":
    unittest.main()
