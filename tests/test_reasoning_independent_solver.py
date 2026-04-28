from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rubric_gen.compiled.reasoning_independent_solver_verifier import (
    _canonicalize_answer,
    _extract_candidate_answer,
    _parse_solver_answer,
    evaluate_reasoning_independent_solver,
    run_reasoning_solver,
)
from rubric_gen.types import ModelSpec


def _make_model() -> ModelSpec:
    return ModelSpec(
        alias="solver",
        provider="anthropic",
        model="claude-opus-4-20250514",
        api_key_env="K",
    )


class CanonicalizeTests(unittest.TestCase):
    def test_strips_bold_and_lowers(self) -> None:
        self.assertEqual(_canonicalize_answer("**Yes, No, Yes**"), "yes, no, yes")

    def test_normalizes_comma_spacing(self) -> None:
        self.assertEqual(_canonicalize_answer("yes ,no, yes"), "yes, no, yes")

    def test_strips_trailing_punct(self) -> None:
        self.assertEqual(_canonicalize_answer("Square pyramid."), "square pyramid")


class ParseSolverAnswerTests(unittest.TestCase):
    def test_finds_final_answer(self) -> None:
        text = "Step 1.\nStep 2.\nFINAL_ANSWER: yes, no, yes"
        self.assertEqual(_parse_solver_answer(text), "yes, no, yes")

    def test_unknown_returns_empty(self) -> None:
        self.assertEqual(_parse_solver_answer("FINAL_ANSWER: UNKNOWN"), "")


class ExtractCandidateAnswerTests(unittest.TestCase):
    def test_uses_last_bold(self) -> None:
        text = "Reasoning bla bla.\n\n**yes, yes, no**"
        self.assertEqual(_extract_candidate_answer(text), "yes, yes, no")

    def test_skips_label_bold(self) -> None:
        text = "**Conclusion:** something happened.\n\n**square pyramid**"
        self.assertEqual(_extract_candidate_answer(text), "square pyramid")

    def test_falls_back_to_last_line(self) -> None:
        text = "Plain reasoning here.\nyes, no, yes"
        self.assertEqual(_extract_candidate_answer(text), "yes, no, yes")


class RunSolverTests(unittest.TestCase):
    def test_returns_canonical_answer(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="Step 1.\nFINAL_ANSWER: yes, no, yes",
            text="Step 1.\nFINAL_ANSWER: yes, no, yes",
            metadata={},
        )
        result = run_reasoning_solver(
            question="puzzle",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertEqual(result.canonical_answer, "yes, no, yes")


class EvaluateTests(unittest.TestCase):
    def test_a_matches_solver_high_confidence(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="FINAL_ANSWER: yes, no, yes",
            text="FINAL_ANSWER: yes, no, yes",
            metadata={},
        )
        outcome = evaluate_reasoning_independent_solver(
            question="puzzle",
            response_a="...** yes, no, yes**",
            response_b="...** no, no, yes**",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "A>B")

    def test_b_matches_solver(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="FINAL_ANSWER: square pyramid",
            text="FINAL_ANSWER: square pyramid",
            metadata={},
        )
        outcome = evaluate_reasoning_independent_solver(
            question="puzzle",
            response_a="**pentagon**",
            response_b="**square pyramid**",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertEqual(outcome.recommended_decision, "B>A")

    def test_neither_matches_does_not_trigger(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text="FINAL_ANSWER: tetrahedron",
            text="FINAL_ANSWER: tetrahedron",
            metadata={},
        )
        outcome = evaluate_reasoning_independent_solver(
            question="puzzle",
            response_a="**pentagon**",
            response_b="**square pyramid**",
            model_spec=_make_model(),
            router=router,
            cache=None,
        )
        self.assertFalse(outcome.triggered)


if __name__ == "__main__":
    unittest.main()
