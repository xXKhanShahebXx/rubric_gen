"""
Regression tests for ``evaluate_rubric_satisfaction`` robustness.

Background: a blind-350 final-eval crashed mid-run because GPT-4o returned
``<EVALUATION> UNKNOWN </EVALUATION>`` once. The original parser only accepted YES/NO and the
catch-all branch raised, killing the whole 350-example run on a single off-spec response.

These tests guarantee:
- ``UNKNOWN`` / ``CANNOT_ASSESS`` / ``UNCLEAR`` are mapped to ``False`` (criterion not satisfied)
  rather than ``None`` (which retries) or raising.
- The catch-all path (no attempt yields a parseable verdict) returns ``(False, metadata)`` with
  ``parse_error == "unparseable_evaluation"`` rather than raising, so a single weird response
  never fails the entire run.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, List

from rubric_gen.compiled.judgebench_eval import (
    _extract_yes_no,
    evaluate_rubric_satisfaction,
)
from rubric_gen.storage import JsonlCache
from rubric_gen.types import ModelSpec


class ExtractYesNoTests(unittest.TestCase):
    def test_yes_returns_true(self) -> None:
        self.assertTrue(_extract_yes_no("<EVALUATION> YES </EVALUATION>"))

    def test_no_returns_false(self) -> None:
        self.assertEqual(_extract_yes_no("<EVALUATION> NO </EVALUATION>"), False)

    def test_unknown_is_treated_as_false(self) -> None:
        self.assertEqual(_extract_yes_no("<EVALUATION> UNKNOWN </EVALUATION>"), False)

    def test_unclear_is_treated_as_false(self) -> None:
        self.assertEqual(_extract_yes_no("<EVALUATION> UNCLEAR </EVALUATION>"), False)

    def test_cannot_assess_is_treated_as_false(self) -> None:
        self.assertEqual(_extract_yes_no("<EVALUATION> CANNOT_ASSESS </EVALUATION>"), False)
        self.assertEqual(_extract_yes_no("<EVALUATION> CANNOT ASSESS </EVALUATION>"), False)

    def test_undetermined_is_treated_as_false(self) -> None:
        self.assertEqual(_extract_yes_no("<EVALUATION> UNDETERMINED </EVALUATION>"), False)

    def test_unrecognized_returns_none(self) -> None:
        self.assertIsNone(_extract_yes_no("garbage that isn't a verdict"))


class _UnparseableRouter:
    """Returns a payload that doesn't parse as YES / NO / UNKNOWN."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, model_spec: ModelSpec, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            text="OOPS not a verdict",
            raw_text="OOPS not a verdict",
            metadata={},
        )


class EvaluateRubricSatisfactionFallbackTests(unittest.TestCase):
    def test_unparseable_response_returns_false_with_parse_error(self) -> None:
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "cache.jsonl", enabled=False)
            judge_model = ModelSpec(
                alias="test",
                provider="openai",
                model="gpt-4o-2024-05-13",
                api_key_env="TEST_API_KEY",
            )
            router = _UnparseableRouter()
            verdict, meta = evaluate_rubric_satisfaction(
                response_text="some response",
                rubric_text="some criterion",
                judge_model=judge_model,
                cache=cache,
                router=router,
            )
        self.assertFalse(verdict)
        self.assertEqual(meta["parse_error"], "unparseable_evaluation")
        self.assertEqual(meta["fallback_verdict"], False)
        self.assertEqual(meta["judge_model"], "openai:gpt-4o-2024-05-13")
        self.assertGreater(router.calls, 0)


class _UnknownVerdictRouter:
    """Returns the off-spec verdict that triggered the original crash."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, model_spec: ModelSpec, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            text="<EVALUATION> UNKNOWN </EVALUATION>",
            raw_text="<EVALUATION> UNKNOWN </EVALUATION>",
            metadata={},
        )


class EvaluateRubricSatisfactionUnknownTests(unittest.TestCase):
    def test_unknown_verdict_short_circuits_to_false(self) -> None:
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "cache.jsonl", enabled=False)
            judge_model = ModelSpec(
                alias="test",
                provider="openai",
                model="gpt-4o-2024-05-13",
                api_key_env="TEST_API_KEY",
            )
            router = _UnknownVerdictRouter()
            verdict, meta = evaluate_rubric_satisfaction(
                response_text="some response",
                rubric_text="some criterion",
                judge_model=judge_model,
                cache=cache,
                router=router,
            )
        self.assertFalse(verdict)
        self.assertEqual(meta["attempt_index"], 0)
        self.assertEqual(router.calls, 1)


if __name__ == "__main__":
    unittest.main()
