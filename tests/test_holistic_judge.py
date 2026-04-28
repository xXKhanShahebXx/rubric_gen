from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

from rubric_gen.compiled.holistic_judge import (
    HOLISTIC_JUDGE_PROMPT_VERSION,
    _aggregate_holistic_attempts,
    apply_holistic_judge_to_scoring,
    run_holistic_pair_judge,
)
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec, RubricCriterion


def _make_example() -> ExampleRecord:
    return ExampleRecord(
        example_id="ex_1",
        source="livebench-reasoning",
        source_id="src_1",
        dataset_subset="livebench-reasoning",
        conversation="",
        task_prompt="Solve the puzzle",
        task_profile_id="general_instruction_following",
        task_family_id="general_instruction_following",
        artifact_kind="response",
        reference_artifact="",
        metadata={"pair_id": "pair_1"},
    )


def _make_candidate(text: str, position: str) -> CandidateNote:
    return CandidateNote(
        candidate_id=f"cand_{position}",
        example_id="ex_1",
        text=text,
        source_label=f"response_{position}",
        quality_bucket="pair_candidate",
        origin_kind="judgebench_pair",
        metadata={"pair_position": position},
        artifact_kind="response",
        task_profile_id="general_instruction_following",
        task_family_id="general_instruction_following",
    )


def _make_rubric(rubric_id: str, text: str) -> RubricCriterion:
    return RubricCriterion(
        rubric_id=rubric_id,
        text=text,
        source_stage="test",
        depth=0,
        round_index=0,
    )


def _make_model_spec() -> ModelSpec:
    return ModelSpec(
        alias="test_gpt4o",
        provider="openai",
        model="gpt-4o-2024-05-13",
        api_key_env="OPENAI_API_KEY",
    )


class AggregateHolisticTests(unittest.TestCase):
    def test_agreeing_attempts_return_consistent_decision(self) -> None:
        attempts = [
            {"order": "AB", "decision": "A>B", "confidence": "high", "distinguishing_behavior": "A is clearer"},
            {"order": "BA", "decision": "A>B", "confidence": "medium", "distinguishing_behavior": ""},
        ]
        result = _aggregate_holistic_attempts(attempts)
        self.assertEqual(result["decision"], "A>B")
        self.assertIn(result["confidence"], {"medium", "high"})
        self.assertTrue(result["order_consistent"])

    def test_order_disagreement_returns_inconsistent(self) -> None:
        attempts = [
            {"order": "AB", "decision": "A>B", "confidence": "medium"},
            {"order": "BA", "decision": "B>A", "confidence": "medium"},
        ]
        result = _aggregate_holistic_attempts(attempts)
        self.assertFalse(result["order_consistent"])
        self.assertEqual(result["decision"], "")

    def test_all_ties_return_tie(self) -> None:
        attempts = [
            {"order": "AB", "decision": "A=B", "confidence": "low"},
            {"order": "BA", "decision": "A=B", "confidence": "low"},
        ]
        result = _aggregate_holistic_attempts(attempts)
        self.assertEqual(result["decision"], "A=B")


class ApplyHolisticJudgeToScoringTests(unittest.TestCase):
    def _base_scoring(self, *, decision: str, margin: float, whitening_unstable: bool = False) -> Dict[str, Any]:
        half = 0.5
        score_a = half + margin / 2 if decision == "A>B" else (half - margin / 2 if decision == "B>A" else half)
        score_b = half - (score_a - half)
        return {
            "whitened_uniform": {
                "result": {
                    "decision": decision,
                    "score_A": score_a,
                    "score_B": score_b,
                    "whitening_unstable": whitening_unstable,
                    "decision_policy": "whitened_uniform",
                }
            },
            "uniform": {
                "result": {
                    "decision": decision,
                    "score_A": score_a,
                    "score_B": score_b,
                }
            },
        }

    def test_holistic_does_not_override_high_margin_agreeing_decision(self) -> None:
        scoring = self._base_scoring(decision="A>B", margin=0.2)
        holistic = {"decision": "A>B", "confidence": "high", "distinguishing_behavior": ""}
        updated = apply_holistic_judge_to_scoring(scoring=scoring, holistic=holistic)
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "A>B")
        self.assertEqual(
            updated["whitened_uniform"]["result"].get("decision_policy"),
            "whitened_uniform",
        )

    def test_holistic_lifts_tie_when_high_confidence(self) -> None:
        scoring = self._base_scoring(decision="A=B", margin=0.0)
        scoring["whitened_uniform"]["result"]["score_A"] = 0.5
        scoring["whitened_uniform"]["result"]["score_B"] = 0.5
        holistic = {
            "decision": "A>B",
            "confidence": "high",
            "distinguishing_behavior": "",
            "order_consistent": True,
        }
        updated = apply_holistic_judge_to_scoring(scoring=scoring, holistic=holistic)
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "A>B")
        self.assertEqual(updated["whitened_uniform"]["result"]["decision_policy"], "holistic_judge")

    def test_holistic_does_not_override_low_margin_with_default_strict_gate(self) -> None:
        # New strict defaults (margin <= 0.005, HIGH confidence) -- this case stays as WU.
        scoring = self._base_scoring(decision="A>B", margin=0.01)
        holistic = {
            "decision": "B>A",
            "confidence": "medium",
            "distinguishing_behavior": "",
            "order_consistent": True,
        }
        updated = apply_holistic_judge_to_scoring(scoring=scoring, holistic=holistic)
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "A>B")

    def test_holistic_legacy_loose_gate_overrides_when_explicitly_opted_in(self) -> None:
        # The legacy v2.0 (loose) gate still works when opt-in flags are set.
        scoring = self._base_scoring(decision="A>B", margin=0.01)
        holistic = {
            "decision": "B>A",
            "confidence": "medium",
            "distinguishing_behavior": "",
            "order_consistent": True,
        }
        updated = apply_holistic_judge_to_scoring(
            scoring=scoring,
            holistic=holistic,
            low_margin_threshold=0.05,
            require_high_confidence=False,
        )
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "B>A")
        self.assertEqual(updated["whitened_uniform"]["result"]["decision_policy"], "holistic_judge")

    def test_holistic_overrides_true_tie_at_high_confidence(self) -> None:
        scoring = self._base_scoring(decision="A>B", margin=0.0)
        holistic = {
            "decision": "B>A",
            "confidence": "high",
            "distinguishing_behavior": "",
            "order_consistent": True,
        }
        updated = apply_holistic_judge_to_scoring(scoring=scoring, holistic=holistic)
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "B>A")
        self.assertEqual(updated["whitened_uniform"]["result"]["decision_policy"], "holistic_judge")

    def test_holistic_low_confidence_does_not_override(self) -> None:
        scoring = self._base_scoring(decision="A=B", margin=0.0)
        scoring["whitened_uniform"]["result"]["score_A"] = 0.5
        scoring["whitened_uniform"]["result"]["score_B"] = 0.5
        holistic = {
            "decision": "A>B",
            "confidence": "low",
            "distinguishing_behavior": "",
        }
        updated = apply_holistic_judge_to_scoring(scoring=scoring, holistic=holistic)
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "A=B")


class RunHolisticPairJudgeTests(unittest.TestCase):
    def test_missing_candidates_returns_empty(self) -> None:
        router = MagicMock()
        cache = None
        result = run_holistic_pair_judge(
            example_record=_make_example(),
            rubrics=[_make_rubric("r1", "Final answer is correct")],
            pair_candidates=[_make_candidate("only A", "A")],
            task_profile_id="general_instruction_following",
            judge_model=_make_model_spec(),
            router=router,
            cache=cache,
        )
        self.assertEqual(result["decision"], "")
        router.generate.assert_not_called()

    def test_happy_path_runs_order_swap(self) -> None:
        router = MagicMock()
        responses: List[MagicMock] = []
        for raw_decision, confidence in (("X>Y", "medium"), ("Y>X", "medium")):
            mock = MagicMock()
            mock.raw_text = (
                "{"
                f'"decision": "{raw_decision}", '
                f'"confidence": "{confidence}", '
                f'"distinguishing_behavior": "clear preference"'
                "}"
            )
            mock.text = mock.raw_text
            responses.append(mock)
        router.generate.side_effect = responses

        result = run_holistic_pair_judge(
            example_record=_make_example(),
            rubrics=[_make_rubric("r1", "Final answer is correct")],
            pair_candidates=[
                _make_candidate("candidate A text", "A"),
                _make_candidate("candidate B text", "B"),
            ],
            task_profile_id="general_instruction_following",
            judge_model=_make_model_spec(),
            router=router,
            cache=None,
        )
        self.assertEqual(router.generate.call_count, 2)
        self.assertEqual(result["decision"], "A>B")
        self.assertTrue(result["order_consistent"])


if __name__ == "__main__":
    unittest.main()
