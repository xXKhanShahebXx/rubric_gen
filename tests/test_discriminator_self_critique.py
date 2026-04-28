from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

from rubric_gen.compiled.judgebench_eval import _run_pair_discriminator_once
from rubric_gen.compiled.judgebench_eval import JudgeBenchRouteDecision
from rubric_gen.storage import JsonlCache
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec


def _make_route() -> JudgeBenchRouteDecision:
    return JudgeBenchRouteDecision(
        pair_id="p",
        source="livebench-reasoning",
        source_family="livebench-reasoning",
        task_profile_id="general_instruction_following",
        task_family_id="general_instruction_following",
        artifact_kind="response",
        route_kind="source_family",
        bootstrap_used=False,
    )


def _make_example() -> ExampleRecord:
    return ExampleRecord(
        example_id="ex_p",
        source="src",
        source_id="sid",
        dataset_subset="x",
        conversation="",
        task_prompt="prompt",
        task_profile_id="general_instruction_following",
        task_family_id="general_instruction_following",
        artifact_kind="response",
        reference_artifact="",
        metadata={"pair_id": "p"},
    )


def _make_candidate(text: str, position: str) -> CandidateNote:
    return CandidateNote(
        candidate_id=f"cand_{position}",
        example_id="ex_p",
        text=text,
        source_label=f"r_{position}",
        quality_bucket="pair_candidate",
        origin_kind="judgebench_pair",
        metadata={"pair_position": position},
        artifact_kind="response",
        task_profile_id="general_instruction_following",
        task_family_id="general_instruction_following",
    )


def _make_model() -> ModelSpec:
    return ModelSpec(
        alias="t",
        provider="openai",
        model="gpt-4o-2024-05-13",
        api_key_env="K",
    )


class DiscriminatorSelfCritiqueTests(unittest.TestCase):
    def test_self_critique_revises_initial_decision(self) -> None:
        router = MagicMock()
        responses: List[SimpleNamespace] = [
            SimpleNamespace(
                raw_text='{"decision": "X>Y", "confidence": "medium", "distinguishing_behavior": "first"}',
                text='{"decision": "X>Y", "confidence": "medium", "distinguishing_behavior": "first"}',
                metadata={},
            ),
            SimpleNamespace(
                raw_text='{"decision": "Y>X", "confidence": "high", "distinguishing_behavior": "revised after critique"}',
                text='{"decision": "Y>X", "confidence": "high", "distinguishing_behavior": "revised after critique"}',
                metadata={},
            ),
        ]
        router.generate.side_effect = responses
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "c.jsonl", enabled=False)
            result = _run_pair_discriminator_once(
                example_record=_make_example(),
                left_candidate=_make_candidate("A side text", "A"),
                right_candidate=_make_candidate("B side text", "B"),
                left_pair_position="A",
                route_decision=_make_route(),
                calibration_guidance="",
                discovery_model=_make_model(),
                router=router,
                cache=cache,
                left_id="X",
                right_id="Y",
                order_label="AB",
                self_critique_enabled=True,
            )
        self.assertEqual(router.generate.call_count, 2)
        self.assertEqual(result["decision"], "B>A")
        self.assertEqual(result["confidence"], "high")
        self.assertIsNotNone(result.get("self_critique"))

    def test_self_critique_keeps_decision_when_revised_matches(self) -> None:
        router = MagicMock()
        responses: List[SimpleNamespace] = [
            SimpleNamespace(
                raw_text='{"decision": "X>Y", "confidence": "high", "distinguishing_behavior": "clear"}',
                text='{"decision": "X>Y", "confidence": "high", "distinguishing_behavior": "clear"}',
                metadata={},
            ),
            SimpleNamespace(
                raw_text='{"decision": "X>Y", "confidence": "high", "distinguishing_behavior": "still clear"}',
                text='{"decision": "X>Y", "confidence": "high", "distinguishing_behavior": "still clear"}',
                metadata={},
            ),
        ]
        router.generate.side_effect = responses
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "c.jsonl", enabled=False)
            result = _run_pair_discriminator_once(
                example_record=_make_example(),
                left_candidate=_make_candidate("A side text", "A"),
                right_candidate=_make_candidate("B side text", "B"),
                left_pair_position="A",
                route_decision=_make_route(),
                calibration_guidance="",
                discovery_model=_make_model(),
                router=router,
                cache=cache,
                left_id="X",
                right_id="Y",
                order_label="AB",
                self_critique_enabled=True,
            )
        self.assertEqual(result["decision"], "A>B")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(router.generate.call_count, 2)

    def test_self_critique_disabled_runs_single_call(self) -> None:
        router = MagicMock()
        router.generate.return_value = SimpleNamespace(
            raw_text='{"decision": "X>Y", "confidence": "medium", "distinguishing_behavior": "ok"}',
            text='{"decision": "X>Y", "confidence": "medium", "distinguishing_behavior": "ok"}',
            metadata={},
        )
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "c.jsonl", enabled=False)
            result = _run_pair_discriminator_once(
                example_record=_make_example(),
                left_candidate=_make_candidate("A side text", "A"),
                right_candidate=_make_candidate("B side text", "B"),
                left_pair_position="A",
                route_decision=_make_route(),
                calibration_guidance="",
                discovery_model=_make_model(),
                router=router,
                cache=cache,
                left_id="X",
                right_id="Y",
                order_label="AB",
                self_critique_enabled=False,
            )
        self.assertEqual(router.generate.call_count, 1)
        self.assertEqual(result["decision"], "A>B")
        self.assertIsNone(result.get("self_critique"))


if __name__ == "__main__":
    unittest.main()
