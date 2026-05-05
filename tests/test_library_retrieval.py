from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from rubric_gen.compiled import judgebench_eval as jbe
from rubric_gen.compiled.judgebench_eval import (
    _RETRIEVAL_PROFILE_FAMILY_QUESTION_V1,
    _RETRIEVAL_PROFILE_LIBRARY_V1,
    _RETRIEVAL_PROFILE_LIBRARY_V1_PLUS_FAMILY_V1,
    _build_retrieval_guidance,
    _maybe_library_rows_for_example,
    _RUBRIC_LIBRARY_CACHE,
)
from rubric_gen.compiled.rubric_library import (
    RubricLibrary,
    RubricLibraryCriterion,
    save_rubric_library,
)
from rubric_gen.storage import JsonlCache
from rubric_gen.types import ExampleRecord, LLMTextResponse


def _make_library(tmp: Path) -> Path:
    library = RubricLibrary(
        version="v1",
        criteria=[
            RubricLibraryCriterion(
                criterion_id="lib_reason_1",
                dimension="assignment_completeness",
                label="Assignment is complete",
                requirement="Every slot has a committed value",
                severity_tier="hard_gate",
                applicable_families=("livebench-reasoning",),
                source_tag="seed",
                focus_kind="assignment_completeness",
                direction_evidence=2,
            ),
            RubricLibraryCriterion(
                criterion_id="lib_reason_2",
                dimension="conclusion_grounded",
                label="Conclusion follows",
                requirement="Final answer matches the derived solution",
                severity_tier="high",
                applicable_families=("livebench-reasoning",),
                source_tag="seed",
                direction_evidence=1,
            ),
            RubricLibraryCriterion(
                criterion_id="lib_mmlu_1",
                dimension="final_answer_correctness",
                label="Selected option matches",
                requirement="Final option matches the correct choice",
                severity_tier="hard_gate",
                applicable_families=("mmlu-pro",),
                source_tag="seed",
                direction_evidence=2,
            ),
        ],
    )
    path = tmp / "library.json"
    save_rubric_library(library, path)
    return path


class LibraryRowsForExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        _RUBRIC_LIBRARY_CACHE.clear()

    def _make_example(self, source_family: str) -> Any:
        class _FakeExample:
            pass

        example = _FakeExample()
        example.pair_id = "pair_xyz"
        example.source_family = source_family
        # Provide the prompt + response fields the relevance filter consumes; harmless
        # for tests that don't enable the filter.
        example.question = "What is the capital of France?"
        example.response_A = "Paris is the capital of France."
        example.response_B = "I think it's Berlin."
        return example

    def _make_record(self) -> ExampleRecord:
        return ExampleRecord(
            example_id="ex_1",
            source="livebench-reasoning",
            source_id="src_1",
            dataset_subset="livebench-reasoning",
            conversation="",
            task_prompt="prompt",
            task_profile_id="general_instruction_following",
        )

    def test_no_library_returns_no_rows_when_top_k_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 0,
                "retrieval_profile": "off",
            }
            rows, debug = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
            )
            self.assertEqual(rows, [])
            self.assertEqual(debug, {})

    def test_library_v1_retrieval_profile_loads_library_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 0,
                "retrieval_profile": _RETRIEVAL_PROFILE_LIBRARY_V1,
            }
            rows, debug = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
            )
            criterion_ids = {row["rubric_library_criterion_id"] for row in rows}
            self.assertEqual(criterion_ids, {"lib_reason_1", "lib_reason_2"})
            self.assertTrue(
                all(row.get("pair_ids") == ["pair_xyz"] for row in rows)
            )
            # Filter is disabled by default -> empty debug payload.
            self.assertEqual(debug, {})

    def test_library_rows_respect_top_k(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 1,
                "retrieval_profile": "off",
            }
            rows, _debug = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
            )
            self.assertLessEqual(len(rows), 4)
            self.assertGreaterEqual(len(rows), 1)

    def test_library_rows_filter_by_family(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 0,
                "retrieval_profile": _RETRIEVAL_PROFILE_LIBRARY_V1,
            }
            rows, _debug = _maybe_library_rows_for_example(
                example=self._make_example("mmlu-pro"),
                example_record=self._make_record(),
                policy=policy,
            )
            criterion_ids = {row["rubric_library_criterion_id"] for row in rows}
            self.assertIn("lib_mmlu_1", criterion_ids)
            self.assertNotIn("lib_reason_1", criterion_ids)


class _ScriptedRouter:
    """Minimal LLMRouter stand-in for relevance-filter integration tests."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        spec,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMTextResponse:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return LLMTextResponse(
            text=self.response_text,
            raw_text=self.response_text,
            latency_s=0.0,
            model_alias=spec.alias,
            provider=spec.provider,
            metadata={"model": spec.model},
        )


class LibraryRelevanceFilterIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _RUBRIC_LIBRARY_CACHE.clear()

    def _make_example(self, source_family: str) -> Any:
        class _FakeExample:
            pass

        example = _FakeExample()
        example.pair_id = "pair_xyz"
        example.source_family = source_family
        example.question = "Solve the assignment problem with the given constraints."
        example.response_A = "The valid assignment is X."
        example.response_B = "The valid assignment is Y."
        return example

    def _make_record(self) -> ExampleRecord:
        return ExampleRecord(
            example_id="ex_1",
            source="livebench-reasoning",
            source_id="src_1",
            dataset_subset="livebench-reasoning",
            conversation="",
            task_prompt="prompt",
            task_profile_id="general_instruction_following",
        )

    def test_filter_disabled_returns_all_criteria_with_empty_debug(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 0,
                "retrieval_profile": _RETRIEVAL_PROFILE_LIBRARY_V1,
                "library_relevance_filter_enabled": False,
            }
            router = _ScriptedRouter(response_text="should_not_be_called")
            rows, debug = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
                router=router,
                relevance_filter_cache=None,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(debug, {})
            self.assertEqual(router.calls, [])

    def test_filter_enabled_drops_irrelevant_criteria(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 0,
                "retrieval_profile": _RETRIEVAL_PROFILE_LIBRARY_V1,
                "library_relevance_filter_enabled": True,
                "library_relevance_filter_strictness": "conservative",
            }
            router = _ScriptedRouter(
                response_text=(
                    """{
                        "0": {"verdict": "APPLICABLE", "reason": "on-topic"},
                        "1": {"verdict": "IRRELEVANT", "reason": "off-topic"}
                    }"""
                )
            )
            rows, debug = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
                router=router,
                relevance_filter_cache=None,
            )

            # 1 of the 2 retrieved criteria survived the filter.
            self.assertEqual(len(rows), 1)
            self.assertTrue(debug["enabled"])
            self.assertEqual(debug["input_count"], 2)
            self.assertEqual(debug["kept_count"], 1)
            self.assertEqual(debug["dropped_count"], 1)
            self.assertEqual(len(router.calls), 1)

    def test_filter_with_cache_warms_to_avoid_second_call(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            library_path = _make_library(tmp_path)
            cache_path = tmp_path / "rubric_relevance_filter.jsonl"
            cache = JsonlCache(cache_path, enabled=True)
            policy = {
                "rubric_library_path": str(library_path),
                "library_retrieval_top_k": 0,
                "retrieval_profile": _RETRIEVAL_PROFILE_LIBRARY_V1,
                "library_relevance_filter_enabled": True,
            }
            router = _ScriptedRouter(
                response_text=(
                    """{
                        "0": {"verdict": "APPLICABLE", "reason": "on-topic"},
                        "1": {"verdict": "APPLICABLE", "reason": "on-topic"}
                    }"""
                )
            )

            first_rows, _ = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
                router=router,
                relevance_filter_cache=cache,
            )
            self.assertEqual(len(first_rows), 2)
            self.assertEqual(len(router.calls), 1)

            # Second call with the same cache should not hit the router again. Reload
            # the cache from disk to simulate a process restart.
            warm_cache = JsonlCache(cache_path, enabled=True)
            second_router = _ScriptedRouter(response_text="will_blow_up_if_called")
            second_rows, debug = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
                router=second_router,
                relevance_filter_cache=warm_cache,
            )
            self.assertEqual(len(second_rows), 2)
            self.assertEqual(second_router.calls, [])
            self.assertTrue(debug["batches"][0]["cache_hit"])


class LibraryRetrievalProfileBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        _RUBRIC_LIBRARY_CACHE.clear()

    def test_library_v1_profile_skips_exemplar_retrieval(self) -> None:
        example = type("E", (), {"source_family": "livebench-reasoning", "pair_id": "pair_1"})()
        guidance, hits, seed_rows = _build_retrieval_guidance(
            example=example,
            retrieval_examples=[],
            policy={"retrieval_profile": _RETRIEVAL_PROFILE_LIBRARY_V1, "retrieval_top_k": 2},
        )
        self.assertEqual(guidance, "")
        self.assertEqual(hits, [])
        self.assertEqual(seed_rows, [])


if __name__ == "__main__":
    unittest.main()
