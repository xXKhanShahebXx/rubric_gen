from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

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
from rubric_gen.types import ExampleRecord


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
            rows = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
            )
            self.assertEqual(rows, [])

    def test_library_v1_retrieval_profile_loads_library_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 0,
                "retrieval_profile": _RETRIEVAL_PROFILE_LIBRARY_V1,
            }
            rows = _maybe_library_rows_for_example(
                example=self._make_example("livebench-reasoning"),
                example_record=self._make_record(),
                policy=policy,
            )
            criterion_ids = {row["rubric_library_criterion_id"] for row in rows}
            self.assertEqual(criterion_ids, {"lib_reason_1", "lib_reason_2"})
            self.assertTrue(
                all(row.get("pair_ids") == ["pair_xyz"] for row in rows)
            )

    def test_library_rows_respect_top_k(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _make_library(Path(tmp))
            policy = {
                "rubric_library_path": str(path),
                "library_retrieval_top_k": 1,
                "retrieval_profile": "off",
            }
            rows = _maybe_library_rows_for_example(
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
            rows = _maybe_library_rows_for_example(
                example=self._make_example("mmlu-pro"),
                example_record=self._make_record(),
                policy=policy,
            )
            criterion_ids = {row["rubric_library_criterion_id"] for row in rows}
            self.assertIn("lib_mmlu_1", criterion_ids)
            self.assertNotIn("lib_reason_1", criterion_ids)


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
