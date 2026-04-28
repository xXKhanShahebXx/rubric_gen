from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rubric_gen.compiled.external_eval_slices import (
    ExternalSlicePair,
    _default_fallback_judge,
    default_slice_path,
    load_external_slice,
    score_external_slice,
    score_external_slices,
    write_seed_slice_if_missing,
)
from rubric_gen.compiled.rubric_library import RubricLibrary, RubricLibraryCriterion


def _library() -> RubricLibrary:
    return RubricLibrary(
        version="v1",
        criteria=[
            RubricLibraryCriterion(
                criterion_id="lib_gen",
                dimension="instruction_adherence",
                label="final answer clear",
                requirement="response states the final answer clearly",
                severity_tier="high",
                applicable_families=("generic",),
                source_tag="seed",
            )
        ],
    )


class LoadExternalSliceTests(unittest.TestCase):
    def test_missing_file_returns_empty_list(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.jsonl"
            self.assertEqual(load_external_slice(path), [])

    def test_valid_jsonl_loads_pairs(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "slice.jsonl"
            rows = [
                {
                    "pair_id": "p1",
                    "prompt": "test",
                    "response_a": "A strong answer",
                    "response_b": "weak",
                    "label": "A>B",
                    "source_family": "mmlu-pro",
                    "source": "test",
                },
                {
                    "pair_id": "p2",
                    "prompt": "test",
                    "response_a": "weak",
                    "response_b": "Strong",
                    "label": "B>A",
                    "source_family": "livebench-reasoning",
                    "source": "test",
                },
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            pairs = load_external_slice(path)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0].label, "A>B")
            self.assertEqual(pairs[1].source_family, "livebench-reasoning")

    def test_invalid_labels_fallback_to_default(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "slice.jsonl"
            row = {
                "pair_id": "p1",
                "prompt": "test",
                "response_a": "a",
                "response_b": "b",
                "label": "???",
                "source_family": "unknown",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            pairs = load_external_slice(path)
            self.assertEqual(pairs[0].label, "A>B")
            self.assertEqual(pairs[0].source_family, "generic")


class FallbackJudgeTests(unittest.TestCase):
    def test_fallback_picks_the_response_matching_library_terms(self) -> None:
        library = _library()
        pair = ExternalSlicePair(
            pair_id="p1",
            prompt="prompt",
            response_a="final answer clearly stated: 42",
            response_b="...unclear response...",
            label="A>B",
            source_family="generic",
            source="test",
        )
        self.assertEqual(_default_fallback_judge(pair, library), "A>B")


class ScoreExternalSliceTests(unittest.TestCase):
    def test_scoring_aggregates_accuracy(self) -> None:
        library = _library()
        pairs = [
            ExternalSlicePair(
                pair_id=f"p_{i}",
                prompt="prompt",
                response_a="final answer clearly stated",
                response_b="weak vague",
                label="A>B",
                source_family="generic",
                source="test",
            )
            for i in range(4)
        ]
        scoring = score_external_slice(pairs, library=library, slice_name="test_slice")
        self.assertEqual(scoring.pair_count, 4)
        self.assertEqual(scoring.correct_count, 4)
        self.assertEqual(scoring.wu_score, 100.0)
        self.assertIn("generic", scoring.by_family)

    def test_failures_are_recorded(self) -> None:
        library = _library()
        pairs = [
            ExternalSlicePair(
                pair_id="p_fail",
                prompt="prompt",
                response_a="weak",
                response_b="final answer clearly stated",
                label="A>B",
                source_family="generic",
                source="test",
            ),
        ]
        scoring = score_external_slice(pairs, library=library, slice_name="test_slice")
        self.assertEqual(scoring.correct_count, 0)
        self.assertEqual(len(scoring.failures), 1)


class ScoreExternalSlicesTests(unittest.TestCase):
    def test_missing_library_returns_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            summary = score_external_slices(repo_root=Path(tmp))
            self.assertFalse(summary["available"])

    def test_happy_path_uses_seed_slice(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            write_seed_slice_if_missing("helpsteer3_val", repo_root=repo_root)
            summary = score_external_slices(
                repo_root=repo_root,
                slice_names=("helpsteer3_val",),
                library=_library(),
            )
            self.assertTrue(summary["available"])
            slice_data = summary["slices"]["helpsteer3_val"]
            self.assertTrue(slice_data["available"])
            self.assertGreater(slice_data["pair_count"], 0)


class DefaultPathTests(unittest.TestCase):
    def test_default_slice_path_includes_artifact_dir(self) -> None:
        path = default_slice_path("helpsteer3_val", repo_root=Path("/tmp"))
        self.assertIn("external_eval_sets", str(path))
        self.assertTrue(str(path).endswith("helpsteer3_val.jsonl"))


if __name__ == "__main__":
    unittest.main()
