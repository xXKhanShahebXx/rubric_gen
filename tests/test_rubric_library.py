from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rubric_gen.compiled.rubric_library import (
    RUBRIC_LIBRARY_SCHEMA,
    RubricLibrary,
    RubricLibraryCriterion,
    load_rubric_library,
    maybe_load_default_library,
    save_rubric_library,
)
from rubric_gen.compiled.rubric_library_builder import (
    BuilderConfig,
    ExternalPreferencePair,
    ProposedCriterion,
    distill_library,
)
from rubric_gen.compiled.rubric_library_seed import (
    build_seed_pair_set,
    seed_proposer,
)


def _make_criterion(
    criterion_id: str = "lib_abc",
    *,
    applicable_families=("generic",),
    focus_kind: str = "",
) -> RubricLibraryCriterion:
    return RubricLibraryCriterion(
        criterion_id=criterion_id,
        dimension="completeness",
        label="label",
        requirement="requirement text",
        severity_tier="medium",
        applicable_families=applicable_families,
        source_tag="test",
        focus_kind=focus_kind,
    )


class RubricLibraryModelTests(unittest.TestCase):
    def test_matches_family_with_generic_tag_matches_everything(self) -> None:
        c = _make_criterion(applicable_families=("generic",))
        self.assertTrue(c.matches_family("mmlu-pro"))
        self.assertTrue(c.matches_family("livebench-reasoning"))

    def test_matches_family_respects_explicit_tags(self) -> None:
        c = _make_criterion(applicable_families=("mmlu-pro",))
        self.assertTrue(c.matches_family("mmlu-pro"))
        self.assertFalse(c.matches_family("livebench-reasoning"))

    def test_to_canonical_row_is_compatible_with_merge_shape(self) -> None:
        c = _make_criterion("lib_xyz")
        row = c.to_canonical_row(example_id="ex1", pair_id="pair1")

        self.assertEqual(row["merge_key"], "library::lib_xyz")
        self.assertEqual(row["example_ids"], ["ex1"])
        self.assertEqual(row["pair_ids"], ["pair1"])
        self.assertEqual(row["rubric_library_criterion_id"], "lib_xyz")
        self.assertIn("rubric_library_source", row)

    def test_filter_by_family_prioritises_hard_gates_then_specific_matches(self) -> None:
        library = RubricLibrary(
            version="v1",
            criteria=[
                RubricLibraryCriterion(
                    criterion_id="lib_1",
                    dimension="completeness",
                    label="L1",
                    requirement="R1",
                    severity_tier="medium",
                    applicable_families=("generic",),
                    source_tag="t",
                ),
                RubricLibraryCriterion(
                    criterion_id="lib_2",
                    dimension="final_answer_correctness",
                    label="L2",
                    requirement="R2",
                    severity_tier="hard_gate",
                    applicable_families=("mmlu-pro",),
                    source_tag="t",
                ),
                RubricLibraryCriterion(
                    criterion_id="lib_3",
                    dimension="grounding",
                    label="L3",
                    requirement="R3",
                    severity_tier="high",
                    applicable_families=("mmlu-pro",),
                    source_tag="t",
                ),
            ],
        )

        result = library.filter_by_family("mmlu-pro", limit=2)
        self.assertEqual([c.criterion_id for c in result], ["lib_2", "lib_3"])

    def test_by_focus_kinds_selects_matching_criteria(self) -> None:
        library = RubricLibrary(
            version="v1",
            criteria=[
                _make_criterion("lib_a", focus_kind="final_answer"),
                _make_criterion("lib_b", focus_kind="grounding"),
            ],
        )
        result = library.by_focus_kinds("mmlu-pro", ["final_answer"])
        self.assertEqual([c.criterion_id for c in result], ["lib_a"])

    def test_save_and_load_round_trip(self) -> None:
        library = RubricLibrary(
            version="v1",
            criteria=[_make_criterion("lib_rt")],
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            save_rubric_library(library, path)
            loaded = load_rubric_library(path)
            self.assertEqual(loaded.version, "v1")
            self.assertEqual(loaded.criterion_count, 1)
            self.assertEqual(loaded.criteria[0].criterion_id, "lib_rt")

    def test_load_rejects_wrong_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            path.write_text(json.dumps({"schema": "bogus", "criteria": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rubric_library(path)

    def test_maybe_load_default_library_returns_none_when_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertIsNone(maybe_load_default_library(Path(tmp)))


class RubricLibraryBuilderTests(unittest.TestCase):
    def test_distill_library_builds_library_from_seed_corpus(self) -> None:
        pairs = build_seed_pair_set()
        result = distill_library(
            pairs,
            proposer=seed_proposer,
            config=BuilderConfig(target_total=30, per_family_target=10, max_per_dimension=6),
        )
        self.assertGreater(result.accepted_count, 10)
        families = {
            fam
            for c in result.library.criteria
            for fam in c.applicable_families
        }
        self.assertIn("mmlu-pro", families)
        self.assertIn("livebench-reasoning", families)
        self.assertIn("livebench-math", families)
        self.assertIn("livecodebench", families)

    def test_distill_library_respects_budget(self) -> None:
        pairs = build_seed_pair_set()
        result = distill_library(
            pairs,
            proposer=seed_proposer,
            config=BuilderConfig(target_total=5, per_family_target=2, max_per_dimension=2),
        )
        self.assertLessEqual(result.accepted_count, 5)

    def test_distill_library_drops_misaligned_proposals(self) -> None:
        pair = ExternalPreferencePair(
            pair_id="pair",
            prompt="prompt",
            chosen="short",
            rejected="a long rejected response that contains much more content",
            source="test",
            source_family="generic",
        )

        def proposer(pair: ExternalPreferencePair):
            return [
                ProposedCriterion(
                    dimension="surface_coverage",
                    label="contains more content than rejected",
                    requirement="response has much more content than the rejected side",
                    severity_tier="medium",
                    focus_kind="",
                    source_tag="test",
                )
            ]

        result = distill_library([pair], proposer=proposer, config=BuilderConfig())
        self.assertEqual(result.accepted_count, 0)
        self.assertGreaterEqual(result.rejected_misaligned, 1)


class SeedProposerTests(unittest.TestCase):
    def test_seed_proposer_returns_family_specific_criteria(self) -> None:
        pair = ExternalPreferencePair(
            pair_id="seed",
            prompt="prompt",
            chosen="chosen",
            rejected="rejected",
            source="seed",
            source_family="livebench-reasoning",
        )
        proposals = seed_proposer(pair)
        self.assertGreater(len(proposals), 0)
        self.assertTrue(any(p.dimension == "assignment_completeness" for p in proposals))


class DefaultLibraryArtifactTests(unittest.TestCase):
    def test_default_library_artifact_is_valid_when_present(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        library = maybe_load_default_library(repo_root)
        if library is None:
            self.skipTest("default library artifact not built")
        self.assertEqual(library.version, "v1")
        self.assertGreaterEqual(library.criterion_count, 1)


if __name__ == "__main__":
    unittest.main()
