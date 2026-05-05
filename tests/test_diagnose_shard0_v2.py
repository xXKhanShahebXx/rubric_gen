"""Tests for the v2 shard 0 diagnostics script (Phase 1)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.diagnose_shard0_v2 import ShardAccumulator


def _example_artifact(
    *,
    example_id: str,
    task_profile_id: str = "general_instruction_following",
    rubrics: list = None,
    rejected: list = None,
    candidates: list = None,
    evaluations: list = None,
    termination_rejections_field: int = 15,
) -> dict:
    return {
        "example": {
            "example_id": example_id,
            "source": "medical",
            "source_id": example_id,
            "task_profile_id": task_profile_id,
            "task_prompt": "Q?",
            "conversation": "",
        },
        "candidates": candidates or [
            {
                "candidate_id": f"{example_id}__c1",
                "origin_kind": "anchor",
                "source_label": "reference_note",
                "quality_bucket": "gold_like",
            },
            {
                "candidate_id": f"{example_id}__c2",
                "origin_kind": "generated",
                "source_label": "generated_direct",
                "quality_bucket": "frontier_generated",
            },
        ],
        "methods": {
            "rrd_uniform": {
                "rubrics": rubrics or [],
                "evaluations": evaluations or [],
                "artifact": {
                    "rejected": rejected or [],
                    "termination_rejections": termination_rejections_field,
                },
            }
        },
    }


class AngleATests(unittest.TestCase):
    def test_term_rejections_field_recognised_as_constant(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        for i in range(3):
            acc.consume(_example_artifact(example_id=f"ex_{i}", rejected=[], termination_rejections_field=15))
        # All three should report the constant 15 in the budget field.
        self.assertEqual(set(acc.budget_unique.keys()), {15})
        # Real rejection counts should all be zero (we passed empty rejected lists).
        self.assertEqual(acc.real_reject_counts, [0.0, 0.0, 0.0])

    def test_real_rejection_count_uses_len_rejected(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        rejs = [{"reason": "overlap", "rubric": "x"} for _ in range(7)]
        acc.consume(_example_artifact(example_id="ex_1", rejected=rejs))
        self.assertEqual(acc.real_reject_counts, [7.0])


class AngleBTests(unittest.TestCase):
    def test_per_reason_histogram(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        rejs = [
            {"reason": "overlap"},
            {"reason": "overlap"},
            {"reason": "misaligned"},
            {"reason": "insufficient_decomposition_gain"},
        ]
        acc.consume(_example_artifact(example_id="ex_1", rejected=rejs))
        self.assertEqual(acc.reject_reason_counter["overlap"], 2)
        self.assertEqual(acc.reject_reason_counter["misaligned"], 1)
        self.assertEqual(acc.reject_reason_counter["insufficient_decomposition_gain"], 1)


class AngleCTests(unittest.TestCase):
    def test_family_split_by_task_profile_id(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        acc.consume(_example_artifact(
            example_id="ex_1",
            task_profile_id="documentation_variants",
            rubrics=[{"text": "the note correctly identifies x", "rubric_id": "r1", "depth": 0}],
        ))
        acc.consume(_example_artifact(
            example_id="ex_2",
            task_profile_id="general_instruction_following",
            rubrics=[{"text": "the note includes a y", "rubric_id": "r2", "depth": 0}],
        ))
        self.assertEqual(acc.family_examples["documentation_variants"]["n"], 1)
        self.assertEqual(acc.family_examples["general_instruction_following"]["n"], 1)


class AngleDTests(unittest.TestCase):
    def test_useless_bucket_for_p_near_1(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        # rubric fires on all 4 candidates -> p=1.0 -> p(1-p)=0 -> useless
        evals = [{"rubric_id": "r1", "candidate_id": f"c{i}", "satisfied": True} for i in range(4)]
        acc.consume(_example_artifact(
            example_id="ex_1",
            rubrics=[{"text": "always-fire rubric", "rubric_id": "r1"}],
            evaluations=evals,
        ))
        self.assertEqual(acc.discrim_useless, 1)
        self.assertEqual(acc.discrim_total, 1)

    def test_strong_bucket_for_p_near_half(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        # 2 yes, 2 no -> p=0.5 -> p(1-p)=0.25 -> strong
        evals = [
            {"rubric_id": "r1", "candidate_id": "c1", "satisfied": True},
            {"rubric_id": "r1", "candidate_id": "c2", "satisfied": False},
            {"rubric_id": "r1", "candidate_id": "c3", "satisfied": True},
            {"rubric_id": "r1", "candidate_id": "c4", "satisfied": False},
        ]
        acc.consume(_example_artifact(
            example_id="ex_1",
            rubrics=[{"text": "discriminating rubric", "rubric_id": "r1"}],
            evaluations=evals,
        ))
        self.assertEqual(acc.discrim_strong, 1)


class AngleETests(unittest.TestCase):
    def test_pool_breakdown_by_origin(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        cands = [
            {"candidate_id": "c1", "origin_kind": "anchor"},
            {"candidate_id": "c2", "origin_kind": "generated"},
            {"candidate_id": "c3", "origin_kind": "generated"},
            {"candidate_id": "c4", "origin_kind": "synthetic"},
        ]
        acc.consume(_example_artifact(example_id="ex_1", candidates=cands))
        pool = acc.family_candidate_pool["general_instruction_following"][0]
        self.assertEqual(pool["anchor"], 1)
        self.assertEqual(pool["generated"], 2)
        self.assertEqual(pool["synthetic"], 1)
        self.assertEqual(pool["total"], 4)


class AngleFTests(unittest.TestCase):
    def test_depth1_count_and_parent_attribution(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        rubrics = [
            {"rubric_id": "r0", "text": "parent", "depth": 0},
            {"rubric_id": "r1", "text": "child a", "depth": 1, "parent_id": "rp0"},
            {"rubric_id": "r2", "text": "child b", "depth": 1, "parent_id": "rp0"},
            {"rubric_id": "r3", "text": "lone child", "depth": 1, "parent_id": "rp1"},
        ]
        acc.consume(_example_artifact(example_id="ex_1", rubrics=rubrics))
        self.assertEqual(acc.depth1_rubrics_seen, 3)
        self.assertEqual(acc.depth1_rubrics_with_parent_id_in_bank, 3)
        # Two unique parents (rp0 produced 2 children, rp1 produced 1).
        self.assertEqual(len(acc.parents_of_surviving_children), 2)


class AngleGTests(unittest.TestCase):
    def test_head_token_clustering(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        rubrics = [
            {"rubric_id": "r1", "text": "the note correctly identifies the diagnosis"},
            {"rubric_id": "r2", "text": "the note correctly identifies the medication"},
            {"rubric_id": "r3", "text": "the note includes a follow-up"},
        ]
        acc.consume(_example_artifact(example_id="ex_1", rubrics=rubrics))
        self.assertEqual(acc.head4_counter["the note correctly identifies"], 2)
        self.assertEqual(acc.head4_counter["the note includes a"], 1)


class AngleHTests(unittest.TestCase):
    def test_anchor_vs_generated_means_recorded(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        # 1 anchor satisfies 2/2; 1 generated satisfies 0/2 -> diff +1.0
        rubrics = [
            {"rubric_id": "r1", "text": "x"},
            {"rubric_id": "r2", "text": "y"},
        ]
        evals = [
            {"rubric_id": "r1", "candidate_id": "ex_1__c1", "satisfied": True},
            {"rubric_id": "r2", "candidate_id": "ex_1__c1", "satisfied": True},
            {"rubric_id": "r1", "candidate_id": "ex_1__c2", "satisfied": False},
            {"rubric_id": "r2", "candidate_id": "ex_1__c2", "satisfied": False},
        ]
        acc.consume(_example_artifact(
            example_id="ex_1",
            rubrics=rubrics,
            evaluations=evals,
        ))
        self.assertEqual(acc.per_example_anchor_mean, [1.0])
        self.assertEqual(acc.per_example_generated_mean, [0.0])
        self.assertEqual(acc.per_example_anchor_minus_generated, [1.0])


class MarkdownReportTests(unittest.TestCase):
    def test_markdown_summary_runs_and_mentions_headlines(self) -> None:
        acc = ShardAccumulator(method="rrd_uniform")
        for i in range(3):
            acc.consume(_example_artifact(example_id=f"ex_{i}"))
        with TemporaryDirectory() as td:
            md = acc.markdown_summary(Path(td), 3)
            self.assertIn("Shard 0 v2 Forensics", md)
            self.assertIn("Angle A", md)
            self.assertIn("Angle H", md)


if __name__ == "__main__":
    unittest.main()
