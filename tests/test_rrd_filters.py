from __future__ import annotations

import unittest
from typing import Any, Dict, List

from rubric_gen.compiled.discovery import (
    merge_proposal_entries,
    merge_proposal_entries_with_rrd_filters,
)
from rubric_gen.compiled.rrd_filters import (
    FilterStats,
    PairContext,
    apply_misalignment_filter,
    apply_redundancy_filter,
    apply_rrd_filters,
    default_misalignment_evaluator,
)


def _make_row(
    *,
    pair_id: str = "pair_1",
    dimension: str = "completeness",
    label: str = "label",
    requirement: str = "response references relevant concepts x y z",
    severity_tier: str = "medium",
    count: int = 1,
    merge_key: str | None = None,
) -> Dict[str, Any]:
    merge_key = merge_key or f"{dimension}||{severity_tier}||{label}||{requirement}"
    return {
        "merge_key": merge_key,
        "dimension": dimension,
        "label": label,
        "requirement": requirement,
        "severity_tier": severity_tier,
        "count": count,
        "example_ids": ["ex1"],
        "pair_ids": [pair_id],
        "criterion_ids": [],
        "parent_criterion_ids": [],
        "root_pair_ids": [pair_id],
        "recursion_depths": [],
        "recursion_reasons": [],
        "decomposition_sources": [],
    }


def _pair(pair_id: str, *, strong: str, weak: str) -> PairContext:
    return PairContext(pair_id=pair_id, strong_text=strong, weak_text=weak)


class MisalignmentFilterTests(unittest.TestCase):
    def test_aligned_row_is_kept(self) -> None:
        row = _make_row(requirement="response references concept alpha beta")
        ctx = _pair("pair_1", strong="response references concept alpha beta and more detail", weak="short reply")
        self.assertTrue(default_misalignment_evaluator(row, [ctx]))

    def test_misaligned_row_is_dropped(self) -> None:
        row = _make_row(requirement="response includes unique marker xyz")
        ctx = _pair("pair_1", strong="different content entirely", weak="response includes unique marker xyz and more")
        self.assertFalse(default_misalignment_evaluator(row, [ctx]))

    def test_apply_misalignment_filter_accumulates_stats(self) -> None:
        rows = [
            _make_row(requirement="alpha beta gamma"),
            _make_row(
                pair_id="pair_2",
                requirement="weak only content",
                merge_key="mk_weak",
            ),
        ]
        contexts = [
            _pair("pair_1", strong="alpha beta gamma more", weak="unrelated"),
            _pair("pair_2", strong="unrelated reference", weak="weak only content"),
        ]
        stats = FilterStats()
        kept = apply_misalignment_filter(rows, contexts, stats=stats)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.dropped_misaligned, 1)

    def test_apply_misalignment_filter_keeps_rows_when_no_context(self) -> None:
        rows = [_make_row()]
        stats = FilterStats()
        kept = apply_misalignment_filter(rows, [], stats=stats)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats.dropped_misaligned, 0)


class RedundancyFilterTests(unittest.TestCase):
    def test_similar_rows_are_collapsed(self) -> None:
        rows = [
            _make_row(
                label="references concept alpha",
                requirement="response references concept alpha in full detail",
                merge_key="mk_a",
            ),
            _make_row(
                label="references concept alpha",
                requirement="response references concept alpha in full detail",
                merge_key="mk_b",
                severity_tier="hard_gate",
            ),
        ]
        stats = FilterStats()
        kept = apply_redundancy_filter(rows, threshold=0.6, stats=stats)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["severity_tier"], "hard_gate")
        self.assertEqual(stats.dropped_redundant, 1)

    def test_distinct_rows_are_preserved(self) -> None:
        rows = [
            _make_row(
                label="A",
                requirement="requirement alpha",
                merge_key="mk_a",
            ),
            _make_row(
                label="B",
                requirement="completely different beta gamma requirement",
                merge_key="mk_b",
            ),
        ]
        kept = apply_redundancy_filter(rows, threshold=0.8)
        self.assertEqual(len(kept), 2)


class RrdPipelineTests(unittest.TestCase):
    def test_apply_rrd_filters_runs_both_filters(self) -> None:
        rows = [
            _make_row(
                label="covers alpha beta gamma",
                requirement="response covers concept alpha beta gamma",
                merge_key="mk_1",
            ),
            _make_row(
                pair_id="pair_2",
                label="weak marker",
                requirement="response contains exclusive weak marker zzz",
                merge_key="mk_2",
            ),
            _make_row(
                label="covers alpha beta gamma",
                requirement="response covers concept alpha beta gamma",
                merge_key="mk_3",
            ),
        ]
        contexts = [
            _pair(
                "pair_1",
                strong="response covers concept alpha beta gamma in depth",
                weak="unrelated response",
            ),
            _pair(
                "pair_2",
                strong="unrelated reference",
                weak="response contains exclusive weak marker zzz here",
            ),
        ]
        filtered, stats = apply_rrd_filters(rows, pair_contexts=contexts, redundancy_threshold=0.7)
        self.assertEqual(len(filtered), 1)
        self.assertGreaterEqual(stats.dropped_misaligned, 1)
        self.assertGreaterEqual(stats.dropped_redundant, 1)

    def test_merge_proposal_entries_with_rrd_filters_end_to_end(self) -> None:
        local_rows = [
            {
                "dimension": "completeness",
                "label": "covers concept alpha",
                "requirement": "response covers concept alpha in depth",
                "severity_tier": "high",
                "pair_id": "pair_1",
                "example_id": "ex_1",
            },
            {
                "dimension": "completeness",
                "label": "covers concept alpha redux",
                "requirement": "response covers concept alpha in depth",
                "severity_tier": "medium",
                "pair_id": "pair_1",
                "example_id": "ex_1",
            },
            {
                "dimension": "grounding",
                "label": "weak only marker",
                "requirement": "response contains exclusive weak marker zzz",
                "severity_tier": "medium",
                "pair_id": "pair_2",
                "example_id": "ex_1",
            },
        ]
        baseline = merge_proposal_entries(local_rows)
        self.assertGreaterEqual(len(baseline["canonical_proposals"]), 2)

        filtered = merge_proposal_entries_with_rrd_filters(
            local_rows,
            pair_contexts=[
                PairContext(
                    pair_id="pair_1",
                    strong_text="response covers concept alpha in great depth and nuance",
                    weak_text="short weak response",
                ),
                PairContext(
                    pair_id="pair_2",
                    strong_text="unrelated content",
                    weak_text="response contains exclusive weak marker zzz which is irrelevant",
                ),
            ],
            redundancy_threshold=0.7,
        )
        self.assertEqual(len(filtered["canonical_proposals"]), 1)
        rrd_payload = filtered["rrd_filters"]
        self.assertEqual(rrd_payload["schema"], "compiled_discovery_rrd_filters_v1")
        self.assertGreaterEqual(rrd_payload["stats"]["dropped_misaligned"], 1)
        self.assertGreaterEqual(rrd_payload["stats"]["dropped_redundant"], 1)

    def test_merge_proposal_entries_with_rrd_filters_preserves_shape(self) -> None:
        rows = [
            {
                "dimension": "completeness",
                "label": "a",
                "requirement": "solid detail kept as is",
                "severity_tier": "high",
                "pair_id": "p1",
                "example_id": "ex_1",
            },
        ]
        filtered = merge_proposal_entries_with_rrd_filters(rows)
        proposal = filtered["canonical_proposals"][0]
        for key in (
            "merge_key",
            "dimension",
            "label",
            "requirement",
            "severity_tier",
            "count",
            "example_ids",
            "pair_ids",
        ):
            self.assertIn(key, proposal)


if __name__ == "__main__":
    unittest.main()
