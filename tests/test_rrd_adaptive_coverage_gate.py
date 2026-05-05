"""Tests for the v2 adaptive decomposition coverage gate (Tier A2).

The original gate ``coverage_count > config.decomposition_threshold`` (default
4) effectively required 5+ satisfied candidates to trigger decomposition.
With shard 0's mean candidate pool of 5.82 (vs target 8) that meant ~85-100%
satisfaction was required, which is why only 1.9% of attempts succeeded.

The v2 rule replaces the static threshold with ``ceil(len(candidates) / 2)``
clamped at the configured threshold, so smaller pools get a proportionally
smaller absolute trigger.

Rather than wire up the full ``run_rrd`` codepath (which requires LLMs and a
full candidate pool), these tests exercise the gate arithmetic directly via
a small replication of the conditional in the engine.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from rubric_gen.config import PipelineConfig


def _config(decomposition_threshold: int) -> PipelineConfig:
    return PipelineConfig(
        dataset_path=Path("/tmp/none"),
        output_dir=Path("/tmp/none"),
        run_name="adaptive_gate_unit",
        decomposition_threshold=decomposition_threshold,
    )


def _adaptive_threshold(config: PipelineConfig, n_candidates: int) -> int:
    """Mirror of the gate computation in ``run_rrd``."""
    half_pool = (n_candidates + 1) // 2
    adaptive_threshold = max(2, half_pool)
    return min(
        adaptive_threshold,
        max(1, int(config.decomposition_threshold)),
    )


class AdaptiveCoverageGateTests(unittest.TestCase):
    def test_full_pool_matches_configured_threshold(self) -> None:
        # pool=8, configured=4 -> threshold 4 (need coverage >= 4 for trigger)
        self.assertEqual(_adaptive_threshold(_config(4), 8), 4)

    def test_small_pool_drops_below_configured(self) -> None:
        # pool=5, configured=4 -> ceil(5/2)=3 -> threshold 3 (vs old 4-5)
        self.assertEqual(_adaptive_threshold(_config(4), 5), 3)

    def test_tiny_pool_floored_at_two(self) -> None:
        # pool=3, configured=4 -> ceil(3/2)=2 -> threshold 2 (the floor)
        self.assertEqual(_adaptive_threshold(_config(4), 3), 2)

    def test_pool_two_floored_at_two(self) -> None:
        # pool=2 -> ceil(2/2)=1 -> floor at 2 -> threshold 2
        self.assertEqual(_adaptive_threshold(_config(4), 2), 2)

    def test_large_pool_capped_at_configured(self) -> None:
        # pool=16, configured=4 -> adaptive=8, capped at configured=4
        self.assertEqual(_adaptive_threshold(_config(4), 16), 4)

    def test_higher_configured_threshold_passes_through(self) -> None:
        # pool=10, configured=6 -> adaptive=5 (smaller wins, not capped)
        self.assertEqual(_adaptive_threshold(_config(6), 10), 5)

    def test_rule_is_strictly_more_permissive_than_old(self) -> None:
        # For every pool size between 3 and 12, the new threshold should be
        # <= the old threshold (which was always config.decomposition_threshold).
        cfg = _config(4)
        old_threshold = cfg.decomposition_threshold
        for n in range(3, 13):
            new_threshold = _adaptive_threshold(cfg, n)
            self.assertLessEqual(
                new_threshold,
                old_threshold,
                f"pool size {n}: new threshold {new_threshold} > old {old_threshold}",
            )


if __name__ == "__main__":
    unittest.main()
