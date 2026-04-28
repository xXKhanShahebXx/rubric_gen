from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import Any, Dict, List

from rubric_gen.compiled import judgebench_eval as jbe
from rubric_gen.compiled.judgebench_eval import (
    _aggregate_pair_discriminator_attempts,
    _normalize_v2_config,
    _policy_self_consistency_n,
    _policy_self_consistency_temperature,
    _policy_v2_wide_discriminator_gate,
    _should_run_blind_pair_discriminator,
)


def _attempt(
    order: str,
    decision: str,
    *,
    confidence: str = "medium",
    sample_index: int = 0,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return {
        "order": order,
        "decision": decision,
        "raw_decision": decision,
        "distinguishing_behavior": f"reason_{order}",
        "confidence": confidence,
        "raw_response": "",
        "cache_hit": False,
        "parse_error": "",
        "temperature": temperature,
        "sample_index": sample_index,
    }


class PolicyAccessorTests(unittest.TestCase):
    def test_self_consistency_defaults(self) -> None:
        self.assertEqual(_policy_self_consistency_n({}), 1)
        self.assertGreaterEqual(_policy_self_consistency_temperature({}), 0.0)

    def test_self_consistency_bounds_are_clamped(self) -> None:
        self.assertEqual(_policy_self_consistency_n({"self_consistency_n": 50}), 9)
        self.assertEqual(_policy_self_consistency_n({"self_consistency_n": -2}), 1)
        self.assertEqual(_policy_self_consistency_temperature({"self_consistency_temperature": 5.0}), 1.5)
        self.assertEqual(_policy_self_consistency_temperature({"self_consistency_temperature": -1.0}), 0.0)

    def test_wide_gate_toggle(self) -> None:
        self.assertFalse(_policy_v2_wide_discriminator_gate({}))
        self.assertTrue(_policy_v2_wide_discriminator_gate({"v2_wide_discriminator_gate": True}))


class NormalizeV2ConfigTests(unittest.TestCase):
    def test_normalize_returns_defaults(self) -> None:
        normalized = _normalize_v2_config(None)
        self.assertEqual(normalized["self_consistency_n"], 1)
        self.assertFalse(normalized["v2_wide_discriminator_gate"])
        self.assertFalse(normalized["holistic_judge_enabled"])
        self.assertEqual(normalized["library_retrieval_top_k"], 0)
        self.assertEqual(normalized["rubric_library_path"], "")
        self.assertFalse(normalized["enable_rrd_filters"])

    def test_normalize_clamps_out_of_range(self) -> None:
        normalized = _normalize_v2_config({
            "self_consistency_n": 100,
            "library_retrieval_top_k": 999,
            "rrd_redundancy_threshold": 10.0,
        })
        self.assertEqual(normalized["self_consistency_n"], 9)
        self.assertEqual(normalized["library_retrieval_top_k"], 16)
        self.assertLessEqual(normalized["rrd_redundancy_threshold"], 1.0)


class AggregatePairDiscriminatorAttemptsTests(unittest.TestCase):
    def test_two_attempts_agreeing_high_confidence(self) -> None:
        attempts = [
            _attempt("AB_0", "A>B", confidence="high"),
            _attempt("BA_0", "A>B", confidence="high"),
        ]
        result = _aggregate_pair_discriminator_attempts(attempts)
        self.assertEqual(result["decision"], "A>B")
        self.assertEqual(result["confidence"], "high")
        self.assertTrue(result["order_consistent"])

    def test_self_consistency_majority_reaches_agreement(self) -> None:
        # 4 A>B votes across 2 orders times 2 samples, 1 minority vote; majority wins.
        attempts = [
            _attempt("AB_0", "A>B", confidence="medium", sample_index=0),
            _attempt("BA_0", "A>B", confidence="medium", sample_index=0),
            _attempt("AB_1", "A>B", confidence="medium", sample_index=1),
            _attempt("BA_1", "A=B", confidence="low", sample_index=1),
            _attempt("AB_2", "A>B", confidence="medium", sample_index=2),
        ]
        result = _aggregate_pair_discriminator_attempts(attempts)
        self.assertEqual(result["decision"], "A>B")

    def test_order_disagreement_flagged(self) -> None:
        attempts = [
            _attempt("AB_0", "A>B", confidence="medium", sample_index=0),
            _attempt("BA_0", "B>A", confidence="medium", sample_index=0),
        ]
        result = _aggregate_pair_discriminator_attempts(attempts)
        self.assertFalse(result["order_consistent"])
        self.assertEqual(result["parse_error"], "pair_discriminator_order_disagreement")


class WideDiscriminatorGateTests(unittest.TestCase):
    def _make_policy(
        self,
        *,
        wide: bool,
    ) -> Dict[str, Any]:
        return {
            "blind_scoring_profile": jbe._BLIND_SCORING_PROFILE_PRUNED_DISC_V1,
            "blind_discriminator_mode_by_family": {},
            "v2_wide_discriminator_gate": wide,
        }

    def _make_example(self, family: str = "livebench-reasoning") -> Any:
        example = type("E", (), {})()
        example.source_family = family
        example.source = "livebench-reasoning/logic-puzzle/person-right-left"
        example.prompt = "Example prompt"
        example.response_A = "A"
        example.response_B = "B"
        example.reference_answer = ""
        example.label = "A>B"
        example.pair_id = "pair_1"
        example.metadata = {}
        return example

    def test_baseline_gate_does_not_fire_when_margin_slightly_above_base(self) -> None:
        policy = self._make_policy(wide=False)
        example = self._make_example("livebench-reasoning")
        scoring = {
            "whitened_uniform": {
                "result": {
                    "decision": "A>B",
                    "score_A": 0.52,
                    "score_B": 0.51,
                    "whitening_unstable": False,
                },
            }
        }
        with patch.object(jbe, "_reasoning_small_margin_discriminator_candidate", return_value=False):
            self.assertFalse(
                _should_run_blind_pair_discriminator(
                    policy=policy,
                    example=example,
                    scoring=scoring,
                    rubric_count=16,
                    verifier_outcome={},
                )
            )

    def test_wide_gate_fires_on_small_reasoning_margin(self) -> None:
        policy = self._make_policy(wide=True)
        example = self._make_example("livebench-reasoning")
        scoring = {
            "whitened_uniform": {
                "result": {
                    "decision": "A>B",
                    "score_A": 0.52,
                    "score_B": 0.49,
                    "whitening_unstable": False,
                },
            }
        }
        with patch.object(jbe, "_reasoning_small_margin_discriminator_candidate", return_value=False):
            self.assertTrue(
                _should_run_blind_pair_discriminator(
                    policy=policy,
                    example=example,
                    scoring=scoring,
                    rubric_count=12,
                    verifier_outcome={},
                )
            )

    def test_wide_gate_respects_verifier_override(self) -> None:
        policy = self._make_policy(wide=True)
        example = self._make_example("livebench-reasoning")
        scoring = {
            "whitened_uniform": {
                "result": {
                    "decision": "A=B",
                    "score_A": 0.5,
                    "score_B": 0.5,
                    "whitening_unstable": False,
                },
            }
        }
        verifier_outcome = {"recommended_decision": "A>B", "confidence": "high"}
        with patch.object(jbe, "_reasoning_small_margin_discriminator_candidate", return_value=False):
            self.assertFalse(
                _should_run_blind_pair_discriminator(
                    policy=policy,
                    example=example,
                    scoring=scoring,
                    rubric_count=12,
                    verifier_outcome=verifier_outcome,
                )
            )


if __name__ == "__main__":
    unittest.main()
