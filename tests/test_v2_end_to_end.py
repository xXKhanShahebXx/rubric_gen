"""
End-to-end wiring tests for the JudgeBench v2 pipeline.

These tests avoid invoking the full per-example discovery pipeline (which is covered by the
component-level tests) and instead verify that v2 policy configuration threads through all the
accessor helpers and decision gates. If any refactor breaks the v2 policy shape, these tests
will fail before an expensive OOF run is kicked off.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

from rubric_gen.compiled import judgebench_eval as jbe
from rubric_gen.compiled.discovery import RecursiveDiscoveryConfig
from rubric_gen.compiled.holistic_judge import apply_holistic_judge_to_scoring
from rubric_gen.compiled.reasoning_process_verifier import ReasoningProcessVerifierConfig
from rubric_gen.compiled.rubric_library import (
    RubricLibrary,
    RubricLibraryCriterion,
    save_rubric_library,
)


def _build_v2_policy(rubric_library_path: Path) -> Dict[str, Any]:
    example = jbe.JudgeBenchJoinedExample(
        split_name="smoke",
        pair_id="pair_1",
        source="livebench-reasoning",
        source_family="livebench-reasoning",
        question="q",
        reference_answer="a",
        response_model="gpt-4o-2024-05-13",
        response_A="A",
        response_B="B",
        label="A>B",
        metadata={"pair_id": "pair_1"},
    )
    return jbe.build_initial_frozen_policy(
        train_examples=[example],
        bootstrap_iterations=1,
        recursive_config=RecursiveDiscoveryConfig(),
        protocol_mode=jbe._PROTOCOL_MODE_GENERIC_BASELINE,
        reference_answer_access=False,
        blind_scoring_profile=jbe._BLIND_SCORING_PROFILE_PRUNED_DISC_V1,
        blind_budget_profile=jbe._BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V2,
        blind_wu_profile=jbe._BLIND_WU_PROFILE_STABLE_V1,
        retrieval_profile=jbe._RETRIEVAL_PROFILE_LIBRARY_V1,
        retrieval_top_k=2,
        blind_discriminator_mode_by_family={"livebench-reasoning": "strict"},
        self_consistency_n=5,
        self_consistency_temperature=0.7,
        v2_wide_discriminator_gate=True,
        holistic_judge_enabled=True,
        library_retrieval_top_k=6,
        rubric_library_path=rubric_library_path,
        enable_rrd_filters=True,
        rrd_redundancy_threshold=0.88,
    )


class V2PolicyWiringTests(unittest.TestCase):
    def test_full_v2_policy_exposes_every_new_accessor(self) -> None:
        with TemporaryDirectory() as tmp:
            lib_path = Path(tmp) / "lib.json"
            save_rubric_library(
                RubricLibrary(
                    version="v1",
                    criteria=[
                        RubricLibraryCriterion(
                            criterion_id="lib_abc",
                            dimension="assignment_completeness",
                            label="L",
                            requirement="R",
                            severity_tier="hard_gate",
                            applicable_families=("livebench-reasoning",),
                            source_tag="seed",
                        )
                    ],
                ),
                lib_path,
            )
            policy = _build_v2_policy(lib_path)

            self.assertEqual(jbe._policy_self_consistency_n(policy), 5)
            self.assertAlmostEqual(jbe._policy_self_consistency_temperature(policy), 0.7, places=4)
            self.assertTrue(jbe._policy_v2_wide_discriminator_gate(policy))
            self.assertTrue(jbe._policy_holistic_judge_enabled(policy))
            self.assertEqual(jbe._policy_library_retrieval_top_k(policy), 6)
            self.assertTrue(policy["enable_rrd_filters"])
            self.assertEqual(policy["rubric_library_path"], str(lib_path))
            self.assertAlmostEqual(policy["rrd_redundancy_threshold"], 0.88, places=4)

    def test_wide_gate_policy_fires_discriminator_on_low_margin(self) -> None:
        with TemporaryDirectory() as tmp:
            policy = _build_v2_policy(Path(tmp) / "lib.json")
            example = jbe.JudgeBenchJoinedExample(
                split_name="smoke",
                pair_id="pair_1",
                source="livebench-reasoning",
                source_family="livebench-reasoning",
                question="q",
                reference_answer="a",
                response_model="gpt-4o-2024-05-13",
                response_A="A",
                response_B="B",
                label="A>B",
                metadata={"pair_id": "pair_1"},
            )
            scoring = {
                "whitened_uniform": {
                    "result": {
                        "decision": "A>B",
                        "score_A": 0.53,
                        "score_B": 0.50,
                        "whitening_unstable": False,
                    }
                }
            }
            should_fire = jbe._should_run_blind_pair_discriminator(
                policy=policy,
                example=example,
                scoring=scoring,
                rubric_count=12,
                verifier_outcome={},
            )
            self.assertTrue(should_fire)

    def test_holistic_judge_integrates_with_wu_result(self) -> None:
        scoring = {
            "whitened_uniform": {
                "result": {
                    "decision": "A=B",
                    "score_A": 0.5,
                    "score_B": 0.5,
                    "whitening_unstable": False,
                    "decision_policy": "whitened_uniform",
                }
            },
        }
        updated = apply_holistic_judge_to_scoring(
            scoring=scoring,
            holistic={
                "decision": "A>B",
                "confidence": "high",
                "order_consistent": True,
                "distinguishing_behavior": "A sustains the argument better.",
            },
        )
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "A>B")
        self.assertEqual(updated["whitened_uniform"]["result"]["decision_policy"], "holistic_judge")

    def test_process_verifier_config_defaults_are_reasonable(self) -> None:
        config = ReasoningProcessVerifierConfig()
        self.assertGreater(config.trigger_margin, 0.0)
        self.assertGreater(config.medium_confidence_margin, config.trigger_margin)
        self.assertGreater(config.high_confidence_margin, config.medium_confidence_margin)


if __name__ == "__main__":
    unittest.main()
