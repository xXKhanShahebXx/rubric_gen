import unittest

from rubric_gen.compiled.gold_refinement import (
    apply_calibration_hints_to_generated_row,
    build_prompt_calibration_guidance,
    classify_granularity_gaps,
    derive_calibration_profile_policy,
    derive_calibration_hints,
    refine_generated_rows,
)
from rubric_gen.compiled.healthbench_eval import (
    HealthBenchCompletion,
    HealthBenchExample,
    HealthBenchExpertCriterion,
    HealthBenchGoldProvider,
    _has_high_priority_gold_gaps,
    _post_refine_realign_reasons,
    _preserve_pre_refinement_matches,
)


def _example() -> HealthBenchExample:
    return HealthBenchExample(
        prompt_id="hb_gold_1",
        dialogue="What should I do next, and when should I follow up if the cough gets worse?",
        reference_answer=(
            "See a healthcare professional. Follow up within 48 hours if you are not improving, "
            "and seek urgent care right away for worsening breathing trouble."
        ),
        completions=[
            HealthBenchCompletion(
                completion_id="ideal",
                text=(
                    "See a healthcare professional. Follow up within 48 hours if you are not improving, "
                    "and seek urgent care right away for worsening breathing trouble."
                ),
                source="ideal_completion",
            )
        ],
        expert_rubrics=[
            HealthBenchExpertCriterion(
                criterion="The response gives a specific follow-up interval for reassessment.",
                points=8,
                tags=["level:example"],
            ),
            HealthBenchExpertCriterion(
                criterion="The response includes clear return precautions for worsening breathing trouble.",
                points=9,
                tags=["level:example"],
            ),
        ],
        is_multi_turn=False,
        n_turns=1,
        themes=["communication"],
        example_tags=["physician_agreed_category:consumer"],
    )


def _generated_rows() -> list[dict]:
    return [
        {
            "generated_index": 0,
            "dimension": "content_coverage",
            "label": "Main next steps covered",
            "requirement": "The response covers the main next steps and safety guidance.",
            "family": "content_coverage",
            "polarity": "positive",
        },
        {
            "generated_index": 1,
            "dimension": "format_compliance",
            "label": "Bullet formatting used",
            "requirement": "The response uses bullet formatting.",
            "family": "other",
            "polarity": "positive",
        },
    ]


class GoldCalibrationTests(unittest.TestCase):
    def test_healthbench_gold_provider_normalizes_example_rubrics(self) -> None:
        provider = HealthBenchGoldProvider()
        gold_rows = provider.gold_rows_for_example(_example())

        self.assertEqual(len(gold_rows), 2)
        self.assertEqual(gold_rows[0]["family"], "follow_up")
        self.assertEqual(gold_rows[1]["family"], "return_precautions_escalation")
        self.assertEqual(gold_rows[0]["polarity"], "positive")

    def test_refinement_improves_gold_alignment_metrics(self) -> None:
        provider = HealthBenchGoldProvider()
        example = _example()
        pre_rows = _generated_rows()
        pre_result = provider.compare_generated_rows(
            example=example,
            generated_rows=pre_rows,
            model_spec=None,
            router=None,
            cache=None,
            use_heuristic_only=True,
        )
        granularity = classify_granularity_gaps(
            provider_id=provider.provider_id,
            prompt_id=example.prompt_id,
            task_profile_id="clinical_decision_support",
            gold_rows=pre_result.gold_rows,
            generated_rows=pre_rows,
            alignment=pre_result.alignment,
        )
        refined = refine_generated_rows(
            prompt_id=example.prompt_id,
            task_profile_id="clinical_decision_support",
            gold_rows=pre_result.gold_rows,
            generated_rows=pre_rows,
            alignment=pre_result.alignment,
            granularity_report=granularity,
            calibration_hints=None,
        )
        post_result = provider.compare_generated_rows(
            example=example,
            generated_rows=refined["generated_rows"],
            model_spec=None,
            router=None,
            cache=None,
            use_heuristic_only=True,
        )

        self.assertTrue(refined["changed"])
        self.assertGreater(post_result.metrics["weighted_recall"], pre_result.metrics["weighted_recall"])
        self.assertGreater(post_result.metrics["expert_direct_matches"], pre_result.metrics["expert_direct_matches"])
        self.assertLess(post_result.metrics["generated_off_target"], pre_result.metrics["generated_off_target"])
        self.assertTrue(
            any(row.get("refinement_origin") == "gold_refinement" for row in refined["generated_rows"])
        )

    def test_calibration_hints_capture_split_bias_and_prompt_guidance(self) -> None:
        provider = HealthBenchGoldProvider()
        example = _example()
        pre_rows = _generated_rows()
        pre_result = provider.compare_generated_rows(
            example=example,
            generated_rows=pre_rows,
            model_spec=None,
            router=None,
            cache=None,
            use_heuristic_only=True,
        )
        granularity = classify_granularity_gaps(
            provider_id=provider.provider_id,
            prompt_id=example.prompt_id,
            task_profile_id="clinical_decision_support",
            gold_rows=pre_result.gold_rows,
            generated_rows=pre_rows,
            alignment=pre_result.alignment,
        )
        hints = derive_calibration_hints(
            [granularity, granularity, granularity, granularity],
            provider_id=provider.provider_id,
        )
        hints["eligibility"] = {"enabled_profiles": ["clinical_decision_support"]}
        guidance = build_prompt_calibration_guidance(
            hints,
            task_profile_id="clinical_decision_support",
        )

        profile_hints = hints["by_task_profile"]["clinical_decision_support"]
        self.assertIn("follow_up", profile_hints["split_bias_families"])
        self.assertIn("return_precautions_escalation", profile_hints["split_bias_families"])
        self.assertIn("follow-up", guidance.lower())
        self.assertIn("return precautions", guidance.lower())

    def test_note_profile_skips_prompt_guidance(self) -> None:
        hints = {
            "by_task_profile": {
                "note_documentation": {
                    "prompt_nudges": [
                        "Keep follow-up timing, surveillance intervals, and next-step monitoring as dedicated criteria instead of broad completeness checks."
                    ],
                    "observed_missing_family_counts": {
                        "follow_up": 10
                    },
                }
            },
            "eligibility": {
                "enabled_profiles": ["note_documentation"]
            },
        }

        guidance = build_prompt_calibration_guidance(
            hints,
            task_profile_id="note_documentation",
        )

        self.assertEqual(guidance, "")

    def test_prompt_guidance_filters_low_signal_nudges(self) -> None:
        hints = {
            "by_task_profile": {
                "clinical_decision_support": {
                    "prompt_nudges": [
                        "Keep follow-up timing, surveillance intervals, and next-step monitoring as dedicated criteria instead of broad completeness checks.",
                        "Use a dedicated grounding criterion when the stronger artifact stays within the provided context and the weaker artifact invents or assumes details.",
                    ],
                    "observed_missing_family_counts": {
                        "follow_up": 8,
                        "context_grounding": 2,
                    },
                }
            },
            "eligibility": {
                "enabled_profiles": ["clinical_decision_support"]
            },
        }

        guidance = build_prompt_calibration_guidance(
            hints,
            task_profile_id="clinical_decision_support",
        )

        self.assertIn("follow-up", guidance.lower())
        self.assertNotIn("grounding criterion", guidance.lower())

    def test_apply_calibration_hints_can_override_generic_family(self) -> None:
        row = {
            "generated_index": 0,
            "dimension": "instruction_adherence",
            "label": "Requested details included",
            "requirement": "The response includes the requested follow-up timing.",
            "family": "content_coverage",
            "polarity": "positive",
        }
        hints = {
            "by_task_profile": {
                "clinical_decision_support": {
                    "dimension_family_bias": {"instruction_adherence": "follow_up"},
                    "observed_missing_family_counts": {"follow_up": 8},
                }
            },
            "eligibility": {
                "enabled_profiles": ["clinical_decision_support"]
            },
        }

        updated = apply_calibration_hints_to_generated_row(
            row,
            task_profile_id="clinical_decision_support",
            calibration_hints=hints,
        )

        self.assertEqual(updated["family"], "follow_up")
        self.assertEqual(updated["family_pre_calibration"], "content_coverage")

    def test_note_profile_skips_family_override(self) -> None:
        row = {
            "generated_index": 0,
            "dimension": "instruction_adherence",
            "label": "Requested details included",
            "requirement": "The note includes the requested follow-up timing.",
            "family": "content_coverage",
            "polarity": "positive",
        }
        hints = {
            "by_task_profile": {
                "note_documentation": {
                    "dimension_family_bias": {"instruction_adherence": "follow_up"},
                    "observed_missing_family_counts": {"follow_up": 10},
                }
            },
            "eligibility": {
                "enabled_profiles": ["note_documentation"]
            },
        }

        updated = apply_calibration_hints_to_generated_row(
            row,
            task_profile_id="note_documentation",
            calibration_hints=hints,
        )

        self.assertEqual(updated["family"], "content_coverage")
        self.assertNotIn("family_pre_calibration", updated)

    def test_family_override_requires_high_signal_count(self) -> None:
        row = {
            "generated_index": 0,
            "dimension": "instruction_adherence",
            "label": "Requested details included",
            "requirement": "The response includes the requested follow-up timing.",
            "family": "content_coverage",
            "polarity": "positive",
        }
        hints = {
            "by_task_profile": {
                "clinical_decision_support": {
                    "dimension_family_bias": {"instruction_adherence": "follow_up"},
                    "observed_missing_family_counts": {"follow_up": 3},
                }
            },
            "eligibility": {
                "enabled_profiles": ["clinical_decision_support"]
            },
        }

        updated = apply_calibration_hints_to_generated_row(
            row,
            task_profile_id="clinical_decision_support",
            calibration_hints=hints,
        )

        self.assertEqual(updated["family"], "content_coverage")
        self.assertNotIn("family_pre_calibration", updated)

    def test_missing_eligibility_disables_calibration_reuse(self) -> None:
        hints = {
            "by_task_profile": {
                "clinical_decision_support": {
                    "prompt_nudges": [
                        "Keep follow-up timing, surveillance intervals, and next-step monitoring as dedicated criteria instead of broad completeness checks."
                    ],
                    "dimension_family_bias": {"instruction_adherence": "follow_up"},
                    "observed_missing_family_counts": {"follow_up": 10},
                }
            }
        }
        row = {
            "generated_index": 0,
            "dimension": "instruction_adherence",
            "label": "Requested details included",
            "requirement": "The response includes the requested follow-up timing.",
            "family": "content_coverage",
            "polarity": "positive",
        }

        guidance = build_prompt_calibration_guidance(
            hints,
            task_profile_id="clinical_decision_support",
        )
        updated = apply_calibration_hints_to_generated_row(
            row,
            task_profile_id="clinical_decision_support",
            calibration_hints=hints,
        )

        self.assertEqual(guidance, "")
        self.assertEqual(updated["family"], "content_coverage")

    def test_derive_calibration_profile_policy_selects_only_safe_profiles(self) -> None:
        baseline_summary = {
            "pre_refinement_alignment_by_task_profile": {
                "general_instruction_following": {
                    "examples_scored": 14,
                    "weighted_recall": 0.60,
                    "expert_recall": 0.55,
                    "generated_precision": 0.40,
                    "generated_off_target": 10,
                },
                "note_documentation": {
                    "examples_scored": 9,
                    "weighted_recall": 0.70,
                    "expert_recall": 0.65,
                    "generated_precision": 0.50,
                    "generated_off_target": 4,
                },
                "agentic_workflows": {
                    "examples_scored": 3,
                    "weighted_recall": 0.60,
                    "expert_recall": 0.60,
                    "generated_precision": 0.50,
                    "generated_off_target": 1,
                },
                "clinical_decision_support": {
                    "examples_scored": 100,
                    "weighted_recall": 0.70,
                    "expert_recall": 0.70,
                    "generated_precision": 0.50,
                    "generated_off_target": 20,
                },
            }
        }
        apply_summary = {
            "pre_refinement_alignment_by_task_profile": {
                "general_instruction_following": {
                    "examples_scored": 14,
                    "weighted_recall": 0.62,
                    "expert_recall": 0.55,
                    "generated_precision": 0.55,
                    "generated_off_target": 5,
                },
                "note_documentation": {
                    "examples_scored": 9,
                    "weighted_recall": 0.75,
                    "expert_recall": 0.70,
                    "generated_precision": 0.55,
                    "generated_off_target": 3,
                },
                "agentic_workflows": {
                    "examples_scored": 3,
                    "weighted_recall": 0.70,
                    "expert_recall": 0.65,
                    "generated_precision": 0.60,
                    "generated_off_target": 0,
                },
                "clinical_decision_support": {
                    "examples_scored": 100,
                    "weighted_recall": 0.69,
                    "expert_recall": 0.69,
                    "generated_precision": 0.52,
                    "generated_off_target": 15,
                },
            }
        }

        policy = derive_calibration_profile_policy(
            baseline_summary=baseline_summary,
            apply_summary=apply_summary,
            min_examples=5,
        )

        self.assertEqual(policy["enabled_profiles"], ["general_instruction_following"])
        self.assertEqual(
            policy["by_task_profile"]["note_documentation"]["reason"],
            "protected_profile",
        )
        self.assertEqual(
            policy["by_task_profile"]["agentic_workflows"]["reason"],
            "insufficient_heldout_examples",
        )
        self.assertFalse(policy["by_task_profile"]["clinical_decision_support"]["eligible"])

    def test_high_priority_gold_gap_detection_targets_missing_and_family_mismatch(self) -> None:
        self.assertTrue(
            _has_high_priority_gold_gaps(
                {
                    "gaps": [
                        {
                            "source": "gold",
                            "gap_type": "missing_gold_criterion",
                            "priority": 8,
                        }
                    ]
                }
            )
        )
        self.assertTrue(
            _has_high_priority_gold_gaps(
                {
                    "gaps": [
                        {
                            "source": "gold",
                            "gap_type": "family_mismatch",
                            "priority": 7,
                        }
                    ]
                }
            )
        )
        self.assertFalse(
            _has_high_priority_gold_gaps(
                {
                    "gaps": [
                        {
                            "source": "gold",
                            "gap_type": "too_coarse",
                            "priority": 10,
                        },
                        {
                            "source": "gold",
                            "gap_type": "missing_gold_criterion",
                            "priority": 4,
                        },
                    ]
                }
            )
        )

    def test_post_refine_realign_reasons_include_recursive_change_and_priority_gaps(self) -> None:
        reasons = _post_refine_realign_reasons(
            recursive_structure_changed=True,
            granularity_report={
                "gaps": [
                    {
                        "source": "gold",
                        "gap_type": "family_mismatch",
                        "priority": 9,
                    }
                ]
            },
        )

        self.assertEqual(
            reasons,
            ["recursive_structure_changed", "persistent_high_priority_gold_gaps"],
        )

    def test_preserve_pre_refinement_matches_keeps_existing_partial_coverage(self) -> None:
        merged = _preserve_pre_refinement_matches(
            pre_alignment={
                "expert_matches": [
                    {
                        "expert_index": 0,
                        "best_generated_index": 0,
                        "match_label": "partial",
                        "reason": "broad pre-refinement match",
                    }
                ],
                "generated_assessments": [
                    {
                        "generated_index": 0,
                        "matched_expert_indices": [0],
                        "precision_label": "broader_but_valid",
                        "reason": "broad pre-refinement assessment",
                    }
                ],
            },
            post_alignment={
                "expert_matches": [
                    {
                        "expert_index": 0,
                        "best_generated_index": None,
                        "match_label": "none",
                        "reason": "heuristic lost broad match",
                    }
                ],
                "generated_assessments": [
                    {
                        "generated_index": 0,
                        "matched_expert_indices": [],
                        "precision_label": "off_target",
                        "reason": "heuristic lost assessment",
                    }
                ],
            },
            post_generated_rows=[
                {
                    "generated_index": 0,
                    "pre_refinement_generated_index": 0,
                    "label": "Main next steps covered",
                    "requirement": "The response covers the main next steps and safety guidance.",
                }
            ],
        )

        self.assertEqual(merged["expert_matches"][0]["match_label"], "partial")
        self.assertEqual(merged["expert_matches"][0]["best_generated_index"], 0)
        self.assertEqual(merged["generated_assessments"][0]["precision_label"], "broader_but_valid")
        self.assertEqual(merged["generated_assessments"][0]["matched_expert_indices"], [0])


if __name__ == "__main__":
    unittest.main()
