import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rubric_gen.compiled.healthbench_eval import (
    GoldAlignmentArtifact,
    HealthBenchCompletion,
    HealthBenchExample,
    HealthBenchExpertCriterion,
    HealthBenchRoutingDecision,
    SubsetDecision,
    build_alignment_lookup,
    classify_task_compatibility,
    infer_criterion_polarity,
    map_expert_rubric_family,
    map_generated_criterion_family,
    route_healthbench_examples,
    route_healthbench_task,
    run_healthbench_evaluation,
)
from rubric_gen.types import ModelSpec


def _example(
    *,
    dialogue: str,
    ideal_text: str,
    expert_criteria: list[tuple[str, int, list[str]]],
    example_tags: list[str] | None = None,
) -> HealthBenchExample:
    return HealthBenchExample(
        prompt_id="hb_1",
        dialogue=dialogue,
        reference_answer=ideal_text,
        completions=[
            HealthBenchCompletion(completion_id="ideal", text=ideal_text, source="ideal_completion"),
            HealthBenchCompletion(completion_id="alt_1", text="Weaker alternative.", source="ref_completion"),
        ],
        expert_rubrics=[
            HealthBenchExpertCriterion(criterion=text, points=points, tags=tags)
            for text, points, tags in expert_criteria
        ],
        is_multi_turn=False,
        n_turns=1,
        themes=["communication"],
        example_tags=example_tags or [],
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
        alias="test",
        provider="test",
        model="test-model",
        api_key_env="TEST_API_KEY",
    )


def _alignment_artifact(
    *,
    gold_rows: list[dict],
    generated_rows: list[dict],
) -> GoldAlignmentArtifact:
    indexed_gold_rows = [
        {
            "expert_index": idx,
            **row,
        }
        for idx, row in enumerate(gold_rows)
    ]
    alignment = {
        "expert_matches": [
            {
                "expert_index": 0,
                "best_generated_index": 0,
                "match_label": "direct",
                "reason": "unit_test_alignment",
            }
        ],
        "generated_assessments": [
            {
                "generated_index": 0,
                "matched_expert_indices": [0],
                "precision_label": "aligned",
                "reason": "unit_test_alignment",
            }
        ],
        "fallback": False,
        "parse_error": None,
    }
    metrics = {
        "expert_criteria_total": 1,
        "expert_criteria_matched": 1,
        "expert_direct_matches": 1,
        "expert_partial_matches": 0,
        "weighted_recall": 1.0,
        "precision": 1.0,
        "generated_criteria_total": 1,
        "generated_aligned": 1,
        "generated_broader_but_valid": 0,
        "generated_valid_extra": 0,
        "generated_off_target": 0,
        "polarity_accuracy": 1.0,
        "expert_recall": 1.0,
    }
    family_summary = {
        "follow_up": {
            "expert_count": 1,
            "expert_weight": 8,
            "matched_expert_count": 1,
            "matched_expert_weight": 8,
            "generated_count": 1,
        }
    }
    return GoldAlignmentArtifact(
        provider_id="healthbench",
        gold_rows=indexed_gold_rows,
        alignment=alignment,
        metrics=metrics,
        family_summary=family_summary,
        alignment_lookup=build_alignment_lookup(alignment),
        cache_hit=False,
        parse_error=None,
        metadata={"use_heuristic_only": True},
    )


class HealthBenchEvalTests(unittest.TestCase):
    def test_filter_selects_primary_note_task_from_dialogue_and_headers(self) -> None:
        example = _example(
            dialogue="Draft an outpatient progress note for a patient with knee pain.",
            ideal_text=(
                "Subjective:\nKnee pain for 2 weeks.\n\n"
                "Objective:\nVitals normal.\n\n"
                "Assessment and Plan:\nLikely strain. Conservative care.\n"
            ),
            expert_criteria=[
                (
                    "Provides an outpatient progress note including chief complaint, HPI, assessment, and plan.",
                    6,
                    ["level:example", "axis:instruction_following"],
                )
            ],
        )

        decision = classify_task_compatibility(example)

        self.assertTrue(decision.selected)
        self.assertEqual(decision.category, "primary_note_task")
        self.assertGreaterEqual(decision.ideal_header_count, 2)

    def test_filter_excludes_general_lay_advice_task(self) -> None:
        example = _example(
            dialogue="What causes mild insomnia?",
            ideal_text="Mild insomnia can happen due to stress, caffeine, or schedule changes.",
            expert_criteria=[
                (
                    "Uses plain language and provides a concise explanation of common causes of insomnia.",
                    6,
                    ["level:example", "axis:communication_quality"],
                )
            ],
            example_tags=["physician_agreed_category:consumer"],
        )

        decision = classify_task_compatibility(example)

        self.assertFalse(decision.selected)
        self.assertEqual(decision.category, "excluded")

    def test_filter_marks_secondary_documentation_task(self) -> None:
        example = _example(
            dialogue="I'm an informatics nurse building a text alert in our EHR for critically low potassium.",
            ideal_text="Critical potassium alert: repeat potassium, get ECG, replace potassium, escalate if no improvement.",
            expert_criteria=[
                (
                    "Provides evidence-based interventions and escalation steps for the EHR alert.",
                    6,
                    ["level:example", "axis:completeness"],
                )
            ],
            example_tags=["physician_agreed_category:health-professional"],
        )

        decision = classify_task_compatibility(example)

        self.assertFalse(decision.selected)
        self.assertEqual(decision.category, "secondary_documentation_task")

    def test_routing_keeps_note_slice_and_routes_note_profile(self) -> None:
        example = _example(
            dialogue="Draft a clinical note for a patient with fever and cough.",
            ideal_text=(
                "Subjective:\nFever and cough for 3 days.\n\n"
                "Objective:\nNo distress.\n\n"
                "Assessment and Plan:\nLikely viral illness. Supportive care.\n"
            ),
            expert_criteria=[
                (
                    "Provides a clinical note with grounded assessment and plan.",
                    6,
                    ["level:example", "axis:instruction_following"],
                )
            ],
        )

        decision = route_healthbench_task(example)

        self.assertTrue(decision.selected)
        self.assertEqual(decision.task_profile_id, "note_documentation")
        self.assertTrue(decision.note_regression_selected)

    def test_routing_assigns_documentation_variants_for_alerts(self) -> None:
        example = _example(
            dialogue="Write an EHR alert for severe hyperkalemia with escalation guidance.",
            ideal_text="Critical alert: repeat potassium urgently, obtain ECG, and escalate immediately.",
            expert_criteria=[
                (
                    "Produces an EHR alert with concise escalation and action items.",
                    6,
                    ["level:example", "axis:instruction_following"],
                )
            ],
            example_tags=["physician_agreed_category:health-professional"],
        )

        decision = route_healthbench_task(example)

        self.assertTrue(decision.selected)
        self.assertEqual(decision.task_profile_id, "documentation_variants")
        self.assertEqual(decision.task_family_id, "alert_message")
        self.assertFalse(decision.note_regression_selected)

    def test_routing_prefers_clinical_decision_support_for_care_guidance(self) -> None:
        example = _example(
            dialogue=(
                "My throat hurts really bad and it's hard to swallow. What should I do? "
                "Can I just wait it out?"
            ),
            ideal_text=(
                "It is important to see a healthcare professional for diagnosis. In the meantime, stay "
                "hydrated and seek care urgently if symptoms worsen."
            ),
            expert_criteria=[
                (
                    "Encourages seeing a healthcare professional and includes clear warning signs.",
                    7,
                    ["level:example", "axis:completeness"],
                )
            ],
            example_tags=["physician_agreed_category:consumer"],
        )

        decision = route_healthbench_task(example)

        self.assertEqual(decision.task_profile_id, "clinical_decision_support")

    def test_routing_selects_general_instruction_tasks_in_full_batch(self) -> None:
        lay_example = _example(
            dialogue="What causes mild insomnia?",
            ideal_text="Mild insomnia can happen due to stress, caffeine, or schedule changes.",
            expert_criteria=[
                (
                    "Uses plain language and provides a concise explanation of common causes of insomnia.",
                    6,
                    ["level:example", "axis:communication_quality"],
                )
            ],
            example_tags=["physician_agreed_category:consumer"],
        )
        note_example = _example(
            dialogue="Draft an outpatient progress note for a patient with knee pain.",
            ideal_text=(
                "Subjective:\nKnee pain for 2 weeks.\n\n"
                "Objective:\nVitals normal.\n\n"
                "Assessment and Plan:\nLikely strain. Conservative care.\n"
            ),
            expert_criteria=[
                (
                    "Provides an outpatient progress note including chief complaint, HPI, assessment, and plan.",
                    6,
                    ["level:example", "axis:instruction_following"],
                )
            ],
        )
        note_example.prompt_id = "hb_2"

        selected, decisions, summary = route_healthbench_examples([lay_example, note_example])

        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["counts"]["selected"], 2)
        self.assertEqual(summary["counts"]["note_regression_selected"], 1)
        by_prompt = {decision.prompt_id: decision for decision in decisions}
        self.assertEqual(by_prompt["hb_1"].task_profile_id, "general_instruction_following")
        self.assertEqual(by_prompt["hb_2"].task_profile_id, "note_documentation")

    def test_infer_polarity_prefers_negative_points(self) -> None:
        polarity = infer_criterion_polarity(
            "Fails to provide a clinical note with an assessment and plan.",
            points=-9,
        )
        self.assertEqual(polarity, "negative")

    def test_map_expert_family_prefers_context_grounding(self) -> None:
        family = map_expert_rubric_family(
            HealthBenchExpertCriterion(
                criterion=(
                    "Does not seek context and writes details about allergies, review of symptoms, physical "
                    "examination, assessment, and plan that were not given by the user."
                ),
                points=-10,
                tags=["level:example", "axis:context_awareness"],
            )
        )
        self.assertEqual(family, "context_grounding")

    def test_map_generated_family_uses_dimension_first(self) -> None:
        family = map_generated_criterion_family(
            {
                "dimension": "medication_management",
                "label": "Grounded medication plan documented",
                "requirement": "The note preserves medication names and dosing guidance when discussed.",
            }
        )
        self.assertEqual(family, "medication_plan")

    def test_map_generated_family_handles_general_instruction_dimensions(self) -> None:
        family = map_generated_criterion_family(
            {
                "dimension": "instruction_adherence",
                "label": "Requested format followed",
                "requirement": "The response follows the user's requested output format.",
            }
        )
        self.assertEqual(family, "instruction_adherence")

    def test_map_generated_family_uses_specific_follow_up_over_generic_dimension(self) -> None:
        family = map_generated_criterion_family(
            {
                "dimension": "meaning_preservation",
                "label": "Include explicit reference to Fleischner Society guidelines",
                "requirement": "The rewrite should confirm follow-up intervals per the Fleischner guidelines.",
            }
        )
        self.assertEqual(family, "follow_up")

    def test_map_generated_family_routes_care_guidance_to_content_coverage(self) -> None:
        family = map_generated_criterion_family(
            {
                "dimension": "final_response_quality",
                "label": "Encourages consulting healthcare professional",
                "requirement": "The response explicitly advises consulting a healthcare professional for diagnosis and treatment.",
            }
        )
        self.assertEqual(family, "content_coverage")

    def test_family_mapping_does_not_match_short_tokens_inside_words(self) -> None:
        family = map_expert_rubric_family(
            HealthBenchExpertCriterion(
                criterion="Correctly adheres to the user's instructions by drafting a preoperative note.",
                points=6,
                tags=["level:example", "axis:instruction_following"],
            )
        )
        self.assertEqual(family, "structure_documentation")

    def test_family_mapping_uses_context_for_requesting_missing_task_details(self) -> None:
        family = map_expert_rubric_family(
            HealthBenchExpertCriterion(
                criterion="Requests a specific task before drafting a preoperative note.",
                points=7,
                tags=["level:example", "axis:context_awareness"],
            )
        )
        self.assertEqual(family, "context_grounding")

    def test_run_healthbench_evaluation_uses_recursive_discovery_wrapper(self) -> None:
        example = _example(
            dialogue="What should I do next, and when should I follow up if my cough gets worse?",
            ideal_text=(
                "See a healthcare professional. Follow up within 48 hours if you are not improving, "
                "and seek urgent care right away for worsening breathing trouble."
            ),
            expert_criteria=[
                (
                    "The response gives a specific follow-up interval for reassessment.",
                    8,
                    ["level:example"],
                )
            ],
            example_tags=["physician_agreed_category:consumer"],
        )
        example.prompt_id = "hb_recursive"
        pair_key = "pair_recursive_1"
        root_row = {
            "dimension": "content_coverage",
            "label": "Main next steps covered",
            "requirement": "The response covers the main next steps and follow-up guidance.",
            "severity_tier": "high",
            "rationale": "This is the broad parent criterion.",
            "example_id": "healthbench__hb_recursive",
            "pair_id": pair_key,
            "strong_candidate_id": "strong",
            "weak_candidate_id": "weak",
            "proposal_index": 0,
            "task_profile_id": "clinical_decision_support",
            "criterion_id": "root_criterion",
            "root_pair_id": pair_key,
            "recursion_depth": 0,
            "recursion_reason": "",
            "decomposition_source": "root_pair_discovery",
        }
        child_row = {
            "dimension": "follow_up",
            "label": "Give a specific follow-up interval",
            "requirement": "The response states a specific follow-up interval such as reassessment within 48 hours.",
            "severity_tier": "high",
            "rationale": "This is the recursive child criterion.",
            "example_id": "healthbench__hb_recursive",
            "pair_id": pair_key,
            "strong_candidate_id": "strong",
            "weak_candidate_id": "weak",
            "proposal_index": 0,
            "task_profile_id": "clinical_decision_support",
            "criterion_id": "child_criterion",
            "parent_criterion_id": "root_criterion",
            "root_pair_id": pair_key,
            "recursion_depth": 1,
            "recursion_reason": "generic_dimension_broad_requirement",
            "decomposition_source": "criterion_decomposition",
        }
        pair_result = {
            "pair_id": pair_key,
            "raw_proposals": [root_row],
            "promoted_root_proposals": [root_row],
            "rejected_root_proposals": [],
            "proposals": [child_row],
            "rejected_proposals": [],
            "grounding": {"mode": "unfiltered", "accepted_count": 1, "rejected_count": 0},
            "cache": "miss",
            "parse_error": None,
            "recursive_steps": [
                {
                    "parent_criterion_id": "root_criterion",
                    "changed_structure": True,
                    "promoted_child_proposals": [child_row],
                    "rejected_child_proposals": [],
                }
            ],
            "recursion": {
                "recursive_calls": 1,
                "recursive_cache_hits": 0,
                "recursive_parse_failures": 0,
                "recursive_parents_considered": 1,
                "recursive_parents_expanded": 1,
                "recursive_children_raw_total": 1,
                "recursive_children_promoted": 1,
                "recursive_children_rejected_grounding": 0,
                "changed_structure": True,
            },
            "raw_proposals_total": 2,
            "promoted_proposals_total": 1,
            "rejected_proposals_total": 0,
        }
        gold_rows = [
            {
                "criterion_id": "g1",
                "criterion": "The response gives a specific follow-up interval for reassessment.",
                "points": 8,
                "polarity": "positive",
                "family": "follow_up",
                "tags": ["level:example"],
                "metadata": {},
            }
        ]

        with TemporaryDirectory() as tmpdir, patch(
            "rubric_gen.compiled.healthbench_eval.load_healthbench_dataset",
            return_value=({"schema": "test"}, [example]),
        ), patch(
            "rubric_gen.compiled.healthbench_eval.select_healthbench_subset",
            return_value=(
                [],
                [SubsetDecision(prompt_id=example.prompt_id, selected=False, category="excluded")],
                {"schema": "subset_test", "counts": {"selected": 0}},
            ),
        ), patch(
            "rubric_gen.compiled.healthbench_eval.route_healthbench_examples",
            return_value=(
                [example],
                [
                    HealthBenchRoutingDecision(
                        prompt_id=example.prompt_id,
                        selected=True,
                        category="clinical_decision_support",
                        task_profile_id="clinical_decision_support",
                        task_family_id="recommendation_plan",
                        artifact_kind="response",
                        route_confidence="high",
                    )
                ],
                {
                    "schema": "routing_test",
                    "counts": {"selected": 1, "note_regression_selected": 0},
                    "task_profile_counts": {"clinical_decision_support": 1},
                },
            ),
        ), patch(
            "rubric_gen.compiled.healthbench_eval.resolve_compiled_judge_spec",
            return_value=_model_spec(),
        ), patch(
            "rubric_gen.compiled.healthbench_eval._resolve_alignment_spec",
            return_value=None,
        ), patch(
            "rubric_gen.compiled.healthbench_eval.discover_pair_criteria",
            return_value=pair_result,
        ) as mock_discover, patch(
            "rubric_gen.compiled.healthbench_eval.classify_granularity_gaps",
            return_value={
                "schema": "compiled_gold_granularity_report_v1",
                "provider_id": "healthbench",
                "prompt_id": example.prompt_id,
                "task_profile_id": "clinical_decision_support",
                "gap_counts": {},
                "family_gap_counts": {},
                "gaps": [],
            },
        ), patch.object(
            run_healthbench_evaluation.__globals__["HealthBenchGoldProvider"],
            "compare_generated_rows",
            return_value=_alignment_artifact(gold_rows=gold_rows, generated_rows=[child_row]),
        ):
            run_dir, summary = run_healthbench_evaluation(
                dataset_path=Path(tmpdir) / "dataset.json",
                run_dir=Path(tmpdir) / "run",
                start=0,
                limit=0,
                discovery_model_override="test:test-model",
                alignment_model_override=None,
                adjudication_model_override=None,
                use_cache=False,
                max_criteria=4,
                max_pairs_per_example=1,
                disagreement_sample_size=0,
                refine_iterations=0,
                emit_calibration_hints=False,
            )
            payload = json.loads(
                (run_dir / "discovery" / "examples" / f"{example.prompt_id}.json").read_text(encoding="utf-8")
            )

        mock_discover.assert_called_once()
        self.assertEqual(summary["discovery"]["recursive_calls"], 1)
        self.assertEqual(summary["discovery"]["recursive_children_promoted"], 1)
        self.assertEqual(summary["discovery"]["examples_with_recursive_change"], 1)
        self.assertTrue(payload["recursive_discovery"]["changed_structure"])
        self.assertEqual(payload["pairs"][0]["recursion"]["recursive_calls"], 1)


if __name__ == "__main__":
    unittest.main()
