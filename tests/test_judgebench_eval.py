import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rubric_gen.compiled.contrast_strategies import ContrastStrategy, mutation_function_for_id
from rubric_gen.compiled.discovery import RecursiveDiscoveryConfig
from rubric_gen.compiled.judgebench_eval import (
    JudgeBenchJoinedExample,
    JudgeBenchRouteDecision,
    _apply_pair_discriminator_result,
    _example_choice_value_map,
    _example_for_reference_access,
    _example_is_exact_answer_task,
    _build_retrieval_guidance,
    _retrieval_similarity,
    _build_mechanism_spec,
    _mechanism_hash,
    _policy_core_hash,
    _apply_weight_adjustments,
    _prepare_rows_for_scoring,
    _policy_recursion_config,
    _score_pair_with_weights,
    _should_run_blind_pair_discriminator,
    _extract_exact_answer_candidate_for_example,
    build_calibration_guidance,
    build_balanced_judgebench_folds,
    build_judgebench_candidates,
    build_initial_frozen_policy,
    evaluate_rubric_satisfaction,
    joined_example_to_example_record,
    render_satisfaction_prompt,
    run_judgebench_final_evaluation,
    run_judgebench_train_only_development,
    canonical_rows_to_rubrics,
    score_discovered_rubrics_for_pair,
)
from rubric_gen.storage import JsonlCache, make_cache_key
from rubric_gen.types import CandidateNote, ModelSpec, RubricCriterion, RubricEvaluation


def _exact_answer_example() -> JudgeBenchJoinedExample:
    return JudgeBenchJoinedExample(
        split_name="validation_270",
        pair_id="pair_exact",
        source="mmlu-pro-math",
        source_family="mmlu-pro",
        question=(
            "What is the correct option? Once you have your answer, duplicate that letter five times in a single string."
        ),
        reference_answer="BBBBB",
        response_model="test-model",
        response_A="BBBBB",
        response_B="CCCCC",
        label="A>B",
        original_id="orig_1",
    )


def _strict_reasoning_example() -> JudgeBenchJoinedExample:
    return JudgeBenchJoinedExample(
        split_name="validation_270",
        pair_id="pair_reasoning",
        source="livebench-reasoning-logic",
        source_family="livebench-reasoning",
        question="Solve the puzzle and return a single digit in the format ***N***.",
        reference_answer="***3***",
        response_model="test-model",
        response_A="Answer: ***2***",
        response_B="Answer: ***3***",
        label="B>A",
        original_id="orig_2",
    )


def _generic_pair_example() -> JudgeBenchJoinedExample:
    return JudgeBenchJoinedExample(
        split_name="validation_270",
        pair_id="pair_generic",
        source="livebench-reasoning-open",
        source_family="livebench-reasoning",
        question="Choose the better response.",
        reference_answer="reference",
        response_model="test-model",
        response_A="A",
        response_B="B",
        label="A>B",
        original_id="orig_3",
    )


def _code_like_reasoning_example() -> JudgeBenchJoinedExample:
    return JudgeBenchJoinedExample(
        split_name="validation_270",
        pair_id="pair_reasoning_code_like",
        source="livebench-reasoning-logic",
        source_family="livebench-reasoning",
        question=(
            "Solve the logic puzzle. Track each clue update, respect all constraints, and report the final assignment.\n"
            "Input: the clue list.\n"
            "Output: the final occupant-to-role mapping."
        ),
        reference_answer="The chef is in seat 4.",
        response_model="test-model",
        response_A="Seat 4 is the chef because the last two clues force that assignment.",
        response_B="Seat 4 remains UNKNOWN, so the final mapping is incomplete.",
        label="A>B",
        original_id="orig_reasoning_code_like",
    )


def _mmlu_conflict_example() -> JudgeBenchJoinedExample:
    return JudgeBenchJoinedExample(
        split_name="validation_270",
        pair_id="pair_mmlu_conflict",
        source="mmlu-pro-clinical",
        source_family="mmlu-pro",
        question=(
            "Pick the best answer. Once you have your answer, duplicate that letter five times in a single string."
        ),
        reference_answer="DDDDD",
        response_model="test-model",
        response_A="Reasoning supports option E.\nFinal answer: EEEEE",
        response_B="Brief explanation.\nFinal answer: DDDDD",
        label="A>B",
        original_id="orig_4",
    )


def _choice_map_exact_answer_example() -> JudgeBenchJoinedExample:
    return JudgeBenchJoinedExample(
        split_name="validation_270",
        pair_id="pair_choice_map_exact",
        source="mmlu-pro-math",
        source_family="mmlu-pro",
        question=(
            "Solve the problem and pick the correct option. (A) 30 (B) 45 (C) 60 "
            "Once you have your answer, duplicate that letter five times in a single string."
        ),
        reference_answer="AAAAA",
        response_model="test-model",
        response_A="We compute 30.\nFinal answer: AAAAA",
        response_B="We compute 30.\nFinal answer: CCCCC",
        label="A>B",
        original_id="orig_choice_map",
    )


def _livecode_example() -> JudgeBenchJoinedExample:
    return JudgeBenchJoinedExample(
        split_name="validation_270",
        pair_id="pair_code",
        source="livecodebench",
        source_family="livecodebench",
        question="Write code that checks whether a 3-digit number is fascinating.",
        reference_answer="class Solution:\n    def isFascinating(self, n: int) -> bool:\n        s = str(n) + str(2 * n) + str(3 * n)\n        return len(s) == 9 and set(s) == set('123456789')",
        response_model="test-model",
        response_A="class Solution:\n    pass",
        response_B="class Solution:\n    pass",
        label="A>B",
        original_id="orig_5",
    )


def _candidate(example: JudgeBenchJoinedExample, pair_position: str, text: str) -> CandidateNote:
    return CandidateNote(
        candidate_id=f"{example.pair_id}__{pair_position}",
        example_id=example.pair_id,
        text=text,
        source_label=f"response_{pair_position}",
        quality_bucket="pair_candidate",
        origin_kind="judgebench_pair",
        artifact_kind="response",
        task_profile_id="test_profile",
        task_family_id="test_family",
    )


class JudgeBenchEvalTests(unittest.TestCase):
    def test_build_initial_policy_generic_baseline_disables_family_tuning(self) -> None:
        example = _exact_answer_example()
        policy = build_initial_frozen_policy(
            train_examples=[example],
            bootstrap_iterations=1,
            recursive_config=RecursiveDiscoveryConfig(
                max_depth=1,
                max_recursive_parents_per_pair=2,
                max_children_per_parent=3,
                max_recursive_calls_per_pair=2,
            ),
            protocol_mode="generic_baseline",
        )

        self.assertEqual(policy["protocol_mode"], "generic_baseline")
        self.assertEqual(policy["blind_scoring_profile"], "baseline")
        self.assertEqual(policy["blind_budget_profile"], "family_v1")
        self.assertEqual(policy["blind_guidance_profile"], "off")
        self.assertEqual(policy["blind_wu_profile"], "raw")
        self.assertEqual(policy["retrieval_profile"], "off")
        self.assertEqual(policy["family_recursion_config"], {})
        self.assertEqual(policy["prompt_nudges"], {"global": []})

    def test_generic_baseline_blind_guidance_profile_enables_calibration_guidance(self) -> None:
        example = _strict_reasoning_example()
        policy = build_initial_frozen_policy(
            train_examples=[example],
            bootstrap_iterations=1,
            recursive_config=RecursiveDiscoveryConfig(
                max_depth=1,
                max_recursive_parents_per_pair=2,
                max_children_per_parent=3,
                max_recursive_calls_per_pair=2,
            ),
            protocol_mode="generic_baseline",
            blind_guidance_profile="family_v2",
        )

        guidance = build_calibration_guidance(
            policy,
            example.source_family,
            example=JudgeBenchJoinedExample(**{**example.__dict__, "reference_answer": ""}),
            reference_answer_access=False,
        )

        self.assertIn("Additional calibration guidance", guidance)
        self.assertIn("Avoid broad completeness or correctness rubrics", guidance)
        self.assertIn("constraint satisfaction", guidance)

    def test_canonical_rows_tag_letter_correctness_and_add_fallback_if_missing(self) -> None:
        example = _exact_answer_example()
        correctness_rows = [
            {
                "dimension": "instruction_adherence",
                "label": "Final answer letter correctness",
                "requirement": "The response must select the letter 'B' as the final answer corresponding to the calculated value.",
                "severity_tier": "hard_gate",
            }
        ]
        rubrics = canonical_rows_to_rubrics("pair_exact", correctness_rows, example=example)

        self.assertEqual(len(rubrics), 1)
        self.assertEqual(rubrics[0].metadata.get("exact_answer_kind"), "correctness")

        format_rows = [
            {
                "dimension": "instruction_adherence",
                "label": "Final answer format correctness",
                "requirement": "The response must provide the final answer as a single letter repeated five times in a single string.",
                "severity_tier": "hard_gate",
            }
        ]
        rubrics = canonical_rows_to_rubrics("pair_exact", format_rows, example=example)

        kinds = [rubric.metadata.get("exact_answer_kind") for rubric in rubrics]
        self.assertIn("format", kinds)
        self.assertIn("correctness", kinds)

    def test_exact_answer_tie_break_uses_extracted_reference_match(self) -> None:
        example = _strict_reasoning_example()
        rows = [
            {
                "dimension": "instruction_adherence",
                "label": "Answer format correctness",
                "requirement": "The response must present the final answer in the format ***N***.",
                "severity_tier": "hard_gate",
            }
        ]
        rubrics = canonical_rows_to_rubrics("pair_reasoning", rows, example=example)
        candidate_a = _candidate(example, "A", example.response_A)
        candidate_b = _candidate(example, "B", example.response_B)
        evaluations = [
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=True),
        ]
        weights = {rubric.rubric_id: 1.0 / len(rubrics) for rubric in rubrics}

        result = _score_pair_with_weights(
            example=example,
            rubrics=rubrics,
            evaluations=evaluations,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            weights=weights,
        )

        self.assertEqual(result["decision"], "B>A")
        self.assertEqual(result["tie_break_reason"], "exact_answer_reference_match")
        self.assertEqual(result["exact_answer_value_A"], "2")
        self.assertEqual(result["exact_answer_value_B"], "3")

    def test_hidden_reference_preserves_private_exact_answer_features(self) -> None:
        example = _choice_map_exact_answer_example()
        hidden = _example_for_reference_access(example, reference_answer_access=False)

        self.assertEqual(hidden.reference_answer, "")
        self.assertTrue(_example_is_exact_answer_task(hidden))
        choice_map = _example_choice_value_map(hidden)
        self.assertEqual(choice_map["a"], "30")
        self.assertEqual(choice_map["b"], "45")
        self.assertIn("c", choice_map)
        extraction = _extract_exact_answer_candidate_for_example(hidden, "We compute 30.\nFinal answer: AAAAA")
        self.assertIsNotNone(extraction)
        self.assertEqual(extraction.value, "aaaaa")

    def test_evaluate_rubric_satisfaction_refreshes_poisoned_cached_response(self) -> None:
        class FakeRouter:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, *args, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    text="<EVALUATION> YES </EVALUATION>",
                    raw_text="<EVALUATION> YES </EVALUATION>",
                    metadata={},
                )

        with tempfile.TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "scoring.jsonl", enabled=True)
            judge_model = ModelSpec(
                alias="test",
                provider="openai",
                model="gpt-5.4",
                api_key_env="TEST_API_KEY",
            )
            response_text = "Candidate text"
            rubric_text = "The answer must mention blue."
            prompt = render_satisfaction_prompt(response_text, rubric_text)
            cache_key = make_cache_key(
                "rubric_satisfaction",
                {
                    "model": {
                        "provider": judge_model.provider,
                        "model": judge_model.model,
                        "base_url": judge_model.base_url or "",
                    },
                    "system_prompt": "",
                    "user_prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": 64,
                },
            )
            cache.set(cache_key, {"text": "IIII", "raw_text": "IIII", "metadata": {}})

            router = FakeRouter()
            verdict, meta = evaluate_rubric_satisfaction(
                response_text=response_text,
                rubric_text=rubric_text,
                judge_model=judge_model,
                cache=cache,
                router=router,
            )

            self.assertTrue(verdict)
            self.assertEqual(router.calls, 1)
            self.assertEqual(meta["attempt_index"], 0)
            self.assertEqual(meta["cache_hits"], 1)
            self.assertEqual(meta["raw_response"], "<EVALUATION> YES </EVALUATION>")
            self.assertEqual(cache.get(cache_key)["text"], "<EVALUATION> YES </EVALUATION>")

    def test_exact_answer_presence_requires_explicit_final_answer_span(self) -> None:
        example = _exact_answer_example()
        rows = [
            {
                "dimension": "instruction_adherence",
                "label": "Final answer format correctness",
                "requirement": "The response must provide the final answer as a single repeated-letter string.",
                "severity_tier": "hard_gate",
            }
        ]
        rubrics = canonical_rows_to_rubrics("pair_exact_presence", rows, example=example)
        candidate_a = _candidate(example, "A", "I think the answer is probably B, but I am not certain.")
        candidate_b = _candidate(example, "B", "A mnemonic appears here: CCCCC, but there is no explicit final answer.")
        evaluations = [
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=True),
        ]
        weights = {rubric.rubric_id: 1.0 / len(rubrics) for rubric in rubrics}

        result = _score_pair_with_weights(
            example=example,
            rubrics=rubrics,
            evaluations=evaluations,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            weights=weights,
        )

        self.assertEqual(result["decision"], "A=B")
        self.assertEqual(result["tie_break_reason"], "")
        self.assertTrue(result["exact_answer_value_B"])
        self.assertFalse(result["exact_answer_explicit_B"])

    def test_mmlu_explicit_reference_match_is_not_overridden_by_non_exact_score(self) -> None:
        example = _mmlu_conflict_example()
        rows = [
            {
                "dimension": "final_answer_correctness",
                "label": "Correct final answer selection",
                "requirement": "The response must select the correct repeated-letter answer DDDDD.",
                "severity_tier": "hard_gate",
            },
            {
                "dimension": "reasoning_support",
                "label": "Reasoning supports option E",
                "requirement": "The response justifies why option E is preferred based on the task evidence.",
                "severity_tier": "high",
            },
            {
                "dimension": "reasoning_support",
                "label": "Explains why option E is better",
                "requirement": "The response gives a grounded explanation for choosing option E over the alternatives.",
                "severity_tier": "medium",
            },
        ]
        rubrics = canonical_rows_to_rubrics("pair_mmlu_conflict", rows, example=example)
        candidate_a = _candidate(example, "A", example.response_A)
        candidate_b = _candidate(example, "B", example.response_B)
        evaluations = [
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[1].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[1].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id=rubrics[2].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[2].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=False),
        ]
        weights = {
            rubrics[0].rubric_id: 0.4,
            rubrics[1].rubric_id: 0.3,
            rubrics[2].rubric_id: 0.3,
        }

        result = _score_pair_with_weights(
            example=example,
            rubrics=rubrics,
            evaluations=evaluations,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            weights=weights,
        )

        self.assertEqual(result["decision"], "B>A")
        self.assertEqual(result["tie_break_reason"], "")
        self.assertTrue(result["exact_answer_match_B"])

    def test_mmlu_nonexplicit_reference_match_is_not_overridden_by_non_exact_score(self) -> None:
        example = JudgeBenchJoinedExample(
            split_name="validation_270",
            pair_id="pair_mmlu_nonexplicit_conflict",
            source="mmlu-pro-ethics",
            source_family="mmlu-pro",
            question=(
                "Pick the best answer. Once you have your answer, duplicate that letter five times in a single string."
            ),
            reference_answer="CCCCC",
            response_model="test-model",
            response_A="Reasoning supports option A.",
            response_B="A possible string is CCCCC, but I will not state a final answer explicitly.",
            label="A>B",
            original_id="orig_6",
        )
        rows = [
            {
                "dimension": "final_answer_correctness",
                "label": "Correct final answer selection",
                "requirement": "The response must select the correct repeated-letter answer CCCCC.",
                "severity_tier": "hard_gate",
            },
            {
                "dimension": "reasoning_support",
                "label": "Reasoning supports option A",
                "requirement": "The response gives grounded reasoning for choosing option A.",
                "severity_tier": "high",
            },
        ]
        rubrics = canonical_rows_to_rubrics("pair_mmlu_nonexplicit_conflict", rows, example=example)
        candidate_a = _candidate(example, "A", example.response_A)
        candidate_b = _candidate(example, "B", example.response_B)
        evaluations = [
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[1].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[1].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=False),
        ]
        weights = {
            rubrics[0].rubric_id: 0.52,
            rubrics[1].rubric_id: 0.48,
        }

        result = _score_pair_with_weights(
            example=example,
            rubrics=rubrics,
            evaluations=evaluations,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            weights=weights,
        )

        self.assertEqual(result["decision"], "B>A")
        self.assertEqual(result["tie_break_reason"], "")
        self.assertFalse(result["exact_answer_explicit_B"])

    def test_code_proxy_rubrics_are_downweighted(self) -> None:
        example = _livecode_example()
        rows = [
            {
                "dimension": "executable_correctness",
                "label": "Use explicit digit presence check instead of set equality",
                "requirement": "The solution should use an explicit digit presence check instead of set equality.",
                "severity_tier": "medium",
            },
            {
                "dimension": "executable_correctness",
                "label": "Checks all 8 directions for the target",
                "requirement": "The solution must check all 8 directions when searching for the target sequence.",
                "severity_tier": "high",
            },
        ]
        rubrics = canonical_rows_to_rubrics("pair_code_weights", rows, example=example)
        adjusted = _apply_weight_adjustments(
            example=example,
            rubrics=rubrics,
            weights={rubrics[0].rubric_id: 0.5, rubrics[1].rubric_id: 0.5},
        )

        self.assertLess(adjusted[rubrics[0].rubric_id], adjusted[rubrics[1].rubric_id])

    def test_generic_baseline_skips_exact_answer_fallback_and_tie_breaks(self) -> None:
        example = _strict_reasoning_example()
        rows = [
            {
                "dimension": "instruction_adherence",
                "label": "Answer format correctness",
                "requirement": "The response must present the final answer in the format ***N***.",
                "severity_tier": "hard_gate",
            }
        ]
        rubrics = canonical_rows_to_rubrics(
            "pair_reasoning_generic",
            rows,
            example=example,
            protocol_mode="generic_baseline",
        )
        candidate_a = _candidate(example, "A", example.response_A)
        candidate_b = _candidate(example, "B", example.response_B)
        evaluations = [
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=True),
        ]

        self.assertEqual(len(rubrics), 1)
        self.assertEqual(rubrics[0].metadata.get("exact_answer_kind"), "")

        result = _score_pair_with_weights(
            example=example,
            rubrics=rubrics,
            evaluations=evaluations,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            weights={rubrics[0].rubric_id: 1.0},
            protocol_mode="generic_baseline",
        )

        self.assertEqual(result["decision"], "A=B")
        self.assertEqual(result["tie_break_reason"], "")

    def test_prepare_rows_for_scoring_adds_blind_exact_signal_rows(self) -> None:
        example = _choice_map_exact_answer_example()
        pair_candidates = [
            _candidate(example, "A", example.response_A),
            _candidate(example, "B", "We compute 30.\nFinal answer: C"),
        ]

        prepared = _prepare_rows_for_scoring(
            [],
            example=example,
            pair_candidates=pair_candidates,
            policy={"protocol_mode": "generic_baseline", "blind_scoring_profile": "pruned_v2"},
            reference_answer_access=False,
        )

        signal_kinds = {row.get("blind_exact_signal_kind") for row in prepared}
        self.assertIn("format", signal_kinds)
        self.assertIn("consistency", signal_kinds)
        self.assertIn("choice_value_consistency", signal_kinds)

        rubrics = canonical_rows_to_rubrics(
            "pair_choice_map_generic",
            prepared,
            example=example,
            protocol_mode="generic_baseline",
        )
        rubric_kinds = {
            rubric.metadata.get("blind_exact_signal_kind"): rubric.metadata.get("exact_answer_kind")
            for rubric in rubrics
            if rubric.metadata.get("blind_exact_signal_kind")
        }
        self.assertEqual(rubric_kinds["format"], "format")
        self.assertEqual(rubric_kinds["consistency"], "correctness")
        self.assertEqual(rubric_kinds["choice_value_consistency"], "correctness")

    def test_prepare_rows_for_scoring_skips_nondifferentiating_blind_exact_rows(self) -> None:
        example = _exact_answer_example()
        pair_candidates = [
            _candidate(example, "A", "Final answer: BBBBB"),
            _candidate(example, "B", "Final answer: CCCCC"),
        ]

        prepared = _prepare_rows_for_scoring(
            [],
            example=example,
            pair_candidates=pair_candidates,
            policy={"protocol_mode": "generic_baseline", "blind_scoring_profile": "pruned_v2"},
            reference_answer_access=False,
        )

        signal_kinds = {row.get("blind_exact_signal_kind") for row in prepared if row.get("blind_exact_signal_kind")}
        self.assertEqual(signal_kinds, set())

    def test_generic_baseline_tie_breaks_with_blind_exact_signal(self) -> None:
        example = _choice_map_exact_answer_example()
        rows = [
            {
                "dimension": "reasoning_support",
                "label": "Explains the calculation",
                "requirement": "The response gives a worked explanation for the chosen answer.",
                "severity_tier": "high",
            }
        ]
        rubrics = canonical_rows_to_rubrics(
            "pair_choice_map_scoring",
            rows,
            example=example,
            protocol_mode="generic_baseline",
        )
        candidate_a = _candidate(example, "A", example.response_A)
        candidate_b = _candidate(example, "B", example.response_B)
        evaluations = [
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id=rubrics[0].rubric_id, candidate_id=candidate_b.candidate_id, satisfied=True),
        ]

        result = _score_pair_with_weights(
            example=example,
            rubrics=rubrics,
            evaluations=evaluations,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            weights={rubrics[0].rubric_id: 1.0},
            protocol_mode="generic_baseline",
        )

        self.assertEqual(result["decision"], "A>B")
        self.assertEqual(result["tie_break_reason"], "blind_exact_answer_signal")
        self.assertTrue(result["blind_exact_signal_A"]["consistent"])
        self.assertFalse(result["blind_exact_signal_B"]["consistent"])

    def test_retrieval_seed_profile_returns_seed_rows(self) -> None:
        query = _choice_map_exact_answer_example()
        query.pair_id = "query_pair"
        retrieved = _choice_map_exact_answer_example()
        retrieved.pair_id = "retrieved_pair"

        guidance, hits, seed_rows = _build_retrieval_guidance(
            example=query,
            retrieval_examples=[retrieved],
            policy={"retrieval_profile": "family_question_seed_v1", "retrieval_top_k": 1},
        )

        self.assertIn("rubric seeding", guidance)
        self.assertEqual(len(hits), 1)
        self.assertIn("exact_answer_consistency", hits[0]["focus_kinds"])
        self.assertTrue(any(row.get("retrieval_seed_row") for row in seed_rows))
        self.assertTrue(any(row.get("blind_exact_signal_kind") == "consistency" for row in seed_rows))

    def test_retrieval_v2_profile_stays_calibration_only(self) -> None:
        query = _choice_map_exact_answer_example()
        query.pair_id = "query_pair"
        retrieved = _choice_map_exact_answer_example()
        retrieved.pair_id = "retrieved_pair"

        guidance, hits, seed_rows = _build_retrieval_guidance(
            example=query,
            retrieval_examples=[retrieved],
            policy={"retrieval_profile": "family_question_v2", "retrieval_top_k": 1},
        )

        self.assertIn("preferred response", guidance)
        self.assertEqual(len(hits), 1)
        self.assertEqual(seed_rows, [])
        self.assertIn("similarity_components", hits[0])

    def test_retrieval_family_override_can_enable_single_family(self) -> None:
        query = _choice_map_exact_answer_example()
        query.pair_id = "query_pair"
        retrieved_one = _choice_map_exact_answer_example()
        retrieved_one.pair_id = "retrieved_one"
        retrieved_two = _choice_map_exact_answer_example()
        retrieved_two.pair_id = "retrieved_two"
        code_query = _livecode_example()
        code_query.pair_id = "code_query"

        guidance, hits, seed_rows = _build_retrieval_guidance(
            example=query,
            retrieval_examples=[retrieved_one, retrieved_two],
            policy={
                "retrieval_profile": "off",
                "retrieval_top_k": 2,
                "retrieval_profile_by_family": {"mmlu-pro": "family_question_v1"},
                "retrieval_top_k_by_family": {"mmlu-pro": 1},
            },
        )
        code_guidance, code_hits, code_seed_rows = _build_retrieval_guidance(
            example=code_query,
            retrieval_examples=[retrieved_one, retrieved_two],
            policy={
                "retrieval_profile": "off",
                "retrieval_top_k": 2,
                "retrieval_profile_by_family": {"mmlu-pro": "family_question_v1"},
                "retrieval_top_k_by_family": {"mmlu-pro": 1},
            },
        )

        self.assertIn("Retrieved training exemplars", guidance)
        self.assertEqual(len(hits), 1)
        self.assertEqual(seed_rows, [])
        self.assertEqual(code_guidance, "")
        self.assertEqual(code_hits, [])
        self.assertEqual(code_seed_rows, [])

    def test_blind_pair_discriminator_family_modes_can_disable_or_tighten(self) -> None:
        scoring = {
            "whitened_uniform": {
                "result": {
                    "decision": "A>B",
                    "score_A": 0.5015,
                    "score_B": 0.4995,
                    "whitening_unstable": True,
                }
            }
        }

        self.assertFalse(
            _should_run_blind_pair_discriminator(
                policy={
                    "blind_scoring_profile": "pruned_disc_v1",
                    "blind_discriminator_mode_by_family": {"mmlu-pro": "off"},
                },
                example=_exact_answer_example(),
                scoring=scoring,
                rubric_count=18,
            )
        )

        reasoning_example = _generic_pair_example()
        self.assertFalse(
            _should_run_blind_pair_discriminator(
                policy={
                    "blind_scoring_profile": "pruned_disc_v1",
                    "blind_discriminator_mode_by_family": {"livebench-reasoning": "strict"},
                },
                example=reasoning_example,
                scoring=scoring,
                rubric_count=15,
            )
        )
        self.assertTrue(
            _should_run_blind_pair_discriminator(
                policy={
                    "blind_scoring_profile": "pruned_disc_v1",
                    "blind_discriminator_mode_by_family": {"livebench-reasoning": "strict"},
                },
                example=reasoning_example,
                scoring=scoring,
                rubric_count=16,
            )
        )

    def test_blind_pair_discriminator_routes_small_margin_reasoning_code_tasks(self) -> None:
        scoring = {
            "whitened_uniform": {
                "result": {
                    "decision": "B>A",
                    "score_A": 0.495,
                    "score_B": 0.501,
                    "whitening_unstable": False,
                }
            }
        }

        self.assertTrue(
            _should_run_blind_pair_discriminator(
                policy={
                    "blind_scoring_profile": "pruned_disc_v1",
                    "blind_discriminator_mode_by_family": {"livebench-reasoning": "strict"},
                },
                example=_code_like_reasoning_example(),
                scoring=scoring,
                rubric_count=14,
            )
        )

    def test_blind_pair_discriminator_does_not_expand_exact_answer_reasoning_route(self) -> None:
        scoring = {
            "whitened_uniform": {
                "result": {
                    "decision": "A>B",
                    "score_A": 0.501,
                    "score_B": 0.495,
                    "whitening_unstable": False,
                }
            }
        }

        self.assertFalse(
            _should_run_blind_pair_discriminator(
                policy={
                    "blind_scoring_profile": "pruned_disc_v1",
                    "blind_discriminator_mode_by_family": {"livebench-reasoning": "strict"},
                },
                example=_strict_reasoning_example(),
                scoring=scoring,
                rubric_count=18,
            )
        )

    def test_retrieval_v2_mmlu_prefers_lexically_closer_exemplar(self) -> None:
        query = JudgeBenchJoinedExample(
            split_name="train_80",
            pair_id="query_pair",
            source="mmlu-pro-engineering",
            source_family="mmlu-pro",
            question=(
                "A pair of mating spur gears increases the speed of the shaft approximately by 4 times. "
                "Determine the actual velocity ratio and center distance. "
                "(A) 4.5, 12.5 in. (B) 3.948, 12.0 in. (C) 4.125, 11.5 in. "
                "If you cannot determine the correct multiple-choice answer, take your best guess."
            ),
            reference_answer="B",
            response_model="test-model",
            response_A="A",
            response_B="B",
            label="A>B",
            original_id="orig_query",
        )
        lexically_closer = JudgeBenchJoinedExample(
            split_name="train_80",
            pair_id="candidate_close",
            source="mmlu-pro-physics",
            source_family="mmlu-pro",
            question=(
                "When 100 g of water at 0 C are mixed with 50 g of water at 50 C, what is the change of entropy on "
                "mixing? (A) 0.8 cal/K/deg (B) 0.5 cal/K/deg (C) 0.3 cal/K/deg "
                "If you cannot determine the correct multiple-choice answer, take your best guess."
            ),
            reference_answer="A",
            response_model="test-model",
            response_A="A",
            response_B="B",
            label="A>B",
            original_id="orig_close",
        )
        same_subdomain = JudgeBenchJoinedExample(
            split_name="train_80",
            pair_id="candidate_same_source",
            source="mmlu-pro-engineering",
            source_family="mmlu-pro",
            question=(
                "A 4 in. schedule 40 wrought iron pipe is covered with 2 in. thick layer of magnesia insulation. "
                "Determine the rate of heat transfer through the insulation. "
                "(A) 120 Btu/hr (B) 220 Btu/hr (C) 320 Btu/hr "
                "If you cannot determine the correct multiple-choice answer, take your best guess."
            ),
            reference_answer="B",
            response_model="test-model",
            response_A="A",
            response_B="B",
            label="A>B",
            original_id="orig_same_source",
        )

        closer_score = _retrieval_similarity(
            query,
            lexically_closer,
            retrieval_profile="family_question_v2",
        )
        same_source_score = _retrieval_similarity(
            query,
            same_subdomain,
            retrieval_profile="family_question_v2",
        )

        self.assertGreater(closer_score, same_source_score)

    def test_pair_discriminator_updates_no_signal_ties(self) -> None:
        scoring = {
            "uniform": {
                "weights": {},
                "result": {
                    "decision": "A=B",
                    "decision_reversed": "A=B",
                    "decision_policy": "uniform",
                    "tie_break_reason": "",
                    "pair_evaluations": [
                        {
                            "rubric_id": "r1",
                            "response_A_satisfied": True,
                            "response_B_satisfied": True,
                            "severity_tier": "high",
                        }
                    ],
                },
            },
            "whitened_uniform": {
                "weights": {},
                "debug": {},
                "result": {
                    "decision": "A=B",
                    "decision_reversed": "A=B",
                    "decision_policy": "whitened_uniform",
                    "tie_break_reason": "",
                    "pair_evaluations": [
                        {
                            "rubric_id": "r1",
                            "response_A_satisfied": True,
                            "response_B_satisfied": True,
                            "severity_tier": "high",
                        }
                    ],
                },
            },
        }

        updated = _apply_pair_discriminator_result(
            scoring=scoring,
            discriminator={
                "decision": "B>A",
                "distinguishing_behavior": "Candidate B handles the required behavior correctly.",
                "confidence": "medium",
            },
        )

        self.assertEqual(updated["uniform"]["result"]["decision"], "B>A")
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "B>A")
        self.assertEqual(updated["uniform"]["result"]["tie_break_reason"], "pairwise_discriminator")

    def test_pair_discriminator_is_disabled_for_mmlu(self) -> None:
        scoring = {
            "uniform": {
                "weights": {},
                "result": {
                    "decision": "A=B",
                    "decision_reversed": "A=B",
                    "decision_policy": "uniform",
                    "tie_break_reason": "",
                    "pair_evaluations": [
                        {
                            "rubric_id": "r1",
                            "response_A_satisfied": True,
                            "response_B_satisfied": True,
                            "severity_tier": "high",
                        }
                    ],
                },
            },
            "whitened_uniform": {
                "weights": {},
                "debug": {},
                "result": {
                    "decision": "A=B",
                    "decision_reversed": "A=B",
                    "decision_policy": "whitened_uniform",
                    "tie_break_reason": "",
                    "pair_evaluations": [
                        {
                            "rubric_id": "r1",
                            "response_A_satisfied": True,
                            "response_B_satisfied": True,
                            "severity_tier": "high",
                        }
                    ],
                },
            },
        }

        updated = _apply_pair_discriminator_result(
            scoring=scoring,
            discriminator={
                "decision": "B>A",
                "distinguishing_behavior": "Candidate B handles the required behavior correctly.",
                "confidence": "medium",
            },
            source_family="mmlu-pro",
        )

        self.assertEqual(updated["uniform"]["result"]["decision"], "A=B")
        self.assertEqual(updated["whitened_uniform"]["result"]["decision"], "A=B")
        self.assertNotIn("pair_discriminator", updated)

    def test_policy_recursion_config_uses_family_specific_overrides(self) -> None:
        policy = {
            "recursion_config": {
                "max_depth": 1,
                "max_recursive_parents_per_pair": 2,
                "max_children_per_parent": 3,
                "max_recursive_calls_per_pair": 2,
            },
            "family_recursion_config": {
                "mmlu-pro": {
                    "max_depth": 2,
                    "max_recursive_parents_per_pair": 3,
                    "max_children_per_parent": 4,
                },
                "livecodebench": {
                    "max_depth": 1,
                    "max_recursive_parents_per_pair": 2,
                    "max_children_per_parent": 3,
                },
            },
        }

        self.assertEqual(
            _policy_recursion_config(policy, source_family="mmlu-pro"),
            {
                "max_depth": 2,
                "max_recursive_parents_per_pair": 3,
                "max_children_per_parent": 4,
                "max_recursive_calls_per_pair": 2,
            },
        )
        self.assertEqual(
            _policy_recursion_config(policy, source_family="livecodebench"),
            {
                "max_depth": 1,
                "max_recursive_parents_per_pair": 2,
                "max_children_per_parent": 3,
                "max_recursive_calls_per_pair": 2,
            },
        )

    def test_unstable_whitening_prefers_uniform_decision(self) -> None:
        example = _generic_pair_example()
        rubrics = [
            RubricCriterion(
                rubric_id="r1",
                text="Criterion one",
                source_stage="compiled_recursive",
                depth=0,
                round_index=0,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
            RubricCriterion(
                rubric_id="r2",
                text="Criterion two",
                source_stage="compiled_recursive",
                depth=0,
                round_index=1,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
            RubricCriterion(
                rubric_id="r3",
                text="Criterion three",
                source_stage="compiled_recursive",
                depth=0,
                round_index=2,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
        ]
        candidate_a = _candidate(example, "A", "response a")
        candidate_b = _candidate(example, "B", "response b")
        evaluations = [
            RubricEvaluation(rubric_id="r1", candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id="r2", candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id="r3", candidate_id=candidate_a.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r1", candidate_id=candidate_b.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r2", candidate_id=candidate_b.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r3", candidate_id=candidate_b.candidate_id, satisfied=True),
        ]

        with patch(
            "rubric_gen.compiled.judgebench_eval.compute_whitened_uniform_weights",
            return_value=(
                {"r1": 0.1, "r2": 0.1, "r3": 0.8},
                {"raw_weights": [0.1, 0.1, 0.8], "clipped_weights": [0.1, 0.1, 0.8], "eigenvalues": [0.001, 0.001, 0.5]},
            ),
        ):
            scoring = score_discovered_rubrics_for_pair(
                example=example,
                rubrics=rubrics,
                scoring_candidates=[candidate_a, candidate_b],
                pair_candidates=[candidate_a, candidate_b],
                evaluations=evaluations,
                covariance_ridge=1e-3,
            )

        self.assertEqual(scoring["uniform"]["result"]["decision"], "A>B")
        self.assertEqual(scoring["whitened_uniform"]["result"]["decision"], "A>B")
        self.assertEqual(
            scoring["whitened_uniform"]["result"]["decision_policy"],
            "uniform_overrides_unstable_whitening",
        )

    def test_generic_baseline_keeps_raw_whitened_uniform_decision(self) -> None:
        example = _generic_pair_example()
        rubrics = [
            RubricCriterion(
                rubric_id="r1",
                text="Criterion one",
                source_stage="compiled_recursive",
                depth=0,
                round_index=0,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
            RubricCriterion(
                rubric_id="r2",
                text="Criterion two",
                source_stage="compiled_recursive",
                depth=0,
                round_index=1,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
            RubricCriterion(
                rubric_id="r3",
                text="Criterion three",
                source_stage="compiled_recursive",
                depth=0,
                round_index=2,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
        ]
        candidate_a = _candidate(example, "A", "response a")
        candidate_b = _candidate(example, "B", "response b")
        evaluations = [
            RubricEvaluation(rubric_id="r1", candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id="r2", candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id="r3", candidate_id=candidate_a.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r1", candidate_id=candidate_b.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r2", candidate_id=candidate_b.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r3", candidate_id=candidate_b.candidate_id, satisfied=True),
        ]

        with patch(
            "rubric_gen.compiled.judgebench_eval.compute_whitened_uniform_weights",
            return_value=(
                {"r1": 0.1, "r2": 0.1, "r3": 0.8},
                {"raw_weights": [0.1, 0.1, 0.8], "clipped_weights": [0.1, 0.1, 0.8], "eigenvalues": [0.001, 0.001, 0.5]},
            ),
        ):
            scoring = score_discovered_rubrics_for_pair(
                example=example,
                rubrics=rubrics,
                scoring_candidates=[candidate_a, candidate_b],
                pair_candidates=[candidate_a, candidate_b],
                evaluations=evaluations,
                covariance_ridge=1e-3,
                protocol_mode="generic_baseline",
            )

        self.assertEqual(scoring["uniform"]["result"]["decision"], "A>B")
        self.assertEqual(scoring["whitened_uniform"]["result"]["decision"], "B>A")
        self.assertEqual(scoring["whitened_uniform"]["result"]["decision_policy"], "whitened_uniform")

    def test_generic_baseline_blind_wu_profile_can_override_unstable_low_margin_decision(self) -> None:
        example = _generic_pair_example()
        rubrics = [
            RubricCriterion(
                rubric_id="r1",
                text="Criterion one",
                source_stage="compiled_recursive",
                depth=0,
                round_index=0,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
            RubricCriterion(
                rubric_id="r2",
                text="Criterion two",
                source_stage="compiled_recursive",
                depth=0,
                round_index=1,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
            RubricCriterion(
                rubric_id="r3",
                text="Criterion three",
                source_stage="compiled_recursive",
                depth=0,
                round_index=2,
                metadata={"canonical_row": {"severity_tier": "high"}},
            ),
        ]
        candidate_a = _candidate(example, "A", "response a")
        candidate_b = _candidate(example, "B", "response b")
        evaluations = [
            RubricEvaluation(rubric_id="r1", candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id="r2", candidate_id=candidate_a.candidate_id, satisfied=True),
            RubricEvaluation(rubric_id="r3", candidate_id=candidate_a.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r1", candidate_id=candidate_b.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r2", candidate_id=candidate_b.candidate_id, satisfied=False),
            RubricEvaluation(rubric_id="r3", candidate_id=candidate_b.candidate_id, satisfied=True),
        ]

        with patch(
            "rubric_gen.compiled.judgebench_eval.compute_whitened_uniform_weights",
            return_value=(
                {"r1": 0.2515, "r2": 0.246, "r3": 0.5025},
                {"raw_weights": [0.2515, 0.246, 0.5025], "clipped_weights": [0.2515, 0.246, 0.5025], "eigenvalues": [0.001, 0.001, 0.5]},
            ),
        ):
            scoring = score_discovered_rubrics_for_pair(
                example=example,
                rubrics=rubrics,
                scoring_candidates=[candidate_a, candidate_b],
                pair_candidates=[candidate_a, candidate_b],
                evaluations=evaluations,
                covariance_ridge=1e-3,
                protocol_mode="generic_baseline",
                policy={"blind_wu_profile": "stable_v1"},
            )

        self.assertEqual(scoring["uniform"]["result"]["decision"], "A>B")
        self.assertEqual(scoring["whitened_uniform"]["result"]["decision"], "A>B")
        self.assertEqual(
            scoring["whitened_uniform"]["result"]["decision_policy"],
            "blind_uniform_overrides_unstable_whitening",
        )

    def test_balanced_folds_keep_family_counts_even(self) -> None:
        examples = []
        families = ("mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench")
        for family in families:
            for index in range(4):
                examples.append(
                    JudgeBenchJoinedExample(
                        split_name="train_80",
                        pair_id=f"{family}_{index}",
                        source=family,
                        source_family=family,
                        question=f"question {family} {index}",
                        reference_answer="ref",
                        response_model="test",
                        response_A="A",
                        response_B="B",
                        label="A>B",
                        original_id=f"orig_{family}_{index}",
                    )
                )
        folds = build_balanced_judgebench_folds(examples, fold_count=4)

        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertEqual(fold["dev_source_family_counts"], {family: 1 for family in families})

    def test_balanced_folds_can_shuffle_deterministically(self) -> None:
        examples = []
        families = ("mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench")
        for family in families:
            for index in range(4):
                examples.append(
                    JudgeBenchJoinedExample(
                        split_name="train_80",
                        pair_id=f"{family}_{index}",
                        source=family,
                        source_family=family,
                        question=f"question {family} {index}",
                        reference_answer="ref",
                        response_model="test",
                        response_A="A",
                        response_B="B",
                        label="A>B",
                        original_id=f"orig_{family}_{index}",
                    )
                )
        folds = build_balanced_judgebench_folds(examples, fold_count=4)
        shuffled_a = build_balanced_judgebench_folds(examples, fold_count=4, shuffle_seed=7)
        shuffled_b = build_balanced_judgebench_folds(examples, fold_count=4, shuffle_seed=7)

        fold_ids = [[row.pair_id for row in fold["dev_examples"]] for fold in folds]
        shuffled_ids_a = [[row.pair_id for row in fold["dev_examples"]] for fold in shuffled_a]
        shuffled_ids_b = [[row.pair_id for row in fold["dev_examples"]] for fold in shuffled_b]

        self.assertNotEqual(fold_ids, shuffled_ids_a)
        self.assertEqual(shuffled_ids_a, shuffled_ids_b)

    def test_blind_validation_candidates_hide_reference_answer(self) -> None:
        example = _generic_pair_example()
        route_decision = JudgeBenchRouteDecision(
            pair_id=example.pair_id,
            source=example.source,
            source_family=example.source_family,
            task_profile_id="general_instruction_following",
            task_family_id="general_instruction_following",
            artifact_kind="response",
            route_kind="fallback",
            bootstrap_used=False,
        )
        example_record = joined_example_to_example_record(
            example,
            task_profile_id=route_decision.task_profile_id,
            reference_answer_access=False,
        )
        strategy = ContrastStrategy(
            strategy_id="test_strategy",
            mutation_ids=("mutate_one", "mutate_two"),
            mutation_grounding_profiles={},
        )

        with patch(
            "rubric_gen.compiled.judgebench_eval.get_contrast_strategy",
            return_value=strategy,
        ), patch(
            "rubric_gen.compiled.judgebench_eval.mutation_function_for_id",
            side_effect=lambda mutation_id: (lambda text, suffix=mutation_id: f"{text} :: {suffix}"),
        ):
            discovery_pairs, pair_candidates, scoring_candidates = build_judgebench_candidates(
                example=example,
                example_record=example_record,
                route_decision=route_decision,
                max_pairs_per_example=4,
                reference_answer_access=False,
            )

        self.assertEqual(example_record.reference_artifact, "")
        self.assertEqual(len(pair_candidates), 2)
        self.assertEqual(len(discovery_pairs), 4)
        self.assertEqual({strong.source_label for strong, _ in discovery_pairs}, {"response_A", "response_B"})
        self.assertEqual(
            {(strong.source_label, weak.source_label) for strong, weak in discovery_pairs[:2]},
            {("response_A", "response_B"), ("response_B", "response_A")},
        )
        self.assertEqual(
            sum(1 for _, weak in discovery_pairs if weak.source_label.startswith("synthetic_mutation:")),
            2,
        )
        self.assertNotIn("reference_answer", {candidate.source_label for candidate in scoring_candidates})

    def test_blind_validation_candidates_cap_reasoning_budget_even_when_raised(self) -> None:
        example = _generic_pair_example()
        route_decision = JudgeBenchRouteDecision(
            pair_id=example.pair_id,
            source=example.source,
            source_family=example.source_family,
            task_profile_id="general_instruction_following",
            task_family_id="general_instruction_following",
            artifact_kind="response",
            route_kind="fallback",
            bootstrap_used=False,
        )
        example_record = joined_example_to_example_record(
            example,
            task_profile_id=route_decision.task_profile_id,
            reference_answer_access=False,
        )
        strategy = ContrastStrategy(
            strategy_id="test_strategy",
            mutation_ids=("mutate_one", "mutate_two", "mutate_three"),
            mutation_grounding_profiles={},
        )

        with patch(
            "rubric_gen.compiled.judgebench_eval.get_contrast_strategy",
            return_value=strategy,
        ), patch(
            "rubric_gen.compiled.judgebench_eval.mutation_function_for_id",
            side_effect=lambda mutation_id: (lambda text, suffix=mutation_id: f"{text} :: {suffix}"),
        ):
            discovery_pairs, pair_candidates, scoring_candidates = build_judgebench_candidates(
                example=example,
                example_record=example_record,
                route_decision=route_decision,
                max_pairs_per_example=6,
                reference_answer_access=False,
            )

        self.assertEqual(len(pair_candidates), 2)
        self.assertEqual(len(scoring_candidates), 4)
        self.assertEqual(len(discovery_pairs), 4)
        self.assertEqual(
            sum(1 for _, weak in discovery_pairs if weak.source_label.startswith("synthetic_mutation:")),
            2,
        )

    def test_blind_validation_candidates_expand_mmlu_budget_when_raised(self) -> None:
        example = _mmlu_conflict_example()
        route_decision = JudgeBenchRouteDecision(
            pair_id=example.pair_id,
            source=example.source,
            source_family=example.source_family,
            task_profile_id="general_instruction_following",
            task_family_id="general_instruction_following",
            artifact_kind="response",
            route_kind="fallback",
            bootstrap_used=False,
        )
        example_record = joined_example_to_example_record(
            example,
            task_profile_id=route_decision.task_profile_id,
            reference_answer_access=False,
        )
        strategy = ContrastStrategy(
            strategy_id="test_strategy",
            mutation_ids=("mutate_one", "mutate_two", "mutate_three"),
            mutation_grounding_profiles={},
        )

        with patch(
            "rubric_gen.compiled.judgebench_eval.get_contrast_strategy",
            return_value=strategy,
        ), patch(
            "rubric_gen.compiled.judgebench_eval.mutation_function_for_id",
            side_effect=lambda mutation_id: (lambda text, suffix=mutation_id: f"{text} :: {suffix}"),
        ):
            discovery_pairs, pair_candidates, scoring_candidates = build_judgebench_candidates(
                example=example,
                example_record=example_record,
                route_decision=route_decision,
                max_pairs_per_example=6,
                reference_answer_access=False,
            )

        self.assertEqual(len(pair_candidates), 2)
        self.assertEqual(len(scoring_candidates), 6)
        self.assertEqual(len(discovery_pairs), 6)
        self.assertEqual(
            sum(1 for _, weak in discovery_pairs if weak.source_label.startswith("synthetic_mutation:")),
            4,
        )

    def test_blind_validation_candidates_use_profile_budget_overrides(self) -> None:
        example = _strict_reasoning_example()
        route_decision = JudgeBenchRouteDecision(
            pair_id=example.pair_id,
            source=example.source,
            source_family=example.source_family,
            task_profile_id="judgebench_source_family_livebench_reasoning_auto_general_instruction_following_person_truth_says_demo",
            task_family_id="general_instruction_following",
            artifact_kind="response",
            route_kind="fallback",
            bootstrap_used=False,
        )
        example_record = joined_example_to_example_record(
            example,
            task_profile_id=route_decision.task_profile_id,
            reference_answer_access=False,
        )
        strategy = ContrastStrategy(
            strategy_id="test_strategy",
            mutation_ids=("mutate_one", "mutate_two", "mutate_three"),
            mutation_grounding_profiles={},
        )
        policy = {
            "blind_budget_profile": "family_profile_v1",
        }

        with patch(
            "rubric_gen.compiled.judgebench_eval.get_contrast_strategy",
            return_value=strategy,
        ), patch(
            "rubric_gen.compiled.judgebench_eval.mutation_function_for_id",
            side_effect=lambda mutation_id: (lambda text, suffix=mutation_id: f"{text} :: {suffix}"),
        ):
            discovery_pairs, pair_candidates, scoring_candidates = build_judgebench_candidates(
                example=example,
                example_record=example_record,
                route_decision=route_decision,
                max_pairs_per_example=6,
                reference_answer_access=False,
                policy=policy,
            )

        self.assertEqual(len(pair_candidates), 2)
        self.assertEqual(len(scoring_candidates), 3)
        self.assertEqual(len(discovery_pairs), 3)
        self.assertEqual(
            sum(1 for _, weak in discovery_pairs if weak.source_label.startswith("synthetic_mutation:")),
            1,
        )

    def test_blind_validation_candidates_use_profile_budget_overrides_v2(self) -> None:
        example = _strict_reasoning_example()
        route_decision = JudgeBenchRouteDecision(
            pair_id=example.pair_id,
            source=example.source,
            source_family=example.source_family,
            task_profile_id="judgebench_source_family_livebench_reasoning_auto_general_instruction_following_logic_grid_demo",
            task_family_id="general_instruction_following",
            artifact_kind="response",
            route_kind="fallback",
            bootstrap_used=False,
        )
        example_record = joined_example_to_example_record(
            example,
            task_profile_id=route_decision.task_profile_id,
            reference_answer_access=False,
        )
        strategy = ContrastStrategy(
            strategy_id="test_strategy",
            mutation_ids=(
                "remove_format_markers",
                "drop_steps",
                "drop_constraints",
                "corrupt_final_answer",
                "drop_supporting_evidence",
                "add_unsupported_detail",
            ),
            mutation_grounding_profiles={},
        )
        policy = {
            "blind_budget_profile": "family_profile_v2",
        }

        with patch(
            "rubric_gen.compiled.judgebench_eval.get_contrast_strategy",
            return_value=strategy,
        ), patch(
            "rubric_gen.compiled.judgebench_eval.mutation_function_for_id",
            side_effect=lambda mutation_id: (lambda text, suffix=mutation_id: f"{text} :: {suffix}"),
        ):
            discovery_pairs, pair_candidates, scoring_candidates = build_judgebench_candidates(
                example=example,
                example_record=example_record,
                route_decision=route_decision,
                max_pairs_per_example=6,
                reference_answer_access=False,
                policy=policy,
            )

        self.assertEqual(len(pair_candidates), 2)
        self.assertEqual(len(scoring_candidates), 5)
        synthetic_weak_labels = [
            weak.source_label
            for _, weak in discovery_pairs
            if weak.source_label.startswith("synthetic_mutation:")
        ]
        self.assertEqual(
            synthetic_weak_labels,
            [
                "synthetic_mutation:drop_constraints:response_A",
                "synthetic_mutation:drop_constraints:response_B",
                "synthetic_mutation:drop_supporting_evidence:response_A",
            ],
        )

    def test_corrupt_final_answer_mutation_targets_final_span(self) -> None:
        mutate = mutation_function_for_id("corrupt_final_answer")
        self.assertIsNotNone(mutate)
        mutated = mutate(
            "Reason through the options carefully.\n\nThe correct answer is:\nC, therefore the final answer string is CCCCC"
        )
        self.assertIn("DDDDD", mutated)
        self.assertNotIn("b careful", mutated.lower())

    def test_prepare_rows_for_scoring_generalizes_candidate_specific_answer_rows(self) -> None:
        example = _strict_reasoning_example()
        rows = [
            {
                "dimension": "conclusion_correctness",
                "label": "Correct final triangle count",
                "requirement": "The response states the final number of triangles as 2 in bold.",
                "severity_tier": "hard_gate",
                "count": 4,
            },
            {
                "dimension": "conclusion_correctness",
                "label": "Correct final triangle count",
                "requirement": "The response concludes with the final number of triangles as 3 in bold.",
                "severity_tier": "hard_gate",
                "count": 5,
            },
            {
                "dimension": "clue_consistency",
                "label": "Track cuts consistently",
                "requirement": "The response must keep the polygon state tracking consistent after each cut.",
                "severity_tier": "high",
                "count": 3,
            },
        ]
        pair_candidates = [
            _candidate(example, "A", example.response_A),
            _candidate(example, "B", example.response_B),
        ]
        prepared = _prepare_rows_for_scoring(
            rows,
            example=JudgeBenchJoinedExample(**{**example.__dict__, "reference_answer": ""}),
            pair_candidates=pair_candidates,
            policy={"blind_scoring_profile": "pruned_v1"},
            reference_answer_access=False,
        )

        texts = [_row["requirement"] for _row in prepared]
        self.assertEqual(len(prepared), 2)
        self.assertTrue(any("state tracking" in text.lower() for text in texts))
        self.assertTrue(any("final answer" in text.lower() and "consistent" in text.lower() for text in texts))

    def test_prepare_rows_for_scoring_pruned_v2_caps_broad_rows_more_aggressively(self) -> None:
        example = _strict_reasoning_example()
        rows = [
            {
                "dimension": "conclusion_correctness",
                "label": "Correct final triangle count",
                "requirement": "The response states the final number of triangles as 2 in bold.",
                "severity_tier": "hard_gate",
                "count": 4,
            },
            {
                "dimension": "conclusion_correctness",
                "label": "Correct final triangle count",
                "requirement": "The response concludes with the final number of triangles as 3 in bold.",
                "severity_tier": "hard_gate",
                "count": 5,
            },
            {
                "dimension": "clue_consistency",
                "label": "Track cuts consistently",
                "requirement": "The response must keep the polygon state tracking consistent after each cut.",
                "severity_tier": "high",
                "count": 3,
            },
        ]
        rows.extend(
            {
                "dimension": "completeness",
                "label": f"Overall completeness {index}",
                "requirement": f"The response should be complete overall in criterion {index}.",
                "severity_tier": "medium",
                "count": 1,
            }
            for index in range(20)
        )
        pair_candidates = [
            _candidate(example, "A", example.response_A),
            _candidate(example, "B", example.response_B),
        ]
        prepared = _prepare_rows_for_scoring(
            rows,
            example=JudgeBenchJoinedExample(**{**example.__dict__, "reference_answer": ""}),
            pair_candidates=pair_candidates,
            policy={"blind_scoring_profile": "pruned_v2"},
            reference_answer_access=False,
        )

        broad_rows = [row for row in prepared if row["dimension"] == "completeness"]
        self.assertLessEqual(len(broad_rows), 2)
        self.assertTrue(any("state tracking" in row["requirement"].lower() for row in prepared))
        self.assertTrue(any(bool(row.get("blind_generalized_answer_row", False)) for row in prepared))

    def test_train_only_development_writes_locked_policy(self) -> None:
        train_examples = [
            JudgeBenchJoinedExample(
                split_name="train_80",
                pair_id=f"pair_{index}",
                source="mmlu-pro-math" if index % 2 == 0 else "livebench-reasoning",
                source_family="mmlu-pro" if index % 2 == 0 else "livebench-reasoning",
                question=f"question {index}",
                reference_answer="BBBBB" if index % 2 == 0 else "***3***",
                response_model="test",
                response_A="A",
                response_B="B",
                label="A>B",
                original_id=f"orig_{index}",
            )
            for index in range(4)
        ]

        def fake_run_judgebench_split(**kwargs):
            split_dir = Path(kwargs["split_dir"])
            split_dir.mkdir(parents=True, exist_ok=True)
            summary_path = split_dir / "summaries" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "schema": "compiled_judgebench_split_summary_v1",
                "split_name": kwargs["split_name"],
                "policy_hash": "fake",
                "reference_answer_access": bool(kwargs.get("reference_answer_access", True)),
                "max_workers": 1,
                "pair_count": len(kwargs["examples"]),
                "wu_metrics": {
                    "mmlu-pro": 100.0,
                    "livebench-reasoning": 100.0,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": 100.0,
                },
                "uniform_metrics": {
                    "mmlu-pro": 100.0,
                    "livebench-reasoning": 100.0,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": 100.0,
                },
                "decision_counts": {"A>B": len(kwargs["examples"])},
                "source_family_counts": dict(
                    json.loads(json.dumps({family: sum(1 for row in kwargs["examples"] if row.source_family == family) for family in {row.source_family for row in kwargs["examples"]}}))
                ),
                "task_profile_counts": {},
                "avg_rubric_count": 1.0,
                "failure_count": 0,
                "stats": {
                    "pairs_total": len(kwargs["examples"]),
                    "pairs_succeeded": len(kwargs["examples"]),
                    "pairs_failed_parse": 0,
                    "local_proposals_total": 0,
                    "local_proposals_promoted": 0,
                    "local_proposals_rejected_grounding": 0,
                    "recursive_calls": 0,
                    "recursive_cache_hits": 0,
                    "recursive_parse_failures": 0,
                    "recursive_parents_considered": 0,
                    "recursive_parents_expanded": 0,
                    "recursive_children_raw_total": 0,
                    "recursive_children_promoted": 0,
                    "recursive_children_rejected_grounding": 0,
                    "examples_with_recursive_change": 0,
                    "rubric_evaluations_total": 0,
                    "rubric_evaluation_cache_hits": 0,
                },
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return {
                "summary": summary,
                "wu_rows": [
                    {
                        "pair_id": row.pair_id,
                        "source": row.source,
                        "label": row.label,
                        "decision_original": row.label,
                        "decision_reversed": "B>A" if row.label == "A>B" else "A>B",
                        "rubric_count": 1,
                    }
                    for row in kwargs["examples"]
                ],
                "uniform_rows": [
                    {
                        "pair_id": row.pair_id,
                        "source": row.source,
                        "label": row.label,
                        "decision_original": row.label,
                        "decision_reversed": "B>A" if row.label == "A>B" else "A>B",
                        "rubric_count": 1,
                    }
                    for row in kwargs["examples"]
                ],
                "failures": [],
                "analysis_rows": [
                    {
                        "pair_id": row.pair_id,
                        "source_family": row.source_family,
                        "weak_source_labels": ["synthetic_mutation:corrupt_final_answer"],
                        "routing_task_profile_id": "test_profile",
                        "broad_rubric_count": 0,
                        "exact_answer_task": False,
                        "code_task": False,
                        "decision": row.label,
                        "rubric_count": 1,
                    }
                    for row in kwargs["examples"]
                ],
                "route_decisions": [],
                "paths": {"summary": str(summary_path.resolve()), "split_dir": str(split_dir.resolve())},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.json"
            official_path = tmp_path / "official.jsonl"
            train_path.write_text("[]", encoding="utf-8")
            official_path.write_text("", encoding="utf-8")
            with patch("rubric_gen.compiled.judgebench_eval.load_local_judgebench_subset", return_value=[]), patch(
                "rubric_gen.compiled.judgebench_eval.load_official_judgebench_pairs",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.join_local_subset_to_official_pairs",
                return_value=train_examples,
            ), patch(
                "rubric_gen.compiled.judgebench_eval.run_judgebench_split",
                side_effect=fake_run_judgebench_split,
            ) as split_mock:
                _, summary = run_judgebench_train_only_development(
                    train_dataset_path=train_path,
                    train_split_name="train_80",
                    run_dir=tmp_path / "run",
                    official_dataset_path=official_path,
                    discovery_model_override=None,
                    judge_model_override=None,
                    use_cache=False,
                    max_criteria=4,
                    max_pairs_per_example=2,
                    bootstrap_iterations=1,
                    recursive_config=RecursiveDiscoveryConfig(
                        max_depth=1,
                        max_recursive_parents_per_pair=2,
                        max_children_per_parent=3,
                        max_recursive_calls_per_pair=2,
                    ),
                    covariance_ridge=1e-3,
                    max_workers=1,
                    fold_count=2,
                    fold_shuffle_seed=11,
                    protocol_mode="generic_baseline",
                    resume=False,
                )

            locked_policy = json.loads((tmp_path / "run" / "frozen_policy" / "locked_policy.json").read_text(encoding="utf-8"))
            self.assertTrue((tmp_path / "run" / "frozen_policy" / "locked_policy.json").exists())
            self.assertEqual(summary["protocol_mode"], "generic_baseline")
            self.assertEqual(summary["fold_count"], 2)
            self.assertEqual(summary["fold_shuffle_seed"], 11)
            self.assertEqual(summary["parallelism"]["fold_processes"], 1)
            self.assertFalse(summary["train_reference_answer_access"])
            self.assertTrue(summary["blind_parity_bootstrap"])
            self.assertFalse(summary["oof_reference_answer_access"])
            self.assertFalse(summary["write_train_fit"])
            self.assertFalse(summary["train_fit_reference_answer_access"])
            self.assertEqual(summary["blind_guidance_profile"], "off")
            self.assertEqual(summary["blind_wu_profile"], "raw")
            self.assertEqual(summary["blind_discriminator_mode_by_family"], {})
            self.assertFalse(summary["oof_summary"]["reference_answer_access"])
            self.assertIsNone(summary["train_fit_summary"])
            self.assertIsNone(summary["train_fit_failure_analysis"])
            self.assertTrue((tmp_path / "run" / "summaries" / "oof_failures.json").exists())
            self.assertTrue((tmp_path / "run" / "summaries" / "oof_analysis_rows.json").exists())
            self.assertTrue((tmp_path / "run" / "summaries" / "locked_policy_alignment.json").exists())
            self.assertTrue(summary["paths"]["oof_failures"].endswith("oof_failures.json"))
            self.assertTrue(summary["paths"]["oof_analysis_rows"].endswith("oof_analysis_rows.json"))
            self.assertTrue(summary["paths"]["locked_policy_alignment"].endswith("locked_policy_alignment.json"))
            self.assertEqual(summary["locked_policy_alignment"]["export_strategy"], "full_train_rebuild")
            self.assertFalse(summary["locked_policy_alignment"]["train_fit_available"])
            self.assertEqual(summary["paths"]["train_fit_summary"], "")
            self.assertFalse(locked_policy["locking_metadata"]["train_reference_answer_access"])
            self.assertTrue(locked_policy["locking_metadata"]["blind_parity_bootstrap"])
            self.assertFalse(locked_policy["locking_metadata"]["oof_reference_answer_access"])
            self.assertFalse(locked_policy["locking_metadata"]["write_train_fit"])
            self.assertFalse(locked_policy["locking_metadata"]["train_fit_reference_answer_access"])
            self.assertEqual(split_mock.call_count, 2)
            self.assertTrue(all(not call.kwargs["reference_answer_access"] for call in split_mock.call_args_list))

    def test_train_only_development_can_write_blind_train_fit_artifacts(self) -> None:
        train_examples = [
            JudgeBenchJoinedExample(
                split_name="train_80",
                pair_id=f"pair_{index}",
                source="mmlu-pro-math" if index % 2 == 0 else "livebench-reasoning",
                source_family="mmlu-pro" if index % 2 == 0 else "livebench-reasoning",
                question=f"question {index}",
                reference_answer="BBBBB" if index % 2 == 0 else "***3***",
                response_model="test",
                response_A="A",
                response_B="B",
                label="A>B",
                original_id=f"orig_{index}",
            )
            for index in range(4)
        ]

        def fake_run_judgebench_split(**kwargs):
            split_dir = Path(kwargs["split_dir"])
            split_dir.mkdir(parents=True, exist_ok=True)
            summary_path = split_dir / "summaries" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            overall = 95.0 if kwargs["split_name"].endswith("_train_fit") else 82.5
            summary = {
                "schema": "compiled_judgebench_split_summary_v1",
                "split_name": kwargs["split_name"],
                "policy_hash": "fake",
                "reference_answer_access": bool(kwargs.get("reference_answer_access", True)),
                "max_workers": 1,
                "pair_count": len(kwargs["examples"]),
                "wu_metrics": {
                    "mmlu-pro": overall,
                    "livebench-reasoning": overall,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": overall,
                },
                "uniform_metrics": {
                    "mmlu-pro": overall,
                    "livebench-reasoning": overall,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": overall,
                },
                "decision_counts": {"A>B": len(kwargs["examples"])},
                "source_family_counts": {},
                "task_profile_counts": {},
                "avg_rubric_count": 1.0,
                "failure_count": 0,
                "stats": {
                    "pairs_total": len(kwargs["examples"]),
                    "pairs_succeeded": len(kwargs["examples"]),
                    "pairs_failed_parse": 0,
                    "local_proposals_total": 0,
                    "local_proposals_promoted": 0,
                    "local_proposals_rejected_grounding": 0,
                    "recursive_calls": 0,
                    "recursive_cache_hits": 0,
                    "recursive_parse_failures": 0,
                    "recursive_parents_considered": 0,
                    "recursive_parents_expanded": 0,
                    "recursive_children_raw_total": 0,
                    "recursive_children_promoted": 0,
                    "recursive_children_rejected_grounding": 0,
                    "examples_with_recursive_change": 0,
                    "rubric_evaluations_total": 0,
                    "rubric_evaluation_cache_hits": 0,
                },
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return {
                "summary": summary,
                "wu_rows": [],
                "uniform_rows": [],
                "failures": [],
                "analysis_rows": [
                    {
                        "pair_id": row.pair_id,
                        "source_family": row.source_family,
                        "weak_source_labels": [],
                        "routing_task_profile_id": "test_profile",
                        "broad_rubric_count": 0,
                        "exact_answer_task": False,
                        "code_task": False,
                        "decision": row.label,
                        "rubric_count": 1,
                    }
                    for row in kwargs["examples"]
                ],
                "route_decisions": [],
                "paths": {"summary": str(summary_path.resolve()), "split_dir": str(split_dir.resolve())},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.json"
            official_path = tmp_path / "official.jsonl"
            train_path.write_text("[]", encoding="utf-8")
            official_path.write_text("", encoding="utf-8")
            with patch("rubric_gen.compiled.judgebench_eval.load_local_judgebench_subset", return_value=[]), patch(
                "rubric_gen.compiled.judgebench_eval.load_official_judgebench_pairs",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.join_local_subset_to_official_pairs",
                return_value=train_examples,
            ), patch(
                "rubric_gen.compiled.judgebench_eval.run_judgebench_split",
                side_effect=fake_run_judgebench_split,
            ) as split_mock:
                _, summary = run_judgebench_train_only_development(
                    train_dataset_path=train_path,
                    train_split_name="train_80",
                    run_dir=tmp_path / "run",
                    official_dataset_path=official_path,
                    discovery_model_override=None,
                    judge_model_override=None,
                    use_cache=False,
                    max_criteria=4,
                    max_pairs_per_example=2,
                    bootstrap_iterations=1,
                    recursive_config=RecursiveDiscoveryConfig(
                        max_depth=1,
                        max_recursive_parents_per_pair=2,
                        max_children_per_parent=3,
                        max_recursive_calls_per_pair=2,
                    ),
                    covariance_ridge=1e-3,
                    max_workers=1,
                    fold_count=2,
                    fold_shuffle_seed=11,
                    protocol_mode="generic_baseline",
                    resume=False,
                    write_train_fit=True,
                )

            train_fit_summary = dict(summary["train_fit_summary"] or {})
            self.assertTrue(summary["write_train_fit"])
            self.assertFalse(summary["train_fit_reference_answer_access"])
            self.assertEqual(train_fit_summary["split_name"], "train_80_train_fit")
            self.assertFalse(train_fit_summary["reference_answer_access"])
            self.assertEqual(train_fit_summary["pair_count"], len(train_examples))
            self.assertEqual(train_fit_summary["wu_metrics"]["overall"], 95.0)
            self.assertTrue((tmp_path / "run" / "summaries" / "train_fit_summary.json").exists())
            self.assertTrue((tmp_path / "run" / "summaries" / "train_fit_failure_analysis.json").exists())
            self.assertTrue((tmp_path / "run" / "summaries" / "train_fit_analysis_rows.json").exists())
            self.assertTrue(summary["paths"]["train_fit_summary"].endswith("train_fit_summary.json"))
            self.assertEqual(summary["locked_policy_alignment"]["export_strategy"], "full_train_rebuild")
            self.assertGreater(summary["locked_policy_alignment"]["locked_train_fit_minus_oof_wu"], 0.0)
            self.assertEqual(split_mock.call_count, 3)
            self.assertEqual(split_mock.call_args_list[-1].kwargs["split_name"], "train_80_train_fit")
            self.assertEqual(len(split_mock.call_args_list[-1].kwargs["examples"]), len(train_examples))
            self.assertFalse(split_mock.call_args_list[-1].kwargs["reference_answer_access"])

    def test_train_only_development_can_disable_train_reference_access(self) -> None:
        train_examples = [
            JudgeBenchJoinedExample(
                split_name="train_80",
                pair_id=f"pair_{index}",
                source="mmlu-pro-math" if index % 2 == 0 else "livebench-reasoning",
                source_family="mmlu-pro" if index % 2 == 0 else "livebench-reasoning",
                question=f"question {index}",
                reference_answer="BBBBB" if index % 2 == 0 else "***3***",
                response_model="test",
                response_A="A",
                response_B="B",
                label="A>B",
                original_id=f"orig_{index}",
            )
            for index in range(4)
        ]
        build_calls = []

        def fake_build_initial_frozen_policy(**kwargs):
            build_calls.append(dict(kwargs))
            return {
                "schema": "compiled_judgebench_policy_v1",
                "protocol_mode": kwargs["protocol_mode"],
                "blind_scoring_profile": kwargs.get("blind_scoring_profile", "baseline"),
                "blind_budget_profile": kwargs.get("blind_budget_profile", "family_v1"),
                "blind_guidance_profile": kwargs.get("blind_guidance_profile", "off"),
                "blind_wu_profile": kwargs.get("blind_wu_profile", "raw"),
                "retrieval_profile": kwargs.get("retrieval_profile", "off"),
                "retrieval_top_k": kwargs.get("retrieval_top_k", 2),
                "source_family_routes": {},
                "fallback_route": {},
                "prompt_nudges": {"global": []},
                "recursion_config": {
                    "max_depth": kwargs["recursive_config"].max_depth,
                    "max_recursive_parents_per_pair": kwargs["recursive_config"].max_recursive_parents_per_pair,
                    "max_children_per_parent": kwargs["recursive_config"].max_children_per_parent,
                    "max_recursive_calls_per_pair": kwargs["recursive_config"].max_recursive_calls_per_pair,
                },
                "family_recursion_config": {},
                "refinement_history": [],
            }

        def fake_run_judgebench_split(**kwargs):
            split_dir = Path(kwargs["split_dir"])
            split_dir.mkdir(parents=True, exist_ok=True)
            summary_path = split_dir / "summaries" / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "schema": "compiled_judgebench_split_summary_v1",
                "split_name": kwargs["split_name"],
                "policy_hash": "fake",
                "reference_answer_access": bool(kwargs.get("reference_answer_access", True)),
                "max_workers": 1,
                "pair_count": len(kwargs["examples"]),
                "wu_metrics": {
                    "mmlu-pro": 100.0,
                    "livebench-reasoning": 100.0,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": 100.0,
                },
                "uniform_metrics": {
                    "mmlu-pro": 100.0,
                    "livebench-reasoning": 100.0,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": 100.0,
                },
                "decision_counts": {"A>B": len(kwargs["examples"])},
                "source_family_counts": {},
                "task_profile_counts": {},
                "avg_rubric_count": 1.0,
                "failure_count": 0,
                "stats": {
                    "pairs_total": len(kwargs["examples"]),
                    "pairs_succeeded": len(kwargs["examples"]),
                    "pairs_failed_parse": 0,
                    "local_proposals_total": 0,
                    "local_proposals_promoted": 0,
                    "local_proposals_rejected_grounding": 0,
                    "recursive_calls": 0,
                    "recursive_cache_hits": 0,
                    "recursive_parse_failures": 0,
                    "recursive_parents_considered": 0,
                    "recursive_parents_expanded": 0,
                    "recursive_children_raw_total": 0,
                    "recursive_children_promoted": 0,
                    "recursive_children_rejected_grounding": 0,
                    "examples_with_recursive_change": 0,
                    "rubric_evaluations_total": 0,
                    "rubric_evaluation_cache_hits": 0,
                },
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return {
                "summary": summary,
                "wu_rows": [],
                "uniform_rows": [],
                "failures": [],
                "analysis_rows": [],
                "route_decisions": [],
                "paths": {"summary": str(summary_path.resolve()), "split_dir": str(split_dir.resolve())},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.json"
            official_path = tmp_path / "official.jsonl"
            train_path.write_text("[]", encoding="utf-8")
            official_path.write_text("", encoding="utf-8")
            with patch("rubric_gen.compiled.judgebench_eval.load_local_judgebench_subset", return_value=[]), patch(
                "rubric_gen.compiled.judgebench_eval.load_official_judgebench_pairs",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.join_local_subset_to_official_pairs",
                return_value=train_examples,
            ), patch(
                "rubric_gen.compiled.judgebench_eval.build_initial_frozen_policy",
                side_effect=fake_build_initial_frozen_policy,
            ), patch(
                "rubric_gen.compiled.judgebench_eval.run_judgebench_split",
                side_effect=fake_run_judgebench_split,
            ):
                _, summary = run_judgebench_train_only_development(
                    train_dataset_path=train_path,
                    train_split_name="train_80",
                    run_dir=tmp_path / "run",
                    official_dataset_path=official_path,
                    discovery_model_override=None,
                    judge_model_override=None,
                    use_cache=False,
                    max_criteria=4,
                    max_pairs_per_example=2,
                    bootstrap_iterations=1,
                    recursive_config=RecursiveDiscoveryConfig(
                        max_depth=1,
                        max_recursive_parents_per_pair=2,
                        max_children_per_parent=3,
                        max_recursive_calls_per_pair=2,
                    ),
                    covariance_ridge=1e-3,
                    max_workers=1,
                    fold_count=2,
                    fold_shuffle_seed=None,
                    protocol_mode="generic_baseline",
                    resume=False,
                    train_reference_answer_access=False,
                    oof_reference_answer_access=False,
                )

            locked_policy = json.loads((tmp_path / "run" / "frozen_policy" / "locked_policy.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["train_reference_answer_access"])
            self.assertFalse(summary["oof_reference_answer_access"])
            self.assertTrue(build_calls)
            self.assertTrue(all(not call["reference_answer_access"] for call in build_calls))
            self.assertFalse(locked_policy["locking_metadata"]["train_reference_answer_access"])
            self.assertFalse(locked_policy["locking_metadata"]["oof_reference_answer_access"])

    def test_final_eval_defaults_to_aggregate_only_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_run_dir = tmp_path / "train_run"
            (train_run_dir / "summaries").mkdir(parents=True, exist_ok=True)
            (train_run_dir / "frozen_policy").mkdir(parents=True, exist_ok=True)
            mechanism_spec = _build_mechanism_spec(
                protocol_mode="generic_baseline",
                bootstrap_iterations=1,
                recursive_config=RecursiveDiscoveryConfig(
                    max_depth=1,
                    max_recursive_parents_per_pair=2,
                    max_children_per_parent=3,
                    max_recursive_calls_per_pair=2,
                ),
                discovery_model_override=None,
                judge_model_override=None,
                max_criteria=4,
                max_pairs_per_example=2,
                covariance_ridge=1e-3,
            )
            locked_policy = {
                "schema": "compiled_judgebench_policy_v1",
                "protocol_mode": "generic_baseline",
                "blind_scoring_profile": "baseline",
                "blind_budget_profile": "family_v1",
                "retrieval_profile": "off",
                "retrieval_top_k": 2,
                "source_family_routes": {},
                "fallback_route": {},
                "prompt_nudges": {"global": []},
                "recursion_config": {
                    "max_depth": 1,
                    "max_recursive_parents_per_pair": 2,
                    "max_children_per_parent": 3,
                    "max_recursive_calls_per_pair": 2,
                },
                "family_recursion_config": {},
                "refinement_history": [],
            }
            locked_policy["locking_metadata"] = {
                "train_pair_ids": ["train_pair_1"],
                "mechanism_spec": mechanism_spec,
                "mechanism_hash": _mechanism_hash(mechanism_spec),
                "frozen_policy_hash": _policy_core_hash(locked_policy),
            }
            (train_run_dir / "frozen_policy" / "locked_policy.json").write_text(
                json.dumps(locked_policy),
                encoding="utf-8",
            )
            (train_run_dir / "summaries" / "summary.json").write_text(json.dumps({"mechanism_hash": locked_policy["locking_metadata"]["mechanism_hash"]}), encoding="utf-8")

            validation_path = tmp_path / "validation_270.json"
            official_path = tmp_path / "official.jsonl"
            validation_path.write_text("[]", encoding="utf-8")
            official_path.write_text("", encoding="utf-8")

            fake_summary_path = tmp_path / "final_run" / "validation_270" / "final" / "summaries" / "summary.json"
            fake_summary_path.parent.mkdir(parents=True, exist_ok=True)
            fake_summary_path.write_text(
                json.dumps(
                    {
                        "schema": "compiled_judgebench_split_summary_v1",
                        "split_name": "validation_270",
                        "policy_hash": "fake",
                        "max_workers": 1,
                        "pair_count": 0,
                        "wu_metrics": {
                            "mmlu-pro": 0.0,
                            "livebench-reasoning": 0.0,
                            "livebench-math": 0.0,
                            "livecodebench": 0.0,
                            "overall": 0.0,
                        },
                        "uniform_metrics": {
                            "mmlu-pro": 0.0,
                            "livebench-reasoning": 0.0,
                            "livebench-math": 0.0,
                            "livecodebench": 0.0,
                            "overall": 0.0,
                        },
                        "decision_counts": {},
                        "source_family_counts": {},
                        "task_profile_counts": {},
                        "avg_rubric_count": 0.0,
                        "failure_count": 0,
                        "stats": {
                            "pairs_total": 0,
                            "pairs_succeeded": 0,
                            "pairs_failed_parse": 0,
                            "local_proposals_total": 0,
                            "local_proposals_promoted": 0,
                            "local_proposals_rejected_grounding": 0,
                            "recursive_calls": 0,
                            "recursive_cache_hits": 0,
                            "recursive_parse_failures": 0,
                            "recursive_parents_considered": 0,
                            "recursive_parents_expanded": 0,
                            "recursive_children_raw_total": 0,
                            "recursive_children_promoted": 0,
                            "recursive_children_rejected_grounding": 0,
                            "examples_with_recursive_change": 0,
                            "rubric_evaluations_total": 0,
                            "rubric_evaluation_cache_hits": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("rubric_gen.compiled.judgebench_eval.load_official_judgebench_pairs", return_value=[]), patch(
                "rubric_gen.compiled.judgebench_eval.load_local_judgebench_subset",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.join_local_subset_to_official_pairs",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.run_judgebench_split",
                return_value={
                    "summary": json.loads(fake_summary_path.read_text(encoding="utf-8")),
                    "wu_rows": [],
                    "uniform_rows": [],
                    "failures": [],
                    "analysis_rows": [],
                    "route_decisions": [],
                    "paths": {"summary": str(fake_summary_path.resolve()), "split_dir": str(fake_summary_path.parent.parent.resolve())},
                },
            ) as split_mock, patch(
                "rubric_gen.compiled.judgebench_eval._assert_split_summary_persisted",
                return_value=None,
            ):
                _, summary = run_judgebench_final_evaluation(
                    train_run_dir=train_run_dir,
                    validation_dataset_path=validation_path,
                    validation_split_name="validation_270",
                    run_dir=tmp_path / "final_run",
                    official_dataset_path=official_path,
                    max_workers=1,
                    write_detailed_outputs=False,
                    resume=False,
                )

            self.assertFalse(split_mock.call_args.kwargs["write_example_artifacts"])
            self.assertFalse(split_mock.call_args.kwargs["write_reports"])
            self.assertFalse(split_mock.call_args.kwargs["reference_answer_access"])
            self.assertFalse(summary["reference_answer_access"])

    def test_final_eval_loads_retrieval_examples_from_locked_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_run_dir = tmp_path / "train_run"
            (train_run_dir / "summaries").mkdir(parents=True, exist_ok=True)
            (train_run_dir / "frozen_policy").mkdir(parents=True, exist_ok=True)
            (train_run_dir / "dataset").mkdir(parents=True, exist_ok=True)
            mechanism_spec = _build_mechanism_spec(
                protocol_mode="generic_baseline",
                bootstrap_iterations=1,
                recursive_config=RecursiveDiscoveryConfig(
                    max_depth=1,
                    max_recursive_parents_per_pair=2,
                    max_children_per_parent=3,
                    max_recursive_calls_per_pair=2,
                ),
                discovery_model_override=None,
                judge_model_override=None,
                max_criteria=4,
                max_pairs_per_example=2,
                covariance_ridge=1e-3,
                retrieval_profile="family_question_v1",
                retrieval_top_k=1,
            )
            locked_policy = {
                "schema": "compiled_judgebench_policy_v1",
                "protocol_mode": "generic_baseline",
                "blind_scoring_profile": "baseline",
                "blind_budget_profile": "family_v1",
                "retrieval_profile": "family_question_v1",
                "retrieval_top_k": 1,
                "source_family_routes": {},
                "fallback_route": {},
                "prompt_nudges": {"global": []},
                "recursion_config": {
                    "max_depth": 1,
                    "max_recursive_parents_per_pair": 2,
                    "max_children_per_parent": 3,
                    "max_recursive_calls_per_pair": 2,
                },
                "family_recursion_config": {},
                "refinement_history": [],
            }
            locked_policy["locking_metadata"] = {
                "train_pair_ids": ["train_pair_1"],
                "train_split_name": "train_80",
                "mechanism_spec": mechanism_spec,
                "mechanism_hash": _mechanism_hash(mechanism_spec),
                "frozen_policy_hash": _policy_core_hash(locked_policy),
            }
            (train_run_dir / "frozen_policy" / "locked_policy.json").write_text(
                json.dumps(locked_policy),
                encoding="utf-8",
            )
            (train_run_dir / "summaries" / "summary.json").write_text(
                json.dumps(
                    {
                        "mechanism_hash": locked_policy["locking_metadata"]["mechanism_hash"],
                        "train_split_name": "train_80",
                    }
                ),
                encoding="utf-8",
            )
            (train_run_dir / "dataset" / "joined_train_80.json").write_text(
                json.dumps(
                    [
                        {
                            "split_name": "train_80",
                            "pair_id": "train_pair_1",
                            "source": "mmlu-pro-math",
                            "source_family": "mmlu-pro",
                            "question": _exact_answer_example().question,
                            "reference_answer": _exact_answer_example().reference_answer,
                            "response_model": "test",
                            "response_A": "BBBBB",
                            "response_B": "CCCCC",
                            "label": "A>B",
                            "original_id": "orig_train",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            validation_path = tmp_path / "validation_270.json"
            official_path = tmp_path / "official.jsonl"
            validation_path.write_text("[]", encoding="utf-8")
            official_path.write_text("", encoding="utf-8")

            fake_summary_path = tmp_path / "final_run" / "validation_270" / "final" / "summaries" / "summary.json"
            fake_summary_path.parent.mkdir(parents=True, exist_ok=True)
            fake_summary_path.write_text(
                json.dumps(
                    {
                        "schema": "compiled_judgebench_split_summary_v1",
                        "split_name": "validation_270",
                        "policy_hash": "fake",
                        "reference_answer_access": False,
                        "retrieval_profile": "family_question_v1",
                        "max_workers": 1,
                        "pair_count": 0,
                        "wu_metrics": {
                            "mmlu-pro": 0.0,
                            "livebench-reasoning": 0.0,
                            "livebench-math": 0.0,
                            "livecodebench": 0.0,
                            "overall": 0.0,
                        },
                        "uniform_metrics": {
                            "mmlu-pro": 0.0,
                            "livebench-reasoning": 0.0,
                            "livebench-math": 0.0,
                            "livecodebench": 0.0,
                            "overall": 0.0,
                        },
                        "decision_counts": {},
                        "source_family_counts": {},
                        "task_profile_counts": {},
                        "avg_rubric_count": 0.0,
                        "failure_count": 0,
                        "stats": {"pairs_total": 0, "pairs_succeeded": 0, "pairs_failed_parse": 0},
                    }
                ),
                encoding="utf-8",
            )

            with patch("rubric_gen.compiled.judgebench_eval.load_official_judgebench_pairs", return_value=[]), patch(
                "rubric_gen.compiled.judgebench_eval.load_local_judgebench_subset",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.join_local_subset_to_official_pairs",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.run_judgebench_split",
                return_value={
                    "summary": json.loads(fake_summary_path.read_text(encoding="utf-8")),
                    "wu_rows": [],
                    "uniform_rows": [],
                    "failures": [],
                    "analysis_rows": [],
                    "route_decisions": [],
                    "paths": {"summary": str(fake_summary_path.resolve()), "split_dir": str(fake_summary_path.parent.parent.resolve())},
                },
            ) as split_mock, patch(
                "rubric_gen.compiled.judgebench_eval._assert_split_summary_persisted",
                return_value=None,
            ):
                _, summary = run_judgebench_final_evaluation(
                    train_run_dir=train_run_dir,
                    validation_dataset_path=validation_path,
                    validation_split_name="validation_270",
                    run_dir=tmp_path / "final_run",
                    official_dataset_path=official_path,
                    max_workers=1,
                    write_detailed_outputs=False,
                    resume=False,
                )

            self.assertEqual(summary["retrieval_profile"], "family_question_v1")
            self.assertEqual(len(split_mock.call_args.kwargs["retrieval_examples"]), 1)
            self.assertEqual(split_mock.call_args.kwargs["policy"]["retrieval_profile"], "family_question_v1")

    def test_final_eval_can_override_family_retrieval_and_discriminator_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_run_dir = tmp_path / "train_run"
            (train_run_dir / "summaries").mkdir(parents=True, exist_ok=True)
            (train_run_dir / "frozen_policy").mkdir(parents=True, exist_ok=True)
            (train_run_dir / "dataset").mkdir(parents=True, exist_ok=True)
            mechanism_spec = _build_mechanism_spec(
                protocol_mode="generic_baseline",
                bootstrap_iterations=1,
                recursive_config=RecursiveDiscoveryConfig(
                    max_depth=1,
                    max_recursive_parents_per_pair=2,
                    max_children_per_parent=3,
                    max_recursive_calls_per_pair=2,
                ),
                discovery_model_override=None,
                judge_model_override=None,
                max_criteria=4,
                max_pairs_per_example=2,
                covariance_ridge=1e-3,
                retrieval_profile="off",
                retrieval_top_k=2,
            )
            locked_policy = {
                "schema": "compiled_judgebench_policy_v1",
                "protocol_mode": "generic_baseline",
                "blind_scoring_profile": "pruned_disc_v1",
                "blind_budget_profile": "family_v1",
                "retrieval_profile": "off",
                "retrieval_top_k": 2,
                "blind_discriminator_mode_by_family": {},
                "retrieval_profile_by_family": {},
                "retrieval_top_k_by_family": {},
                "source_family_routes": {},
                "fallback_route": {},
                "prompt_nudges": {"global": []},
                "recursion_config": {
                    "max_depth": 1,
                    "max_recursive_parents_per_pair": 2,
                    "max_children_per_parent": 3,
                    "max_recursive_calls_per_pair": 2,
                },
                "family_recursion_config": {},
                "refinement_history": [],
            }
            locked_policy["locking_metadata"] = {
                "train_pair_ids": ["train_pair_1"],
                "train_split_name": "train_120",
                "mechanism_spec": mechanism_spec,
                "mechanism_hash": _mechanism_hash(mechanism_spec),
                "frozen_policy_hash": _policy_core_hash(locked_policy),
            }
            (train_run_dir / "frozen_policy" / "locked_policy.json").write_text(
                json.dumps(locked_policy),
                encoding="utf-8",
            )
            (train_run_dir / "summaries" / "summary.json").write_text(
                json.dumps(
                    {
                        "mechanism_hash": locked_policy["locking_metadata"]["mechanism_hash"],
                        "train_split_name": "train_120",
                    }
                ),
                encoding="utf-8",
            )
            (train_run_dir / "dataset" / "joined_train_120.json").write_text(
                json.dumps(
                    [
                        {
                            "split_name": "train_120",
                            "pair_id": "train_pair_1",
                            "source": "mmlu-pro-math",
                            "source_family": "mmlu-pro",
                            "question": _exact_answer_example().question,
                            "reference_answer": _exact_answer_example().reference_answer,
                            "response_model": "test",
                            "response_A": "BBBBB",
                            "response_B": "CCCCC",
                            "label": "A>B",
                            "original_id": "orig_train",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            validation_path = tmp_path / "validation_270.json"
            official_path = tmp_path / "official.jsonl"
            validation_path.write_text("[]", encoding="utf-8")
            official_path.write_text("", encoding="utf-8")

            fake_summary_path = tmp_path / "final_run" / "validation_270" / "final" / "summaries" / "summary.json"
            fake_summary_path.parent.mkdir(parents=True, exist_ok=True)
            fake_summary_path.write_text(
                json.dumps(
                    {
                        "schema": "compiled_judgebench_split_summary_v1",
                        "split_name": "validation_270",
                        "policy_hash": "fake",
                        "reference_answer_access": False,
                        "retrieval_profile": "off",
                        "retrieval_profile_by_family": {"mmlu-pro": "family_question_v1"},
                        "retrieval_top_k": 2,
                        "retrieval_top_k_by_family": {"mmlu-pro": 1},
                        "blind_discriminator_mode_by_family": {"mmlu-pro": "off"},
                        "max_workers": 1,
                        "pair_count": 0,
                        "wu_metrics": {
                            "mmlu-pro": 0.0,
                            "livebench-reasoning": 0.0,
                            "livebench-math": 0.0,
                            "livecodebench": 0.0,
                            "overall": 0.0,
                        },
                        "uniform_metrics": {
                            "mmlu-pro": 0.0,
                            "livebench-reasoning": 0.0,
                            "livebench-math": 0.0,
                            "livecodebench": 0.0,
                            "overall": 0.0,
                        },
                        "decision_counts": {},
                        "source_family_counts": {},
                        "task_profile_counts": {},
                        "avg_rubric_count": 0.0,
                        "failure_count": 0,
                        "stats": {"pairs_total": 0, "pairs_succeeded": 0, "pairs_failed_parse": 0},
                    }
                ),
                encoding="utf-8",
            )

            with patch("rubric_gen.compiled.judgebench_eval.load_official_judgebench_pairs", return_value=[]), patch(
                "rubric_gen.compiled.judgebench_eval.load_local_judgebench_subset",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.join_local_subset_to_official_pairs",
                return_value=[],
            ), patch(
                "rubric_gen.compiled.judgebench_eval.run_judgebench_split",
                return_value={
                    "summary": json.loads(fake_summary_path.read_text(encoding="utf-8")),
                    "wu_rows": [],
                    "uniform_rows": [],
                    "failures": [],
                    "analysis_rows": [],
                    "route_decisions": [],
                    "paths": {"summary": str(fake_summary_path.resolve()), "split_dir": str(fake_summary_path.parent.parent.resolve())},
                },
            ) as split_mock, patch(
                "rubric_gen.compiled.judgebench_eval._assert_split_summary_persisted",
                return_value=None,
            ):
                _, summary = run_judgebench_final_evaluation(
                    train_run_dir=train_run_dir,
                    validation_dataset_path=validation_path,
                    validation_split_name="validation_270",
                    run_dir=tmp_path / "final_run",
                    official_dataset_path=official_path,
                    max_workers=1,
                    write_detailed_outputs=False,
                    resume=False,
                    retrieval_profile_by_family={"mmlu-pro": "family_question_v1"},
                    retrieval_top_k_by_family={"mmlu-pro": 1},
                    blind_discriminator_mode_by_family={"mmlu-pro": "off"},
                )

            self.assertEqual(summary["retrieval_profile"], "off")
            self.assertEqual(summary["retrieval_profile_by_family"], {"mmlu-pro": "family_question_v1"})
            self.assertEqual(summary["retrieval_top_k_by_family"], {"mmlu-pro": 1})
            self.assertEqual(summary["blind_discriminator_mode_by_family"], {"mmlu-pro": "off"})
            self.assertEqual(len(split_mock.call_args.kwargs["retrieval_examples"]), 1)
            self.assertEqual(
                split_mock.call_args.kwargs["policy"]["retrieval_profile_by_family"],
                {"mmlu-pro": "family_question_v1"},
            )
            self.assertEqual(
                split_mock.call_args.kwargs["policy"]["retrieval_top_k_by_family"],
                {"mmlu-pro": 1},
            )
            self.assertEqual(
                split_mock.call_args.kwargs["policy"]["blind_discriminator_mode_by_family"],
                {"mmlu-pro": "off"},
            )

    def test_final_eval_persists_failure_bundle_and_same_run_subset_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_run_dir = tmp_path / "train_run"
            (train_run_dir / "summaries").mkdir(parents=True, exist_ok=True)
            (train_run_dir / "frozen_policy").mkdir(parents=True, exist_ok=True)
            mechanism_spec = _build_mechanism_spec(
                protocol_mode="generic_baseline",
                bootstrap_iterations=1,
                recursive_config=RecursiveDiscoveryConfig(
                    max_depth=1,
                    max_recursive_parents_per_pair=2,
                    max_children_per_parent=3,
                    max_recursive_calls_per_pair=2,
                ),
                discovery_model_override=None,
                judge_model_override=None,
                max_criteria=4,
                max_pairs_per_example=2,
                covariance_ridge=1e-3,
                retrieval_profile="off",
                retrieval_top_k=2,
            )
            locked_policy = {
                "schema": "compiled_judgebench_policy_v1",
                "protocol_mode": "generic_baseline",
                "blind_scoring_profile": "baseline",
                "blind_budget_profile": "family_v1",
                "retrieval_profile": "off",
                "retrieval_top_k": 2,
                "source_family_routes": {},
                "fallback_route": {},
                "prompt_nudges": {"global": []},
                "recursion_config": {
                    "max_depth": 1,
                    "max_recursive_parents_per_pair": 2,
                    "max_children_per_parent": 3,
                    "max_recursive_calls_per_pair": 2,
                },
                "family_recursion_config": {},
                "refinement_history": [],
            }
            locked_policy["locking_metadata"] = {
                "train_pair_ids": ["train_pair_1"],
                "train_split_name": "train_120",
                "mechanism_spec": mechanism_spec,
                "mechanism_hash": _mechanism_hash(mechanism_spec),
                "frozen_policy_hash": _policy_core_hash(locked_policy),
            }
            (train_run_dir / "frozen_policy" / "locked_policy.json").write_text(
                json.dumps(locked_policy),
                encoding="utf-8",
            )
            (train_run_dir / "summaries" / "summary.json").write_text(
                json.dumps(
                    {
                        "mechanism_hash": locked_policy["locking_metadata"]["mechanism_hash"],
                        "train_split_name": "train_120",
                    }
                ),
                encoding="utf-8",
            )

            validation_examples = [
                JudgeBenchJoinedExample(
                    split_name="validation_350",
                    pair_id="pair_80",
                    source="mmlu-pro-math",
                    source_family="mmlu-pro",
                    question="question 80",
                    reference_answer="BBBBB",
                    response_model="test",
                    response_A="A",
                    response_B="B",
                    label="A>B",
                    original_id="orig_80",
                ),
                JudgeBenchJoinedExample(
                    split_name="validation_350",
                    pair_id="pair_270",
                    source="livebench-reasoning",
                    source_family="livebench-reasoning",
                    question="question 270",
                    reference_answer="***3***",
                    response_model="test",
                    response_A="A",
                    response_B="B",
                    label="B>A",
                    original_id="orig_270",
                ),
            ]
            validation_path = tmp_path / "validation_350.json"
            official_path = tmp_path / "official.jsonl"
            validation_path.write_text("[]", encoding="utf-8")
            official_path.write_text("", encoding="utf-8")
            fake_summary_path = tmp_path / "final_run" / "validation_350" / "final" / "summaries" / "summary.json"
            fake_summary_path.parent.mkdir(parents=True, exist_ok=True)
            fake_summary = {
                "schema": "compiled_judgebench_split_summary_v1",
                "split_name": "validation_350",
                "policy_hash": "fake",
                "reference_answer_access": False,
                "blind_guidance_profile": "off",
                "blind_wu_profile": "raw",
                "retrieval_profile": "off",
                "retrieval_profile_by_family": {},
                "retrieval_top_k": 2,
                "retrieval_top_k_by_family": {},
                "blind_discriminator_mode_by_family": {},
                "max_workers": 1,
                "pair_count": 2,
                "wu_metrics": {
                    "mmlu-pro": 100.0,
                    "livebench-reasoning": 0.0,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": 50.0,
                },
                "uniform_metrics": {
                    "mmlu-pro": 100.0,
                    "livebench-reasoning": 0.0,
                    "livebench-math": 0.0,
                    "livecodebench": 0.0,
                    "overall": 50.0,
                },
                "decision_counts": {"A>B": 1, "A=B": 1},
                "source_family_counts": {"mmlu-pro": 1, "livebench-reasoning": 1},
                "task_profile_counts": {"test_profile": 2},
                "avg_rubric_count": 1.0,
                "failure_count": 1,
                "stats": {"pairs_total": 2, "pairs_succeeded": 2, "pairs_failed_parse": 0},
            }
            fake_summary_path.write_text(json.dumps(fake_summary), encoding="utf-8")
            final_result = {
                "summary": fake_summary,
                "wu_rows": [
                    {
                        "pair_id": "pair_80",
                        "source": "mmlu-pro-math",
                        "label": "A>B",
                        "decision_original": "A>B",
                        "decision_reversed": "B>A",
                        "rubric_count": 1,
                    },
                    {
                        "pair_id": "pair_270",
                        "source": "livebench-reasoning",
                        "label": "B>A",
                        "decision_original": "A=B",
                        "decision_reversed": "A=B",
                        "rubric_count": 1,
                    },
                ],
                "uniform_rows": [
                    {
                        "pair_id": "pair_80",
                        "source": "mmlu-pro-math",
                        "label": "A>B",
                        "decision_original": "A>B",
                        "decision_reversed": "B>A",
                        "rubric_count": 1,
                    },
                    {
                        "pair_id": "pair_270",
                        "source": "livebench-reasoning",
                        "label": "B>A",
                        "decision_original": "A=B",
                        "decision_reversed": "A=B",
                        "rubric_count": 1,
                    },
                ],
                "failures": [
                    {
                        "pair_id": "pair_270",
                        "source_family": "livebench-reasoning",
                        "routing_task_profile_id": "test_profile",
                        "exact_answer_task": True,
                        "code_task": False,
                        "decision": "A=B",
                        "broad_rubric_count": 0,
                        "rubric_count": 1,
                    }
                ],
                "analysis_rows": [
                    {
                        "pair_id": "pair_80",
                        "source_family": "mmlu-pro",
                        "weak_source_labels": ["synthetic_mutation:corrupt_final_answer"],
                        "routing_task_profile_id": "test_profile",
                        "broad_rubric_count": 0,
                        "exact_answer_task": True,
                        "code_task": False,
                        "decision": "A>B",
                        "rubric_count": 1,
                    },
                    {
                        "pair_id": "pair_270",
                        "source_family": "livebench-reasoning",
                        "weak_source_labels": ["synthetic_mutation:drop_constraint"],
                        "routing_task_profile_id": "test_profile",
                        "broad_rubric_count": 0,
                        "exact_answer_task": True,
                        "code_task": False,
                        "decision": "A=B",
                        "rubric_count": 1,
                    },
                ],
                "route_decisions": [
                    {"pair_id": "pair_80", "task_profile_id": "test_profile"},
                    {"pair_id": "pair_270", "task_profile_id": "test_profile"},
                ],
                "paths": {"summary": str(fake_summary_path.resolve()), "split_dir": str(fake_summary_path.parent.parent.resolve())},
            }

            def fake_load_local(path: Path):
                name = Path(path).name
                if name == "judgebench_80_human.json":
                    return [SimpleNamespace(pair_id="pair_80")]
                if name == "judgebench_270_generated.json":
                    return [SimpleNamespace(pair_id="pair_270")]
                return []

            with patch("rubric_gen.compiled.judgebench_eval.load_official_judgebench_pairs", return_value=[]), patch(
                "rubric_gen.compiled.judgebench_eval.load_local_judgebench_subset",
                side_effect=fake_load_local,
            ), patch(
                "rubric_gen.compiled.judgebench_eval.join_local_subset_to_official_pairs",
                return_value=validation_examples,
            ), patch(
                "rubric_gen.compiled.judgebench_eval.run_judgebench_split",
                return_value=final_result,
            ), patch(
                "rubric_gen.compiled.judgebench_eval._assert_split_summary_persisted",
                return_value=None,
            ):
                _, summary = run_judgebench_final_evaluation(
                    train_run_dir=train_run_dir,
                    validation_dataset_path=validation_path,
                    validation_split_name="validation_350",
                    run_dir=tmp_path / "final_run",
                    official_dataset_path=official_path,
                    max_workers=1,
                    write_detailed_outputs=False,
                    resume=False,
                )

            self.assertEqual(summary["validation_failure_analysis"]["failure_count"], 1)
            self.assertTrue((tmp_path / "final_run" / "summaries" / "validation_350_failure_analysis.json").exists())
            self.assertTrue((tmp_path / "final_run" / "summaries" / "validation_350_failures.json").exists())
            self.assertTrue((tmp_path / "final_run" / "summaries" / "validation_350_analysis_rows.json").exists())
            self.assertIn("judgebench_80_human", summary["diagnostic_subsets"])
            self.assertIn("judgebench_270_generated", summary["diagnostic_subsets"])
            self.assertEqual(summary["diagnostic_subsets"]["judgebench_80_human"]["pair_count"], 1)
            self.assertEqual(summary["diagnostic_subsets"]["judgebench_270_generated"]["pair_count"], 1)
            self.assertTrue(
                summary["diagnostic_subsets"]["judgebench_80_human"]["paths"]["summary"].endswith(
                    "judgebench_80_human_summary.json"
                )
            )
            self.assertTrue(summary["paths"]["validation_failures"].endswith("validation_350_failures.json"))


if __name__ == "__main__":
    unittest.main()
