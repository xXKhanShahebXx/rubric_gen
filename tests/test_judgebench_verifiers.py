import unittest

from rubric_gen.compiled.judgebench_verifiers import (
    JudgeBenchVerifierCandidateSignal,
    evaluate_pair_verifier,
)


class JudgeBenchVerifierTests(unittest.TestCase):
    def test_mmlu_exact_match_triggers_high_confidence_decision(self) -> None:
        outcome = evaluate_pair_verifier(
            source_family="mmlu-pro",
            features={
                "exact_answer_task": True,
                "requested_answer_mode": "repeated_choice",
                "choice_map_available": True,
                "reference_value_available": True,
            },
            signal_a=JudgeBenchVerifierCandidateSignal(
                extracted_value="ccccc",
                explicit=True,
                exact_match=False,
                format_ok=True,
                consistent=True,
                option_map_available=True,
                choice_letter="C",
                choice_value="60",
            ),
            signal_b=JudgeBenchVerifierCandidateSignal(
                extracted_value="aaaaa",
                explicit=True,
                exact_match=True,
                format_ok=True,
                consistent=True,
                option_map_available=True,
                choice_letter="A",
                choice_value="30",
            ),
        )

        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "B>A")
        self.assertEqual(outcome.confidence, "high")
        self.assertEqual(outcome.reason, "b_exact_answer_match")

    def test_reasoning_low_margin_does_not_trigger_verifier_override(self) -> None:
        outcome = evaluate_pair_verifier(
            source_family="livebench-reasoning",
            features={
                "exact_answer_task": True,
                "requested_answer_mode": "single_digit",
                "choice_map_available": False,
                "reference_value_available": True,
            },
            signal_a=JudgeBenchVerifierCandidateSignal(
                extracted_value="3",
                explicit=True,
                exact_match=False,
                format_ok=True,
                consistent=True,
            ),
            signal_b=JudgeBenchVerifierCandidateSignal(
                extracted_value="4",
                explicit=True,
                exact_match=False,
                format_ok=True,
                consistent=True,
            ),
        )

        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "")
        self.assertIn(outcome.confidence, {"", "low"})
        self.assertEqual(outcome.reason, "insufficient_verifier_margin")


if __name__ == "__main__":
    unittest.main()
