from __future__ import annotations

import unittest

from rubric_gen.compiled.reasoning_process_verifier import (
    ReasoningProcessVerifier,
    ReasoningProcessVerifierConfig,
    deterministic_extractor,
    score_response,
)


_CLEAN_RESPONSE = """Let me solve this logic puzzle step by step.

Alice is at position 1.
Bob is at position 2.
Carol is at position 3.
Dave is at position 4.

Checking constraints:
- Alice is left of Bob.
- Dave is the rightmost.
- All slots filled with unique entities.

Final answer: Alice sits in position 1.
"""

_INCOMPLETE_RESPONSE = """Let me solve this logic puzzle step by step.

Alice is at position 1.
Bob is unclear.
Carol is unknown.
Dave is somewhere.

Final answer: cannot determine.
"""

_CONTRADICTORY_RESPONSE = """Let me try.

Alice is at position 1.
Bob is at position 1.
Actually wait, Bob should be at position 2.
On second thought, maybe Carol is at 1.
Actually, wait, Alice is at position 2.

Final answer: position assignments unclear.
"""


class DeterministicExtractorTests(unittest.TestCase):
    def test_extracts_entities_and_assignments(self) -> None:
        extract = deterministic_extractor("A", _CLEAN_RESPONSE, {})
        self.assertIn("Alice", extract.entities)
        self.assertIn("Bob", extract.entities)
        self.assertGreaterEqual(len(extract.assignments), 1)
        self.assertIn("final answer", extract.final_answer_line.lower())
        self.assertEqual(extract.unknown_markers, 0)
        self.assertEqual(extract.reversal_markers, 0)

    def test_detects_unknown_markers(self) -> None:
        extract = deterministic_extractor("A", _INCOMPLETE_RESPONSE, {})
        self.assertGreater(extract.unknown_markers, 0)

    def test_detects_reversal_markers(self) -> None:
        extract = deterministic_extractor("A", _CONTRADICTORY_RESPONSE, {})
        self.assertGreaterEqual(extract.reversal_markers, 2)


class ScoringTests(unittest.TestCase):
    def test_clean_response_scores_higher_than_incomplete(self) -> None:
        config = ReasoningProcessVerifierConfig()
        clean = score_response(deterministic_extractor("A", _CLEAN_RESPONSE, {}), config=config)
        incomplete = score_response(deterministic_extractor("B", _INCOMPLETE_RESPONSE, {}), config=config)
        self.assertGreater(clean.composite, incomplete.composite)
        self.assertGreater(clean.assignment_completeness, incomplete.assignment_completeness)
        self.assertGreaterEqual(clean.contradiction_avoidance, incomplete.contradiction_avoidance)

    def test_contradictory_response_scores_low_on_contradiction(self) -> None:
        config = ReasoningProcessVerifierConfig()
        contradictory = score_response(
            deterministic_extractor("A", _CONTRADICTORY_RESPONSE, {}),
            config=config,
        )
        clean = score_response(deterministic_extractor("B", _CLEAN_RESPONSE, {}), config=config)
        self.assertLess(contradictory.contradiction_avoidance, clean.contradiction_avoidance)


class ReasoningProcessVerifierPairTests(unittest.TestCase):
    def test_applies_only_to_livebench_reasoning(self) -> None:
        verifier = ReasoningProcessVerifier()
        self.assertTrue(verifier.applies_to("livebench-reasoning"))
        self.assertFalse(verifier.applies_to("mmlu-pro"))

    def test_triggers_when_one_response_is_clearly_better(self) -> None:
        verifier = ReasoningProcessVerifier(
            config=ReasoningProcessVerifierConfig(trigger_margin=0.05, medium_confidence_margin=0.1),
        )
        outcome = verifier.evaluate(
            source_family="livebench-reasoning",
            response_a=_CLEAN_RESPONSE,
            response_b=_INCOMPLETE_RESPONSE,
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "A>B")
        self.assertIn(outcome.confidence, {"medium", "high"})
        self.assertGreater(outcome.margin, 0.0)

    def test_does_not_trigger_when_both_are_similar(self) -> None:
        verifier = ReasoningProcessVerifier(
            config=ReasoningProcessVerifierConfig(trigger_margin=0.25),
        )
        outcome = verifier.evaluate(
            source_family="livebench-reasoning",
            response_a=_CLEAN_RESPONSE,
            response_b=_CLEAN_RESPONSE,
        )
        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "")


if __name__ == "__main__":
    unittest.main()
