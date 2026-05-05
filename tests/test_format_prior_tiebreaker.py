"""Tests for the format-prior heuristic tiebreaker (Tier C1, v2 plan)."""

from __future__ import annotations

import unittest

from rubric_gen.evaluation.pair_tiebreaker import (
    FormatPriorOutcome,
    format_prior_predict,
    _detect_format_request,
    _format_features,
)


class DetectFormatRequestTests(unittest.TestCase):
    def test_terse_cues(self) -> None:
        self.assertTrue(_detect_format_request("Answer in one sentence.")["wants_terse"])
        self.assertTrue(_detect_format_request("Briefly summarize.")["wants_terse"])
        self.assertFalse(_detect_format_request("What is the diagnosis?")["wants_terse"])

    def test_list_cues(self) -> None:
        self.assertTrue(_detect_format_request("List the differential.")["wants_list"])
        self.assertTrue(_detect_format_request("Use bullet points.")["wants_list"])
        # numbered cue is separate from list
        self.assertTrue(_detect_format_request("Walk me through step-by-step.")["wants_numbered"])

    def test_table_cues(self) -> None:
        self.assertTrue(_detect_format_request("Provide a table comparison.")["wants_table"])
        self.assertFalse(_detect_format_request("Provide a comparison.")["wants_table"])

    def test_explain_cues(self) -> None:
        self.assertTrue(_detect_format_request("Explain in detail.")["wants_explain"])
        self.assertTrue(_detect_format_request("Elaborate on the mechanism.")["wants_explain"])
        self.assertFalse(_detect_format_request("Pick the best option.")["wants_explain"])


class FormatFeaturesTests(unittest.TestCase):
    def test_bullet_count(self) -> None:
        text = "Risk factors:\n- HTN\n- Smoking\n- Diabetes"
        feats = _format_features(text)
        self.assertEqual(feats["bullet_count"], 3.0)

    def test_table_detection(self) -> None:
        text = "| Drug | Dose |\n|------|------|\n| A | 10mg |\n| B | 20mg |"
        feats = _format_features(text)
        self.assertEqual(feats["has_table"], 1.0)
        self.assertGreaterEqual(feats["table_rows"], 3.0)

    def test_no_table_for_pipe_only_text(self) -> None:
        # Single | usage is not a table
        text = "Use option A | option B"
        feats = _format_features(text)
        self.assertEqual(feats["has_table"], 0.0)

    def test_code_blocks(self) -> None:
        text = "Sure:\n```python\nprint('hi')\n```"
        feats = _format_features(text)
        self.assertEqual(feats["code_blocks"], 1.0)


class FormatPriorPredictTests(unittest.TestCase):
    def test_list_request_picks_bulleted_side(self) -> None:
        out = format_prior_predict(
            question="List the top 3 risk factors.",
            response_a="Risk factors include hypertension, smoking, and diabetes.",
            response_b="- Hypertension\n- Smoking\n- Diabetes",
        )
        self.assertEqual(out.verdict, "b")
        self.assertIn("list:B", out.reason)

    def test_table_request_picks_table_side(self) -> None:
        out = format_prior_predict(
            question="Provide a comparison table.",
            response_a="Drug A is better than Drug B for this indication.",
            response_b="| Drug | Dose |\n|------|------|\n| A | 10mg |\n| B | 20mg |",
        )
        self.assertEqual(out.verdict, "b")
        self.assertIn("table:B", out.reason)

    def test_terse_request_picks_short_side(self) -> None:
        out = format_prior_predict(
            question="Answer in one sentence.",
            response_a="D.",
            response_b="The answer is D because " + ("hypochlorous acid " * 50),
        )
        self.assertEqual(out.verdict, "a")
        self.assertIn("terse:A", out.reason)

    def test_explain_request_picks_long_side(self) -> None:
        long_explanation = (
            "When chlorine reacts with water it undergoes hydrolysis to form "
            "hypochlorous acid (HOCl) and hypochlorite ions (OCl-). " * 30
        )
        out = format_prior_predict(
            question="Explain in detail the mechanism.",
            response_a="Hypochlorous acid disinfects.",
            response_b=long_explanation,
        )
        self.assertEqual(out.verdict, "b")
        self.assertIn("explain:B", out.reason)

    def test_no_format_signal_returns_tie(self) -> None:
        out = format_prior_predict(
            question="What is the diagnosis?",
            response_a="A is correct",
            response_b="B is correct",
        )
        self.assertEqual(out.verdict, "tie")
        self.assertEqual(out.reason, "no_format_signal")
        self.assertEqual(out.votes_a, 0)
        self.assertEqual(out.votes_b, 0)

    def test_balanced_format_signal_returns_tie(self) -> None:
        # Q wants both terse + list; one side is terse but un-listed, the
        # other is bulleted but verbose.  Each side wins one vote -> tie.
        out = format_prior_predict(
            question="List the top 3 risk factors briefly.",
            response_a="HTN.",
            response_b="- Hypertension is a major risk factor for cardiovascular events\n"
                       "- Smoking damages endothelium and accelerates atherosclerosis\n"
                       "- Diabetes mellitus contributes via metabolic and vascular pathways",
        )
        self.assertEqual(out.verdict, "tie")
        self.assertEqual(out.votes_a, 1)
        self.assertEqual(out.votes_b, 1)
        self.assertIn("format_prior_tied", out.reason)

    def test_outcome_dataclass_carries_cues(self) -> None:
        out = format_prior_predict(
            question="Briefly: what is the diagnosis?",
            response_a="MI",
            response_b="It looks like a myocardial infarction based on " + ("the ECG and troponin " * 50),
        )
        self.assertIsInstance(out, FormatPriorOutcome)
        self.assertTrue(out.cues["wants_terse"])


if __name__ == "__main__":
    unittest.main()
