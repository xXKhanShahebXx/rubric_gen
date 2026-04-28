from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rubric_gen.compiled.judgebench_selection_audit import (
    _SPLIT_SUMMARY_SCHEMA,
    _TRAIN_ONLY_SUMMARY_SCHEMA,
    _load_summary,
    evaluate_v2_promotion_gates,
)


def _build_split_summary(
    *,
    overall: float = 86.5,
    mmlu_pro: float = 82.5,
    reasoning: float = 83.0,
    math: float = 82.5,
    livecodebench: float = 97.5,
    pair_count: int = 320,
    tie_failures: int = 10,
    exact_answer_failures: int = 15,
    reasoning_trigger: float = 0.15,
    order_disagreement: float = 0.0,
) -> dict:
    return {
        "schema": _SPLIT_SUMMARY_SCHEMA,
        "pair_count": pair_count,
        "wu_metrics": {
            "overall": overall,
            "mmlu-pro": mmlu_pro,
            "livebench-reasoning": reasoning,
            "livebench-math": math,
            "livecodebench": livecodebench,
        },
        "calibration_metrics": {
            "verifier_trigger_rate_by_family": {"livebench-reasoning": reasoning_trigger},
            "discriminator_order_disagreement_rate": order_disagreement,
        },
        "tie_failures": tie_failures,
        "exact_answer_failures": exact_answer_failures,
    }


def _build_failure_analysis(
    *,
    pair_count: int = 320,
    tie_failures: int = 10,
    exact_answer_failures: int = 15,
) -> dict:
    return {
        "schema": "compiled_judgebench_failure_analysis_v1",
        "pair_count": pair_count,
        "tie_failures": tie_failures,
        "exact_answer_failures": exact_answer_failures,
    }


class EvaluateV2PromotionGatesTests(unittest.TestCase):
    def test_all_gates_pass_on_strong_summary(self) -> None:
        summary = _build_split_summary(
            overall=87.0,
            mmlu_pro=83.0,
            reasoning=83.0,
            math=83.0,
            tie_failures=8,
            exact_answer_failures=15,
            reasoning_trigger=0.2,
        )
        result = evaluate_v2_promotion_gates(summary)
        self.assertTrue(result["passes_all"])
        self.assertTrue(result["gates"]["overall_wu"])
        self.assertTrue(result["gates"]["reasoning_wu"])

    def test_failing_reasoning_gate_blocks_promotion(self) -> None:
        summary = _build_split_summary(reasoning=77.5, reasoning_trigger=0.0)
        result = evaluate_v2_promotion_gates(summary)
        self.assertFalse(result["passes_all"])
        self.assertFalse(result["gates"]["reasoning_wu"])
        self.assertFalse(result["gates"]["reasoning_verifier_trigger_rate"])

    def test_tie_failure_rate_above_threshold_fails(self) -> None:
        summary = _build_split_summary(tie_failures=32)
        result = evaluate_v2_promotion_gates(summary)
        self.assertFalse(result["gates"]["tie_failure_rate"])

    def test_external_slice_min_wu_is_included_when_available(self) -> None:
        summary = _build_split_summary()
        summary["external_slice_summary"] = {
            "available": True,
            "slices": {
                "helpsteer3_val": {"available": True, "wu_score": 76.5},
                "ppe_mmlu_gpqa_val": {"available": True, "wu_score": 72.0},
            },
        }
        result = evaluate_v2_promotion_gates(summary)
        self.assertIn("external_slice_wu", result["gates"])
        self.assertFalse(result["gates"]["external_slice_wu"])
        self.assertEqual(result["metrics"]["external_slice_wu_min"], 72.0)

    def test_train_only_summary_schema_is_accepted(self) -> None:
        oof = _build_split_summary()
        failure = _build_failure_analysis()
        train_only_summary = {
            "schema": _TRAIN_ONLY_SUMMARY_SCHEMA,
            "oof_summary": oof,
            "failure_analysis": failure,
            "external_slice_summary": {
                "available": True,
                "slices": {
                    "helpsteer3_val": {"available": True, "wu_score": 78.0},
                    "ppe_mmlu_gpqa_val": {"available": True, "wu_score": 81.0},
                },
            },
        }
        result = evaluate_v2_promotion_gates(train_only_summary)
        self.assertTrue(result["passes_all"])


class LoadSummaryAcceptsSplitSchemaTests(unittest.TestCase):
    def test_load_summary_from_split_schema_oof_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "summaries").mkdir(parents=True)
            oof = _build_split_summary()
            failure = _build_failure_analysis()
            (run_dir / "summaries" / "oof_summary.json").write_text(
                json.dumps(oof), encoding="utf-8"
            )
            (run_dir / "summaries" / "oof_failure_analysis.json").write_text(
                json.dumps(failure), encoding="utf-8"
            )
            summary = _load_summary(run_dir)
            self.assertEqual(summary["schema"], _TRAIN_ONLY_SUMMARY_SCHEMA)
            self.assertEqual(summary["oof_summary"]["schema"], _SPLIT_SUMMARY_SCHEMA)
            self.assertEqual(
                summary["failure_analysis"]["schema"], "compiled_judgebench_failure_analysis_v1"
            )


if __name__ == "__main__":
    unittest.main()
