from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rubric_gen.compiled.judgebench_selection_audit import build_selection_audit


def _write_summary(
    root: Path,
    name: str,
    *,
    pair_count: int = 80,
    overall_wu: float,
    overall_uniform: float,
    family_wu: dict[str, float],
    failure_count: int,
    exact_answer_failures: int,
    tie_failures: int,
    fold_overalls: list[float],
    train_fit_overall_wu: float | None = None,
    train_fit_overall_uniform: float | None = None,
    train_fit_family_wu: dict[str, float] | None = None,
    train_fit_failure_count: int = 0,
    train_fit_exact_answer_failures: int = 0,
    train_fit_tie_failures: int = 0,
    blind_parity_bootstrap: bool = True,
    focus_verifier_coverage_rate: float = 0.9,
    low_confidence_bucket_accuracy: float = 0.8,
    exact_answer_parser_success_rate: float = 0.8,
) -> Path:
    run_dir = root / name
    (run_dir / "summaries").mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "compiled_judgebench_train_only_summary_v1",
        "run_dir": str(run_dir),
        "oof_summary": {
            "pair_count": pair_count,
            "wu_metrics": {"overall": overall_wu, **family_wu},
            "uniform_metrics": {"overall": overall_uniform},
            "calibration_metrics": {
                "verifier_coverage_rate": focus_verifier_coverage_rate,
                "focus_verifier_coverage_rate": focus_verifier_coverage_rate,
                "low_confidence_bucket_accuracy": low_confidence_bucket_accuracy,
                "exact_answer_parser_success_rate": exact_answer_parser_success_rate,
                "discriminator_usage_rate": 0.2,
            },
        },
        "blind_parity_bootstrap": blind_parity_bootstrap,
        "failure_analysis": {
            "pair_count": pair_count,
            "failure_count": failure_count,
            "failure_rate": failure_count / max(1, pair_count),
            "exact_answer_failures": exact_answer_failures,
            "tie_failures": tie_failures,
        },
        "fold_summaries": [
            {"wu_metrics": {"overall": value}}
            for value in fold_overalls
        ],
    }
    if train_fit_overall_wu is not None:
        summary["train_fit_summary"] = {
            "pair_count": pair_count,
            "wu_metrics": {"overall": train_fit_overall_wu, **(train_fit_family_wu or {})},
            "uniform_metrics": {
                "overall": train_fit_overall_uniform
                if train_fit_overall_uniform is not None
                else train_fit_overall_wu
            },
        }
        summary["train_fit_failure_analysis"] = {
            "pair_count": pair_count,
            "failure_count": train_fit_failure_count,
            "failure_rate": train_fit_failure_count / max(1, pair_count),
            "exact_answer_failures": train_fit_exact_answer_failures,
            "tie_failures": train_fit_tie_failures,
        }
    (run_dir / "summaries" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


class JudgeBenchSelectionAuditTests(unittest.TestCase):
    def test_prefers_gate_passing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stronger = _write_summary(
                root,
                "stronger",
                overall_wu=75.0,
                overall_uniform=70.0,
                family_wu={
                    "mmlu-pro": 70.0,
                    "livebench-reasoning": 68.75,
                    "livebench-math": 100.0,
                    "livecodebench": 76.92,
                },
                failure_count=20,
                exact_answer_failures=17,
                tie_failures=4,
                fold_overalls=[81.0, 75.0, 70.0, 74.0],
            )
            noisy = _write_summary(
                root,
                "noisy",
                overall_wu=75.0,
                overall_uniform=65.0,
                family_wu={
                    "mmlu-pro": 72.5,
                    "livebench-reasoning": 62.5,
                    "livebench-math": 90.9,
                    "livecodebench": 84.6,
                },
                failure_count=20,
                exact_answer_failures=18,
                tie_failures=3,
                fold_overalls=[81.0, 70.0, 60.0, 89.0],
            )

            audit = build_selection_audit({"stronger": [stronger], "noisy": [noisy]})

            self.assertEqual(audit["recommendation"], "stronger")
            records = {row["label"]: row for row in audit["candidates"]}
            self.assertTrue(records["stronger"]["gates"]["passes_all"])
            self.assertFalse(records["noisy"]["gates"]["results"]["min_family_wu"])
            self.assertFalse(records["noisy"]["gates"]["results"]["max_wu_uniform_gap"])

    def test_blind_validation_gate_tracks_blind_parity_and_verifier_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _write_summary(
                root,
                "candidate",
                overall_wu=86.0,
                overall_uniform=84.0,
                family_wu={
                    "mmlu-pro": 83.0,
                    "livebench-reasoning": 82.0,
                    "livebench-math": 88.0,
                    "livecodebench": 87.0,
                },
                failure_count=8,
                exact_answer_failures=4,
                tie_failures=1,
                fold_overalls=[85.5, 86.0, 86.5, 86.0],
                blind_parity_bootstrap=False,
                focus_verifier_coverage_rate=0.4,
                low_confidence_bucket_accuracy=0.8,
                exact_answer_parser_success_rate=0.9,
            )

            audit = build_selection_audit({"candidate": [candidate]})
            gate = audit["candidates"][0]["blind_validation_gate"]

            self.assertFalse(gate["passes_all"])
            self.assertFalse(gate["results"]["blind_parity_bootstrap"])
            self.assertFalse(gate["results"]["min_focus_verifier_coverage_rate"])

    def test_aggregates_replicates_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rep1 = _write_summary(
                root,
                "rep1",
                overall_wu=73.0,
                overall_uniform=70.0,
                family_wu={
                    "mmlu-pro": 68.0,
                    "livebench-reasoning": 67.0,
                    "livebench-math": 90.0,
                    "livecodebench": 80.0,
                },
                failure_count=21,
                exact_answer_failures=18,
                tie_failures=2,
                fold_overalls=[74.0, 72.0, 71.0, 75.0],
            )
            rep2 = _write_summary(
                root,
                "rep2",
                overall_wu=71.0,
                overall_uniform=69.0,
                family_wu={
                    "mmlu-pro": 66.0,
                    "livebench-reasoning": 65.5,
                    "livebench-math": 88.0,
                    "livecodebench": 79.0,
                },
                failure_count=22,
                exact_answer_failures=19,
                tie_failures=1,
                fold_overalls=[70.0, 71.0, 72.0, 71.0],
            )

            audit = build_selection_audit({"candidate": [rep1, rep2]})
            record = audit["candidates"][0]

            self.assertEqual(record["run_count"], 2)
            self.assertAlmostEqual(record["metrics"]["mean_overall_wu"], 72.0)
            self.assertAlmostEqual(record["metrics"]["worst_family_wu"], 65.5)
            self.assertEqual(record["metrics"]["max_failure_count"], 22)
            self.assertTrue(record["gates"]["passes_all"])

    def test_prefers_target_mix_candidate_when_weights_emphasize_mmlu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = _write_summary(
                root,
                "broad",
                pair_count=120,
                overall_wu=74.0,
                overall_uniform=71.0,
                family_wu={
                    "mmlu-pro": 58.0,
                    "livebench-reasoning": 82.0,
                    "livebench-math": 78.0,
                    "livecodebench": 76.0,
                },
                failure_count=30,
                exact_answer_failures=16,
                tie_failures=8,
                fold_overalls=[75.0, 74.0, 73.0, 74.0, 74.0],
            )
            mmlu_weighted = _write_summary(
                root,
                "mmlu_weighted",
                pair_count=120,
                overall_wu=72.0,
                overall_uniform=70.0,
                family_wu={
                    "mmlu-pro": 72.0,
                    "livebench-reasoning": 68.0,
                    "livebench-math": 71.0,
                    "livecodebench": 74.0,
                },
                failure_count=29,
                exact_answer_failures=15,
                tie_failures=7,
                fold_overalls=[72.0, 71.0, 73.0, 72.0, 72.0],
            )

            audit = build_selection_audit(
                {"broad": [broad], "mmlu_weighted": [mmlu_weighted]},
                target_family_weights={
                    "mmlu-pro": 154,
                    "livebench-reasoning": 98,
                    "livebench-math": 56,
                    "livecodebench": 42,
                },
                hard_family_weights={
                    "mmlu-pro": 40,
                    "livebench-reasoning": 16,
                    "livebench-math": 11,
                    "livecodebench": 13,
                },
                max_failure_threshold=40,
                max_exact_answer_failure_threshold=25,
                max_tie_threshold=15,
            )

            self.assertEqual(audit["recommendation"], "mmlu_weighted")
            records = {row["label"]: row for row in audit["candidates"]}
            self.assertGreater(
                records["mmlu_weighted"]["metrics"]["mean_target_mix_wu"],
                records["broad"]["metrics"]["mean_target_mix_wu"],
            )
            self.assertGreater(
                records["mmlu_weighted"]["selection_score"],
                records["broad"]["selection_score"],
            )

    def test_prefers_oof_primary_when_train_fit_is_only_tiebreak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof_only_better = _write_summary(
                root,
                "oof_only_better",
                overall_wu=81.0,
                overall_uniform=79.0,
                family_wu={
                    "mmlu-pro": 78.0,
                    "livebench-reasoning": 80.0,
                    "livebench-math": 84.0,
                    "livecodebench": 82.0,
                },
                failure_count=10,
                exact_answer_failures=6,
                tie_failures=1,
                fold_overalls=[80.0, 81.0, 82.0, 81.0],
                train_fit_overall_wu=86.0,
                train_fit_overall_uniform=85.0,
                train_fit_family_wu={
                    "mmlu-pro": 84.0,
                    "livebench-reasoning": 86.0,
                    "livebench-math": 88.0,
                    "livecodebench": 86.0,
                },
                train_fit_failure_count=6,
                train_fit_exact_answer_failures=4,
                train_fit_tie_failures=0,
            )
            train_fit_better = _write_summary(
                root,
                "train_fit_better",
                overall_wu=79.0,
                overall_uniform=77.5,
                family_wu={
                    "mmlu-pro": 76.0,
                    "livebench-reasoning": 78.0,
                    "livebench-math": 82.0,
                    "livecodebench": 80.0,
                },
                failure_count=12,
                exact_answer_failures=7,
                tie_failures=1,
                fold_overalls=[79.0, 79.0, 78.0, 80.0],
                train_fit_overall_wu=92.0,
                train_fit_overall_uniform=91.0,
                train_fit_family_wu={
                    "mmlu-pro": 90.0,
                    "livebench-reasoning": 91.0,
                    "livebench-math": 94.0,
                    "livecodebench": 93.0,
                },
                train_fit_failure_count=3,
                train_fit_exact_answer_failures=2,
                train_fit_tie_failures=0,
            )

            audit = build_selection_audit({"oof_only_better": [oof_only_better], "train_fit_better": [train_fit_better]})

            self.assertEqual(audit["recommendation"], "oof_only_better")
            records = {row["label"]: row for row in audit["candidates"]}
            self.assertTrue(records["train_fit_better"]["metrics"]["train_fit_available"])
            self.assertGreater(
                records["train_fit_better"]["metrics"]["mean_train_fit_wu"],
                records["oof_only_better"]["metrics"]["mean_train_fit_wu"],
            )
            self.assertGreater(
                records["oof_only_better"]["selection_score"],
                records["train_fit_better"]["selection_score"],
            )
            self.assertTrue(records["train_fit_better"]["gates"]["results"]["min_train_fit_family_wu"])
            self.assertLess(
                records["oof_only_better"]["metrics"]["mean_locked_policy_train_oof_wu_gap"],
                records["train_fit_better"]["metrics"]["mean_locked_policy_train_oof_wu_gap"],
            )
            self.assertIn("blind_validation_gate", records["oof_only_better"])
            self.assertIn("recovery_guidance", audit)
            self.assertIn("mmlu-pro", audit["recovery_guidance"]["focus_families"])


if __name__ == "__main__":
    unittest.main()
