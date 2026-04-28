from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from rubric_gen.compiled.archived_replay import (
    ReplayReport,
    replay_run,
    write_replay_report,
)
from rubric_gen.compiled.rubric_library import (
    RubricLibrary,
    RubricLibraryCriterion,
)


_CLEAN_REASONING_A = (
    "Let me solve the puzzle step by step.\n"
    "Alice sits at position 1.\n"
    "Bob sits at position 2.\n"
    "Carol sits at position 3.\n"
    "All clues satisfied.\n"
    "Final answer: Alice is leftmost.\n"
)
_INCOMPLETE_REASONING_B = (
    "Let me try. Alice might be at position 1. Unclear.\n"
    "Cannot determine the rest.\n"
    "Final answer: unknown.\n"
)


def _build_run_dir(tmp: Path, *, failures: List[Dict[str, Any]], examples: Dict[str, Dict[str, Any]]) -> Path:
    run_dir = tmp / "run"
    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(parents=True)
    (summaries_dir / "oof_failures.json").write_text(json.dumps(failures), encoding="utf-8")
    (summaries_dir / "oof_failure_analysis.json").write_text(
        json.dumps(
            {
                "schema": "compiled_judgebench_failure_analysis_v1",
                "pair_count": len(failures),
                "failure_count": len(failures),
                "tie_failures": sum(1 for f in failures if f.get("decision") == "A=B"),
                "exact_answer_failures": sum(1 for f in failures if f.get("exact_answer_task")),
            }
        ),
        encoding="utf-8",
    )
    fold_dir = run_dir / "folds" / "fold_00" / "dev" / "examples"
    fold_dir.mkdir(parents=True)
    for pair_id, artifact in examples.items():
        (fold_dir / f"{pair_id}.json").write_text(json.dumps(artifact), encoding="utf-8")
    return run_dir


def _build_artifact(response_a: str, response_b: str) -> Dict[str, Any]:
    return {
        "schema": "compiled_judgebench_example_v2",
        "pair": {},
        "candidates": [
            {
                "candidate_id": "a",
                "text": response_a,
                "metadata": {"pair_position": "A"},
            },
            {
                "candidate_id": "b",
                "text": response_b,
                "metadata": {"pair_position": "B"},
            },
        ],
        "scoring": {
            "whitened_uniform": {
                "result": {
                    "decision": "A=B",
                    "score_A": 0.5,
                    "score_B": 0.5,
                    "whitening_unstable": False,
                }
            }
        },
    }


class ReplayRunTests(unittest.TestCase):
    def test_process_verifier_resolves_clear_reasoning_tie(self) -> None:
        with TemporaryDirectory() as tmp:
            failures = [
                {
                    "pair_id": "pair_reason_1",
                    "source_family": "livebench-reasoning",
                    "label": "A>B",
                    "decision": "A=B",
                    "score_A": 0.5,
                    "score_B": 0.5,
                    "exact_answer_task": False,
                    "code_task": False,
                }
            ]
            examples = {
                "pair_reason_1": _build_artifact(_CLEAN_REASONING_A, _INCOMPLETE_REASONING_B),
            }
            run_dir = _build_run_dir(Path(tmp), failures=failures, examples=examples)
            report = replay_run(run_dir)
            self.assertEqual(report.total_failures, 1)
            self.assertEqual(report.tie_failures_in_run, 1)
            self.assertEqual(report.process_verifier_would_resolve, 1)
            self.assertEqual(report.widened_gate_would_route_to_discriminator, 1)
            details = report.details
            self.assertEqual(len(details), 1)
            self.assertTrue(details[0].process_verifier_resolves)

    def test_missing_fold_artifact_is_counted(self) -> None:
        with TemporaryDirectory() as tmp:
            failures = [
                {
                    "pair_id": "pair_missing",
                    "source_family": "livebench-reasoning",
                    "label": "A>B",
                    "decision": "A=B",
                    "score_A": 0.5,
                    "score_B": 0.5,
                }
            ]
            run_dir = _build_run_dir(Path(tmp), failures=failures, examples={})
            report = replay_run(run_dir)
            self.assertEqual(report.fold_artifacts_missing, 1)

    def test_library_criteria_injected(self) -> None:
        with TemporaryDirectory() as tmp:
            failures = [
                {
                    "pair_id": "pair_reason_1",
                    "source_family": "livebench-reasoning",
                    "label": "A>B",
                    "decision": "A=B",
                    "score_A": 0.5,
                    "score_B": 0.5,
                }
            ]
            examples = {
                "pair_reason_1": _build_artifact(_CLEAN_REASONING_A, _INCOMPLETE_REASONING_B),
            }
            run_dir = _build_run_dir(Path(tmp), failures=failures, examples=examples)
            library = RubricLibrary(
                version="v1",
                criteria=[
                    RubricLibraryCriterion(
                        criterion_id="lib_1",
                        dimension="assignment_completeness",
                        label="L1",
                        requirement="R1",
                        severity_tier="hard_gate",
                        applicable_families=("livebench-reasoning",),
                        source_tag="test",
                    ),
                    RubricLibraryCriterion(
                        criterion_id="lib_2",
                        dimension="conclusion_grounded",
                        label="L2",
                        requirement="R2",
                        severity_tier="high",
                        applicable_families=("livebench-reasoning",),
                        source_tag="test",
                    ),
                ],
            )
            report = replay_run(run_dir, library=library, library_top_k=4)
            self.assertEqual(report.details[0].library_criteria_count, 2)


class WriteReplayReportTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        report = ReplayReport(
            run_dir="run",
            total_failures=0,
            tie_failures_in_run=0,
            exact_answer_failures_in_run=0,
            process_verifier_would_resolve=0,
            widened_gate_would_route_to_discriminator=0,
            library_criteria_avg_per_example=0.0,
            fold_artifacts_missing=0,
            source_family_counts={},
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_replay_report(report, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "compiled_judgebench_archived_replay_v1")


if __name__ == "__main__":
    unittest.main()
