import json
import tempfile
import unittest
from pathlib import Path

from rubric_gen.compiled.compiler import build_task_family_spec, build_task_ontology, compile_task_case_rubric
from rubric_gen.compiled.contrast_strategies import clear_dynamic_contrast_strategies
from rubric_gen.compiled.pilot_runner import run_pilot
from rubric_gen.compiled.profile_bootstrap import resolve_or_bootstrap_task_profile
from rubric_gen.compiled.task_profiles import clear_dynamic_task_profiles
from rubric_gen.types import ExampleRecord


def _agentic_example() -> ExampleRecord:
    return ExampleRecord(
        example_id="workflow__1",
        source="ops_suite",
        source_id="1",
        dataset_subset="pilot",
        conversation=(
            "Observation: query returned 12 incidents.\n"
            "Action: grouped incidents by service.\n"
            "Result: payments had the highest error count.\n"
            "Verified the counts and noted retry failures for one shard."
        ),
        task_prompt="Create an incident workflow summary from the observed steps, tool results, and verification notes.",
        reference_artifact=(
            "Step 1: Reviewed tool results for 12 incidents.\n"
            "Step 2: Grouped incidents by service.\n"
            "Result: Payments had the highest error count.\n"
            "Verification: Counts rechecked.\n"
            "Failure handling: Retry failures remained on one shard."
        ),
        augmented_artifact="Payments had the highest error count.",
        artifact_kind="workflow_output",
    )


def _rewrite_example() -> ExampleRecord:
    return ExampleRecord(
        example_id="rewrite__auto",
        source="rewrite_suite",
        source_id="1",
        dataset_subset="pilot",
        conversation="Original text: The client missed one deadline but delivered the revised summary the next day.",
        task_prompt="Rewrite this paragraph into concise bullet points while preserving meaning.",
        reference_artifact="- Client missed one deadline\n- Revised summary delivered the next day",
        augmented_artifact="The client missed a deadline.",
        artifact_kind="artifact",
    )


def _judgebench_reasoning_example() -> ExampleRecord:
    return ExampleRecord(
        example_id="judgebench_reasoning__1",
        source="livebench-reasoning-logic",
        source_id="pair_1",
        dataset_subset="train_80",
        conversation="",
        task_prompt=(
            "A logic puzzle gives several clues and asks for a final answer. "
            "Return a single digit in the format ***N***."
        ),
        reference_artifact="***3***",
        augmented_artifact="The answer might be 2.",
        artifact_kind="response",
        metadata={"source_family": "livebench-reasoning"},
    )


def _judgebench_code_example() -> ExampleRecord:
    return ExampleRecord(
        example_id="judgebench_code__1",
        source="livecodebench",
        source_id="pair_code",
        dataset_subset="train_80",
        conversation="",
        task_prompt="Given input constraints, write Python code that returns the correct answer for each test case.",
        reference_artifact=(
            "class Solution:\n"
            "    def solve(self, nums):\n"
            "        data = input().split()\n"
            "        for i in range(len(nums)):\n"
            "            if nums[i] == 0:\n"
            "                return False\n"
            "        return True"
        ),
        augmented_artifact="return True",
        artifact_kind="response",
        metadata={"source_family": "livecodebench"},
    )


class ProfileBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_dynamic_task_profiles()
        clear_dynamic_contrast_strategies()

    def tearDown(self) -> None:
        clear_dynamic_task_profiles()
        clear_dynamic_contrast_strategies()

    def test_confident_builtin_match_reuses_existing_profile(self) -> None:
        result = resolve_or_bootstrap_task_profile([_rewrite_example()])

        self.assertFalse(result.bootstrap_used)
        self.assertEqual(result.profile.task_profile_id, "rewrite_editing")

    def test_forced_auto_bootstrap_registers_runtime_profile(self) -> None:
        result = resolve_or_bootstrap_task_profile(
            [_agentic_example()],
            explicit="auto",
            bootstrap_iterations=2,
        )

        self.assertTrue(result.bootstrap_used)
        self.assertTrue(result.profile.task_profile_id.startswith("auto_"))
        self.assertFalse(result.profile.built_in)
        self.assertEqual(result.profile.parent_profile_id, "agentic_workflows")
        self.assertGreaterEqual(len(result.strategy.mutation_ids), 3)

        ontology = build_task_ontology(result.profile.task_profile_id)
        family = build_task_family_spec(
            result.profile.default_task_family_id,
            ontology,
            task_profile_id=result.profile.task_profile_id,
        )
        rubric = compile_task_case_rubric(
            _agentic_example(),
            ontology,
            family,
            task_profile_id=result.profile.task_profile_id,
        )

        self.assertEqual(rubric.task_profile_id, result.profile.task_profile_id)
        self.assertEqual(rubric.artifact_label, result.profile.artifact_label)

    def test_run_pilot_with_auto_profile_records_bootstrap_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "auto_dataset.json"
            dataset.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "source": "ops_suite",
                                "source_id": "row_1",
                                "instruction": "Create an incident workflow summary from the observed steps, tool results, and verification notes.",
                                "input": (
                                    "Observation: query returned 12 incidents.\n"
                                    "Action: grouped incidents by service.\n"
                                    "Result: payments had the highest error count.\n"
                                    "Verified the counts and noted retry failures for one shard."
                                ),
                                "reference_output": (
                                    "Step 1: Reviewed tool results for 12 incidents.\n"
                                    "Step 2: Grouped incidents by service.\n"
                                    "Result: Payments had the highest error count.\n"
                                    "Verification: Counts rechecked.\n"
                                    "Failure handling: Retry failures remained on one shard."
                                ),
                                "candidate_output": "Payments had the highest error count.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            run_dir = run_pilot(
                dataset_path=dataset,
                run_name="auto_profile_smoke",
                start=0,
                limit=0,
                design_n=0,
                validation_n=0,
                pilot_n=1,
                source_filter=None,
                out_root=Path(tmp),
                write_csv=False,
                judge_mode="heuristic",
                task_profile="auto",
                bootstrap_iterations=2,
            )
            summary = json.loads((run_dir / "summaries" / "run_summary.json").read_text(encoding="utf-8"))

        self.assertTrue(summary["profile_resolution"]["bootstrap_used"])
        self.assertTrue(summary["profile_resolution"]["resolved_task_profile_id"].startswith("auto_"))
        self.assertEqual(summary["params"]["bootstrap_iterations"], 2)

    def test_judgebench_reasoning_bootstrap_prioritizes_final_answer_mutations(self) -> None:
        result = resolve_or_bootstrap_task_profile(
            [_judgebench_reasoning_example()],
            explicit="auto",
            bootstrap_iterations=1,
        )

        self.assertTrue(result.bootstrap_used)
        self.assertGreaterEqual(len(result.strategy.mutation_ids), 2)
        self.assertEqual(result.strategy.mutation_ids[0], "corrupt_final_answer")
        self.assertIn("clue_consistency", result.profile.discovery_dimensions)
        self.assertEqual(result.diagnostics.get("source_family_hint"), "livebench-reasoning")

    def test_judgebench_code_bootstrap_prioritizes_code_bug_mutations(self) -> None:
        result = resolve_or_bootstrap_task_profile(
            [_judgebench_code_example()],
            explicit="auto",
            bootstrap_iterations=1,
        )

        self.assertTrue(result.bootstrap_used)
        self.assertGreaterEqual(len(result.strategy.mutation_ids), 3)
        self.assertIn("code_flip_condition_branch", result.strategy.mutation_ids[:3])
        self.assertIn("code_corrupt_input_parsing", result.strategy.mutation_ids[:4])
        self.assertEqual(result.diagnostics.get("source_family_hint"), "livecodebench")


if __name__ == "__main__":
    unittest.main()
