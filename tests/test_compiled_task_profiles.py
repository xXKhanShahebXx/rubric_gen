import json
import tempfile
import unittest
from pathlib import Path

from rubric_gen.compiled.compiler import (
    build_starter_ontology,
    build_task_family_spec,
    build_task_ontology,
    compile_task_case_rubric,
    infer_task_family,
)
from rubric_gen.compiled.discovered_augmentation import build_augmented_ontology
from rubric_gen.compiled.pilot_runner import run_pilot
from rubric_gen.types import ExampleRecord


def _rewrite_example() -> ExampleRecord:
    return ExampleRecord(
        example_id="rewrite__1",
        source="rewrite_suite",
        source_id="1",
        dataset_subset="pilot",
        conversation="Original text: Patient has cough for two days. No fever. Return if symptoms worsen.",
        task_prompt="Rewrite the source into bullet points and preserve meaning.",
        reference_artifact="- Cough for 2 days\n- No fever\n- Return if symptoms worsen",
        augmented_artifact="Cough for two days.",
        task_profile_id="rewrite_editing",
        task_family_id="structured_transform",
        artifact_kind="response",
    )


class CompiledTaskProfileTests(unittest.TestCase):
    def test_build_task_ontology_keeps_note_profile_baseline(self) -> None:
        task_ontology = build_task_ontology("note_documentation")
        baseline = build_starter_ontology()

        self.assertEqual(task_ontology.ontology_id, baseline.ontology_id)
        self.assertEqual(task_ontology.version, baseline.version)
        self.assertEqual(len(task_ontology.criterion_templates), len(baseline.criterion_templates))

    def test_compile_task_case_rubric_supports_rewrite_profile(self) -> None:
        example = _rewrite_example()
        ontology = build_task_ontology("rewrite_editing")
        family_id = infer_task_family(example, task_profile_id="rewrite_editing")
        family = build_task_family_spec(
            family_id,
            ontology,
            task_profile_id="rewrite_editing",
        )

        rubric = compile_task_case_rubric(
            example,
            ontology,
            family,
            task_profile_id="rewrite_editing",
        )

        self.assertEqual(rubric.task_profile_id, "rewrite_editing")
        self.assertEqual(rubric.task_family_id, "structured_transform")
        self.assertEqual(rubric.artifact_label, "rewrite")
        self.assertTrue(any(c.eval_kind == "generic_unsupported_assertions" for c in rubric.hard_gates))
        self.assertTrue(
            any(c.eval_kind in {"artifact_marker_presence", "anchor_terms_presence"} for c in rubric.soft_checks)
        )

    def test_build_augmented_ontology_scopes_generic_templates_to_task_family(self) -> None:
        merged = {
            "canonical_proposals": [
                {
                    "merge_key": "instruction_adherence||high||preserve bullet format||preserve bullet format",
                    "dimension": "format_compliance",
                    "label": "Preserve bullet format",
                    "requirement": "The rewrite should preserve the requested bullet-point format.",
                    "severity_tier": "high",
                    "count": 1,
                    "example_ids": ["rewrite__1"],
                    "pair_ids": ["rewrite__1__reference_artifact__vs__rewrite__1__mut__remove_format_markers"],
                }
            ]
        }

        ontology, selected = build_augmented_ontology(
            merged_proposals=merged,
            design_examples=[_rewrite_example()],
            support_threshold=1,
            task_profile_id="rewrite_editing",
        )

        discovered = [template for template in ontology.criterion_templates if template.provisional_discovered]
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].task_family_scope, ["structured_transform"])
        self.assertEqual(selected[0]["task_family_scope"], ["structured_transform"])

    def test_run_pilot_supports_non_note_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "rewrite_dataset.json"
            dataset.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "source": "rewrite_suite",
                                "source_id": "row_1",
                                "instruction": "Rewrite the text into bullet points and preserve meaning.",
                                "input": "Patient has cough for two days. No fever. Return if symptoms worsen.",
                                "reference_output": "- Cough for 2 days\n- No fever\n- Return if symptoms worsen",
                                "candidate_output": "Cough for two days.",
                                "task_profile_id": "rewrite_editing",
                                "task_family_id": "structured_transform",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            run_dir = run_pilot(
                dataset_path=dataset,
                run_name="rewrite_smoke",
                start=0,
                limit=0,
                design_n=0,
                validation_n=0,
                pilot_n=1,
                source_filter=None,
                out_root=Path(tmp),
                write_csv=False,
                judge_mode="heuristic",
                task_profile="rewrite_editing",
            )

            summary = json.loads((run_dir / "summaries" / "run_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["params"]["task_profile"], "rewrite_editing")
        self.assertEqual(summary["stats"]["task_profile_counts"], {"rewrite_editing": 1})
        self.assertEqual(summary["stats"]["task_family_counts"], {"structured_transform": 1})


if __name__ == "__main__":
    unittest.main()
