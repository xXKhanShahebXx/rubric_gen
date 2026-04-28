import json
import tempfile
import unittest
from pathlib import Path

from rubric_gen.compiled.task_profiles import infer_task_profile_id_from_text, resolve_task_profile
from rubric_gen.dataio import load_examples, strongest_anchor_text
from rubric_gen.types import ExampleRecord


class TaskProfileTests(unittest.TestCase):
    def test_load_examples_populates_generic_artifact_fields(self) -> None:
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

            examples = load_examples(dataset)

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.task_profile_id, "rewrite_editing")
        self.assertEqual(example.task_family_id, "structured_transform")
        self.assertEqual(example.reference_artifact, "- Cough for 2 days\n- No fever\n- Return if symptoms worsen")
        self.assertEqual(example.augmented_artifact, "Cough for two days.")
        self.assertEqual(strongest_anchor_text(example), example.reference_artifact)

    def test_resolve_task_profile_preserves_note_baseline(self) -> None:
        example = ExampleRecord(
            example_id="note_1",
            source="medical_notes",
            source_id="1",
            dataset_subset="pilot",
            conversation="Doctor: Please draft a SOAP note from this transcript.",
            task_prompt="You are a healthcare scribe. Write a clinically faithful medical note.",
            reference_note="S: Abdominal pain\nO: Afebrile\nA: Suspected biliary colic\nP: Follow-up arranged",
            task_profile_id="note_documentation",
        )

        profile = resolve_task_profile(example)

        self.assertEqual(profile.task_profile_id, "note_documentation")
        self.assertEqual(profile.artifact_label, "note")

    def test_resolve_task_profile_infers_rewrite_from_prompt(self) -> None:
        example = ExampleRecord(
            example_id="rewrite_1",
            source="rewrite_suite",
            source_id="1",
            dataset_subset="pilot",
            conversation="Original text that needs cleanup.",
            task_prompt="Rewrite this paragraph in a more formal tone and preserve meaning.",
            reference_artifact="Formal rewritten paragraph.",
            augmented_artifact="Original text.",
            task_profile_id="",
        )

        profile = resolve_task_profile(example)

        self.assertEqual(profile.task_profile_id, "rewrite_editing")

    def test_infer_task_profile_uses_phrase_boundaries(self) -> None:
        inferred = infer_task_profile_id_from_text(
            "General patient engagement advice about school accommodations and recovery."
        )

        self.assertEqual(inferred, "general_instruction_following")


if __name__ == "__main__":
    unittest.main()
