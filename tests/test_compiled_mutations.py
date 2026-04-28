import unittest

from rubric_gen.compiled.mutations import build_contrast_candidates
from rubric_gen.types import ExampleRecord


def _example(reference_note: str) -> ExampleRecord:
    return ExampleRecord(
        example_id="ex_1",
        source="test",
        source_id="src_1",
        dataset_subset="unit",
        conversation="Patient encounter.",
        task_prompt="Write the note.",
        reference_note=reference_note,
    )


def _synthetic_text(example: ExampleRecord, mutation_id: str) -> str | None:
    candidates = build_contrast_candidates(example, mutation_ids=[mutation_id])
    for candidate in candidates:
        if candidate.metadata.get("mutation_id") == mutation_id:
            return candidate.text
    return None


class CompiledMutationTests(unittest.TestCase):
    def test_drop_medication_lines_keeps_nonmedication_monitoring(self) -> None:
        example = _example(
            "Medication: Continue metformin 500 mg twice daily.\n"
            "Plan: Continue to monitor blood sugars and call with readings in 2 weeks.\n"
        )

        mutated = _synthetic_text(example, "drop_medication_lines")

        self.assertIsNotNone(mutated)
        self.assertNotIn("metformin 500 mg", mutated.lower())
        self.assertIn("monitor blood sugars", mutated.lower())

    def test_drop_medication_lines_ignores_bare_continue_monitor_language(self) -> None:
        example = _example(
            "Plan: Continue to monitor blood sugars and follow with primary care as needed.\n"
        )

        mutated = _synthetic_text(example, "drop_medication_lines")

        self.assertIsNone(mutated)

    def test_drop_followup_lines_keeps_home_monitoring_instructions(self) -> None:
        example = _example(
            "Plan: Follow up in 2 weeks for reassessment.\n"
            "Instructions: Record daily blood pressure readings and upload them to the patient portal.\n"
        )

        mutated = _synthetic_text(example, "drop_followup_lines")

        self.assertIsNotNone(mutated)
        self.assertNotIn("follow up in 2 weeks", mutated.lower())
        self.assertIn("daily blood pressure readings", mutated.lower())


if __name__ == "__main__":
    unittest.main()
