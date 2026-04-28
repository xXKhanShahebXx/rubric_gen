import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rubric_gen.compiled.judgebench_strict_disjoint_builder import _OpenAIGenerator, build_strict_disjoint_split


class JudgeBenchStrictDisjointBuilderTests(unittest.TestCase):
    def test_openai_generator_retries_with_timeout(self) -> None:
        create_calls = []

        class _FakeCompletions:
            def __init__(self) -> None:
                self._attempt = 0

            def create(self, **kwargs):
                create_calls.append(dict(kwargs))
                self._attempt += 1
                if self._attempt == 1:
                    raise TimeoutError("simulated timeout")

                class _Message:
                    content = "print('ok')"

                class _Choice:
                    message = _Message()

                class _Response:
                    choices = [_Choice()]

                return _Response()

        class _FakeChat:
            def __init__(self) -> None:
                self.completions = _FakeCompletions()

        class _FakeClient:
            def __init__(self) -> None:
                self.chat = _FakeChat()

        generator = object.__new__(_OpenAIGenerator)
        generator._client = _FakeClient()
        generator.noncode_model = "gpt-4o-mini"
        generator.code_model = "gpt-4o"

        with patch("rubric_gen.compiled.judgebench_strict_disjoint_builder.time.sleep", return_value=None):
            code = generator.generate_code(question="Print ok.")

        self.assertEqual(code, "print('ok')")
        self.assertEqual(len(create_calls), 2)
        self.assertTrue(all(call["timeout"] == 120 for call in create_calls))
        self.assertTrue(all(call["model"] == "gpt-4o" for call in create_calls))

    def test_build_strict_disjoint_split_uses_explicit_120_350_names(self) -> None:
        validation_rows = [
            {"pair_id": f"val_{index}", "question": f"validation question {index}"}
            for index in range(350)
        ]
        official_rows = [{"pair_id": f"val_{index}"} for index in range(350)]
        train_local = [
            {
                "pair_id": "train_mmlu",
                "builder_family": "mmlu-pro",
                "question": "train question mmlu",
            },
            {
                "pair_id": "train_reasoning",
                "builder_family": "livebench-reasoning",
                "question": "train question reasoning",
            },
            {
                "pair_id": "train_math",
                "builder_family": "livebench-math",
                "question": "train question math",
            },
            {
                "pair_id": "train_code",
                "builder_family": "livecodebench",
                "question": "train question code",
            },
        ]
        train_official = [{"pair_id": row["pair_id"]} for row in train_local]
        selection_manifest = {"families": {"mmlu-pro": [{"pair_id": "train_mmlu"}]}}

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "split_bundle"
            with patch(
                "rubric_gen.compiled.judgebench_strict_disjoint_builder._load_current_validation_context",
                return_value=(validation_rows, official_rows, set(), set()),
            ), patch(
                "rubric_gen.compiled.judgebench_strict_disjoint_builder._OpenAIGenerator",
                return_value=object(),
            ), patch(
                "rubric_gen.compiled.judgebench_strict_disjoint_builder._sample_train_rows",
                return_value=(train_local, train_official, selection_manifest),
            ):
                artifacts = build_strict_disjoint_split(
                    output_dir=output_dir,
                    seed=7,
                    per_family=30,
                    noncode_model="gpt-4o-mini",
                    code_model="gpt-4o",
                    max_code_attempts=3,
                )

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(artifacts.train_dataset_path.name, "train_120_strict.json")
        self.assertEqual(artifacts.validation_dataset_path.name, "validation_350.json")
        self.assertEqual(artifacts.official_dataset_path.name, "official_train_120_validation_350.jsonl")
        self.assertEqual(manifest["train_split_name"], "train_120")
        self.assertEqual(manifest["validation_split_name"], "validation_350")
        self.assertEqual(manifest["train_dataset_filename"], "train_120_strict.json")
        self.assertEqual(manifest["validation_dataset_filename"], "validation_350.json")
        self.assertEqual(manifest["official_dataset_filename"], "official_train_120_validation_350.jsonl")
        self.assertEqual(manifest["validation_pair_count"], 350)
        self.assertEqual(manifest["family_pair_counts"]["mmlu-pro"], 1)
        self.assertEqual(manifest["family_pair_counts"]["livecodebench"], 1)


if __name__ == "__main__":
    unittest.main()
