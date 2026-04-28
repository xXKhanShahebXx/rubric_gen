from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rubric_gen.compiled.reward_bench_2_loader import (
    expand_item_to_pairs,
    expand_items_to_pairs,
    subset_to_family,
    write_joined_dataset,
    write_official_jsonl,
)


def _factuality_item() -> dict:
    return {
        "id": "0",
        "subset": "Factuality",
        "prompt": "Do you know how to fix the deadline_exceeded error?",
        "chosen": ["Long correct answer about deadline_exceeded"],
        "rejected": [
            "Wrong answer 1",
            "Wrong answer 2",
            "Wrong answer 3",
        ],
        "num_correct": 1,
        "num_incorrect": 3,
        "total_completions": 4,
        "models": ["modelC", "modelR1", "modelR2", "modelR3"],
        "additional_metadata": {"correct": "ONE", "method": "natural"},
    }


def _math_item() -> dict:
    item = _factuality_item()
    item["id"] = "501"
    item["subset"] = "Math"
    return item


def _ties_item() -> dict:
    return {
        "id": "T1",
        "subset": "Ties",
        "prompt": "Name a color of the rainbow.",
        "chosen": ["red", "orange", "yellow", "green"],
        "rejected": ["pink", "brown", "black", "white", "magenta", "cyan"],
        "num_correct": 4,
        "num_incorrect": 6,
        "total_completions": 10,
        "models": ["m"] * 10,
        "additional_metadata": {},
    }


class SubsetMappingTests(unittest.TestCase):
    def test_known_mappings(self) -> None:
        self.assertEqual(subset_to_family("Factuality"), "mmlu-pro")
        self.assertEqual(subset_to_family("Precise IF"), "livebench-reasoning")
        self.assertEqual(subset_to_family("Math"), "livebench-math")
        self.assertEqual(subset_to_family("Safety"), "mmlu-pro")
        self.assertEqual(subset_to_family("Focus"), "mmlu-pro")
        self.assertEqual(subset_to_family("Ties"), "mmlu-pro")

    def test_unknown_falls_back(self) -> None:
        self.assertEqual(subset_to_family("Unknown"), "mmlu-pro")


class ExpandItemTests(unittest.TestCase):
    def test_factuality_item_expands_to_three_rows(self) -> None:
        rows = expand_item_to_pairs(_factuality_item())
        self.assertEqual(len(rows), 3)
        for idx, row in enumerate(rows):
            self.assertEqual(row.subset, "Factuality")
            self.assertEqual(row.rejected_index, idx)
            self.assertEqual(row.joined_example.label, "A>B")
            self.assertEqual(row.joined_example.source_family, "mmlu-pro")
            # response_A always carries the chosen text
            self.assertIn("correct", row.joined_example.response_A.lower())
            self.assertIn("Wrong", row.joined_example.response_B)

    def test_math_item_routes_to_livebench_math(self) -> None:
        rows = expand_item_to_pairs(_math_item())
        self.assertEqual(rows[0].joined_example.source_family, "livebench-math")

    def test_ties_item_expands_to_full_cross_product(self) -> None:
        rows = expand_item_to_pairs(_ties_item())
        # 4 chosen × 6 rejected
        self.assertEqual(len(rows), 24)
        chosen_indices = {row.joined_example.metadata["reward_bench_2"]["chosen_index"] for row in rows}
        rejected_indices = {row.joined_example.metadata["reward_bench_2"]["rejected_index"] for row in rows}
        self.assertEqual(chosen_indices, {0, 1, 2, 3})
        self.assertEqual(rejected_indices, {0, 1, 2, 3, 4, 5})

    def test_max_caps_apply_to_ties(self) -> None:
        rows = expand_item_to_pairs(_ties_item(), max_chosen=2, max_rejected=3)
        self.assertEqual(len(rows), 6)

    def test_metadata_round_trips(self) -> None:
        rows = expand_item_to_pairs(_factuality_item())
        m = rows[0].joined_example.metadata["reward_bench_2"]
        self.assertEqual(m["item_id"], "0")
        self.assertEqual(m["subset"], "Factuality")
        self.assertEqual(m["num_correct"], 1)
        self.assertEqual(m["chosen_index"], 0)
        self.assertEqual(m["rejected_index"], 0)


class ExpandItemsTests(unittest.TestCase):
    def test_subset_filter_excludes_unwanted(self) -> None:
        items = [_factuality_item(), _math_item()]
        rows = expand_items_to_pairs(items, subsets=["Math"])
        self.assertEqual({r.subset for r in rows}, {"Math"})

    def test_items_per_subset_caps_count(self) -> None:
        items = []
        for i in range(5):
            it = _factuality_item()
            it["id"] = str(i)
            items.append(it)
        rows = expand_items_to_pairs(items, items_per_subset=2)
        item_ids = {r.rb2_item_id for r in rows}
        self.assertEqual(len(item_ids), 2)
        self.assertEqual(len(rows), 6)


class PersistTests(unittest.TestCase):
    def test_writes_validation_and_official(self) -> None:
        items = [_factuality_item(), _math_item()]
        rows = expand_items_to_pairs(items)
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            v = tmp_path / "validation.json"
            o = tmp_path / "official.jsonl"
            write_joined_dataset(rows, v)
            write_official_jsonl(rows, o)
            v_payload = json.loads(v.read_text(encoding="utf-8"))
            self.assertEqual(len(v_payload), 6)
            with o.open(encoding="utf-8") as fh:
                lines = [json.loads(ln) for ln in fh]
            self.assertEqual(len(lines), 6)
            self.assertTrue(all(line["label"] == "A>B" for line in lines))


if __name__ == "__main__":
    unittest.main()
