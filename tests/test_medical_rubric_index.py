from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Sequence

from rubric_gen.compiled.medical_rubric_index import (
    DEFAULT_EMBEDDING_DIM,
    MEDICAL_RUBRIC_INDEX_SCHEMA,
    MedicalRubricIndex,
    MedicalRubricIndexItem,
    OpenAIEmbedder,
    retrieve_top_k_for_query,
)


def _normalized(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def _make_item(rubric_id: str, text: str, embedding: Sequence[float]) -> MedicalRubricIndexItem:
    return MedicalRubricIndexItem(
        rubric_id=rubric_id,
        text=text,
        embedding=tuple(embedding),
        source_runs=("medical_v47_train_shard0",),
        source_examples=("medical_o1_subset_b__0005000-test",),
        occurrence_count=1,
        coverage_count=3,
        depth=0,
        severity_tier="medium",
    )


def _make_index(items: List[MedicalRubricIndexItem], dim: int) -> MedicalRubricIndex:
    return MedicalRubricIndex(
        version="v1",
        embedding_model="test-embed",
        embedding_dim=dim,
        items=items,
        build_metadata={"unit_test": True},
    )


class MedicalRubricIndexRetrievalTests(unittest.TestCase):
    def test_top_k_returns_items_in_descending_cosine_order(self) -> None:
        # 4-dim toy embeddings; the query exactly matches item_a.
        items = [
            _make_item("med_a", "criterion A", [1.0, 0.0, 0.0, 0.0]),
            _make_item("med_b", "criterion B", [0.5, 0.5, 0.0, 0.0]),
            _make_item("med_c", "criterion C", [0.0, 1.0, 0.0, 0.0]),
            _make_item("med_d", "criterion D", [0.0, 0.0, 0.0, 1.0]),  # orthogonal -> last
        ]
        index = _make_index(items, dim=4)

        scored = index.retrieve_top_k([1.0, 0.0, 0.0, 0.0], k=4)

        # med_c and med_d are both orthogonal to the query (cosine 0.0); the order of
        # the orthogonal pair is settled deterministically by rubric_id ascending.
        self.assertEqual([item.rubric_id for item, _ in scored], ["med_a", "med_b", "med_c", "med_d"])
        self.assertGreater(scored[0][1], scored[1][1])
        self.assertGreater(scored[1][1], scored[2][1])
        self.assertGreaterEqual(scored[2][1], scored[3][1])

    def test_top_k_respects_k_cap(self) -> None:
        items = [_make_item(f"med_{i}", f"text {i}", [float(i == j) for j in range(4)]) for i in range(4)]
        index = _make_index(items, dim=4)
        scored = index.retrieve_top_k([1.0, 0.0, 0.0, 0.0], k=2)
        self.assertEqual(len(scored), 2)

    def test_empty_index_returns_empty_list(self) -> None:
        index = _make_index([], dim=4)
        self.assertEqual(index.retrieve_top_k([1.0, 0.0, 0.0, 0.0], k=5), [])

    def test_zero_k_returns_empty_list(self) -> None:
        items = [_make_item("med_a", "criterion A", [1.0, 0.0, 0.0, 0.0])]
        index = _make_index(items, dim=4)
        self.assertEqual(index.retrieve_top_k([1.0, 0.0, 0.0, 0.0], k=0), [])

    def test_zero_query_norm_returns_empty(self) -> None:
        items = [_make_item("med_a", "criterion A", [1.0, 0.0, 0.0, 0.0])]
        index = _make_index(items, dim=4)
        self.assertEqual(index.retrieve_top_k([0.0, 0.0, 0.0, 0.0], k=3), [])

    def test_dim_mismatch_raises(self) -> None:
        items = [_make_item("med_a", "criterion A", [1.0, 0.0, 0.0, 0.0])]
        index = _make_index(items, dim=4)
        with self.assertRaises(ValueError):
            index.retrieve_top_k([1.0, 0.0, 0.0], k=1)

    def test_zero_embedding_items_are_skipped(self) -> None:
        items = [
            _make_item("med_a", "criterion A", [1.0, 0.0, 0.0, 0.0]),
            _make_item("med_zero", "criterion zero", [0.0, 0.0, 0.0, 0.0]),
        ]
        index = _make_index(items, dim=4)
        scored = index.retrieve_top_k([1.0, 0.0, 0.0, 0.0], k=2)
        self.assertEqual([item.rubric_id for item, _ in scored], ["med_a"])


class MedicalRubricIndexRoundtripTests(unittest.TestCase):
    def test_save_and_load_preserves_items(self) -> None:
        items = [
            _make_item("med_a", "criterion A", [1.0, 0.0, 0.5, 0.5]),
            _make_item("med_b", "criterion B", [0.5, 0.5, 0.0, 0.0]),
        ]
        index = _make_index(items, dim=4)
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "index.json"
            saved_path = index.save(out_path)
            self.assertEqual(saved_path, out_path)
            loaded = MedicalRubricIndex.load(out_path)
        self.assertEqual(loaded.embedding_model, "test-embed")
        self.assertEqual(loaded.embedding_dim, 4)
        self.assertEqual(loaded.item_count, 2)
        self.assertEqual([item.rubric_id for item in loaded.items], ["med_a", "med_b"])
        # Check embedding fidelity (within float tolerance after JSON round-trip).
        for original, loaded_item in zip(items, loaded.items):
            self.assertEqual(len(original.embedding), len(loaded_item.embedding))
            for o, l in zip(original.embedding, loaded_item.embedding):
                self.assertAlmostEqual(o, l)
        self.assertEqual(loaded.build_metadata.get("unit_test"), True)

    def test_load_rejects_bad_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema": "wrong", "items": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                MedicalRubricIndex.load(path)

    def test_load_rejects_non_object(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError):
                MedicalRubricIndex.load(path)

    def test_save_writes_documented_schema(self) -> None:
        index = _make_index([_make_item("med_a", "x", [1.0])], dim=1)
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "index.json"
            index.save(out_path)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], MEDICAL_RUBRIC_INDEX_SCHEMA)
        self.assertEqual(payload["item_count"], 1)
        self.assertIn("items", payload)


class MedicalRubricIndexCriterionConversionTests(unittest.TestCase):
    def test_to_criterion_packs_text_into_requirement(self) -> None:
        item = _make_item("med_a", "Always answer in plain English.", [0.5, 0.5])
        criterion = item.to_criterion()
        self.assertEqual(criterion.criterion_id, "med_a")
        self.assertEqual(criterion.requirement, "Always answer in plain English.")
        self.assertEqual(criterion.applicable_families, ("generic",))
        self.assertEqual(criterion.source_tag, "medical_rrd_index_v1")
        self.assertEqual(criterion.dimension, "medical_qa")
        self.assertEqual(criterion.severity_tier, "medium")
        # Direction evidence is borrowed from occurrence_count for downstream sorting.
        self.assertEqual(criterion.direction_evidence, 1)


class _StubEmbedder(OpenAIEmbedder):
    """In-memory embedder that returns deterministic per-text vectors without API."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [list(self._mapping[t]) for t in texts]

    def embed_text(self, text):
        return list(self._mapping[text])


class RetrieveTopKForQueryTests(unittest.TestCase):
    def test_combined_helper_embeds_query_then_retrieves(self) -> None:
        items = [
            _make_item("med_a", "criterion A", [1.0, 0.0, 0.0, 0.0]),
            _make_item("med_b", "criterion B", [0.0, 1.0, 0.0, 0.0]),
        ]
        index = _make_index(items, dim=4)
        embedder = _StubEmbedder({"query about A": [1.0, 0.0, 0.0, 0.0]})

        scored = retrieve_top_k_for_query(
            index,
            "query about A",
            embedder=embedder,
            k=1,
        )

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0][0].rubric_id, "med_a")
        self.assertEqual(embedder.calls, 0)  # embed_text bypasses embed_texts in stub


if __name__ == "__main__":
    unittest.main()
