"""
Embedding-based rubric index for the medical workflow.

The medical pipeline (``RubricPipeline`` with the ``judgebench-v47-medical`` preset)
discovers per-example RRD rubrics during training. To use those rubrics during
validation, we aggregate them into a single in-memory pool indexed by an OpenAI
embedding of each rubric's text. At validation time, the pipeline embeds the
validation prompt and retrieves the top-K nearest rubrics by cosine similarity.

Why embeddings instead of family-strict library retrieval (the JudgeBench path)?

- The medical_rl_prompts dataset has a single ``source`` value (``medical_o1_subset_b``),
  so JudgeBench-style family tags would collapse to a single bucket and add no
  precision. Embedding similarity gives much finer per-prompt selection.
- The Sonnet relevance filter (``rubric_gen.compiled.relevance_filter``) then
  catches the residual false positives where a same-cluster rubric is technically
  embedding-near but topically off (cancer-X retrieved for cancer-Y).

Index file format (JSON, ~37 MB at 6K rubrics x 1536-dim embeddings):

    {
      "schema": "compiled_medical_rubric_index_v1",
      "version": "v1",
      "embedding_model": "text-embedding-3-small",
      "embedding_dim": 1536,
      "build_metadata": {...},
      "items": [
        {
          "rubric_id": "med_lib_<sha1>",
          "text": "...",
          "embedding": [float, ...],
          "source_runs": ["medical_v47_train_shard0", ...],
          "source_examples": ["medical_o1_subset_b__0005000-..."],
          "occurrence_count": 1,
          "coverage_count": 7,
          "depth": 0,
          "severity_tier": "medium"
        }
      ]
    }

The index keeps the criterion as a :class:`RubricLibraryCriterion` at retrieval time
so the same downstream code paths (``to_canonical_row``, the relevance filter) work.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.rubric_library import RubricLibraryCriterion
from rubric_gen.types import ModelSpec


MEDICAL_RUBRIC_INDEX_SCHEMA = "compiled_medical_rubric_index_v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536  # text-embedding-3-small native dim

DEFAULT_EMBEDDING_MODEL_SPEC = ModelSpec(
    alias="medical_rubric_embedder",
    provider="openai",
    model=DEFAULT_EMBEDDING_MODEL,
    api_key_env="OPENAI_API_KEY",
)


@dataclass
class MedicalRubricIndexItem:
    """One rubric in the index, with the embedding that powers retrieval."""

    rubric_id: str
    text: str
    embedding: Tuple[float, ...]
    source_runs: Tuple[str, ...] = ()
    source_examples: Tuple[str, ...] = ()
    occurrence_count: int = 1
    coverage_count: int = 0
    depth: int = 0
    severity_tier: str = "medium"

    def to_criterion(self) -> RubricLibraryCriterion:
        """Render as a :class:`RubricLibraryCriterion` so downstream code paths work.

        We pack the rubric text into ``requirement`` (the field downstream renderers
        actually display) and tag the criterion with the ``generic`` family because
        the medical workflow has no family taxonomy. ``source_tag`` records that this
        criterion came from the medical embedding index so forensics can disambiguate
        it from JudgeBench library criteria.
        """
        return RubricLibraryCriterion(
            criterion_id=self.rubric_id,
            dimension="medical_qa",
            label="Medical RRD rubric",
            requirement=self.text,
            severity_tier=self.severity_tier or "medium",
            applicable_families=("generic",),
            source_tag="medical_rrd_index_v1",
            focus_kind="",
            verification_notes="",
            direction_evidence=int(self.occurrence_count),
        )


@dataclass
class MedicalRubricIndex:
    """In-memory embedding index over RRD rubrics distilled from training runs."""

    version: str
    embedding_model: str
    embedding_dim: int
    items: List[MedicalRubricIndexItem]
    build_metadata: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def item_count(self) -> int:
        return len(self.items)

    # ----- save/load ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "MedicalRubricIndex":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Medical rubric index at {path} is not a JSON object.")
        schema = str(payload.get("schema", "") or "")
        if schema != MEDICAL_RUBRIC_INDEX_SCHEMA:
            raise ValueError(
                f"Medical rubric index at {path} has schema={schema!r}, expected {MEDICAL_RUBRIC_INDEX_SCHEMA!r}."
            )
        items_raw = payload.get("items", [])
        if not isinstance(items_raw, list):
            raise ValueError(f"Medical rubric index at {path} is missing 'items' list.")
        items: List[MedicalRubricIndexItem] = []
        for row in items_raw:
            if not isinstance(row, Mapping):
                continue
            rubric_id = str(row.get("rubric_id", "") or "").strip()
            text = str(row.get("text", "") or "").strip()
            embedding_raw = row.get("embedding", []) or []
            if not rubric_id or not text or not isinstance(embedding_raw, (list, tuple)):
                continue
            try:
                embedding = tuple(float(v) for v in embedding_raw)
            except (TypeError, ValueError):
                continue
            items.append(
                MedicalRubricIndexItem(
                    rubric_id=rubric_id,
                    text=text,
                    embedding=embedding,
                    source_runs=tuple(str(s) for s in (row.get("source_runs") or [])),
                    source_examples=tuple(str(s) for s in (row.get("source_examples") or [])),
                    occurrence_count=int(row.get("occurrence_count", 1) or 1),
                    coverage_count=int(row.get("coverage_count", 0) or 0),
                    depth=int(row.get("depth", 0) or 0),
                    severity_tier=str(row.get("severity_tier", "medium") or "medium"),
                )
            )
        return cls(
            version=str(payload.get("version", "v1") or "v1"),
            embedding_model=str(payload.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
            embedding_dim=int(payload.get("embedding_dim", DEFAULT_EMBEDDING_DIM) or DEFAULT_EMBEDDING_DIM),
            items=items,
            build_metadata=dict(payload.get("build_metadata") or {}),
            path=path,
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": MEDICAL_RUBRIC_INDEX_SCHEMA,
            "version": self.version,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "item_count": self.item_count,
            "build_metadata": dict(self.build_metadata or {}),
            "items": [
                {
                    "rubric_id": item.rubric_id,
                    "text": item.text,
                    "embedding": list(item.embedding),
                    "source_runs": list(item.source_runs),
                    "source_examples": list(item.source_examples),
                    "occurrence_count": int(item.occurrence_count),
                    "coverage_count": int(item.coverage_count),
                    "depth": int(item.depth),
                    "severity_tier": item.severity_tier,
                }
                for item in self.items
            ],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        self.path = path
        return path

    # ----- retrieval ---------------------------------------------------------

    def retrieve_top_k(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 8,
    ) -> List[Tuple[MedicalRubricIndexItem, float]]:
        """Return the top-``k`` items by cosine similarity, paired with their score.

        Returns an empty list when the index is empty or ``k <= 0``. This method does
        not call any embedding model itself -- pass the query embedding in. The
        :class:`OpenAIEmbedder` companion class produces embeddings if needed.
        """
        if not self.items or k <= 0:
            return []
        query = list(query_embedding)
        if len(query) != self.embedding_dim:
            raise ValueError(
                f"Query embedding has dim={len(query)}; index expects dim={self.embedding_dim}."
            )
        query_norm = _l2_norm(query)
        if query_norm == 0.0:
            return []
        scored: List[Tuple[MedicalRubricIndexItem, float]] = []
        for item in self.items:
            item_norm = _l2_norm(item.embedding)
            if item_norm == 0.0:
                continue
            dot = sum(q * v for q, v in zip(query, item.embedding))
            cosine = dot / (query_norm * item_norm)
            scored.append((item, float(cosine)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].rubric_id))
        return scored[: int(k)]

    def summarize(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "item_count": self.item_count,
            "build_metadata": dict(self.build_metadata or {}),
        }


def _l2_norm(vec: Iterable[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """Tiny wrapper around the OpenAI embeddings endpoint with batching.

    Kept dependency-light: instantiate ``openai.OpenAI`` lazily on first use so the
    module import stays free for tests that mock embeddings.
    """

    def __init__(
        self,
        *,
        model_spec: Optional[ModelSpec] = None,
        batch_size: int = 128,
    ) -> None:
        self.model_spec = model_spec or DEFAULT_EMBEDDING_MODEL_SPEC
        self.batch_size = max(1, int(batch_size))
        self._client: Any = None

    def _client_or_create(self) -> Any:
        if self._client is None:
            api_key = os.getenv(self.model_spec.api_key_env, "").strip()
            if not api_key:
                raise ValueError(
                    f"Environment variable '{self.model_spec.api_key_env}' is required for the embedder."
                )
            from openai import OpenAI  # local import keeps unit tests lightweight

            self._client = OpenAI(api_key=api_key, base_url=self.model_spec.base_url)
        return self._client

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        return list(vectors[0]) if vectors else []

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        normalized = [str(t or "").strip() for t in texts]
        if not normalized:
            return []
        client = self._client_or_create()
        out: List[List[float]] = []
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            response = client.embeddings.create(
                model=self.model_spec.model,
                input=batch,
            )
            # Provider returns embeddings in input order; sort defensively.
            data = sorted(response.data, key=lambda d: d.index)
            for entry in data:
                out.append([float(v) for v in entry.embedding])
        return out


# ---------------------------------------------------------------------------
# Top-K convenience: combine embed + retrieve in one call
# ---------------------------------------------------------------------------


def retrieve_top_k_for_query(
    index: MedicalRubricIndex,
    query_text: str,
    *,
    embedder: OpenAIEmbedder,
    k: int = 8,
) -> List[Tuple[MedicalRubricIndexItem, float]]:
    """Embed a single query string and return top-``k`` matching items."""
    embedding = embedder.embed_text(query_text)
    if not embedding:
        return []
    return index.retrieve_top_k(embedding, k=k)
