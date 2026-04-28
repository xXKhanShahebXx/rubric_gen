"""
Few-shot retrieval over a labeled JudgeBench train split.

Given the train_320 (or train_240, etc.) joined dataset, this module builds a lightweight TF-IDF
index over question texts and serves the top-K most similar labeled pairs for any blind-350
question. The retrieved demonstrations are formatted as compact in-context examples that the
discriminator / holistic-judge prompts can prepend to the system message.

Design constraints:

- Labels come exclusively from the train split. The blind-350 validation questions are NEVER
  used for retrieval, ensuring no leakage.
- Demonstrations are capped per-pair (default 1500 chars) and per-block (default 5000 chars
  total) so the prompt stays within model token limits.
- A single ``LabeledTrainFewShotIndex`` is built once at policy-build time and shared across
  all blind-350 examples. Retrieval is deterministic for a given (index, query) pair.
- The TF-IDF computation is implemented in pure Python (no scikit-learn dependency).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class LabeledTrainExample:
    pair_id: str
    source: str
    source_family: str
    question: str
    response_a: str
    response_b: str
    label: str

    def truncate(self, max_chars: int) -> "LabeledTrainExample":
        def _trunc(s: str) -> str:
            return s if len(s) <= max_chars else s[:max_chars] + "..."

        return LabeledTrainExample(
            pair_id=self.pair_id,
            source=self.source,
            source_family=self.source_family,
            question=_trunc(self.question),
            response_a=_trunc(self.response_a),
            response_b=_trunc(self.response_b),
            label=self.label,
        )


@dataclass
class LabeledTrainFewShotIndex:
    examples: List[LabeledTrainExample]
    document_freq: Dict[str, int]
    doc_token_lists: List[List[str]]
    doc_norms: List[float]
    doc_count: int

    @classmethod
    def from_examples(cls, examples: Sequence[LabeledTrainExample]) -> "LabeledTrainFewShotIndex":
        examples = list(examples)
        doc_token_lists: List[List[str]] = []
        document_freq: Dict[str, int] = {}
        for ex in examples:
            tokens = _tokenize(ex.question)
            doc_token_lists.append(tokens)
            for token in set(tokens):
                document_freq[token] = document_freq.get(token, 0) + 1
        doc_count = len(examples)
        doc_norms: List[float] = []
        for tokens in doc_token_lists:
            term_freq: Dict[str, int] = {}
            for tok in tokens:
                term_freq[tok] = term_freq.get(tok, 0) + 1
            sq_sum = 0.0
            for tok, tf in term_freq.items():
                idf = math.log((doc_count + 1) / (document_freq.get(tok, 0) + 1)) + 1.0
                sq_sum += (tf * idf) ** 2
            doc_norms.append(math.sqrt(sq_sum) if sq_sum > 0 else 0.0)
        return cls(
            examples=examples,
            document_freq=document_freq,
            doc_token_lists=doc_token_lists,
            doc_norms=doc_norms,
            doc_count=doc_count,
        )

    def _query_idf_vec(self, query_tokens: Sequence[str]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        if not query_tokens:
            return vec
        for tok in query_tokens:
            if tok in vec:
                vec[tok] += 1.0
            else:
                vec[tok] = 1.0
        out: Dict[str, float] = {}
        for tok, tf in vec.items():
            df = self.document_freq.get(tok, 0)
            idf = math.log((self.doc_count + 1) / (df + 1)) + 1.0
            out[tok] = tf * idf
        return out

    def top_k(
        self,
        query: str,
        *,
        top_k: int = 3,
        same_family: Optional[str] = None,
    ) -> List[Tuple[float, LabeledTrainExample]]:
        query_tokens = _tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []
        query_vec = self._query_idf_vec(query_tokens)
        query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if query_norm <= 0.0:
            return []
        scored: List[Tuple[float, LabeledTrainExample]] = []
        for idx, ex in enumerate(self.examples):
            if same_family is not None and ex.source_family != same_family:
                continue
            doc_norm = self.doc_norms[idx]
            if doc_norm <= 0.0:
                continue
            doc_tokens = self.doc_token_lists[idx]
            term_freq: Dict[str, int] = {}
            for tok in doc_tokens:
                term_freq[tok] = term_freq.get(tok, 0) + 1
            dot = 0.0
            for tok, qv in query_vec.items():
                tf = term_freq.get(tok, 0)
                if tf == 0:
                    continue
                df = self.document_freq.get(tok, 0)
                idf = math.log((self.doc_count + 1) / (df + 1)) + 1.0
                dot += qv * (tf * idf)
            sim = dot / (query_norm * doc_norm)
            if sim > 0:
                scored.append((sim, ex))
        scored.sort(key=lambda kv: (-kv[0], kv[1].pair_id))
        return scored[: int(top_k) if top_k else 0]


def load_labeled_train_few_shot_index(
    *,
    train_dataset_path: Path,
    official_dataset_path: Path,
) -> LabeledTrainFewShotIndex:
    """Build the index from a JudgeBench train split + the official labelled jsonl."""
    from rubric_gen.compiled.judgebench_eval import (
        join_local_subset_to_official_pairs,
        load_local_judgebench_subset,
        load_official_judgebench_pairs,
    )

    local_rows = load_local_judgebench_subset(train_dataset_path)
    official_pairs = load_official_judgebench_pairs(official_dataset_path)
    joined = join_local_subset_to_official_pairs(
        local_rows=local_rows,
        official_pairs=official_pairs,
        split_name=Path(train_dataset_path).stem,
    )
    examples = [
        LabeledTrainExample(
            pair_id=ex.pair_id,
            source=ex.source,
            source_family=ex.source_family,
            question=ex.question,
            response_a=ex.response_A,
            response_b=ex.response_B,
            label=ex.label,
        )
        for ex in joined
        if ex.label in {"A>B", "B>A"}
    ]
    return LabeledTrainFewShotIndex.from_examples(examples)


def format_demonstrations(
    selected: Sequence[Tuple[float, LabeledTrainExample]],
    *,
    max_chars_per_pair: int = 1500,
    max_chars_total: int = 5000,
) -> str:
    if not selected:
        return ""
    parts: List[str] = []
    total = 0
    for idx, (similarity, ex) in enumerate(selected):
        truncated = ex.truncate(max_chars_per_pair)
        block = (
            f"Demonstration {idx + 1} ({ex.source_family}, similarity={similarity:.3f}):\n"
            f"Prompt: {truncated.question}\n"
            f"Candidate X: {truncated.response_a}\n"
            f"Candidate Y: {truncated.response_b}\n"
            f"Gold preference: {ex.label.replace('A', 'X').replace('B', 'Y')}\n"
        )
        if total + len(block) > int(max_chars_total):
            break
        parts.append(block)
        total += len(block)
    if not parts:
        return ""
    header = (
        "You will see labeled demonstrations from a labeled training split below. "
        "Use them as calibration for the kinds of preference patterns the gold data follows; "
        "do not treat any demonstration as the answer.\n\n"
    )
    return header + "\n".join(parts)


def build_few_shot_block(
    *,
    index: Optional[LabeledTrainFewShotIndex],
    query: str,
    top_k: int = 3,
    same_family: Optional[str] = None,
    max_chars_per_pair: int = 1500,
    max_chars_total: int = 5000,
) -> str:
    if index is None or top_k <= 0 or not query:
        return ""
    selected = index.top_k(query, top_k=top_k, same_family=same_family)
    return format_demonstrations(
        selected,
        max_chars_per_pair=max_chars_per_pair,
        max_chars_total=max_chars_total,
    )
