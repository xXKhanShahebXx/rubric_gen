"""
Adapter that turns ``allenai/reward-bench-2`` items into ``JudgeBenchJoinedExample`` rows
that the existing rubric-judge pipeline can score without modification.

RewardBench 2 structure:

* Non-Ties subsets (Factuality / Precise IF / Math / Safety / Focus): each item is a
  best-of-4 — one ``chosen`` completion vs three ``rejected`` completions. Scoring success
  = the reward model rates ``chosen`` higher than EVERY rejected. We turn each item into 3
  pairwise rows ``(prompt, chosen, rejected_i)`` and aggregate per-item at the metrics
  layer.
* Ties subset: item has multiple correct + multiple rejected. Different metric. Handled
  in :mod:`reward_bench_2_metrics` (see ``compute_ties_score``); the loader still emits
  pairwise rows so the pipeline runs uniformly.

Subset → family routing:

* ``Factuality``    → ``mmlu-pro``           (factual recall, hallucination detection)
* ``Precise IF``    → ``livebench-reasoning`` (verifier-checkable instruction following)
* ``Math``          → ``livebench-math``     (direct — math solver applies)
* ``Safety``        → ``mmlu-pro``           (open-text profile; no specific fit)
* ``Focus``         → ``mmlu-pro``           (on-topic answers to general queries)
* ``Ties``          → ``mmlu-pro``           (multi-correct; special metric)

The mapping reuses existing routes; we never had to register new task profiles.

Each emitted row keeps the original RB2 metadata under
``metadata["reward_bench_2"]`` so downstream metric code can reconstruct best-of-4 grouping.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from rubric_gen.compiled.judgebench_eval import JudgeBenchJoinedExample


_SUBSET_TO_FAMILY: Dict[str, str] = {
    "Factuality": "mmlu-pro",
    "Precise IF": "livebench-reasoning",
    "Math": "livebench-math",
    "Safety": "mmlu-pro",
    "Focus": "mmlu-pro",
    "Ties": "mmlu-pro",
}


def subset_to_family(subset: str) -> str:
    """Return the JudgeBench source_family that the existing pipeline routes for this subset."""
    return _SUBSET_TO_FAMILY.get(subset, "mmlu-pro")


@dataclass(frozen=True)
class RewardBench2PairRow:
    """A single pairwise row emitted by the loader. Mirrors JudgeBenchJoinedExample fields."""

    pair_id: str
    rb2_item_id: str
    subset: str
    rejected_index: int
    joined_example: JudgeBenchJoinedExample


def _normalize_completion(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _build_metadata(
    *,
    rb2_item: Mapping[str, Any],
    rejected_index: int,
    chosen_index: int,
    chosen_text: str,
    rejected_text: str,
) -> Dict[str, Any]:
    return {
        "reward_bench_2": {
            "item_id": str(rb2_item.get("id", "")),
            "subset": rb2_item.get("subset", ""),
            "rejected_index": int(rejected_index),
            "chosen_index": int(chosen_index),
            "num_correct": int(rb2_item.get("num_correct", 0) or 0),
            "num_incorrect": int(rb2_item.get("num_incorrect", 0) or 0),
            "total_completions": int(rb2_item.get("total_completions", 0) or 0),
            "models": list(rb2_item.get("models", []) or []),
            "additional_metadata": rb2_item.get("additional_metadata", {}) or {},
        }
    }


def _slug_subset(subset: str) -> str:
    """Lowercase + slug-safe subset name (so it composes into a pair_id)."""
    return (
        (subset or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        or "unknown"
    )


def _build_pair_id(subset: str, rb2_id: str, chosen_index: int, rejected_index: int) -> str:
    """
    Build a globally-unique pair id.

    RewardBench 2 IDs are *only* unique within a subset (Factuality '0', Precise IF '0',
    and Math '10' coexist). We embed the subset slug so multi-subset eval runs don't
    collide on storage paths or metric grouping keys.
    """
    safe_id = (rb2_id or "").replace("/", "_").replace(":", "-")
    return f"rb2-{_slug_subset(subset)}-{safe_id}-c{chosen_index}-r{rejected_index}"


def expand_item_to_pairs(
    rb2_item: Mapping[str, Any],
    *,
    split_name: str = "reward_bench_2",
    max_rejected: Optional[int] = None,
    max_chosen: Optional[int] = None,
) -> List[RewardBench2PairRow]:
    """
    Expand a single RewardBench 2 item into pairwise rows.

    For non-Ties items (1 chosen, 3 rejected), this emits 3 rows. For Ties items
    (multiple chosen, multiple rejected), it emits ``len(chosen) * len(rejected)`` rows
    unless ``max_chosen`` / ``max_rejected`` are set.

    Each row has the chosen completion in position ``response_A`` and the rejected
    completion in position ``response_B``, so the gold ``label`` is always ``"A>B"``.
    """
    rb2_id = str(rb2_item.get("id", "")).strip()
    if not rb2_id:
        return []
    subset = str(rb2_item.get("subset", "")).strip()
    family = subset_to_family(subset)
    chosen_list: Sequence[Any] = list(rb2_item.get("chosen", []) or [])
    rejected_list: Sequence[Any] = list(rb2_item.get("rejected", []) or [])
    if max_chosen is not None and max_chosen >= 0:
        chosen_list = chosen_list[: int(max_chosen)]
    if max_rejected is not None and max_rejected >= 0:
        rejected_list = rejected_list[: int(max_rejected)]
    prompt = str(rb2_item.get("prompt", "")).strip()

    rows: List[RewardBench2PairRow] = []
    models_list = list(rb2_item.get("models", []) or [])
    for c_idx, chosen_raw in enumerate(chosen_list):
        chosen_text = _normalize_completion(chosen_raw)
        if not chosen_text:
            continue
        for r_idx, rejected_raw in enumerate(rejected_list):
            rejected_text = _normalize_completion(rejected_raw)
            if not rejected_text:
                continue
            pair_id = _build_pair_id(subset, rb2_id, c_idx, r_idx)
            chosen_model = (
                models_list[c_idx] if c_idx < len(models_list) else ""
            )
            rejected_model_offset = len(chosen_list) + r_idx
            rejected_model = (
                models_list[rejected_model_offset]
                if rejected_model_offset < len(models_list)
                else ""
            )
            example = JudgeBenchJoinedExample(
                split_name=split_name,
                pair_id=pair_id,
                # ``source`` must start with a recognised family prefix because the
                # eval pipeline re-derives ``source_family`` from it via
                # :func:`judgebench_source_family`. We therefore lead with the family
                # and keep the RB2 lineage in ``metadata.reward_bench_2``.
                source=f"{family}:reward_bench_2:{subset}",
                source_family=family,
                question=prompt,
                reference_answer="",
                response_model=str(chosen_model or "rb2-chosen"),
                response_A=chosen_text,
                response_B=rejected_text,
                label="A>B",
                original_id=rb2_id,
                candidate_models=[
                    str(chosen_model or ""),
                    str(rejected_model or ""),
                ],
                verifier_model="",
                metadata=_build_metadata(
                    rb2_item=rb2_item,
                    rejected_index=r_idx,
                    chosen_index=c_idx,
                    chosen_text=chosen_text,
                    rejected_text=rejected_text,
                ),
            )
            rows.append(
                RewardBench2PairRow(
                    pair_id=pair_id,
                    rb2_item_id=rb2_id,
                    subset=subset,
                    rejected_index=r_idx,
                    joined_example=example,
                )
            )
    return rows


def expand_items_to_pairs(
    items: Iterable[Mapping[str, Any]],
    *,
    split_name: str = "reward_bench_2",
    max_rejected: Optional[int] = None,
    max_chosen: Optional[int] = None,
    subsets: Optional[Sequence[str]] = None,
    items_per_subset: Optional[int] = None,
) -> List[RewardBench2PairRow]:
    """
    Expand many items into pairwise rows. Optional filters:

    - ``subsets``: keep only items whose subset matches.
    - ``items_per_subset``: cap each subset to N items (after filtering).
    - ``max_chosen`` / ``max_rejected``: cap completions per item (useful for Ties).
    """
    subset_filter = set(subsets) if subsets else None
    counters: Dict[str, int] = {}
    rows: List[RewardBench2PairRow] = []
    for item in items:
        subset = str(item.get("subset", "")).strip()
        if subset_filter is not None and subset not in subset_filter:
            continue
        if items_per_subset is not None:
            seen = counters.get(subset, 0)
            if seen >= items_per_subset:
                continue
            counters[subset] = seen + 1
        rows.extend(
            expand_item_to_pairs(
                item,
                split_name=split_name,
                max_rejected=max_rejected,
                max_chosen=max_chosen,
            )
        )
    return rows


def load_reward_bench_2(
    *,
    cache_dir: Optional[Path] = None,
    split: str = "test",
):
    """
    Lazy import of ``datasets`` so the rest of the loader can be unit-tested without it.
    Returns a HuggingFace ``Dataset``. Caller is responsible for filtering / iteration.
    """
    from datasets import load_dataset  # local import on purpose

    return load_dataset(
        "allenai/reward-bench-2",
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def write_joined_dataset(
    rows: Sequence[RewardBench2PairRow],
    out_path: Path,
) -> None:
    """
    Persist the expanded pairwise rows in the same JSON shape that
    ``run_judgebench_final_evaluation`` expects from
    ``--validation-dataset path/to/validation.json``.
    """
    payload = [asdict(row.joined_example) for row in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_official_jsonl(
    rows: Sequence[RewardBench2PairRow],
    out_path: Path,
) -> None:
    """
    Persist a JudgeBench-style 'official' JSONL with one record per pair_id.

    The eval pipeline's :func:`join_local_subset_to_official_pairs` joins the local
    validation file against this official JSONL **and overrides** ``response_A`` /
    ``response_B`` / ``question`` / ``response_model`` from the official record. We
    therefore write the FULL pair payload here (not just metadata) so the joined examples
    carry the correct candidate texts. The ``reward_bench_2_*`` fields are kept for
    leakage / lineage audits.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            example = row.joined_example
            record = {
                "pair_id": example.pair_id,
                "original_id": example.original_id,
                "source": example.source,
                "source_family": example.source_family,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "response_model": example.response_model,
                "response_A": example.response_A,
                "response_B": example.response_B,
                "label": example.label,
                "reward_bench_2_item_id": row.rb2_item_id,
                "reward_bench_2_subset": row.subset,
                "reward_bench_2_rejected_index": row.rejected_index,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
