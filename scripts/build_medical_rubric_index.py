"""Build the medical RRD-rubric embedding index from one or more training run dirs.

Reads ``examples/<id>.json`` files under each provided run directory, extracts the
discovered RRD rubrics from ``methods.rrd_uniform.rubrics``, dedupes by normalized
lowercased text, embeds via the OpenAI text-embedding-3-small model in batches, and
writes a single JSON index that the medical validation pipeline can load.

Cost: ~$0.30 one-time for ~6000 unique rubric texts (5750 expected from a 3000-row
training set after RRD's filtering, post-dedup ~6K assuming low duplication).

Usage:

  python scripts/build_medical_rubric_index.py \\
    artifacts/medical_rl/runs/medical_v47_train_shard0 \\
    artifacts/medical_rl/runs/medical_v47_train_shard1 \\
    artifacts/medical_rl/runs/medical_v47_train_shard2 \\
    --out artifacts/medical_rl/rubric_index/medical_v47_index.json

Pass ``--dry-run`` to skip the embedding API calls; only the dedupe/aggregate stats
are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from rubric_gen.compiled.medical_rubric_index import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_SPEC,
    MedicalRubricIndex,
    MedicalRubricIndexItem,
    OpenAIEmbedder,
)
from rubric_gen.types import ModelSpec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more training run directories (e.g. artifacts/medical_rl/runs/medical_v47_train_shard0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output index path (e.g. artifacts/medical_rl/rubric_index/medical_v47_index.json).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embeddings model name (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Expected embedding dimension (default: {DEFAULT_EMBEDDING_DIM}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size (default: 128).",
    )
    parser.add_argument(
        "--include-decomposed",
        action="store_true",
        help=(
            "Include rubrics whose source_stage is 'decomposition'. By default only "
            "'initial' RRD rubrics are indexed because decomposed children are often "
            "rejected by RRD's gain check anyway."
        ),
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rubrics where accepted=False. Off by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Aggregate and dedupe but skip the embedding API calls. Writes a stub "
            "index with empty embeddings; useful for sanity-checking the dedupe stats."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only embed this many unique texts. Useful for cost-bounded smoke runs.",
    )
    parser.add_argument(
        "--method-key",
        type=str,
        default="rrd_uniform",
        help="Which methods.<key>.rubrics array to harvest (default: rrd_uniform).",
    )
    return parser


def _normalize_for_dedupe(text: str) -> str:
    """Same normalization the candidate / rubric pipeline uses for dedupe purposes."""
    return " ".join((text or "").strip().lower().split())


def _rubric_id_for_text(text: str) -> str:
    digest = hashlib.sha1(_normalize_for_dedupe(text).encode("utf-8")).hexdigest()
    return f"med_lib_{digest[:16]}"


def _iter_rubrics_from_artifact(
    artifact: Mapping[str, Any],
    *,
    method_key: str,
    include_decomposed: bool,
    include_rejected: bool,
) -> Iterable[Mapping[str, Any]]:
    methods = artifact.get("methods") or {}
    method_payload = methods.get(method_key) or {}
    rubrics = method_payload.get("rubrics") or []
    if not isinstance(rubrics, list):
        return []
    out: List[Mapping[str, Any]] = []
    for entry in rubrics:
        if not isinstance(entry, Mapping):
            continue
        if not include_rejected and not bool(entry.get("accepted", True)):
            continue
        stage = str(entry.get("source_stage", "") or "").strip().lower()
        if not include_decomposed and stage and stage != "initial":
            continue
        text = str(entry.get("text", "") or "").strip()
        if not text:
            continue
        out.append(entry)
    return out


def _aggregate_run(
    run_dir: Path,
    *,
    method_key: str,
    include_decomposed: bool,
    include_rejected: bool,
    aggregate: Dict[str, Dict[str, Any]],
) -> Tuple[int, int]:
    """Walk one run dir, append rubrics into the dedupe-by-text aggregate.

    Returns ``(examples_seen, rubrics_added)``.
    """
    examples_dir = run_dir / "examples"
    if not examples_dir.is_dir():
        return 0, 0
    examples_seen = 0
    rubrics_added = 0
    for example_path in sorted(examples_dir.glob("*.json")):
        try:
            artifact = json.loads(example_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        examples_seen += 1
        example_id = str((artifact.get("example") or {}).get("example_id") or example_path.stem)
        for entry in _iter_rubrics_from_artifact(
            artifact,
            method_key=method_key,
            include_decomposed=include_decomposed,
            include_rejected=include_rejected,
        ):
            text = str(entry.get("text", "") or "").strip()
            normalized = _normalize_for_dedupe(text)
            if not normalized:
                continue
            entry_record = aggregate.get(normalized)
            if entry_record is None:
                entry_record = {
                    "rubric_id": _rubric_id_for_text(text),
                    "text": text,
                    "source_runs": set(),
                    "source_examples": set(),
                    "occurrence_count": 0,
                    "coverage_count_total": 0,
                    "depth_min": int(entry.get("depth", 0) or 0),
                    "severity_tier": str(entry.get("severity_tier", "medium") or "medium"),
                }
                aggregate[normalized] = entry_record
                rubrics_added += 1
            entry_record["source_runs"].add(run_dir.name)
            entry_record["source_examples"].add(example_id)
            entry_record["occurrence_count"] += 1
            try:
                entry_record["coverage_count_total"] += int(entry.get("coverage_count", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                entry_record["depth_min"] = min(
                    entry_record["depth_min"], int(entry.get("depth", 0) or 0)
                )
            except (TypeError, ValueError):
                pass
    return examples_seen, rubrics_added


def _aggregate_to_index_items(
    aggregate: Mapping[str, Mapping[str, Any]],
    embeddings_by_id: Mapping[str, List[float]],
) -> List[MedicalRubricIndexItem]:
    items: List[MedicalRubricIndexItem] = []
    for record in aggregate.values():
        rubric_id = str(record["rubric_id"])
        embedding = tuple(embeddings_by_id.get(rubric_id, ()))
        items.append(
            MedicalRubricIndexItem(
                rubric_id=rubric_id,
                text=str(record["text"]),
                embedding=embedding,
                source_runs=tuple(sorted(record["source_runs"])),
                source_examples=tuple(sorted(record["source_examples"])),
                occurrence_count=int(record["occurrence_count"]),
                coverage_count=int(record["coverage_count_total"]),
                depth=int(record["depth_min"]),
                severity_tier=str(record["severity_tier"]),
            )
        )
    items.sort(key=lambda item: (-item.occurrence_count, item.rubric_id))
    return items


def _build_embeddings(
    aggregate: Mapping[str, Mapping[str, Any]],
    *,
    embedder: Optional[OpenAIEmbedder],
    limit: int,
) -> Dict[str, List[float]]:
    if embedder is None:
        return {}
    rubric_ids: List[str] = []
    texts: List[str] = []
    for record in aggregate.values():
        rubric_ids.append(str(record["rubric_id"]))
        texts.append(str(record["text"]))
    if limit > 0:
        rubric_ids = rubric_ids[: int(limit)]
        texts = texts[: int(limit)]
    if not texts:
        return {}
    print(f"  Embedding {len(texts)} unique rubrics in batches of {embedder.batch_size}...", flush=True)
    vectors = embedder.embed_texts(texts)
    out: Dict[str, List[float]] = {}
    for rid, vec in zip(rubric_ids, vectors):
        out[rid] = vec
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists():
        print(f"WARNING: {out_path} already exists; it will be overwritten.", file=sys.stderr)

    aggregate: Dict[str, Dict[str, Any]] = {}
    total_examples = 0
    print(f"Scanning {len(args.run_dirs)} run dirs...", flush=True)
    for run_dir in args.run_dirs:
        examples_seen, rubrics_added = _aggregate_run(
            Path(run_dir),
            method_key=args.method_key,
            include_decomposed=bool(args.include_decomposed),
            include_rejected=bool(args.include_rejected),
            aggregate=aggregate,
        )
        total_examples += examples_seen
        print(
            f"  {run_dir}: {examples_seen} examples seen, {rubrics_added} new unique rubrics added",
            flush=True,
        )

    unique_count = len(aggregate)
    if total_examples == 0:
        print("ERROR: no examples found under any of the provided run dirs.", file=sys.stderr)
        return 1
    print(f"Aggregate: {total_examples} examples seen, {unique_count} unique rubrics after dedupe.", flush=True)

    embedder: Optional[OpenAIEmbedder] = None
    if not args.dry_run:
        model_spec = ModelSpec(
            alias=DEFAULT_EMBEDDING_MODEL_SPEC.alias,
            provider=DEFAULT_EMBEDDING_MODEL_SPEC.provider,
            model=str(args.embedding_model or DEFAULT_EMBEDDING_MODEL),
            api_key_env=DEFAULT_EMBEDDING_MODEL_SPEC.api_key_env,
        )
        embedder = OpenAIEmbedder(model_spec=model_spec, batch_size=int(args.batch_size))

    embeddings_by_id = _build_embeddings(
        aggregate,
        embedder=embedder,
        limit=int(args.limit or 0),
    )
    items = _aggregate_to_index_items(aggregate, embeddings_by_id)

    embedding_dim = int(args.embedding_dim or DEFAULT_EMBEDDING_DIM)
    sample_dims = {len(item.embedding) for item in items if item.embedding}
    if sample_dims and len(sample_dims) == 1:
        embedding_dim = next(iter(sample_dims))

    build_metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_run_dirs": [str(p) for p in args.run_dirs],
        "method_key": args.method_key,
        "include_decomposed": bool(args.include_decomposed),
        "include_rejected": bool(args.include_rejected),
        "dry_run": bool(args.dry_run),
        "examples_scanned": int(total_examples),
        "unique_rubric_count": int(unique_count),
        "embedded_count": int(sum(1 for item in items if item.embedding)),
    }

    index = MedicalRubricIndex(
        version="v1",
        embedding_model=str(args.embedding_model or DEFAULT_EMBEDDING_MODEL),
        embedding_dim=embedding_dim,
        items=items,
        build_metadata=build_metadata,
    )
    out_path = index.save(out_path)
    print(f"Wrote index to {out_path}", flush=True)
    print(json.dumps(index.summarize(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
