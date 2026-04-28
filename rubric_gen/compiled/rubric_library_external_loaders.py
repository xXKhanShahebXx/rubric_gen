"""
Loaders for external preference sources used by the rubric library builder.

These loaders are intentionally *lazy*: importing this module must NOT require ``datasets`` or
network access. Each loader checks for the backing dependency at call time and returns an empty
iterator with a warning if unavailable. Production builds will have ``datasets`` installed and
network access; tests use the seed manifest and do not touch this module.

Supported sources:

- ``helpsteer3_preference`` (``nvidia/HelpSteer3``)
- ``ultrafeedback_binarized`` (``HuggingFaceH4/ultrafeedback_binarized``)
- ``ppe_mmlu_pro`` and ``ppe_gpqa`` (``lmarena-ai/ppe``)
- ``arena_hard`` (``lmsys/arena-hard-auto``)
- ``synthetic_multi_model`` (local JSONL produced by ``synthetic_pair_generator.py``)
- ``jsonl_file`` (freeform local JSONL with ``prompt``, ``chosen``, ``rejected``, ``source_family``)
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from rubric_gen.compiled.rubric_library_builder import (
    ExternalPreferencePair,
    ProposedCriterion,
)
from rubric_gen.compiled.rubric_library_seed import seed_proposer


_LOG = logging.getLogger(__name__)


def _warn(message: str) -> None:
    _LOG.warning(message)
    print(f"[rubric_library_external_loaders] {message}", file=sys.stderr)


def _load_hf_dataset(name: str, *, split: str, subset: Optional[str] = None) -> Optional[Iterable[Mapping[str, Any]]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        _warn(f"datasets package unavailable ({exc}); skipping {name}")
        return None
    try:
        if subset:
            return load_dataset(name, subset, split=split)
        return load_dataset(name, split=split)
    except Exception as exc:
        _warn(f"Failed to load {name} split={split}: {exc}")
        return None


def _limited(iter_like: Optional[Iterable[Mapping[str, Any]]], limit: Optional[int]) -> Iterable[Mapping[str, Any]]:
    if iter_like is None:
        return []
    if limit is None or limit <= 0:
        return iter_like
    out: List[Mapping[str, Any]] = []
    for idx, row in enumerate(iter_like):
        if idx >= limit:
            break
        out.append(row)
    return out


def _helpsteer3_loader(source: Mapping[str, Any]) -> Iterable[ExternalPreferencePair]:
    split = str(source.get("split", "train") or "train")
    limit = source.get("limit")
    rows = _load_hf_dataset("nvidia/HelpSteer3", split=split)
    for idx, row in enumerate(_limited(rows, int(limit) if isinstance(limit, int) else None)):
        prompt = str(row.get("prompt", "") or row.get("conversation_input", ""))
        chosen = str(row.get("response1", "") or "")
        rejected = str(row.get("response2", "") or "")
        preference = str(row.get("preference", "")).strip().lower()
        if preference == "response2":
            chosen, rejected = rejected, chosen
        if not chosen or not rejected:
            continue
        yield ExternalPreferencePair(
            pair_id=f"helpsteer3_{idx}",
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source="helpsteer3_preference",
            source_family=str(source.get("source_family", "generic") or "generic"),
            focus_kind=str(source.get("focus_kind", "") or ""),
            metadata={"hf_index": idx},
        )


def _ultrafeedback_loader(source: Mapping[str, Any]) -> Iterable[ExternalPreferencePair]:
    split = str(source.get("split", "train_prefs") or "train_prefs")
    limit = source.get("limit")
    rows = _load_hf_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    for idx, row in enumerate(_limited(rows, int(limit) if isinstance(limit, int) else None)):
        prompt = str(row.get("prompt", "") or "")
        chosen_rows = row.get("chosen", [])
        rejected_rows = row.get("rejected", [])
        chosen_text = ""
        rejected_text = ""
        if isinstance(chosen_rows, list) and chosen_rows:
            chosen_text = str(chosen_rows[-1].get("content", "") if isinstance(chosen_rows[-1], Mapping) else chosen_rows[-1])
        if isinstance(rejected_rows, list) and rejected_rows:
            rejected_text = str(rejected_rows[-1].get("content", "") if isinstance(rejected_rows[-1], Mapping) else rejected_rows[-1])
        if not chosen_text or not rejected_text:
            continue
        yield ExternalPreferencePair(
            pair_id=f"ultrafeedback_{idx}",
            prompt=prompt,
            chosen=chosen_text,
            rejected=rejected_text,
            source="ultrafeedback_binarized",
            source_family=str(source.get("source_family", "generic") or "generic"),
            focus_kind=str(source.get("focus_kind", "") or ""),
            metadata={"hf_index": idx},
        )


def _ppe_loader(source: Mapping[str, Any]) -> Iterable[ExternalPreferencePair]:
    subset = str(source.get("subset", "mmlu_pro") or "mmlu_pro")
    split = str(source.get("split", "test") or "test")
    limit = source.get("limit")
    rows = _load_hf_dataset("lmarena-ai/ppe", split=split, subset=subset)
    for idx, row in enumerate(_limited(rows, int(limit) if isinstance(limit, int) else None)):
        prompt = str(row.get("prompt", "") or "")
        chosen = str(row.get("chosen", row.get("response_a", "")) or "")
        rejected = str(row.get("rejected", row.get("response_b", "")) or "")
        winner = str(row.get("winner", "")).strip().lower()
        if winner == "b":
            chosen, rejected = rejected, chosen
        if not chosen or not rejected:
            continue
        yield ExternalPreferencePair(
            pair_id=f"ppe_{subset}_{idx}",
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source=f"ppe_{subset}",
            source_family=str(source.get("source_family", "mmlu-pro") or "mmlu-pro"),
            focus_kind=str(source.get("focus_kind", "") or ""),
            metadata={"subset": subset, "hf_index": idx},
        )


def _arena_hard_loader(source: Mapping[str, Any]) -> Iterable[ExternalPreferencePair]:
    split = str(source.get("split", "train") or "train")
    limit = source.get("limit")
    rows = _load_hf_dataset("lmsys/arena-hard-auto", split=split)
    for idx, row in enumerate(_limited(rows, int(limit) if isinstance(limit, int) else None)):
        prompt = str(row.get("prompt", "") or "")
        chosen = str(row.get("chosen", "") or "")
        rejected = str(row.get("rejected", "") or "")
        if not chosen or not rejected:
            continue
        yield ExternalPreferencePair(
            pair_id=f"arena_hard_{idx}",
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source="arena_hard",
            source_family=str(source.get("source_family", "generic") or "generic"),
            focus_kind=str(source.get("focus_kind", "") or ""),
            metadata={"hf_index": idx},
        )


def _jsonl_loader(source: Mapping[str, Any]) -> Iterable[ExternalPreferencePair]:
    path_value = source.get("path")
    if not path_value:
        _warn("jsonl_file source missing 'path'")
        return []
    path = Path(str(path_value))
    if not path.exists():
        _warn(f"jsonl_file source path does not exist: {path}")
        return []
    limit = source.get("limit")
    source_family = str(source.get("source_family", "generic") or "generic")
    source_tag = str(source.get("source_tag", path.stem) or path.stem)
    out: List[ExternalPreferencePair] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if isinstance(limit, int) and limit > 0 and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, Mapping):
                continue
            chosen = str(row.get("chosen", "") or "")
            rejected = str(row.get("rejected", "") or "")
            if not chosen or not rejected:
                continue
            out.append(
                ExternalPreferencePair(
                    pair_id=str(row.get("pair_id", f"{source_tag}_{idx}") or f"{source_tag}_{idx}"),
                    prompt=str(row.get("prompt", "") or ""),
                    chosen=chosen,
                    rejected=rejected,
                    source=source_tag,
                    source_family=str(row.get("source_family", source_family) or source_family),
                    focus_kind=str(row.get("focus_kind", "") or ""),
                    metadata=dict(row.get("metadata", {}) or {}),
                )
            )
    return out


REGISTERED_LOADERS: Dict[str, Callable[[Mapping[str, Any]], Iterable[ExternalPreferencePair]]] = {
    "helpsteer3_preference": _helpsteer3_loader,
    "ultrafeedback_binarized": _ultrafeedback_loader,
    "ppe": _ppe_loader,
    "arena_hard": _arena_hard_loader,
    "jsonl_file": _jsonl_loader,
}


def build_default_proposer(config: Mapping[str, Any]) -> Callable[[ExternalPreferencePair], List[ProposedCriterion]]:
    """
    Return an LLM-backed proposer if API access is configured; otherwise fall back to the
    deterministic seed proposer. The LLM proposer is wired to ``rubric_gen.compiled.discovery``'s
    multi-model discovery flow in production; here we keep the default light-weight so the test
    suite does not require network access.
    """
    kind = str(config.get("kind", "seed") or "seed").strip().lower()
    if kind == "seed":
        return seed_proposer
    try:
        from rubric_gen.compiled.rubric_library_llm_proposer import make_llm_proposer  # noqa: WPS433
    except Exception as exc:
        _warn(f"LLM proposer unavailable ({exc}); using seed proposer.")
        return seed_proposer
    return make_llm_proposer(config)


def build_default_verifier(config: Mapping[str, Any]):
    """
    Mirror of ``build_default_proposer`` for the verifier. Returns ``None`` to use the deterministic
    default verifier in ``rubric_library_builder``.
    """
    kind = str(config.get("kind", "default") or "default").strip().lower()
    if kind == "default":
        return None
    try:
        from rubric_gen.compiled.rubric_library_llm_proposer import make_llm_verifier  # noqa: WPS433
    except Exception as exc:
        _warn(f"LLM verifier unavailable ({exc}); using default verifier.")
        return None
    return make_llm_verifier(config)
