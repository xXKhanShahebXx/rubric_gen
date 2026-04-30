from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rubric_gen.types import ExampleRecord


NOTE_TASK_PROMPT = """You are a healthcare scribe.

Write a clinically faithful medical note from the transcript below.

Requirements:
- Use only information grounded in the transcript.
- Preserve important symptoms, diagnoses, medications, plans, follow-up, and clinically relevant negatives.
- Exclude small talk unless it changes the clinical picture or adherence context.
- Do not invent diagnoses, medications, test results, or follow-up instructions.
- Produce a clear clinical note with section headers.
"""

DOCUMENTATION_VARIANT_PROMPT = """Produce the requested documentation artifact using only the provided context.

Requirements:
- Stay grounded in the source context.
- Preserve important actions, findings, and follow-up details.
- Match the requested document style and structure when specified.
- Do not invent facts, plans, or results.
"""

REWRITE_EDIT_PROMPT = """Rewrite or edit the provided text exactly as requested.

Requirements:
- Preserve the underlying meaning unless the instruction explicitly asks for a substantive change.
- Improve wording, structure, tone, or grammar according to the task.
- Do not add unsupported facts or requirements.
"""

CLINICAL_DECISION_SUPPORT_PROMPT = """Generate a grounded clinical decision-support artifact from the provided context.

Requirements:
- Base recommendations and reasoning only on the provided evidence.
- Include the requested next steps, options, or management details when appropriate.
- Avoid unsupported certainty or fabricated findings.
"""

GENERAL_INSTRUCTION_PROMPT = """Complete the requested task using the provided context.

Requirements:
- Follow the user's instructions and output constraints.
- Stay grounded in the available context.
- Be complete, accurate, and avoid unsupported additions.
"""

AGENTIC_WORKFLOW_PROMPT = """Produce the requested workflow output from the provided context.

Requirements:
- Reflect the completed steps, tool results, or blockers that are actually present.
- Keep conclusions grounded in the observed evidence.
- Preserve failure handling, verification, and final action status when relevant.
"""

_DEFAULT_PROMPTS = {
    "note_documentation": NOTE_TASK_PROMPT,
    "documentation_variants": DOCUMENTATION_VARIANT_PROMPT,
    "rewrite_editing": REWRITE_EDIT_PROMPT,
    "clinical_decision_support": CLINICAL_DECISION_SUPPORT_PROMPT,
    "general_instruction_following": GENERAL_INSTRUCTION_PROMPT,
    "agentic_workflows": AGENTIC_WORKFLOW_PROMPT,
}

_SOURCE_CONTEXT_KEYS: Sequence[str] = (
    "conversation",
    "dialogue",
    "transcript",
    "input",
    "context",
    "source_context",
    "source_text",
    "request_context",
)

_TASK_PROMPT_KEYS: Sequence[str] = (
    "task_prompt",
    "instruction",
    "instructions",
    "prompt",
    "task",
    "request",
    "user_request",
    "Question",
    "question",
)

_REFERENCE_ARTIFACT_KEYS: Sequence[str] = (
    "reference_artifact",
    "reference_output",
    "reference_response",
    "reference_text",
    "reference_note",
    "ideal_completion",
    "ideal_output",
    "gold_output",
    "target",
)

_AUGMENTED_ARTIFACT_KEYS: Sequence[str] = (
    "augmented_artifact",
    "augmented_output",
    "augmented_response",
    "augmented_text",
    "augmented_note",
    "candidate_output",
    "candidate_response",
    "response",
    "assistant_response",
    "completion",
    "output",
    "alt_completion",
)

_TRUNCATED_ARTIFACT_KEYS: Sequence[str] = (
    "artifact_truncated",
    "output_truncated",
    "response_truncated",
    "note_truncated",
)

_STRUCTURED_SUMMARY_KEYS: Sequence[str] = (
    "structured_summary",
    "summary_json",
    "structured_context",
)

_KNOWN_ROW_KEYS = {
    "source",
    "source_id",
    "id",
    "dataset_subset",
    "task_profile_id",
    "task_family_id",
    "artifact_kind",
    "note_family_id",
    *_SOURCE_CONTEXT_KEYS,
    *_TASK_PROMPT_KEYS,
    *_REFERENCE_ARTIFACT_KEYS,
    *_AUGMENTED_ARTIFACT_KEYS,
    *_TRUNCATED_ARTIFACT_KEYS,
    *_STRUCTURED_SUMMARY_KEYS,
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False).strip()
    return str(value).strip()


def _parse_structured_summary(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, "", {}):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"raw": text}
    return {"raw": _clean_text(value)}


def _first_present_text(row: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row:
            value = _clean_text(row.get(key))
            if value:
                return value
    return ""


def _collect_source_context(row: Dict[str, Any]) -> str:
    seen: set[str] = set()
    parts: List[str] = []
    for key in _SOURCE_CONTEXT_KEYS:
        value = _clean_text(row.get(key))
        if value and value not in seen:
            seen.add(value)
            parts.append(value)
    return "\n\n".join(parts)


def _infer_task_profile_id(row: Dict[str, Any]) -> str:
    explicit = _clean_text(row.get("task_profile_id"))
    if explicit:
        return explicit

    blob = " ".join(
        value
        for value in (
            _first_present_text(row, _TASK_PROMPT_KEYS),
            _collect_source_context(row),
            _clean_text(row.get("task_family_id")),
            _clean_text(row.get("artifact_kind")),
        )
        if value
    ).lower()

    if row.get("reference_note") or row.get("augmented_note") or "healthcare scribe" in blob:
        return "note_documentation"
    if any(token in blob for token in ("discharge", "progress note", "pre-op", "preop", "summary", "handoff", "patient message", "alert")):
        return "documentation_variants"
    if any(token in blob for token in ("rewrite", "rephrase", "edit", "grammar", "proofread", "tone", "shorten", "expand")):
        return "rewrite_editing"
    if any(token in blob for token in ("differential", "recommendation", "next step", "management plan", "triage", "workup")):
        return "clinical_decision_support"
    if any(token in blob for token in ("tool result", "workflow", "step by step", "failure", "retry", "verification", "agent")):
        return "agentic_workflows"
    return "general_instruction_following"


def _default_prompt_for_profile(task_profile_id: str) -> str:
    return _DEFAULT_PROMPTS.get(task_profile_id, GENERAL_INSTRUCTION_PROMPT)


def _default_artifact_kind(task_profile_id: str, row: Dict[str, Any]) -> str:
    explicit = _clean_text(row.get("artifact_kind"))
    if explicit:
        return explicit
    if task_profile_id in {"note_documentation", "documentation_variants"}:
        return "note"
    if task_profile_id in {"general_instruction_following", "clinical_decision_support"}:
        return "response"
    return "artifact"


def _extract_extra_metadata(row: Dict[str, Any], payload: Any, row_index: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "row_index": row_index,
    }
    if isinstance(payload, dict):
        if "aci_dataset" in payload:
            metadata["aci_dataset"] = payload.get("aci_dataset")
        if "agbonnet_dataset" in payload:
            metadata["agbonnet_dataset"] = payload.get("agbonnet_dataset")

    extras = {key: value for key, value in row.items() if key not in _KNOWN_ROW_KEYS}
    if extras:
        metadata["extra_fields"] = extras
    return metadata


def _rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        examples = payload.get("examples")
        if isinstance(examples, list):
            return [row for row in examples if isinstance(row, dict)]
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _read_dataset_rows(dataset_path: Path) -> Tuple[List[Dict[str, Any]], Any]:
    """Read rows from either a JSON or JSONL dataset file.

    Returns (rows, payload). For JSONL files `payload` is the list of rows.
    """
    suffix = dataset_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: List[Dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    rows.append(record)
        return rows, rows

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _rows_from_payload(payload), payload


def _apply_split_and_shard(
    rows: List[Dict[str, Any]],
    *,
    split: Optional[str],
    train_size: int,
    val_size: int,
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    """Slice rows by train/val split and shard.

    Splitting is deterministic and order-preserving so that team members on
    different machines select identical shards from the same source file:

    * `split="train"` selects the first `train_size` rows, then partitions
      them into `num_shards` contiguous shards and returns shard `shard_index`.
    * `split="val"`   selects rows `[train_size, train_size + val_size)`.
    * `split="all"` / `None` returns all rows unchanged.

    The shard partition is contiguous (rows 0..N/K-1 → shard 0, etc.). For 3000
    train rows with `num_shards=3`, shard 0 covers rows [0, 1000), shard 1
    covers [1000, 2000), and shard 2 covers [2000, 3000).
    """
    if split is None or split == "all":
        return rows

    if split not in {"train", "val"}:
        raise ValueError(
            f"split must be one of 'train', 'val', or 'all'; got '{split}'."
        )

    if train_size < 0 or val_size < 0:
        raise ValueError("train_size and val_size must be non-negative.")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}."
        )

    if split == "val":
        return rows[train_size : train_size + val_size]

    train_rows = rows[:train_size]
    if num_shards == 1:
        return train_rows

    shard_size = len(train_rows) // num_shards
    remainder = len(train_rows) % num_shards
    if remainder != 0:
        raise ValueError(
            f"train_size ({train_size}) is not evenly divisible by "
            f"num_shards ({num_shards}); refusing to silently truncate. "
            "Either pick a divisible train_size or pre-trim the source file."
        )
    start = shard_index * shard_size
    end = start + shard_size
    return train_rows[start:end]


def _build_example_id(source: str, source_id: str, row_index: int) -> str:
    left = source.lower().replace(" ", "_").replace("-", "_")
    right = source_id.strip().replace(" ", "_") or str(row_index)
    return f"{left}__{right}"


def load_examples(
    dataset_path: Path,
    start: int = 0,
    limit: int = 0,
    source_filter: Optional[str] = None,
    split: Optional[str] = None,
    train_size: int = 0,
    val_size: int = 0,
    num_shards: int = 1,
    shard_index: int = 0,
    reference_field_overrides: Optional[Sequence[str]] = None,
) -> List[ExampleRecord]:
    rows, payload = _read_dataset_rows(dataset_path)
    rows = _apply_split_and_shard(
        rows,
        split=split,
        train_size=train_size,
        val_size=val_size,
        num_shards=num_shards,
        shard_index=shard_index,
    )

    # Field names supplied via reference_field_overrides take priority over the
    # default _REFERENCE_ARTIFACT_KEYS, letting a workflow promote a generic
    # column (e.g. medical_rl_prompts' "response") into the gold/reference slot
    # without changing the global default for other datasets.
    effective_reference_keys: List[str] = []
    seen_keys: set[str] = set()
    for key in list(reference_field_overrides or []) + list(_REFERENCE_ARTIFACT_KEYS):
        if key and key not in seen_keys:
            effective_reference_keys.append(key)
            seen_keys.add(key)

    normalized: List[ExampleRecord] = []

    for row_index, row in enumerate(rows):
        task_profile_id = _infer_task_profile_id(row)
        source = _clean_text(row.get("source")) or task_profile_id
        if source_filter and source_filter.lower() not in source.lower():
            continue

        source = source or task_profile_id
        source_id = (
            _clean_text(row.get("source_id"))
            or _clean_text(row.get("id"))
            or str(row_index)
        )
        source_context = _collect_source_context(row)
        task_prompt = _first_present_text(row, _TASK_PROMPT_KEYS) or _default_prompt_for_profile(task_profile_id)
        reference_artifact = _first_present_text(row, effective_reference_keys)
        augmented_artifact = _first_present_text(row, _AUGMENTED_ARTIFACT_KEYS)
        # When a field appears in both lists (e.g. `response` after the medical
        # preset promotes it), the same text would otherwise build two anchor
        # candidates that dedupe down to one. Suppress the augmented copy so the
        # gold semantics are preserved end-to-end.
        if (
            reference_artifact
            and augmented_artifact
            and reference_artifact == augmented_artifact
        ):
            augmented_artifact = ""
        artifact_truncated = _first_present_text(row, _TRUNCATED_ARTIFACT_KEYS)
        structured_summary = None
        for key in _STRUCTURED_SUMMARY_KEYS:
            if key in row:
                structured_summary = _parse_structured_summary(row.get(key))
                if structured_summary is not None:
                    break

        example = ExampleRecord(
            example_id=_build_example_id(source, source_id, row_index),
            source=source,
            source_id=source_id,
            dataset_subset=_clean_text(row.get("dataset_subset")),
            conversation=source_context,
            task_prompt=task_prompt,
            reference_note=_clean_text(row.get("reference_note")) or reference_artifact,
            augmented_note=_clean_text(row.get("augmented_note")) or augmented_artifact,
            note_truncated=_clean_text(row.get("note_truncated")) or artifact_truncated,
            structured_summary=structured_summary,
            metadata=_extract_extra_metadata(row, payload, row_index),
            task_profile_id=task_profile_id,
            task_family_id=_clean_text(row.get("task_family_id")) or _clean_text(row.get("note_family_id")),
            artifact_kind=_default_artifact_kind(task_profile_id, row),
            reference_artifact=reference_artifact or _clean_text(row.get("reference_note")),
            augmented_artifact=augmented_artifact or _clean_text(row.get("augmented_note")),
            artifact_truncated=artifact_truncated or _clean_text(row.get("note_truncated")),
        )
        normalized.append(example)

    normalized = normalized[start:]
    if limit > 0:
        normalized = normalized[:limit]
    return normalized


def example_to_prompt(example: ExampleRecord) -> str:
    context_label = str(example.metadata.get("source_context_label") or "")
    if not context_label:
        context_label = "Transcript" if example.task_profile_id == "note_documentation" else "Input"
    if example.conversation.strip():
        return (
            f"{example.task_prompt}\n\n"
            f"{context_label}:\n{example.conversation}\n"
        )
    return example.task_prompt


def strongest_anchor_text(example: ExampleRecord) -> str:
    for candidate in [
        example.reference_artifact,
        example.augmented_artifact,
        example.artifact_truncated,
        example.reference_note,
        example.augmented_note,
        example.note_truncated,
    ]:
        if candidate:
            return candidate
    return ""
