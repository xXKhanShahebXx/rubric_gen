from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from rubric_gen.config import PipelineConfig
from rubric_gen.dataio import strongest_anchor_text
from rubric_gen.llm_client import LLMRouter
from rubric_gen.storage import JsonlCache, make_cache_key
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec


NOTE_STYLE_INSTRUCTIONS: Dict[str, str] = {
    "structured": (
        "Write a clinically faithful note with clear section headers such as chief complaint, "
        "history, review of systems, exam, assessment, and plan."
    ),
    "soap": (
        "Write the note in a SOAP-style structure. Keep the note clinically grounded and concise."
    ),
    "concise": (
        "Write a concise problem-oriented note that keeps only clinically relevant details."
    ),
    "plan_focused": (
        "Write a clinically faithful note with special care around assessment, medication, and follow-up plan details."
    ),
}

# QA / response-style writers used for non-note task profiles such as
# `general_instruction_following`, `clinical_decision_support`, and the medical
# Q&A workflow. The prior STYLE_INSTRUCTIONS-only setup forced every writer to
# emit a clinical note even when the task was a plain question; that biased the
# discovered rubrics toward clinical-note structure and made gold Q&A answers
# fail rubric satisfaction.
QA_STYLE_INSTRUCTIONS: Dict[str, str] = {
    "direct": (
        "Answer the question directly and concisely. Lead with the answer, then "
        "give one sentence of justification grounded in established medical facts."
    ),
    "explained": (
        "Answer the question and briefly explain the reasoning. Cite the key "
        "mechanism, finding, or guideline that supports the answer in 2-4 "
        "sentences."
    ),
    "comprehensive": (
        "Answer the question thoroughly. Cover the conclusion, the supporting "
        "reasoning, and any clinically relevant nuances or caveats. Stay grounded "
        "in established medical knowledge and avoid speculation."
    ),
    "clinical_reasoning": (
        "Answer the question using a structured clinical reasoning chain: name "
        "the key findings, walk through the differential or mechanism, then state "
        "the final answer. Keep the answer focused and avoid invented details."
    ),
}

# Backwards-compatible alias preserved for any external code that imported the
# old name. Defaults to the note styles, which is the original behaviour.
STYLE_INSTRUCTIONS: Dict[str, str] = NOTE_STYLE_INSTRUCTIONS

NOTE_TASK_PROFILE_IDS: Tuple[str, ...] = (
    "note_documentation",
    "documentation_variants",
)


def _is_note_task(example: ExampleRecord) -> bool:
    profile_id = (example.task_profile_id or "").strip().lower()
    return profile_id in NOTE_TASK_PROFILE_IDS


def _styles_for(example: ExampleRecord) -> Dict[str, str]:
    return NOTE_STYLE_INSTRUCTIONS if _is_note_task(example) else QA_STYLE_INSTRUCTIONS


def _system_prompt_for(example: ExampleRecord) -> str:
    if _is_note_task(example):
        return (
            "You are an expert clinical documentation assistant. "
            "Produce only the clinical note."
        )
    return (
        "You are an expert medical assistant answering a medical question for a "
        "clinician. Produce only the answer to the question."
    )


def _context_label_for(example: ExampleRecord) -> str:
    return "Transcript" if _is_note_task(example) else "Context"


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _candidate_prompt(example: ExampleRecord, style_name: str) -> str:
    styles = _styles_for(example)
    style_instruction = styles[style_name]
    parts = [example.task_prompt, f"Style instruction: {style_instruction}"]
    if example.conversation.strip():
        parts.append(f"\n{_context_label_for(example)}:\n{example.conversation}")
    return "\n".join(parts) + "\n"


def _quality_bucket_for_model(spec: ModelSpec) -> str:
    if spec.provider in {"openai", "anthropic"}:
        return "frontier_generated"
    return "open_generated"


def _anchor_candidates(example: ExampleRecord) -> List[CandidateNote]:
    anchors: List[CandidateNote] = []
    if example.reference_note:
        anchors.append(
            CandidateNote(
                candidate_id=f"{example.example_id}__reference_note",
                example_id=example.example_id,
                text=example.reference_note,
                source_label="reference_note",
                quality_bucket="gold_like",
                origin_kind="anchor",
            )
        )
    if example.augmented_note:
        anchors.append(
            CandidateNote(
                candidate_id=f"{example.example_id}__augmented_note",
                example_id=example.example_id,
                text=example.augmented_note,
                source_label="augmented_note",
                quality_bucket="strong_anchor",
                origin_kind="anchor",
            )
        )
    if example.note_truncated:
        anchors.append(
            CandidateNote(
                candidate_id=f"{example.example_id}__note_truncated",
                example_id=example.example_id,
                text=example.note_truncated,
                source_label="note_truncated",
                quality_bucket="synthetically_degraded",
                origin_kind="anchor",
            )
        )
    return anchors


def _generation_plan(
    writer_models: Sequence[ModelSpec],
    target_count: int,
    example: Optional[ExampleRecord] = None,
) -> List[Tuple[ModelSpec, str, float, int]]:
    if not writer_models or target_count <= 0:
        return []
    style_names = list(_styles_for(example) if example is not None else NOTE_STYLE_INSTRUCTIONS)
    temperatures = [0.0, 0.2, 0.3, 0.1]
    jobs: List[Tuple[ModelSpec, str, float, int]] = []
    for index in range(target_count):
        spec = writer_models[index % len(writer_models)]
        style_name = style_names[index % len(style_names)]
        temperature = temperatures[index % len(temperatures)]
        jobs.append((spec, style_name, temperature, index))
    return jobs


def _paper_generation_plan(
    writer_models: Sequence[ModelSpec],
    target_count: int,
) -> List[Tuple[ModelSpec, float, int]]:
    if not writer_models or target_count <= 0:
        return []

    if len(writer_models) >= 2 and target_count >= 8:
        temperatures = [0.0, 0.2, 0.4, 0.6]
        jobs: List[Tuple[ModelSpec, float, int]] = []
        first = writer_models[0]
        second = writer_models[1]
        for index, temperature in enumerate(temperatures):
            jobs.append((first, temperature, index))
        for index, temperature in enumerate(temperatures, start=len(temperatures)):
            jobs.append((second, temperature, index))
        return jobs[:target_count]

    temperatures = [0.0, 0.2, 0.4, 0.6]
    jobs = []
    for index in range(target_count):
        spec = writer_models[index % len(writer_models)]
        temperature = temperatures[index % len(temperatures)]
        jobs.append((spec, temperature, index))
    return jobs


def _truncate_note(note: str) -> str:
    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", note) if chunk.strip()]
    if paragraphs:
        keep = max(1, len(paragraphs) // 2)
        return "\n\n".join(paragraphs[:keep]).strip()
    lines = [line for line in note.splitlines() if line.strip()]
    keep = max(1, len(lines) // 2)
    return "\n".join(lines[:keep]).strip()


def _remove_plan_details(note: str) -> str:
    lines = []
    for line in note.splitlines():
        if re.search(r"follow up|follow-up|assessment|plan|medication|treatment", line, re.IGNORECASE):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    text = re.sub(r"\b\d+(\.\d+)?\b", "[unspecified]", text)
    return text.strip()


def _synthetic_degradations(example: ExampleRecord, count: int) -> List[CandidateNote]:
    seed_note = strongest_anchor_text(example)
    if not seed_note:
        return []

    degraded_texts = [
        ("weak_truncate", _truncate_note(seed_note)),
        ("weak_remove_plan", _remove_plan_details(seed_note)),
        ("weak_short", _truncate_note(_remove_plan_details(seed_note))),
    ]

    candidates: List[CandidateNote] = []
    for index, (label, text) in enumerate(degraded_texts[:count]):
        if not text:
            continue
        candidates.append(
            CandidateNote(
                candidate_id=f"{example.example_id}__{label}_{index}",
                example_id=example.example_id,
                text=text,
                source_label=label,
                quality_bucket="synthetically_degraded",
                origin_kind="synthetic",
                parent_candidate_id=f"{example.example_id}__reference_note" if example.reference_note else None,
            )
        )
    return candidates


def _dedupe_candidates(candidates: List[CandidateNote]) -> List[CandidateNote]:
    deduped: List[CandidateNote] = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_for_dedupe(candidate.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def build_paper_candidate_pool(
    example: ExampleRecord,
    config: PipelineConfig,
    router: Optional[LLMRouter],
    generation_cache: JsonlCache,
) -> List[CandidateNote]:
    candidates: List[CandidateNote] = []

    if not config.dry_run and router is not None:
        for spec, temperature, index in _paper_generation_plan(config.writer_models, config.target_candidate_count):
            if _is_note_task(example):
                instruction = (
                    "Produce one clinically faithful note for this case. Use your "
                    "own best judgment on structure and wording, but stay grounded "
                    "in the prompt."
                )
            else:
                instruction = (
                    "Answer the question accurately using your own best judgment on "
                    "structure and wording. Stay grounded in established medical "
                    "knowledge and avoid speculation."
                )
            context_section = (
                f"\n\n{_context_label_for(example)}:\n{example.conversation}\n"
                if example.conversation.strip()
                else "\n"
            )
            prompt = f"{example.task_prompt}\n{instruction}{context_section}"
            system_prompt = _system_prompt_for(example)
            cache_key = make_cache_key(
                "paper_candidate_generation",
                {
                    "example_id": example.example_id,
                    "model": spec.model,
                    "provider": spec.provider,
                    "temperature": temperature,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                },
            )
            cached = generation_cache.get(cache_key)
            if cached is None:
                try:
                    response = router.generate(
                        spec=spec,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        temperature=temperature,
                    )
                except Exception:
                    continue
                cached = generation_cache.set(
                    cache_key,
                    {
                        "candidate": {
                            "candidate_id": f"{example.example_id}__paper_generated_{index}",
                            "example_id": example.example_id,
                            "text": response.text,
                            "source_label": f"paper_generated_{spec.alias}_{index}",
                            "quality_bucket": _quality_bucket_for_model(spec),
                            "origin_kind": "generated",
                            "model_alias": spec.alias,
                            "provider": spec.provider,
                            "prompt_style": "paper_mode",
                            "temperature": temperature,
                            "parent_candidate_id": None,
                            "metadata": {
                                "model": spec.model,
                                "latency_s": response.latency_s,
                                "paper_mode": True,
                            },
                        }
                    },
                )
            candidates.append(CandidateNote(**cached["candidate"]))

    candidates = _dedupe_candidates(candidates)
    return candidates[: config.target_candidate_count]


def build_candidate_pool(
    example: ExampleRecord,
    config: PipelineConfig,
    router: Optional[LLMRouter],
    generation_cache: JsonlCache,
) -> List[CandidateNote]:
    candidates = _anchor_candidates(example)

    weak_existing = sum(1 for candidate in candidates if candidate.quality_bucket == "synthetically_degraded")
    reserved_weak_slots = max(0, 2 - weak_existing)
    needed_generated = max(0, config.target_candidate_count - len(candidates) - reserved_weak_slots)

    if not config.dry_run and router is not None:
        for spec, style_name, temperature, index in _generation_plan(
            config.writer_models, needed_generated, example=example
        ):
            prompt = _candidate_prompt(example, style_name)
            system_prompt = _system_prompt_for(example)
            cache_key = make_cache_key(
                "candidate_generation",
                {
                    "example_id": example.example_id,
                    "model": spec.model,
                    "provider": spec.provider,
                    "style": style_name,
                    "temperature": temperature,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                },
            )
            cached = generation_cache.get(cache_key)
            if cached is None:
                try:
                    response = router.generate(
                        spec=spec,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        temperature=temperature,
                    )
                except Exception as exc:
                    continue
                cached = generation_cache.set(
                    cache_key,
                    {
                        "candidate": {
                            "candidate_id": f"{example.example_id}__generated_{index}",
                            "example_id": example.example_id,
                            "text": response.text,
                            "source_label": f"generated_{style_name}",
                            "quality_bucket": _quality_bucket_for_model(spec),
                            "origin_kind": "generated",
                            "model_alias": spec.alias,
                            "provider": spec.provider,
                            "prompt_style": style_name,
                            "temperature": temperature,
                            "parent_candidate_id": None,
                            "metadata": {
                                "model": spec.model,
                                "latency_s": response.latency_s,
                            },
                        }
                    },
                )
            candidate_payload = cached["candidate"]
            candidates.append(CandidateNote(**candidate_payload))

    candidates = _dedupe_candidates(candidates)
    remaining = max(0, config.target_candidate_count - len(candidates))
    if remaining > 0:
        candidates.extend(_synthetic_degradations(example, remaining))

    candidates = _dedupe_candidates(candidates)
    return candidates[: config.target_candidate_count]
