from __future__ import annotations

from typing import Iterable

from rubric_gen.types import CandidateNote, ExampleRecord


INITIAL_RUBRIC_SYSTEM = """You are a rubric designer for an LLM-as-a-judge system.

Design rubrics that distinguish stronger and weaker clinical notes for the given transcript.
Only propose rubrics you are confident about.
Each rubric must be:
- binary and judgeable,
- clinically grounded in the transcript and note-writing task,
- atomic (one criterion per rubric),
- useful for distinguishing note quality rather than generic writing advice,
- not redundant with other rubric items,
- not a direct copy of a candidate note.

Output only rubric tags in this exact format:
<RUBRIC> ... </RUBRIC>
"""


INITIAL_RUBRIC_SYSTEM_PROMPT_ONLY = """You are a rubric designer for an LLM-as-a-judge system.

Design a comprehensive set of prompt-specific rubrics for evaluating responses to the given task.
Only propose rubrics you are confident about.
Each rubric must be:
- binary and judgeable,
- prompt-specific,
- atomic (one criterion per rubric),
- self-contained,
- useful for distinguishing stronger from weaker responses,
- not generic writing advice.

Output only rubric tags in this exact format:
<RUBRIC> ... </RUBRIC>
"""


DECOMPOSITION_SYSTEM = """You are a rubric designer for an LLM-as-a-judge system.

You will receive a current rubric that is too coarse because it is satisfied by too many candidate notes.
Propose exactly two more granular rubrics that:
- preserve the important intent of the original rubric,
- split it into more discriminative sub-dimensions,
- remain clinically grounded and binary,
- do not overlap substantially with the supplied existing rubrics,
- do not copy any candidate note.

Only produce children when each child reflects a genuinely distinct clinical quality dimension that could change
how candidate notes are ranked.

Do NOT decompose by:
- minor wording changes,
- section-placement preferences,
- splitting the same instruction into tiny timing/frequency fragments unless each fragment is independently useful,
- transcript-only microfacts that are too narrow to matter for note quality,
- near-paraphrases of the parent rubric.

Each child should:
- be narrower than the parent,
- improve discrimination among the candidate notes,
- avoid enforcing exact phrasing when a clinically equivalent statement would suffice.

Output only:
<RUBRIC> ... </RUBRIC>
<RUBRIC> ... </RUBRIC>
"""


OVERLAP_SYSTEM = """You are checking whether a new rubric substantially overlaps with any existing rubric.

Return YES if the new rubric has the same intent, a subset/superset meaning, or enough semantic overlap that
using both would not materially change scoring. Return NO otherwise.

Output only:
<EVALUATION> YES/NO </EVALUATION>
"""


CONFLICT_SYSTEM = """You are checking whether a new rubric conflicts with any existing rubric.

Return YES if the new rubric reverses the direction or meaning of any existing rubric on the same axis.
Return NO otherwise.

Output only:
<EVALUATION> YES/NO </EVALUATION>
"""


SATISFACTION_SYSTEM = """You are a clinical note judge.

Evaluate whether the candidate note satisfies the rubric using only the transcript and the rubric.
Be strict. Do not consider criteria not written in the rubric.

Output JSON only:
{"verdict":"YES or NO","reasoning":"brief explanation"}
"""


SATISFACTION_SYSTEM_RESPONSE_ONLY = """You are a judge evaluating whether a response satisfies a rubric.

Evaluate only the response and the rubric. Do not use outside knowledge or assume extra context.
The rubric should be treated as fully self-contained.

Output JSON only:
{"verdict":"YES or NO","reasoning":"brief explanation"}
"""


def _candidate_block(candidates: Iterable[CandidateNote]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            f"[{candidate.candidate_id} | {candidate.source_label} | {candidate.quality_bucket}]\n"
            f"{candidate.text}"
        )
    return "\n\n".join(blocks)


def render_initial_rubric_prompt(example: ExampleRecord, candidates: Iterable[CandidateNote], max_rubrics: int) -> str:
    return (
        "Task: Generate prompt-specific rubrics for evaluating clinical notes.\n\n"
        f"Transcript:\n{example.conversation}\n\n"
        "Candidate notes:\n"
        f"{_candidate_block(candidates)}\n\n"
        f"Target rubric count: propose about {max_rubrics} strong rubric items.\n"
        "Focus on clinical fidelity, completeness, structure, grounding, and plan correctness."
    )


def render_initial_rubric_prompt_without_responses(example: ExampleRecord, max_rubrics: int) -> str:
    return (
        "Task: Generate prompt-specific rubrics for evaluating responses to this task.\n\n"
        f"Prompt:\n{example.task_prompt}\n\n"
        f"Task instance:\n{example.conversation}\n\n"
        f"Target rubric count: propose about {max_rubrics} strong rubric items.\n"
        "Focus on the important dimensions needed to distinguish a high-quality note from a weaker note."
    )


def render_decomposition_prompt(
    example: ExampleRecord,
    candidates: Iterable[CandidateNote],
    rubric_text: str,
    other_rubrics: Iterable[str],
) -> str:
    other_text = "\n".join(f"- {rubric}" for rubric in other_rubrics) or "- (none)"
    return (
        f"Transcript:\n{example.conversation}\n\n"
        "Candidate notes:\n"
        f"{_candidate_block(candidates)}\n\n"
        f"Current rubric:\n{rubric_text}\n\n"
        f"Other rubrics that new rubrics must not overlap with:\n{other_text}\n"
    )


def render_overlap_prompt(existing_rubrics: Iterable[str], new_rubric: str) -> str:
    existing_text = "\n".join(f"- {rubric}" for rubric in existing_rubrics) or "- (none)"
    return (
        f"EXISTING_RUBRICS:\n{existing_text}\n\n"
        f"NEW_RUBRIC:\n- {new_rubric}\n"
    )


def render_conflict_prompt(existing_rubrics: Iterable[str], new_rubric: str) -> str:
    existing_text = "\n".join(f"- {rubric}" for rubric in existing_rubrics) or "- (none)"
    return (
        f"EXISTING_RUBRICS:\n{existing_text}\n\n"
        f"NEW_RUBRIC:\n- {new_rubric}\n"
    )


def render_satisfaction_prompt(example: ExampleRecord, candidate: CandidateNote, rubric_text: str) -> str:
    return (
        f"Transcript:\n{example.conversation}\n\n"
        f"Candidate note:\n{candidate.text}\n\n"
        f"Rubric:\n{rubric_text}\n"
    )


def render_satisfaction_prompt_response_only(candidate: CandidateNote, rubric_text: str) -> str:
    return (
        f"Response:\n{candidate.text}\n\n"
        f"Rubric:\n{rubric_text}\n"
    )
