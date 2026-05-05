from __future__ import annotations

from typing import Iterable

from rubric_gen.types import CandidateNote, ExampleRecord


# Forced rules shared by every task-type-specific system prompt.  These are
# the v2 deltas designed to attack the diagnostic findings:
#   * 74% of shard-0 rejections are `overlap`, driven in part by the proposer
#     producing many near-duplicates that all start with the same template.
#   * Top-50 4-token prefixes cover 72.2% of all rubrics; "the note correctly
#     identifies" alone is 9.0% of the library.
#   * 11.8% of rubrics have p(1-p)<0.05 (fire on >95% or <5% of candidates),
#     i.e. provide effectively no discrimination signal.
# The forced rules below ban the over-used templates and require each rubric
# to be a substantive discrimination axis.
_FORCED_RUBRIC_RULES = """\
HARD CONSTRAINTS:
- Each rubric must be a binary YES/NO judgement (no scales, no "both A and B").
- Each rubric must be atomic (one criterion per rubric).
- Each rubric must be a substantive discrimination axis: at least one
  plausible answer would satisfy it AND at least one plausible answer would
  fail it.  Do NOT propose criteria that virtually every reasonable answer
  would satisfy (or fail).
- Do NOT copy or paraphrase any specific candidate response.
- Do NOT propose more than one rubric on the same intent.

BANNED OPENING TEMPLATES (these are over-used and produce non-discriminative
rubrics; do not start any rubric with them or close paraphrases). The ban
covers all noun-substitutions ("note", "answer", "response", "explanation",
"output"):
- "The {note,answer,response,explanation,output} correctly identifies ..."
- "The {note,answer,response,explanation,output} includes a ..." / "... includes the ..."
- "The {note,answer,response,explanation,output} provides a ..."
- "The {note,answer,response,explanation,output} discusses the ..."
- "The {note,answer,response,explanation,output} accurately identifies ..."
- "The {note,answer,response,explanation,output} accurately describes ..."
- "The {note,answer,response,explanation,output} addresses the ..."
- "The {note,answer,response,explanation,output} mentions the ..."

Instead, lead with the SPECIFIC clinical claim or test the rubric requires.
For example: "Hypochlorous acid is named as the disinfecting agent" is
preferred over "The answer correctly identifies the disinfecting agent".

PREFER discrimination dimensions (when applicable to the task):
1. Factual correctness on the central claim or required answer.
2. Completeness on the explicit sub-questions or steps the prompt asks for.
3. Explicit, traceable reasoning chain (when the task requires reasoning).
4. Safety, contraindications, or harms when relevant.
5. Format / structure faithful to what the prompt explicitly requested.
6. Calibration: presence of warranted uncertainty, absence of unwarranted
   certainty.

AVOID:
- Generic writing advice ("is well-structured", "is clear", "is well-written").
- Vague qualitative claims ("is clinically accurate" without saying about
  what).
- Criteria that depend on style preference rather than substance.
"""


_OUTPUT_FORMAT = """\
Output only rubric tags in this exact format, one per rubric:
<RUBRIC> ... </RUBRIC>
"""


# Task-type-specific framing.  The dispatcher in ``select_initial_rubric_system``
# picks the right one off ``ExampleRecord.task_profile_id``.  All variants stack
# the forced rules + output format above.

_INITIAL_RUBRIC_SYSTEM_NOTE = """You are a rubric designer for an LLM-as-a-judge system that evaluates clinical notes.

Design rubrics that distinguish stronger and weaker clinical notes for the given transcript or task.
Each rubric should be clinically grounded in the documentation task at hand.
"""


_INITIAL_RUBRIC_SYSTEM_QA = """You are a rubric designer for an LLM-as-a-judge system that evaluates answers to medical questions.

The task is question-answering, not note-writing.  Design rubrics that distinguish a correct,
well-reasoned answer from a wrong, hand-wavy, or incomplete one.  Focus first on whether the
answer reaches the correct conclusion; secondarily on whether the supporting reasoning is sound.
"""


_INITIAL_RUBRIC_SYSTEM_AGENTIC = """You are a rubric designer for an LLM-as-a-judge system that evaluates outputs of an agentic medical workflow.

The task is producing the artifact a workflow step asks for (e.g. a triage decision, an action
plan, a tool-call rationale).  Design rubrics that distinguish a complete, faithful, actionable
output from one that fabricates state, drops required steps, or misrepresents tool results.
"""


_INITIAL_RUBRIC_SYSTEM_DOCUMENTATION = """You are a rubric designer for an LLM-as-a-judge system that evaluates a requested clinical documentation artifact.

The task is producing a specific documentation artifact (a discharge summary, problem list,
patient instructions, etc.) -- not arbitrary free text.  Design rubrics that distinguish an
artifact that matches the requested style and content from one that is generic, off-format, or
missing required sections.
"""


_INITIAL_RUBRIC_SYSTEM_REWRITE = """You are a rubric designer for an LLM-as-a-judge system that evaluates a rewrite or edit task.

The task is rewriting or editing an existing text per an explicit instruction.  Design rubrics
that distinguish a faithful, instruction-following rewrite from one that drifts in meaning,
adds unsupported facts, or fails to apply the requested change.
"""


_INITIAL_RUBRIC_SYSTEM_CDS = """You are a rubric designer for an LLM-as-a-judge system that evaluates a clinical decision-support output.

The task is providing decision support grounded in the clinical context (e.g. recommended next
step, risk assessment, options comparison).  Design rubrics that distinguish a recommendation
that is correct, justified by the evidence in the prompt, and safe, from one that is wrong,
unsupported, or risky.
"""


def _compose_system(framing: str) -> str:
    return framing.strip() + "\n\n" + _FORCED_RUBRIC_RULES + "\n" + _OUTPUT_FORMAT


# Public symbols.  ``INITIAL_RUBRIC_SYSTEM`` is kept (with the new forced
# rules + clinical-note framing) so external callers and tests that import the
# name still resolve.  ``INITIAL_RUBRIC_SYSTEM_PROMPT_ONLY`` likewise.
INITIAL_RUBRIC_SYSTEM = _compose_system(_INITIAL_RUBRIC_SYSTEM_NOTE)
INITIAL_RUBRIC_SYSTEM_QA = _compose_system(_INITIAL_RUBRIC_SYSTEM_QA)
INITIAL_RUBRIC_SYSTEM_AGENTIC = _compose_system(_INITIAL_RUBRIC_SYSTEM_AGENTIC)
INITIAL_RUBRIC_SYSTEM_DOCUMENTATION = _compose_system(_INITIAL_RUBRIC_SYSTEM_DOCUMENTATION)
INITIAL_RUBRIC_SYSTEM_REWRITE = _compose_system(_INITIAL_RUBRIC_SYSTEM_REWRITE)
INITIAL_RUBRIC_SYSTEM_CDS = _compose_system(_INITIAL_RUBRIC_SYSTEM_CDS)
INITIAL_RUBRIC_SYSTEM_PROMPT_ONLY = _compose_system(
    "You are a rubric designer for an LLM-as-a-judge system.\n\n"
    "Design a comprehensive set of prompt-specific rubrics for evaluating responses to the given\n"
    "task.  Tailor the rubrics to the prompt; avoid generic writing advice."
)


# Dispatcher off ExampleRecord.task_profile_id.  Falls back to the QA framing
# (the most common shard-0 family at 88%) when the task profile is unknown,
# rather than the original clinical-note framing -- because the medical_o1
# corpus is dominated by Q&A, not free-form notes, which is why the
# "the note correctly identifies" template was so over-used.
_TASK_PROMPT_DISPATCH = {
    "general_instruction_following": INITIAL_RUBRIC_SYSTEM_QA,
    "clinical_decision_support": INITIAL_RUBRIC_SYSTEM_CDS,
    "agentic_workflows": INITIAL_RUBRIC_SYSTEM_AGENTIC,
    "documentation_variants": INITIAL_RUBRIC_SYSTEM_DOCUMENTATION,
    "rewrite_editing": INITIAL_RUBRIC_SYSTEM_REWRITE,
    "note_documentation": INITIAL_RUBRIC_SYSTEM,
}


def select_initial_rubric_system(example: ExampleRecord, include_responses: bool = True) -> str:
    """Pick the initial-rubric system prompt based on task type.

    Falls through to the prompt-only variant when ``include_responses`` is
    False (used by the one-shot baseline path).  Otherwise dispatches off
    ``example.task_profile_id``; unknown profiles fall back to the Q&A
    variant, which is the modal shard-0 family.
    """
    if not include_responses:
        return INITIAL_RUBRIC_SYSTEM_PROMPT_ONLY
    profile = (example.task_profile_id or "").strip().lower()
    return _TASK_PROMPT_DISPATCH.get(profile, INITIAL_RUBRIC_SYSTEM_QA)


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
