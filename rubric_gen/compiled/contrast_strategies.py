from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.mutations import (
    MUTATION_IDS as NOTE_MUTATION_IDS,
    build_contrast_candidates as build_note_contrast_candidates,
    is_synthetic_candidate as is_note_synthetic_candidate,
)
from rubric_gen.compiled.task_profiles import TaskProfile, get_task_profile, resolve_task_profile
from rubric_gen.dataio import strongest_anchor_text
from rubric_gen.types import CandidateNote, ExampleRecord


@dataclass(frozen=True)
class ContrastStrategy:
    strategy_id: str
    mutation_ids: Tuple[str, ...]
    mutation_grounding_profiles: Mapping[str, Mapping[str, Any]]


_ACTION_PATTERNS = re.compile(
    r"\b(next\s+steps?|plan|plans|recommend(?:ation|ations)?|follow[\s-]*up|"
    r"return\s+if|return\s+to|call\s+if|schedule|book|arrange|continue|start|stop)\b",
    re.IGNORECASE,
)
_EVIDENCE_PATTERNS = re.compile(
    r"\b(because|due\s+to|based\s+on|supported\s+by|evidence|reason(?:ing)?|"
    r"rationale|as\s+shown|findings?)\b",
    re.IGNORECASE,
)
_FORMAT_PATTERNS = re.compile(
    r"(?m)^\s*(#{1,6}\s+|[-*]\s+|\d+\.\s+|[A-Z][A-Z /_-]{2,}:|\{|\[)",
    re.IGNORECASE,
)
_CONSTRAINT_PATTERNS = re.compile(
    r"\b(must|should|exactly|only|under\s+\d+|limit(?:ed)?\s+to|do\s+not|"
    r"without|preserve|keep|maintain)\b",
    re.IGNORECASE,
)
_STEP_PATTERNS = re.compile(
    r"(?m)^\s*(step\s+\d+[:.)-]?|\d+\.\s+|observation:|action:|result:)",
    re.IGNORECASE,
)
_TOOL_RESULT_PATTERNS = re.compile(
    r"\b(tool|search|query|returned|result|results|observation|stdout|stderr|"
    r"exit code|found that)\b",
    re.IGNORECASE,
)
_VERIFICATION_PATTERNS = re.compile(
    r"\b(verify|verified|double-check|double checked|confirmed|validation|checked)\b",
    re.IGNORECASE,
)
_FAILURE_PATTERNS = re.compile(
    r"\b(failed|failure|unable|could not|couldn't|blocked|retry|fallback|error)\b",
    re.IGNORECASE,
)
_FINAL_ANSWER_PATTERNS = re.compile(
    r"\b(final answer|in summary|done|completed|next action|deliverable)\b",
    re.IGNORECASE,
)
_REWRITE_PATTERNS = re.compile(
    r"\b(clear|clarity|concise|formal|polished|rewrite|rephrase|grammar|tone|style)\b",
    re.IGNORECASE,
)
_REPEATED_CHOICE_RE = re.compile(r"\b([A-J])\1{0,4}\b", re.IGNORECASE)
_STAR_WRAPPED_ANSWER_RE = re.compile(r"\*\*\*([^*]+)\*\*\*")
_BOLD_ANSWER_RE = re.compile(r"\*\*([^*]+)\*\*")
_FINAL_ANSWER_LINE_RE = re.compile(r"(?im)^(?:final answer|answer)\s*[:\-]\s*(.+)$")
_INTEGER_RE = re.compile(r"-?\d+")
_BOOLEAN_RE = re.compile(r"\b(?:yes|no|true|false)\b", re.IGNORECASE)


def _drop_matching_lines(text: str, pattern: re.Pattern[str]) -> str:
    if not text.strip():
        return text
    kept: List[str] = []
    dropped = False
    for line in text.splitlines():
        if pattern.search(line):
            dropped = True
            continue
        kept.append(line)
    if not dropped:
        return text
    out = "\n".join(kept).strip()
    return out or text


def _flatten_document_scaffold(text: str) -> str:
    if not text.strip():
        return text
    out = re.sub(r"(?m)^\s*(#{1,6}\s+|[-*]\s+|\d+\.\s+)", "", text)
    out = re.sub(r"(?m)^\s*[A-Z][A-Z /_-]{2,}:\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()
    return out or text


def _drop_action_items(text: str) -> str:
    return _drop_matching_lines(text, _ACTION_PATTERNS)


def _drop_supporting_evidence(text: str) -> str:
    return _drop_matching_lines(text, _EVIDENCE_PATTERNS)


def _remove_format_markers(text: str) -> str:
    if not text.strip():
        return text
    out = re.sub(r"(?m)^\s*(#{1,6}\s+|[-*]\s+|\d+\.\s+)", "", text)
    out = out.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    out = out.strip()
    return out or text


def _drop_constraints(text: str) -> str:
    return _drop_matching_lines(text, _CONSTRAINT_PATTERNS)


def _drop_steps(text: str) -> str:
    return _drop_matching_lines(text, _STEP_PATTERNS)


def _drop_tool_results(text: str) -> str:
    return _drop_matching_lines(text, _TOOL_RESULT_PATTERNS)


def _drop_verification(text: str) -> str:
    return _drop_matching_lines(text, _VERIFICATION_PATTERNS)


def _drop_failure_handling(text: str) -> str:
    return _drop_matching_lines(text, _FAILURE_PATTERNS)


def _drop_final_answer(text: str) -> str:
    return _drop_matching_lines(text, _FINAL_ANSWER_PATTERNS)


def _drop_rewrite_signal(text: str) -> str:
    return _drop_matching_lines(text, _REWRITE_PATTERNS)


def _introduce_typos(text: str) -> str:
    out = text
    replacements = (
        (" the ", " teh "),
        (" and ", " adn "),
        (" with ", " wtih "),
        (" patient ", " paitent "),
    )
    changed = False
    for src, dst in replacements:
        if src in out:
            out = out.replace(src, dst, 1)
            changed = True
    if not changed:
        out = (out.rstrip() + "\n\nteh").strip()
    return out


def _add_unsupported_detail(text: str) -> str:
    suffix = "\n\nAdditional unsupported claim: This is definitively proven by the available information."
    return (text.rstrip() + suffix).strip()


def _inflate_certainty_generic(text: str) -> str:
    suffix = "\n\nConclusion: The answer is definitively correct and requires no further uncertainty handling."
    return (text.rstrip() + suffix).strip()


def _rotate_choice_token(token: str) -> str:
    upper = token.upper()
    if not upper:
        return token
    first = upper[0]
    if first < "A" or first > "J":
        return token
    rotated = chr(((ord(first) - ord("A") + 1) % 10) + ord("A"))
    if len(set(upper)) == 1:
        replacement = rotated * len(token)
    else:
        replacement = rotated
    return replacement if token.isupper() else replacement.lower()


def _last_nonempty_line(text: str) -> str:
    nonempty = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return nonempty[-1] if nonempty else ""


def _corrupt_final_answer_token(token: str) -> str:
    stripped = token.strip()
    if not stripped:
        return "UNKNOWN"
    if _REPEATED_CHOICE_RE.fullmatch(stripped):
        return _rotate_choice_token(stripped)
    if re.fullmatch(r"[A-Ja-j]", stripped):
        return _rotate_choice_token(stripped)
    lowered = stripped.lower()
    if lowered == "yes":
        return "no"
    if lowered == "no":
        return "yes"
    if lowered == "true":
        return "false"
    if lowered == "false":
        return "true"
    if _INTEGER_RE.fullmatch(stripped):
        value = int(stripped)
        return str(value + 1 if value >= 0 else value - 1)
    return "UNKNOWN"


def _replace_last_supported_token(text: str) -> str:
    patterns = (
        _REPEATED_CHOICE_RE,
        re.compile(r"\b[A-Ja-j]\b"),
        _INTEGER_RE,
        _BOOLEAN_RE,
    )
    for pattern in patterns:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        match = matches[-1]
        replacement = _corrupt_final_answer_token(match.group(0))
        return text[: match.start()] + replacement + text[match.end() :]
    return text


def _corrupt_final_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text

    replacement = _corrupt_final_answer_token(stripped)
    if replacement != "UNKNOWN":
        return replacement

    star_matches = list(_STAR_WRAPPED_ANSWER_RE.finditer(text))
    if star_matches:
        star_match = star_matches[-1]
        replacement = _corrupt_final_answer_token(star_match.group(1).strip())
        return text[: star_match.start(1)] + replacement + text[star_match.end(1) :]

    bold_matches = list(_BOLD_ANSWER_RE.finditer(text))
    if bold_matches:
        bold_match = bold_matches[-1]
        replacement = _corrupt_final_answer_token(bold_match.group(1).strip())
        return text[: bold_match.start(1)] + replacement + text[bold_match.end(1) :]

    final_answer_matches = list(_FINAL_ANSWER_LINE_RE.finditer(text))
    if final_answer_matches:
        final_answer_match = final_answer_matches[-1]
        replaced_line = _replace_last_supported_token(final_answer_match.group(1))
        if replaced_line != final_answer_match.group(1):
            return (
                text[: final_answer_match.start(1)]
                + replaced_line
                + text[final_answer_match.end(1) :]
            )

    last_line = _last_nonempty_line(text)
    if last_line:
        replaced_last_line = _replace_last_supported_token(last_line)
        if replaced_last_line != last_line:
            last_line_index = text.rfind(last_line)
            if last_line_index >= 0:
                return text[:last_line_index] + replaced_last_line + text[last_line_index + len(last_line):]

    suffix = "\n\nFinal answer: UNKNOWN"
    return (text.rstrip() + suffix).strip()


def _code_off_by_one_loop(text: str) -> str:
    if not text.strip():
        return text
    out, count = re.subn(
        r"range\(([^,\n()]+)\)",
        lambda match: f"range(max(0, ({match.group(1).strip()}) - 1))",
        text,
        count=1,
    )
    if count:
        return out
    out, count = re.subn(r"<=\s*([A-Za-z_][A-Za-z0-9_]*)", r"< \1", text, count=1)
    if count:
        return out
    return text


def _code_flip_condition_branch(text: str) -> str:
    if not text.strip():
        return text
    replacements = (
        (r"==", "!="),
        (r">=", ">"),
        (r"<=", "<"),
        (r"\bis None\b", "is not None"),
        (r"\bis not None\b", "is None"),
    )
    for pattern, replacement in replacements:
        out, count = re.subn(pattern, replacement, text, count=1)
        if count:
            return out
    return text


def _code_corrupt_input_parsing(text: str) -> str:
    if not text.strip():
        return text
    replacements = (
        ("input().split()", "input()"),
        ("sys.stdin.read().split()", "sys.stdin.read().splitlines()"),
        ("sys.stdin.read()", "sys.stdin.readline()"),
        (".split()", ".splitlines()"),
    )
    for src, dst in replacements:
        if src in text:
            return text.replace(src, dst, 1)
    return text


def _code_drop_negative_directions(text: str) -> str:
    if not text.strip():
        return text
    out = text.replace("(-1, 0, 1)", "(0, 1)", 1)
    if out != text:
        return out
    kept: List[str] = []
    dropped = 0
    for line in text.splitlines():
        normalized = line.replace(" ", "")
        if dropped < 2 and any(
            token in normalized
            for token in ("(-1,", ",-1)", "(1,-1)", "(-1,1)", "(-1,-1)")
        ):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        return "\n".join(kept).strip() or text
    return text


def _code_remove_zero_guard(text: str) -> str:
    if not text.strip():
        return text
    kept: List[str] = []
    removed_guard = False
    skip_return_false = False
    for line in text.splitlines():
        normalized = line.lower().replace(" ", "")
        if not removed_guard and any(
            token in normalized
            for token in ("'0'in", '"0"in', "==0", "count('0')", 'count("0")')
        ):
            removed_guard = True
            skip_return_false = True
            continue
        if skip_return_false and line.lstrip().startswith("return False"):
            skip_return_false = False
            continue
        skip_return_false = False
        kept.append(line)
    if removed_guard:
        return "\n".join(kept).strip() or text
    return text


def _code_drop_reset_logic(text: str) -> str:
    if not text.strip():
        return text
    out, count = re.subn(
        r"(?m)^\s*(?:cur|curr|current|running|window|count|start|left|right|l|r)\s*=\s*0\s*(?:#.*)?$",
        "",
        text,
        count=1,
    )
    if count:
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out or text
    return text


_GENERIC_MUTATION_FUNCS: Dict[str, Callable[[str], str]] = {
    "flatten_document_scaffold": _flatten_document_scaffold,
    "drop_action_items": _drop_action_items,
    "drop_supporting_evidence": _drop_supporting_evidence,
    "remove_format_markers": _remove_format_markers,
    "drop_constraints": _drop_constraints,
    "drop_steps": _drop_steps,
    "drop_tool_results": _drop_tool_results,
    "drop_verification": _drop_verification,
    "drop_failure_handling": _drop_failure_handling,
    "drop_final_answer": _drop_final_answer,
    "drop_rewrite_signal": _drop_rewrite_signal,
    "introduce_typos": _introduce_typos,
    "add_unsupported_detail": _add_unsupported_detail,
    "inflate_certainty_generic": _inflate_certainty_generic,
    "corrupt_final_answer": _corrupt_final_answer,
    "code_off_by_one_loop": _code_off_by_one_loop,
    "code_flip_condition_branch": _code_flip_condition_branch,
    "code_corrupt_input_parsing": _code_corrupt_input_parsing,
    "code_drop_negative_directions": _code_drop_negative_directions,
    "code_remove_zero_guard": _code_remove_zero_guard,
    "code_drop_reset_logic": _code_drop_reset_logic,
}


_DOCUMENTATION_MUTATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "flatten_document_scaffold": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker artifact mainly loses section structure or formatting scaffold.",
        "keywords": ("section", "header", "structure", "scaffold", "format", "layout"),
    },
    "drop_action_items": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker artifact mainly drops action items, follow-up steps, or recommendations.",
        "keywords": ("plan", "next step", "follow-up", "follow up", "recommendation", "action item"),
        "allowed_phrases": ("plan", "next step", "follow-up", "recommendation", "action"),
        "blocked_phrases": ("grammar", "tone", "style"),
    },
    "drop_supporting_evidence": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker artifact mainly drops the reasoning or evidence supporting its claims.",
        "keywords": ("because", "due to", "based on", "evidence", "reasoning", "rationale"),
    },
    "inflate_certainty_generic": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker artifact mainly adds unsupported certainty language.",
        "keywords": ("certainty", "overconfident", "unsupported", "definitive"),
    },
}

_REWRITE_MUTATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "remove_format_markers": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker rewrite mainly loses the requested structure or format cues.",
        "keywords": ("format", "structure", "bullet", "heading", "layout"),
    },
    "drop_rewrite_signal": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker rewrite mainly fails to carry through the requested editing transformation.",
        "keywords": ("rewrite", "rephrase", "tone", "style", "clarity", "grammar"),
    },
    "introduce_typos": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker rewrite mainly introduces grammar or spelling errors.",
        "keywords": ("grammar", "spelling", "clarity", "readability"),
    },
    "add_unsupported_detail": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker rewrite mainly adds unsupported content that was not present in the source.",
        "keywords": ("fabrication", "unsupported", "added content", "preserve meaning"),
    },
}

_DECISION_SUPPORT_MUTATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "drop_action_items": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker response mainly drops recommended next steps or management actions.",
        "keywords": ("recommendation", "management", "next step", "plan", "follow-up"),
    },
    "drop_supporting_evidence": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker response mainly drops evidence-based justification for the recommendation.",
        "keywords": ("because", "reasoning", "evidence", "supported by", "based on"),
    },
    "drop_constraints": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker response mainly drops safety constraints or stated limitations.",
        "keywords": ("safety", "warning", "monitor", "contraindication", "limitation"),
    },
    "inflate_certainty_generic": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly overstates certainty beyond the available evidence.",
        "keywords": ("certainty", "definitive", "unsupported", "guaranteed"),
    },
}

_INSTRUCTION_MUTATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "remove_format_markers": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker response mainly loses the requested formatting or output structure.",
        "keywords": ("format", "bullet", "json", "table", "heading", "final answer", "bold", "triple asterisks"),
    },
    "drop_constraints": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker response mainly drops explicit constraints from the instruction.",
        "keywords": ("constraint", "must", "only", "do not", "instruction", "clue", "condition", "logic"),
    },
    "drop_supporting_evidence": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker response mainly drops evidence or rationale that grounded the answer.",
        "keywords": ("grounded", "evidence", "reasoning", "because", "source", "verification", "derivation", "consistency"),
    },
    "add_unsupported_detail": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly adds unsupported detail outside the provided context.",
        "keywords": ("unsupported", "fabricated", "made up", "invented"),
    },
    "corrupt_final_answer": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly gives the wrong final answer or violates the requested final-answer format.",
        "keywords": (
            "final answer",
            "answer format",
            "single word",
            "single digit",
            "output format",
            "correct option",
            "correct letter",
            "correct digit",
            "final conclusion",
            "yes/no list",
            "repeated letter",
        ),
    },
    "code_off_by_one_loop": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly introduces an off-by-one loop or boundary bug.",
        "keywords": ("range", "loop", "boundary", "index", "off-by-one", "last element", "iteration"),
    },
    "code_flip_condition_branch": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly flips a comparison or branch condition and therefore chooses the wrong path.",
        "keywords": ("condition", "branch", "comparison", "threshold", "equality", "inequality", "reset"),
    },
    "code_corrupt_input_parsing": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly reads or tokenizes the input incorrectly.",
        "keywords": ("input", "parse", "token", "split", "stdin", "read", "line", "whitespace"),
    },
    "code_drop_negative_directions": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly misses directions, neighbors, or movement cases that require negative deltas.",
        "keywords": ("direction", "neighbor", "diagonal", "up", "left", "negative delta", "all 8 directions"),
    },
    "code_remove_zero_guard": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly fails to reject zero or other explicitly forbidden values.",
        "keywords": ("zero", "digit", "forbidden value", "invalid", "guard", "reject"),
    },
    "code_drop_reset_logic": {
        "delta_mode": "weak_only",
        "prompt_hint": "The weaker response mainly forgets to reset state when a segment or window becomes invalid.",
        "keywords": ("reset", "restart", "window", "segment", "running state", "subarray"),
    },
}

_AGENTIC_MUTATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "drop_steps": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker workflow mainly drops execution steps or ordered progress markers.",
        "keywords": ("step", "sequence", "workflow", "completion"),
    },
    "drop_tool_results": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker workflow mainly drops evidence from tool outputs or observed results.",
        "keywords": ("tool result", "observation", "stdout", "stderr", "search result"),
    },
    "drop_verification": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker workflow mainly drops verification or validation of the final state.",
        "keywords": ("verify", "validated", "checked", "confirmed"),
    },
    "drop_failure_handling": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker workflow mainly drops blockers, retries, or failure handling.",
        "keywords": ("failed", "retry", "blocked", "fallback", "error"),
    },
    "drop_final_answer": {
        "delta_mode": "strong_only",
        "prompt_hint": "The weaker workflow mainly drops the final deliverable or completion summary.",
        "keywords": ("final answer", "deliverable", "completed", "summary"),
    },
}

_STRATEGIES: Dict[str, ContrastStrategy] = {
    "note_documentation": ContrastStrategy(
        strategy_id="note_documentation",
        mutation_ids=NOTE_MUTATION_IDS,
        mutation_grounding_profiles={},
    ),
    "documentation_variants": ContrastStrategy(
        strategy_id="documentation_variants",
        mutation_ids=("flatten_document_scaffold", "drop_action_items", "drop_supporting_evidence", "inflate_certainty_generic"),
        mutation_grounding_profiles=_DOCUMENTATION_MUTATION_PROFILES,
    ),
    "rewrite_editing": ContrastStrategy(
        strategy_id="rewrite_editing",
        mutation_ids=("remove_format_markers", "drop_rewrite_signal", "introduce_typos", "add_unsupported_detail"),
        mutation_grounding_profiles=_REWRITE_MUTATION_PROFILES,
    ),
    "clinical_decision_support": ContrastStrategy(
        strategy_id="clinical_decision_support",
        mutation_ids=("drop_action_items", "drop_supporting_evidence", "drop_constraints", "inflate_certainty_generic"),
        mutation_grounding_profiles=_DECISION_SUPPORT_MUTATION_PROFILES,
    ),
    "general_instruction_following": ContrastStrategy(
        strategy_id="general_instruction_following",
        mutation_ids=("remove_format_markers", "drop_constraints", "drop_supporting_evidence", "add_unsupported_detail"),
        mutation_grounding_profiles=_INSTRUCTION_MUTATION_PROFILES,
    ),
    "agentic_workflows": ContrastStrategy(
        strategy_id="agentic_workflows",
        mutation_ids=("drop_steps", "drop_tool_results", "drop_verification", "drop_failure_handling", "drop_final_answer"),
        mutation_grounding_profiles=_AGENTIC_MUTATION_PROFILES,
    ),
}

_DYNAMIC_STRATEGIES: Dict[str, ContrastStrategy] = {}


def get_contrast_strategy(strategy_id: str | None) -> ContrastStrategy:
    sid = (strategy_id or "general_instruction_following").strip() or "general_instruction_following"
    if sid in _STRATEGIES:
        return _STRATEGIES[sid]
    if sid in _DYNAMIC_STRATEGIES:
        return _DYNAMIC_STRATEGIES[sid]
    return _STRATEGIES["general_instruction_following"]


def register_contrast_strategy(strategy: ContrastStrategy) -> ContrastStrategy:
    _DYNAMIC_STRATEGIES[strategy.strategy_id] = strategy
    return strategy


def clear_dynamic_contrast_strategies() -> None:
    _DYNAMIC_STRATEGIES.clear()


def mutation_function_for_id(mutation_id: str) -> Optional[Callable[[str], str]]:
    return _GENERIC_MUTATION_FUNCS.get(mutation_id)


def generic_mutation_catalog() -> Mapping[str, Callable[[str], str]]:
    return dict(_GENERIC_MUTATION_FUNCS)


def mutation_grounding_profiles_for_profile(task_profile_id: str) -> Mapping[str, Mapping[str, Any]]:
    if task_profile_id == "note_documentation":
        return {}
    profile = get_task_profile(task_profile_id)
    return get_contrast_strategy(profile.contrast_strategy_id).mutation_grounding_profiles


def _original_candidate(
    example: ExampleRecord,
    *,
    field_name: str,
    text: str,
    profile: TaskProfile,
) -> CandidateNote:
    safe = example.example_id.replace("/", "_")
    return CandidateNote(
        candidate_id=f"{safe}__{field_name}",
        example_id=example.example_id,
        text=text,
        source_label=field_name,
        quality_bucket=f"dataset_{profile.artifact_kind}",
        origin_kind="original",
        metadata={"synthetic": False, "artifact_field": field_name},
        artifact_kind=example.artifact_kind or profile.artifact_kind,
        task_profile_id=profile.task_profile_id,
        task_family_id=example.task_family_id,
    )


def _synthetic_candidate(
    example: ExampleRecord,
    *,
    mutation_id: str,
    text: str,
    anchor_preview: str,
    profile: TaskProfile,
) -> CandidateNote:
    safe = example.example_id.replace("/", "_")
    return CandidateNote(
        candidate_id=f"{safe}__mut__{mutation_id}",
        example_id=example.example_id,
        text=text,
        source_label=f"synthetic_mutation:{mutation_id}",
        quality_bucket="synthetic_contrast",
        origin_kind="synthetic_mutation",
        metadata={
            "synthetic": True,
            "mutation_id": mutation_id,
            "anchor_char_len": len(anchor_preview),
        },
        artifact_kind=example.artifact_kind or profile.artifact_kind,
        task_profile_id=profile.task_profile_id,
        task_family_id=example.task_family_id,
    )


def _generic_original_candidates(example: ExampleRecord, profile: TaskProfile) -> List[CandidateNote]:
    out: List[CandidateNote] = []
    seen: set[str] = set()
    fields = (
        ("reference_artifact", example.reference_artifact),
        ("augmented_artifact", example.augmented_artifact),
        ("artifact_truncated", example.artifact_truncated),
        ("reference_note", example.reference_note),
        ("augmented_note", example.augmented_note),
        ("note_truncated", example.note_truncated),
    )
    for field_name, text in fields:
        cleaned = (text or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(_original_candidate(example, field_name=field_name, text=text, profile=profile))
    return out


def _build_generic_candidates(
    example: ExampleRecord,
    *,
    profile: TaskProfile,
    mutation_ids: Sequence[str] | None = None,
) -> List[CandidateNote]:
    out = _generic_original_candidates(example, profile)
    anchor = strongest_anchor_text(example)
    if not anchor.strip():
        return out

    seen_text = {candidate.text.strip() for candidate in out if candidate.text.strip()}
    strategy = get_contrast_strategy(profile.contrast_strategy_id)
    mids = tuple(mutation_ids) if mutation_ids is not None else strategy.mutation_ids

    for mutation_id in mids:
        fn = _GENERIC_MUTATION_FUNCS.get(mutation_id)
        if fn is None:
            continue
        mutated = fn(anchor)
        if not mutated.strip():
            continue
        if mutated.strip() == anchor.strip():
            continue
        if mutated.strip() in seen_text:
            continue
        seen_text.add(mutated.strip())
        out.append(
            _synthetic_candidate(
                example,
                mutation_id=mutation_id,
                text=mutated,
                anchor_preview=anchor[:200],
                profile=profile,
            )
        )
    return out


def build_task_contrast_candidates(
    example: ExampleRecord,
    *,
    task_profile_id: Optional[str] = None,
    mutation_ids: Sequence[str] | None = None,
) -> List[CandidateNote]:
    profile = resolve_task_profile(example, explicit=task_profile_id)
    if profile.task_profile_id == "note_documentation":
        return build_note_contrast_candidates(example, mutation_ids=mutation_ids)
    return _build_generic_candidates(example, profile=profile, mutation_ids=mutation_ids)


def is_synthetic_candidate(candidate: CandidateNote) -> bool:
    if candidate.task_profile_id == "note_documentation" or (
        candidate.metadata.get("artifact_field") in {"reference_note", "augmented_note", "note_truncated"}
    ):
        return is_note_synthetic_candidate(candidate)
    return bool(candidate.metadata.get("synthetic")) or candidate.origin_kind == "synthetic_mutation"
