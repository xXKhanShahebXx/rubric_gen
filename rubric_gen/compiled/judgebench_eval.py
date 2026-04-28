"""
JudgeBench evaluation for the compiled recursive rubric discovery pipeline.

This module joins local JudgeBench subset files to the official pairwise benchmark rows by `pair_id`,
learns a frozen routing/granularity policy on the 80-example design split, and evaluates that frozen
policy on the 270-example validation split.
"""

from __future__ import annotations

import copy
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.contrast_strategies import (
    ContrastStrategy,
    clear_dynamic_contrast_strategies,
    get_contrast_strategy,
    mutation_function_for_id,
    register_contrast_strategy,
)
from rubric_gen.compiled.discovery import (
    RecursiveDiscoveryConfig,
    build_pair_discriminator_prompts,
    discover_pair_criteria,
    merge_proposal_entries,
    merge_proposal_entries_with_rrd_filters,
)
from rubric_gen.compiled.rrd_filters import PairContext as _RrdPairContext
from rubric_gen.compiled.llm_judge import resolve_compiled_judge_spec
from rubric_gen.compiled.profile_bootstrap import resolve_or_bootstrap_task_profile
from rubric_gen.compiled.serialize import to_json_dict, write_json
from rubric_gen.compiled.task_profiles import (
    TaskProfile,
    clear_dynamic_task_profiles,
    get_task_profile,
    register_task_profile,
)
from rubric_gen.compiled.judgebench_verifiers import (
    JudgeBenchVerifierCandidateSignal,
    JudgeBenchVerifierOutcome,
    evaluate_pair_verifier,
)
from rubric_gen.compiled.reasoning_process_verifier import (
    ReasoningProcessVerifier,
    ReasoningProcessVerifierConfig,
)
from rubric_gen.compiled.math_independent_solver_verifier import (
    evaluate_math_independent_solver,
)
from rubric_gen.compiled.code_execution_verifier import (
    evaluate_code_pair_verifier,
)
from rubric_gen.compiled.leetcode_test_runner import (
    evaluate_leetcode_pair_verifier,
)
from rubric_gen.compiled.mmlu_independent_answerer_verifier import (
    evaluate_mmlu_independent_answerer,
)
from rubric_gen.compiled.reasoning_independent_solver_verifier import (
    evaluate_reasoning_independent_solver,
)
from rubric_gen.compiled.holistic_judge import (
    apply_holistic_judge_to_scoring,
    run_holistic_pair_judge,
)
from rubric_gen.compiled.rubric_library import (
    RubricLibrary,
    load_rubric_library,
    maybe_load_default_library,
)
from rubric_gen.config import discover_default_comparison_judge_model, parse_model_spec
from rubric_gen.llm_client import LLMRouter, extract_json_object, parse_yes_no
from rubric_gen.rrd.weighting import compute_uniform_weights, compute_whitened_uniform_weights
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import CandidateNote, ExampleRecord, ModelSpec, RubricCriterion, RubricEvaluation


JUDGEBENCH_GPT4O_URL = (
    "https://raw.githubusercontent.com/ScalerLab/JudgeBench/main/"
    "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
)

EVALUATION_RE = re.compile(r"<EVALUATION>\s*(YES|NO)\s*</EVALUATION>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_SAFE_ID_RE = re.compile(r"[^a-z0-9]+")

_BUILT_IN_SUBSET_PREFIXES: Mapping[str, str] = {
    "mmlu-pro": "mmlu-pro",
    "livebench-reasoning": "livebench-reasoning",
    "livebench-math": "livebench-math",
    "livecodebench": "livecodebench",
}

_BLIND_SOURCE_FAMILY_SCORING_CANDIDATE_BUDGET: Mapping[str, int] = {
    "livebench-reasoning": 4,
    "livebench-math": 4,
    "mmlu-pro": 6,
    "livecodebench": 6,
}
_BLIND_SCORING_PROFILE_BASELINE = "baseline"
_BLIND_SCORING_PROFILE_PRUNED_V1 = "pruned_v1"
_BLIND_SCORING_PROFILE_PRUNED_V2 = "pruned_v2"
_BLIND_SCORING_PROFILE_PRUNED_DISC_V1 = "pruned_disc_v1"
_ALLOWED_BLIND_SCORING_PROFILES = {
    _BLIND_SCORING_PROFILE_BASELINE,
    _BLIND_SCORING_PROFILE_PRUNED_V1,
    _BLIND_SCORING_PROFILE_PRUNED_V2,
    _BLIND_SCORING_PROFILE_PRUNED_DISC_V1,
}
_BLIND_DISCRIMINATOR_MODE_DEFAULT = "default"
_BLIND_DISCRIMINATOR_MODE_OFF = "off"
_BLIND_DISCRIMINATOR_MODE_STRICT = "strict"
_ALLOWED_BLIND_DISCRIMINATOR_FAMILY_MODES = {
    _BLIND_DISCRIMINATOR_MODE_DEFAULT,
    _BLIND_DISCRIMINATOR_MODE_OFF,
    _BLIND_DISCRIMINATOR_MODE_STRICT,
}
_BLIND_BUDGET_PROFILE_FAMILY_V1 = "family_v1"
_BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V1 = "family_profile_v1"
_BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V2 = "family_profile_v2"
_ALLOWED_BLIND_BUDGET_PROFILES = {
    _BLIND_BUDGET_PROFILE_FAMILY_V1,
    _BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V1,
    _BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V2,
}
_BLIND_GUIDANCE_PROFILE_OFF = "off"
_BLIND_GUIDANCE_PROFILE_FAMILY_V1 = "family_v1"
_BLIND_GUIDANCE_PROFILE_FAMILY_V2 = "family_v2"
_ALLOWED_BLIND_GUIDANCE_PROFILES = {
    _BLIND_GUIDANCE_PROFILE_OFF,
    _BLIND_GUIDANCE_PROFILE_FAMILY_V1,
    _BLIND_GUIDANCE_PROFILE_FAMILY_V2,
}
_BLIND_WU_PROFILE_RAW = "raw"
_BLIND_WU_PROFILE_STABLE_V1 = "stable_v1"
_ALLOWED_BLIND_WU_PROFILES = {
    _BLIND_WU_PROFILE_RAW,
    _BLIND_WU_PROFILE_STABLE_V1,
}
_RETRIEVAL_PROFILE_OFF = "off"
_RETRIEVAL_PROFILE_FAMILY_QUESTION_V1 = "family_question_v1"
_RETRIEVAL_PROFILE_FAMILY_QUESTION_V2 = "family_question_v2"
_RETRIEVAL_PROFILE_FAMILY_QUESTION_SEED_V1 = "family_question_seed_v1"
_RETRIEVAL_PROFILE_LIBRARY_V1 = "library_v1"
_RETRIEVAL_PROFILE_LIBRARY_V1_PLUS_FAMILY_V1 = "library_v1_plus_family_v1"
_ALLOWED_RETRIEVAL_PROFILES = {
    _RETRIEVAL_PROFILE_OFF,
    _RETRIEVAL_PROFILE_FAMILY_QUESTION_V1,
    _RETRIEVAL_PROFILE_FAMILY_QUESTION_V2,
    _RETRIEVAL_PROFILE_FAMILY_QUESTION_SEED_V1,
    _RETRIEVAL_PROFILE_LIBRARY_V1,
    _RETRIEVAL_PROFILE_LIBRARY_V1_PLUS_FAMILY_V1,
}

_BROAD_DIMENSIONS = {
    "",
    "content_coverage",
    "completeness",
    "instruction_adherence",
    "grounding",
    "format_communication",
}

_FORMAT_PROMPT_NUDGE = (
    "Use dedicated criteria for the exact requested final-answer format when the prompt requires a single word, "
    "single digit, repeated letters, or other rigid output syntax."
)
_MCQ_PROMPT_NUDGE = (
    "Use a dedicated criterion for the final selected option matching the correct answer exactly, not only for "
    "partially correct reasoning."
)
_REASONING_PROMPT_NUDGE = (
    "Prefer narrower criteria for constraint satisfaction, clue consistency, contradiction-free state tracking, and "
    "the exact final deliverable rather than broad completeness checks. Do not let format-only rubrics outweigh a "
    "wrong final conclusion."
)
_CODE_PROMPT_NUDGE = (
    "Prefer criteria about executable correctness, required input/output behavior, edge cases, and respecting "
    "problem constraints. Do not reward implementation-specific details such as memoization decorators, import "
    "choices, or naming style unless they are required by the prompt or clearly change behavior under the stated "
    "constraints."
)
_MATH_PROMPT_NUDGE = (
    "Prefer criteria for mathematical correctness, the final numeric or symbolic answer, and any exact output "
    "format requested by the prompt. Do not let format-only rubrics outweigh a wrong computed value."
)
_FINAL_ANSWER_SUPPORT_PROMPT_NUDGE = (
    "Prefer criteria that connect the final answer to the supporting deductions, rejected alternatives, and "
    "preserved constraints. Do not spend multiple criteria on the same answer-format requirement."
)
_GRANULARITY_PROMPT_NUDGE = (
    "Avoid broad completeness or correctness rubrics. Decompose them into smaller, self-contained checks that each "
    "evaluate one concrete aspect."
)
_BOLD_SPAN_RE = re.compile(r"\*\*([^*]+)\*\*")
_TRIPLE_ASTERISK_SPAN_RE = re.compile(r"\*{3}([^*]+?)\*{3}")
_REPEATED_CHOICE_TOKEN_RE = re.compile(r"([A-Ja-j])\1{0,9}")
_FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^(?:final answer|answer|thus(?:,)?(?: the)? answer is|therefore(?:,)?(?: the)? answer is|i (?:select|choose))\s*[:\-]?\s*(.+)$"
)
_EXACT_CORRECTNESS_HINTS = (
    "correct answer",
    "correct multiple-choice answer",
    "correct multiple choice answer",
    "final answer correctness",
    "matches correct",
    "correct option",
    "correct letter",
    "correct shape",
    "correct numeric",
    "correct value",
    "final answer letter correctness",
    "final answer digit correctness",
    "exact final answer",
    "matches the reference answer",
    "matches the correct final answer",
    "final answer matches",
    "letter correctness",
    "digit correctness",
    "identify the resulting shape",
    "must identify the resulting shape",
    "must identify the correct",
    "must select the correct",
    "select the correct",
    "select the letter",
    "select the digit",
    "corresponding to the calculated",
    "corresponding to the correct",
)
_EXACT_FORMAT_HINTS = (
    "format correctness",
    "output format",
    "single phrase",
    "single string",
    "single word",
    "single digit",
    "single letter",
    "exactly five characters long",
    "all characters in final answer are identical",
    "single uppercase letter",
    "single lowercase letter",
    "repeated-letter answer",
    "repeated letter answer",
    "single contiguous string",
    "single repeated-letter string",
    "bold",
    "double asterisks",
    "repeated",
    "five times",
    "no extra characters",
    "one of the given multiple-choice letters",
    "one of the given multiple choice letters",
    "clearly separated",
    "without unnecessary",
    "without lengthy",
    "direct final answer",
    "conciseness",
)
_CODE_STYLE_HINTS = (
    "lru_cache",
    "memoization",
    "memoize",
    "caching decorator",
    "imports lru_cache",
    "imports functools",
    "variable naming",
    "descriptive variable names",
    "clear function structure",
    "clear structure",
    "readability",
)
_SEVERITY_RANK = {"hard_gate": 3, "high": 2, "medium": 1, "low": 0}
_PAIR_DISCRIMINATOR_PROMPT_VERSION = "judgebench_pair_discriminator_v3"
_ARTIFACT_FINGERPRINT_VERSION = "judgebench_eval_v8"

# v2 self-consistency configuration.
#
# When the policy enables ``self_consistency_n`` (see :func:`_policy_self_consistency_n`), the
# pair discriminator runs ``N`` sampling passes per (A,B) and (B,A) order with temperature > 0
# and aggregates by majority vote. This widens the discriminator from 2 calls to 2N calls on the
# subset of pairs that route to it, and is gated by the margin / whitening-instability logic in
# :func:`_should_run_blind_pair_discriminator`.
_DEFAULT_SELF_CONSISTENCY_N = 1
_DEFAULT_SELF_CONSISTENCY_TEMPERATURE = 0.7
_V2_WIDER_GATE_LOW_MARGIN = 0.02
_V2_WIDER_GATE_REASONING_MARGIN = 0.05
_PROTOCOL_MODE_GENERIC_BASELINE = "generic_baseline"
_PROTOCOL_MODE_JUDGEBENCH_TUNED = "judgebench_tuned"
_DEFAULT_PROTOCOL_MODE = _PROTOCOL_MODE_JUDGEBENCH_TUNED
_ALLOWED_PROTOCOL_MODES = {
    _PROTOCOL_MODE_GENERIC_BASELINE,
    _PROTOCOL_MODE_JUDGEBENCH_TUNED,
}
_RETRIEVAL_WORD_RE = re.compile(r"[a-z0-9]+")
_RETRIEVAL_STOPWORDS = {
    "about",
    "after",
    "again",
    "answer",
    "before",
    "between",
    "choose",
    "correct",
    "count",
    "determine",
    "each",
    "example",
    "final",
    "first",
    "following",
    "from",
    "given",
    "have",
    "into",
    "letter",
    "make",
    "must",
    "only",
    "please",
    "response",
    "return",
    "select",
    "should",
    "single",
    "solve",
    "step",
    "string",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "times",
    "using",
    "what",
    "when",
    "with",
    "would",
    "write",
}
_PRIVATE_ANSWER_KEY_METADATA_KEY = "_private_answer_key_features"


@dataclass
class JudgeBenchLocalExample:
    pair_id: str
    source: str
    question: str
    reference_answer: str
    candidate_models: List[str] = field(default_factory=list)
    verifier_model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeBenchPairRecord:
    pair_id: str
    original_id: str
    source: str
    question: str
    response_model: str
    response_A: str
    response_B: str
    label: str


@dataclass
class JudgeBenchJoinedExample:
    split_name: str
    pair_id: str
    source: str
    source_family: str
    question: str
    reference_answer: str
    response_model: str
    response_A: str
    response_B: str
    label: str
    original_id: str = ""
    candidate_models: List[str] = field(default_factory=list)
    verifier_model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeBenchRouteDecision:
    pair_id: str
    source: str
    source_family: str
    task_profile_id: str
    task_family_id: str
    artifact_kind: str
    route_kind: str
    bootstrap_used: bool
    parent_task_profile_id: str = ""
    prompt_nudges: List[str] = field(default_factory=list)
    recursion_config: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactAnswerExtraction:
    value: str
    source: str
    explicit: bool


@dataclass(frozen=True)
class BlindExactAnswerSignal:
    value: str = ""
    explicit: bool = False
    format_ok: bool = False
    consistent: bool = False
    option_map_available: bool = False
    conflicting_markers: bool = False
    marker_count: int = 0

    def tie_break_key(self) -> Tuple[int, int, int, int]:
        return (
            int(self.consistent),
            int(self.format_ok),
            int(self.explicit),
            -int(self.conflicting_markers),
        )


@dataclass(frozen=True)
class JudgeBenchAnswerKeyFeatures:
    source_family: str = ""
    exact_answer_task: bool = False
    requested_answer_mode: str = ""
    normalized_reference_value: str = ""
    choice_value_map: Dict[str, str] = field(default_factory=dict)
    mcq_option_letters: Tuple[str, ...] = ()
    repeated_choice_length: int = 0
    reference_answer_visible: bool = False


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _safe_slug(value: str) -> str:
    slug = _SAFE_ID_RE.sub("_", (value or "").strip().lower()).strip("_")
    return slug or "unknown"


def _compiled_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_private_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(metadata or {}).items()
        if str(key).strip() and not str(key).startswith("_")
    }


def _question_choice_value_map(question: str) -> Dict[str, str]:
    cleaned = _normalize_question_choice_text(question)
    if not cleaned:
        return {}
    patterns = (
        re.compile(r"\(([A-Ja-j])\)"),
        re.compile(r"(?:(?<=\s)|^)([A-Ja-j])[\)\.:]\s+"),
    )
    for pattern in patterns:
        matches = list(pattern.finditer(cleaned))
        if len(matches) < 2:
            continue
        options: Dict[str, str] = {}
        for index, match in enumerate(matches):
            letter = match.group(1).lower()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
            option_text = cleaned[start:end].strip(" :-.;,")
            normalized = _normalize_exact_answer_value(option_text)
            if normalized:
                options[letter] = normalized
        if len(options) >= 2:
            return options
    return {}


def _build_private_answer_key_features(
    example: JudgeBenchJoinedExample,
    *,
    reference_answer_visible: bool,
) -> JudgeBenchAnswerKeyFeatures:
    normalized_reference_value = _normalize_exact_answer_value(example.reference_answer)
    requested_answer_mode = _question_requested_answer_mode(example.question, example.reference_answer)
    choice_value_map = _question_choice_value_map(example.question)
    exact_answer_task = bool(
        choice_value_map
        or _is_exact_answer_task(example.question, example.reference_answer)
        or requested_answer_mode
    )
    return JudgeBenchAnswerKeyFeatures(
        source_family=example.source_family,
        exact_answer_task=exact_answer_task,
        requested_answer_mode=requested_answer_mode,
        normalized_reference_value=normalized_reference_value,
        choice_value_map=choice_value_map,
        mcq_option_letters=tuple(sorted(letter.upper() for letter in choice_value_map))
        or _mcq_option_letters(example.question),
        repeated_choice_length=_required_repeated_choice_length(example.question) if requested_answer_mode == "repeated_choice" else 0,
        reference_answer_visible=bool(reference_answer_visible),
    )


def _answer_key_features(example: JudgeBenchJoinedExample) -> JudgeBenchAnswerKeyFeatures:
    raw_payload = dict((example.metadata or {}).get(_PRIVATE_ANSWER_KEY_METADATA_KEY, {}) or {})
    if raw_payload:
        return JudgeBenchAnswerKeyFeatures(
            source_family=str(raw_payload.get("source_family", example.source_family)).strip(),
            exact_answer_task=bool(raw_payload.get("exact_answer_task")),
            requested_answer_mode=str(raw_payload.get("requested_answer_mode", "")).strip(),
            normalized_reference_value=str(raw_payload.get("normalized_reference_value", "")).strip(),
            choice_value_map={
                str(key).strip().lower(): str(value).strip().lower()
                for key, value in dict(raw_payload.get("choice_value_map", {}) or {}).items()
                if str(key).strip() and str(value).strip()
            },
            mcq_option_letters=tuple(str(item).strip().upper() for item in list(raw_payload.get("mcq_option_letters", ()) or []) if str(item).strip()),
            repeated_choice_length=int(raw_payload.get("repeated_choice_length", 0) or 0),
            reference_answer_visible=bool(raw_payload.get("reference_answer_visible")),
        )
    return _build_private_answer_key_features(
        example,
        reference_answer_visible=bool(_normalize_text(example.reference_answer)),
    )


def _example_reference_visible(example: JudgeBenchJoinedExample) -> bool:
    if _normalize_text(example.reference_answer):
        return True
    return bool(_answer_key_features(example).reference_answer_visible)


def _example_requested_answer_mode(example: JudgeBenchJoinedExample) -> str:
    features = _answer_key_features(example)
    if features.requested_answer_mode:
        return features.requested_answer_mode
    return _question_requested_answer_mode(example.question, example.reference_answer)


def _example_choice_value_map(example: JudgeBenchJoinedExample) -> Dict[str, str]:
    features = _answer_key_features(example)
    if features.choice_value_map:
        return dict(features.choice_value_map)
    return _question_choice_value_map(example.question)


def _example_is_exact_answer_task(example: JudgeBenchJoinedExample) -> bool:
    features = _answer_key_features(example)
    if features.exact_answer_task:
        return True
    return _is_exact_answer_task(example.question, example.reference_answer)


def _example_reference_value(example: JudgeBenchJoinedExample) -> str:
    features = _answer_key_features(example)
    if features.normalized_reference_value:
        return features.normalized_reference_value
    return _normalize_exact_answer_value(example.reference_answer)


def _extract_exact_answer_candidate_for_example(
    example: JudgeBenchJoinedExample,
    candidate_text: str,
) -> Optional[ExactAnswerExtraction]:
    return _extract_exact_answer_candidate(
        candidate_text,
        reference_answer=_example_reference_value(example),
        question=example.question,
    )


def _public_example_payload(example: JudgeBenchJoinedExample) -> Dict[str, Any]:
    payload = to_json_dict(example)
    payload["metadata"] = _strip_private_metadata(payload.get("metadata", {}) or {})
    return payload


def _example_for_reference_access(
    example: JudgeBenchJoinedExample,
    *,
    reference_answer_access: bool,
) -> JudgeBenchJoinedExample:
    payload = asdict(example)
    metadata = dict(payload.get("metadata", {}) or {})
    metadata[_PRIVATE_ANSWER_KEY_METADATA_KEY] = to_json_dict(
        _build_private_answer_key_features(
            example,
            reference_answer_visible=reference_answer_access,
        )
    )
    payload["metadata"] = metadata
    if not reference_answer_access:
        payload["reference_answer"] = ""
    return JudgeBenchJoinedExample(**payload)


def _normalize_protocol_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return _DEFAULT_PROTOCOL_MODE
    if normalized not in _ALLOWED_PROTOCOL_MODES:
        raise ValueError(f"Unsupported JudgeBench protocol_mode: {mode!r}")
    return normalized


def _normalize_blind_scoring_profile(profile: Optional[str]) -> str:
    normalized = str(profile or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return _BLIND_SCORING_PROFILE_BASELINE
    if normalized not in _ALLOWED_BLIND_SCORING_PROFILES:
        raise ValueError(f"Unsupported blind scoring profile: {profile!r}")
    return normalized


def _normalize_blind_budget_profile(profile: Optional[str]) -> str:
    normalized = str(profile or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return _BLIND_BUDGET_PROFILE_FAMILY_V1
    if normalized not in _ALLOWED_BLIND_BUDGET_PROFILES:
        raise ValueError(f"Unsupported blind budget profile: {profile!r}")
    return normalized


def _normalize_blind_guidance_profile(profile: Optional[str]) -> str:
    normalized = str(profile or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return _BLIND_GUIDANCE_PROFILE_OFF
    if normalized not in _ALLOWED_BLIND_GUIDANCE_PROFILES:
        raise ValueError(f"Unsupported blind guidance profile: {profile!r}")
    return normalized


def _normalize_blind_wu_profile(profile: Optional[str]) -> str:
    normalized = str(profile or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return _BLIND_WU_PROFILE_RAW
    if normalized not in _ALLOWED_BLIND_WU_PROFILES:
        raise ValueError(f"Unsupported blind WU profile: {profile!r}")
    return normalized


def _normalize_retrieval_profile(profile: Optional[str]) -> str:
    normalized = str(profile or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return _RETRIEVAL_PROFILE_OFF
    if normalized not in _ALLOWED_RETRIEVAL_PROFILES:
        raise ValueError(f"Unsupported retrieval profile: {profile!r}")
    return normalized


def _normalize_source_family_key(source_family: Any) -> str:
    return judgebench_source_family(str(source_family or "").strip())


def _normalize_retrieval_profile_by_family(mapping: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for raw_family, raw_profile in dict(mapping or {}).items():
        family = _normalize_source_family_key(raw_family)
        if not family:
            continue
        normalized[family] = _normalize_retrieval_profile(raw_profile)
    return dict(sorted(normalized.items()))


def _normalize_retrieval_top_k_by_family(mapping: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for raw_family, raw_value in dict(mapping or {}).items():
        family = _normalize_source_family_key(raw_family)
        if not family:
            continue
        normalized[family] = max(1, int(raw_value or 2))
    return dict(sorted(normalized.items()))


def _score_external_slices_for_run(
    *,
    v2_config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Score the locked rubric library on external preference slices (HelpSteer3 / PPE) to provide a
    transport proxy for the blind-350 score. Returns an empty dict when the library or slice
    files are missing -- this is always fail-open because the primary OOF run must not block on
    the external signal.
    """
    from rubric_gen.compiled.external_eval_slices import score_external_slices as _score_slices

    normalized = _normalize_v2_config(v2_config)
    library_path = normalized.get("rubric_library_path", "")
    if not library_path:
        return {}
    try:
        library = load_rubric_library(Path(library_path))
    except Exception:
        library = None
    if library is None:
        return {}
    repo_root = Path(__file__).resolve().parents[2]
    return _score_slices(repo_root=repo_root, library=library)


def _normalize_v2_config(v2_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Coerce a v2 config mapping into a stable, serialisable dict with safe defaults."""
    v2 = dict(v2_config or {})
    try:
        sc_n = int(v2.get("self_consistency_n", _DEFAULT_SELF_CONSISTENCY_N) or _DEFAULT_SELF_CONSISTENCY_N)
    except (TypeError, ValueError):
        sc_n = _DEFAULT_SELF_CONSISTENCY_N
    try:
        sc_temp = float(
            v2.get("self_consistency_temperature", _DEFAULT_SELF_CONSISTENCY_TEMPERATURE)
            or _DEFAULT_SELF_CONSISTENCY_TEMPERATURE
        )
    except (TypeError, ValueError):
        sc_temp = _DEFAULT_SELF_CONSISTENCY_TEMPERATURE
    try:
        library_top_k = int(v2.get("library_retrieval_top_k", 0) or 0)
    except (TypeError, ValueError):
        library_top_k = 0
    try:
        redundancy_threshold = float(v2.get("rrd_redundancy_threshold", 0.9) or 0.9)
    except (TypeError, ValueError):
        redundancy_threshold = 0.9
    try:
        math_samples = int(v2.get("math_solver_samples", 1) or 1)
    except (TypeError, ValueError):
        math_samples = 1
    try:
        math_temp = float(v2.get("math_solver_temperature", 0.5) or 0.5)
    except (TypeError, ValueError):
        math_temp = 0.5
    try:
        code_timeout = float(v2.get("code_execution_timeout_s", 10.0) or 10.0)
    except (TypeError, ValueError):
        code_timeout = 10.0
    try:
        code_margin = float(v2.get("code_execution_min_margin", 0.34) or 0.34)
    except (TypeError, ValueError):
        code_margin = 0.34
    return {
        "self_consistency_n": max(1, min(9, sc_n)),
        "self_consistency_temperature": max(0.0, min(1.5, sc_temp)),
        "v2_wide_discriminator_gate": bool(v2.get("v2_wide_discriminator_gate", False)),
        "holistic_judge_enabled": bool(v2.get("holistic_judge_enabled", False)),
        "library_retrieval_top_k": max(0, min(16, library_top_k)),
        "rubric_library_path": str(v2.get("rubric_library_path") or ""),
        "enable_rrd_filters": bool(v2.get("enable_rrd_filters", False)),
        "rrd_redundancy_threshold": max(0.0, min(1.0, redundancy_threshold)),
        "library_retrieval_top_k_by_family": _normalize_retrieval_top_k_by_family(
            v2.get("library_retrieval_top_k_by_family")
        ),
        "family_strict_library_mode": bool(v2.get("family_strict_library_mode", False)),
        "math_independent_solver_enabled": bool(v2.get("math_independent_solver_enabled", False)),
        "math_solver_samples": max(1, min(9, math_samples)),
        "math_solver_temperature": max(0.0, min(1.5, math_temp)),
        "code_execution_verifier_enabled": bool(v2.get("code_execution_verifier_enabled", False)),
        "code_execution_timeout_s": max(1.0, min(60.0, code_timeout)),
        "code_execution_min_margin": max(0.0, min(1.0, code_margin)),
        "math_solver_use_sympy": bool(v2.get("math_solver_use_sympy", False)),
        "math_solver_model": str(v2.get("math_solver_model", "") or ""),
        "few_shot_train_enabled": bool(v2.get("few_shot_train_enabled", False)),
        "few_shot_top_k": max(0, min(10, int(v2.get("few_shot_top_k", 3) or 3))),
        "few_shot_train_dataset_path": str(v2.get("few_shot_train_dataset_path", "") or ""),
        "few_shot_official_dataset_path": str(v2.get("few_shot_official_dataset_path", "") or ""),
        "rubric_satisfaction_samples": max(1, min(9, int(v2.get("rubric_satisfaction_samples", 1) or 1))),
        "rubric_satisfaction_temperature": max(0.0, min(1.5, float(v2.get("rubric_satisfaction_temperature", 0.4) or 0.4))),
        "discriminator_self_critique_enabled": bool(v2.get("discriminator_self_critique_enabled", False)),
        "mmlu_independent_answerer_enabled": bool(v2.get("mmlu_independent_answerer_enabled", False)),
        "mmlu_answerer_samples": max(1, min(9, int(v2.get("mmlu_answerer_samples", 1) or 1))),
        "mmlu_answerer_temperature": max(0.0, min(1.5, float(v2.get("mmlu_answerer_temperature", 0.5) or 0.5))),
        "mmlu_answerer_model": str(v2.get("mmlu_answerer_model", "") or ""),
        "mmlu_answerer_secondary_model": str(v2.get("mmlu_answerer_secondary_model", "") or ""),
        "mmlu_answerer_secondary_samples": max(
            1, min(9, int(v2.get("mmlu_answerer_secondary_samples", 1) or 1))
        ),
        "mmlu_answerer_secondary_temperature": max(
            0.0, min(1.5, float(v2.get("mmlu_answerer_secondary_temperature", 0.0) or 0.0))
        ),
        "reasoning_independent_solver_enabled": bool(v2.get("reasoning_independent_solver_enabled", False)),
        "reasoning_solver_samples": max(1, min(9, int(v2.get("reasoning_solver_samples", 1) or 1))),
        "reasoning_solver_temperature": max(0.0, min(1.5, float(v2.get("reasoning_solver_temperature", 0.5) or 0.5))),
        "reasoning_solver_model": str(v2.get("reasoning_solver_model", "") or ""),
    }


def _normalize_blind_discriminator_family_mode(mode: Optional[str]) -> str:
    normalized = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return _BLIND_DISCRIMINATOR_MODE_DEFAULT
    if normalized not in _ALLOWED_BLIND_DISCRIMINATOR_FAMILY_MODES:
        raise ValueError(f"Unsupported blind discriminator family mode: {mode!r}")
    return normalized


def _normalize_blind_discriminator_mode_by_family(mapping: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for raw_family, raw_mode in dict(mapping or {}).items():
        family = _normalize_source_family_key(raw_family)
        if not family:
            continue
        normalized[family] = _normalize_blind_discriminator_family_mode(raw_mode)
    return dict(sorted(normalized.items()))


def _policy_blind_scoring_profile(policy: Mapping[str, Any]) -> str:
    return _normalize_blind_scoring_profile(policy.get("blind_scoring_profile"))


def _policy_blind_discriminator_mode(policy: Mapping[str, Any], source_family: Optional[str] = None) -> str:
    normalized = _BLIND_DISCRIMINATOR_MODE_DEFAULT
    if source_family is None:
        return normalized
    family_overrides = _normalize_blind_discriminator_mode_by_family(policy.get("blind_discriminator_mode_by_family"))
    family_key = _normalize_source_family_key(source_family)
    return family_overrides.get(family_key, normalized)


def _policy_blind_budget_profile(policy: Mapping[str, Any]) -> str:
    return _normalize_blind_budget_profile(policy.get("blind_budget_profile"))


def _policy_blind_guidance_profile(policy: Mapping[str, Any]) -> str:
    return _normalize_blind_guidance_profile(policy.get("blind_guidance_profile"))


def _policy_blind_wu_profile(policy: Mapping[str, Any]) -> str:
    return _normalize_blind_wu_profile(policy.get("blind_wu_profile"))


def _policy_retrieval_profile(policy: Mapping[str, Any], source_family: Optional[str] = None) -> str:
    normalized = _normalize_retrieval_profile(policy.get("retrieval_profile"))
    if source_family is None:
        return normalized
    family_overrides = _normalize_retrieval_profile_by_family(policy.get("retrieval_profile_by_family"))
    family_key = _normalize_source_family_key(source_family)
    return family_overrides.get(family_key, normalized)


def _policy_retrieval_top_k(policy: Mapping[str, Any], source_family: Optional[str] = None) -> int:
    normalized = max(1, int(policy.get("retrieval_top_k", 2) or 2))
    if source_family is None:
        return normalized
    family_overrides = _normalize_retrieval_top_k_by_family(policy.get("retrieval_top_k_by_family"))
    family_key = _normalize_source_family_key(source_family)
    return family_overrides.get(family_key, normalized)


def _policy_has_any_retrieval(policy: Mapping[str, Any]) -> bool:
    if _policy_retrieval_profile(policy) != _RETRIEVAL_PROFILE_OFF:
        return True
    family_overrides = _normalize_retrieval_profile_by_family(policy.get("retrieval_profile_by_family"))
    return any(profile != _RETRIEVAL_PROFILE_OFF for profile in family_overrides.values())


def _policy_self_consistency_n(policy: Mapping[str, Any]) -> int:
    try:
        value = int(policy.get("self_consistency_n", _DEFAULT_SELF_CONSISTENCY_N) or _DEFAULT_SELF_CONSISTENCY_N)
    except (TypeError, ValueError):
        return _DEFAULT_SELF_CONSISTENCY_N
    return max(1, min(value, 9))


def _policy_self_consistency_temperature(policy: Mapping[str, Any]) -> float:
    try:
        value = float(
            policy.get("self_consistency_temperature", _DEFAULT_SELF_CONSISTENCY_TEMPERATURE)
            or _DEFAULT_SELF_CONSISTENCY_TEMPERATURE
        )
    except (TypeError, ValueError):
        return _DEFAULT_SELF_CONSISTENCY_TEMPERATURE
    return max(0.0, min(1.5, value))


def _policy_v2_wide_discriminator_gate(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("v2_wide_discriminator_gate", False))


def _policy_holistic_judge_enabled(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("holistic_judge_enabled", False))


def _policy_library_retrieval_top_k(
    policy: Mapping[str, Any], source_family: Optional[str] = None
) -> int:
    try:
        default = int(policy.get("library_retrieval_top_k", 0) or 0)
    except (TypeError, ValueError):
        default = 0
    default = max(0, min(default, 16))
    if source_family is None:
        return default
    overrides = policy.get("library_retrieval_top_k_by_family") or {}
    family_key = _normalize_source_family_key(source_family)
    if family_key in overrides:
        try:
            override_value = int(overrides[family_key] or 0)
        except (TypeError, ValueError):
            override_value = default
        return max(0, min(override_value, 16))
    return default


def _policy_family_strict_library(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("family_strict_library_mode", False))


def _policy_math_independent_solver_enabled(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("math_independent_solver_enabled", False))


def _policy_math_solver_samples(policy: Mapping[str, Any]) -> int:
    try:
        value = int(policy.get("math_solver_samples", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, 9))


def _policy_math_solver_temperature(policy: Mapping[str, Any]) -> float:
    try:
        value = float(policy.get("math_solver_temperature", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(value, 1.5))


def _policy_math_solver_use_sympy(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("math_solver_use_sympy", False))


def _policy_math_solver_model(policy: Mapping[str, Any]) -> str:
    """
    Optional provider:model for the math independent solver.

    Empty string -> use the default GPT-4o scoring model. Like the MMLU answerer, the math
    solver does NOT cast the final A>B verdict; it only produces a canonical answer that the
    pair verifier compares against candidate answers, so non-OpenAI models are allowed here.
    """
    return str(policy.get("math_solver_model", "") or "").strip()


def _policy_few_shot_train_enabled(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("few_shot_train_enabled", False))


def _policy_few_shot_top_k(policy: Mapping[str, Any]) -> int:
    try:
        value = int(policy.get("few_shot_top_k", 3) or 3)
    except (TypeError, ValueError):
        return 3
    return max(0, min(value, 10))


def _policy_few_shot_train_dataset_path(policy: Mapping[str, Any]) -> str:
    return str(policy.get("few_shot_train_dataset_path", "") or "")


def _policy_few_shot_official_dataset_path(policy: Mapping[str, Any]) -> str:
    return str(policy.get("few_shot_official_dataset_path", "") or "")


def _policy_rubric_satisfaction_samples(policy: Mapping[str, Any]) -> int:
    try:
        value = int(policy.get("rubric_satisfaction_samples", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, 9))


def _policy_rubric_satisfaction_temperature(policy: Mapping[str, Any]) -> float:
    try:
        value = float(policy.get("rubric_satisfaction_temperature", 0.4) or 0.4)
    except (TypeError, ValueError):
        return 0.4
    return max(0.0, min(value, 1.5))


def _policy_discriminator_self_critique_enabled(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("discriminator_self_critique_enabled", False))


def _policy_mmlu_independent_answerer_enabled(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("mmlu_independent_answerer_enabled", False))


def _policy_mmlu_answerer_samples(policy: Mapping[str, Any]) -> int:
    try:
        value = int(policy.get("mmlu_answerer_samples", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, 9))


def _policy_mmlu_answerer_temperature(policy: Mapping[str, Any]) -> float:
    try:
        value = float(policy.get("mmlu_answerer_temperature", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(value, 1.5))


def _policy_mmlu_answerer_model(policy: Mapping[str, Any]) -> str:
    """
    Model spec used for the MMLU independent answerer.

    The answerer is a tool the GPT-4o judge uses to ground its mmlu-pro decisions; it does
    NOT replace the judge. Per the project policy, non-OpenAI models are allowed in
    rubric generation / discovery / synthetic-contrast / external-data labeling roles, and
    the independent-answerer role falls into the same category (it never produces the
    final A>B / B>A decision text — it only emits a grounding letter).

    Empty string -> use the default GPT-4o scoring model.
    """
    return str(policy.get("mmlu_answerer_model", "") or "").strip()


def _policy_mmlu_answerer_secondary_model(policy: Mapping[str, Any]) -> str:
    """Optional second answerer; require dual consensus before overriding base verdict."""
    return str(policy.get("mmlu_answerer_secondary_model", "") or "").strip()


def _policy_reasoning_independent_solver_enabled(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("reasoning_independent_solver_enabled", False))


def _policy_reasoning_solver_samples(policy: Mapping[str, Any]) -> int:
    try:
        value = int(policy.get("reasoning_solver_samples", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, 9))


def _policy_reasoning_solver_temperature(policy: Mapping[str, Any]) -> float:
    try:
        value = float(policy.get("reasoning_solver_temperature", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(value, 1.5))


def _policy_reasoning_solver_model(policy: Mapping[str, Any]) -> str:
    """Optional provider:model for the reasoning independent solver. Empty -> GPT-4o."""
    return str(policy.get("reasoning_solver_model", "") or "").strip()


_FEW_SHOT_INDEX_CACHE: Dict[Tuple[str, str], Any] = {}


def _load_policy_few_shot_index(policy: Mapping[str, Any]):
    if not _policy_few_shot_train_enabled(policy):
        return None
    train_path = _policy_few_shot_train_dataset_path(policy)
    official_path = _policy_few_shot_official_dataset_path(policy)
    if not train_path or not official_path:
        return None
    key = (train_path, official_path)
    if key in _FEW_SHOT_INDEX_CACHE:
        return _FEW_SHOT_INDEX_CACHE[key]
    try:
        from rubric_gen.compiled.labeled_train_few_shot import (
            load_labeled_train_few_shot_index,
        )

        index = load_labeled_train_few_shot_index(
            train_dataset_path=Path(train_path),
            official_dataset_path=Path(official_path),
        )
    except Exception:
        index = None
    _FEW_SHOT_INDEX_CACHE[key] = index
    return index


def _policy_code_execution_verifier_enabled(policy: Mapping[str, Any]) -> bool:
    return bool(policy.get("code_execution_verifier_enabled", False))


def _policy_code_execution_timeout_s(policy: Mapping[str, Any]) -> float:
    try:
        value = float(policy.get("code_execution_timeout_s", 10.0) or 10.0)
    except (TypeError, ValueError):
        return 10.0
    return max(1.0, min(value, 60.0))


def _policy_code_execution_min_margin(policy: Mapping[str, Any]) -> float:
    try:
        value = float(policy.get("code_execution_min_margin", 0.34) or 0.34)
    except (TypeError, ValueError):
        return 0.34
    return max(0.0, min(value, 1.0))


def _policy_protocol_mode(policy: Mapping[str, Any]) -> str:
    return _normalize_protocol_mode(str(policy.get("protocol_mode", _DEFAULT_PROTOCOL_MODE)))


def _protocol_uses_judgebench_tuning(mode_or_policy: Any) -> bool:
    if isinstance(mode_or_policy, Mapping):
        protocol_mode = _policy_protocol_mode(mode_or_policy)
    else:
        protocol_mode = _normalize_protocol_mode(str(mode_or_policy))
    return protocol_mode == _PROTOCOL_MODE_JUDGEBENCH_TUNED


def judgebench_source_family(source: str) -> str:
    normalized = (source or "").strip().lower()
    for family, prefix in _BUILT_IN_SUBSET_PREFIXES.items():
        if normalized.startswith(prefix):
            return family
    if normalized.startswith("mmlu-pro"):
        return "mmlu-pro"
    if normalized.startswith("livebench-reasoning"):
        return "livebench-reasoning"
    if normalized.startswith("livebench-math"):
        return "livebench-math"
    if normalized.startswith("livecodebench"):
        return "livecodebench"
    if normalized.startswith("livebench"):
        return "livebench"
    if "-" in normalized:
        return normalized.split("-", 1)[0]
    return normalized or "unknown"


def _question_mismatch(local_question: str, official_question: str) -> bool:
    return _normalize_text(local_question).lower() != _normalize_text(official_question).lower()


def load_local_judgebench_subset(path: Path) -> List[JudgeBenchLocalExample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level JSON array in {path}.")

    rows: List[JudgeBenchLocalExample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id", "")).strip()
        question = str(item.get("question", "")).strip()
        source = str(item.get("source", "")).strip()
        if not pair_id or not question or not source:
            raise ValueError(f"Malformed JudgeBench subset row {index} in {path}.")
        candidate_models = [str(x).strip() for x in item.get("candidate_models", []) if str(x).strip()]
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"pair_id", "source", "question", "reference_answer", "candidate_models", "verifier_model"}
        }
        rows.append(
            JudgeBenchLocalExample(
                pair_id=pair_id,
                source=source,
                question=question,
                reference_answer=str(item.get("reference_answer", "")).strip(),
                candidate_models=candidate_models,
                verifier_model=str(item.get("verifier_model", "")).strip(),
                metadata=metadata,
            )
        )
    return rows


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def ensure_official_judgebench_dataset(dataset_path: Optional[Path], data_dir: Path) -> Path:
    if dataset_path is not None:
        target = Path(dataset_path)
        if not target.exists():
            raise FileNotFoundError(f"Official JudgeBench dataset was not found: {target}")
        return target
    target = data_dir / "dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
    if not target.exists():
        _download_file(JUDGEBENCH_GPT4O_URL, target)
    return target


def load_official_judgebench_pairs(path: Path) -> List[JudgeBenchPairRecord]:
    rows: List[JudgeBenchPairRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                JudgeBenchPairRecord(
                    pair_id=str(payload.get("pair_id", "")).strip(),
                    original_id=str(payload.get("original_id", "")).strip(),
                    source=str(payload.get("source", "")).strip(),
                    question=str(payload.get("question", "")).strip(),
                    response_model=str(payload.get("response_model", "")).strip(),
                    response_A=str(payload.get("response_A", "")).strip(),
                    response_B=str(payload.get("response_B", "")).strip(),
                    label=str(payload.get("label", "")).strip(),
                )
            )
    return rows


def join_local_subset_to_official_pairs(
    *,
    local_rows: Sequence[JudgeBenchLocalExample],
    official_pairs: Sequence[JudgeBenchPairRecord],
    split_name: str,
) -> List[JudgeBenchJoinedExample]:
    pair_lookup = {row.pair_id: row for row in official_pairs}
    joined: List[JudgeBenchJoinedExample] = []
    missing: List[str] = []
    for row in local_rows:
        pair = pair_lookup.get(row.pair_id)
        if pair is None:
            missing.append(row.pair_id)
            continue
        metadata = dict(row.metadata)
        if _question_mismatch(row.question, pair.question):
            metadata["question_mismatch"] = {
                "local_question_hash": stable_hash(row.question),
                "official_question_hash": stable_hash(pair.question),
            }
        joined.append(
            JudgeBenchJoinedExample(
                split_name=split_name,
                pair_id=row.pair_id,
                source=pair.source or row.source,
                source_family=judgebench_source_family(pair.source or row.source),
                question=pair.question or row.question,
                reference_answer=row.reference_answer,
                response_model=pair.response_model,
                response_A=pair.response_A,
                response_B=pair.response_B,
                label=pair.label,
                original_id=pair.original_id,
                candidate_models=list(row.candidate_models),
                verifier_model=row.verifier_model,
                metadata=metadata,
            )
        )
    if missing:
        raise ValueError(
            f"{split_name} could not be joined to the official JudgeBench dataset for {len(missing)} pair_ids. "
            f"First missing ids: {missing[:8]}"
        )
    return joined


def ensure_disjoint_pair_ids(train_rows: Sequence[JudgeBenchJoinedExample], val_rows: Sequence[JudgeBenchJoinedExample]) -> None:
    overlap = {row.pair_id for row in train_rows} & {row.pair_id for row in val_rows}
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(f"Train/validation pair_id overlap detected: {preview}")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _serialize_task_profile(profile: TaskProfile) -> Dict[str, Any]:
    return {
        "task_profile_id": profile.task_profile_id,
        "label": profile.label,
        "artifact_label": profile.artifact_label,
        "artifact_kind": profile.artifact_kind,
        "default_task_prompt": profile.default_task_prompt,
        "default_task_family_id": profile.default_task_family_id,
        "contrast_strategy_id": profile.contrast_strategy_id,
        "strong_source_priority": list(profile.strong_source_priority),
        "discovery_context": profile.discovery_context,
        "discovery_dimensions": list(profile.discovery_dimensions),
        "parent_profile_id": profile.parent_profile_id,
        "built_in": profile.built_in,
        "feature_tags": list(profile.feature_tags),
        "metadata": to_json_dict(profile.metadata),
    }


def _serialize_contrast_strategy(strategy: ContrastStrategy) -> Dict[str, Any]:
    return {
        "strategy_id": strategy.strategy_id,
        "mutation_ids": list(strategy.mutation_ids),
        "mutation_grounding_profiles": to_json_dict(dict(strategy.mutation_grounding_profiles)),
    }


def _deserialize_task_profile(payload: Mapping[str, Any]) -> TaskProfile:
    return TaskProfile(
        task_profile_id=str(payload.get("task_profile_id", "")).strip(),
        label=str(payload.get("label", "")).strip(),
        artifact_label=str(payload.get("artifact_label", "")).strip(),
        artifact_kind=str(payload.get("artifact_kind", "")).strip(),
        default_task_prompt=str(payload.get("default_task_prompt", "")).strip(),
        default_task_family_id=str(payload.get("default_task_family_id", "")).strip(),
        contrast_strategy_id=str(payload.get("contrast_strategy_id", "")).strip(),
        strong_source_priority=tuple(str(x) for x in payload.get("strong_source_priority", []) if str(x).strip()),
        discovery_context=str(payload.get("discovery_context", "")).strip(),
        discovery_dimensions=tuple(str(x) for x in payload.get("discovery_dimensions", []) if str(x).strip()),
        parent_profile_id=str(payload.get("parent_profile_id", "")).strip() or None,
        built_in=bool(payload.get("built_in", False)),
        feature_tags=tuple(str(x) for x in payload.get("feature_tags", []) if str(x).strip()),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _deserialize_contrast_strategy(payload: Mapping[str, Any]) -> ContrastStrategy:
    mutation_grounding = payload.get("mutation_grounding_profiles", {}) or {}
    return ContrastStrategy(
        strategy_id=str(payload.get("strategy_id", "")).strip(),
        mutation_ids=tuple(str(x) for x in payload.get("mutation_ids", []) if str(x).strip()),
        mutation_grounding_profiles={
            str(key): dict(value) if isinstance(value, Mapping) else {}
            for key, value in mutation_grounding.items()
        },
    )


def _register_route_payload(route_payload: Mapping[str, Any]) -> None:
    register_contrast_strategy(_deserialize_contrast_strategy(route_payload["strategy_bundle"]))
    register_task_profile(_deserialize_task_profile(route_payload["profile_bundle"]))


def _make_dynamic_route_payload(
    *,
    route_key: str,
    route_kind: str,
    resolution: Any,
) -> Dict[str, Any]:
    source_slug = _safe_slug(route_key)
    base_profile = resolution.profile
    base_strategy = resolution.strategy
    profile_id = f"judgebench_{route_kind}_{source_slug}_{_safe_slug(base_profile.task_profile_id)}"
    strategy_id = f"{profile_id}__strategy"
    profile_label_prefix = "JudgeBench Validation Fallback" if route_kind == "fallback" else f"JudgeBench {route_key}"
    strategy_bundle = _serialize_contrast_strategy(
        ContrastStrategy(
            strategy_id=strategy_id,
            mutation_ids=tuple(base_strategy.mutation_ids),
            mutation_grounding_profiles=dict(base_strategy.mutation_grounding_profiles),
        )
    )
    profile_bundle = _serialize_task_profile(
        TaskProfile(
            task_profile_id=profile_id,
            label=f"{profile_label_prefix} {base_profile.label}",
            artifact_label=base_profile.artifact_label,
            artifact_kind=base_profile.artifact_kind,
            default_task_prompt=base_profile.default_task_prompt,
            default_task_family_id=base_profile.default_task_family_id,
            contrast_strategy_id=strategy_id,
            strong_source_priority=base_profile.strong_source_priority,
            discovery_context=base_profile.discovery_context,
            discovery_dimensions=base_profile.discovery_dimensions,
            parent_profile_id=base_profile.task_profile_id,
            built_in=False,
            feature_tags=base_profile.feature_tags,
            metadata={
                "judgebench_route_key": route_key,
                "judgebench_route_kind": route_kind,
                "bootstrap_used": bool(resolution.bootstrap_used),
                "bootstrap_diagnostics": to_json_dict(resolution.diagnostics),
            },
        )
    )
    return {
        "route_key": route_key,
        "route_kind": route_kind,
        "task_profile_id": profile_id,
        "contrast_strategy_id": strategy_id,
        "bootstrap_used": bool(resolution.bootstrap_used),
        "parent_task_profile_id": base_profile.task_profile_id,
        "profile_bundle": profile_bundle,
        "strategy_bundle": strategy_bundle,
        "diagnostics": to_json_dict(resolution.diagnostics),
    }


def build_initial_frozen_policy(
    *,
    train_examples: Sequence[JudgeBenchJoinedExample],
    bootstrap_iterations: int,
    recursive_config: RecursiveDiscoveryConfig,
    protocol_mode: str = _DEFAULT_PROTOCOL_MODE,
    reference_answer_access: bool = False,
    blind_scoring_profile: str = _BLIND_SCORING_PROFILE_BASELINE,
    blind_budget_profile: str = _BLIND_BUDGET_PROFILE_FAMILY_V1,
    blind_guidance_profile: str = _BLIND_GUIDANCE_PROFILE_OFF,
    blind_wu_profile: str = _BLIND_WU_PROFILE_RAW,
    retrieval_profile: str = _RETRIEVAL_PROFILE_OFF,
    retrieval_top_k: int = 2,
    blind_discriminator_mode_by_family: Optional[Mapping[str, Any]] = None,
    retrieval_profile_by_family: Optional[Mapping[str, Any]] = None,
    retrieval_top_k_by_family: Optional[Mapping[str, Any]] = None,
    self_consistency_n: int = _DEFAULT_SELF_CONSISTENCY_N,
    self_consistency_temperature: float = _DEFAULT_SELF_CONSISTENCY_TEMPERATURE,
    v2_wide_discriminator_gate: bool = False,
    holistic_judge_enabled: bool = False,
    library_retrieval_top_k: int = 0,
    rubric_library_path: Optional[Path] = None,
    enable_rrd_filters: bool = False,
    rrd_redundancy_threshold: float = 0.9,
    library_retrieval_top_k_by_family: Optional[Mapping[str, Any]] = None,
    family_strict_library_mode: bool = False,
    math_independent_solver_enabled: bool = False,
    math_solver_samples: int = 1,
    math_solver_temperature: float = 0.5,
    code_execution_verifier_enabled: bool = False,
    code_execution_timeout_s: float = 10.0,
    code_execution_min_margin: float = 0.34,
    math_solver_use_sympy: bool = False,
    math_solver_model: str = "",
    few_shot_train_enabled: bool = False,
    few_shot_top_k: int = 3,
    few_shot_train_dataset_path: str = "",
    few_shot_official_dataset_path: str = "",
    rubric_satisfaction_samples: int = 1,
    rubric_satisfaction_temperature: float = 0.4,
    discriminator_self_critique_enabled: bool = False,
    mmlu_independent_answerer_enabled: bool = False,
    mmlu_answerer_samples: int = 1,
    mmlu_answerer_temperature: float = 0.5,
    mmlu_answerer_model: str = "",
    mmlu_answerer_secondary_model: str = "",
    mmlu_answerer_secondary_samples: int = 1,
    mmlu_answerer_secondary_temperature: float = 0.0,
    reasoning_independent_solver_enabled: bool = False,
    reasoning_solver_samples: int = 1,
    reasoning_solver_temperature: float = 0.5,
    reasoning_solver_model: str = "",
) -> Dict[str, Any]:
    protocol_mode = _normalize_protocol_mode(protocol_mode)
    blind_scoring_profile = _normalize_blind_scoring_profile(blind_scoring_profile)
    blind_budget_profile = _normalize_blind_budget_profile(blind_budget_profile)
    blind_guidance_profile = _normalize_blind_guidance_profile(blind_guidance_profile)
    blind_wu_profile = _normalize_blind_wu_profile(blind_wu_profile)
    retrieval_profile = _normalize_retrieval_profile(retrieval_profile)
    blind_discriminator_mode_by_family = _normalize_blind_discriminator_mode_by_family(
        blind_discriminator_mode_by_family
    )
    retrieval_profile_by_family = _normalize_retrieval_profile_by_family(retrieval_profile_by_family)
    retrieval_top_k_by_family = _normalize_retrieval_top_k_by_family(retrieval_top_k_by_family)
    clear_dynamic_task_profiles()
    clear_dynamic_contrast_strategies()

    grouped: Dict[str, List[JudgeBenchJoinedExample]] = defaultdict(list)
    for row in train_examples:
        grouped[row.source_family].append(row)

    route_payloads: Dict[str, Dict[str, Any]] = {}
    for source_family, rows in sorted(grouped.items()):
        bootstrap_examples = [
            joined_example_to_example_record(
                row,
                task_profile_id="general_instruction_following",
                reference_answer_access=reference_answer_access,
            )
            for row in rows
        ]
        resolution = resolve_or_bootstrap_task_profile(
            bootstrap_examples,
            explicit="auto",
            bootstrap_iterations=bootstrap_iterations,
        )
        route_payload = _make_dynamic_route_payload(
            route_key=source_family,
            route_kind="source_family",
            resolution=resolution,
        )
        route_payloads[source_family] = route_payload
        _register_route_payload(route_payload)

    fallback_examples = [
        joined_example_to_example_record(
            row,
            task_profile_id="general_instruction_following",
            reference_answer_access=reference_answer_access,
        )
        for row in train_examples
    ]
    fallback_resolution = resolve_or_bootstrap_task_profile(
        fallback_examples,
        explicit="auto",
        bootstrap_iterations=bootstrap_iterations,
    )
    fallback_payload = _make_dynamic_route_payload(
        route_key="global_fallback",
        route_kind="fallback",
        resolution=fallback_resolution,
    )
    _register_route_payload(fallback_payload)

    prompt_nudges: Dict[str, List[str]] = {"global": []}
    family_recursion_config: Dict[str, Dict[str, int]] = {}
    if _protocol_uses_judgebench_tuning(protocol_mode):
        for source_family in route_payloads:
            family_nudges = _family_prompt_nudges_for_source_family(source_family)
            if family_nudges:
                prompt_nudges[source_family] = family_nudges
        family_recursion_config = {
            "mmlu-pro": {
                "max_depth": max(recursive_config.max_depth, 2),
                "max_recursive_parents_per_pair": max(recursive_config.max_recursive_parents_per_pair, 3),
                "max_children_per_parent": max(recursive_config.max_children_per_parent, 4),
                "max_recursive_calls_per_pair": recursive_config.max_recursive_calls_per_pair,
            },
            "livecodebench": {
                "max_depth": 1,
                "max_recursive_parents_per_pair": 2,
                "max_children_per_parent": 3,
                "max_recursive_calls_per_pair": recursive_config.max_recursive_calls_per_pair,
            },
        }
    elif blind_guidance_profile != _BLIND_GUIDANCE_PROFILE_OFF:
        prompt_nudges["global"] = _blind_guidance_global_nudges(blind_guidance_profile)
        for source_family in route_payloads:
            family_nudges = _family_prompt_nudges_for_source_family(source_family)
            if family_nudges:
                prompt_nudges[source_family] = family_nudges

    return {
        "schema": "compiled_judgebench_policy_v1",
        "protocol_mode": protocol_mode,
        "blind_parity_bootstrap": not bool(reference_answer_access),
        "blind_scoring_profile": blind_scoring_profile,
        "blind_budget_profile": blind_budget_profile,
        "blind_guidance_profile": blind_guidance_profile,
        "blind_wu_profile": blind_wu_profile,
        "retrieval_profile": retrieval_profile,
        "retrieval_top_k": max(1, int(retrieval_top_k)),
        "blind_discriminator_mode_by_family": blind_discriminator_mode_by_family,
        "retrieval_profile_by_family": retrieval_profile_by_family,
        "retrieval_top_k_by_family": retrieval_top_k_by_family,
        "source_family_routes": route_payloads,
        "fallback_route": fallback_payload,
        "prompt_nudges": prompt_nudges,
        "recursion_config": {
            "max_depth": recursive_config.max_depth,
            "max_recursive_parents_per_pair": recursive_config.max_recursive_parents_per_pair,
            "max_children_per_parent": recursive_config.max_children_per_parent,
            "max_recursive_calls_per_pair": recursive_config.max_recursive_calls_per_pair,
        },
        "family_recursion_config": family_recursion_config,
        "refinement_history": [],
        "self_consistency_n": max(1, min(9, int(self_consistency_n or 1))),
        "self_consistency_temperature": max(0.0, min(1.5, float(self_consistency_temperature or 0.0))),
        "v2_wide_discriminator_gate": bool(v2_wide_discriminator_gate),
        "holistic_judge_enabled": bool(holistic_judge_enabled),
        "library_retrieval_top_k": max(0, min(16, int(library_retrieval_top_k or 0))),
        "rubric_library_path": str(rubric_library_path) if rubric_library_path else "",
        "enable_rrd_filters": bool(enable_rrd_filters),
        "rrd_redundancy_threshold": float(rrd_redundancy_threshold),
        "library_retrieval_top_k_by_family": _normalize_retrieval_top_k_by_family(
            library_retrieval_top_k_by_family
        ),
        "family_strict_library_mode": bool(family_strict_library_mode),
        "math_independent_solver_enabled": bool(math_independent_solver_enabled),
        "math_solver_samples": max(1, min(9, int(math_solver_samples or 1))),
        "math_solver_temperature": max(0.0, min(1.5, float(math_solver_temperature or 0.5))),
        "code_execution_verifier_enabled": bool(code_execution_verifier_enabled),
        "code_execution_timeout_s": max(1.0, min(60.0, float(code_execution_timeout_s or 10.0))),
        "code_execution_min_margin": max(0.0, min(1.0, float(code_execution_min_margin or 0.34))),
        "math_solver_use_sympy": bool(math_solver_use_sympy),
        "math_solver_model": str(math_solver_model or ""),
        "few_shot_train_enabled": bool(few_shot_train_enabled),
        "few_shot_top_k": max(0, min(10, int(few_shot_top_k or 3))),
        "few_shot_train_dataset_path": str(few_shot_train_dataset_path or ""),
        "few_shot_official_dataset_path": str(few_shot_official_dataset_path or ""),
        "rubric_satisfaction_samples": max(1, min(9, int(rubric_satisfaction_samples or 1))),
        "rubric_satisfaction_temperature": max(0.0, min(1.5, float(rubric_satisfaction_temperature or 0.4))),
        "discriminator_self_critique_enabled": bool(discriminator_self_critique_enabled),
        "mmlu_independent_answerer_enabled": bool(mmlu_independent_answerer_enabled),
        "mmlu_answerer_samples": max(1, min(9, int(mmlu_answerer_samples or 1))),
        "mmlu_answerer_temperature": max(0.0, min(1.5, float(mmlu_answerer_temperature or 0.5))),
        "mmlu_answerer_model": str(mmlu_answerer_model or ""),
        "mmlu_answerer_secondary_model": str(mmlu_answerer_secondary_model or ""),
        "mmlu_answerer_secondary_samples": max(1, min(9, int(mmlu_answerer_secondary_samples or 1))),
        "mmlu_answerer_secondary_temperature": max(
            0.0, min(1.5, float(mmlu_answerer_secondary_temperature or 0.0))
        ),
        "reasoning_independent_solver_enabled": bool(reasoning_independent_solver_enabled),
        "reasoning_solver_samples": max(1, min(9, int(reasoning_solver_samples or 1))),
        "reasoning_solver_temperature": max(0.0, min(1.5, float(reasoning_solver_temperature or 0.5))),
        "reasoning_solver_model": str(reasoning_solver_model or ""),
    }


def apply_frozen_policy(policy: Mapping[str, Any]) -> None:
    clear_dynamic_task_profiles()
    clear_dynamic_contrast_strategies()
    for route in (policy.get("source_family_routes", {}) or {}).values():
        _register_route_payload(route)
    fallback = policy.get("fallback_route")
    if isinstance(fallback, Mapping):
        _register_route_payload(fallback)


def joined_example_to_example_record(
    example: JudgeBenchJoinedExample,
    *,
    task_profile_id: str,
    reference_answer_access: bool = True,
) -> ExampleRecord:
    effective_example = _example_for_reference_access(example, reference_answer_access=reference_answer_access)
    profile = get_task_profile(task_profile_id)
    return ExampleRecord(
        example_id=f"{effective_example.split_name}__{effective_example.pair_id}",
        source=effective_example.source,
        source_id=effective_example.pair_id,
        dataset_subset=effective_example.split_name,
        conversation="",
        task_prompt=effective_example.question,
        reference_artifact=effective_example.reference_answer,
        task_profile_id=task_profile_id,
        task_family_id=profile.default_task_family_id or effective_example.source_family,
        artifact_kind=profile.artifact_kind or "response",
        metadata={
            "pair_id": effective_example.pair_id,
            "source_family": effective_example.source_family,
            "candidate_models": list(effective_example.candidate_models),
            "verifier_model": effective_example.verifier_model,
            **dict(effective_example.metadata),
        },
    )


def _policy_recursion_config(policy: Mapping[str, Any], *, source_family: str) -> Dict[str, int]:
    base = dict((policy.get("recursion_config", {}) or {}))
    overrides = dict(((policy.get("family_recursion_config", {}) or {}).get(source_family, {}) or {}))
    for key, value in overrides.items():
        if value is None:
            continue
        base[key] = int(value)
    return base


def _ordered_unique_nudges(nudges: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for nudge in nudges:
        text = _normalize_text(str(nudge))
        lowered = text.lower()
        if not text or lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(text)
    return ordered


def build_route_decision(
    example: JudgeBenchJoinedExample,
    policy: Mapping[str, Any],
    *,
    reference_answer_access: bool = True,
) -> JudgeBenchRouteDecision:
    effective_example = _example_for_reference_access(example, reference_answer_access=reference_answer_access)
    source_family = effective_example.source_family
    route = (policy.get("source_family_routes", {}) or {}).get(source_family)
    route_kind = "source_family"
    if route is None:
        route = policy.get("fallback_route", {})
        route_kind = "fallback"
    task_profile_id = str(route.get("task_profile_id", "")).strip()
    profile = get_task_profile(task_profile_id)
    uses_judgebench_tuning = _protocol_uses_judgebench_tuning(policy)
    prompt_nudges = list((policy.get("prompt_nudges", {}) or {}).get("global", []))
    prompt_nudges.extend(list((policy.get("prompt_nudges", {}) or {}).get(source_family, [])))
    if uses_judgebench_tuning:
        prompt_nudges.extend(_source_family_nudges(effective_example))
    return JudgeBenchRouteDecision(
        pair_id=effective_example.pair_id,
        source=effective_example.source,
        source_family=source_family,
        task_profile_id=task_profile_id,
        task_family_id=profile.default_task_family_id or source_family,
        artifact_kind=profile.artifact_kind or "response",
        route_kind=route_kind,
        bootstrap_used=bool(route.get("bootstrap_used", False)),
        parent_task_profile_id=str(route.get("parent_task_profile_id", "")).strip(),
        prompt_nudges=_ordered_unique_nudges(prompt_nudges),
        recursion_config=_policy_recursion_config(policy, source_family=source_family),
        diagnostics=dict(route.get("diagnostics", {}) or {}),
    )


def build_calibration_guidance(
    policy: Mapping[str, Any],
    source_family: str,
    *,
    example: Optional[JudgeBenchJoinedExample] = None,
    reference_answer_access: bool = True,
) -> str:
    uses_judgebench_tuning = _protocol_uses_judgebench_tuning(policy)
    blind_guidance_profile = _policy_blind_guidance_profile(policy)
    if not uses_judgebench_tuning and blind_guidance_profile == _BLIND_GUIDANCE_PROFILE_OFF:
        return ""
    nudges = list((policy.get("prompt_nudges", {}) or {}).get("global", []))
    nudges.extend(list((policy.get("prompt_nudges", {}) or {}).get(source_family, [])))
    if example is not None and (uses_judgebench_tuning or blind_guidance_profile != _BLIND_GUIDANCE_PROFILE_OFF):
        nudges.extend(
            _source_family_nudges(
                _example_for_reference_access(example, reference_answer_access=reference_answer_access)
            )
        )
    ordered = _ordered_unique_nudges(nudges)
    if not ordered:
        return ""
    lines = ["Additional calibration guidance for this task family:"]
    lines.extend(f"- {item}" for item in ordered[:6])
    return "\n".join(lines)


def _retrieval_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _RETRIEVAL_WORD_RE.findall((text or "").lower()):
        if len(token) < 4 or token in _RETRIEVAL_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _source_domain_tokens(source: str) -> set[str]:
    parts = [part for part in re.split(r"[^a-z0-9]+", (source or "").lower()) if part]
    return {
        part
        for part in parts
        if len(part) >= 4 and part not in {"mmlu", "livebench", "livecodebench", "reasoning", "math", "code"}
    }


def _mcq_option_letters(question: str) -> Tuple[str, ...]:
    letters = sorted(
        {
            group.upper()
            for match in re.finditer(r"\(([A-Ja-j])\)|(?:(?<=\s)|^)([A-Ja-j])[\)\.:]\s+", question or "")
            for group in match.groups()
            if group
        }
    )
    return tuple(letters)


def _question_word_count_bucket(question: str) -> str:
    count = len(_normalize_text(question).split())
    if count <= 20:
        return "short"
    if count <= 60:
        return "medium"
    return "long"


def _question_line_count_bucket(question: str) -> str:
    count = len([line for line in (question or "").splitlines() if line.strip()])
    if count <= 2:
        return "compact"
    if count <= 6:
        return "structured"
    return "multi_section"


def _question_shape_tags(example: JudgeBenchJoinedExample) -> set[str]:
    question = example.question or ""
    tags: set[str] = {
        f"word_bucket:{_question_word_count_bucket(question)}",
        f"line_bucket:{_question_line_count_bucket(question)}",
    }
    answer_mode = _example_requested_answer_mode(example)
    if answer_mode:
        tags.add(f"answer_mode:{answer_mode}")
    option_letters = _mcq_option_letters(question)
    if option_letters:
        tags.add("question:multiple_choice")
        tags.add(f"question:options:{min(len(option_letters), 6)}")
    if _example_is_exact_answer_task(example):
        tags.add("task:exact_answer")
    if _looks_like_code_task(example):
        tags.add("task:code")
    lowered = question.lower()
    for needle, tag in (
        ("let's think step by step", "prompt:step_by_step"),
        ("best guess", "prompt:best_guess"),
        ("single string", "format:single_string"),
        ("single digit", "format:single_digit"),
        ("single word", "format:single_word"),
        ("single phrase", "format:single_phrase"),
        ("duplicate that letter five times", "format:repeated_letter"),
        ("input", "code:input"),
        ("output", "code:output"),
        ("constraint", "question:constraints"),
        ("prove", "question:proof"),
    ):
        if needle in lowered:
            tags.add(tag)
    for token in sorted(_source_domain_tokens(example.source)):
        tags.add(f"source:{token}")
    return tags


def _option_structure_similarity(query: JudgeBenchJoinedExample, candidate: JudgeBenchJoinedExample) -> float:
    query_letters = _mcq_option_letters(query.question)
    candidate_letters = _mcq_option_letters(candidate.question)
    if not query_letters or not candidate_letters:
        return 0.0
    query_set = set(query_letters)
    candidate_set = set(candidate_letters)
    overlap = len(query_set & candidate_set)
    union = len(query_set | candidate_set) or 1
    score = overlap / union
    if len(query_letters) == len(candidate_letters):
        score += 0.25
    return min(score, 1.25)


def _shape_similarity(query: JudgeBenchJoinedExample, candidate: JudgeBenchJoinedExample) -> float:
    query_tags = _question_shape_tags(query)
    candidate_tags = _question_shape_tags(candidate)
    overlap = len(query_tags & candidate_tags)
    union = len(query_tags | candidate_tags) or 1
    return overlap / union


def _retrieval_v2_weights(query: JudgeBenchJoinedExample) -> Dict[str, float]:
    # MMLU questions are already restricted to same-family retrieval. Overweighting
    # subdomain / shape cues there caused the retriever to prefer superficially
    # similar MCQ templates over lexically closer exemplars that calibrated better.
    if query.source_family == "mmlu-pro":
        return {
            "lexical": 0.95,
            "same_family": 0.75,
            "answer_mode_match": 0.20,
            "exact_answer_match": 0.15,
            "code_task_match": 0.05,
            "option_structure": 0.08,
            "shape_similarity": 0.03,
            "source_domain": 0.0,
        }
    return {
        "lexical": 0.35,
        "same_family": 0.75,
        "answer_mode_match": 0.25,
        "exact_answer_match": 0.20,
        "code_task_match": 0.20,
        "option_structure": 0.30,
        "shape_similarity": 0.45,
        "source_domain": 0.25,
    }


def _retrieval_similarity_components(
    query: JudgeBenchJoinedExample,
    candidate: JudgeBenchJoinedExample,
    *,
    retrieval_profile: str,
) -> Dict[str, float]:
    query_tokens = _retrieval_tokens(f"{query.source}\n{query.question}")
    candidate_tokens = _retrieval_tokens(f"{candidate.source}\n{candidate.question}")
    token_overlap = len(query_tokens & candidate_tokens)
    token_union = len(query_tokens | candidate_tokens) or 1
    lexical = token_overlap / token_union
    same_family = 1.0 if query.source_family == candidate.source_family else 0.0
    answer_mode_match = float(
        _example_requested_answer_mode(query) == _example_requested_answer_mode(candidate)
    )
    exact_answer_match = float(
        _example_is_exact_answer_task(query) == _example_is_exact_answer_task(candidate)
    )
    code_task_match = float(_looks_like_code_task(query) and _looks_like_code_task(candidate))
    option_structure = _option_structure_similarity(query, candidate)
    shape_similarity = _shape_similarity(query, candidate)
    source_tokens_query = _source_domain_tokens(query.source)
    source_tokens_candidate = _source_domain_tokens(candidate.source)
    source_domain = 0.0
    if source_tokens_query and source_tokens_candidate:
        source_domain = len(source_tokens_query & source_tokens_candidate) / max(
            1,
            len(source_tokens_query | source_tokens_candidate),
        )
    if retrieval_profile == _RETRIEVAL_PROFILE_FAMILY_QUESTION_V2:
        weights = _retrieval_v2_weights(query)
        total = (
            lexical * weights["lexical"]
            + same_family * weights["same_family"]
            + answer_mode_match * weights["answer_mode_match"]
            + exact_answer_match * weights["exact_answer_match"]
            + code_task_match * weights["code_task_match"]
            + option_structure * weights["option_structure"]
            + shape_similarity * weights["shape_similarity"]
            + source_domain * weights["source_domain"]
        )
    else:
        total = lexical
        total += same_family * 0.25
        total += answer_mode_match * 0.05
        total += exact_answer_match * 0.05
        total += code_task_match * 0.05
    return {
        "lexical": round(float(lexical), 6),
        "same_family": round(float(same_family), 6),
        "answer_mode_match": round(float(answer_mode_match), 6),
        "exact_answer_match": round(float(exact_answer_match), 6),
        "code_task_match": round(float(code_task_match), 6),
        "option_structure": round(float(option_structure), 6),
        "shape_similarity": round(float(shape_similarity), 6),
        "source_domain": round(float(source_domain), 6),
        "total": round(float(total), 6),
    }


def _retrieval_similarity(
    query: JudgeBenchJoinedExample,
    candidate: JudgeBenchJoinedExample,
    *,
    retrieval_profile: str,
) -> float:
    return float(
        _retrieval_similarity_components(
            query,
            candidate,
            retrieval_profile=retrieval_profile,
        )["total"]
    )


def _preferred_pair_response(example: JudgeBenchJoinedExample) -> str:
    return example.response_A if example.label == "A>B" else example.response_B


def _weaker_pair_response(example: JudgeBenchJoinedExample) -> str:
    return example.response_B if example.label == "A>B" else example.response_A


def _trim_preview(text: str, *, limit: int = 140) -> str:
    preview = _normalize_text(text)
    if len(preview) <= limit:
        return preview
    return preview[: max(0, limit - 3)].rstrip() + "..."


def _retrieval_exemplar_summary(
    query: JudgeBenchJoinedExample,
    example: JudgeBenchJoinedExample,
    *,
    retrieval_profile: str,
) -> str:
    preferred = _preferred_pair_response(example)
    weaker = _weaker_pair_response(example)
    preferred_answer = _extract_exact_answer_candidate_for_example(example, preferred)
    weaker_answer = _extract_exact_answer_candidate_for_example(example, weaker)
    question_preview = _trim_preview(example.question, limit=120)
    preferred_signal = _blind_exact_answer_signal(example, preferred)
    weaker_signal = _blind_exact_answer_signal(example, weaker)
    if example.source_family == "livecodebench":
        return (
            f"Similar code task `{question_preview}`: the preferred training response preserved the required program "
            "behavior, I/O contract, and prompt constraints; the weaker response lost behavioral coverage or drifted "
            "toward style-only differences."
        )
    if example.source_family == "livebench-reasoning":
        return (
            f"Similar reasoning task `{question_preview}`: the preferred training response kept clue/state tracking "
            "consistent and ended with a supported conclusion, while the weaker response introduced contradiction or "
            "unsupported deduction."
        )
    if preferred_answer is not None and weaker_answer is not None and preferred_answer.value != weaker_answer.value:
        if retrieval_profile == _RETRIEVAL_PROFILE_FAMILY_QUESTION_V2 and (
            preferred_signal.consistent != weaker_signal.consistent
            or preferred_signal.format_ok != weaker_signal.format_ok
        ):
            return (
                f"Similar `{example.source_family}` task `{question_preview}`: the preferred response made `{preferred_answer.value}` "
                "explicit and kept it consistent with the reasoning/format, while the weaker response let the final "
                f"answer drift toward `{weaker_answer.value}` or an unsupported conclusion."
            )
        return (
            f"Similar `{example.source_family}` task `{question_preview}`: the preferred training response ended with "
            f"`{preferred_answer.value}` while the weaker response ended with `{weaker_answer.value}`; prioritize "
            "whether the final answer is actually supported by the worked reasoning."
        )
    if retrieval_profile == _RETRIEVAL_PROFILE_FAMILY_QUESTION_V2 and preferred_signal.explicit and not weaker_signal.explicit:
        return (
            f"Similar `{example.source_family}` task `{question_preview}`: the preferred response made its final answer "
            "explicit and supported, while the weaker response left the choice implicit or under-specified."
        )
    if preferred_answer is not None:
        return (
            f"Similar `{example.source_family}` task `{question_preview}`: the preferred training response made its "
            f"final answer explicit as `{preferred_answer.value}` and kept it aligned with the reasoning rather than "
            "relying on broad completeness."
        )
    return (
        f"Similar `{example.source_family}` task `{question_preview}`: the preferred training response aligned its "
        "final answer with the supporting reasoning, while the weaker response did not preserve that comparison logic."
    )


def _retrieval_diversity_signature(example: JudgeBenchJoinedExample) -> str:
    answer_mode = _example_requested_answer_mode(example) or "none"
    option_letters = _mcq_option_letters(example.question)
    source_bucket = example.source or example.source_family
    return f"{example.source_family}|{source_bucket}|{answer_mode}|{min(len(option_letters), 6)}"


def _select_retrieval_examples(
    scored: Sequence[Tuple[float, JudgeBenchJoinedExample]],
    *,
    top_k: int,
    retrieval_profile: str,
) -> List[Tuple[float, JudgeBenchJoinedExample]]:
    if retrieval_profile != _RETRIEVAL_PROFILE_FAMILY_QUESTION_V2:
        return [(score, candidate) for score, candidate in scored[:top_k] if score > 0.0]
    selected: List[Tuple[float, JudgeBenchJoinedExample]] = []
    seen_signatures: set[str] = set()
    for score, candidate in scored:
        if score <= 0.0:
            continue
        signature = _retrieval_diversity_signature(candidate)
        if signature in seen_signatures:
            continue
        selected.append((score, candidate))
        seen_signatures.add(signature)
        if len(selected) >= top_k:
            return selected
    if len(selected) >= top_k:
        return selected
    for score, candidate in scored:
        if score <= 0.0:
            continue
        if any(candidate.pair_id == existing.pair_id for _, existing in selected):
            continue
        selected.append((score, candidate))
        if len(selected) >= top_k:
            break
    return selected


def _retrieval_exemplar_focus_kinds(example: JudgeBenchJoinedExample) -> List[str]:
    focus: List[str] = []
    if example.source_family == "livecodebench":
        focus.append("code_behavior_constraints")
    if example.source_family == "livebench-reasoning":
        focus.append("reasoning_state_tracking")
    if _example_is_exact_answer_task(example):
        focus.append("exact_answer_format")
        preferred_signal = _blind_exact_answer_signal(example, _preferred_pair_response(example))
        weaker_signal = _blind_exact_answer_signal(example, _weaker_pair_response(example))
        if (
            preferred_signal.consistent != weaker_signal.consistent
            or preferred_signal.format_ok != weaker_signal.format_ok
            or weaker_signal.conflicting_markers
        ):
            focus.append("exact_answer_consistency")
            if _example_choice_value_map(example):
                focus.append("choice_value_consistency")
    if not focus:
        focus.append("final_answer_supported")
    return focus


def _retrieval_seed_row_from_focus(
    example: JudgeBenchJoinedExample,
    *,
    focus_kind: str,
    count: int,
) -> Optional[Dict[str, Any]]:
    if focus_kind == "exact_answer_format":
        row = _blind_exact_signal_row(example, kind="format")
        row["label"] = "Retrieved exemplar focus: final answer is explicit in the requested format"
        row["requirement"] = (
            "Across retrieved training exemplars, the preferred response made the final answer explicit in the "
            "requested format instead of leaving the choice implicit in the reasoning."
        )
    elif focus_kind == "exact_answer_consistency":
        row = _blind_exact_signal_row(example, kind="consistency")
        row["label"] = "Retrieved exemplar focus: final answer markers stay internally consistent"
        row["requirement"] = (
            "Across retrieved training exemplars, the preferred response kept its explicit answer markers aligned "
            "across the conclusion, last line, and final-answer cue rather than contradicting itself."
        )
    elif focus_kind == "choice_value_consistency":
        if not _example_choice_value_map(example):
            return None
        row = _blind_exact_signal_row(example, kind="choice_value_consistency")
        row["label"] = "Retrieved exemplar focus: option letter and option value agree"
        row["requirement"] = (
            "Across retrieved training exemplars, the preferred response kept any explicit option letter, option "
            "value, and final answer string aligned with the same choice."
        )
    elif focus_kind == "reasoning_state_tracking":
        row = {
            "dimension": "constraint_satisfaction",
            "label": "Retrieved exemplar focus: keep clue and state tracking consistent",
            "requirement": (
                "Across retrieved reasoning exemplars, the preferred response tracked the clue/state updates "
                "consistently and avoided unsupported deductions in the final conclusion."
            ),
            "severity_tier": "high",
            "count": 0,
        }
    elif focus_kind == "code_behavior_constraints":
        row = {
            "dimension": "executable_correctness",
            "label": "Retrieved exemplar focus: preserve required behavior and I/O constraints",
            "requirement": (
                "Across retrieved code exemplars, the preferred response respected the required program behavior, "
                "input/output contract, and prompt-stated constraints rather than style-only preferences."
            ),
            "severity_tier": "high",
            "count": 0,
        }
    else:
        row = {
            "dimension": "final_answer_correctness",
            "label": "Retrieved exemplar focus: final answer is supported by the reasoning",
            "requirement": (
                "Across retrieved exemplars, the preferred response tied the final answer directly to the supporting "
                "reasoning instead of relying on broad completeness."
            ),
            "severity_tier": "high",
            "count": 0,
        }
    row["count"] = max(int(row.get("count", 0) or 0), int(count))
    row["retrieval_seed_row"] = True
    row["retrieval_seed_focus_kind"] = focus_kind
    return row


def _build_retrieval_seed_rows(
    example: JudgeBenchJoinedExample,
    selected_examples: Sequence[JudgeBenchJoinedExample],
) -> List[Dict[str, Any]]:
    focus_counts: Counter[str] = Counter()
    for candidate in selected_examples:
        focus_counts.update(_retrieval_exemplar_focus_kinds(candidate))
    rows: List[Dict[str, Any]] = []
    for focus_kind, count in sorted(focus_counts.items()):
        row = _retrieval_seed_row_from_focus(example, focus_kind=focus_kind, count=count)
        if row is not None:
            rows.append(row)
    return rows


_RUBRIC_LIBRARY_CACHE: Dict[str, Optional[RubricLibrary]] = {}


def _load_policy_rubric_library(policy: Mapping[str, Any]) -> Optional[RubricLibrary]:
    library_path = str(policy.get("rubric_library_path", "") or "").strip()
    if not library_path:
        return None
    if library_path in _RUBRIC_LIBRARY_CACHE:
        return _RUBRIC_LIBRARY_CACHE[library_path]
    try:
        library = load_rubric_library(Path(library_path))
    except Exception:
        library = None
    _RUBRIC_LIBRARY_CACHE[library_path] = library
    return library


def _maybe_library_rows_for_example(
    *,
    example: JudgeBenchJoinedExample,
    example_record: ExampleRecord,
    policy: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    family_top_k = _policy_library_retrieval_top_k(policy, example.source_family)
    profile = _policy_retrieval_profile(policy, example.source_family)
    uses_library = family_top_k > 0 or profile in {
        _RETRIEVAL_PROFILE_LIBRARY_V1,
        _RETRIEVAL_PROFILE_LIBRARY_V1_PLUS_FAMILY_V1,
    }
    # Per-family override of 0 wins even when the retrieval profile would otherwise inject.
    family_overrides = policy.get("library_retrieval_top_k_by_family") or {}
    family_key = _normalize_source_family_key(example.source_family)
    if family_key in family_overrides:
        try:
            override_value = int(family_overrides[family_key] or 0)
        except (TypeError, ValueError):
            override_value = family_top_k
        if override_value <= 0:
            return []
    if not uses_library:
        return []
    library = _load_policy_rubric_library(policy)
    if library is None:
        return []
    effective_top_k = max(family_top_k or 0, 4)
    effective_top_k = min(effective_top_k, 12)
    strict = _policy_family_strict_library(policy)
    criteria = library.filter_by_family(
        example.source_family,
        limit=effective_top_k,
        strict=strict,
    )
    rows: List[Dict[str, Any]] = []
    for criterion in criteria:
        row = criterion.to_canonical_row(
            example_id=example_record.example_id,
            pair_id=str(example.pair_id),
        )
        rows.append(row)
    return rows


def _build_pair_contexts_for_rrd(raw: Sequence[Mapping[str, Any]]) -> List[_RrdPairContext]:
    contexts: List[_RrdPairContext] = []
    for payload in raw or []:
        pair_id = str(payload.get("pair_id", "") or "")
        if not pair_id:
            continue
        contexts.append(
            _RrdPairContext(
                pair_id=pair_id,
                strong_text=str(payload.get("strong_text", "") or ""),
                weak_text=str(payload.get("weak_text", "") or ""),
            )
        )
    return contexts


def _build_retrieval_guidance(
    *,
    example: JudgeBenchJoinedExample,
    retrieval_examples: Sequence[JudgeBenchJoinedExample],
    policy: Mapping[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    retrieval_profile = _policy_retrieval_profile(policy, example.source_family)
    if retrieval_profile == _RETRIEVAL_PROFILE_OFF:
        return "", [], []
    if retrieval_profile == _RETRIEVAL_PROFILE_LIBRARY_V1:
        return "", [], []
    if retrieval_profile == _RETRIEVAL_PROFILE_LIBRARY_V1_PLUS_FAMILY_V1:
        retrieval_profile = _RETRIEVAL_PROFILE_FAMILY_QUESTION_V1
    candidates = [row for row in retrieval_examples if row.pair_id != example.pair_id]
    if not candidates:
        return "", [], []
    same_family = [row for row in candidates if row.source_family == example.source_family]
    pool = same_family or candidates
    scored = sorted(
        (
            (
                _retrieval_similarity(
                    example,
                    candidate,
                    retrieval_profile=retrieval_profile,
                ),
                candidate,
            )
            for candidate in pool
        ),
        key=lambda item: (-item[0], item[1].pair_id),
    )
    top_k = _policy_retrieval_top_k(policy, example.source_family)
    if retrieval_profile == _RETRIEVAL_PROFILE_FAMILY_QUESTION_SEED_V1:
        top_k = max(top_k, 3)
    selected = _select_retrieval_examples(
        scored,
        top_k=top_k,
        retrieval_profile=retrieval_profile,
    )
    if not selected:
        return "", [], []
    if retrieval_profile == _RETRIEVAL_PROFILE_FAMILY_QUESTION_SEED_V1:
        lines = [
            "Retrieved training exemplars for calibration and rubric seeding; transfer the comparison structure that "
            "distinguished the preferred response, not the answer itself:"
        ]
    else:
        lines = [
            "Retrieved training exemplars for calibration only; use them as style and consistency cues rather than as a direct answer lookup:"
        ]
    hits: List[Dict[str, Any]] = []
    for score, candidate in selected:
        focus_kinds = _retrieval_exemplar_focus_kinds(candidate)
        summary = _retrieval_exemplar_summary(
            example,
            candidate,
            retrieval_profile=retrieval_profile,
        )
        similarity_components = _retrieval_similarity_components(
            example,
            candidate,
            retrieval_profile=retrieval_profile,
        )
        if retrieval_profile == _RETRIEVAL_PROFILE_FAMILY_QUESTION_SEED_V1:
            summary = f"{summary} Focus transfer: {', '.join(focus_kinds)}."
        lines.append(f"- {summary}")
        hits.append(
            {
                "pair_id": candidate.pair_id,
                "source_family": candidate.source_family,
                "similarity": round(float(score), 6),
                "focus_kinds": focus_kinds,
                "similarity_components": similarity_components,
            }
        )
    seed_rows = (
        _build_retrieval_seed_rows(example, [candidate for _, candidate in selected])
        if retrieval_profile == _RETRIEVAL_PROFILE_FAMILY_QUESTION_SEED_V1
        else []
    )
    return "\n".join(lines), hits, seed_rows


def build_judgebench_candidates(
    *,
    example: JudgeBenchJoinedExample,
    example_record: ExampleRecord,
    route_decision: JudgeBenchRouteDecision,
    max_pairs_per_example: Optional[int],
    reference_answer_access: bool = True,
    policy: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Tuple[CandidateNote, CandidateNote]], List[CandidateNote], List[CandidateNote]]:
    profile = get_task_profile(route_decision.task_profile_id)
    pair_candidates = [
        CandidateNote(
            candidate_id=f"{example_record.example_id}__response_A",
            example_id=example_record.example_id,
            text=example.response_A,
            source_label="response_A",
            quality_bucket="pair_candidate",
            origin_kind="judgebench_pair",
            metadata={"pair_position": "A", "response_model": example.response_model},
            artifact_kind=route_decision.artifact_kind,
            task_profile_id=route_decision.task_profile_id,
            task_family_id=route_decision.task_family_id,
        ),
        CandidateNote(
            candidate_id=f"{example_record.example_id}__response_B",
            example_id=example_record.example_id,
            text=example.response_B,
            source_label="response_B",
            quality_bucket="pair_candidate",
            origin_kind="judgebench_pair",
            metadata={"pair_position": "B", "response_model": example.response_model},
            artifact_kind=route_decision.artifact_kind,
            task_profile_id=route_decision.task_profile_id,
            task_family_id=route_decision.task_family_id,
        ),
    ]
    strategy = get_contrast_strategy(profile.contrast_strategy_id)
    ordered_mutation_ids = _ordered_blind_mutation_ids(
        strategy.mutation_ids,
        source_family=example.source_family,
        task_profile_id=route_decision.task_profile_id,
        budget_profile=_policy_blind_budget_profile(policy or {}),
    )
    synthetic_candidates: List[CandidateNote] = []
    discovery_pairs: List[Tuple[CandidateNote, CandidateNote]] = []

    if reference_answer_access:
        strong = CandidateNote(
            candidate_id=f"{example_record.example_id}__reference_answer",
            example_id=example_record.example_id,
            text=example.reference_answer,
            source_label="reference_answer",
            quality_bucket="reference",
            origin_kind="judgebench_reference",
            metadata={"split": example.split_name},
            artifact_kind=route_decision.artifact_kind,
            task_profile_id=route_decision.task_profile_id,
            task_family_id=route_decision.task_family_id,
        )
        seen_text = {strong.text.strip(), *(candidate.text.strip() for candidate in pair_candidates if candidate.text.strip())}
        extra_budget = None if max_pairs_per_example is None else max(0, max_pairs_per_example - len(pair_candidates))
        for mutation_id in ordered_mutation_ids:
            if extra_budget is not None and len(synthetic_candidates) >= extra_budget:
                break
            fn = mutation_function_for_id(mutation_id)
            if fn is None:
                continue
            mutated = fn(strong.text)
            mutated_text = (mutated or "").strip()
            if not mutated_text or mutated_text == strong.text.strip() or mutated_text in seen_text:
                continue
            seen_text.add(mutated_text)
            synthetic = CandidateNote(
                candidate_id=f"{example_record.example_id}__synthetic_{mutation_id}",
                example_id=example_record.example_id,
                text=mutated_text,
                source_label=f"synthetic_mutation:{mutation_id}",
                quality_bucket="synthetically_degraded",
                origin_kind="synthetic_mutation",
                metadata={
                    "mutation_id": mutation_id,
                    "synthetic": True,
                    "anchor_preview": strong.text[:200],
                },
                artifact_kind=route_decision.artifact_kind,
                task_profile_id=route_decision.task_profile_id,
                task_family_id=route_decision.task_family_id,
            )
            synthetic_candidates.append(synthetic)
        discovery_pairs = [(strong, weak) for weak in [*pair_candidates, *synthetic_candidates]]
        scoring_candidates = [strong, *pair_candidates, *synthetic_candidates]
        return discovery_pairs, pair_candidates, scoring_candidates

    seen_text = {candidate.text.strip() for candidate in pair_candidates if candidate.text.strip()}
    direct_pair_candidates: List[Tuple[CandidateNote, CandidateNote]] = []
    if pair_candidates[0].text.strip() != pair_candidates[1].text.strip():
        direct_pair_candidates = [
            (pair_candidates[0], pair_candidates[1]),
            (pair_candidates[1], pair_candidates[0]),
        ]
    discovery_pairs.extend(direct_pair_candidates)
    total_scoring_candidate_budget = _blind_scoring_candidate_budget(
        example.source_family,
        route_decision.task_profile_id,
        max_pairs_per_example,
        budget_profile=_policy_blind_budget_profile(policy or {}),
    )
    synthetic_budget = max(0, total_scoring_candidate_budget - len(pair_candidates))
    for mutation_id in ordered_mutation_ids:
        fn = mutation_function_for_id(mutation_id)
        if fn is None:
            continue
        for anchor in pair_candidates:
            if synthetic_budget is not None and len(synthetic_candidates) >= synthetic_budget:
                break
            mutated = fn(anchor.text)
            mutated_text = (mutated or "").strip()
            if not mutated_text or mutated_text == anchor.text.strip() or mutated_text in seen_text:
                continue
            seen_text.add(mutated_text)
            anchor_slug = _safe_slug(str(anchor.metadata.get("pair_position", anchor.source_label)))
            synthetic = CandidateNote(
                candidate_id=f"{example_record.example_id}__synthetic_{anchor_slug}_{mutation_id}",
                example_id=example_record.example_id,
                text=mutated_text,
                source_label=f"synthetic_mutation:{mutation_id}:{anchor.source_label}",
                quality_bucket="synthetically_degraded",
                origin_kind="synthetic_mutation",
                metadata={
                    "mutation_id": mutation_id,
                    "synthetic": True,
                    "anchor_source_label": anchor.source_label,
                    "anchor_preview": anchor.text[:200],
                    "reference_answer_access": False,
                },
                artifact_kind=route_decision.artifact_kind,
                task_profile_id=route_decision.task_profile_id,
                task_family_id=route_decision.task_family_id,
            )
            synthetic_candidates.append(synthetic)
            discovery_pairs.append((anchor, synthetic))
        if synthetic_budget is not None and len(synthetic_candidates) >= synthetic_budget:
            break

    if not discovery_pairs and pair_candidates[0].text.strip() != pair_candidates[1].text.strip():
        discovery_pairs = [
            (pair_candidates[0], pair_candidates[1]),
            (pair_candidates[1], pair_candidates[0]),
        ]
    scoring_candidates = [*pair_candidates, *synthetic_candidates]
    return discovery_pairs, pair_candidates, scoring_candidates


def _profile_budget_override(task_profile_id: str, *, source_family: str, budget_profile: str) -> Optional[int]:
    normalized = str(task_profile_id or "").strip().lower()
    if budget_profile == _BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V1:
        if source_family == "livebench-reasoning" and any(
            token in normalized for token in ("person_truth_says", "person_following_likes")
        ):
            return 3
        return None
    if budget_profile == _BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V2:
        if source_family == "livebench-reasoning":
            return 5
        if source_family == "mmlu-pro" and "answer_your_step" in normalized:
            return 5
    return None


def _ordered_blind_mutation_ids(
    mutation_ids: Sequence[str],
    *,
    source_family: str,
    task_profile_id: str,
    budget_profile: str,
) -> List[str]:
    priority: Tuple[str, ...] = ()
    normalized = str(task_profile_id or "").strip().lower()
    if budget_profile == _BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V1:
        if source_family == "livebench-reasoning" and any(
            token in normalized for token in ("person_truth_says", "person_following_likes")
        ):
            priority = (
                "drop_constraints",
                "corrupt_final_answer",
                "drop_steps",
                "drop_supporting_evidence",
                "remove_format_markers",
                "add_unsupported_detail",
            )
        elif source_family == "mmlu-pro" and "answer_your_step" in normalized:
            priority = (
                "corrupt_final_answer",
                "remove_format_markers",
                "drop_constraints",
                "drop_supporting_evidence",
                "add_unsupported_detail",
            )
    elif budget_profile == _BLIND_BUDGET_PROFILE_FAMILY_PROFILE_V2:
        if source_family == "livebench-reasoning":
            priority = (
                "drop_constraints",
                "drop_supporting_evidence",
                "add_unsupported_detail",
                "corrupt_final_answer",
                "drop_steps",
                "remove_format_markers",
            )
        elif source_family == "mmlu-pro" and "answer_your_step" in normalized:
            priority = (
                "drop_constraints",
                "drop_supporting_evidence",
                "corrupt_final_answer",
                "add_unsupported_detail",
                "remove_format_markers",
            )
    if not priority:
        return list(mutation_ids)
    priority_index = {mutation_id: index for index, mutation_id in enumerate(priority)}
    original_index = {mutation_id: index for index, mutation_id in enumerate(mutation_ids)}
    return sorted(
        mutation_ids,
        key=lambda mutation_id: (
            priority_index.get(mutation_id, len(priority) + original_index[mutation_id]),
            original_index[mutation_id],
        ),
    )


def _blind_scoring_candidate_budget(
    source_family: str,
    task_profile_id: str,
    max_pairs_per_example: Optional[int],
    *,
    budget_profile: str,
) -> int:
    family_budget = int(_BLIND_SOURCE_FAMILY_SCORING_CANDIDATE_BUDGET.get(source_family, 4))
    profile_budget = _profile_budget_override(
        task_profile_id,
        source_family=source_family,
        budget_profile=budget_profile,
    )
    if profile_budget is not None:
        family_budget = int(profile_budget)
    if max_pairs_per_example is None:
        return max(2, family_budget)
    return max(2, min(int(max_pairs_per_example), family_budget))


def render_satisfaction_prompt(response_text: str, rubric_text: str) -> str:
    return (
        "Judge prompt for rubric-based response evaluation.\n"
        "You are a judge, evaluating whether a response satisfies the given rubric. If the response satisfies the\n"
        "criterion of the rubric, output YES; otherwise output NO.\n"
        "Requirement:\n"
        "• You must follow the rubric strictly, and only consider the criteria listed in the rubric.\n"
        "• You must NOT consider any other factors, such as your own opinions or external knowledge.\n"
        "Below between <RESPONSE> and </RESPONSE> is the response to evaluate on\n"
        f"• <RESPONSE> {response_text} </RESPONSE>\n"
        "Below between <RUBRIC> and </RUBRIC> is the rubric to evaluate on\n"
        f"• <RUBRIC> {rubric_text} </RUBRIC>\n"
        "Output STRICTLY in below format. No other text is allowed:\n"
        "• <EVALUATION> YES/NO </EVALUATION>"
    )


def render_satisfaction_retry_prompt(response_text: str, rubric_text: str) -> str:
    return (
        f"{render_satisfaction_prompt(response_text, rubric_text)}\n\n"
        "IMPORTANT: Your previous answer did not follow the required output format.\n"
        "Respond again.\n"
        "Output ONLY one of the following and nothing else:\n"
        "<EVALUATION> YES </EVALUATION>\n"
        "<EVALUATION> NO </EVALUATION>"
    )


_UNCERTAIN_VERDICT_RE = re.compile(
    r"<EVALUATION>\s*(UNKNOWN|UNCLEAR|CANNOT[_\s]?ASSESS|UNDETERMINED|N/A)\s*</EVALUATION>",
    re.IGNORECASE | re.DOTALL,
)


def _extract_yes_no(raw_text: str) -> Optional[bool]:
    text = raw_text or ""
    match = EVALUATION_RE.search(text)
    if match:
        return match.group(1).strip().upper() == "YES"
    if _UNCERTAIN_VERDICT_RE.search(text):
        return False
    verdict, _ = parse_yes_no(text)
    return verdict


def _generation_cache_key(
    *,
    stage: str,
    model: ModelSpec,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str = "",
) -> str:
    return make_cache_key(
        stage,
        {
            "model": {
                "provider": model.provider,
                "model": model.model,
                "base_url": model.base_url or "",
            },
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )


def _generate_payload(
    *,
    model: ModelSpec,
    user_prompt: str,
    router: LLMRouter,
    temperature: float,
    max_tokens: int,
    system_prompt: str = "",
) -> Dict[str, Any]:
    response = router.generate(
        model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "text": response.text,
        "raw_text": response.raw_text,
        "metadata": dict(response.metadata),
    }


def _cached_generation(
    *,
    cache: JsonlCache,
    stage: str,
    model: ModelSpec,
    user_prompt: str,
    router: LLMRouter,
    temperature: float,
    max_tokens: int,
    system_prompt: str = "",
) -> Tuple[str, Dict[str, Any], bool]:
    cache_key = _generation_cache_key(
        stage=stage,
        model=model,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cache_key, cached, True
    payload = _generate_payload(
        model=model,
        user_prompt=user_prompt,
        router=router,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    cache.set(cache_key, payload)
    return cache_key, payload, False


def _evaluate_rubric_satisfaction_single(
    *,
    response_text: str,
    rubric_text: str,
    judge_model: ModelSpec,
    cache: JsonlCache,
    router: LLMRouter,
    sample_index: int = 0,
    temperature: float = 0.0,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    sample_label = "" if sample_index == 0 and temperature == 0.0 else f"_s{int(sample_index)}_t{int(round(float(temperature) * 100))}"
    attempts = [
        (
            f"rubric_satisfaction{sample_label}",
            render_satisfaction_prompt(response_text, rubric_text),
            64,
        ),
        (
            f"rubric_satisfaction_retry_1{sample_label}",
            render_satisfaction_retry_prompt(response_text, rubric_text),
            32,
        ),
        (
            f"rubric_satisfaction_retry_2{sample_label}",
            render_satisfaction_retry_prompt(response_text, rubric_text),
            32,
        ),
    ]
    last_payload: Optional[Dict[str, Any]] = None
    cache_hits = 0
    for attempt_index, (stage, prompt, max_tokens) in enumerate(attempts):
        cache_key, payload, cache_hit = _cached_generation(
            cache=cache,
            stage=stage,
            model=judge_model,
            user_prompt=prompt,
            router=router,
            temperature=float(temperature),
            max_tokens=max_tokens,
        )
        if cache_hit:
            cache_hits += 1
        last_payload = payload
        verdict = _extract_yes_no(str(payload.get("text", "")))
        if verdict is None and cache_hit:
            refreshed_payload = _generate_payload(
                model=judge_model,
                user_prompt=prompt,
                router=router,
                temperature=float(temperature),
                max_tokens=max_tokens,
            )
            last_payload = refreshed_payload
            refreshed_verdict = _extract_yes_no(str(refreshed_payload.get("text", "")))
            if refreshed_verdict is not None:
                cache.set(cache_key, refreshed_payload)
                return refreshed_verdict, {
                    "attempt_index": attempt_index,
                    "cache_hits": cache_hits,
                    "raw_response": refreshed_payload.get("raw_text", refreshed_payload.get("text", "")),
                    "sample_index": int(sample_index),
                    "temperature": round(float(temperature), 4),
                }
        if verdict is None:
            continue
        return verdict, {
            "attempt_index": attempt_index,
            "cache_hits": cache_hits,
            "raw_response": payload.get("raw_text", payload.get("text", "")),
            "sample_index": int(sample_index),
            "temperature": round(float(temperature), 4),
        }
    preview = _normalize_text(str((last_payload or {}).get("text", "")))[:200]
    return None, {
        "attempt_index": -1,
        "cache_hits": cache_hits,
        "raw_response": preview,
        "parse_error": "unparseable_evaluation",
        "sample_index": int(sample_index),
        "temperature": round(float(temperature), 4),
    }


def evaluate_rubric_satisfaction(
    *,
    response_text: str,
    rubric_text: str,
    judge_model: ModelSpec,
    cache: JsonlCache,
    router: LLMRouter,
    samples: int = 1,
    temperature: float = 0.4,
) -> Tuple[bool, Dict[str, Any]]:
    samples = max(1, int(samples or 1))
    if samples == 1:
        verdict, meta = _evaluate_rubric_satisfaction_single(
            response_text=response_text,
            rubric_text=rubric_text,
            judge_model=judge_model,
            cache=cache,
            router=router,
            sample_index=0,
            temperature=0.0,
        )
        if verdict is None:
            return False, {
                **meta,
                "fallback_verdict": False,
                "judge_model": f"{judge_model.provider}:{judge_model.model}",
            }
        return verdict, meta

    yes_votes = 0
    no_votes = 0
    abstain_votes = 0
    sample_metas: List[Dict[str, Any]] = []
    primary_meta: Optional[Dict[str, Any]] = None
    for idx in range(samples):
        sample_temp = 0.0 if idx == 0 else max(0.1, float(temperature))
        verdict, meta = _evaluate_rubric_satisfaction_single(
            response_text=response_text,
            rubric_text=rubric_text,
            judge_model=judge_model,
            cache=cache,
            router=router,
            sample_index=idx,
            temperature=sample_temp,
        )
        sample_metas.append(meta)
        if idx == 0:
            primary_meta = meta
        if verdict is True:
            yes_votes += 1
        elif verdict is False:
            no_votes += 1
        else:
            abstain_votes += 1

    if yes_votes == 0 and no_votes == 0:
        return False, {
            "attempt_index": -1,
            "cache_hits": sum(int(m.get("cache_hits", 0) or 0) for m in sample_metas),
            "raw_response": (primary_meta or {}).get("raw_response", ""),
            "parse_error": "unparseable_evaluation",
            "fallback_verdict": False,
            "judge_model": f"{judge_model.provider}:{judge_model.model}",
            "self_consistency_samples": samples,
            "self_consistency_temperature": float(temperature),
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "abstain_votes": abstain_votes,
        }
    final = yes_votes > no_votes
    return final, {
        "attempt_index": (primary_meta or {}).get("attempt_index", 0),
        "cache_hits": sum(int(m.get("cache_hits", 0) or 0) for m in sample_metas),
        "raw_response": (primary_meta or {}).get("raw_response", ""),
        "self_consistency_samples": samples,
        "self_consistency_temperature": float(temperature),
        "yes_votes": yes_votes,
        "no_votes": no_votes,
        "abstain_votes": abstain_votes,
    }


def evaluate_rubrics_on_candidates(
    *,
    example: JudgeBenchJoinedExample,
    rubrics: Sequence[RubricCriterion],
    candidates: Sequence[CandidateNote],
    judge_model: ModelSpec,
    cache: JsonlCache,
    router: LLMRouter,
    protocol_mode: str = _DEFAULT_PROTOCOL_MODE,
    rubric_satisfaction_samples: int = 1,
    rubric_satisfaction_temperature: float = 0.4,
) -> Tuple[List[RubricEvaluation], Dict[str, int]]:
    evaluations: List[RubricEvaluation] = []
    stats = {"cache_hits": 0, "evaluations_total": 0, "deterministic_evaluations": 0}
    for candidate in candidates:
        for rubric in rubrics:
            deterministic = _deterministic_rubric_satisfaction(
                example=example,
                rubric=rubric,
                candidate=candidate,
                protocol_mode=protocol_mode,
            )
            if deterministic is not None:
                satisfied, meta = deterministic
                stats["evaluations_total"] += 1
                stats["deterministic_evaluations"] += 1
                evaluations.append(
                    RubricEvaluation(
                        rubric_id=rubric.rubric_id,
                        candidate_id=candidate.candidate_id,
                        satisfied=satisfied,
                        raw_response="",
                        metadata=meta,
                    )
                )
                continue
            satisfied, meta = evaluate_rubric_satisfaction(
                response_text=candidate.text,
                rubric_text=rubric.text,
                judge_model=judge_model,
                cache=cache,
                router=router,
                samples=int(rubric_satisfaction_samples or 1),
                temperature=float(rubric_satisfaction_temperature or 0.4),
            )
            stats["cache_hits"] += int(meta.get("cache_hits", 0))
            stats["evaluations_total"] += 1
            evaluations.append(
                RubricEvaluation(
                    rubric_id=rubric.rubric_id,
                    candidate_id=candidate.candidate_id,
                    satisfied=satisfied,
                    raw_response=str(meta.get("raw_response", "")),
                    metadata={
                        "attempt_index": meta.get("attempt_index", 0),
                        **(
                            {"self_consistency_samples": int(meta["self_consistency_samples"])}
                            if "self_consistency_samples" in meta
                            else {}
                        ),
                        **(
                            {"yes_votes": int(meta.get("yes_votes", 0)), "no_votes": int(meta.get("no_votes", 0))}
                            if "self_consistency_samples" in meta
                            else {}
                        ),
                    },
                )
            )
    return evaluations, stats


def _proposal_to_rubric_text(row: Mapping[str, Any]) -> str:
    label = _normalize_text(str(row.get("label", "")))
    requirement = _normalize_text(str(row.get("requirement", "")))
    if label and requirement and label.lower() not in requirement.lower():
        return f"{label}: {requirement}"
    return requirement or label


def _fallback_exact_answer_correctness_row(example: JudgeBenchJoinedExample) -> Optional[Dict[str, Any]]:
    reference_value = _normalize_exact_answer_value(example.reference_answer)
    if not reference_value:
        return None
    reference_display = _normalize_text(example.reference_answer) or reference_value
    if _is_repeated_choice_value(reference_value):
        label = "Final answer matches the correct repeated-letter answer"
    elif reference_value.isdigit():
        label = "Final answer matches the correct digit"
    elif "," in reference_value and all(part.strip() in {"yes", "no"} for part in reference_value.split(",")):
        label = "Final answer matches the correct yes/no list"
    else:
        label = "Final answer matches the correct exact answer"
    return {
        "dimension": "instruction_adherence",
        "label": label,
        "requirement": f"The response must match the correct final answer exactly as {reference_display}.",
        "severity_tier": "hard_gate",
        "count": 0,
        "fallback_exact_answer_correctness": True,
    }


def canonical_rows_to_rubrics(
    example_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    example: JudgeBenchJoinedExample,
    protocol_mode: str = _DEFAULT_PROTOCOL_MODE,
) -> List[RubricCriterion]:
    rubrics: List[RubricCriterion] = []
    seen: set[str] = set()
    uses_judgebench_tuning = _protocol_uses_judgebench_tuning(protocol_mode)
    reference_visible = _example_reference_visible(example)
    collapsed_rows = _collapse_semantic_rows(rows, example=example) if uses_judgebench_tuning else list(rows)
    if uses_judgebench_tuning and reference_visible and _example_is_exact_answer_task(example):
        has_correctness_row = any(
            _exact_answer_rubric_kind(_proposal_to_rubric_text(row), reference_answer=example.reference_answer) == "correctness"
            for row in collapsed_rows
        )
        if not has_correctness_row:
            fallback_row = _fallback_exact_answer_correctness_row(example)
            if fallback_row is not None:
                collapsed_rows = list(collapsed_rows) + [fallback_row]
    for index, row in enumerate(collapsed_rows):
        text = _proposal_to_rubric_text(row)
        normalized = text.lower()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        blind_exact_signal_kind = str(row.get("blind_exact_signal_kind", "")).strip().lower()
        if blind_exact_signal_kind in {"consistency", "choice_value_consistency"}:
            exact_answer_kind = "correctness"
        elif blind_exact_signal_kind == "format":
            exact_answer_kind = "format"
        else:
            exact_answer_kind = (
                _exact_answer_rubric_kind(text, reference_answer=example.reference_answer)
                if uses_judgebench_tuning and reference_visible and _example_is_exact_answer_task(example)
                else None
            )
        code_style = _code_style_cluster(text) if uses_judgebench_tuning and example.source_family == "livecodebench" else None
        code_proxy = _code_proxy_cluster(text) if uses_judgebench_tuning and example.source_family == "livecodebench" else None
        rubrics.append(
            RubricCriterion(
                rubric_id=f"{example_id}__compiled_recursive_{index}",
                text=text,
                source_stage="compiled_recursive",
                depth=min((row.get("recursion_depths") or [0])),
                round_index=index,
                metadata={
                    "canonical_row": to_json_dict(dict(row)),
                    "semantic_cluster": _semantic_cluster_key(row, example) if uses_judgebench_tuning else "",
                    "exact_answer_kind": exact_answer_kind or "",
                    "blind_exact_signal_kind": blind_exact_signal_kind,
                    "code_style_cluster": code_style or "",
                    "code_proxy_cluster": code_proxy or "",
                },
            )
        )
    return rubrics


def _score_pair_with_weights(
    *,
    example: JudgeBenchJoinedExample,
    rubrics: Sequence[RubricCriterion],
    evaluations: Sequence[RubricEvaluation],
    candidate_a: CandidateNote,
    candidate_b: CandidateNote,
    weights: Mapping[str, float],
    protocol_mode: str = _DEFAULT_PROTOCOL_MODE,
) -> Dict[str, Any]:
    by_pair = {(row.candidate_id, row.rubric_id): bool(row.satisfied) for row in evaluations}
    uses_judgebench_tuning = _protocol_uses_judgebench_tuning(protocol_mode)
    score_a = 0.0
    score_b = 0.0
    non_exact_score_a = 0.0
    non_exact_score_b = 0.0
    satisfied_a = 0
    satisfied_b = 0
    exact_correctness_a = 0
    exact_correctness_b = 0
    exact_correctness_weight_a = 0.0
    exact_correctness_weight_b = 0.0
    severity_counts_a = {tier: 0 for tier in _SEVERITY_RANK}
    severity_counts_b = {tier: 0 for tier in _SEVERITY_RANK}
    pair_rows: List[Dict[str, Any]] = []
    exact_answer_task = _example_is_exact_answer_task(example)
    reference_value = _example_reference_value(example) if exact_answer_task else ""
    extraction_a = (
        _extract_exact_answer_candidate_for_example(example, candidate_a.text)
        if exact_answer_task
        else None
    )
    extraction_b = (
        _extract_exact_answer_candidate_for_example(example, candidate_b.text)
        if exact_answer_task
        else None
    )
    candidate_value_a = extraction_a.value if extraction_a is not None else None
    candidate_value_b = extraction_b.value if extraction_b is not None else None
    explicit_answer_a = bool(extraction_a is not None and extraction_a.explicit)
    explicit_answer_b = bool(extraction_b is not None and extraction_b.explicit)
    extracted_match_a = bool(candidate_value_a) and candidate_value_a == reference_value
    extracted_match_b = bool(candidate_value_b) and candidate_value_b == reference_value
    blind_signal_a = _blind_exact_answer_signal(example, candidate_a.text) if exact_answer_task else BlindExactAnswerSignal()
    blind_signal_b = _blind_exact_answer_signal(example, candidate_b.text) if exact_answer_task else BlindExactAnswerSignal()
    for rubric in rubrics:
        weight = float(weights.get(rubric.rubric_id, 0.0))
        a_value = bool(by_pair.get((candidate_a.candidate_id, rubric.rubric_id), False))
        b_value = bool(by_pair.get((candidate_b.candidate_id, rubric.rubric_id), False))
        canonical_row = rubric.metadata.get("canonical_row", {}) or {}
        severity_tier = str((rubric.metadata.get("canonical_row", {}) or {}).get("severity_tier", "")).strip().lower()
        if a_value:
            score_a += weight
            satisfied_a += 1
            if severity_tier in severity_counts_a:
                severity_counts_a[severity_tier] += 1
        if b_value:
            score_b += weight
            satisfied_b += 1
            if severity_tier in severity_counts_b:
                severity_counts_b[severity_tier] += 1
        exact_answer_kind = str(rubric.metadata.get("exact_answer_kind", "")).strip().lower()
        dimension = str(canonical_row.get("dimension", "")).strip().lower()
        if exact_answer_kind not in {"correctness", "format"}:
            if a_value:
                non_exact_score_a += weight
            if b_value:
                non_exact_score_b += weight
        if exact_answer_kind == "correctness":
            if a_value:
                exact_correctness_a += 1
                exact_correctness_weight_a += weight
            if b_value:
                exact_correctness_b += 1
                exact_correctness_weight_b += weight
        pair_rows.append(
            {
                "rubric_id": rubric.rubric_id,
                "text": rubric.text,
                "weight": weight,
                "response_A_satisfied": a_value,
                "response_B_satisfied": b_value,
                "severity_tier": severity_tier,
                "dimension": dimension,
                "exact_answer_kind": exact_answer_kind,
            }
        )
    if exact_answer_task:
        key_a = (exact_correctness_a, exact_correctness_weight_a, score_a, satisfied_a)
        key_b = (exact_correctness_b, exact_correctness_weight_b, score_b, satisfied_b)
    else:
        key_a = (score_a, satisfied_a)
        key_b = (score_b, satisfied_b)
    if key_a > key_b:
        decision = "A>B"
    elif key_b > key_a:
        decision = "B>A"
    else:
        decision = "A=B"
    tie_break_reason = ""
    explicit_match_a = extracted_match_a and explicit_answer_a
    explicit_match_b = extracted_match_b and explicit_answer_b
    if uses_judgebench_tuning and decision == "A=B" and exact_answer_task:
        if explicit_match_a != explicit_match_b:
            decision = "A>B" if explicit_match_a else "B>A"
            tie_break_reason = "exact_answer_reference_match"
        elif explicit_answer_a != explicit_answer_b:
            decision = "A>B" if explicit_answer_a else "B>A"
            tie_break_reason = "exact_answer_presence"
    if uses_judgebench_tuning and decision == "A=B" and _strict_pairwise_task(example):
        severity_key_a = tuple(severity_counts_a[tier] for tier in ("hard_gate", "high", "medium", "low"))
        severity_key_b = tuple(severity_counts_b[tier] for tier in ("hard_gate", "high", "medium", "low"))
        if severity_key_a > severity_key_b:
            decision = "A>B"
            tie_break_reason = "strict_pair_severity_counts"
        elif severity_key_b > severity_key_a:
            decision = "B>A"
            tie_break_reason = "strict_pair_severity_counts"
    if not uses_judgebench_tuning and decision == "A=B" and exact_answer_task:
        signal_key_a = blind_signal_a.tie_break_key()
        signal_key_b = blind_signal_b.tie_break_key()
        if signal_key_a > signal_key_b:
            decision = "A>B"
            tie_break_reason = "blind_exact_answer_signal"
        elif signal_key_b > signal_key_a:
            decision = "B>A"
            tie_break_reason = "blind_exact_answer_signal"
    return {
        "decision": decision,
        "decision_reversed": flip_decision(decision),
        "score_A": score_a,
        "score_B": score_b,
        "non_exact_score_A": non_exact_score_a,
        "non_exact_score_B": non_exact_score_b,
        "satisfied_A": satisfied_a,
        "satisfied_B": satisfied_b,
        "exact_answer_correctness_A": exact_correctness_a,
        "exact_answer_correctness_B": exact_correctness_b,
        "exact_answer_value_A": candidate_value_a or "",
        "exact_answer_value_B": candidate_value_b or "",
        "exact_answer_match_A": extracted_match_a,
        "exact_answer_match_B": extracted_match_b,
        "exact_answer_explicit_A": explicit_answer_a,
        "exact_answer_explicit_B": explicit_answer_b,
        "blind_exact_signal_A": {
            "value": blind_signal_a.value,
            "explicit": blind_signal_a.explicit,
            "format_ok": blind_signal_a.format_ok,
            "consistent": blind_signal_a.consistent,
            "conflicting_markers": blind_signal_a.conflicting_markers,
            "marker_count": blind_signal_a.marker_count,
        },
        "blind_exact_signal_B": {
            "value": blind_signal_b.value,
            "explicit": blind_signal_b.explicit,
            "format_ok": blind_signal_b.format_ok,
            "consistent": blind_signal_b.consistent,
            "conflicting_markers": blind_signal_b.conflicting_markers,
            "marker_count": blind_signal_b.marker_count,
        },
        "tie_break_reason": tie_break_reason,
        "strict_pairwise_task": _strict_pairwise_task(example),
        "pair_evaluations": pair_rows,
    }


def score_discovered_rubrics_for_pair(
    *,
    example: JudgeBenchJoinedExample,
    rubrics: Sequence[RubricCriterion],
    scoring_candidates: Sequence[CandidateNote],
    pair_candidates: Sequence[CandidateNote],
    evaluations: Sequence[RubricEvaluation],
    covariance_ridge: float,
    protocol_mode: str = _DEFAULT_PROTOCOL_MODE,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if len(pair_candidates) != 2:
        raise ValueError("Expected exactly two pair candidates (A and B).")
    blind_wu_profile = _policy_blind_wu_profile(policy or {})
    uniform_weights = compute_uniform_weights(rubrics)
    wu_weights, wu_debug = compute_whitened_uniform_weights(
        list(rubrics),
        list(scoring_candidates),
        evaluations,
        ridge=covariance_ridge,
    )
    if _protocol_uses_judgebench_tuning(protocol_mode):
        uniform_weights = _apply_weight_adjustments(example=example, rubrics=rubrics, weights=uniform_weights)
        wu_weights = _apply_weight_adjustments(example=example, rubrics=rubrics, weights=wu_weights)
    uniform_result = _score_pair_with_weights(
        example=example,
        rubrics=rubrics,
        evaluations=evaluations,
        candidate_a=pair_candidates[0],
        candidate_b=pair_candidates[1],
        weights=uniform_weights,
        protocol_mode=protocol_mode,
    )
    wu_result = _score_pair_with_weights(
        example=example,
        rubrics=rubrics,
        evaluations=evaluations,
        candidate_a=pair_candidates[0],
        candidate_b=pair_candidates[1],
        weights=wu_weights,
        protocol_mode=protocol_mode,
    )
    uniform_result["decision_policy"] = "uniform"
    if _protocol_uses_judgebench_tuning(protocol_mode):
        wu_result = _stabilize_whitened_uniform_result(
            example=example,
            uniform_result=uniform_result,
            wu_result=wu_result,
            wu_debug=wu_debug,
            covariance_ridge=covariance_ridge,
        )
    else:
        wu_result = _stabilize_blind_whitened_uniform_result(
            example=example,
            uniform_result=uniform_result,
            wu_result=wu_result,
            wu_debug=wu_debug,
            covariance_ridge=covariance_ridge,
            blind_wu_profile=blind_wu_profile,
        )
    return {
        "uniform": {"weights": uniform_weights, "result": uniform_result},
        "whitened_uniform": {"weights": wu_weights, "debug": wu_debug, "result": wu_result},
    }


def flip_decision(decision: Optional[str]) -> Optional[str]:
    if decision == "A>B":
        return "B>A"
    if decision == "B>A":
        return "A>B"
    return decision


def compute_double_order_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        decision_1 = row.get("decision_original")
        decision_2 = flip_decision(row.get("decision_reversed"))
        counter = 0
        for decision in (decision_1, decision_2):
            if decision == row.get("label"):
                counter += 1
            elif decision == flip_decision(row.get("label")):
                counter -= 1
        if counter > 0:
            correct += 1
    return 100.0 * correct / len(rows)


def metric_breakdown(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name, prefix in _BUILT_IN_SUBSET_PREFIXES.items():
        subset = [row for row in rows if str(row.get("source", "")).startswith(prefix)]
        results[name] = compute_double_order_accuracy(subset)
    results["overall"] = compute_double_order_accuracy(rows)
    return results


def _is_choice_only_reference(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"[A-Ja-j]{1,5}", stripped))


def _strict_pairwise_task(example: JudgeBenchJoinedExample) -> bool:
    return example.label in {"A>B", "B>A"}


def _is_exact_answer_task(question: str, reference_answer: str) -> bool:
    normalized = question.lower()
    if any(
        cue in normalized
        for cue in (
            "duplicate that letter five times",
            "single string",
            "single word",
            "single digit",
            "***x***",
            "put your answer in **bold**",
            "return your answer as",
        )
    ):
        return True
    return _is_choice_only_reference(reference_answer)


def _looks_like_code_task(example: JudgeBenchJoinedExample) -> bool:
    text = f"{example.source}\n{example.question}\n{example.reference_answer}".lower()
    return any(
        cue in text
        for cue in (
            "class solution",
            "def ",
            "input",
            "output",
            "constraints",
            "# your code here",
            "livecodebench",
        )
    )


def _family_prompt_nudges_for_source_family(source_family: str) -> List[str]:
    nudges: List[str] = []
    if source_family == "mmlu-pro":
        nudges.append(_MCQ_PROMPT_NUDGE)
    if source_family == "livebench-reasoning":
        nudges.append(_REASONING_PROMPT_NUDGE)
    if source_family == "livebench-math":
        nudges.append(_MATH_PROMPT_NUDGE)
    if source_family == "livecodebench":
        nudges.append(_CODE_PROMPT_NUDGE)
    return nudges


def _source_family_nudges(example: JudgeBenchJoinedExample) -> List[str]:
    nudges = _family_prompt_nudges_for_source_family(example.source_family)
    if _looks_like_code_task(example) and _CODE_PROMPT_NUDGE not in nudges:
        nudges.append(_CODE_PROMPT_NUDGE)
    if _example_is_exact_answer_task(example):
        nudges.append(_FORMAT_PROMPT_NUDGE)
    return nudges


def _blind_guidance_global_nudges(profile: str) -> List[str]:
    normalized = _normalize_blind_guidance_profile(profile)
    if normalized == _BLIND_GUIDANCE_PROFILE_OFF:
        return []
    nudges = [_GRANULARITY_PROMPT_NUDGE]
    if normalized == _BLIND_GUIDANCE_PROFILE_FAMILY_V2:
        nudges.append(_FINAL_ANSWER_SUPPORT_PROMPT_NUDGE)
    return nudges


def _count_broad_canonical_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        dimension = _normalize_text(str(row.get("dimension", ""))).lower()
        text = _proposal_to_rubric_text(row).lower()
        if dimension in _BROAD_DIMENSIONS or any(token in text for token in ("complete", "coverage", "overall", "correctness")):
            count += 1
    return count


def _severity_rank_for_row(row: Mapping[str, Any]) -> int:
    return _SEVERITY_RANK.get(str(row.get("severity_tier", "")).strip().lower(), 0)


def _normalize_exact_answer_value(text: str) -> str:
    out = (text or "").strip()
    if out.startswith("```") and out.endswith("```"):
        out = out.strip("`").strip()
    if out.startswith("**") and out.endswith("**") and len(out) > 4:
        out = out[2:-2]
    out = out.strip("`*_ \t\r\n.,:;!?\"'")
    return _normalize_text(out).lower()


def _is_repeated_choice_value(text: str) -> bool:
    return bool(re.fullmatch(r"([a-j])\1{0,9}", _normalize_exact_answer_value(text)))


def _question_requested_answer_mode(question: str, reference_answer: str) -> str:
    normalized = (question or "").lower()
    reference_value = _normalize_exact_answer_value(reference_answer)
    if _is_repeated_choice_value(reference_value) or "duplicate that letter five times" in normalized:
        return "repeated_choice"
    if "***" in normalized or "bold" in normalized or "double asterisks" in normalized:
        return "decorated_span"
    if "single digit" in normalized:
        return "single_digit"
    if "single phrase" in normalized:
        return "single_phrase"
    if "single word" in normalized or "single string" in normalized:
        return "single_word"
    return ""


def _last_nonempty_line(text: str) -> str:
    nonempty = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return nonempty[-1] if nonempty else ""


def _make_exact_answer_extraction(
    value: str,
    *,
    source: str,
    explicit: bool,
) -> Optional[ExactAnswerExtraction]:
    normalized = _normalize_exact_answer_value(value)
    if not normalized:
        return None
    return ExactAnswerExtraction(value=normalized, source=source, explicit=explicit)


def _extract_last_match_value(
    text: str,
    pattern: re.Pattern[str],
    *,
    source: str,
    explicit: bool,
) -> Optional[ExactAnswerExtraction]:
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return _make_exact_answer_extraction(matches[-1].group(0), source=source, explicit=explicit)


def _extract_exact_answer_candidate(
    candidate_text: str,
    *,
    reference_answer: str,
    question: str,
) -> Optional[ExactAnswerExtraction]:
    reference_value = _normalize_exact_answer_value(reference_answer)
    answer_mode = _question_requested_answer_mode(question, reference_answer)

    if _is_repeated_choice_value(reference_value):
        repeated_re = re.compile(rf"\b([A-Ja-j])\1{{{len(reference_value) - 1}}}\b")
        stripped = _normalize_exact_answer_value(candidate_text)
        if stripped and re.fullmatch(rf"([a-j])\1{{{len(reference_value) - 1}}}", stripped):
            return _make_exact_answer_extraction(stripped, source="entire_response", explicit=True)

        triple_matches = list(_TRIPLE_ASTERISK_SPAN_RE.finditer(candidate_text or ""))
        if triple_matches:
            inner = _normalize_exact_answer_value(triple_matches[-1].group(1))
            if re.fullmatch(rf"([a-j])\1{{{len(reference_value) - 1}}}", inner):
                return _make_exact_answer_extraction(inner, source="triple_asterisk_span", explicit=True)

        bold_matches = list(_BOLD_SPAN_RE.finditer(candidate_text or ""))
        if bold_matches:
            inner = _normalize_exact_answer_value(bold_matches[-1].group(1))
            if re.fullmatch(rf"([a-j])\1{{{len(reference_value) - 1}}}", inner):
                return _make_exact_answer_extraction(inner, source="bold_span", explicit=True)

        answer_lines = list(_FINAL_ANSWER_LINE_RE.finditer(candidate_text or ""))
        if answer_lines:
            line = answer_lines[-1].group(1)
            matches = list(repeated_re.finditer(line))
            if matches:
                return _make_exact_answer_extraction(matches[-1].group(0), source="final_answer_line", explicit=True)

        last_line = _last_nonempty_line(candidate_text)
        if last_line:
            normalized_last_line = _normalize_exact_answer_value(last_line)
            if re.fullmatch(rf"([a-j])\1{{{len(reference_value) - 1}}}", normalized_last_line):
                return _make_exact_answer_extraction(normalized_last_line, source="last_line", explicit=True)

        matches = list(repeated_re.finditer(candidate_text or ""))
        if matches:
            return _make_exact_answer_extraction(matches[-1].group(0), source="inline_repeated_choice", explicit=False)
        return None

    if reference_value:
        triple_matches = list(_TRIPLE_ASTERISK_SPAN_RE.finditer(candidate_text or ""))
        if triple_matches:
            return _make_exact_answer_extraction(triple_matches[-1].group(1), source="triple_asterisk_span", explicit=True)

        bold_matches = list(_BOLD_SPAN_RE.finditer(candidate_text or ""))
        if bold_matches:
            return _make_exact_answer_extraction(bold_matches[-1].group(1), source="bold_span", explicit=True)

        answer_lines = list(_FINAL_ANSWER_LINE_RE.finditer(candidate_text or ""))
        if answer_lines:
            return _make_exact_answer_extraction(answer_lines[-1].group(1), source="final_answer_line", explicit=True)

        if "single phrase" in question.lower():
            last_line = _last_nonempty_line(candidate_text)
            if last_line:
                return _make_exact_answer_extraction(last_line, source="last_line", explicit=True)
        return None

    if answer_mode == "repeated_choice":
        repeated_re = re.compile(r"\b([A-Ja-j])\1{1,9}\b")
        simple_choice_re = re.compile(r"\b[A-Ja-j]\b")
        stripped = _normalize_exact_answer_value(candidate_text)
        if stripped and re.fullmatch(r"([a-j])\1{1,9}", stripped):
            return _make_exact_answer_extraction(stripped, source="entire_response", explicit=True)
        triple_matches = list(_TRIPLE_ASTERISK_SPAN_RE.finditer(candidate_text or ""))
        if triple_matches:
            inner = triple_matches[-1].group(1)
            repeated = _extract_last_match_value(inner, repeated_re, source="triple_asterisk_span", explicit=True)
            if repeated is not None:
                return repeated
            explicit_choice = _extract_last_match_value(inner, simple_choice_re, source="triple_asterisk_span", explicit=True)
            if explicit_choice is not None:
                return explicit_choice
        bold_matches = list(_BOLD_SPAN_RE.finditer(candidate_text or ""))
        if bold_matches:
            inner = bold_matches[-1].group(1)
            repeated = _extract_last_match_value(inner, repeated_re, source="bold_span", explicit=True)
            if repeated is not None:
                return repeated
            explicit_choice = _extract_last_match_value(inner, simple_choice_re, source="bold_span", explicit=True)
            if explicit_choice is not None:
                return explicit_choice
        answer_lines = list(_FINAL_ANSWER_LINE_RE.finditer(candidate_text or ""))
        if answer_lines:
            line = answer_lines[-1].group(1)
            repeated = _extract_last_match_value(line, repeated_re, source="final_answer_line", explicit=True)
            if repeated is not None:
                return repeated
            explicit_choice = _extract_last_match_value(line, simple_choice_re, source="final_answer_line", explicit=True)
            if explicit_choice is not None:
                return explicit_choice
        last_line = _last_nonempty_line(candidate_text)
        if last_line:
            repeated = _extract_last_match_value(last_line, repeated_re, source="last_line", explicit=True)
            if repeated is not None:
                return repeated
            explicit_choice = _extract_last_match_value(last_line, simple_choice_re, source="last_line", explicit=True)
            if explicit_choice is not None:
                return explicit_choice
        inline_repeated = _extract_last_match_value(
            candidate_text,
            repeated_re,
            source="inline_repeated_choice",
            explicit=False,
        )
        if inline_repeated is not None:
            return inline_repeated
        return None

    triple_matches = list(_TRIPLE_ASTERISK_SPAN_RE.finditer(candidate_text or ""))
    if triple_matches:
        return _make_exact_answer_extraction(triple_matches[-1].group(1), source="triple_asterisk_span", explicit=True)

    bold_matches = list(_BOLD_SPAN_RE.finditer(candidate_text or ""))
    if bold_matches:
        return _make_exact_answer_extraction(bold_matches[-1].group(1), source="bold_span", explicit=True)

    answer_lines = list(_FINAL_ANSWER_LINE_RE.finditer(candidate_text or ""))
    if answer_lines:
        return _make_exact_answer_extraction(answer_lines[-1].group(1), source="final_answer_line", explicit=True)

    if answer_mode == "single_digit":
        integer_re = re.compile(r"-?\d+")
        last_line = _last_nonempty_line(candidate_text)
        if last_line:
            explicit_integer = _extract_last_match_value(last_line, integer_re, source="last_line", explicit=True)
            if explicit_integer is not None:
                return explicit_integer
        inline_integer = _extract_last_match_value(candidate_text, integer_re, source="inline_integer", explicit=False)
        if inline_integer is not None:
            return inline_integer
        return None

    if "single phrase" in question.lower():
        last_line = _last_nonempty_line(candidate_text)
        if last_line:
            return _make_exact_answer_extraction(last_line, source="last_line", explicit=True)
    if answer_mode in {"single_word", "single_phrase", "decorated_span"}:
        last_line = _last_nonempty_line(candidate_text)
        if last_line:
            return _make_exact_answer_extraction(last_line, source="last_line", explicit=True)
    return None


def _normalize_question_choice_text(question: str) -> str:
    cleaned = question or ""
    for old, new in (
        ("\\textbf{", " "),
        ("\\mathbf{", " "),
        ("\\boxed{", " "),
        ("\\qquad", " "),
        ("\\quad", " "),
        ("~", " "),
    ):
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    return _normalize_text(cleaned)


def _choice_letter_from_answer_value(value: str) -> str:
    normalized = _normalize_exact_answer_value(value)
    if not normalized:
        return ""
    if _is_repeated_choice_value(normalized):
        return normalized[0]
    if re.fullmatch(r"[a-j]", normalized):
        return normalized
    return ""


def _required_repeated_choice_length(question: str) -> int:
    normalized = (question or "").lower()
    if "five times" in normalized:
        return 5
    return 2


def _explicit_answer_segments(candidate_text: str) -> List[str]:
    segments: List[str] = []
    for match in _FINAL_ANSWER_LINE_RE.finditer(candidate_text or ""):
        segments.append(match.group(1))
    for match in _TRIPLE_ASTERISK_SPAN_RE.finditer(candidate_text or ""):
        segments.append(match.group(1))
    for match in _BOLD_SPAN_RE.finditer(candidate_text or ""):
        segments.append(match.group(1))
    last_line = _last_nonempty_line(candidate_text)
    if last_line:
        segments.append(last_line)
    ordered: List[str] = []
    seen: set[str] = set()
    for segment in segments:
        normalized = _normalize_text(segment)
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(normalized)
    return ordered


def _segment_choice_letters(segment: str, choice_map: Mapping[str, str]) -> List[str]:
    lowered = _normalize_text(segment).lower()
    if not lowered:
        return []
    letters: set[str] = set()
    repeated_matches = re.findall(r"\b([a-j])\1{1,9}\b", lowered)
    letters.update(match.lower() for match in repeated_matches)
    cue_matches = re.findall(
        r"(?:letter|option|answer|string|choice)\s*[\(\[\{\"']?([a-j])[\)\]\}\"']?",
        lowered,
    )
    letters.update(match.lower() for match in cue_matches)
    if re.fullmatch(r"[a-j]", lowered):
        letters.add(lowered)
    for letter, option_value in choice_map.items():
        if option_value and len(option_value) > 1 and option_value in lowered:
            letters.add(letter)
    return sorted(letters)


def _blind_exact_answer_signal(example: JudgeBenchJoinedExample, candidate_text: str) -> BlindExactAnswerSignal:
    extraction = _extract_exact_answer_candidate_for_example(example, candidate_text)
    extracted_value = extraction.value if extraction is not None else ""
    explicit = bool(extraction is not None and extraction.explicit)
    answer_mode = _example_requested_answer_mode(example)
    choice_map = _example_choice_value_map(example)
    explicit_segments = _explicit_answer_segments(candidate_text)
    explicit_values = {
        item.value
        for item in (
            _extract_exact_answer_candidate_for_example(example, segment)
            for segment in explicit_segments
        )
        if item is not None and item.value
    }
    segment_letters: set[str] = set()
    for segment in explicit_segments:
        segment_letters.update(_segment_choice_letters(segment, choice_map))
    response_value_letters: set[str] = set()
    normalized_response = _normalize_text(candidate_text).lower()
    for letter, option_value in choice_map.items():
        if option_value and len(option_value) > 1 and option_value in normalized_response:
            response_value_letters.add(letter)
    if len(response_value_letters) == 1:
        segment_letters.update(response_value_letters)
    final_letter = _choice_letter_from_answer_value(extracted_value)
    conflicting_markers = len(explicit_values) > 1
    if choice_map and segment_letters:
        if len(segment_letters) > 1:
            conflicting_markers = True
        elif final_letter and final_letter not in segment_letters:
            conflicting_markers = True
    if answer_mode == "repeated_choice":
        required_length = max(1, _answer_key_features(example).repeated_choice_length or _required_repeated_choice_length(example.question))
        format_ok = (
            explicit
            and bool(extracted_value)
            and len(set(extracted_value)) == 1
            and len(extracted_value) >= required_length
        )
    else:
        format_ok = explicit and bool(extracted_value)
    consistent = explicit and not conflicting_markers
    marker_count = len(explicit_values | ({final_letter} if final_letter else set()) | segment_letters)
    return BlindExactAnswerSignal(
        value=extracted_value,
        explicit=explicit,
        format_ok=format_ok,
        consistent=consistent,
        option_map_available=bool(choice_map),
        conflicting_markers=conflicting_markers,
        marker_count=marker_count,
    )


def _blind_exact_signal_row(example: JudgeBenchJoinedExample, *, kind: str) -> Dict[str, Any]:
    if kind == "format":
        if _example_requested_answer_mode(example) == "repeated_choice":
            requirement = (
                "The response must end with an explicit repeated-letter final answer in the requested format instead of "
                "only implying a choice in the reasoning."
            )
        else:
            requirement = (
                "The response must make its final answer explicit in the requested output format instead of leaving the "
                "choice implicit in the reasoning."
            )
        return {
            "dimension": "final_answer_correctness",
            "label": "Final answer is explicit in the requested format",
            "requirement": requirement,
            "severity_tier": "high",
            "count": 0,
            "blind_exact_signal_row": True,
            "blind_exact_signal_kind": kind,
        }
    if kind == "choice_value_consistency":
        return {
            "dimension": "final_answer_correctness",
            "label": "Final answer letter and selected option/value agree",
            "requirement": (
                "When the response names both an option letter and an option value or option text, those explicit "
                "answer markers must point to the same final choice."
            ),
            "severity_tier": "hard_gate",
            "count": 0,
            "blind_exact_signal_row": True,
            "blind_exact_signal_kind": kind,
        }
    return {
        "dimension": "final_answer_correctness",
        "label": "Final answer markers stay internally consistent",
        "requirement": (
            "The response must use one explicit final answer and keep its answer markers consistent instead of "
            "contradicting itself across the conclusion, last line, or final-answer cue."
        ),
        "severity_tier": "hard_gate",
        "count": 0,
        "blind_exact_signal_row": True,
        "blind_exact_signal_kind": kind,
    }


def _blind_exact_signal_rows(example: JudgeBenchJoinedExample) -> List[Dict[str, Any]]:
    if not _example_is_exact_answer_task(example):
        return []
    rows = [
        _blind_exact_signal_row(example, kind="format"),
        _blind_exact_signal_row(example, kind="consistency"),
    ]
    if _example_choice_value_map(example):
        rows.append(_blind_exact_signal_row(example, kind="choice_value_consistency"))
    return rows


def _differentiating_blind_exact_signal_kinds(
    example: JudgeBenchJoinedExample,
    pair_candidates: Sequence[CandidateNote],
) -> set[str]:
    if len(pair_candidates) != 2 or not _example_is_exact_answer_task(example):
        return set()
    signal_a = _blind_exact_answer_signal(example, pair_candidates[0].text)
    signal_b = _blind_exact_answer_signal(example, pair_candidates[1].text)
    kinds: set[str] = set()
    if signal_a.format_ok != signal_b.format_ok:
        kinds.add("format")
    if (
        signal_a.consistent != signal_b.consistent
        or signal_a.conflicting_markers != signal_b.conflicting_markers
        or signal_a.marker_count != signal_b.marker_count
    ):
        kinds.add("consistency")
    if signal_a.option_map_available and signal_b.option_map_available and (
        signal_a.consistent != signal_b.consistent
        or signal_a.conflicting_markers != signal_b.conflicting_markers
    ):
        kinds.add("choice_value_consistency")
    return kinds


def _rubric_mentions_reference_answer(text: str, *, reference_answer: str) -> bool:
    lowered = _normalize_text(text).lower()
    reference_value = _normalize_exact_answer_value(reference_answer)
    if not lowered or not reference_value:
        return False
    if len(reference_value) > 1 and reference_value in lowered:
        return True
    if _is_repeated_choice_value(reference_value):
        letter = reference_value[0]
        return (
            re.search(rf"(?:letter|option|answer)\s*[\(\"']?{re.escape(letter)}[\)\"']?", lowered) is not None
            or re.search(rf"(?:select|choose|matches?)\s+[\(\"']?{re.escape(letter)}[\)\"']?", lowered) is not None
        )
    if reference_value.isdigit():
        return re.search(rf"(?:digit|value|position|answer)\D{{0,16}}{re.escape(reference_value)}\b", lowered) is not None
    return False


def _exact_answer_rubric_kind(text: str, *, reference_answer: str = "") -> Optional[str]:
    lowered = _normalize_text(text).lower()
    format_match = any(hint in lowered for hint in _EXACT_FORMAT_HINTS)
    correctness_match = any(hint in lowered for hint in _EXACT_CORRECTNESS_HINTS)
    explicit_correctness = (
        "correctness" in lowered and "format correctness" not in lowered and "answer format" not in lowered
    )
    reference_match = _rubric_mentions_reference_answer(lowered, reference_answer=reference_answer)
    if correctness_match or explicit_correctness or reference_match:
        return "correctness"
    if format_match:
        return "format"
    return None


def _code_style_cluster(text: str) -> Optional[str]:
    lowered = _normalize_text(text).lower()
    if "variable naming" in lowered or "readability" in lowered or "clear function structure" in lowered:
        return "code_style_readability"
    if "imports lru_cache" in lowered or "imports functools" in lowered:
        return "code_style_import"
    if any(hint in lowered for hint in ("lru_cache", "memoization", "memoize", "caching decorator")):
        return "code_style_memoization"
    return None


def _code_proxy_cluster(text: str) -> Optional[str]:
    lowered = _normalize_text(text).lower()
    if not lowered:
        return None
    if "instead of" in lowered or "avoid redundant" in lowered or "redundant" in lowered:
        return "code_impl_preference"
    if "single set comparison" in lowered or "set equality" in lowered or "use a single" in lowered:
        return "code_impl_preference"
    if "xor" in lowered or "sum parity" in lowered:
        return "code_operator_preference"
    if "prunes search" in lowered or "prune search" in lowered:
        return "code_optimization_preference"
    return None


def _semantic_cluster_key(row: Mapping[str, Any], example: JudgeBenchJoinedExample) -> Optional[str]:
    text = _proposal_to_rubric_text(row)
    exact_kind = _exact_answer_rubric_kind(text, reference_answer=example.reference_answer)
    if _example_is_exact_answer_task(example) and exact_kind is not None:
        return f"exact_answer_{exact_kind}"
    if example.source_family == "livecodebench":
        code_style = _code_style_cluster(text)
        if code_style:
            return code_style
        code_proxy = _code_proxy_cluster(text)
        if code_proxy:
            return code_proxy
    return None


def _canonical_row_priority(row: Mapping[str, Any]) -> Tuple[int, int, int]:
    return (
        _severity_rank_for_row(row),
        int(row.get("count", 0) or 0),
        len(_proposal_to_rubric_text(row)),
    )


def _collapse_semantic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    example: JudgeBenchJoinedExample,
) -> List[Mapping[str, Any]]:
    chosen_by_cluster: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cluster = _semantic_cluster_key(row, example)
        if cluster is None:
            continue
        current = chosen_by_cluster.get(cluster)
        if current is None or _canonical_row_priority(row) > _canonical_row_priority(current):
            chosen_by_cluster[cluster] = row

    emitted_clusters: set[str] = set()
    collapsed: List[Mapping[str, Any]] = []
    for row in rows:
        cluster = _semantic_cluster_key(row, example)
        if cluster is None:
            collapsed.append(row)
            continue
        if cluster in emitted_clusters:
            continue
        if row is chosen_by_cluster.get(cluster):
            collapsed.append(row)
            emitted_clusters.add(cluster)
    return collapsed


def _row_is_broad(row: Mapping[str, Any]) -> bool:
    dimension = _normalize_text(str(row.get("dimension", ""))).lower()
    text = _proposal_to_rubric_text(row).lower()
    return dimension in _BROAD_DIMENSIONS or any(token in text for token in ("complete", "coverage", "overall"))


def _blind_pruning_cluster_key(row: Mapping[str, Any], example: JudgeBenchJoinedExample) -> Optional[str]:
    signal_kind = str(row.get("blind_exact_signal_kind", "")).strip().lower()
    if signal_kind:
        return f"blind_exact_signal_{signal_kind}"
    if bool(row.get("blind_generalized_answer_row", False)):
        return "blind_generalized_answer"
    semantic = _semantic_cluster_key(row, example)
    if semantic:
        return semantic
    if _row_is_broad(row):
        dimension = _normalize_text(str(row.get("dimension", ""))).lower() or "overall"
        return f"broad_{dimension}"
    return None


def _blind_row_priority(
    row: Mapping[str, Any],
    *,
    example: JudgeBenchJoinedExample,
    blind_scoring_profile: str,
) -> Tuple[int, int, int, int, int, int]:
    targeted_bonus = 0
    non_format_bonus = 0
    if blind_scoring_profile == _BLIND_SCORING_PROFILE_PRUNED_V2:
        dimension = _normalize_text(str(row.get("dimension", ""))).lower()
        signal_kind = str(row.get("blind_exact_signal_kind", "")).strip().lower()
        exact_kind = _exact_answer_rubric_kind(
            _proposal_to_rubric_text(row),
            reference_answer=example.reference_answer,
        )
        if signal_kind in {"consistency", "choice_value_consistency"}:
            exact_kind = "correctness"
        elif signal_kind == "format":
            exact_kind = "format"
        if bool(row.get("blind_generalized_answer_row", False)):
            targeted_bonus += 3
        if bool(row.get("blind_exact_signal_row", False)):
            targeted_bonus += 4
        elif exact_kind == "correctness":
            targeted_bonus += 2
        if dimension in {
            "constraint_satisfaction",
            "reasoning_support",
            "conclusion_correctness",
            "final_answer_correctness",
        }:
            targeted_bonus += 1
        non_format_bonus = int(exact_kind != "format")
    return (
        int(bool(row.get("blind_exact_signal_row", False) or row.get("blind_generalized_answer_row", False))),
        _severity_rank_for_row(row),
        targeted_bonus,
        int(not _row_is_broad(row)),
        non_format_bonus,
        int(row.get("count", 0) or 0),
    )


def _rubric_mentions_candidate_value(text: str, value: str) -> bool:
    lowered = _normalize_text(text).lower()
    normalized_value = _normalize_exact_answer_value(value)
    if not lowered or not normalized_value:
        return False
    if len(normalized_value) > 1 and normalized_value in lowered:
        return True
    if _is_repeated_choice_value(normalized_value):
        letter = normalized_value[0]
        return (
            re.search(rf"(?:letter|option|answer|string)\s*[\(\"']?{re.escape(letter)}[\)\"']?", lowered) is not None
            or re.search(rf"\b{re.escape(letter)}\b", lowered) is not None
        )
    if normalized_value.isdigit():
        return re.search(rf"\b{re.escape(normalized_value)}\b", lowered) is not None
    return False


def _blind_answer_consistency_row(example: JudgeBenchJoinedExample, *, severity_tier: str) -> Dict[str, Any]:
    if example.source_family == "livebench-reasoning":
        label = "Final conclusion is consistent with the state tracking"
        requirement = (
            "The response must end with the correct final conclusion, and that final answer must stay consistent "
            "with the clue constraints, state tracking, and deductions shown earlier in the response."
        )
        dimension = "conclusion_correctness"
    elif example.source_family in {"mmlu-pro", "livebench-math"}:
        label = "Final answer is supported by the worked reasoning"
        requirement = (
            "The response must choose the correct final answer, and that final answer must agree with the "
            "calculations or deductions shown earlier in the response while also following the requested output format."
        )
        dimension = "final_answer_correctness"
    else:
        label = "Final answer is internally consistent"
        requirement = (
            "The response must end with the correct final answer, and that final answer must remain consistent with "
            "the reasoning presented earlier in the response and the requested output format."
        )
        dimension = "conclusion_correctness"
    return {
        "dimension": dimension,
        "label": label,
        "requirement": requirement,
        "severity_tier": severity_tier,
        "count": 0,
        "blind_generalized_answer_row": True,
    }


def _generalize_blind_candidate_specific_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    example: JudgeBenchJoinedExample,
    pair_candidates: Sequence[CandidateNote],
) -> List[Mapping[str, Any]]:
    if not _example_is_exact_answer_task(example) or len(pair_candidates) != 2:
        return list(rows)
    candidate_values = []
    for candidate in pair_candidates:
        extraction = _extract_exact_answer_candidate(
            candidate.text,
            reference_answer=_example_reference_value(example),
            question=example.question,
        )
        if extraction is not None and extraction.value:
            candidate_values.append(extraction.value)
    distinct_values = sorted(set(candidate_values))
    if len(distinct_values) < 2:
        return list(rows)

    transformed: List[Mapping[str, Any]] = []
    for row in rows:
        text = _proposal_to_rubric_text(row)
        mentions_candidate_value = any(
            _rubric_mentions_candidate_value(text, value)
            for value in distinct_values
        )
        dimension = _normalize_text(str(row.get("dimension", ""))).lower()
        if mentions_candidate_value and (
            dimension in {"final_answer_correctness", "conclusion_correctness", "instruction_adherence"}
            or _exact_answer_rubric_kind(text, reference_answer=example.reference_answer) == "correctness"
            or "triangle count" in text.lower()
        ):
            generalized = _blind_answer_consistency_row(
                example,
                severity_tier=str(row.get("severity_tier", "")).strip() or "high",
            )
            generalized["count"] = int(row.get("count", 0) or 0)
            generalized["blind_generalized_answer_values"] = distinct_values
            generalized["blind_generalized_from"] = text
            transformed.append(generalized)
            continue
        transformed.append(dict(row))
    return transformed


def _blind_row_pruning_limits(example: JudgeBenchJoinedExample, *, blind_scoring_profile: str) -> Tuple[Optional[int], Optional[int]]:
    if blind_scoring_profile not in {
        _BLIND_SCORING_PROFILE_PRUNED_V1,
        _BLIND_SCORING_PROFILE_PRUNED_V2,
        _BLIND_SCORING_PROFILE_PRUNED_DISC_V1,
    }:
        return None, None
    if blind_scoring_profile == _BLIND_SCORING_PROFILE_PRUNED_V2:
        if example.source_family == "mmlu-pro":
            return 16, 3
        if example.source_family == "livebench-reasoning":
            return 14, 2
        if example.source_family == "livebench-math":
            return 15, 3
        if example.source_family == "livecodebench":
            return 18, 3
        return 16, 3
    if example.source_family == "mmlu-pro":
        return 18, 5
    if example.source_family == "livebench-reasoning":
        return 16, 4
    if example.source_family == "livebench-math":
        return 16, 4
    return 18, 4


def _prepare_rows_for_scoring(
    rows: Sequence[Mapping[str, Any]],
    *,
    example: JudgeBenchJoinedExample,
    pair_candidates: Sequence[CandidateNote],
    policy: Mapping[str, Any],
    reference_answer_access: bool,
) -> List[Mapping[str, Any]]:
    prepared: List[Mapping[str, Any]] = [dict(row) for row in rows]
    blind_scoring_profile = _policy_blind_scoring_profile(policy)
    if reference_answer_access or blind_scoring_profile == _BLIND_SCORING_PROFILE_BASELINE:
        return prepared

    prepared = _generalize_blind_candidate_specific_rows(
        prepared,
        example=example,
        pair_candidates=pair_candidates,
    )
    if (
        not reference_answer_access
        and not _protocol_uses_judgebench_tuning(policy)
        and _example_is_exact_answer_task(example)
    ):
        differentiating_signal_kinds = _differentiating_blind_exact_signal_kinds(example, pair_candidates)
        existing_signal_kinds = {
            str(row.get("blind_exact_signal_kind", "")).strip().lower()
            for row in prepared
            if str(row.get("blind_exact_signal_kind", "")).strip()
        }
        for row in _blind_exact_signal_rows(example):
            signal_kind = str(row.get("blind_exact_signal_kind", "")).strip().lower()
            if signal_kind and signal_kind in differentiating_signal_kinds and signal_kind not in existing_signal_kinds:
                prepared.append(row)

    chosen_by_cluster: Dict[str, Mapping[str, Any]] = {}
    for row in prepared:
        cluster = _blind_pruning_cluster_key(row, example)
        if cluster is None:
            continue
        current = chosen_by_cluster.get(cluster)
        if current is None or _blind_row_priority(
            row,
            example=example,
            blind_scoring_profile=blind_scoring_profile,
        ) > _blind_row_priority(
            current,
            example=example,
            blind_scoring_profile=blind_scoring_profile,
        ):
            chosen_by_cluster[cluster] = row

    collapsed: List[Mapping[str, Any]] = []
    emitted_clusters: set[str] = set()
    for row in prepared:
        cluster = _blind_pruning_cluster_key(row, example)
        if cluster is None:
            collapsed.append(row)
            continue
        if cluster in emitted_clusters:
            continue
        if row is chosen_by_cluster.get(cluster):
            collapsed.append(row)
            emitted_clusters.add(cluster)

    if blind_scoring_profile == _BLIND_SCORING_PROFILE_PRUNED_V2 and _example_is_exact_answer_task(example):
        format_rows = [
            row
            for row in collapsed
            if _exact_answer_rubric_kind(
                _proposal_to_rubric_text(row),
                reference_answer=example.reference_answer,
            )
            == "format"
        ]
        if len(format_rows) > 1:
            best_format_row = max(
                format_rows,
                key=lambda row: _blind_row_priority(
                    row,
                    example=example,
                    blind_scoring_profile=blind_scoring_profile,
                ),
            )
            collapsed = [
                row
                for row in collapsed
                if _exact_answer_rubric_kind(
                    _proposal_to_rubric_text(row),
                    reference_answer=example.reference_answer,
                )
                != "format"
                or row is best_format_row
            ]

    max_total, max_broad = _blind_row_pruning_limits(example, blind_scoring_profile=blind_scoring_profile)
    if max_total is None or len(collapsed) <= max_total:
        return collapsed

    indexed_rows = list(enumerate(collapsed))
    selected_indexes: set[int] = set()
    broad_count = 0
    for index, row in sorted(
        indexed_rows,
        key=lambda item: _blind_row_priority(
            item[1],
            example=example,
            blind_scoring_profile=blind_scoring_profile,
        ),
        reverse=True,
    ):
        is_broad = _row_is_broad(row)
        if is_broad and max_broad is not None and broad_count >= max_broad:
            continue
        selected_indexes.add(index)
        broad_count += int(is_broad)
        if len(selected_indexes) >= max_total:
            break
    return [row for index, row in indexed_rows if index in selected_indexes]


def _deterministic_rubric_satisfaction(
    *,
    example: JudgeBenchJoinedExample,
    rubric: RubricCriterion,
    candidate: CandidateNote,
    protocol_mode: str = _DEFAULT_PROTOCOL_MODE,
) -> Optional[Tuple[bool, Dict[str, Any]]]:
    if not _example_is_exact_answer_task(example):
        return None
    blind_signal_kind = str(rubric.metadata.get("blind_exact_signal_kind", "")).strip().lower()
    if blind_signal_kind:
        signal = _blind_exact_answer_signal(example, candidate.text)
        if blind_signal_kind == "format":
            satisfied = signal.format_ok
        elif blind_signal_kind == "choice_value_consistency":
            satisfied = signal.option_map_available and signal.consistent
        elif blind_signal_kind == "consistency":
            satisfied = signal.consistent
        else:
            return None
        return satisfied, {
            "mode": "deterministic_blind_exact_signal",
            "blind_exact_signal_kind": blind_signal_kind,
            "candidate_value": signal.value,
            "explicit": signal.explicit,
            "format_ok": signal.format_ok,
            "consistent": signal.consistent,
            "option_map_available": signal.option_map_available,
            "conflicting_markers": signal.conflicting_markers,
            "marker_count": signal.marker_count,
        }
    if not _protocol_uses_judgebench_tuning(protocol_mode):
        return None
    rubric_kind = str(rubric.metadata.get("exact_answer_kind", "")).strip().lower()
    if rubric_kind != "correctness":
        return None
    if not _example_reference_visible(example):
        return None
    reference_value = _example_reference_value(example)
    extraction = _extract_exact_answer_candidate_for_example(example, candidate.text)
    candidate_value = extraction.value if extraction is not None else None
    satisfied = bool(candidate_value) and candidate_value == reference_value
    return satisfied, {
        "mode": "deterministic_exact_answer_correctness",
        "reference_value": reference_value,
        "candidate_value": candidate_value or "",
    }


def _apply_weight_adjustments(
    *,
    example: JudgeBenchJoinedExample,
    rubrics: Sequence[RubricCriterion],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    adjusted: Dict[str, float] = {}
    exact_answer_task = _example_is_exact_answer_task(example)
    code_task = example.source_family == "livecodebench" or _looks_like_code_task(example)
    for rubric in rubrics:
        weight = float(weights.get(rubric.rubric_id, 0.0))
        multiplier = 1.0
        exact_kind = str(rubric.metadata.get("exact_answer_kind", "")).strip().lower()
        if exact_answer_task:
            if exact_kind == "correctness":
                multiplier *= 3.0
            elif exact_kind == "format":
                multiplier *= 0.35
        code_style = str(rubric.metadata.get("code_style_cluster", "")).strip()
        code_proxy = str(rubric.metadata.get("code_proxy_cluster", "")).strip()
        if code_task and code_style:
            multiplier *= 0.15
        if code_task and code_proxy:
            multiplier *= 0.35
        adjusted[rubric.rubric_id] = weight * multiplier
    total = sum(adjusted.values())
    if total <= 0.0:
        return {key: float(value) for key, value in weights.items()}
    return {key: value / total for key, value in adjusted.items()}


def _whitening_is_unstable(wu_debug: Mapping[str, Any], *, covariance_ridge: float) -> bool:
    eigenvalues = [float(value) for value in wu_debug.get("eigenvalues", []) or []]
    if not eigenvalues:
        return False
    near_ridge = sum(value <= covariance_ridge * 1.05 for value in eigenvalues)
    return near_ridge >= max(2, len(eigenvalues) // 2)


def _stabilize_whitened_uniform_result(
    *,
    example: JudgeBenchJoinedExample,
    uniform_result: Mapping[str, Any],
    wu_result: Mapping[str, Any],
    wu_debug: Mapping[str, Any],
    covariance_ridge: float,
) -> Dict[str, Any]:
    stabilized = copy.deepcopy(dict(wu_result))
    stabilized["decision_policy"] = "whitened_uniform"
    stabilized["whitening_unstable"] = _whitening_is_unstable(wu_debug, covariance_ridge=covariance_ridge)
    if not _strict_pairwise_task(example):
        return stabilized
    uniform_decision = str(uniform_result.get("decision", "")).strip()
    wu_decision = str(wu_result.get("decision", "")).strip()
    if wu_decision == "A=B" and uniform_decision in {"A>B", "B>A"}:
        chosen = copy.deepcopy(dict(uniform_result))
        chosen["decision_policy"] = "uniform_breaks_strict_tie"
        chosen["base_decision"] = wu_decision
        chosen["whitening_unstable"] = stabilized["whitening_unstable"]
        return chosen
    if (
        uniform_decision in {"A>B", "B>A"}
        and wu_decision in {"A>B", "B>A"}
        and uniform_decision != wu_decision
        and stabilized["whitening_unstable"]
    ):
        chosen = copy.deepcopy(dict(uniform_result))
        chosen["decision_policy"] = "uniform_overrides_unstable_whitening"
        chosen["base_decision"] = wu_decision
        chosen["whitening_unstable"] = True
        return chosen
    return stabilized


def _stabilize_blind_whitened_uniform_result(
    *,
    example: JudgeBenchJoinedExample,
    uniform_result: Mapping[str, Any],
    wu_result: Mapping[str, Any],
    wu_debug: Mapping[str, Any],
    covariance_ridge: float,
    blind_wu_profile: str,
) -> Dict[str, Any]:
    stabilized = copy.deepcopy(dict(wu_result))
    stabilized["decision_policy"] = "whitened_uniform"
    stabilized["whitening_unstable"] = _whitening_is_unstable(wu_debug, covariance_ridge=covariance_ridge)
    if _normalize_blind_wu_profile(blind_wu_profile) != _BLIND_WU_PROFILE_STABLE_V1:
        return stabilized
    if example.source_family not in {"mmlu-pro", "livebench-reasoning", "livebench-math"}:
        return stabilized
    if not _strict_pairwise_task(example):
        return stabilized
    uniform_decision = str(uniform_result.get("decision", "")).strip()
    wu_decision = str(wu_result.get("decision", "")).strip()
    margin = abs(float(wu_result.get("score_A", 0.0) or 0.0) - float(wu_result.get("score_B", 0.0) or 0.0))
    if (
        wu_decision == "A=B"
        and uniform_decision in {"A>B", "B>A"}
        and _pairwise_tie_has_no_signal(stabilized)
    ):
        chosen = copy.deepcopy(dict(uniform_result))
        chosen["decision_policy"] = "blind_uniform_breaks_tie"
        chosen["base_decision"] = wu_decision
        chosen["whitening_unstable"] = stabilized["whitening_unstable"]
        return chosen
    if (
        stabilized["whitening_unstable"]
        and margin <= 0.006
        and uniform_decision in {"A>B", "B>A"}
        and wu_decision in {"A>B", "B>A"}
        and uniform_decision != wu_decision
    ):
        chosen = copy.deepcopy(dict(uniform_result))
        chosen["decision_policy"] = "blind_uniform_overrides_unstable_whitening"
        chosen["base_decision"] = wu_decision
        chosen["whitening_unstable"] = True
        return chosen
    return stabilized


def _pairwise_tie_has_no_signal(result: Mapping[str, Any]) -> bool:
    pair_rows = list(result.get("pair_evaluations", []) or [])
    if not pair_rows:
        return True
    return all(bool(row.get("response_A_satisfied")) == bool(row.get("response_B_satisfied")) for row in pair_rows)


def _confidence_rank(value: str) -> int:
    return {"": 0, "low": 1, "medium": 2, "high": 3}.get(str(value or "").strip().lower(), 0)


def _build_verifier_candidate_signal(
    example: JudgeBenchJoinedExample,
    candidate_text: str,
) -> JudgeBenchVerifierCandidateSignal:
    extraction = _extract_exact_answer_candidate_for_example(example, candidate_text)
    signal = _blind_exact_answer_signal(example, candidate_text) if _example_is_exact_answer_task(example) else BlindExactAnswerSignal()
    extracted_value = extraction.value if extraction is not None else ""
    choice_letter = _choice_letter_from_answer_value(extracted_value)
    choice_map = _example_choice_value_map(example)
    return JudgeBenchVerifierCandidateSignal(
        extracted_value=extracted_value,
        extracted_source=str(extraction.source) if extraction is not None else "",
        explicit=bool(extraction is not None and extraction.explicit),
        exact_match=bool(extracted_value) and extracted_value == _example_reference_value(example),
        format_ok=bool(signal.format_ok),
        consistent=bool(signal.consistent),
        conflicting_markers=bool(signal.conflicting_markers),
        option_map_available=bool(signal.option_map_available),
        marker_count=int(signal.marker_count),
        choice_letter=choice_letter.upper(),
        choice_value=str(choice_map.get(choice_letter.lower(), "")),
        final_line=_last_nonempty_line(candidate_text),
    )


_REASONING_PROCESS_VERIFIER = ReasoningProcessVerifier(
    config=ReasoningProcessVerifierConfig(),
)


def _run_pair_verifier(
    *,
    example: JudgeBenchJoinedExample,
    pair_candidates: Sequence[CandidateNote],
    enable_reasoning_process_verifier: bool = True,
    math_solver_config: Optional[Mapping[str, Any]] = None,
    math_solver_router: Optional[LLMRouter] = None,
    math_solver_cache: Optional[JsonlCache] = None,
    math_solver_model: Optional[ModelSpec] = None,
    code_execution_config: Optional[Mapping[str, Any]] = None,
    mmlu_answerer_config: Optional[Mapping[str, Any]] = None,
    mmlu_answerer_router: Optional[LLMRouter] = None,
    mmlu_answerer_cache: Optional[JsonlCache] = None,
    mmlu_answerer_model: Optional[ModelSpec] = None,
    reasoning_solver_config: Optional[Mapping[str, Any]] = None,
    reasoning_solver_router: Optional[LLMRouter] = None,
    reasoning_solver_cache: Optional[JsonlCache] = None,
    reasoning_solver_model: Optional[ModelSpec] = None,
) -> Dict[str, Any]:
    features = _answer_key_features(example)
    code_verifier_eligible = (
        example.source_family == "livecodebench"
        and bool((code_execution_config or {}).get("enabled", False))
    )
    available = bool(
        features.exact_answer_task
        or example.source_family in {"mmlu-pro", "livebench-reasoning", "livebench-math"}
        or code_verifier_eligible
    )
    if len(pair_candidates) != 2 or not available:
        return {
            "available": available,
            "triggered": False,
            "recommended_decision": "",
            "confidence": "",
            "reason": "",
            "margin": 0.0,
            "candidate_signals": {},
            "features": {
                "exact_answer_task": bool(features.exact_answer_task),
                "requested_answer_mode": features.requested_answer_mode,
                "choice_map_available": bool(features.choice_value_map),
                "reference_value_available": bool(features.normalized_reference_value),
                "reference_answer_visible": bool(features.reference_answer_visible),
            },
        }
    code_only_path = code_verifier_eligible and example.source_family == "livecodebench"
    if code_only_path:
        signal_a = JudgeBenchVerifierCandidateSignal()
        signal_b = JudgeBenchVerifierCandidateSignal()
        outcome = JudgeBenchVerifierOutcome(
            source_family=example.source_family,
            triggered=False,
            recommended_decision="",
            confidence="",
            reason="livecodebench_pre_code_verifier",
            margin=0.0,
            candidate_signals={"A": asdict(signal_a), "B": asdict(signal_b)},
            features={
                "exact_answer_task": bool(features.exact_answer_task),
                "requested_answer_mode": features.requested_answer_mode,
                "choice_map_available": bool(features.choice_value_map),
                "reference_value_available": bool(features.normalized_reference_value),
                "reference_answer_visible": bool(features.reference_answer_visible),
            },
        )
        payload = asdict(outcome)
        payload["available"] = available
        payload.setdefault("decision_source", "exact_answer_verifier")
        if (
            example.source_family == "livecodebench"
            and bool((code_execution_config or {}).get("enabled", False))
        ):
            from dataclasses import asdict as _asdict_code

            # AtCoder-style stdin/stdout (legacy LiveCodeBench format)
            code_outcome = evaluate_code_pair_verifier(
                question=example.question or "",
                response_a=pair_candidates[0].text,
                response_b=pair_candidates[1].text,
                timeout_s=float((code_execution_config or {}).get("timeout_s", 10.0)),
                min_margin=float((code_execution_config or {}).get("min_margin", 0.34)),
            )
            code_payload = _asdict_code(code_outcome)
            payload["code_execution_verifier"] = code_payload
            code_decision = str(code_payload.get("recommended_decision", "")).strip()
            code_confidence = str(code_payload.get("confidence", "")).strip().lower()
            if (
                code_outcome.triggered
                and code_decision in {"A>B", "B>A"}
                and code_confidence in {"medium", "high"}
            ):
                payload["triggered"] = True
                payload["recommended_decision"] = code_decision
                payload["confidence"] = code_confidence
                payload["reason"] = code_payload.get("reason") or payload.get("reason", "")
                payload["margin"] = 1.0
                payload["decision_source"] = "code_execution_verifier"

            # LeetCode-style class-based test harness (most LiveCodeBench problems)
            if not payload.get("triggered"):
                leet_outcome = evaluate_leetcode_pair_verifier(
                    question=example.question or "",
                    response_a=pair_candidates[0].text,
                    response_b=pair_candidates[1].text,
                    timeout_s=float((code_execution_config or {}).get("timeout_s", 10.0)),
                    min_margin=float((code_execution_config or {}).get("min_margin", 0.34)),
                )
                leet_payload = _asdict_code(leet_outcome)
                payload["leetcode_test_runner"] = leet_payload
                leet_decision = str(leet_payload.get("recommended_decision", "")).strip()
                leet_confidence = str(leet_payload.get("confidence", "")).strip().lower()
                if (
                    leet_outcome.triggered
                    and leet_decision in {"A>B", "B>A"}
                    and leet_confidence in {"medium", "high"}
                ):
                    payload["triggered"] = True
                    payload["recommended_decision"] = leet_decision
                    payload["confidence"] = leet_confidence
                    payload["reason"] = leet_payload.get("reason") or payload.get("reason", "")
                    payload["margin"] = 1.0
                    payload["decision_source"] = "leetcode_test_runner"
        return payload
    signal_a = _build_verifier_candidate_signal(example, pair_candidates[0].text)
    signal_b = _build_verifier_candidate_signal(example, pair_candidates[1].text)
    outcome = evaluate_pair_verifier(
        source_family=example.source_family,
        features={
            "exact_answer_task": bool(features.exact_answer_task),
            "requested_answer_mode": features.requested_answer_mode,
            "choice_map_available": bool(features.choice_value_map),
            "reference_value_available": bool(features.normalized_reference_value),
            "reference_answer_visible": bool(features.reference_answer_visible),
        },
        signal_a=signal_a,
        signal_b=signal_b,
    )
    payload = asdict(outcome)
    payload["available"] = available
    if (
        enable_reasoning_process_verifier
        and _REASONING_PROCESS_VERIFIER.applies_to(example.source_family)
    ):
        process_outcome = _REASONING_PROCESS_VERIFIER.evaluate(
            source_family=example.source_family,
            response_a=pair_candidates[0].text,
            response_b=pair_candidates[1].text,
            prompt_features={
                "exact_answer_task": bool(features.exact_answer_task),
                "requested_answer_mode": features.requested_answer_mode,
                "choice_map_available": bool(features.choice_value_map),
                "reference_value_available": bool(features.normalized_reference_value),
                "reference_answer_visible": bool(features.reference_answer_visible),
            },
        )
        process_payload = asdict(process_outcome)
        payload["reasoning_process_verifier"] = process_payload
        base_decision = str(payload.get("recommended_decision", "")).strip()
        base_confidence = str(payload.get("confidence", "")).strip().lower()
        base_has_decision = base_decision in {"A>B", "B>A"} and base_confidence in {"medium", "high"}
        proc_decision = str(process_payload.get("recommended_decision", "")).strip()
        proc_confidence = str(process_payload.get("confidence", "")).strip().lower()
        if not base_has_decision and proc_decision in {"A>B", "B>A"} and proc_confidence in {"medium", "high"}:
            payload["triggered"] = True
            payload["recommended_decision"] = proc_decision
            payload["confidence"] = proc_confidence
            payload["reason"] = process_payload.get("reason") or payload.get("reason", "")
            payload["margin"] = float(process_payload.get("margin", 0.0) or 0.0)
            payload["decision_source"] = "reasoning_process_verifier"
        else:
            payload.setdefault("decision_source", "exact_answer_verifier")
    else:
        payload.setdefault("decision_source", "exact_answer_verifier")

    if (
        example.source_family == "mmlu-pro"
        and bool((mmlu_answerer_config or {}).get("enabled", False))
        and mmlu_answerer_router is not None
        and mmlu_answerer_model is not None
    ):
        from dataclasses import asdict as _asdict_mmlu

        secondary_spec_str = str((mmlu_answerer_config or {}).get("secondary_model", "") or "")
        secondary_spec = (
            parse_model_spec(secondary_spec_str, default_alias="mmlu-answerer-secondary")
            if secondary_spec_str
            else None
        )
        mmlu_outcome = evaluate_mmlu_independent_answerer(
            question=example.question or "",
            response_a=pair_candidates[0].text,
            response_b=pair_candidates[1].text,
            model_spec=mmlu_answerer_model,
            router=mmlu_answerer_router,
            cache=mmlu_answerer_cache,
            samples=int((mmlu_answerer_config or {}).get("samples", 1)),
            temperature=float((mmlu_answerer_config or {}).get("temperature", 0.5)),
            secondary_model_spec=secondary_spec,
            secondary_samples=int(
                (mmlu_answerer_config or {}).get("secondary_samples", 1)
            ),
            secondary_temperature=float(
                (mmlu_answerer_config or {}).get("secondary_temperature", 0.0)
            ),
        )
        mmlu_payload = _asdict_mmlu(mmlu_outcome)
        payload["mmlu_independent_answerer"] = mmlu_payload
        mmlu_decision = str(mmlu_payload.get("recommended_decision", "")).strip()
        mmlu_confidence = str(mmlu_payload.get("confidence", "")).strip().lower()
        reference_visible = bool(features.reference_answer_visible)
        base_decision = str(payload.get("recommended_decision", "")).strip()
        base_confidence = str(payload.get("confidence", "")).strip().lower()
        base_has_decision = (
            base_decision in {"A>B", "B>A"} and base_confidence in {"medium", "high"}
        )
        answerer_can_fire = (
            mmlu_outcome.triggered
            and mmlu_decision in {"A>B", "B>A"}
            and mmlu_confidence in {"medium", "high"}
        )
        if answerer_can_fire and not base_has_decision:
            payload["triggered"] = True
            payload["recommended_decision"] = mmlu_decision
            payload["confidence"] = mmlu_confidence
            payload["reason"] = mmlu_payload.get("reason") or payload.get("reason", "")
            payload["margin"] = 1.0
            payload["decision_source"] = "mmlu_independent_answerer"
        elif (
            answerer_can_fire
            and base_has_decision
            and not reference_visible
            and base_decision != mmlu_decision
        ):
            # Blind mmlu-pro: the format-based exact_answer_verifier scores by consistency/format
            # rather than factual correctness, so it can confidently disagree with an independent
            # solver that actually knows the answer. The solver's letter is grounded in the
            # question content, so we override base disagreements when the answerer fires high.
            payload["triggered"] = True
            payload["recommended_decision"] = mmlu_decision
            payload["confidence"] = mmlu_confidence
            payload["reason"] = (
                f"mmlu_answerer_overrode_{base_decision.replace('>', '_gt_').lower()}"
            )
            payload["margin"] = 1.0
            payload["decision_source"] = "mmlu_independent_answerer"

    if (
        example.source_family == "livebench-math"
        and bool((math_solver_config or {}).get("enabled", False))
        and math_solver_router is not None
        and math_solver_model is not None
    ):
        from dataclasses import asdict as _asdict

        solver_outcome = evaluate_math_independent_solver(
            question=example.question or "",
            response_a=pair_candidates[0].text,
            response_b=pair_candidates[1].text,
            model_spec=math_solver_model,
            router=math_solver_router,
            cache=math_solver_cache,
            samples=int((math_solver_config or {}).get("samples", 1)),
            temperature=float((math_solver_config or {}).get("temperature", 0.5)),
            use_sympy=bool((math_solver_config or {}).get("use_sympy", False)),
        )
        solver_payload = _asdict(solver_outcome)
        payload["math_independent_solver"] = solver_payload
        base_has_decision = (
            str(payload.get("recommended_decision", "")).strip() in {"A>B", "B>A"}
            and str(payload.get("confidence", "")).strip().lower() in {"medium", "high"}
        )
        solver_decision = str(solver_payload.get("recommended_decision", "")).strip()
        solver_confidence = str(solver_payload.get("confidence", "")).strip().lower()
        if (
            not base_has_decision
            and solver_outcome.triggered
            and solver_decision in {"A>B", "B>A"}
            and solver_confidence in {"medium", "high"}
        ):
            payload["triggered"] = True
            payload["recommended_decision"] = solver_decision
            payload["confidence"] = solver_confidence
            payload["reason"] = solver_payload.get("reason") or payload.get("reason", "")
            payload["margin"] = 1.0
            payload["decision_source"] = "math_independent_solver"

    if (
        example.source_family == "livebench-reasoning"
        and bool((reasoning_solver_config or {}).get("enabled", False))
        and reasoning_solver_router is not None
        and reasoning_solver_model is not None
    ):
        from dataclasses import asdict as _asdict_reason

        reason_outcome = evaluate_reasoning_independent_solver(
            question=example.question or "",
            response_a=pair_candidates[0].text,
            response_b=pair_candidates[1].text,
            model_spec=reasoning_solver_model,
            router=reasoning_solver_router,
            cache=reasoning_solver_cache,
            samples=int((reasoning_solver_config or {}).get("samples", 1)),
            temperature=float((reasoning_solver_config or {}).get("temperature", 0.5)),
        )
        reason_payload = _asdict_reason(reason_outcome)
        payload["reasoning_independent_solver"] = reason_payload
        reason_decision = str(reason_payload.get("recommended_decision", "")).strip()
        reason_confidence = str(reason_payload.get("confidence", "")).strip().lower()
        base_decision = str(payload.get("recommended_decision", "")).strip()
        base_confidence = str(payload.get("confidence", "")).strip().lower()
        base_has_decision = (
            base_decision in {"A>B", "B>A"} and base_confidence in {"medium", "high"}
        )
        solver_can_fire = (
            reason_outcome.triggered
            and reason_decision in {"A>B", "B>A"}
            and reason_confidence in {"medium", "high"}
        )
        reference_visible = bool(features.reference_answer_visible)
        if solver_can_fire and not base_has_decision:
            payload["triggered"] = True
            payload["recommended_decision"] = reason_decision
            payload["confidence"] = reason_confidence
            payload["reason"] = reason_payload.get("reason") or payload.get("reason", "")
            payload["margin"] = 1.0
            payload["decision_source"] = "reasoning_independent_solver"
        elif (
            solver_can_fire
            and base_has_decision
            and not reference_visible
            and base_decision != reason_decision
        ):
            # Same rationale as the MMLU answerer override on blind data: the
            # reasoning_process_verifier is a deterministic format-based check that can
            # confidently disagree with an independent solver that actually solved the
            # puzzle. The solver's canonical answer is grounded in the question content
            # rather than candidate format, so we override base disagreements when the
            # solver fires high.
            payload["triggered"] = True
            payload["recommended_decision"] = reason_decision
            payload["confidence"] = reason_confidence
            payload["reason"] = (
                f"reasoning_solver_overrode_{base_decision.replace('>', '_gt_').lower()}"
            )
            payload["margin"] = 1.0
            payload["decision_source"] = "reasoning_independent_solver"

    return payload


def _apply_pair_verifier_result(
    *,
    scoring: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> Dict[str, Any]:
    updated = copy.deepcopy(dict(scoring))
    updated["pair_verifier"] = dict(verifier)
    decision = str(verifier.get("recommended_decision", "")).strip()
    confidence = str(verifier.get("confidence", "")).strip().lower()
    if decision not in {"A>B", "B>A"} or confidence not in {"medium", "high"}:
        return updated
    decision_source = str(verifier.get("decision_source", "")).strip()
    # These verifiers produce deterministic / high-precision factual signals (test pass-rate,
    # canonical solver answer, MCQ letter match). When they fire HIGH confidence we trust
    # them over the rubric scoring regardless of whitened margin. Format-based exact_answer
    # / reasoning_process verifiers retain the original conservative override gating.
    #
    # ``reasoning_independent_solver`` is intentionally excluded: free-form puzzle answers
    # are noisier to canonicalise than test-pass-rates / canonical math answers / MCQ
    # letters, and on blind-350 it overrode at ~63% precision (12 right / 7 wrong of 19).
    high_precision_sources = {
        "code_execution_verifier",
        "leetcode_test_runner",
        "mmlu_independent_answerer",
        "math_independent_solver",
    }
    # Also honor the high-precision pathway when the base ``exact_answer_verifier`` happens
    # to agree with a high-confidence sub-verifier (MMLU answerer / math solver / code).
    # Without this, ``decision_source`` stays as ``exact_answer_verifier`` whenever the
    # format-based check matches the ground-truth solver, and the override fails to
    # propagate against the conservative whitened-margin gate.
    sub_verifier_high_confidence = False
    for sub_key in (
        "mmlu_independent_answerer",
        "math_independent_solver",
        "code_execution_verifier",
        "leetcode_test_runner",
    ):
        sub = verifier.get(sub_key)
        if not isinstance(sub, Mapping):
            continue
        if not sub.get("triggered"):
            continue
        if str(sub.get("recommended_decision", "")).strip() != decision:
            continue
        if str(sub.get("confidence", "")).strip().lower() != "high":
            continue
        sub_verifier_high_confidence = True
        break
    is_high_precision = confidence == "high" and (
        decision_source in high_precision_sources or sub_verifier_high_confidence
    )
    for method_name in ("uniform", "whitened_uniform"):
        result = ((updated.get(method_name) or {}).get("result") or {})
        current_decision = str(result.get("decision", "")).strip()
        margin = abs(float(result.get("score_A", 0.0) or 0.0) - float(result.get("score_B", 0.0) or 0.0))
        if current_decision == decision and current_decision != "A=B":
            continue
        if current_decision == "A=B":
            if not _pairwise_tie_has_no_signal(result):
                continue
        elif is_high_precision:
            pass
        elif not (confidence == "high" and bool(result.get("whitening_unstable")) and margin <= 0.006):
            continue
        result["base_decision"] = current_decision
        result["base_decision_policy"] = str(result.get("decision_policy", "")).strip()
        result["decision_policy"] = "pair_verifier"
        result["decision"] = decision
        result["decision_reversed"] = flip_decision(decision)
        result["tie_break_reason"] = "pair_verifier"
        result["verifier_reason"] = str(verifier.get("reason", "")).strip()
        result["verifier_confidence"] = confidence
        result["verifier_margin"] = float(verifier.get("margin", 0.0) or 0.0)
    return updated


def _normalize_pair_discriminator_decision(
    raw_decision: str,
    *,
    left_id: str,
    right_id: str,
    left_pair_position: str,
) -> str:
    normalized = _normalize_text(raw_decision).upper().replace(" ", "")
    left_id = str(left_id).strip().upper()
    right_id = str(right_id).strip().upper()
    if normalized in {f"{left_id}={right_id}", f"{right_id}={left_id}", "A=B"}:
        return "A=B"
    if normalized in {f"{left_id}>{right_id}", "A>B"}:
        return "A>B" if str(left_pair_position).strip().upper() == "A" else "B>A"
    if normalized in {f"{right_id}>{left_id}", "B>A"}:
        return "B>A" if str(left_pair_position).strip().upper() == "A" else "A>B"
    return ""


def _run_pair_discriminator_once(
    *,
    example_record: ExampleRecord,
    left_candidate: CandidateNote,
    right_candidate: CandidateNote,
    left_pair_position: str,
    route_decision: JudgeBenchRouteDecision,
    calibration_guidance: str,
    discovery_model: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    left_id: str,
    right_id: str,
    order_label: str,
    temperature: float = 0.0,
    sample_index: int = 0,
    few_shot_block: str = "",
    self_critique_enabled: bool = False,
) -> Dict[str, Any]:
    system_prompt, user_prompt = build_pair_discriminator_prompts(
        example=example_record,
        candidate_a=left_candidate,
        candidate_b=right_candidate,
        task_profile_id=route_decision.task_profile_id,
        artifact_label=get_task_profile(route_decision.task_profile_id).artifact_label,
        calibration_guidance=calibration_guidance,
        candidate_a_id=left_id,
        candidate_b_id=right_id,
        few_shot_block=few_shot_block,
    )
    payload = {
        "pair_id": example_record.metadata.get("pair_id", example_record.example_id),
        "task_profile_id": route_decision.task_profile_id,
        "candidate_left": stable_hash(left_candidate.text),
        "candidate_right": stable_hash(right_candidate.text),
        "left_pair_position": str(left_pair_position),
        "left_id": left_id,
        "right_id": right_id,
        "reference": stable_hash(example_record.reference_artifact),
        "calibration_guidance": calibration_guidance,
    }
    if temperature > 0.0 or sample_index > 0:
        payload["temperature"] = round(float(temperature), 4)
        payload["sample_index"] = int(sample_index)
    cache_key = make_cache_key(_PAIR_DISCRIMINATOR_PROMPT_VERSION, payload)
    cache_hit = False
    raw_text = ""
    if cache and cache.enabled:
        cache.load()
        hit = cache.get(cache_key)
        if hit and isinstance(hit.get("raw_response"), str):
            raw_text = hit["raw_response"]
            cache_hit = True
    if not raw_text:
        response = router.generate(
            discovery_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(temperature),
        )
        raw_text = response.raw_text or response.text
        if cache and cache.enabled:
            cache.set(cache_key, {"raw_response": raw_text, "kind": "pair_discriminator"})
    raw_decision = ""
    explanation = ""
    confidence = ""
    parse_error = ""
    obj = extract_json_object(raw_text)
    if isinstance(obj, Mapping):
        raw_decision = _normalize_text(str(obj.get("decision", "")))
        explanation = _normalize_text(str(obj.get("distinguishing_behavior", "")))
        confidence = _normalize_text(str(obj.get("confidence", ""))).lower()
    decision = _normalize_pair_discriminator_decision(
        raw_decision,
        left_id=left_id,
        right_id=right_id,
        left_pair_position=left_pair_position,
    )
    if decision not in {"A>B", "B>A", "A=B"}:
        parse_error = "pair_discriminator_invalid_or_missing_decision"
        decision = ""
    if confidence not in {"high", "medium", "low"}:
        confidence = ""

    self_critique_payload: Optional[Dict[str, Any]] = None
    if self_critique_enabled and decision in {"A>B", "B>A", "A=B"}:
        critique_user_prompt = (
            f"{user_prompt}\n\n"
            "Your previous verdict was the JSON below. Critique it: identify any flaws in the "
            "reasoning, missed constraints, or biases (positional, length, surface polish). "
            "If you find a flaw that would change the verdict, output a corrected JSON in the "
            "exact same shape. If the verdict is sound, repeat it.\n\n"
            f"Previous verdict: {raw_text.strip()}"
        )
        critique_payload_for_key = dict(payload)
        critique_payload_for_key["self_critique"] = True
        critique_cache_key = make_cache_key(
            _PAIR_DISCRIMINATOR_PROMPT_VERSION + "_self_critique",
            critique_payload_for_key,
        )
        critique_raw = ""
        critique_cache_hit = False
        if cache and cache.enabled:
            cache.load()
            critique_hit = cache.get(critique_cache_key)
            if critique_hit and isinstance(critique_hit.get("raw_response"), str):
                critique_raw = critique_hit["raw_response"]
                critique_cache_hit = True
        if not critique_raw:
            try:
                critique_response = router.generate(
                    discovery_model,
                    system_prompt=system_prompt,
                    user_prompt=critique_user_prompt,
                    temperature=float(temperature),
                )
                critique_raw = critique_response.raw_text or critique_response.text
                if cache and cache.enabled:
                    cache.set(
                        critique_cache_key,
                        {"raw_response": critique_raw, "kind": "pair_discriminator_self_critique"},
                    )
            except Exception:
                critique_raw = ""
        critique_obj = extract_json_object(critique_raw or "")
        if isinstance(critique_obj, Mapping):
            critique_raw_decision = _normalize_text(str(critique_obj.get("decision", "")))
            critique_confidence = _normalize_text(str(critique_obj.get("confidence", ""))).lower()
            critique_explanation = _normalize_text(
                str(critique_obj.get("distinguishing_behavior", ""))
            )
            critique_decision = _normalize_pair_discriminator_decision(
                critique_raw_decision,
                left_id=left_id,
                right_id=right_id,
                left_pair_position=left_pair_position,
            )
            if critique_decision in {"A>B", "B>A", "A=B"}:
                self_critique_payload = {
                    "decision": critique_decision,
                    "confidence": critique_confidence
                    if critique_confidence in {"high", "medium", "low"}
                    else "",
                    "distinguishing_behavior": critique_explanation,
                    "raw_response": critique_raw,
                    "cache_hit": bool(critique_cache_hit),
                }
                # Adopt the critique's verdict if it changed the decision OR if the original
                # was empty (parse error). Otherwise keep the initial decision.
                if (
                    critique_decision != decision
                    and critique_decision in {"A>B", "B>A", "A=B"}
                ):
                    decision = critique_decision
                    confidence = self_critique_payload["confidence"] or confidence
                    if self_critique_payload["distinguishing_behavior"]:
                        explanation = self_critique_payload["distinguishing_behavior"]
                    parse_error = ""

    return {
        "order": order_label,
        "decision": decision,
        "raw_decision": raw_decision,
        "distinguishing_behavior": explanation,
        "confidence": confidence,
        "raw_response": raw_text,
        "cache_hit": cache_hit,
        "parse_error": parse_error,
        "temperature": round(float(temperature), 4),
        "sample_index": int(sample_index),
        "self_critique": self_critique_payload,
    }


def _aggregate_pair_discriminator_attempts(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    normalized_attempts = [dict(attempt) for attempt in attempts]
    valid_attempts = [attempt for attempt in normalized_attempts if str(attempt.get("decision", "")).strip() in {"A>B", "B>A", "A=B"}]
    directional_attempts = [
        attempt for attempt in valid_attempts if str(attempt.get("decision", "")).strip() in {"A>B", "B>A"}
    ]
    decision = ""
    confidence = ""
    explanation = ""
    parse_error = ""
    order_consistent = True
    if not valid_attempts:
        parse_error = next(
            (
                str(attempt.get("parse_error", "")).strip()
                for attempt in normalized_attempts
                if str(attempt.get("parse_error", "")).strip()
            ),
            "pair_discriminator_no_valid_attempts",
        )
    elif not directional_attempts:
        decision = "A=B"
        confidence = "low"
    else:
        direction_set = {str(attempt.get("decision", "")).strip() for attempt in directional_attempts}
        if len(direction_set) > 1:
            order_consistent = False
            parse_error = "pair_discriminator_order_disagreement"
        else:
            decision = next(iter(direction_set))
            supporting = [attempt for attempt in directional_attempts if str(attempt.get("decision", "")).strip() == decision]
            best_attempt = max(supporting, key=lambda attempt: (_confidence_rank(str(attempt.get("confidence", ""))), attempt.get("order", "")))
            explanation = str(best_attempt.get("distinguishing_behavior", "")).strip()
            if len(supporting) >= 2 and min(_confidence_rank(str(item.get("confidence", ""))) for item in supporting) >= 2:
                confidence = "high"
            elif _confidence_rank(str(best_attempt.get("confidence", ""))) >= 2:
                confidence = "medium"
            else:
                confidence = "low"
            if any(str(attempt.get("decision", "")).strip() == "A=B" for attempt in valid_attempts) and confidence == "high":
                confidence = "medium"
    return {
        "decision": decision,
        "distinguishing_behavior": explanation,
        "confidence": confidence,
        "raw_response": "\n\n".join(
            str(attempt.get("raw_response", "")).strip()
            for attempt in normalized_attempts
            if str(attempt.get("raw_response", "")).strip()
        ),
        "cache_hit": bool(normalized_attempts) and all(bool(attempt.get("cache_hit")) for attempt in normalized_attempts),
        "parse_error": parse_error,
        "order_consistent": order_consistent,
        "attempts": normalized_attempts,
    }


def _run_pair_discriminator(
    *,
    example_record: ExampleRecord,
    pair_candidates: Sequence[CandidateNote],
    route_decision: JudgeBenchRouteDecision,
    calibration_guidance: str,
    discovery_model: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    self_consistency_n: int = 1,
    self_consistency_temperature: float = 0.0,
    few_shot_block: str = "",
    self_critique_enabled: bool = False,
) -> Dict[str, Any]:
    if len(pair_candidates) < 2:
        return {
            "decision": "",
            "distinguishing_behavior": "",
            "confidence": "",
            "raw_response": "",
            "cache_hit": False,
            "parse_error": "pair_discriminator_missing_candidates",
            "order_consistent": False,
            "attempts": [],
        }
    samples = max(1, int(self_consistency_n or 1))
    temperature = float(self_consistency_temperature) if samples > 1 else 0.0
    attempts: List[Dict[str, Any]] = []
    for sample_index in range(samples):
        sample_temp = temperature if sample_index > 0 else (0.0 if samples == 1 else temperature)
        attempts.append(
            _run_pair_discriminator_once(
                example_record=example_record,
                left_candidate=pair_candidates[0],
                right_candidate=pair_candidates[1],
                left_pair_position="A",
                route_decision=route_decision,
                calibration_guidance=calibration_guidance,
                discovery_model=discovery_model,
                router=router,
                cache=cache,
                left_id="X",
                right_id="Y",
                order_label=f"AB_{sample_index}" if samples > 1 else "AB",
                temperature=sample_temp,
                sample_index=sample_index,
                few_shot_block=few_shot_block,
                self_critique_enabled=self_critique_enabled,
            )
        )
        attempts.append(
            _run_pair_discriminator_once(
                example_record=example_record,
                left_candidate=pair_candidates[1],
                right_candidate=pair_candidates[0],
                left_pair_position="B",
                route_decision=route_decision,
                calibration_guidance=calibration_guidance,
                discovery_model=discovery_model,
                router=router,
                cache=cache,
                left_id="X",
                right_id="Y",
                order_label=f"BA_{sample_index}" if samples > 1 else "BA",
                temperature=sample_temp,
                sample_index=sample_index,
                few_shot_block=few_shot_block,
                self_critique_enabled=self_critique_enabled,
            )
        )
    aggregated = _aggregate_pair_discriminator_attempts(attempts)
    aggregated["self_consistency_n"] = samples
    aggregated["self_consistency_temperature"] = round(temperature, 4)
    return aggregated


def _apply_pair_discriminator_result(
    *,
    scoring: Mapping[str, Any],
    discriminator: Mapping[str, Any],
    source_family: str = "",
) -> Dict[str, Any]:
    updated = copy.deepcopy(dict(scoring))
    if source_family == "mmlu-pro":
        return updated
    decision = str(discriminator.get("decision", "")).strip()
    confidence = str(discriminator.get("confidence", "")).strip().lower()
    updated["pair_discriminator"] = dict(discriminator)
    if decision not in {"A>B", "B>A"} or confidence == "low":
        return updated
    for method_name in ("uniform", "whitened_uniform"):
        result = ((updated.get(method_name) or {}).get("result") or {})
        if str(result.get("decision", "")).strip() != "A=B":
            continue
        if not _pairwise_tie_has_no_signal(result):
            continue
        result["base_decision_policy"] = str(result.get("decision_policy", "")).strip()
        result["decision_policy"] = "pairwise_discriminator"
        result["decision"] = decision
        result["decision_reversed"] = flip_decision(decision)
        result["tie_break_reason"] = "pairwise_discriminator"
        result["tie_discriminator_explanation"] = str(discriminator.get("distinguishing_behavior", "")).strip()
        result["tie_discriminator_confidence"] = confidence
        result["tie_discriminator_order_consistent"] = bool(discriminator.get("order_consistent", False))
    return updated


def _reasoning_small_margin_discriminator_candidate(example: JudgeBenchJoinedExample) -> bool:
    return (
        example.source_family == "livebench-reasoning"
        and not _example_is_exact_answer_task(example)
        and _looks_like_code_task(example)
    )


def _should_run_blind_pair_discriminator(
    *,
    policy: Mapping[str, Any],
    example: JudgeBenchJoinedExample,
    scoring: Mapping[str, Any],
    rubric_count: int,
    verifier_outcome: Optional[Mapping[str, Any]] = None,
) -> bool:
    if _policy_blind_scoring_profile(policy) != _BLIND_SCORING_PROFILE_PRUNED_DISC_V1:
        return False
    if example.source_family not in {"mmlu-pro", "livebench-reasoning", "livebench-math"}:
        return False
    mode = _policy_blind_discriminator_mode(policy, example.source_family)
    if mode == _BLIND_DISCRIMINATOR_MODE_OFF:
        return False
    verifier_decision = str((verifier_outcome or {}).get("recommended_decision", "")).strip()
    verifier_confidence = str((verifier_outcome or {}).get("confidence", "")).strip().lower()
    if verifier_decision in {"A>B", "B>A"} and verifier_confidence in {"medium", "high"}:
        return False
    reasoning_small_margin_route = _reasoning_small_margin_discriminator_candidate(example)
    min_rubric_count = 16 if mode == _BLIND_DISCRIMINATOR_MODE_STRICT else 14
    unstable_margin = 0.0025 if mode == _BLIND_DISCRIMINATOR_MODE_STRICT else 0.0035
    if reasoning_small_margin_route:
        min_rubric_count = 14 if mode == _BLIND_DISCRIMINATOR_MODE_STRICT else 12
        unstable_margin = 0.01 if mode == _BLIND_DISCRIMINATOR_MODE_STRICT else 0.0125
    wide_gate = _policy_v2_wide_discriminator_gate(policy)
    if wide_gate:
        min_rubric_count = min(min_rubric_count, 10)
        unstable_margin = max(unstable_margin, _V2_WIDER_GATE_LOW_MARGIN)
        if example.source_family == "livebench-reasoning":
            unstable_margin = max(unstable_margin, _V2_WIDER_GATE_REASONING_MARGIN)
    result = ((scoring.get("whitened_uniform") or {}).get("result") or {})
    decision = str(result.get("decision", "")).strip()
    if decision not in {"A>B", "B>A", "A=B"}:
        return False
    if rubric_count < min_rubric_count:
        return False
    if decision == "A=B":
        return _pairwise_tie_has_no_signal(result)
    margin = abs(float(result.get("score_A", 0.0) or 0.0) - float(result.get("score_B", 0.0) or 0.0))
    if wide_gate:
        family_margin = _V2_WIDER_GATE_LOW_MARGIN
        if example.source_family == "livebench-reasoning":
            family_margin = _V2_WIDER_GATE_REASONING_MARGIN
        if margin <= family_margin or bool(result.get("whitening_unstable")):
            return True
    if reasoning_small_margin_route:
        return margin <= unstable_margin
    return bool(result.get("whitening_unstable")) and margin <= unstable_margin


def _apply_blind_pair_discriminator_result(
    *,
    scoring: Mapping[str, Any],
    discriminator: Mapping[str, Any],
) -> Dict[str, Any]:
    updated = copy.deepcopy(dict(scoring))
    decision = str(discriminator.get("decision", "")).strip()
    confidence = str(discriminator.get("confidence", "")).strip().lower()
    updated["pair_discriminator"] = dict(discriminator)
    if decision not in {"A>B", "B>A"} or confidence == "low":
        return updated
    result = ((updated.get("whitened_uniform") or {}).get("result") or {})
    current_decision = str(result.get("decision", "")).strip()
    if current_decision == decision and current_decision != "A=B":
        return updated
    result["base_decision"] = current_decision
    result["base_decision_policy"] = str(result.get("decision_policy", "")).strip()
    result["decision_policy"] = "blind_pair_discriminator"
    result["decision"] = decision
    result["decision_reversed"] = flip_decision(decision)
    result["tie_break_reason"] = "blind_pair_discriminator"
    result["tie_discriminator_explanation"] = str(discriminator.get("distinguishing_behavior", "")).strip()
    result["tie_discriminator_confidence"] = confidence
    result["tie_discriminator_order_consistent"] = bool(discriminator.get("order_consistent", False))
    return updated


def _resolve_scoring_model(explicit_model: Optional[str]) -> ModelSpec:
    if explicit_model and explicit_model.strip():
        return parse_model_spec(explicit_model.strip(), default_alias="judgebench-scoring")
    spec = discover_default_comparison_judge_model()
    if spec is None:
        raise ValueError(
            "No JudgeBench scoring model configured. Set RUBRIC_GEN_COMPARISON_JUDGE_MODEL or provide --judge-model."
        )
    return spec


def _policy_route_bundle(policy: Dict[str, Any], source_family: str) -> Dict[str, Any]:
    return (policy.get("source_family_routes", {}) or {}).get(source_family) or policy["fallback_route"]


def _append_unique(items: List[str], new_text: str) -> bool:
    normalized = _normalize_text(new_text)
    if not normalized:
        return False
    lowered = normalized.lower()
    existing = {_normalize_text(item).lower() for item in items}
    if lowered in existing:
        return False
    items.append(normalized)
    return True


def _maybe_add_mutation(route_bundle: Dict[str, Any], mutation_id: str) -> bool:
    strategy_bundle = route_bundle.get("strategy_bundle", {})
    mutation_ids = [str(item) for item in strategy_bundle.get("mutation_ids", []) if str(item).strip()]
    if mutation_id in mutation_ids:
        return False
    mutation_ids.append(mutation_id)
    strategy_bundle["mutation_ids"] = mutation_ids
    route_bundle["strategy_bundle"] = strategy_bundle
    return True


def _propose_policy_refinement(
    *,
    current_policy: Mapping[str, Any],
    split_result: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not _protocol_uses_judgebench_tuning(current_policy):
        return None, []
    failures = list(split_result.get("failures", []))
    if not failures:
        return None, []

    policy = copy.deepcopy(dict(current_policy))
    actions: List[Dict[str, Any]] = []
    failures_by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    broad_failures = 0
    exact_format_failures = 0
    for failure in failures:
        source_family = str(failure.get("source_family", "")).strip() or "unknown"
        failures_by_family[source_family].append(failure)
        broad_failures += int(failure.get("broad_rubric_count", 0) or 0) > 0
        if bool(failure.get("exact_answer_task", False)):
            exact_format_failures += 1

    for source_family, family_failures in sorted(failures_by_family.items()):
        family_nudges = list((policy.get("prompt_nudges", {}) or {}).get(source_family, []))
        before_count = len(family_nudges)
        for failure in family_failures:
            for nudge in failure.get("suggested_nudges", []) or []:
                _append_unique(family_nudges, str(nudge))
        if len(family_nudges) > before_count:
            policy.setdefault("prompt_nudges", {})[source_family] = family_nudges[:6]
            actions.append(
                {
                    "kind": "add_prompt_nudges",
                    "source_family": source_family,
                    "count_added": len(family_nudges) - before_count,
                }
            )
        if any(bool(failure.get("exact_answer_task", False)) for failure in family_failures):
            route_bundle = _policy_route_bundle(policy, source_family)
            if _maybe_add_mutation(route_bundle, "corrupt_final_answer"):
                actions.append(
                    {
                        "kind": "add_mutation",
                        "source_family": source_family,
                        "mutation_id": "corrupt_final_answer",
                    }
                )

    recursion_config = dict(policy.get("recursion_config", {}) or {})
    family_recursion_config = copy.deepcopy(dict(policy.get("family_recursion_config", {}) or {}))
    if broad_failures >= max(2, len(failures) // 3):
        if int(recursion_config.get("max_recursive_parents_per_pair", 2)) < 4:
            recursion_config["max_recursive_parents_per_pair"] = int(recursion_config.get("max_recursive_parents_per_pair", 2)) + 1
            actions.append(
                {
                    "kind": "increase_recursive_parents",
                    "value": recursion_config["max_recursive_parents_per_pair"],
                }
            )
        if int(recursion_config.get("max_children_per_parent", 3)) < 5:
            recursion_config["max_children_per_parent"] = int(recursion_config.get("max_children_per_parent", 3)) + 1
            actions.append(
                {
                    "kind": "increase_recursive_children",
                    "value": recursion_config["max_children_per_parent"],
                }
            )
        if failures_by_family.get("mmlu-pro"):
            mmlu_recursion = dict(family_recursion_config.get("mmlu-pro", {}) or {})
            changed = False
            if int(mmlu_recursion.get("max_depth", recursion_config.get("max_depth", 1))) < 2:
                mmlu_recursion["max_depth"] = 2
                changed = True
            if (
                int(
                    mmlu_recursion.get(
                        "max_recursive_parents_per_pair",
                        recursion_config.get("max_recursive_parents_per_pair", 2),
                    )
                )
                < 3
            ):
                mmlu_recursion["max_recursive_parents_per_pair"] = 3
                changed = True
            if int(mmlu_recursion.get("max_children_per_parent", recursion_config.get("max_children_per_parent", 3))) < 4:
                mmlu_recursion["max_children_per_parent"] = 4
                changed = True
            if changed:
                mmlu_recursion["max_recursive_calls_per_pair"] = int(
                    mmlu_recursion.get(
                        "max_recursive_calls_per_pair",
                        recursion_config.get("max_recursive_calls_per_pair", 2),
                    )
                )
                family_recursion_config["mmlu-pro"] = mmlu_recursion
                actions.append(
                    {
                        "kind": "increase_family_recursive_budget",
                        "source_family": "mmlu-pro",
                        "value": dict(mmlu_recursion),
                    }
                )
    policy["recursion_config"] = recursion_config
    policy["family_recursion_config"] = family_recursion_config

    if exact_format_failures > 0:
        global_nudges = list((policy.get("prompt_nudges", {}) or {}).get("global", []))
        if _append_unique(global_nudges, _GRANULARITY_PROMPT_NUDGE):
            policy.setdefault("prompt_nudges", {})["global"] = global_nudges[:6]
            actions.append({"kind": "add_global_granularity_nudge"})

    if not actions:
        return None, []
    return policy, actions


def _split_comparison_key(summary: Mapping[str, Any]) -> Tuple[float, float, float]:
    metrics = summary.get("wu_metrics", {}) or {}
    overall = float(metrics.get("overall", 0.0))
    source_values = [float(value) for key, value in metrics.items() if key != "overall"]
    worst_source = min(source_values) if source_values else overall
    pair_count = max(1, int(summary.get("pair_count", 0) or 0))
    ties = int((summary.get("decision_counts", {}) or {}).get("A=B", 0) or 0)
    tie_rate = ties / pair_count
    return overall, worst_source, -tie_rate


def _is_better_split(candidate_summary: Mapping[str, Any], best_summary: Mapping[str, Any]) -> bool:
    cand = _split_comparison_key(candidate_summary)
    best = _split_comparison_key(best_summary)
    if cand[0] > best[0] + 1e-9:
        return True
    if abs(cand[0] - best[0]) <= 1e-9 and cand[1] > best[1] + 1e-9:
        return True
    if abs(cand[0] - best[0]) <= 1e-9 and abs(cand[1] - best[1]) <= 1e-9 and cand[2] > best[2] + 1e-9:
        return True
    return False


def _example_signature(
    example: JudgeBenchJoinedExample,
    *,
    reference_answer_access: bool = True,
) -> str:
    effective_example = _example_for_reference_access(example, reference_answer_access=reference_answer_access)
    return stable_hash(to_json_dict(effective_example))


def _artifact_fingerprint(
    *,
    split_name: str,
    example: JudgeBenchJoinedExample,
    policy: Mapping[str, Any],
    discovery_model: ModelSpec,
    judge_model: ModelSpec,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    covariance_ridge: float,
    reference_answer_access: bool = True,
    retrieval_fingerprint: str = "",
) -> str:
    return stable_hash(
        {
            "fingerprint_version": _ARTIFACT_FINGERPRINT_VERSION,
            "split_name": split_name,
            "example_signature": _example_signature(example, reference_answer_access=reference_answer_access),
            "policy": to_json_dict(dict(policy)),
            "discovery_model": {
                "provider": discovery_model.provider,
                "model": discovery_model.model,
                "base_url": discovery_model.base_url or "",
            },
            "judge_model": {
                "provider": judge_model.provider,
                "model": judge_model.model,
                "base_url": judge_model.base_url or "",
            },
            "max_criteria": max_criteria,
            "max_pairs_per_example": max_pairs_per_example,
            "covariance_ridge": covariance_ridge,
            "reference_answer_access": bool(reference_answer_access),
            "retrieval_fingerprint": str(retrieval_fingerprint or ""),
        }
    )


def _accumulate_split_stats_from_artifact(stats: Dict[str, int], artifact: Mapping[str, Any]) -> None:
    discovery = artifact.get("discovery", {}) or {}
    for pair_payload in discovery.get("pairs", []) or []:
        if bool(pair_payload.get("parse_error")):
            stats["pairs_failed_parse"] += 1
        else:
            stats["pairs_succeeded"] += 1
        recursion = dict(pair_payload.get("recursion", {}) or {})
        stats["local_proposals_total"] += len(pair_payload.get("raw_proposals", []) or []) + int(
            recursion.get("recursive_children_raw_total", 0) or 0
        )
        stats["local_proposals_promoted"] += len(pair_payload.get("proposals", []) or [])
        stats["local_proposals_rejected_grounding"] += len(pair_payload.get("rejected_proposals", []) or [])
        for key in (
            "recursive_calls",
            "recursive_cache_hits",
            "recursive_parse_failures",
            "recursive_parents_considered",
            "recursive_parents_expanded",
            "recursive_children_raw_total",
            "recursive_children_promoted",
            "recursive_children_rejected_grounding",
        ):
            stats[key] += int(recursion.get(key, 0) or 0)
    if bool(discovery.get("recursive_changed", False)):
        stats["examples_with_recursive_change"] += 1
    stats["rubric_evaluations_total"] += len(artifact.get("evaluations", []) or [])
    evaluation_stats = artifact.get("evaluation_stats", {}) or {}
    stats["rubric_evaluation_cache_hits"] += int(evaluation_stats.get("cache_hits", 0) or 0)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _clear_split_bookkeeping(split_dir: Path) -> None:
    for rel_path in (
        Path("routing/decisions.json"),
        Path("reports/rrd_wu_predictions.jsonl"),
        Path("reports/rrd_uniform_predictions.jsonl"),
        Path("summaries/summary.json"),
        Path("rejected_refinement.json"),
    ):
        _unlink_if_exists(split_dir / rel_path)


def _clear_run_bookkeeping(
    *,
    train_root: Path,
    refinement_dir: Path,
    frozen_policy_dir: Path,
    calibration_dir: Path,
    summaries_dir: Path,
) -> None:
    for path in (
        train_root / "best_iteration.json",
        train_root / "accepted_refinements.json",
        train_root / "best_summary.json",
        refinement_dir / "accepted_refinements.json",
        calibration_dir / "prompt_nudges.json",
        calibration_dir / "frozen_recursion_config.json",
        frozen_policy_dir / "best_policy.json",
        summaries_dir / "summary.json",
    ):
        _unlink_if_exists(path)
    for path in frozen_policy_dir.glob("accepted_policy_iter_*.json"):
        _unlink_if_exists(path)


def _make_split_stats(*, pairs_total: int = 0) -> Dict[str, int]:
    return {
        "pairs_total": pairs_total,
        "pairs_succeeded": 0,
        "pairs_failed_parse": 0,
        "local_proposals_total": 0,
        "local_proposals_promoted": 0,
        "local_proposals_rejected_grounding": 0,
        "recursive_calls": 0,
        "recursive_cache_hits": 0,
        "recursive_parse_failures": 0,
        "recursive_parents_considered": 0,
        "recursive_parents_expanded": 0,
        "recursive_children_raw_total": 0,
        "recursive_children_promoted": 0,
        "recursive_children_rejected_grounding": 0,
        "examples_with_recursive_change": 0,
        "rubric_evaluations_total": 0,
        "rubric_evaluation_cache_hits": 0,
    }


def _merge_split_stats(target: Dict[str, int], source: Mapping[str, Any]) -> None:
    for key in target:
        if key == "pairs_total":
            continue
        target[key] += int(source.get(key, 0) or 0)


def _process_judgebench_example(
    *,
    example: JudgeBenchJoinedExample,
    split_name: str,
    examples_dir: Path,
    policy: Mapping[str, Any],
    discovery_model: ModelSpec,
    judge_model: ModelSpec,
    discovery_cache: JsonlCache,
    scoring_cache: JsonlCache,
    recursive_config: RecursiveDiscoveryConfig,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    covariance_ridge: float,
    resume: bool,
    write_example_artifacts: bool,
    reference_answer_access: bool = True,
    retrieval_examples: Optional[Sequence[JudgeBenchJoinedExample]] = None,
    retrieval_fingerprint: str = "",
) -> Dict[str, Any]:
    effective_example = _example_for_reference_access(example, reference_answer_access=reference_answer_access)
    example_path = examples_dir / f"{example.pair_id}.json"
    route_recorded = False
    route_decisions: List[Dict[str, Any]] = []
    protocol_mode = _policy_protocol_mode(policy)
    expected_fingerprint = _artifact_fingerprint(
        split_name=split_name,
        example=example,
        policy=policy,
        discovery_model=discovery_model,
        judge_model=judge_model,
        max_criteria=max_criteria,
        max_pairs_per_example=max_pairs_per_example,
        covariance_ridge=covariance_ridge,
        reference_answer_access=reference_answer_access,
        retrieval_fingerprint=retrieval_fingerprint,
    )
    expected_example_signature = _example_signature(example, reference_answer_access=reference_answer_access)
    if resume and write_example_artifacts and example_path.exists():
        artifact = json.loads(example_path.read_text(encoding="utf-8"))
        if (
            artifact.get("artifact_fingerprint") != expected_fingerprint
            or artifact.get("example_signature") != expected_example_signature
        ):
            artifact = {}
    else:
        artifact = {}

    if not artifact:
        router = LLMRouter()
        route_decision = build_route_decision(
            example,
            policy,
            reference_answer_access=reference_answer_access,
        )
        route_decisions.append(to_json_dict(route_decision))
        route_recorded = True
        effective_recursive_config = RecursiveDiscoveryConfig(
            max_depth=int(route_decision.recursion_config.get("max_depth", recursive_config.max_depth)),
            max_recursive_parents_per_pair=int(
                route_decision.recursion_config.get(
                    "max_recursive_parents_per_pair",
                    recursive_config.max_recursive_parents_per_pair,
                )
            ),
            max_children_per_parent=int(
                route_decision.recursion_config.get("max_children_per_parent", recursive_config.max_children_per_parent)
            ),
            max_recursive_calls_per_pair=int(
                route_decision.recursion_config.get(
                    "max_recursive_calls_per_pair",
                    recursive_config.max_recursive_calls_per_pair,
                )
            ),
        )
        example_record = joined_example_to_example_record(
            example,
            task_profile_id=route_decision.task_profile_id,
            reference_answer_access=reference_answer_access,
        )
        discovery_pairs, pair_candidates, scoring_candidates = build_judgebench_candidates(
            example=example,
            example_record=example_record,
            route_decision=route_decision,
            max_pairs_per_example=max_pairs_per_example,
            reference_answer_access=reference_answer_access,
            policy=policy,
        )
        calibration_guidance = build_calibration_guidance(
            policy,
            example.source_family,
            example=example,
            reference_answer_access=reference_answer_access,
        )
        retrieval_hits: List[Dict[str, Any]] = []
        retrieval_seed_rows: List[Dict[str, Any]] = []
        if retrieval_examples:
            retrieval_guidance, retrieval_hits, retrieval_seed_rows = _build_retrieval_guidance(
                example=example,
                retrieval_examples=retrieval_examples,
                policy=policy,
            )
            if retrieval_guidance:
                calibration_guidance = (
                    f"{calibration_guidance}\n\n{retrieval_guidance}"
                    if calibration_guidance.strip()
                    else retrieval_guidance
                )
        example_local_rows: List[Dict[str, Any]] = []
        pair_payloads: List[Dict[str, Any]] = []
        example_recursive_changed = False

        for strong, weak in discovery_pairs:
            pair_result = discover_pair_criteria(
                example=example_record,
                strong=strong,
                weak=weak,
                model_spec=discovery_model,
                router=router,
                cache=discovery_cache,
                max_criteria=max_criteria,
                task_profile_id=route_decision.task_profile_id,
                artifact_label=get_task_profile(route_decision.task_profile_id).artifact_label,
                calibration_guidance=calibration_guidance,
                recursive_config=effective_recursive_config,
            )
            recursion = dict(pair_result.get("recursion", {}))
            if bool(recursion.get("changed_structure")):
                example_recursive_changed = True
            example_local_rows.extend(pair_result["proposals"])
            pair_payloads.append(
                {
                    "pair_id": pair_result["pair_id"],
                    "strong_candidate_id": strong.candidate_id,
                    "strong_source_label": strong.source_label,
                    "weak_candidate_id": weak.candidate_id,
                    "weak_source_label": weak.source_label,
                    "cache": pair_result["cache"],
                    "parse_error": pair_result["parse_error"],
                    "grounding": pair_result["grounding"],
                    "raw_proposals": pair_result["raw_proposals"],
                    "promoted_root_proposals": pair_result["promoted_root_proposals"],
                    "rejected_root_proposals": pair_result["rejected_root_proposals"],
                    "proposals": pair_result["proposals"],
                    "rejected_proposals": pair_result["rejected_proposals"],
                    "recursive_steps": pair_result["recursive_steps"],
                    "recursion": pair_result["recursion"],
                }
            )

        library_rows = _maybe_library_rows_for_example(
            example=effective_example,
            example_record=example_record,
            policy=policy,
        )
        if bool(policy.get("enable_rrd_filters", False)):
            pair_contexts_for_filters = [
                {
                    "pair_id": payload["pair_id"],
                    "strong_text": payload.get("strong_text", "")
                    or next(
                        (c.text for c in pair_candidates if c.candidate_id == payload["strong_candidate_id"]),
                        "",
                    ),
                    "weak_text": payload.get("weak_text", "")
                    or next(
                        (c.text for c in pair_candidates if c.candidate_id == payload["weak_candidate_id"]),
                        "",
                    ),
                }
                for payload in pair_payloads
            ]
            merged = merge_proposal_entries_with_rrd_filters(
                example_local_rows,
                pair_contexts=_build_pair_contexts_for_rrd(pair_contexts_for_filters),
                redundancy_threshold=float(policy.get("rrd_redundancy_threshold", 0.9)),
            )
        else:
            merged = merge_proposal_entries(example_local_rows)
        if retrieval_seed_rows:
            merged["canonical_proposals"] = list(merged.get("canonical_proposals", []) or []) + retrieval_seed_rows
        if library_rows:
            merged["canonical_proposals"] = library_rows + list(merged.get("canonical_proposals", []) or [])
            merged["library_rows_injected"] = len(library_rows)
        prepared_rows = _prepare_rows_for_scoring(
            merged["canonical_proposals"],
            example=effective_example,
            pair_candidates=pair_candidates,
            policy=policy,
            reference_answer_access=reference_answer_access,
        )
        rubrics = canonical_rows_to_rubrics(
            example_record.example_id,
            prepared_rows,
            example=effective_example,
            protocol_mode=protocol_mode,
        )
        evaluations, evaluation_stats = evaluate_rubrics_on_candidates(
            example=effective_example,
            rubrics=rubrics,
            candidates=scoring_candidates,
            judge_model=judge_model,
            cache=scoring_cache,
            router=router,
            protocol_mode=protocol_mode,
            rubric_satisfaction_samples=_policy_rubric_satisfaction_samples(policy),
            rubric_satisfaction_temperature=_policy_rubric_satisfaction_temperature(policy),
        )
        scoring = score_discovered_rubrics_for_pair(
            example=effective_example,
            rubrics=rubrics,
            scoring_candidates=scoring_candidates,
            pair_candidates=pair_candidates,
            evaluations=evaluations,
            covariance_ridge=covariance_ridge,
            protocol_mode=protocol_mode,
            policy=policy,
        )
        verifier_outcome = _run_pair_verifier(
            example=effective_example,
            pair_candidates=pair_candidates,
            math_solver_config=(
                {
                    "enabled": _policy_math_independent_solver_enabled(policy),
                    "samples": _policy_math_solver_samples(policy),
                    "temperature": _policy_math_solver_temperature(policy),
                    "use_sympy": _policy_math_solver_use_sympy(policy),
                }
                if _policy_math_independent_solver_enabled(policy)
                else None
            ),
            math_solver_router=router if _policy_math_independent_solver_enabled(policy) else None,
            math_solver_cache=scoring_cache if _policy_math_independent_solver_enabled(policy) else None,
            math_solver_model=(
                (
                    parse_model_spec(_policy_math_solver_model(policy), default_alias="math-solver")
                    if _policy_math_solver_model(policy)
                    else judge_model
                )
                if _policy_math_independent_solver_enabled(policy)
                else None
            ),
            code_execution_config=(
                {
                    "enabled": _policy_code_execution_verifier_enabled(policy),
                    "timeout_s": _policy_code_execution_timeout_s(policy),
                    "min_margin": _policy_code_execution_min_margin(policy),
                }
                if _policy_code_execution_verifier_enabled(policy)
                else None
            ),
            mmlu_answerer_config=(
                {
                    "enabled": _policy_mmlu_independent_answerer_enabled(policy),
                    "samples": _policy_mmlu_answerer_samples(policy),
                    "temperature": _policy_mmlu_answerer_temperature(policy),
                    "secondary_model": _policy_mmlu_answerer_secondary_model(policy),
                    "secondary_samples": int(policy.get("mmlu_answerer_secondary_samples", 1) or 1),
                    "secondary_temperature": float(
                        policy.get("mmlu_answerer_secondary_temperature", 0.0) or 0.0
                    ),
                }
                if _policy_mmlu_independent_answerer_enabled(policy)
                else None
            ),
            mmlu_answerer_router=router if _policy_mmlu_independent_answerer_enabled(policy) else None,
            mmlu_answerer_cache=scoring_cache if _policy_mmlu_independent_answerer_enabled(policy) else None,
            mmlu_answerer_model=(
                (
                    parse_model_spec(_policy_mmlu_answerer_model(policy), default_alias="mmlu-answerer")
                    if _policy_mmlu_answerer_model(policy)
                    else judge_model
                )
                if _policy_mmlu_independent_answerer_enabled(policy)
                else None
            ),
            reasoning_solver_config=(
                {
                    "enabled": _policy_reasoning_independent_solver_enabled(policy),
                    "samples": _policy_reasoning_solver_samples(policy),
                    "temperature": _policy_reasoning_solver_temperature(policy),
                }
                if _policy_reasoning_independent_solver_enabled(policy)
                else None
            ),
            reasoning_solver_router=router if _policy_reasoning_independent_solver_enabled(policy) else None,
            reasoning_solver_cache=scoring_cache if _policy_reasoning_independent_solver_enabled(policy) else None,
            reasoning_solver_model=(
                (
                    parse_model_spec(_policy_reasoning_solver_model(policy), default_alias="reasoning-solver")
                    if _policy_reasoning_solver_model(policy)
                    else judge_model
                )
                if _policy_reasoning_independent_solver_enabled(policy)
                else None
            ),
        )
        if not reference_answer_access:
            scoring = _apply_pair_verifier_result(
                scoring=scoring,
                verifier=verifier_outcome,
            )
        if (
            reference_answer_access
            and
            _protocol_uses_judgebench_tuning(protocol_mode)
            and example.source_family != "mmlu-pro"
            and
            str(((scoring.get("uniform") or {}).get("result") or {}).get("decision", "")).strip() == "A=B"
            and str(((scoring.get("whitened_uniform") or {}).get("result") or {}).get("decision", "")).strip() == "A=B"
            and _pairwise_tie_has_no_signal(((scoring.get("whitened_uniform") or {}).get("result") or {}))
        ):
            discriminator = _run_pair_discriminator(
                example_record=example_record,
                pair_candidates=pair_candidates,
                route_decision=route_decision,
                calibration_guidance=calibration_guidance,
                discovery_model=discovery_model,
                router=router,
                cache=discovery_cache,
            )
            scoring = _apply_pair_discriminator_result(
                scoring=scoring,
                discriminator=discriminator,
                source_family=example.source_family,
            )
        elif _should_run_blind_pair_discriminator(
            policy=policy,
            example=effective_example,
            scoring=scoring,
            rubric_count=len(rubrics),
            verifier_outcome=verifier_outcome,
        ):
            few_shot_block = ""
            if _policy_few_shot_train_enabled(policy):
                index = _load_policy_few_shot_index(policy)
                if index is not None:
                    from rubric_gen.compiled.labeled_train_few_shot import (
                        build_few_shot_block,
                    )

                    few_shot_block = build_few_shot_block(
                        index=index,
                        query=example.question or "",
                        top_k=_policy_few_shot_top_k(policy),
                        same_family=example.source_family,
                    )
            discriminator = _run_pair_discriminator(
                example_record=example_record,
                pair_candidates=pair_candidates,
                route_decision=route_decision,
                calibration_guidance=calibration_guidance,
                discovery_model=discovery_model,
                router=router,
                cache=discovery_cache,
                self_consistency_n=_policy_self_consistency_n(policy),
                self_consistency_temperature=_policy_self_consistency_temperature(policy),
                few_shot_block=few_shot_block,
                self_critique_enabled=_policy_discriminator_self_critique_enabled(policy),
            )
            scoring = _apply_blind_pair_discriminator_result(
                scoring=scoring,
                discriminator=discriminator,
            )

        if _policy_holistic_judge_enabled(policy):
            wu_result = (scoring.get("whitened_uniform") or {}).get("result") or {}
            wu_decision = str(wu_result.get("decision", "") or "").strip()
            wu_margin = abs(float(wu_result.get("score_A", 0.0) or 0.0) - float(wu_result.get("score_B", 0.0) or 0.0))
            wu_policy = str(wu_result.get("decision_policy", "") or "").strip()
            low_margin_threshold = 0.05
            needs_holistic = (
                wu_decision == "A=B"
                or wu_margin <= low_margin_threshold
                or bool(wu_result.get("whitening_unstable"))
                or wu_policy in {"", "whitened_uniform"}
            )
            already_resolved_by_discriminator = wu_policy in {"pairwise_discriminator", "blind_pair_discriminator"}
            if needs_holistic and not already_resolved_by_discriminator:
                holistic = run_holistic_pair_judge(
                    example_record=example_record,
                    rubrics=rubrics,
                    pair_candidates=pair_candidates,
                    task_profile_id=route_decision.task_profile_id,
                    judge_model=judge_model,
                    router=router,
                    cache=scoring_cache,
                )
                scoring = apply_holistic_judge_to_scoring(
                    scoring=scoring,
                    holistic=holistic,
                    low_margin_threshold=low_margin_threshold,
                )

        artifact = {
            "schema": "compiled_judgebench_example_v2",
            "split_name": split_name,
            "artifact_fingerprint": expected_fingerprint,
            "example_signature": expected_example_signature,
            "pair": _public_example_payload(effective_example),
            "reference_answer_access": bool(reference_answer_access),
            "routing_decision": to_json_dict(route_decision),
            "calibration_guidance": calibration_guidance or None,
            "recursive_config": {
                "max_depth": effective_recursive_config.max_depth,
                "max_recursive_parents_per_pair": effective_recursive_config.max_recursive_parents_per_pair,
                "max_children_per_parent": effective_recursive_config.max_children_per_parent,
                "max_recursive_calls_per_pair": effective_recursive_config.max_recursive_calls_per_pair,
            },
            "candidates": [to_json_dict(item) for item in scoring_candidates],
            "discovery": {
                "pairs": pair_payloads,
                "merged": {
                    **merged,
                    "prepared_canonical_proposals": [to_json_dict(dict(row)) for row in prepared_rows],
                },
                "recursive_changed": example_recursive_changed,
            },
            "retrieval": {
                "profile": _policy_retrieval_profile(policy, effective_example.source_family),
                "top_k": _policy_retrieval_top_k(policy, effective_example.source_family),
                "profile_by_family": _normalize_retrieval_profile_by_family(policy.get("retrieval_profile_by_family")),
                "top_k_by_family": _normalize_retrieval_top_k_by_family(policy.get("retrieval_top_k_by_family")),
                "hits": retrieval_hits,
                "seed_rows": [to_json_dict(dict(row)) for row in retrieval_seed_rows],
            },
            "rubrics": [to_json_dict(item) for item in rubrics],
            "evaluations": [to_json_dict(item) for item in evaluations],
            "evaluation_stats": evaluation_stats,
            "verifier": verifier_outcome,
            "scoring": scoring,
            "analysis": {
                "broad_rubric_count": _count_broad_canonical_rows(prepared_rows),
                "raw_broad_rubric_count": _count_broad_canonical_rows(merged["canonical_proposals"]),
                "exact_answer_task": _example_is_exact_answer_task(effective_example),
                "code_task": _looks_like_code_task(effective_example),
            },
        }
        if write_example_artifacts:
            write_json(example_path, artifact)

    if not route_recorded and "routing_decision" in artifact:
        route_decisions.append(dict(artifact["routing_decision"]))

    local_stats = _make_split_stats()
    _accumulate_split_stats_from_artifact(local_stats, artifact)
    wu_result = artifact["scoring"]["whitened_uniform"]["result"]
    uniform_result = artifact["scoring"]["uniform"]["result"]
    verifier_payload = dict(artifact.get("verifier", {}) or {})
    verifier_signals = dict(verifier_payload.get("candidate_signals", {}) or {})
    verifier_signal_a = dict(verifier_signals.get("A", {}) or {})
    verifier_signal_b = dict(verifier_signals.get("B", {}) or {})
    pair_discriminator = dict((artifact.get("scoring", {}) or {}).get("pair_discriminator", {}) or {})
    decision_policy = str(wu_result.get("decision_policy", "")).strip()
    tie_break_reason = str(wu_result.get("tie_break_reason", "")).strip()
    pair_margin = abs(float(wu_result.get("score_A", 0.0) or 0.0) - float(wu_result.get("score_B", 0.0) or 0.0))
    wu_row = {
        "pair_id": example.pair_id,
        "source": example.source,
        "label": example.label,
        "decision_original": wu_result["decision"],
        "decision_reversed": wu_result["decision_reversed"],
        "score_A": wu_result["score_A"],
        "score_B": wu_result["score_B"],
        "rubric_count": len(artifact["rubrics"]),
        "source_family": example.source_family,
    }
    uniform_row = {
        "pair_id": example.pair_id,
        "source": example.source,
        "label": example.label,
        "decision_original": uniform_result["decision"],
        "decision_reversed": uniform_result["decision_reversed"],
        "score_A": uniform_result["score_A"],
        "score_B": uniform_result["score_B"],
        "rubric_count": len(artifact["rubrics"]),
        "source_family": example.source_family,
    }
    failure: Optional[Dict[str, Any]] = None
    if wu_result["decision"] != example.label:
        failure = {
            "pair_id": example.pair_id,
            "source": example.source,
            "source_family": example.source_family,
            "label": example.label,
            "decision": wu_result["decision"],
            "score_A": wu_result["score_A"],
            "score_B": wu_result["score_B"],
            "broad_rubric_count": int(artifact.get("analysis", {}).get("broad_rubric_count", 0) or 0),
            "exact_answer_task": bool(artifact.get("analysis", {}).get("exact_answer_task", False)),
            "code_task": bool(artifact.get("analysis", {}).get("code_task", False)),
            "suggested_nudges": (
                _source_family_nudges(effective_example) if _protocol_uses_judgebench_tuning(protocol_mode) else []
            ),
            "routing_task_profile_id": str(artifact.get("routing_decision", {}).get("task_profile_id", "")),
            "rubric_count": len(artifact.get("rubrics", [])),
            "decision_policy": decision_policy,
            "tie_break_reason": tie_break_reason,
            "verifier_confidence": str(verifier_payload.get("confidence", "")).strip(),
            "verifier_reason": str(verifier_payload.get("reason", "")).strip(),
        }
    analysis_row = {
        "pair_id": example.pair_id,
        "source": example.source,
        "source_family": example.source_family,
        "label": example.label,
        "decision": wu_result["decision"],
        "rubric_count": len(artifact.get("rubrics", [])),
        "broad_rubric_count": int(artifact.get("analysis", {}).get("broad_rubric_count", 0) or 0),
        "exact_answer_task": bool(artifact.get("analysis", {}).get("exact_answer_task", False)),
        "code_task": bool(artifact.get("analysis", {}).get("code_task", False)),
        "routing_task_profile_id": str(artifact.get("routing_decision", {}).get("task_profile_id", "")),
        "tie": str(wu_result["decision"]).strip() == "A=B",
        "reference_answer_access": bool(reference_answer_access),
        "weak_source_labels": [str(payload.get("weak_source_label", "")).strip() for payload in artifact.get("discovery", {}).get("pairs", [])],
        "decision_policy": decision_policy,
        "tie_break_reason": tie_break_reason,
        "pair_margin": pair_margin,
        "verifier_available": bool(verifier_payload.get("available")),
        "verifier_triggered": bool(verifier_payload.get("triggered")),
        "verifier_recommended_decision": str(verifier_payload.get("recommended_decision", "")).strip(),
        "verifier_confidence": str(verifier_payload.get("confidence", "")).strip().lower(),
        "verifier_reason": str(verifier_payload.get("reason", "")).strip(),
        "verifier_margin": float(verifier_payload.get("margin", 0.0) or 0.0),
        "verifier_exact_match_A": bool(verifier_signal_a.get("exact_match")),
        "verifier_exact_match_B": bool(verifier_signal_b.get("exact_match")),
        "verifier_consistent_A": bool(verifier_signal_a.get("consistent")),
        "verifier_consistent_B": bool(verifier_signal_b.get("consistent")),
        "exact_answer_parser_success_A": bool(str(verifier_signal_a.get("extracted_value", "")).strip()),
        "exact_answer_parser_success_B": bool(str(verifier_signal_b.get("extracted_value", "")).strip()),
        "discriminator_used": decision_policy in {"pairwise_discriminator", "blind_pair_discriminator"},
        "discriminator_confidence": str(pair_discriminator.get("confidence", "")).strip().lower(),
        "discriminator_order_consistent": bool(pair_discriminator.get("order_consistent", False)),
    }
    return {
        "route_decisions": route_decisions,
        "wu_row": wu_row,
        "uniform_row": uniform_row,
        "failure": failure,
        "stats": local_stats,
        "analysis_row": analysis_row,
    }


def _assert_split_summary_persisted(split_result: Mapping[str, Any]) -> None:
    summary_path = Path(str((split_result.get("paths", {}) or {}).get("summary", "")).strip())
    if not summary_path.exists():
        raise RuntimeError(f"Expected split summary to be written, but it was missing: {summary_path}")
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    in_memory = split_result.get("summary", {})
    if stable_hash(persisted) != stable_hash(in_memory):
        raise RuntimeError(
            "Persisted split summary did not match the in-memory split result. "
            f"path={summary_path}"
        )


def run_judgebench_split(
    *,
    examples: Sequence[JudgeBenchJoinedExample],
    split_name: str,
    split_dir: Path,
    policy: Mapping[str, Any],
    discovery_model_override: Optional[str],
    judge_model_override: Optional[str],
    use_cache: bool,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    covariance_ridge: float,
    max_workers: int,
    resume: bool,
    write_example_artifacts: bool = True,
    write_reports: bool = True,
    reference_answer_access: bool = True,
    retrieval_examples: Optional[Sequence[JudgeBenchJoinedExample]] = None,
    shared_cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    apply_frozen_policy(policy)
    _clear_split_bookkeeping(split_dir)

    routing_dir = split_dir / "routing"
    examples_dir = split_dir / "examples"
    reports_dir = split_dir / "reports"
    summaries_dir = split_dir / "summaries"
    cache_dir = Path(shared_cache_dir) if shared_cache_dir else (split_dir / "cache")
    for path in (routing_dir, examples_dir, reports_dir, summaries_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    discovery_model = resolve_compiled_judge_spec(discovery_model_override)
    judge_model = _resolve_scoring_model(judge_model_override)
    discovery_cache = JsonlCache(cache_dir / "discovery.jsonl", enabled=use_cache)
    scoring_cache = JsonlCache(cache_dir / "scoring.jsonl", enabled=use_cache)
    discovery_cache.load()
    scoring_cache.load()
    retrieval_examples = list(retrieval_examples or [])
    retrieval_fingerprint = (
        stable_hash(
            [
                {
                    "pair_id": row.pair_id,
                    "source_family": row.source_family,
                    "question": row.question,
                    "reference_answer": row.reference_answer,
                    "label": row.label,
                }
                for row in retrieval_examples
            ]
        )
        if retrieval_examples
        else ""
    )
    recursive_config = RecursiveDiscoveryConfig(
        max_depth=int((policy.get("recursion_config", {}) or {}).get("max_depth", 1)),
        max_recursive_parents_per_pair=int(
            (policy.get("recursion_config", {}) or {}).get("max_recursive_parents_per_pair", 2)
        ),
        max_children_per_parent=int((policy.get("recursion_config", {}) or {}).get("max_children_per_parent", 3)),
        max_recursive_calls_per_pair=int(
            (policy.get("recursion_config", {}) or {}).get("max_recursive_calls_per_pair", 2)
        ),
    )

    route_decisions: List[Dict[str, Any]] = []
    wu_rows: List[Dict[str, Any]] = []
    uniform_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    analysis_rows: List[Dict[str, Any]] = []
    stats = _make_split_stats(pairs_total=len(examples))
    worker_count = max(1, min(int(max_workers), len(examples))) if examples else 1

    def process_example(current: JudgeBenchJoinedExample) -> Dict[str, Any]:
        return _process_judgebench_example(
            example=current,
            split_name=split_name,
            examples_dir=examples_dir,
            policy=policy,
            discovery_model=discovery_model,
            judge_model=judge_model,
            discovery_cache=discovery_cache,
            scoring_cache=scoring_cache,
            recursive_config=recursive_config,
            max_criteria=max_criteria,
            max_pairs_per_example=max_pairs_per_example,
            covariance_ridge=covariance_ridge,
            resume=resume,
            write_example_artifacts=write_example_artifacts,
            reference_answer_access=reference_answer_access,
            retrieval_examples=retrieval_examples,
            retrieval_fingerprint=retrieval_fingerprint,
        )

    if worker_count == 1:
        example_results = [process_example(example) for example in examples]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"{split_name}_worker") as executor:
            example_results = list(executor.map(process_example, examples))

    for result in example_results:
        route_decisions.extend(result["route_decisions"])
        wu_rows.append(dict(result["wu_row"]))
        uniform_rows.append(dict(result["uniform_row"]))
        if result["failure"] is not None:
            failures.append(dict(result["failure"]))
        analysis_rows.append(dict(result["analysis_row"]))
        _merge_split_stats(stats, result["stats"])

    route_counts = Counter(str(item.get("task_profile_id", "")) for item in route_decisions)
    calibration_metrics = build_judgebench_calibration_metrics(
        split_result={
            "summary": {"pair_count": len(examples)},
            "analysis_rows": analysis_rows,
            "failures": failures,
        }
    )
    summary = {
        "schema": "compiled_judgebench_split_summary_v1",
        "split_name": split_name,
        "policy_hash": stable_hash(to_json_dict(dict(policy))),
        "reference_answer_access": bool(reference_answer_access),
        "blind_guidance_profile": _policy_blind_guidance_profile(policy),
        "blind_wu_profile": _policy_blind_wu_profile(policy),
        "retrieval_profile": _policy_retrieval_profile(policy),
        "blind_discriminator_mode_by_family": _normalize_blind_discriminator_mode_by_family(
            policy.get("blind_discriminator_mode_by_family")
        ),
        "retrieval_profile_by_family": _normalize_retrieval_profile_by_family(policy.get("retrieval_profile_by_family")),
        "retrieval_top_k": _policy_retrieval_top_k(policy),
        "retrieval_top_k_by_family": _normalize_retrieval_top_k_by_family(policy.get("retrieval_top_k_by_family")),
        "max_workers": worker_count,
        "pair_count": len(examples),
        "wu_metrics": metric_breakdown(wu_rows),
        "uniform_metrics": metric_breakdown(uniform_rows),
        "decision_counts": dict(Counter(str(row.get("decision_original", "")) for row in wu_rows)),
        "source_family_counts": dict(Counter(example.source_family for example in examples)),
        "task_profile_counts": dict(sorted(route_counts.items())),
        "avg_rubric_count": (
            sum(float(row.get("rubric_count", 0) or 0) for row in wu_rows) / max(1, len(wu_rows))
        ),
        "failure_count": len(failures),
        "calibration_metrics": calibration_metrics,
        "stats": stats,
    }
    if write_reports:
        write_json(routing_dir / "decisions.json", {"decisions": route_decisions})
    write_json(summaries_dir / "summary.json", summary)
    if write_reports:
        _write_jsonl(reports_dir / "rrd_wu_predictions.jsonl", wu_rows)
        _write_jsonl(reports_dir / "rrd_uniform_predictions.jsonl", uniform_rows)
    return {
        "summary": summary,
        "wu_rows": wu_rows,
        "uniform_rows": uniform_rows,
        "failures": failures,
        "analysis_rows": analysis_rows,
        "route_decisions": route_decisions,
        "paths": {
            "split_dir": str(split_dir.resolve()),
            "summary": str((summaries_dir / "summary.json").resolve()),
        },
    }


def run_judgebench_recursive_evaluation(
    *,
    train_dataset_path: Path,
    validation_dataset_path: Path,
    train_split_name: str,
    validation_split_name: str,
    protocol_mode: str,
    run_dir: Path,
    official_dataset_path: Optional[Path],
    discovery_model_override: Optional[str],
    judge_model_override: Optional[str],
    use_cache: bool,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    bootstrap_iterations: int,
    refine_iterations: int,
    recursive_config: RecursiveDiscoveryConfig,
    covariance_ridge: float,
    max_workers: int,
    resume: bool,
) -> Tuple[Path, Dict[str, Any]]:
    run_dir = Path(run_dir)
    protocol_mode = _normalize_protocol_mode(protocol_mode)
    train_split_name = _safe_slug(train_split_name)
    validation_split_name = _safe_slug(validation_split_name)
    dataset_dir = run_dir / "dataset"
    frozen_policy_dir = run_dir / "frozen_policy"
    refinement_dir = run_dir / "refinement"
    calibration_dir = run_dir / "calibration"
    train_root = run_dir / train_split_name
    val_root = run_dir / validation_split_name
    summaries_dir = run_dir / "summaries"
    for path in (dataset_dir, frozen_policy_dir, refinement_dir, calibration_dir, train_root, val_root, summaries_dir):
        path.mkdir(parents=True, exist_ok=True)
    _clear_run_bookkeeping(
        train_root=train_root,
        refinement_dir=refinement_dir,
        frozen_policy_dir=frozen_policy_dir,
        calibration_dir=calibration_dir,
        summaries_dir=summaries_dir,
    )

    official_path = ensure_official_judgebench_dataset(official_dataset_path, dataset_dir)
    official_pairs = load_official_judgebench_pairs(official_path)
    train_local = load_local_judgebench_subset(train_dataset_path)
    val_local = load_local_judgebench_subset(validation_dataset_path)
    train_joined = join_local_subset_to_official_pairs(
        local_rows=train_local,
        official_pairs=official_pairs,
        split_name=train_split_name,
    )
    val_joined = join_local_subset_to_official_pairs(
        local_rows=val_local,
        official_pairs=official_pairs,
        split_name=validation_split_name,
    )
    ensure_disjoint_pair_ids(train_joined, val_joined)

    write_json(dataset_dir / f"joined_{train_split_name}.json", train_joined)
    write_json(dataset_dir / f"joined_{validation_split_name}.json", val_joined)
    write_json(
        dataset_dir / "dataset_manifest.json",
        {
            "official_dataset_path": str(official_path),
            "train_dataset_path": str(Path(train_dataset_path).resolve()),
            "validation_dataset_path": str(Path(validation_dataset_path).resolve()),
            "train_split_name": train_split_name,
            "validation_split_name": validation_split_name,
            "train_pair_count": len(train_joined),
            "validation_pair_count": len(val_joined),
        },
    )

    initial_policy = build_initial_frozen_policy(
        train_examples=train_joined,
        bootstrap_iterations=bootstrap_iterations,
        recursive_config=recursive_config,
        protocol_mode=protocol_mode,
    )
    write_json(frozen_policy_dir / "initial_policy.json", initial_policy)

    best_policy = copy.deepcopy(initial_policy)
    accepted_refinements: List[Dict[str, Any]] = []
    train_iterations_root = train_root / "iterations"
    train_iterations_root.mkdir(parents=True, exist_ok=True)

    best_train_result = run_judgebench_split(
        examples=train_joined,
        split_name=train_split_name,
        split_dir=train_iterations_root / "iter_00",
        policy=best_policy,
        discovery_model_override=discovery_model_override,
        judge_model_override=judge_model_override,
        use_cache=use_cache,
        max_criteria=max_criteria,
        max_pairs_per_example=max_pairs_per_example,
        covariance_ridge=covariance_ridge,
        max_workers=max_workers,
        resume=resume,
    )
    _assert_split_summary_persisted(best_train_result)
    best_iteration_index = 0

    for refine_index in range(1, max(0, refine_iterations) + 1):
        candidate_policy, actions = _propose_policy_refinement(
            current_policy=best_policy,
            split_result=best_train_result,
        )
        if candidate_policy is None or not actions:
            break
        candidate_policy["refinement_history"] = list(best_policy.get("refinement_history", [])) + [
            {"iteration": refine_index, "actions": actions}
        ]
        iteration_dir = train_iterations_root / f"iter_{refine_index:02d}"
        write_json(iteration_dir / "candidate_policy.json", candidate_policy)
        candidate_result = run_judgebench_split(
            examples=train_joined,
            split_name=train_split_name,
            split_dir=iteration_dir,
            policy=candidate_policy,
            discovery_model_override=discovery_model_override,
            judge_model_override=judge_model_override,
            use_cache=use_cache,
            max_criteria=max_criteria,
            max_pairs_per_example=max_pairs_per_example,
            covariance_ridge=covariance_ridge,
            max_workers=max_workers,
            resume=resume,
        )
        _assert_split_summary_persisted(candidate_result)
        if _is_better_split(candidate_result["summary"], best_train_result["summary"]):
            best_policy = candidate_policy
            best_train_result = candidate_result
            best_iteration_index = refine_index
            accepted_refinements.append({"iteration": refine_index, "actions": actions})
            write_json(frozen_policy_dir / f"accepted_policy_iter_{refine_index:02d}.json", best_policy)
            _unlink_if_exists(iteration_dir / "rejected_refinement.json")
            continue
        write_json(
            iteration_dir / "rejected_refinement.json",
            {
                "actions": actions,
                "best_train_summary": best_train_result["summary"],
                "candidate_train_summary": candidate_result["summary"],
            },
        )
        break

    write_json(frozen_policy_dir / "best_policy.json", best_policy)
    write_json(train_root / "best_iteration.json", {"best_iteration_index": best_iteration_index})
    write_json(train_root / "accepted_refinements.json", {"accepted_refinements": accepted_refinements})
    write_json(train_root / "best_summary.json", best_train_result["summary"])
    write_json(refinement_dir / "accepted_refinements.json", {"accepted_refinements": accepted_refinements})
    write_json(calibration_dir / "prompt_nudges.json", {"prompt_nudges": best_policy.get("prompt_nudges", {})})
    write_json(calibration_dir / "frozen_recursion_config.json", {"recursion_config": best_policy.get("recursion_config", {})})

    validation_result = run_judgebench_split(
        examples=val_joined,
        split_name=validation_split_name,
        split_dir=val_root / "final",
        policy=best_policy,
        discovery_model_override=discovery_model_override,
        judge_model_override=judge_model_override,
        use_cache=use_cache,
        max_criteria=max_criteria,
        max_pairs_per_example=max_pairs_per_example,
        covariance_ridge=covariance_ridge,
        max_workers=max_workers,
        resume=resume,
    )
    _assert_split_summary_persisted(validation_result)

    summary = {
        "schema": "compiled_judgebench_run_summary_v1",
        "run_dir": str(run_dir.resolve()),
        "official_dataset_path": str(official_path),
        "train_dataset_path": str(Path(train_dataset_path).resolve()),
        "validation_dataset_path": str(Path(validation_dataset_path).resolve()),
        "protocol_mode": protocol_mode,
        "bootstrap_iterations": bootstrap_iterations,
        "refine_iterations_requested": max(0, refine_iterations),
        "max_workers": max(1, int(max_workers)),
        "accepted_refinement_count": len(accepted_refinements),
        "best_train_iteration_index": best_iteration_index,
        "recursion_config_initial": {
            "max_depth": recursive_config.max_depth,
            "max_recursive_parents_per_pair": recursive_config.max_recursive_parents_per_pair,
            "max_children_per_parent": recursive_config.max_children_per_parent,
            "max_recursive_calls_per_pair": recursive_config.max_recursive_calls_per_pair,
        },
        "recursion_config_frozen": best_policy.get("recursion_config", {}),
        "train_summary": best_train_result["summary"],
        "validation_summary": validation_result["summary"],
        "paths": {
            "dataset_dir": str(dataset_dir.resolve()),
            "frozen_policy_dir": str(frozen_policy_dir.resolve()),
            "refinement_dir": str(refinement_dir.resolve()),
            "calibration_dir": str(calibration_dir.resolve()),
            "train_root": str(train_root.resolve()),
            "validation_root": str(val_root.resolve()),
            "train_best_summary": str((train_root / "best_summary.json").resolve()),
            "validation_summary": str(Path(validation_result["paths"]["summary"]).resolve()),
        },
    }
    write_json(summaries_dir / "summary.json", summary)
    return run_dir, summary


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _policy_core_hash(policy: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in dict(policy).items() if key != "locking_metadata"}
    return stable_hash(to_json_dict(payload))


def _build_mechanism_spec(
    *,
    protocol_mode: str,
    bootstrap_iterations: int,
    recursive_config: RecursiveDiscoveryConfig,
    discovery_model_override: Optional[str],
    judge_model_override: Optional[str],
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    covariance_ridge: float,
    train_reference_answer_access: bool = False,
    blind_scoring_profile: str = _BLIND_SCORING_PROFILE_BASELINE,
    blind_budget_profile: str = _BLIND_BUDGET_PROFILE_FAMILY_V1,
    blind_guidance_profile: str = _BLIND_GUIDANCE_PROFILE_OFF,
    blind_wu_profile: str = _BLIND_WU_PROFILE_RAW,
    retrieval_profile: str = _RETRIEVAL_PROFILE_OFF,
    retrieval_top_k: int = 2,
    blind_discriminator_mode_by_family: Optional[Mapping[str, Any]] = None,
    retrieval_profile_by_family: Optional[Mapping[str, Any]] = None,
    retrieval_top_k_by_family: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "protocol_mode": _normalize_protocol_mode(protocol_mode),
        "bootstrap_iterations": int(bootstrap_iterations),
        "train_reference_answer_access": bool(train_reference_answer_access),
        "blind_parity_bootstrap": not bool(train_reference_answer_access),
        "blind_scoring_profile": _normalize_blind_scoring_profile(blind_scoring_profile),
        "blind_budget_profile": _normalize_blind_budget_profile(blind_budget_profile),
        "blind_guidance_profile": _normalize_blind_guidance_profile(blind_guidance_profile),
        "blind_wu_profile": _normalize_blind_wu_profile(blind_wu_profile),
        "retrieval_profile": _normalize_retrieval_profile(retrieval_profile),
        "retrieval_top_k": max(1, int(retrieval_top_k)),
        "blind_discriminator_mode_by_family": _normalize_blind_discriminator_mode_by_family(
            blind_discriminator_mode_by_family
        ),
        "retrieval_profile_by_family": _normalize_retrieval_profile_by_family(retrieval_profile_by_family),
        "retrieval_top_k_by_family": _normalize_retrieval_top_k_by_family(retrieval_top_k_by_family),
        "recursive_config": {
            "max_depth": int(recursive_config.max_depth),
            "max_recursive_parents_per_pair": int(recursive_config.max_recursive_parents_per_pair),
            "max_children_per_parent": int(recursive_config.max_children_per_parent),
            "max_recursive_calls_per_pair": int(recursive_config.max_recursive_calls_per_pair),
        },
        "discovery_model_override": str(discovery_model_override or "").strip(),
        "judge_model_override": str(judge_model_override or "").strip(),
        "max_criteria": int(max_criteria),
        "max_pairs_per_example": None if max_pairs_per_example is None else int(max_pairs_per_example),
        "covariance_ridge": float(covariance_ridge),
    }


def _mechanism_hash(mechanism_spec: Mapping[str, Any]) -> str:
    return stable_hash(to_json_dict(dict(mechanism_spec)))


def build_balanced_judgebench_folds(
    examples: Sequence[JudgeBenchJoinedExample],
    *,
    fold_count: int,
    shuffle_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[JudgeBenchJoinedExample]] = defaultdict(list)
    for example in sorted(examples, key=lambda row: (row.source_family, row.pair_id)):
        grouped[example.source_family].append(example)
    if not grouped:
        return []
    min_family_count = min(len(rows) for rows in grouped.values())
    effective_fold_count = max(1, min(int(fold_count), min_family_count))
    folds: List[Dict[str, Any]] = [
        {"fold_index": fold_index, "dev_examples": []}
        for fold_index in range(effective_fold_count)
    ]
    rng = random.Random(int(shuffle_seed)) if shuffle_seed is not None else None
    for source_family, rows in sorted(grouped.items()):
        rows = list(rows)
        if rng is not None:
            rng.shuffle(rows)
        for index, row in enumerate(rows):
            folds[index % effective_fold_count]["dev_examples"].append(row)
    all_examples = list(examples)
    for fold in folds:
        dev_ids = {row.pair_id for row in fold["dev_examples"]}
        train_examples = [row for row in all_examples if row.pair_id not in dev_ids]
        fold["train_examples"] = train_examples
        fold["dev_examples"] = sorted(fold["dev_examples"], key=lambda row: (row.source_family, row.pair_id))
        fold["train_examples"] = sorted(train_examples, key=lambda row: (row.source_family, row.pair_id))
        fold["dev_source_family_counts"] = dict(Counter(row.source_family for row in fold["dev_examples"]))
        fold["train_source_family_counts"] = dict(Counter(row.source_family for row in fold["train_examples"]))
    return folds


def _aggregate_split_results(
    *,
    split_name: str,
    examples: Sequence[JudgeBenchJoinedExample],
    results: Sequence[Mapping[str, Any]],
    max_workers: int,
) -> Dict[str, Any]:
    route_decisions: List[Dict[str, Any]] = []
    wu_rows: List[Dict[str, Any]] = []
    uniform_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    analysis_rows: List[Dict[str, Any]] = []
    stats = _make_split_stats(pairs_total=len(examples))
    fold_policy_hashes: List[str] = []
    reference_answer_access_values: set[bool] = set()
    blind_guidance_profile_values: set[str] = set()
    blind_wu_profile_values: set[str] = set()
    retrieval_profile_values: set[str] = set()
    blind_discriminator_mode_by_family_values: set[str] = set()
    retrieval_profile_by_family_values: set[str] = set()
    retrieval_top_k_values: set[int] = set()
    retrieval_top_k_by_family_values: set[str] = set()
    for result in results:
        route_decisions.extend(list(result.get("route_decisions", []) or []))
        wu_rows.extend(list(result.get("wu_rows", []) or []))
        uniform_rows.extend(list(result.get("uniform_rows", []) or []))
        failures.extend(list(result.get("failures", []) or []))
        analysis_rows.extend(list(result.get("analysis_rows", []) or []))
        _merge_split_stats(stats, dict(result.get("summary", {}).get("stats", {}) or {}))
        fold_policy_hash = str((result.get("summary", {}) or {}).get("policy_hash", "")).strip()
        if fold_policy_hash:
            fold_policy_hashes.append(fold_policy_hash)
        summary_reference_answer_access = (result.get("summary", {}) or {}).get("reference_answer_access")
        if summary_reference_answer_access is not None:
            reference_answer_access_values.add(bool(summary_reference_answer_access))
        blind_guidance_profile = str((result.get("summary", {}) or {}).get("blind_guidance_profile", "")).strip()
        if blind_guidance_profile:
            blind_guidance_profile_values.add(blind_guidance_profile)
        blind_wu_profile = str((result.get("summary", {}) or {}).get("blind_wu_profile", "")).strip()
        if blind_wu_profile:
            blind_wu_profile_values.add(blind_wu_profile)
        retrieval_profile = str((result.get("summary", {}) or {}).get("retrieval_profile", "")).strip()
        if retrieval_profile:
            retrieval_profile_values.add(retrieval_profile)
        blind_discriminator_mode_by_family = dict(
            (result.get("summary", {}) or {}).get("blind_discriminator_mode_by_family", {}) or {}
        )
        blind_discriminator_mode_by_family_values.add(
            json.dumps(to_json_dict(blind_discriminator_mode_by_family), sort_keys=True)
        )
        retrieval_profile_by_family = dict((result.get("summary", {}) or {}).get("retrieval_profile_by_family", {}) or {})
        retrieval_profile_by_family_values.add(json.dumps(to_json_dict(retrieval_profile_by_family), sort_keys=True))
        retrieval_top_k_values.add(int((result.get("summary", {}) or {}).get("retrieval_top_k", 2) or 2))
        retrieval_top_k_by_family = dict((result.get("summary", {}) or {}).get("retrieval_top_k_by_family", {}) or {})
        retrieval_top_k_by_family_values.add(json.dumps(to_json_dict(retrieval_top_k_by_family), sort_keys=True))
    route_counts = Counter(str(item.get("task_profile_id", "")) for item in route_decisions)
    aggregated_reference_answer_access: Optional[bool] = None
    if len(reference_answer_access_values) == 1:
        aggregated_reference_answer_access = next(iter(reference_answer_access_values))
    aggregated_blind_guidance_profile = ""
    if len(blind_guidance_profile_values) == 1:
        aggregated_blind_guidance_profile = next(iter(blind_guidance_profile_values))
    aggregated_blind_wu_profile = ""
    if len(blind_wu_profile_values) == 1:
        aggregated_blind_wu_profile = next(iter(blind_wu_profile_values))
    aggregated_retrieval_profile = ""
    if len(retrieval_profile_values) == 1:
        aggregated_retrieval_profile = next(iter(retrieval_profile_values))
    aggregated_blind_discriminator_mode_by_family: Dict[str, Any] = {}
    if len(blind_discriminator_mode_by_family_values) == 1:
        aggregated_blind_discriminator_mode_by_family = dict(
            json.loads(next(iter(blind_discriminator_mode_by_family_values)) or "{}")
        )
    aggregated_retrieval_profile_by_family: Dict[str, Any] = {}
    if len(retrieval_profile_by_family_values) == 1:
        aggregated_retrieval_profile_by_family = dict(json.loads(next(iter(retrieval_profile_by_family_values)) or "{}"))
    aggregated_retrieval_top_k = 2
    if len(retrieval_top_k_values) == 1:
        aggregated_retrieval_top_k = next(iter(retrieval_top_k_values))
    aggregated_retrieval_top_k_by_family: Dict[str, Any] = {}
    if len(retrieval_top_k_by_family_values) == 1:
        aggregated_retrieval_top_k_by_family = dict(json.loads(next(iter(retrieval_top_k_by_family_values)) or "{}"))
    calibration_metrics = build_judgebench_calibration_metrics(
        split_result={
            "summary": {"pair_count": len(examples)},
            "analysis_rows": analysis_rows,
            "failures": failures,
        }
    )
    summary = {
        "schema": "compiled_judgebench_split_summary_v1",
        "split_name": split_name,
        "aggregation_kind": "oof_per_fold_policies",
        "policy_hash": stable_hash(sorted(set(fold_policy_hashes))),
        "fold_policy_hashes": sorted(set(fold_policy_hashes)),
        "reference_answer_access": aggregated_reference_answer_access,
        "blind_guidance_profile": aggregated_blind_guidance_profile or _BLIND_GUIDANCE_PROFILE_OFF,
        "blind_wu_profile": aggregated_blind_wu_profile or _BLIND_WU_PROFILE_RAW,
        "retrieval_profile": aggregated_retrieval_profile or _RETRIEVAL_PROFILE_OFF,
        "blind_discriminator_mode_by_family": aggregated_blind_discriminator_mode_by_family,
        "retrieval_profile_by_family": aggregated_retrieval_profile_by_family,
        "retrieval_top_k": max(1, int(aggregated_retrieval_top_k)),
        "retrieval_top_k_by_family": aggregated_retrieval_top_k_by_family,
        "max_workers": max(1, int(max_workers)),
        "pair_count": len(examples),
        "wu_metrics": metric_breakdown(wu_rows),
        "uniform_metrics": metric_breakdown(uniform_rows),
        "decision_counts": dict(Counter(str(row.get("decision_original", "")) for row in wu_rows)),
        "source_family_counts": dict(Counter(example.source_family for example in examples)),
        "task_profile_counts": dict(sorted(route_counts.items())),
        "avg_rubric_count": (
            sum(float(row.get("rubric_count", 0) or 0) for row in wu_rows) / max(1, len(wu_rows))
        ),
        "failure_count": len(failures),
        "calibration_metrics": calibration_metrics,
        "stats": stats,
    }
    return {
        "summary": summary,
        "wu_rows": wu_rows,
        "uniform_rows": uniform_rows,
        "failures": failures,
        "analysis_rows": analysis_rows,
        "route_decisions": route_decisions,
    }


def _slice_split_result(
    *,
    split_name: str,
    examples: Sequence[JudgeBenchJoinedExample],
    split_result: Mapping[str, Any],
) -> Dict[str, Any]:
    pair_ids = {str(example.pair_id).strip() for example in examples}
    base_summary = dict(split_result.get("summary", {}) or {})
    route_decisions = [
        dict(item)
        for item in list(split_result.get("route_decisions", []) or [])
        if str(item.get("pair_id", "")).strip() in pair_ids
    ]
    wu_rows = [
        dict(item)
        for item in list(split_result.get("wu_rows", []) or [])
        if str(item.get("pair_id", "")).strip() in pair_ids
    ]
    uniform_rows = [
        dict(item)
        for item in list(split_result.get("uniform_rows", []) or [])
        if str(item.get("pair_id", "")).strip() in pair_ids
    ]
    failures = [
        dict(item)
        for item in list(split_result.get("failures", []) or [])
        if str(item.get("pair_id", "")).strip() in pair_ids
    ]
    analysis_rows = [
        dict(item)
        for item in list(split_result.get("analysis_rows", []) or [])
        if str(item.get("pair_id", "")).strip() in pair_ids
    ]
    route_counts = Counter(str(item.get("task_profile_id", "")) for item in route_decisions)
    calibration_metrics = build_judgebench_calibration_metrics(
        split_result={
            "summary": {"pair_count": len(examples)},
            "analysis_rows": analysis_rows,
            "failures": failures,
        }
    )
    summary = {
        "schema": "compiled_judgebench_split_summary_v1",
        "split_name": split_name,
        "aggregation_kind": "derived_existing_split_subset",
        "policy_hash": str(base_summary.get("policy_hash", "")).strip(),
        "reference_answer_access": base_summary.get("reference_answer_access"),
        "blind_guidance_profile": (
            str(base_summary.get("blind_guidance_profile", "")).strip() or _BLIND_GUIDANCE_PROFILE_OFF
        ),
        "blind_wu_profile": str(base_summary.get("blind_wu_profile", "")).strip() or _BLIND_WU_PROFILE_RAW,
        "retrieval_profile": str(base_summary.get("retrieval_profile", "")).strip() or _RETRIEVAL_PROFILE_OFF,
        "blind_discriminator_mode_by_family": _normalize_blind_discriminator_mode_by_family(
            base_summary.get("blind_discriminator_mode_by_family")
        ),
        "retrieval_profile_by_family": _normalize_retrieval_profile_by_family(
            base_summary.get("retrieval_profile_by_family")
        ),
        "retrieval_top_k": max(1, int(base_summary.get("retrieval_top_k", 2) or 2)),
        "retrieval_top_k_by_family": _normalize_retrieval_top_k_by_family(
            base_summary.get("retrieval_top_k_by_family")
        ),
        "max_workers": max(1, int(base_summary.get("max_workers", 1) or 1)),
        "pair_count": len(examples),
        "wu_metrics": metric_breakdown(wu_rows),
        "uniform_metrics": metric_breakdown(uniform_rows),
        "decision_counts": dict(Counter(str(row.get("decision_original", "")) for row in wu_rows)),
        "source_family_counts": dict(Counter(example.source_family for example in examples)),
        "task_profile_counts": dict(sorted(route_counts.items())),
        "avg_rubric_count": (
            sum(float(row.get("rubric_count", 0) or 0) for row in wu_rows) / max(1, len(wu_rows))
        ),
        "failure_count": len(failures),
        "calibration_metrics": calibration_metrics,
        "stats": _make_split_stats(pairs_total=len(examples)),
        "derived_from_split_name": str(base_summary.get("split_name", "")).strip(),
    }
    return {
        "summary": summary,
        "wu_rows": wu_rows,
        "uniform_rows": uniform_rows,
        "failures": failures,
        "analysis_rows": analysis_rows,
        "route_decisions": route_decisions,
    }


def _rubric_count_bucket(count: int) -> str:
    if count <= 5:
        return "0_5"
    if count <= 10:
        return "6_10"
    if count <= 15:
        return "11_15"
    return "16_plus"


def build_judgebench_failure_analysis(
    *,
    split_result: Mapping[str, Any],
) -> Dict[str, Any]:
    failures = list(split_result.get("failures", []) or [])
    analysis_rows = list(split_result.get("analysis_rows", []) or [])
    failure_ids = {str(row.get("pair_id", "")).strip() for row in failures}
    family_failures: Dict[str, Counter[str]] = defaultdict(Counter)
    profile_failures: Counter[str] = Counter()
    mutation_coverage: Counter[str] = Counter()
    failure_mutation_coverage: Counter[str] = Counter()
    exact_answer_failures = 0
    code_failures = 0
    tie_failures = 0
    broad_failures = 0
    rubric_count_buckets: Counter[str] = Counter()
    for failure in failures:
        source_family = str(failure.get("source_family", "")).strip() or "unknown"
        decision = str(failure.get("decision", "")).strip() or "unknown"
        family_failures[source_family][decision] += 1
        profile_failures[str(failure.get("routing_task_profile_id", "")).strip() or "unknown"] += 1
        exact_answer_failures += int(bool(failure.get("exact_answer_task", False)))
        code_failures += int(bool(failure.get("code_task", False)))
        tie_failures += int(decision == "A=B")
        broad_failures += int(int(failure.get("broad_rubric_count", 0) or 0) > 0)
        rubric_count_buckets[_rubric_count_bucket(int(failure.get("rubric_count", 0) or 0))] += 1
    for row in analysis_rows:
        weak_labels = [str(item).strip() for item in row.get("weak_source_labels", []) or [] if str(item).strip()]
        for weak_label in weak_labels:
            mutation_coverage[weak_label] += 1
            if str(row.get("pair_id", "")).strip() in failure_ids:
                failure_mutation_coverage[weak_label] += 1
    return {
        "schema": "compiled_judgebench_failure_analysis_v1",
        "pair_count": int((split_result.get("summary", {}) or {}).get("pair_count", 0) or 0),
        "failure_count": len(failures),
        "failure_rate": len(failures) / max(1, int((split_result.get("summary", {}) or {}).get("pair_count", 0) or 0)),
        "tie_failures": tie_failures,
        "exact_answer_failures": exact_answer_failures,
        "code_failures": code_failures,
        "broad_failures": broad_failures,
        "rubric_count_buckets": dict(sorted(rubric_count_buckets.items())),
        "family_failure_clusters": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(family_failures.items())
        },
        "routing_profile_failure_clusters": dict(sorted(profile_failures.items())),
        "mutation_coverage": {
            "overall": dict(sorted(mutation_coverage.items())),
            "failed_examples": dict(sorted(failure_mutation_coverage.items())),
        },
    }


def build_judgebench_calibration_metrics(
    *,
    split_result: Mapping[str, Any],
) -> Dict[str, Any]:
    analysis_rows = list(split_result.get("analysis_rows", []) or [])
    pair_count = max(1, int((split_result.get("summary", {}) or {}).get("pair_count", 0) or len(analysis_rows) or 1))
    family_totals: Counter[str] = Counter()
    verifier_available_by_family: Counter[str] = Counter()
    verifier_triggered_by_family: Counter[str] = Counter()
    exact_task_total = 0
    exact_parser_success = 0
    low_confidence_total = 0
    low_confidence_correct = 0
    discriminator_used = 0
    discriminator_order_disagreement = 0
    decision_policy_counts: Counter[str] = Counter()
    tie_break_reason_counts: Counter[str] = Counter()
    for row in analysis_rows:
        family = str(row.get("source_family", "")).strip() or "unknown"
        family_totals[family] += 1
        if bool(row.get("verifier_available")):
            verifier_available_by_family[family] += 1
        if bool(row.get("verifier_triggered")):
            verifier_triggered_by_family[family] += 1
        if bool(row.get("exact_answer_task")):
            exact_task_total += 1
            exact_parser_success += int(
                bool(row.get("exact_answer_parser_success_A")) and bool(row.get("exact_answer_parser_success_B"))
            )
        if bool(row.get("discriminator_used")):
            discriminator_used += 1
            discriminator_order_disagreement += int(not bool(row.get("discriminator_order_consistent")))
        policy_name = str(row.get("decision_policy", "")).strip()
        if policy_name:
            decision_policy_counts[policy_name] += 1
        tie_reason = str(row.get("tie_break_reason", "")).strip()
        if tie_reason:
            tie_break_reason_counts[tie_reason] += 1
        verifier_confidence = str(row.get("verifier_confidence", "")).strip().lower()
        discriminator_confidence = str(row.get("discriminator_confidence", "")).strip().lower()
        pair_margin = float(row.get("pair_margin", 0.0) or 0.0)
        low_confidence = (
            verifier_confidence in {"", "low"}
            or (bool(row.get("discriminator_used")) and discriminator_confidence in {"", "low"})
            or pair_margin <= 0.006
        )
        if low_confidence:
            low_confidence_total += 1
            low_confidence_correct += int(str(row.get("decision", "")).strip() == str(row.get("label", "")).strip())
    verifier_coverage_by_family = {
        family: round(verifier_available_by_family[family] / max(1, count), 6)
        for family, count in sorted(family_totals.items())
    }
    verifier_trigger_rate_by_family = {
        family: round(verifier_triggered_by_family[family] / max(1, count), 6)
        for family, count in sorted(family_totals.items())
    }
    focus_families = ("mmlu-pro", "livebench-reasoning")
    focus_counts = [family_totals[family] for family in focus_families if family_totals[family] > 0]
    focus_available = [verifier_available_by_family[family] for family in focus_families if family_totals[family] > 0]
    focus_triggered = [verifier_triggered_by_family[family] for family in focus_families if family_totals[family] > 0]
    focus_verifier_coverage = (
        round(sum(focus_available) / max(1, sum(focus_counts)), 6)
        if focus_counts
        else 0.0
    )
    focus_verifier_trigger_rate = (
        round(sum(focus_triggered) / max(1, sum(focus_counts)), 6)
        if focus_counts
        else 0.0
    )
    return {
        "schema": "compiled_judgebench_calibration_metrics_v1",
        "pair_count": pair_count,
        "verifier_coverage_rate": round(sum(verifier_available_by_family.values()) / pair_count, 6),
        "focus_verifier_coverage_rate": focus_verifier_coverage,
        "verifier_trigger_rate": round(sum(verifier_triggered_by_family.values()) / pair_count, 6),
        "focus_verifier_trigger_rate": focus_verifier_trigger_rate,
        "verifier_coverage_by_family": verifier_coverage_by_family,
        "verifier_trigger_rate_by_family": verifier_trigger_rate_by_family,
        "exact_answer_parser_success_rate": round(exact_parser_success / max(1, exact_task_total), 6)
        if exact_task_total
        else 0.0,
        "exact_answer_task_count": exact_task_total,
        "low_confidence_bucket_accuracy": round(low_confidence_correct / max(1, low_confidence_total), 6)
        if low_confidence_total
        else 1.0,
        "low_confidence_bucket_count": low_confidence_total,
        "discriminator_usage_rate": round(discriminator_used / pair_count, 6),
        "discriminator_order_disagreement_rate": round(discriminator_order_disagreement / max(1, discriminator_used), 6)
        if discriminator_used
        else 0.0,
        "decision_policy_counts": dict(sorted(decision_policy_counts.items())),
        "tie_break_reason_counts": dict(sorted(tie_break_reason_counts.items())),
    }


def _write_split_failure_bundle(
    *,
    summaries_dir: Path,
    stem: str,
    split_result: Mapping[str, Any],
) -> Dict[str, Any]:
    failure_analysis = build_judgebench_failure_analysis(split_result=split_result)
    failure_analysis_path = summaries_dir / f"{stem}_failure_analysis.json"
    failures_path = summaries_dir / f"{stem}_failures.json"
    analysis_rows_path = summaries_dir / f"{stem}_analysis_rows.json"
    write_json(failure_analysis_path, failure_analysis)
    write_json(failures_path, list(split_result.get("failures", []) or []))
    write_json(analysis_rows_path, list(split_result.get("analysis_rows", []) or []))
    return {
        "failure_analysis": failure_analysis,
        "paths": {
            "failure_analysis": str(failure_analysis_path.resolve()),
            "failures": str(failures_path.resolve()),
            "analysis_rows": str(analysis_rows_path.resolve()),
        },
    }


def _canonical_final_eval_subset_specs() -> List[Tuple[str, Path]]:
    root = _compiled_repo_root()
    return [
        ("judgebench_80_human", root / "data" / "judgebench_80_human.json"),
        ("judgebench_270_generated", root / "data" / "judgebench_270_generated.json"),
    ]


def _build_locked_policy_alignment_summary(
    *,
    locked_policy: Mapping[str, Any],
    oof_result: Mapping[str, Any],
    train_fit_result: Optional[Mapping[str, Any]],
    fold_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    oof_summary = dict(oof_result.get("summary", {}) or {})
    train_fit_summary = dict((train_fit_result or {}).get("summary", {}) or {})
    fold_policy_hashes = sorted(
        {
            str((result.get("summary", {}) or {}).get("policy_hash", "")).strip()
            for result in fold_results
            if str((result.get("summary", {}) or {}).get("policy_hash", "")).strip()
        }
    )
    family_wu_gap: Dict[str, float] = {}
    if train_fit_summary:
        oof_family_wu = dict((oof_summary.get("wu_metrics", {}) or {}))
        train_fit_family_wu = dict((train_fit_summary.get("wu_metrics", {}) or {}))
        for family_name in sorted(set(oof_family_wu) & set(train_fit_family_wu)):
            if family_name == "overall":
                continue
            family_wu_gap[family_name] = round(
                    _coerce_float(train_fit_family_wu.get(family_name)) - _coerce_float(oof_family_wu.get(family_name)),
                6,
            )
    train_fit_overall_wu = (
        _coerce_float((train_fit_summary.get("wu_metrics", {}) or {}).get("overall")) if train_fit_summary else None
    )
    train_fit_overall_uniform = (
        _coerce_float((train_fit_summary.get("uniform_metrics", {}) or {}).get("overall"))
        if train_fit_summary
        else None
    )
    oof_overall_wu = _coerce_float((oof_summary.get("wu_metrics", {}) or {}).get("overall"))
    oof_overall_uniform = _coerce_float((oof_summary.get("uniform_metrics", {}) or {}).get("overall"))
    return {
        "schema": "compiled_judgebench_locked_policy_alignment_v1",
        "export_strategy": "full_train_rebuild",
        "locked_policy_hash": _policy_core_hash(dict(locked_policy)),
        "fold_policy_hashes": fold_policy_hashes,
        "oof_overall_wu": oof_overall_wu,
        "oof_overall_uniform": oof_overall_uniform,
        "train_fit_available": bool(train_fit_summary),
        "train_fit_overall_wu": train_fit_overall_wu,
        "train_fit_overall_uniform": train_fit_overall_uniform,
        "locked_train_fit_minus_oof_wu": (
            round(train_fit_overall_wu - oof_overall_wu, 6) if train_fit_overall_wu is not None else None
        ),
        "locked_train_fit_minus_oof_uniform": (
            round(train_fit_overall_uniform - oof_overall_uniform, 6)
            if train_fit_overall_uniform is not None
            else None
        ),
        "family_wu_gap": family_wu_gap,
    }


def _clear_train_only_bookkeeping(run_dir: Path) -> None:
    for rel_path in (
        Path("summaries/summary.json"),
        Path("summaries/oof_summary.json"),
        Path("summaries/oof_failure_analysis.json"),
        Path("summaries/oof_failures.json"),
        Path("summaries/oof_analysis_rows.json"),
        Path("summaries/locked_policy_alignment.json"),
        Path("summaries/train_fit_summary.json"),
        Path("summaries/train_fit_failure_analysis.json"),
        Path("summaries/train_fit_failures.json"),
        Path("summaries/train_fit_analysis_rows.json"),
        Path("summaries/fold_manifest.json"),
        Path("frozen_policy/locked_policy.json"),
        Path("train_fit/routing/decisions.json"),
        Path("train_fit/reports/rrd_wu_predictions.jsonl"),
        Path("train_fit/reports/rrd_uniform_predictions.jsonl"),
        Path("train_fit/summaries/summary.json"),
    ):
        _unlink_if_exists(run_dir / rel_path)


def _train_only_parallelism_plan(*, max_workers: int, fold_count: int) -> Dict[str, int]:
    total_workers = max(1, int(max_workers))
    if fold_count <= 1:
        return {
            "total_max_workers": total_workers,
            "fold_processes": 1,
            "split_max_workers": total_workers,
        }
    fold_processes = max(1, min(int(fold_count), total_workers))
    split_max_workers = max(1, total_workers // fold_processes)
    return {
        "total_max_workers": total_workers,
        "fold_processes": fold_processes,
        "split_max_workers": split_max_workers,
    }


def _run_train_only_fold_job(
    *,
    fold_index: int,
    train_examples: Sequence[JudgeBenchJoinedExample],
    dev_examples: Sequence[JudgeBenchJoinedExample],
    fold_dir: Path,
    train_split_name: str,
    bootstrap_iterations: int,
    recursive_config: RecursiveDiscoveryConfig,
    protocol_mode: str,
    discovery_model_override: Optional[str],
    judge_model_override: Optional[str],
    use_cache: bool,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    covariance_ridge: float,
    split_max_workers: int,
    resume: bool,
    train_reference_answer_access: bool,
    oof_reference_answer_access: bool,
    blind_scoring_profile: str,
    blind_budget_profile: str,
    blind_guidance_profile: str,
    blind_wu_profile: str,
    retrieval_profile: str,
    retrieval_top_k: int,
    blind_discriminator_mode_by_family: Optional[Mapping[str, Any]],
    retrieval_profile_by_family: Optional[Mapping[str, Any]],
    retrieval_top_k_by_family: Optional[Mapping[str, Any]],
    v2_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    fold_dir = Path(fold_dir)
    v2 = dict(v2_config or {})
    fold_policy = build_initial_frozen_policy(
        train_examples=train_examples,
        bootstrap_iterations=bootstrap_iterations,
        recursive_config=recursive_config,
        protocol_mode=protocol_mode,
        reference_answer_access=train_reference_answer_access,
        blind_scoring_profile=blind_scoring_profile,
        blind_budget_profile=blind_budget_profile,
        blind_guidance_profile=blind_guidance_profile,
        blind_wu_profile=blind_wu_profile,
        retrieval_profile=retrieval_profile,
        retrieval_top_k=retrieval_top_k,
        blind_discriminator_mode_by_family=blind_discriminator_mode_by_family,
        retrieval_profile_by_family=retrieval_profile_by_family,
        retrieval_top_k_by_family=retrieval_top_k_by_family,
        self_consistency_n=int(v2.get("self_consistency_n", _DEFAULT_SELF_CONSISTENCY_N) or _DEFAULT_SELF_CONSISTENCY_N),
        self_consistency_temperature=float(
            v2.get("self_consistency_temperature", _DEFAULT_SELF_CONSISTENCY_TEMPERATURE)
            or _DEFAULT_SELF_CONSISTENCY_TEMPERATURE
        ),
        v2_wide_discriminator_gate=bool(v2.get("v2_wide_discriminator_gate", False)),
        holistic_judge_enabled=bool(v2.get("holistic_judge_enabled", False)),
        library_retrieval_top_k=int(v2.get("library_retrieval_top_k", 0) or 0),
        rubric_library_path=v2.get("rubric_library_path") or None,
        enable_rrd_filters=bool(v2.get("enable_rrd_filters", False)),
        rrd_redundancy_threshold=float(v2.get("rrd_redundancy_threshold", 0.9) or 0.9),
        library_retrieval_top_k_by_family=v2.get("library_retrieval_top_k_by_family") or {},
        family_strict_library_mode=bool(v2.get("family_strict_library_mode", False)),
        math_independent_solver_enabled=bool(v2.get("math_independent_solver_enabled", False)),
        math_solver_samples=int(v2.get("math_solver_samples", 1) or 1),
        math_solver_temperature=float(v2.get("math_solver_temperature", 0.5) or 0.5),
        code_execution_verifier_enabled=bool(v2.get("code_execution_verifier_enabled", False)),
        code_execution_timeout_s=float(v2.get("code_execution_timeout_s", 10.0) or 10.0),
        code_execution_min_margin=float(v2.get("code_execution_min_margin", 0.34) or 0.34),
        math_solver_use_sympy=bool(v2.get("math_solver_use_sympy", False)),
        math_solver_model=str(v2.get("math_solver_model", "") or ""),
        few_shot_train_enabled=bool(v2.get("few_shot_train_enabled", False)),
        few_shot_top_k=int(v2.get("few_shot_top_k", 3) or 3),
        few_shot_train_dataset_path=str(v2.get("few_shot_train_dataset_path", "") or ""),
        few_shot_official_dataset_path=str(v2.get("few_shot_official_dataset_path", "") or ""),
        rubric_satisfaction_samples=int(v2.get("rubric_satisfaction_samples", 1) or 1),
        rubric_satisfaction_temperature=float(v2.get("rubric_satisfaction_temperature", 0.4) or 0.4),
        discriminator_self_critique_enabled=bool(v2.get("discriminator_self_critique_enabled", False)),
        mmlu_independent_answerer_enabled=bool(v2.get("mmlu_independent_answerer_enabled", False)),
        mmlu_answerer_samples=int(v2.get("mmlu_answerer_samples", 1) or 1),
        mmlu_answerer_temperature=float(v2.get("mmlu_answerer_temperature", 0.5) or 0.5),
        mmlu_answerer_model=str(v2.get("mmlu_answerer_model", "") or ""),
        mmlu_answerer_secondary_model=str(v2.get("mmlu_answerer_secondary_model", "") or ""),
        mmlu_answerer_secondary_samples=int(v2.get("mmlu_answerer_secondary_samples", 1) or 1),
        mmlu_answerer_secondary_temperature=float(v2.get("mmlu_answerer_secondary_temperature", 0.0) or 0.0),
        reasoning_independent_solver_enabled=bool(v2.get("reasoning_independent_solver_enabled", False)),
        reasoning_solver_samples=int(v2.get("reasoning_solver_samples", 1) or 1),
        reasoning_solver_temperature=float(v2.get("reasoning_solver_temperature", 0.5) or 0.5),
        reasoning_solver_model=str(v2.get("reasoning_solver_model", "") or ""),
    )
    write_json(fold_dir / "frozen_policy.json", fold_policy)
    fold_result = run_judgebench_split(
        examples=dev_examples,
        split_name=f"{train_split_name}_fold_{int(fold_index):02d}_dev",
        split_dir=fold_dir / "dev",
        policy=fold_policy,
        discovery_model_override=discovery_model_override,
        judge_model_override=judge_model_override,
        use_cache=use_cache,
        max_criteria=max_criteria,
        max_pairs_per_example=max_pairs_per_example,
        covariance_ridge=covariance_ridge,
        max_workers=split_max_workers,
        resume=resume,
        reference_answer_access=oof_reference_answer_access,
        retrieval_examples=train_examples if _policy_has_any_retrieval(fold_policy) else None,
    )
    _assert_split_summary_persisted(fold_result)
    return {
        "fold_index": int(fold_index),
        "fold_result": fold_result,
    }


def run_judgebench_train_only_development(
    *,
    train_dataset_path: Path,
    train_split_name: str,
    run_dir: Path,
    official_dataset_path: Optional[Path],
    discovery_model_override: Optional[str],
    judge_model_override: Optional[str],
    use_cache: bool,
    max_criteria: int,
    max_pairs_per_example: Optional[int],
    bootstrap_iterations: int,
    recursive_config: RecursiveDiscoveryConfig,
    covariance_ridge: float,
    max_workers: int,
    fold_count: int,
    fold_shuffle_seed: Optional[int],
    protocol_mode: str,
    resume: bool,
    train_reference_answer_access: bool = False,
    oof_reference_answer_access: bool = False,
    write_train_fit: bool = False,
    train_fit_reference_answer_access: bool = False,
    blind_scoring_profile: str = _BLIND_SCORING_PROFILE_BASELINE,
    blind_budget_profile: str = _BLIND_BUDGET_PROFILE_FAMILY_V1,
    blind_guidance_profile: str = _BLIND_GUIDANCE_PROFILE_OFF,
    blind_wu_profile: str = _BLIND_WU_PROFILE_RAW,
    retrieval_profile: str = _RETRIEVAL_PROFILE_OFF,
    retrieval_top_k: int = 2,
    blind_discriminator_mode_by_family: Optional[Mapping[str, Any]] = None,
    retrieval_profile_by_family: Optional[Mapping[str, Any]] = None,
    retrieval_top_k_by_family: Optional[Mapping[str, Any]] = None,
    v2_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    run_dir = Path(run_dir)
    train_split_name = _safe_slug(train_split_name)
    protocol_mode = _normalize_protocol_mode(protocol_mode)
    blind_scoring_profile = _normalize_blind_scoring_profile(blind_scoring_profile)
    blind_budget_profile = _normalize_blind_budget_profile(blind_budget_profile)
    blind_guidance_profile = _normalize_blind_guidance_profile(blind_guidance_profile)
    blind_wu_profile = _normalize_blind_wu_profile(blind_wu_profile)
    retrieval_profile = _normalize_retrieval_profile(retrieval_profile)
    blind_discriminator_mode_by_family = _normalize_blind_discriminator_mode_by_family(
        blind_discriminator_mode_by_family
    )
    retrieval_profile_by_family = _normalize_retrieval_profile_by_family(retrieval_profile_by_family)
    retrieval_top_k_by_family = _normalize_retrieval_top_k_by_family(retrieval_top_k_by_family)
    dataset_dir = run_dir / "dataset"
    folds_root = run_dir / "folds"
    frozen_policy_dir = run_dir / "frozen_policy"
    summaries_dir = run_dir / "summaries"
    for path in (dataset_dir, folds_root, frozen_policy_dir, summaries_dir):
        path.mkdir(parents=True, exist_ok=True)
    _clear_train_only_bookkeeping(run_dir)

    official_path = ensure_official_judgebench_dataset(official_dataset_path, dataset_dir)
    official_pairs = load_official_judgebench_pairs(official_path)
    train_local = load_local_judgebench_subset(train_dataset_path)
    train_joined = join_local_subset_to_official_pairs(
        local_rows=train_local,
        official_pairs=official_pairs,
        split_name=train_split_name,
    )
    write_json(dataset_dir / f"joined_{train_split_name}.json", train_joined)
    write_json(
        dataset_dir / "dataset_manifest.json",
        {
            "official_dataset_path": str(official_path),
            "train_dataset_path": str(Path(train_dataset_path).resolve()),
            "train_split_name": train_split_name,
            "train_pair_count": len(train_joined),
            "train_reference_answer_access": bool(train_reference_answer_access),
            "blind_parity_bootstrap": not bool(train_reference_answer_access),
            "oof_reference_answer_access": bool(oof_reference_answer_access),
            "write_train_fit": bool(write_train_fit),
            "train_fit_reference_answer_access": bool(train_fit_reference_answer_access),
            "fold_shuffle_seed": int(fold_shuffle_seed) if fold_shuffle_seed is not None else None,
            "blind_scoring_profile": blind_scoring_profile,
            "blind_budget_profile": blind_budget_profile,
            "blind_guidance_profile": blind_guidance_profile,
            "blind_wu_profile": blind_wu_profile,
            "retrieval_profile": retrieval_profile,
            "retrieval_top_k": max(1, int(retrieval_top_k)),
            "blind_discriminator_mode_by_family": blind_discriminator_mode_by_family,
            "retrieval_profile_by_family": retrieval_profile_by_family,
            "retrieval_top_k_by_family": retrieval_top_k_by_family,
        },
    )

    folds = build_balanced_judgebench_folds(
        train_joined,
        fold_count=fold_count,
        shuffle_seed=fold_shuffle_seed,
    )
    parallelism = _train_only_parallelism_plan(max_workers=max_workers, fold_count=len(folds))
    write_json(
        summaries_dir / "fold_manifest.json",
        {
            "fold_count": len(folds),
            "parallelism": parallelism,
            "train_reference_answer_access": bool(train_reference_answer_access),
            "blind_parity_bootstrap": not bool(train_reference_answer_access),
            "oof_reference_answer_access": bool(oof_reference_answer_access),
            "write_train_fit": bool(write_train_fit),
            "train_fit_reference_answer_access": bool(train_fit_reference_answer_access),
            "fold_shuffle_seed": int(fold_shuffle_seed) if fold_shuffle_seed is not None else None,
            "blind_scoring_profile": blind_scoring_profile,
            "blind_budget_profile": blind_budget_profile,
            "blind_guidance_profile": blind_guidance_profile,
            "blind_wu_profile": blind_wu_profile,
            "retrieval_profile": retrieval_profile,
            "retrieval_top_k": max(1, int(retrieval_top_k)),
            "blind_discriminator_mode_by_family": blind_discriminator_mode_by_family,
            "retrieval_profile_by_family": retrieval_profile_by_family,
            "retrieval_top_k_by_family": retrieval_top_k_by_family,
            "folds": [
                {
                    "fold_index": fold["fold_index"],
                    "dev_pair_ids": [row.pair_id for row in fold["dev_examples"]],
                    "train_pair_ids": [row.pair_id for row in fold["train_examples"]],
                    "dev_source_family_counts": dict(fold["dev_source_family_counts"]),
                    "train_source_family_counts": dict(fold["train_source_family_counts"]),
                }
                for fold in folds
            ],
        },
    )

    fold_results_by_index: Dict[int, Dict[str, Any]] = {}
    normalized_v2_config = _normalize_v2_config(v2_config)
    if parallelism["fold_processes"] <= 1 or len(folds) <= 1:
        for fold in folds:
            job_result = _run_train_only_fold_job(
                fold_index=int(fold["fold_index"]),
                train_examples=fold["train_examples"],
                dev_examples=fold["dev_examples"],
                fold_dir=folds_root / f"fold_{int(fold['fold_index']):02d}",
                train_split_name=train_split_name,
                bootstrap_iterations=bootstrap_iterations,
                recursive_config=recursive_config,
                protocol_mode=protocol_mode,
                discovery_model_override=discovery_model_override,
                judge_model_override=judge_model_override,
                use_cache=use_cache,
                max_criteria=max_criteria,
                max_pairs_per_example=max_pairs_per_example,
                covariance_ridge=covariance_ridge,
                split_max_workers=parallelism["split_max_workers"],
                resume=resume,
                train_reference_answer_access=train_reference_answer_access,
                oof_reference_answer_access=oof_reference_answer_access,
                blind_scoring_profile=blind_scoring_profile,
                blind_budget_profile=blind_budget_profile,
                blind_guidance_profile=blind_guidance_profile,
                blind_wu_profile=blind_wu_profile,
                retrieval_profile=retrieval_profile,
                retrieval_top_k=max(1, int(retrieval_top_k)),
                blind_discriminator_mode_by_family=blind_discriminator_mode_by_family,
                retrieval_profile_by_family=retrieval_profile_by_family,
                retrieval_top_k_by_family=retrieval_top_k_by_family,
                v2_config=normalized_v2_config,
            )
            fold_results_by_index[int(job_result["fold_index"])] = dict(job_result["fold_result"])
    else:
        with ProcessPoolExecutor(max_workers=parallelism["fold_processes"]) as executor:
            futures = [
                executor.submit(
                    _run_train_only_fold_job,
                    fold_index=int(fold["fold_index"]),
                    train_examples=fold["train_examples"],
                    dev_examples=fold["dev_examples"],
                    fold_dir=folds_root / f"fold_{int(fold['fold_index']):02d}",
                    train_split_name=train_split_name,
                    bootstrap_iterations=bootstrap_iterations,
                    recursive_config=recursive_config,
                    protocol_mode=protocol_mode,
                    discovery_model_override=discovery_model_override,
                    judge_model_override=judge_model_override,
                    use_cache=use_cache,
                    max_criteria=max_criteria,
                    max_pairs_per_example=max_pairs_per_example,
                    covariance_ridge=covariance_ridge,
                    split_max_workers=parallelism["split_max_workers"],
                    resume=resume,
                    train_reference_answer_access=train_reference_answer_access,
                    oof_reference_answer_access=oof_reference_answer_access,
                    blind_scoring_profile=blind_scoring_profile,
                    blind_budget_profile=blind_budget_profile,
                    blind_guidance_profile=blind_guidance_profile,
                    blind_wu_profile=blind_wu_profile,
                    retrieval_profile=retrieval_profile,
                    retrieval_top_k=max(1, int(retrieval_top_k)),
                    blind_discriminator_mode_by_family=blind_discriminator_mode_by_family,
                    retrieval_profile_by_family=retrieval_profile_by_family,
                    retrieval_top_k_by_family=retrieval_top_k_by_family,
                    v2_config=normalized_v2_config,
                )
                for fold in folds
            ]
            for future in futures:
                job_result = future.result()
                fold_results_by_index[int(job_result["fold_index"])] = dict(job_result["fold_result"])
    fold_results = [fold_results_by_index[int(fold["fold_index"])] for fold in folds]

    oof_result = _aggregate_split_results(
        split_name=f"{train_split_name}_oof",
        examples=train_joined,
        results=fold_results,
        max_workers=max_workers,
    )
    failure_analysis = build_judgebench_failure_analysis(split_result=oof_result)
    oof_summary_payload = dict(oof_result["summary"])
    external_slice_summary = _score_external_slices_for_run(v2_config=normalized_v2_config)
    if external_slice_summary:
        oof_summary_payload["external_slice_summary"] = external_slice_summary
    write_json(summaries_dir / "oof_summary.json", oof_summary_payload)
    write_json(summaries_dir / "oof_failure_analysis.json", failure_analysis)
    write_json(summaries_dir / "oof_failures.json", list(oof_result.get("failures", []) or []))
    write_json(summaries_dir / "oof_analysis_rows.json", list(oof_result.get("analysis_rows", []) or []))
    if external_slice_summary:
        write_json(summaries_dir / "external_slice_summary.json", external_slice_summary)

    mechanism_spec = _build_mechanism_spec(
        protocol_mode=protocol_mode,
        bootstrap_iterations=bootstrap_iterations,
        recursive_config=recursive_config,
        discovery_model_override=discovery_model_override,
        judge_model_override=judge_model_override,
        max_criteria=max_criteria,
        max_pairs_per_example=max_pairs_per_example,
        covariance_ridge=covariance_ridge,
        train_reference_answer_access=train_reference_answer_access,
        blind_scoring_profile=blind_scoring_profile,
        blind_budget_profile=blind_budget_profile,
        blind_guidance_profile=blind_guidance_profile,
        blind_wu_profile=blind_wu_profile,
        retrieval_profile=retrieval_profile,
        retrieval_top_k=max(1, int(retrieval_top_k)),
        blind_discriminator_mode_by_family=blind_discriminator_mode_by_family,
        retrieval_profile_by_family=retrieval_profile_by_family,
        retrieval_top_k_by_family=retrieval_top_k_by_family,
    )
    mechanism_hash = _mechanism_hash(mechanism_spec)
    fold_policy_hashes = sorted(
        {
            str((fold_result.get("summary", {}) or {}).get("policy_hash", "")).strip()
            for fold_result in fold_results
            if str((fold_result.get("summary", {}) or {}).get("policy_hash", "")).strip()
        }
    )
    locked_policy = build_initial_frozen_policy(
        train_examples=train_joined,
        bootstrap_iterations=bootstrap_iterations,
        recursive_config=recursive_config,
        protocol_mode=protocol_mode,
        reference_answer_access=train_reference_answer_access,
        blind_scoring_profile=blind_scoring_profile,
        blind_budget_profile=blind_budget_profile,
        blind_guidance_profile=blind_guidance_profile,
        blind_wu_profile=blind_wu_profile,
        retrieval_profile=retrieval_profile,
        retrieval_top_k=max(1, int(retrieval_top_k)),
        blind_discriminator_mode_by_family=blind_discriminator_mode_by_family,
        retrieval_profile_by_family=retrieval_profile_by_family,
        retrieval_top_k_by_family=retrieval_top_k_by_family,
        self_consistency_n=int(
            normalized_v2_config.get("self_consistency_n", _DEFAULT_SELF_CONSISTENCY_N)
            or _DEFAULT_SELF_CONSISTENCY_N
        ),
        self_consistency_temperature=float(
            normalized_v2_config.get(
                "self_consistency_temperature", _DEFAULT_SELF_CONSISTENCY_TEMPERATURE
            )
            or _DEFAULT_SELF_CONSISTENCY_TEMPERATURE
        ),
        v2_wide_discriminator_gate=bool(normalized_v2_config.get("v2_wide_discriminator_gate", False)),
        holistic_judge_enabled=bool(normalized_v2_config.get("holistic_judge_enabled", False)),
        library_retrieval_top_k=int(normalized_v2_config.get("library_retrieval_top_k", 0) or 0),
        rubric_library_path=normalized_v2_config.get("rubric_library_path") or None,
        enable_rrd_filters=bool(normalized_v2_config.get("enable_rrd_filters", False)),
        rrd_redundancy_threshold=float(
            normalized_v2_config.get("rrd_redundancy_threshold", 0.9) or 0.9
        ),
        library_retrieval_top_k_by_family=normalized_v2_config.get(
            "library_retrieval_top_k_by_family"
        ) or {},
        family_strict_library_mode=bool(
            normalized_v2_config.get("family_strict_library_mode", False)
        ),
        math_independent_solver_enabled=bool(
            normalized_v2_config.get("math_independent_solver_enabled", False)
        ),
        math_solver_samples=int(normalized_v2_config.get("math_solver_samples", 1) or 1),
        math_solver_temperature=float(
            normalized_v2_config.get("math_solver_temperature", 0.5) or 0.5
        ),
        code_execution_verifier_enabled=bool(
            normalized_v2_config.get("code_execution_verifier_enabled", False)
        ),
        code_execution_timeout_s=float(
            normalized_v2_config.get("code_execution_timeout_s", 10.0) or 10.0
        ),
        code_execution_min_margin=float(
            normalized_v2_config.get("code_execution_min_margin", 0.34) or 0.34
        ),
        math_solver_use_sympy=bool(
            normalized_v2_config.get("math_solver_use_sympy", False)
        ),
        math_solver_model=str(normalized_v2_config.get("math_solver_model", "") or ""),
        few_shot_train_enabled=bool(
            normalized_v2_config.get("few_shot_train_enabled", False)
        ),
        few_shot_top_k=int(normalized_v2_config.get("few_shot_top_k", 3) or 3),
        few_shot_train_dataset_path=str(
            normalized_v2_config.get("few_shot_train_dataset_path", "") or ""
        ),
        few_shot_official_dataset_path=str(
            normalized_v2_config.get("few_shot_official_dataset_path", "") or ""
        ),
        rubric_satisfaction_samples=int(
            normalized_v2_config.get("rubric_satisfaction_samples", 1) or 1
        ),
        rubric_satisfaction_temperature=float(
            normalized_v2_config.get("rubric_satisfaction_temperature", 0.4) or 0.4
        ),
        discriminator_self_critique_enabled=bool(
            normalized_v2_config.get("discriminator_self_critique_enabled", False)
        ),
        mmlu_independent_answerer_enabled=bool(
            normalized_v2_config.get("mmlu_independent_answerer_enabled", False)
        ),
        mmlu_answerer_samples=int(normalized_v2_config.get("mmlu_answerer_samples", 1) or 1),
        mmlu_answerer_temperature=float(
            normalized_v2_config.get("mmlu_answerer_temperature", 0.5) or 0.5
        ),
        mmlu_answerer_model=str(normalized_v2_config.get("mmlu_answerer_model", "") or ""),
        mmlu_answerer_secondary_model=str(
            normalized_v2_config.get("mmlu_answerer_secondary_model", "") or ""
        ),
        mmlu_answerer_secondary_samples=int(
            normalized_v2_config.get("mmlu_answerer_secondary_samples", 1) or 1
        ),
        mmlu_answerer_secondary_temperature=float(
            normalized_v2_config.get("mmlu_answerer_secondary_temperature", 0.0) or 0.0
        ),
        reasoning_independent_solver_enabled=bool(
            normalized_v2_config.get("reasoning_independent_solver_enabled", False)
        ),
        reasoning_solver_samples=int(
            normalized_v2_config.get("reasoning_solver_samples", 1) or 1
        ),
        reasoning_solver_temperature=float(
            normalized_v2_config.get("reasoning_solver_temperature", 0.5) or 0.5
        ),
        reasoning_solver_model=str(normalized_v2_config.get("reasoning_solver_model", "") or ""),
    )
    locked_policy["locking_metadata"] = {
        "train_split_name": train_split_name,
        "train_pair_count": len(train_joined),
        "train_pair_ids": [row.pair_id for row in train_joined],
        "train_dataset_path": str(Path(train_dataset_path).resolve()),
        "official_dataset_path": str(Path(official_path).resolve()),
        "train_reference_answer_access": bool(train_reference_answer_access),
        "blind_parity_bootstrap": not bool(train_reference_answer_access),
        "oof_reference_answer_access": bool(oof_reference_answer_access),
        "write_train_fit": bool(write_train_fit),
        "train_fit_reference_answer_access": bool(train_fit_reference_answer_access),
        "mechanism_spec": mechanism_spec,
        "mechanism_hash": mechanism_hash,
        "fold_count": len(folds),
        "export_strategy": "full_train_rebuild",
        "fold_policy_hashes": fold_policy_hashes,
        "frozen_policy_hash": _policy_core_hash(locked_policy),
    }
    write_json(frozen_policy_dir / "locked_policy.json", locked_policy)

    train_fit_result: Optional[Dict[str, Any]] = None
    train_fit_failure_analysis: Optional[Dict[str, Any]] = None
    if write_train_fit:
        train_fit_root = run_dir / "train_fit"
        train_fit_result = run_judgebench_split(
            examples=train_joined,
            split_name=f"{train_split_name}_train_fit",
            split_dir=train_fit_root,
            policy=locked_policy,
            discovery_model_override=discovery_model_override,
            judge_model_override=judge_model_override,
            use_cache=use_cache,
            max_criteria=max_criteria,
            max_pairs_per_example=max_pairs_per_example,
            covariance_ridge=covariance_ridge,
            max_workers=max_workers,
            resume=resume,
            reference_answer_access=train_fit_reference_answer_access,
            retrieval_examples=train_joined if _policy_has_any_retrieval(locked_policy) else None,
        )
        _assert_split_summary_persisted(train_fit_result)
        train_fit_failure_analysis = build_judgebench_failure_analysis(split_result=train_fit_result)
        write_json(summaries_dir / "train_fit_summary.json", train_fit_result["summary"])
        write_json(summaries_dir / "train_fit_failure_analysis.json", train_fit_failure_analysis)
        write_json(summaries_dir / "train_fit_failures.json", list(train_fit_result.get("failures", []) or []))
        write_json(
            summaries_dir / "train_fit_analysis_rows.json",
            list(train_fit_result.get("analysis_rows", []) or []),
        )

    locked_policy_alignment = _build_locked_policy_alignment_summary(
        locked_policy=locked_policy,
        oof_result=oof_result,
        train_fit_result=train_fit_result,
        fold_results=fold_results,
    )
    write_json(summaries_dir / "locked_policy_alignment.json", locked_policy_alignment)

    summary = {
        "schema": "compiled_judgebench_train_only_summary_v1",
        "run_dir": str(run_dir.resolve()),
        "train_dataset_path": str(Path(train_dataset_path).resolve()),
        "official_dataset_path": str(Path(official_path).resolve()),
        "train_split_name": train_split_name,
        "protocol_mode": protocol_mode,
        "fold_count": len(folds),
        "parallelism": parallelism,
        "train_reference_answer_access": bool(train_reference_answer_access),
        "blind_parity_bootstrap": not bool(train_reference_answer_access),
        "oof_reference_answer_access": bool(oof_reference_answer_access),
        "write_train_fit": bool(write_train_fit),
        "train_fit_reference_answer_access": bool(train_fit_reference_answer_access),
        "fold_shuffle_seed": int(fold_shuffle_seed) if fold_shuffle_seed is not None else None,
        "blind_scoring_profile": blind_scoring_profile,
        "blind_budget_profile": blind_budget_profile,
        "blind_guidance_profile": blind_guidance_profile,
        "blind_wu_profile": blind_wu_profile,
        "retrieval_profile": retrieval_profile,
        "retrieval_top_k": max(1, int(retrieval_top_k)),
        "blind_discriminator_mode_by_family": blind_discriminator_mode_by_family,
        "retrieval_profile_by_family": retrieval_profile_by_family,
        "retrieval_top_k_by_family": retrieval_top_k_by_family,
        "mechanism_hash": mechanism_hash,
        "frozen_policy_hash": str(locked_policy["locking_metadata"]["frozen_policy_hash"]),
        "oof_summary": oof_result["summary"],
        "failure_analysis": failure_analysis,
        "locked_policy_alignment": locked_policy_alignment,
        "train_fit_summary": (train_fit_result or {}).get("summary"),
        "train_fit_failure_analysis": train_fit_failure_analysis,
        "fold_summaries": [
            {
                "fold_index": fold["fold_index"],
                "split_name": fold_result["summary"]["split_name"],
                "policy_hash": fold_result["summary"]["policy_hash"],
                "wu_metrics": fold_result["summary"]["wu_metrics"],
                "summary_path": fold_result["paths"]["summary"],
            }
            for fold, fold_result in zip(folds, fold_results)
        ],
        "paths": {
            "dataset_dir": str(dataset_dir.resolve()),
            "folds_root": str(folds_root.resolve()),
            "frozen_policy": str((frozen_policy_dir / "locked_policy.json").resolve()),
            "oof_summary": str((summaries_dir / "oof_summary.json").resolve()),
            "oof_failure_analysis": str((summaries_dir / "oof_failure_analysis.json").resolve()),
            "oof_failures": str((summaries_dir / "oof_failures.json").resolve()),
            "oof_analysis_rows": str((summaries_dir / "oof_analysis_rows.json").resolve()),
            "locked_policy_alignment": str((summaries_dir / "locked_policy_alignment.json").resolve()),
            "train_fit_root": str((run_dir / "train_fit").resolve()) if write_train_fit else "",
            "train_fit_summary": str((summaries_dir / "train_fit_summary.json").resolve()) if write_train_fit else "",
            "train_fit_failure_analysis": (
                str((summaries_dir / "train_fit_failure_analysis.json").resolve()) if write_train_fit else ""
            ),
            "train_fit_failures": str((summaries_dir / "train_fit_failures.json").resolve()) if write_train_fit else "",
            "train_fit_analysis_rows": (
                str((summaries_dir / "train_fit_analysis_rows.json").resolve()) if write_train_fit else ""
            ),
        },
    }
    write_json(summaries_dir / "summary.json", summary)
    return run_dir, summary


def run_judgebench_final_evaluation(
    *,
    train_run_dir: Path,
    validation_dataset_path: Path,
    validation_split_name: str,
    run_dir: Path,
    official_dataset_path: Optional[Path],
    max_workers: int,
    write_detailed_outputs: bool,
    resume: bool,
    reference_answer_access: bool = False,
    retrieval_profile: Optional[str] = None,
    retrieval_profile_by_family: Optional[Mapping[str, Any]] = None,
    retrieval_top_k_by_family: Optional[Mapping[str, Any]] = None,
    blind_discriminator_mode_by_family: Optional[Mapping[str, Any]] = None,
    shared_cache_dir: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    train_run_dir = Path(train_run_dir)
    validation_dataset_path = Path(validation_dataset_path)
    run_dir = Path(run_dir)
    validation_split_name = _safe_slug(validation_split_name)

    train_summary = _read_json(train_run_dir / "summaries" / "summary.json")
    locked_policy = _read_json(train_run_dir / "frozen_policy" / "locked_policy.json")
    locking_metadata = dict(locked_policy.get("locking_metadata", {}) or {})
    mechanism_spec = dict(locking_metadata.get("mechanism_spec", {}) or {})
    stored_mechanism_hash = str(locking_metadata.get("mechanism_hash", "")).strip()
    if _mechanism_hash(mechanism_spec) != stored_mechanism_hash:
        raise RuntimeError("Locked policy mechanism hash did not match its embedded mechanism spec.")
    if stored_mechanism_hash != str(train_summary.get("mechanism_hash", "")).strip():
        raise RuntimeError("Locked policy mechanism hash did not match the train-only run summary.")
    if _policy_core_hash(locked_policy) != str(locking_metadata.get("frozen_policy_hash", "")).strip():
        raise RuntimeError("Locked policy hash verification failed.")

    dataset_dir = run_dir / "dataset"
    evaluation_root = run_dir / validation_split_name
    summaries_dir = run_dir / "summaries"
    for path in (dataset_dir, evaluation_root, summaries_dir):
        path.mkdir(parents=True, exist_ok=True)

    stored_official_dataset = str(locking_metadata.get("official_dataset_path", "")).strip() or str(
        train_summary.get("official_dataset_path", "")
    ).strip()
    resolved_official_dataset = (
        Path(official_dataset_path)
        if official_dataset_path is not None
        else (Path(stored_official_dataset) if stored_official_dataset else None)
    )
    if official_dataset_path is None and resolved_official_dataset is not None and not resolved_official_dataset.exists():
        resolved_official_dataset = None
    official_dataset = ensure_official_judgebench_dataset(resolved_official_dataset, dataset_dir)
    official_pairs = load_official_judgebench_pairs(official_dataset)
    validation_local = load_local_judgebench_subset(validation_dataset_path)
    validation_joined = join_local_subset_to_official_pairs(
        local_rows=validation_local,
        official_pairs=official_pairs,
        split_name=validation_split_name,
    )
    train_pair_ids = {str(item).strip() for item in locking_metadata.get("train_pair_ids", []) or []}
    overlap = train_pair_ids & {row.pair_id for row in validation_joined}
    if overlap:
        preview = sorted(overlap)[:10]
        raise RuntimeError(f"Locked train policy overlaps validation pair_ids: {preview}")
    write_json(
        dataset_dir / "dataset_manifest.json",
        {
            "train_run_dir": str(train_run_dir.resolve()),
            "train_dataset_path": str(train_summary.get("train_dataset_path", "")),
            "validation_dataset_path": str(validation_dataset_path.resolve()),
            "official_dataset_path": str(official_dataset.resolve()),
            "validation_split_name": validation_split_name,
            "validation_pair_count": len(validation_joined),
            "reference_answer_access": bool(reference_answer_access),
        },
    )

    recursive_config = RecursiveDiscoveryConfig(
        max_depth=int((mechanism_spec.get("recursive_config", {}) or {}).get("max_depth", 1)),
        max_recursive_parents_per_pair=int(
            (mechanism_spec.get("recursive_config", {}) or {}).get("max_recursive_parents_per_pair", 2)
        ),
        max_children_per_parent=int((mechanism_spec.get("recursive_config", {}) or {}).get("max_children_per_parent", 3)),
        max_recursive_calls_per_pair=int(
            (mechanism_spec.get("recursive_config", {}) or {}).get("max_recursive_calls_per_pair", 2)
        ),
    )
    effective_retrieval_profile = _normalize_retrieval_profile(
        retrieval_profile if retrieval_profile is not None else mechanism_spec.get("retrieval_profile")
    )
    retrieval_profile_by_family = _normalize_retrieval_profile_by_family(retrieval_profile_by_family)
    retrieval_top_k_by_family = _normalize_retrieval_top_k_by_family(retrieval_top_k_by_family)
    blind_discriminator_mode_by_family = _normalize_blind_discriminator_mode_by_family(
        blind_discriminator_mode_by_family
    )
    effective_policy = copy.deepcopy(locked_policy)
    effective_policy["retrieval_profile"] = effective_retrieval_profile
    if retrieval_profile is not None and not retrieval_profile_by_family and not retrieval_top_k_by_family:
        effective_policy["retrieval_profile_by_family"] = {}
        effective_policy["retrieval_top_k_by_family"] = {}
    if retrieval_profile_by_family:
        effective_policy["retrieval_profile_by_family"] = retrieval_profile_by_family
    if retrieval_top_k_by_family:
        effective_policy["retrieval_top_k_by_family"] = retrieval_top_k_by_family
    if blind_discriminator_mode_by_family:
        effective_policy["blind_discriminator_mode_by_family"] = blind_discriminator_mode_by_family
    retrieval_examples: Optional[List[JudgeBenchJoinedExample]] = None
    if _policy_has_any_retrieval(effective_policy):
        train_split_name = str(locking_metadata.get("train_split_name", "")).strip() or str(
            train_summary.get("train_split_name", "")
        ).strip()
        joined_train_path = train_run_dir / "dataset" / f"joined_{train_split_name}.json"
        if joined_train_path.exists():
            retrieval_examples = [
                JudgeBenchJoinedExample(**row)
                for row in json.loads(joined_train_path.read_text(encoding="utf-8"))
            ]
    final_result = run_judgebench_split(
        examples=validation_joined,
        split_name=validation_split_name,
        split_dir=evaluation_root / "final",
        policy=effective_policy,
        discovery_model_override=str(mechanism_spec.get("discovery_model_override", "")).strip() or None,
        judge_model_override=str(mechanism_spec.get("judge_model_override", "")).strip() or None,
        use_cache=True,
        max_criteria=int(mechanism_spec.get("max_criteria", 8) or 8),
        max_pairs_per_example=(
            None
            if mechanism_spec.get("max_pairs_per_example") is None
            else int(mechanism_spec.get("max_pairs_per_example", 4))
        ),
        covariance_ridge=float(mechanism_spec.get("covariance_ridge", 1e-3) or 1e-3),
        max_workers=max_workers,
        resume=resume if write_detailed_outputs else False,
        write_example_artifacts=write_detailed_outputs,
        write_reports=write_detailed_outputs,
        reference_answer_access=reference_answer_access,
        retrieval_examples=retrieval_examples,
        shared_cache_dir=shared_cache_dir,
    )
    _assert_split_summary_persisted(final_result)
    validation_failure_bundle = _write_split_failure_bundle(
        summaries_dir=summaries_dir,
        stem=validation_split_name,
        split_result=final_result,
    )
    diagnostic_subsets: Dict[str, Any] = {}
    diagnostic_subset_errors: Dict[str, str] = {}
    for subset_name, subset_dataset_path in _canonical_final_eval_subset_specs():
        if not subset_dataset_path.exists():
            continue
        try:
            subset_local_rows = load_local_judgebench_subset(subset_dataset_path)
        except Exception as exc:
            diagnostic_subset_errors[subset_name] = str(exc)
            continue
        subset_pair_ids = {str(row.pair_id).strip() for row in subset_local_rows}
        subset_examples = [example for example in validation_joined if str(example.pair_id).strip() in subset_pair_ids]
        if not subset_examples:
            continue
        subset_result = _slice_split_result(
            split_name=subset_name,
            examples=subset_examples,
            split_result=final_result,
        )
        subset_summary_path = summaries_dir / f"{subset_name}_summary.json"
        write_json(subset_summary_path, subset_result["summary"])
        subset_failure_bundle = _write_split_failure_bundle(
            summaries_dir=summaries_dir,
            stem=subset_name,
            split_result=subset_result,
        )
        diagnostic_subsets[subset_name] = {
            "dataset_path": str(subset_dataset_path.resolve()),
            "pair_count": len(subset_examples),
            "summary": subset_result["summary"],
            "failure_analysis": subset_failure_bundle["failure_analysis"],
            "paths": {
                "summary": str(subset_summary_path.resolve()),
                **subset_failure_bundle["paths"],
            },
        }
    summary = {
        "schema": "compiled_judgebench_final_eval_summary_v1",
        "run_dir": str(run_dir.resolve()),
        "train_run_dir": str(train_run_dir.resolve()),
        "train_dataset_path": str(train_summary.get("train_dataset_path", "")),
        "validation_dataset_path": str(validation_dataset_path.resolve()),
        "official_dataset_path": str(official_dataset.resolve()),
        "validation_split_name": validation_split_name,
        "protocol_mode": _policy_protocol_mode(locked_policy),
        "mechanism_hash": stored_mechanism_hash,
        "frozen_policy_hash": str(locking_metadata.get("frozen_policy_hash", "")).strip(),
        "reference_answer_access": bool(reference_answer_access),
        "blind_guidance_profile": _policy_blind_guidance_profile(effective_policy),
        "blind_wu_profile": _policy_blind_wu_profile(effective_policy),
        "retrieval_profile": effective_retrieval_profile,
        "blind_discriminator_mode_by_family": _normalize_blind_discriminator_mode_by_family(
            effective_policy.get("blind_discriminator_mode_by_family")
        ),
        "retrieval_profile_by_family": _normalize_retrieval_profile_by_family(
            effective_policy.get("retrieval_profile_by_family")
        ),
        "retrieval_top_k": _policy_retrieval_top_k(effective_policy),
        "retrieval_top_k_by_family": _normalize_retrieval_top_k_by_family(
            effective_policy.get("retrieval_top_k_by_family")
        ),
        "write_detailed_outputs": bool(write_detailed_outputs),
        "validation_summary": final_result["summary"],
        "validation_failure_analysis": validation_failure_bundle["failure_analysis"],
        "diagnostic_subsets": diagnostic_subsets,
        "diagnostic_subset_errors": diagnostic_subset_errors,
        "paths": {
            "dataset_dir": str(dataset_dir.resolve()),
            "validation_root": str(evaluation_root.resolve()),
            "validation_summary": str(Path(final_result["paths"]["summary"]).resolve()),
            "validation_failure_analysis": validation_failure_bundle["paths"]["failure_analysis"],
            "validation_failures": validation_failure_bundle["paths"]["failures"],
            "validation_analysis_rows": validation_failure_bundle["paths"]["analysis_rows"],
            "diagnostic_subsets": {
                name: dict(payload.get("paths", {}) or {})
                for name, payload in sorted(diagnostic_subsets.items())
            },
        },
    }
    write_json(summaries_dir / "summary.json", summary)
    return run_dir, summary
