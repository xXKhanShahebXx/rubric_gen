from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rubric_gen.compiled.contrast_strategies import (
    ContrastStrategy,
    get_contrast_strategy,
    mutation_function_for_id,
    register_contrast_strategy,
)
from rubric_gen.compiled.task_profiles import (
    TaskProfile,
    get_task_profile,
    infer_task_profile_id_from_text,
    register_task_profile,
)
from rubric_gen.dataio import strongest_anchor_text
from rubric_gen.types import ExampleRecord

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "write",
    "rewrite",
    "task",
    "requested",
    "request",
    "response",
    "output",
    "context",
}
_BUILT_IN_PROFILE_IDS: Tuple[str, ...] = (
    "note_documentation",
    "documentation_variants",
    "rewrite_editing",
    "clinical_decision_support",
    "general_instruction_following",
    "agentic_workflows",
)
_FEATURE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "structure": re.compile(r"(?mi)(^#{1,6}\s+|^\s*[-*]\s+|^\s*\d+\.\s+|^\s*[A-Z][A-Z /_-]{2,}:|section|heading|format|bullet)"),
    "actions": re.compile(r"\b(plan|plans|recommend(?:ation)?|next step|follow[\s-]*up|schedule|book|arrange|continue|start|stop)\b", re.IGNORECASE),
    "evidence": re.compile(r"\b(because|due to|based on|supported by|evidence|rationale|reasoning|findings?)\b", re.IGNORECASE),
    "constraints": re.compile(r"\b(must|should|only|do not|without|preserve|keep|maintain|exactly|under \d+)\b", re.IGNORECASE),
    "steps": re.compile(r"(?mi)(^step\s+\d+[:.)-]?|^\s*\d+\.\s+|observation:|action:|result:)"),
    "tool_results": re.compile(r"\b(tool|query|search|stdout|stderr|observation|result|results|exit code)\b", re.IGNORECASE),
    "verification": re.compile(r"\b(verify|verified|checked|confirmed|validation|double-check)\b", re.IGNORECASE),
    "failure": re.compile(r"\b(failed|failure|unable|blocked|retry|fallback|error)\b", re.IGNORECASE),
    "rewrite": re.compile(r"\b(rewrite|rephrase|edit|grammar|proofread|tone|style|formal|casual|clarity)\b", re.IGNORECASE),
    "recommendation": re.compile(r"\b(recommend(?:ation)?|management|triage|differential|next step|plan)\b", re.IGNORECASE),
    "documentation": re.compile(r"\b(note|summary|progress note|discharge|pre-?op|handoff|alert|message|report)\b", re.IGNORECASE),
    "clinical": re.compile(r"\b(clinical|patient|diagnosis|symptom|exam|medical|medication|follow-up)\b", re.IGNORECASE),
    "answer_format": re.compile(
        r"\b(duplicate that letter five times|single string|single word|single digit|return your answer|format:\s*\*{3}|put your answer in \*{2,})\b",
        re.IGNORECASE,
    ),
    "multiple_choice": re.compile(r"\([A-J]\)|\b[A-J]{5}\b", re.IGNORECASE),
    "code": re.compile(
        r"\b(class\s+solution|def\s+\w+\(|input\b|output\b|constraints\b|#\s*your code here|leetcode|time complexity)\b",
        re.IGNORECASE,
    ),
}
_FALLBACK_PROMPT_HINTS: Dict[str, str] = {
    "documentation_variants": "Produce the requested document from the provided context.",
    "rewrite_editing": "Rewrite the provided text while preserving the requested constraints and meaning.",
    "clinical_decision_support": "Generate a grounded clinical decision-support artifact from the provided context.",
    "general_instruction_following": "Complete the requested task using the provided context.",
    "agentic_workflows": "Produce the requested workflow output grounded in the observed steps and results.",
}
_SOURCE_FAMILY_PROMPT_HINTS: Dict[str, str] = {
    "livebench-reasoning": "Focus on clue consistency, contradiction-free reasoning, and the exact final conclusion.",
    "livebench-math": "Focus on mathematical correctness, the exact final value, and any rigid answer format.",
    "mmlu-pro": "Focus on selecting the exact correct option and emitting the requested final-answer string.",
    "livecodebench": "Focus on executable correctness, required input/output behavior, edge cases, and constraints.",
}
_SOURCE_FAMILY_DISCOVERY_CONTEXT: Dict[str, str] = {
    "livebench-reasoning": "logic-puzzle and constraint-following response quality",
    "livebench-math": "mathematical reasoning and exact-answer response quality",
    "mmlu-pro": "multiple-choice option-selection and exact-answer response quality",
    "livecodebench": "code-solution correctness and input/output behavior quality",
}
_SOURCE_FAMILY_DISCOVERY_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "livebench-reasoning": (
        "constraint_satisfaction",
        "clue_consistency",
        "conclusion_correctness",
        "verification",
        "format_communication",
    ),
    "livebench-math": (
        "mathematical_correctness",
        "constraint_satisfaction",
        "final_answer_correctness",
        "reasoning_support",
        "format_communication",
    ),
    "mmlu-pro": (
        "final_answer_correctness",
        "option_selection",
        "reasoning_support",
        "format_communication",
        "instruction_adherence",
    ),
    "livecodebench": (
        "executable_correctness",
        "input_output_behavior",
        "edge_cases",
        "constraint_satisfaction",
        "format_communication",
    ),
}
_SOURCE_FAMILY_MUTATION_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "livebench-reasoning": (
        "corrupt_final_answer",
        "drop_constraints",
        "drop_supporting_evidence",
        "drop_verification",
        "drop_steps",
        "remove_format_markers",
        "add_unsupported_detail",
    ),
    "livebench-math": (
        "corrupt_final_answer",
        "drop_constraints",
        "drop_supporting_evidence",
        "drop_steps",
        "remove_format_markers",
        "add_unsupported_detail",
    ),
    "mmlu-pro": (
        "corrupt_final_answer",
        "drop_constraints",
        "drop_supporting_evidence",
        "remove_format_markers",
        "add_unsupported_detail",
    ),
    "livecodebench": (
        "code_off_by_one_loop",
        "code_flip_condition_branch",
        "code_corrupt_input_parsing",
        "code_drop_negative_directions",
        "code_remove_zero_guard",
        "code_drop_reset_logic",
        "corrupt_final_answer",
        "drop_constraints",
        "drop_steps",
        "drop_tool_results",
        "remove_format_markers",
        "add_unsupported_detail",
    ),
}


@dataclass(frozen=True)
class TaskProfileResolution:
    profile: TaskProfile
    strategy: ContrastStrategy
    bootstrap_used: bool
    iterations_run: int
    diagnostics: Dict[str, Any]


def _example_blob(example: ExampleRecord) -> str:
    return " ".join(
        part
        for part in (
            example.task_prompt,
            example.conversation,
            example.reference_artifact,
            example.augmented_artifact,
            example.task_family_id,
        )
        if part
    )


def _feature_counts(examples: Sequence[ExampleRecord]) -> Dict[str, int]:
    counts = {name: 0 for name in _FEATURE_PATTERNS}
    for example in examples:
        blob = _example_blob(example)
        for name, pattern in _FEATURE_PATTERNS.items():
            if pattern.search(blob):
                counts[name] += 1
    return counts


def _normalize_source_family(source: str) -> str:
    normalized = (source or "").strip().lower()
    if normalized.startswith("mmlu-pro"):
        return "mmlu-pro"
    if normalized.startswith("livebench-reasoning"):
        return "livebench-reasoning"
    if normalized.startswith("livebench-math"):
        return "livebench-math"
    if normalized.startswith("livecodebench"):
        return "livecodebench"
    return normalized


def _source_family_hint(examples: Sequence[ExampleRecord]) -> str:
    counts: Counter[str] = Counter()
    for example in examples:
        hint = str(example.metadata.get("source_family", "")).strip().lower()
        if not hint:
            hint = _normalize_source_family(example.source)
        if hint:
            counts[hint] += 1
    return counts.most_common(1)[0][0] if counts else ""


def _prioritize_mutations(mutation_ids: Sequence[str], *, source_family: str) -> List[str]:
    priority = _SOURCE_FAMILY_MUTATION_PRIORITY.get(source_family, ())
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


def _top_terms(examples: Sequence[ExampleRecord], *, limit: int = 3) -> List[str]:
    counter: Counter[str] = Counter()
    for example in examples:
        blob = " ".join(
            part
            for part in (
                example.task_prompt,
                strongest_anchor_text(example),
                example.conversation[:400],
            )
            if part
        ).lower()
        for token in _WORD_RE.findall(blob):
            if len(token) < 4 or token in _STOPWORDS:
                continue
            counter[token] += 1
    return [token for token, _ in counter.most_common(limit)]


def _score_builtin_profiles(examples: Sequence[ExampleRecord], feature_counts: Mapping[str, int]) -> Dict[str, float]:
    combined_text = "\n".join(_example_blob(example) for example in examples).lower()
    raw_scores = {profile_id: 0.0 for profile_id in _BUILT_IN_PROFILE_IDS}

    inferred = [infer_task_profile_id_from_text(_example_blob(example)) for example in examples]
    inferred_counts = Counter(inferred)
    for profile_id, count in inferred_counts.items():
        raw_scores[profile_id] += float(count) * 2.0

    raw_scores["note_documentation"] += feature_counts["documentation"] * 0.5 + feature_counts["clinical"] * 1.5
    raw_scores["documentation_variants"] += feature_counts["documentation"] * 2.0 + feature_counts["structure"] * 1.25 + feature_counts["actions"] * 0.75
    raw_scores["rewrite_editing"] += feature_counts["rewrite"] * 3.0 + feature_counts["constraints"] * 1.0 + feature_counts["structure"] * 0.5
    raw_scores["clinical_decision_support"] += feature_counts["recommendation"] * 2.5 + feature_counts["evidence"] * 1.0 + feature_counts["clinical"] * 1.0
    raw_scores["general_instruction_following"] += (
        feature_counts["constraints"] * 1.5
        + feature_counts["structure"] * 0.5
        + feature_counts["answer_format"] * 1.5
        + feature_counts["multiple_choice"] * 1.25
        + feature_counts["code"] * 1.0
        + len(examples) * 0.5
    )
    raw_scores["agentic_workflows"] += (
        feature_counts["steps"] * 2.0
        + feature_counts["tool_results"] * 2.5
        + feature_counts["verification"] * 1.5
        + feature_counts["failure"] * 1.5
    )

    if "healthcare scribe" in combined_text or "soap note" in combined_text:
        raw_scores["note_documentation"] += 6.0
    if "rewrite" in combined_text or "rephrase" in combined_text:
        raw_scores["rewrite_editing"] += 4.0
    if "workflow" in combined_text or "tool result" in combined_text:
        raw_scores["agentic_workflows"] += 4.0
    if "differential" in combined_text or "triage" in combined_text:
        raw_scores["clinical_decision_support"] += 4.0
    return raw_scores


def _pick_parent_profile(scores: Mapping[str, float]) -> str:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[0][0] if ordered else "general_instruction_following"


def _should_bootstrap(
    *,
    explicit: Optional[str],
    inferred_static_profiles: Sequence[str],
    scores: Mapping[str, float],
) -> bool:
    if explicit and explicit.strip().lower() in {"auto", "bootstrap"}:
        return True

    distinct = {profile_id for profile_id in inferred_static_profiles if profile_id}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return True
    top_profile, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else -999.0
    if top_profile == "general_instruction_following":
        return True
    if len(distinct) > 1:
        return True
    if top_score - second_score < 2.0:
        return True
    return False


def _suggest_mutations(
    parent_profile_id: str,
    feature_counts: Mapping[str, int],
    *,
    source_family: str = "",
) -> List[str]:
    selected: List[str] = []

    if parent_profile_id in {"documentation_variants", "general_instruction_following"} and feature_counts["structure"] > 0:
        selected.append("flatten_document_scaffold" if parent_profile_id == "documentation_variants" else "remove_format_markers")
    if parent_profile_id == "rewrite_editing":
        selected.extend(["remove_format_markers", "drop_rewrite_signal", "introduce_typos"])
    if feature_counts["actions"] > 0:
        selected.append("drop_action_items")
    if feature_counts["evidence"] > 0:
        selected.append("drop_supporting_evidence")
    if feature_counts["constraints"] > 0 and parent_profile_id != "documentation_variants":
        selected.append("drop_constraints")
    if feature_counts["answer_format"] > 0 or feature_counts["multiple_choice"] > 0:
        selected.append("corrupt_final_answer")
    if source_family == "livecodebench" and feature_counts["code"] > 0:
        selected.extend(
            [
                "code_off_by_one_loop",
                "code_flip_condition_branch",
                "code_corrupt_input_parsing",
                "code_drop_negative_directions",
                "code_remove_zero_guard",
                "code_drop_reset_logic",
            ]
        )
    if feature_counts["steps"] > 0:
        selected.append("drop_steps")
    if feature_counts["tool_results"] > 0:
        selected.append("drop_tool_results")
    if feature_counts["verification"] > 0:
        selected.append("drop_verification")
    if feature_counts["failure"] > 0:
        selected.append("drop_failure_handling")
    if parent_profile_id == "agentic_workflows":
        selected.append("drop_final_answer")
    if parent_profile_id == "clinical_decision_support":
        selected.append("inflate_certainty_generic")
    if parent_profile_id == "general_instruction_following":
        selected.append("add_unsupported_detail")
    if not selected:
        base_strategy = get_contrast_strategy(parent_profile_id)
        selected.extend(base_strategy.mutation_ids)
    if "add_unsupported_detail" not in selected and parent_profile_id != "clinical_decision_support":
        selected.append("add_unsupported_detail")
    deduped: List[str] = []
    seen: set[str] = set()
    for mutation_id in selected:
        if mutation_id in seen:
            continue
        seen.add(mutation_id)
        deduped.append(mutation_id)
    return _prioritize_mutations(deduped, source_family=source_family)


def _mutation_coverage(examples: Sequence[ExampleRecord], mutation_ids: Sequence[str]) -> Dict[str, Dict[str, int]]:
    coverage: Dict[str, Dict[str, int]] = {}
    for mutation_id in mutation_ids:
        fn = mutation_function_for_id(mutation_id)
        changed = 0
        total = 0
        if fn is not None:
            for example in examples:
                anchor = strongest_anchor_text(example)
                if not anchor.strip():
                    continue
                total += 1
                mutated = fn(anchor)
                if mutated.strip() and mutated.strip() != anchor.strip():
                    changed += 1
        coverage[mutation_id] = {"examples_with_anchor": total, "changed_examples": changed}
    return coverage


def _essential_mutations(
    parent_profile_id: str,
    feature_counts: Mapping[str, int],
    *,
    source_family: str = "",
) -> List[str]:
    essentials: List[str] = []
    if feature_counts["rewrite"] > 0:
        essentials.extend(["drop_rewrite_signal", "introduce_typos"])
    if feature_counts["answer_format"] > 0 or feature_counts["multiple_choice"] > 0:
        essentials.append("corrupt_final_answer")
    if source_family in {"livebench-reasoning", "livebench-math", "mmlu-pro"}:
        essentials.append("corrupt_final_answer")
    if source_family == "livecodebench" and feature_counts["code"] > 0:
        essentials.extend(
            [
                "code_flip_condition_branch",
                "code_corrupt_input_parsing",
                "code_off_by_one_loop",
            ]
        )
    if feature_counts["steps"] > 0:
        essentials.append("drop_steps")
    if feature_counts["tool_results"] > 0:
        essentials.append("drop_tool_results")
    if feature_counts["actions"] > 0:
        essentials.append("drop_action_items")
    if feature_counts["evidence"] > 0:
        essentials.append("drop_supporting_evidence")
    if not essentials:
        essentials.extend(get_contrast_strategy(parent_profile_id).mutation_ids[:2])
    deduped: List[str] = []
    seen: set[str] = set()
    for mutation_id in essentials:
        if mutation_id in seen:
            continue
        seen.add(mutation_id)
        deduped.append(mutation_id)
    return _prioritize_mutations(deduped, source_family=source_family)


def _refine_mutations(
    examples: Sequence[ExampleRecord],
    *,
    parent_profile_id: str,
    initial_mutation_ids: Sequence[str],
    feature_counts: Mapping[str, int],
    source_family: str,
    max_iterations: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    current = list(initial_mutation_ids)
    history: List[Dict[str, Any]] = []
    min_changed = max(1, math.ceil(max(1, len(examples)) * 0.15))
    essentials = set(_essential_mutations(parent_profile_id, feature_counts, source_family=source_family))

    for iteration in range(1, max_iterations + 1):
        coverage = _mutation_coverage(examples, current)
        refined = [
            mutation_id
            for mutation_id in current
            if coverage.get(mutation_id, {}).get("changed_examples", 0) >= min_changed or mutation_id in essentials
        ]

        if feature_counts["structure"] > 0:
            preferred_structure = "flatten_document_scaffold" if parent_profile_id == "documentation_variants" else "remove_format_markers"
            if preferred_structure not in refined and mutation_function_for_id(preferred_structure) is not None:
                refined.insert(0, preferred_structure)
        if feature_counts["constraints"] > 0 and "drop_constraints" not in refined:
            refined.append("drop_constraints")
        if feature_counts["verification"] > 0 and "drop_verification" not in refined:
            refined.append("drop_verification")
        if feature_counts["failure"] > 0 and "drop_failure_handling" not in refined:
            refined.append("drop_failure_handling")
        if "add_unsupported_detail" not in refined and "inflate_certainty_generic" not in refined:
            refined.append("add_unsupported_detail")

        deduped: List[str] = []
        seen: set[str] = set()
        for mutation_id in refined:
            if mutation_id in seen:
                continue
            seen.add(mutation_id)
            deduped.append(mutation_id)
        refined = [mutation_id for mutation_id in deduped if mutation_function_for_id(mutation_id) is not None]
        refined = _prioritize_mutations(refined, source_family=source_family)

        history.append(
            {
                "iteration": iteration,
                "input_mutation_ids": list(current),
                "coverage": coverage,
                "refined_mutation_ids": list(refined),
            }
        )
        if refined == current:
            return refined, history
        current = refined
    return current, history


def _grounding_catalog() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for profile_id in _BUILT_IN_PROFILE_IDS:
        strategy = get_contrast_strategy(profile_id)
        for mutation_id, profile in strategy.mutation_grounding_profiles.items():
            if mutation_id not in catalog:
                catalog[mutation_id] = dict(profile)
    return catalog


def _mutation_default_delta_mode(mutation_id: str) -> str:
    if mutation_id.startswith(("add_", "inflate_", "introduce_")):
        return "weak_only"
    return "strong_only"


def _build_grounding_profiles(
    mutation_ids: Sequence[str],
    *,
    feature_terms: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    catalog = _grounding_catalog()
    grounding: Dict[str, Dict[str, Any]] = {}
    for mutation_id in mutation_ids:
        base = dict(catalog.get(mutation_id, {}))
        keywords = list(base.get("keywords", ()))
        for term in feature_terms:
            if term not in keywords:
                keywords.append(term)
        if keywords:
            base["keywords"] = tuple(keywords[:16])
        base.setdefault("delta_mode", _mutation_default_delta_mode(mutation_id))
        base.setdefault(
            "prompt_hint",
            f"The weaker artifact mainly changes behavior related to `{mutation_id}`.",
        )
        grounding[mutation_id] = base
    return grounding


def _profile_label(parent_profile_id: str, salient_terms: Sequence[str]) -> str:
    parent = get_task_profile(parent_profile_id)
    phrase = " ".join(salient_terms[:2]).strip()
    if phrase:
        return f"Auto {phrase.title()} {parent.label}"
    return f"Auto {parent.label}"


def _profile_prompt(parent_profile_id: str, salient_terms: Sequence[str], *, source_family: str = "") -> str:
    hint = _FALLBACK_PROMPT_HINTS.get(parent_profile_id, "Complete the requested task using the provided context.")
    phrase = " ".join(salient_terms[:2]).strip()
    source_hint = _SOURCE_FAMILY_PROMPT_HINTS.get(source_family, "")
    if phrase:
        hint = f"{hint} Focus on the `{phrase}` task variant when inferring quality checks."
    if source_hint:
        hint = f"{hint} {source_hint}"
    return hint


def _profile_discovery_context(parent_profile_id: str, *, source_family: str = "") -> str:
    parent = get_task_profile(parent_profile_id)
    return _SOURCE_FAMILY_DISCOVERY_CONTEXT.get(source_family, parent.discovery_context)


def _profile_discovery_dimensions(parent_profile_id: str, *, source_family: str = "") -> Tuple[str, ...]:
    parent = get_task_profile(parent_profile_id)
    return _SOURCE_FAMILY_DISCOVERY_DIMENSIONS.get(source_family, parent.discovery_dimensions)


def _profile_id(parent_profile_id: str, salient_terms: Sequence[str], examples: Sequence[ExampleRecord]) -> str:
    slug_terms = [term for term in salient_terms[:3] if term]
    slug = "_".join(slug_terms) if slug_terms else "task"
    digest = hashlib.sha256(
        "|".join(example.example_id for example in examples).encode("utf-8")
    ).hexdigest()[:8]
    return f"auto_{parent_profile_id}_{slug}_{digest}"


def _default_task_family_id(parent_profile_id: str, salient_terms: Sequence[str]) -> str:
    slug_terms = [term for term in salient_terms[:3] if term]
    if slug_terms:
        return f"auto_{'_'.join(slug_terms)}"
    return f"auto_{parent_profile_id}"


def bootstrap_task_profile(
    examples: Sequence[ExampleRecord],
    *,
    explicit: Optional[str] = None,
    bootstrap_iterations: int = 3,
) -> TaskProfileResolution:
    if not examples:
        profile = get_task_profile("general_instruction_following")
        return TaskProfileResolution(
            profile=profile,
            strategy=get_contrast_strategy(profile.contrast_strategy_id),
            bootstrap_used=False,
            iterations_run=0,
            diagnostics={"reason": "no_examples"},
        )

    features = _feature_counts(examples)
    scores = _score_builtin_profiles(examples, features)
    parent_profile_id = _pick_parent_profile(scores)
    source_family = _source_family_hint(examples)
    if parent_profile_id == "note_documentation":
        profile = get_task_profile("note_documentation")
        return TaskProfileResolution(
            profile=profile,
            strategy=get_contrast_strategy(profile.contrast_strategy_id),
            bootstrap_used=False,
            iterations_run=0,
            diagnostics={
                "reason": "note_baseline_preferred",
                "score_by_profile": dict(sorted(scores.items())),
            },
        )
    salient_terms = _top_terms(examples)
    mutation_ids, refinement_history = _refine_mutations(
        examples,
        parent_profile_id=parent_profile_id,
        initial_mutation_ids=_suggest_mutations(parent_profile_id, features, source_family=source_family),
        feature_counts=features,
        source_family=source_family,
        max_iterations=max(1, bootstrap_iterations),
    )

    profile_id = _profile_id(parent_profile_id, salient_terms, examples)
    strategy_id = f"{profile_id}__strategy"
    parent = get_task_profile(parent_profile_id)
    strategy = register_contrast_strategy(
        ContrastStrategy(
            strategy_id=strategy_id,
            mutation_ids=tuple(mutation_ids),
            mutation_grounding_profiles=_build_grounding_profiles(
                mutation_ids,
                feature_terms=salient_terms,
            ),
        )
    )
    diagnostics: Dict[str, Any] = {
        "parent_profile_id": parent_profile_id,
        "source_family_hint": source_family,
        "feature_counts": dict(features),
        "salient_terms": list(salient_terms),
        "score_by_profile": dict(sorted(scores.items())),
        "refinement_history": refinement_history,
    }
    profile = register_task_profile(
        TaskProfile(
            task_profile_id=profile_id,
            label=_profile_label(parent_profile_id, salient_terms),
            artifact_label=parent.artifact_label,
            artifact_kind=parent.artifact_kind,
            default_task_prompt=_profile_prompt(parent_profile_id, salient_terms, source_family=source_family),
            default_task_family_id=_default_task_family_id(parent_profile_id, salient_terms),
            contrast_strategy_id=strategy_id,
            strong_source_priority=parent.strong_source_priority,
            discovery_context=f"auto-bootstrapped {_profile_discovery_context(parent_profile_id, source_family=source_family)}",
            discovery_dimensions=_profile_discovery_dimensions(parent_profile_id, source_family=source_family),
            parent_profile_id=parent_profile_id,
            built_in=False,
            feature_tags=tuple(sorted(name for name, count in features.items() if count > 0)),
            metadata={"bootstrap": diagnostics},
        )
    )
    return TaskProfileResolution(
        profile=profile,
        strategy=strategy,
        bootstrap_used=True,
        iterations_run=len(refinement_history),
        diagnostics=diagnostics,
    )


def resolve_or_bootstrap_task_profile(
    examples: Sequence[ExampleRecord],
    *,
    explicit: Optional[str] = None,
    bootstrap_iterations: int = 3,
) -> TaskProfileResolution:
    if explicit and explicit.strip().lower() not in {"auto", "bootstrap"}:
        profile = get_task_profile(explicit.strip())
        return TaskProfileResolution(
            profile=profile,
            strategy=get_contrast_strategy(profile.contrast_strategy_id),
            bootstrap_used=False,
            iterations_run=0,
            diagnostics={"reason": "explicit_profile"},
        )

    inferred_static = [infer_task_profile_id_from_text(_example_blob(example)) for example in examples]
    features = _feature_counts(examples)
    scores = _score_builtin_profiles(examples, features)
    if not _should_bootstrap(explicit=explicit, inferred_static_profiles=inferred_static, scores=scores):
        parent_profile_id = _pick_parent_profile(scores)
        profile = get_task_profile(parent_profile_id)
        return TaskProfileResolution(
            profile=profile,
            strategy=get_contrast_strategy(profile.contrast_strategy_id),
            bootstrap_used=False,
            iterations_run=0,
            diagnostics={
                "reason": "confident_builtin_match",
                "score_by_profile": dict(sorted(scores.items())),
                "inferred_static_profiles": inferred_static,
            },
        )
    return bootstrap_task_profile(
        examples,
        explicit=explicit,
        bootstrap_iterations=bootstrap_iterations,
    )
