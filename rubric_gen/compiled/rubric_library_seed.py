"""
Seed rubric-library corpus and deterministic proposer.

This module hard-codes a compact, paper-grounded set of seed criteria targeted at the JudgeBench
failure clusters identified in ``docs/workflows/judgebench_350_recovery_handoff.md`` (reasoning
``person_right_*`` profiles, MMLU-Pro final-answer format, LiveBench-Math final-value correctness,
LiveCodeBench behavior). Every seed criterion is marked with an explicit ``source_tag`` so it is
traceable in the final library.

The seed is used when:
- running unit tests for the v2 pipeline, and
- bootstrapping ``artifacts/rubric_library/v1/library.json`` without an LLM distillation run.

A real production build calls :func:`rubric_gen.compiled.rubric_library_builder.distill_library`
with a multi-model proposer wired to GPT-4o / Claude / Gemini on HelpSteer3 / UltraFeedback / PPE
inputs. This seed acts as a high-quality fallback and as a floor the distilled library must beat.
"""

from __future__ import annotations

from typing import Dict, List, Mapping

from rubric_gen.compiled.rubric_library_builder import (
    ExternalPreferencePair,
    ProposedCriterion,
)


SEED_SOURCE_TAG = "hand_curated_seed_v1"


def _mmlu_pro_seeds() -> List[ProposedCriterion]:
    return [
        ProposedCriterion(
            dimension="final_answer_correctness",
            label="Selected option matches the correct answer exactly",
            requirement=(
                "The response states a single final option (letter and/or value) that matches the "
                "correct answer; it does not hedge between multiple choices or select the wrong letter "
                "even when intermediate reasoning is partially right."
            ),
            severity_tier="hard_gate",
            focus_kind="final_answer",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="final_answer_correctness",
            label="Final answer letter and final answer value agree",
            requirement=(
                "When the response names both an option letter and an option value, they refer to "
                "the same choice. Mismatched letter/value pairs count as unmet."
            ),
            severity_tier="high",
            focus_kind="option_consistency",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="format_communication",
            label="Final answer formatted as requested",
            requirement=(
                "If the prompt asks for a specific output syntax (single letter, repeated letter, "
                "single word, etc.), the response's final answer line conforms to that syntax exactly."
            ),
            severity_tier="high",
            focus_kind="final_answer_format",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="reasoning_faithfulness",
            label="Reasoning supports the stated final option",
            requirement=(
                "The derivation in the response concludes with the selected option and does not "
                "contradict earlier reasoning steps or the task's exclusion criteria."
            ),
            severity_tier="medium",
            focus_kind="reasoning",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="grounding",
            label="Does not invent facts absent from the question",
            requirement=(
                "The response uses only information present in the question stem and answer options "
                "when justifying its choice; it does not fabricate numbers, laws, or citations."
            ),
            severity_tier="medium",
            focus_kind="grounding",
            source_tag=SEED_SOURCE_TAG,
        ),
    ]


def _livebench_reasoning_seeds() -> List[ProposedCriterion]:
    return [
        ProposedCriterion(
            dimension="assignment_completeness",
            label="All required slots are filled in the solution",
            requirement=(
                "When the puzzle asks for an assignment of entities to positions or attributes, the "
                "response fills every slot with a committed value and does not leave any as "
                "'UNKNOWN', '?', or 'unclear'."
            ),
            severity_tier="hard_gate",
            focus_kind="assignment_completeness",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="clue_consistency",
            label="Final assignment is consistent with every stated clue",
            requirement=(
                "Each clue (adjacency, direction, ordering, parity, exclusivity) is satisfied by the "
                "final assignment. Violated or ignored clues count as unmet."
            ),
            severity_tier="hard_gate",
            focus_kind="clue_consistency",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="exclusivity",
            label="No two entities occupy the same exclusive slot",
            requirement=(
                "When slots are exclusive (e.g. each person has one unique favorite, each position "
                "holds one person), the response does not assign the same slot to multiple entities."
            ),
            severity_tier="high",
            focus_kind="exclusivity",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="conclusion_grounded",
            label="Final answer is derivable from the committed assignment",
            requirement=(
                "The response's final stated answer (the thing actually asked for) can be read off "
                "the solution directly; it does not contradict the intermediate assignment or leave "
                "the solver to infer it."
            ),
            severity_tier="hard_gate",
            focus_kind="final_answer",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="reasoning_faithfulness",
            label="Deductions use only constraints stated in the puzzle",
            requirement=(
                "The reasoning chain references constraints that appear in the puzzle text; it does "
                "not invent extra rules, symmetries, or domain knowledge."
            ),
            severity_tier="medium",
            focus_kind="reasoning",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="contradiction_avoidance",
            label="No step is later retracted or reversed without resolution",
            requirement=(
                "If the response tentatively assigns a value and later flips it, the final section "
                "explicitly reconciles the change; lingering 'actually wait' contradictions count as "
                "unmet."
            ),
            severity_tier="medium",
            focus_kind="contradiction",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="format_communication",
            label="Final answer uses the exact requested syntax",
            requirement=(
                "If the prompt requests a single letter, a list of values, or a sentence stating the "
                "answer, the final line of the response conforms to that request."
            ),
            severity_tier="medium",
            focus_kind="final_answer_format",
            source_tag=SEED_SOURCE_TAG,
        ),
    ]


def _livebench_math_seeds() -> List[ProposedCriterion]:
    return [
        ProposedCriterion(
            dimension="final_answer_correctness",
            label="Final numeric value equals the correct answer",
            requirement=(
                "The final reported number (or algebraic expression) matches the correct answer "
                "exactly after trivial normalization (trailing zeros, equivalent fractions). A value "
                "that is off by even one unit is unmet."
            ),
            severity_tier="hard_gate",
            focus_kind="final_answer",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="constraint_satisfaction",
            label="Solution satisfies every numeric constraint stated in the problem",
            requirement=(
                "Each explicit constraint (ranges, divisibility, coprimality, inequalities) is "
                "satisfied by the final answer and by intermediate quantities on which the answer "
                "depends."
            ),
            severity_tier="high",
            focus_kind="constraint",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="derivation_grounded",
            label="Intermediate steps match the final answer",
            requirement=(
                "The working shown in the response leads to the stated final number; there is no "
                "unexplained jump or mismatch between the last computation and the reported answer."
            ),
            severity_tier="high",
            focus_kind="derivation",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="format_communication",
            label="Answer is reported in the format requested",
            requirement=(
                "If the prompt asks for the answer in a box, a fraction in lowest terms, or a "
                "specific variable expression, the final line conforms. Plain numbers without the "
                "requested format count as unmet."
            ),
            severity_tier="medium",
            focus_kind="final_answer_format",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="arithmetic_correctness",
            label="No arithmetic or algebraic step is wrong",
            requirement=(
                "Individual computations in the derivation (sums, products, simplifications, modular "
                "arithmetic) are correct; errors in any step make the criterion unmet even if the "
                "final answer happens to be right."
            ),
            severity_tier="medium",
            focus_kind="arithmetic",
            source_tag=SEED_SOURCE_TAG,
        ),
    ]


def _livecodebench_seeds() -> List[ProposedCriterion]:
    return [
        ProposedCriterion(
            dimension="behavior_correctness",
            label="Program output matches the required behavior for all stated I/O",
            requirement=(
                "The program produces the exact expected output for every input scenario described "
                "in the prompt (including edge cases given as examples). Off-by-one, wrong format, "
                "or missing cases count as unmet."
            ),
            severity_tier="hard_gate",
            focus_kind="behavior",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="constraint_satisfaction",
            label="Solution honors explicit constraints (complexity, limits, language)",
            requirement=(
                "The code respects all explicit constraints given in the prompt: time/space limits, "
                "required algorithm class, required language/library, I/O format specified by the "
                "problem."
            ),
            severity_tier="high",
            focus_kind="constraint",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="completeness",
            label="Handles all edge cases mentioned or implied by the problem",
            requirement=(
                "Edge cases implicit in the specification (empty input, single element, zero, "
                "duplicates, maximum size, negative numbers) are handled without crashing or "
                "returning wrong answers."
            ),
            severity_tier="high",
            focus_kind="edge_cases",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="grounding",
            label="Solution addresses the problem as specified, not a reworded version",
            requirement=(
                "The code solves the task the prompt describes, not a simpler or more general "
                "variant; inputs, outputs, and invariants match the problem statement exactly."
            ),
            severity_tier="medium",
            focus_kind="grounding",
            source_tag=SEED_SOURCE_TAG,
        ),
    ]


def _generic_seeds() -> List[ProposedCriterion]:
    return [
        ProposedCriterion(
            dimension="instruction_adherence",
            label="Follows every explicit instruction in the prompt",
            requirement=(
                "The response carries out each explicit instruction (format, length, scope, "
                "prohibitions). Partial compliance counts as unmet when a hard requirement is "
                "dropped."
            ),
            severity_tier="high",
            focus_kind="instruction",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="grounding",
            label="Facts and citations are grounded in the prompt or verifiable",
            requirement=(
                "Any factual claim or citation in the response is either drawn from information "
                "given in the prompt or is objectively verifiable; fabricated references are "
                "unmet."
            ),
            severity_tier="high",
            focus_kind="grounding",
            source_tag=SEED_SOURCE_TAG,
        ),
        ProposedCriterion(
            dimension="completeness",
            label="Addresses all parts of a multi-part question",
            requirement=(
                "If the prompt asks multiple distinct questions or components, the response answers "
                "each one. Skipping or merging separate questions into a single vague answer is "
                "unmet."
            ),
            severity_tier="medium",
            focus_kind="completeness",
            source_tag=SEED_SOURCE_TAG,
        ),
    ]


_FAMILY_TO_SEEDS = {
    "mmlu-pro": _mmlu_pro_seeds,
    "livebench-reasoning": _livebench_reasoning_seeds,
    "livebench-math": _livebench_math_seeds,
    "livecodebench": _livecodebench_seeds,
    "generic": _generic_seeds,
}


def seed_proposer(pair: ExternalPreferencePair) -> List[ProposedCriterion]:
    """
    Deterministic proposer that ignores the pair text and always returns the family-appropriate
    seed criteria. Used for smoke tests and to build the bootstrap library. Real distillation runs
    replace this with an LLM-based proposer.
    """
    family = pair.source_family or "generic"
    builder = _FAMILY_TO_SEEDS.get(family, _generic_seeds)
    return builder()


def build_seed_pair_set() -> List[ExternalPreferencePair]:
    """
    Return a deterministic fake pair set covering every target family. This exists so the
    distillation loop runs end-to-end in tests without network access.
    """
    pairs: List[ExternalPreferencePair] = []
    for family, factory in _FAMILY_TO_SEEDS.items():
        pair_tokens = " ".join(
            f"{c.dimension} {c.label} {c.requirement}" for c in factory()
        )
        pairs.append(
            ExternalPreferencePair(
                pair_id=f"seed_{family}",
                prompt=f"Example evaluation prompt for {family}.",
                chosen=pair_tokens,
                rejected="A weaker response that omits the above requirements.",
                source="hand_curated_seed",
                source_family=family,
                focus_kind="seed",
            )
        )
    return pairs


SEED_MANIFEST: Mapping[str, object] = {
    "description": "Hand-curated seed corpus bootstrapping the JudgeBench v2 rubric library.",
    "inline_pairs": [
        {
            "pair_id": p.pair_id,
            "prompt": p.prompt,
            "chosen": p.chosen,
            "rejected": p.rejected,
            "source": p.source,
            "source_family": p.source_family,
            "focus_kind": p.focus_kind,
        }
        for p in build_seed_pair_set()
    ],
}
