"""
Independent-answerer verifier for ``mmlu-pro`` JudgeBench examples.

The diagnostic on the final v3 runs identified ``mmlu-pro`` as the largest pool of intractable
pairs: 38 of 87 mmlu-pro pairs were wrong on every policy variant (44% intractable). These are
factual / domain-specific MCQ questions where the rubric judge can't score correctness because
both candidates produce plausible reasoning that ends in a letter.

This verifier mirrors the math independent solver: make ONE GPT-4o call asking the model to
solve the MCQ from scratch (no candidate responses visible), extract its chosen letter, then
compare to each candidate's stated final letter. When exactly one candidate's letter matches
the solver's letter, the verifier fires HIGH confidence in that direction.

The verifier:

- Only fires when ``policy.mmlu_independent_answerer_enabled`` is set and the example is
  ``mmlu-pro``.
- Optionally runs N>1 samples and majority-votes the solver's chosen letter (mirrors the math
  solver self-consistency mode).
- Caches via the standard ``JsonlCache`` so re-runs share calls.
- Abstains (no decision, ``confidence=low``) when neither or both candidates match.

Cost per blind-350 run: ~87 GPT-4o calls (one per mmlu-pro pair) when N=1; multiplied by N
when sampling. With caching across runs (the new shared-cache directory) the cost is paid once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


SOLVER_PROMPT_VERSION = "mmlu_independent_answerer_v1"


_SYSTEM_PROMPT = """You are answering a multiple-choice question independently.

Rules:
- Read only the question and answer options. You will NOT see candidate responses.
- Reason briefly (2-5 short steps) about which option is correct.
- End with a single line of the exact form: ``FINAL_ANSWER: <letter>``
  - <letter> is a single uppercase letter A-J corresponding to your chosen option.
- If the prompt asks for a "repeated letter" output (e.g., 'AAAAA'), still output just the
  single letter on the FINAL_ANSWER line.
- If you genuinely cannot decide, end with: ``FINAL_ANSWER: UNKNOWN``"""


_FINAL_ANSWER_RE = re.compile(
    r"FINAL[_\s]?ANSWER\s*[:\-]\s*([A-Za-z])\b",
    re.IGNORECASE,
)
_REPEATED_LETTER_RE = re.compile(r"\b([A-Ja-j])\1{4,}\b")
_PARENTHESIZED_LETTER_RE = re.compile(r"\(([A-Ja-j])\)")
_FINAL_ANSWER_LINE_RE = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:\-])\s*[:\-]?\s*\(?([A-Ja-j])\)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IndependentAnswererResult:
    raw_answer: str
    parsed_letter: str
    cache_hit: bool
    parse_error: str = ""


@dataclass(frozen=True)
class MMLUIndependentAnswererOutcome:
    triggered: bool
    recommended_decision: str
    confidence: str
    reason: str
    solver_letter: str
    a_letter: str
    b_letter: str
    a_match: bool
    b_match: bool
    schema: str = "mmlu_independent_answerer_outcome_v1"


def _parse_solver_letter(text: str) -> str:
    if not text:
        return ""
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        letter = match.group(1).upper()
        if letter in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}:
            return letter
    return ""


def _extract_candidate_letter(response_text: str) -> str:
    """
    Extract the candidate's final answer letter from a free-form response.

    Order of preference:
    1. ``FINAL_ANSWER: <letter>`` marker.
    2. Repeated-letter pattern (``AAAAA``, ``BBBBB``, ...).
    3. ``Final answer: (X)`` / ``The answer is X``.
    4. Last parenthesized letter ``(X)`` near the end of the response.
    """
    if not response_text:
        return ""
    match = _FINAL_ANSWER_RE.search(response_text)
    if match:
        return match.group(1).upper()
    repeated = _REPEATED_LETTER_RE.search(response_text)
    if repeated:
        return repeated.group(1).upper()
    final_marker = _FINAL_ANSWER_LINE_RE.search(response_text)
    if final_marker:
        return final_marker.group(1).upper()
    last_500 = response_text[-500:]
    paren_letters = _PARENTHESIZED_LETTER_RE.findall(last_500)
    if paren_letters:
        return paren_letters[-1].upper()
    return ""


def _run_independent_answerer_single(
    *,
    question: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    sample_index: int = 0,
    temperature: float = 0.0,
) -> IndependentAnswererResult:
    payload: Dict[str, Any] = {
        "prompt_version": SOLVER_PROMPT_VERSION,
        "model": f"{model_spec.provider}:{model_spec.model}",
        "question_hash": stable_hash(question or ""),
    }
    if sample_index > 0 or temperature > 0.0:
        payload["sample_index"] = int(sample_index)
        payload["temperature"] = round(float(temperature), 4)
    cache_key = make_cache_key(SOLVER_PROMPT_VERSION, payload)
    raw_text = ""
    cache_hit = False
    if cache and cache.enabled:
        cache.load()
        hit = cache.get(cache_key)
        if hit and isinstance(hit.get("raw_response"), str):
            raw_text = hit["raw_response"]
            cache_hit = True
    if not raw_text:
        try:
            response = router.generate(
                model_spec,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=str(question or ""),
                temperature=float(temperature),
            )
        except Exception as exc:
            return IndependentAnswererResult(
                raw_answer="",
                parsed_letter="",
                cache_hit=False,
                parse_error=f"router_error:{type(exc).__name__}",
            )
        raw_text = response.raw_text or response.text or ""
        if cache and cache.enabled:
            cache.set(
                cache_key,
                {"raw_response": raw_text, "kind": "mmlu_independent_answerer"},
            )
    letter = _parse_solver_letter(raw_text)
    parse_error = "" if letter else "no_letter_parsed"
    return IndependentAnswererResult(
        raw_answer=raw_text,
        parsed_letter=letter,
        cache_hit=cache_hit,
        parse_error=parse_error,
    )


def run_independent_answerer(
    *,
    question: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    samples: int = 1,
    temperature: float = 0.0,
) -> IndependentAnswererResult:
    samples = max(1, int(samples or 1))
    if samples == 1:
        return _run_independent_answerer_single(
            question=question,
            model_spec=model_spec,
            router=router,
            cache=cache,
            sample_index=0,
            temperature=temperature,
        )
    results: List[IndependentAnswererResult] = []
    for idx in range(samples):
        sample_temp = 0.0 if idx == 0 else max(temperature, 0.5)
        results.append(
            _run_independent_answerer_single(
                question=question,
                model_spec=model_spec,
                router=router,
                cache=cache,
                sample_index=idx,
                temperature=sample_temp,
            )
        )
    votes: Dict[str, int] = {}
    for r in results:
        if r.parsed_letter:
            votes[r.parsed_letter] = votes.get(r.parsed_letter, 0) + 1
    if not votes:
        primary = results[0]
        return IndependentAnswererResult(
            raw_answer=primary.raw_answer,
            parsed_letter="",
            cache_hit=all(r.cache_hit for r in results),
            parse_error="majority_vote_no_letters",
        )
    sorted_votes = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, winner_count = sorted_votes[0]
    runner_up_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    if winner_count == runner_up_count:
        primary = results[0]
        return IndependentAnswererResult(
            raw_answer=primary.raw_answer,
            parsed_letter="",
            cache_hit=all(r.cache_hit for r in results),
            parse_error="majority_vote_inconclusive",
        )
    matching = next(r for r in results if r.parsed_letter == winner)
    return IndependentAnswererResult(
        raw_answer=matching.raw_answer,
        parsed_letter=winner,
        cache_hit=all(r.cache_hit for r in results),
        parse_error="",
    )


def evaluate_mmlu_independent_answerer(
    *,
    question: str,
    response_a: str,
    response_b: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    samples: int = 1,
    temperature: float = 0.0,
    secondary_model_spec: Optional[ModelSpec] = None,
    secondary_samples: int = 1,
    secondary_temperature: float = 0.0,
) -> MMLUIndependentAnswererOutcome:
    """
    Resolve an mmlu-pro pair using an independent-answerer verifier.

    When ``secondary_model_spec`` is provided, the verifier requires BOTH the primary and
    secondary answerers to recommend the same direction before firing. This dual-consensus
    mode trades some override coverage for higher precision; it is appropriate when a
    single strong solver still produces non-trivial false-positive overrides on subtle
    benchmark items.
    """
    solver = run_independent_answerer(
        question=question,
        model_spec=model_spec,
        router=router,
        cache=cache,
        samples=samples,
        temperature=temperature,
    )
    secondary_solver: Optional[IndependentAnswererResult] = None
    if secondary_model_spec is not None:
        secondary_solver = run_independent_answerer(
            question=question,
            model_spec=secondary_model_spec,
            router=router,
            cache=cache,
            samples=secondary_samples,
            temperature=secondary_temperature,
        )
        if (
            secondary_solver.parsed_letter
            and solver.parsed_letter
            and secondary_solver.parsed_letter != solver.parsed_letter
        ):
            return MMLUIndependentAnswererOutcome(
                triggered=False,
                recommended_decision="",
                confidence="low",
                reason=(
                    f"primary_secondary_disagree:"
                    f"{solver.parsed_letter}_vs_{secondary_solver.parsed_letter}"
                ),
                solver_letter=solver.parsed_letter,
                a_letter=_extract_candidate_letter(response_a),
                b_letter=_extract_candidate_letter(response_b),
                a_match=False,
                b_match=False,
            )
    a_letter = _extract_candidate_letter(response_a)
    b_letter = _extract_candidate_letter(response_b)
    if solver.parse_error or not solver.parsed_letter:
        return MMLUIndependentAnswererOutcome(
            triggered=False,
            recommended_decision="",
            confidence="low",
            reason=solver.parse_error or "solver_no_letter",
            solver_letter="",
            a_letter=a_letter,
            b_letter=b_letter,
            a_match=False,
            b_match=False,
        )
    a_match = bool(a_letter) and a_letter == solver.parsed_letter
    b_match = bool(b_letter) and b_letter == solver.parsed_letter
    if a_match and not b_match:
        return MMLUIndependentAnswererOutcome(
            triggered=True,
            recommended_decision="A>B",
            confidence="high",
            reason="a_matches_independent_answerer",
            solver_letter=solver.parsed_letter,
            a_letter=a_letter,
            b_letter=b_letter,
            a_match=True,
            b_match=False,
        )
    if b_match and not a_match:
        return MMLUIndependentAnswererOutcome(
            triggered=True,
            recommended_decision="B>A",
            confidence="high",
            reason="b_matches_independent_answerer",
            solver_letter=solver.parsed_letter,
            a_letter=a_letter,
            b_letter=b_letter,
            a_match=False,
            b_match=True,
        )
    if a_match and b_match:
        return MMLUIndependentAnswererOutcome(
            triggered=False,
            recommended_decision="",
            confidence="medium",
            reason="both_match_solver",
            solver_letter=solver.parsed_letter,
            a_letter=a_letter,
            b_letter=b_letter,
            a_match=True,
            b_match=True,
        )
    return MMLUIndependentAnswererOutcome(
        triggered=False,
        recommended_decision="",
        confidence="low",
        reason="neither_matches_solver",
        solver_letter=solver.parsed_letter,
        a_letter=a_letter,
        b_letter=b_letter,
        a_match=False,
        b_match=False,
    )
