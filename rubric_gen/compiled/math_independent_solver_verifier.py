"""
Independent-solver verifier for ``livebench-math`` JudgeBench examples.

The existing exact-answer verifier needs the reference answer to fire (and even when present, it
often misses on math because both candidate responses fail the rubric in different ways). The
v2 blind-350 ablation showed math at 67.9% even with the holistic judge disabled — a real ceiling
that the rubric-based judges cannot crack alone.

This module adds a *math-specific* verifier that runs ONE additional GPT-4o call per math pair,
asking the model to solve the problem **without seeing either candidate response**. The
independently-derived final value is then compared against the values extracted from A and B; if
exactly one matches, the verifier fires with high confidence in that direction.

The verifier is opt-in via the policy field ``math_independent_solver_enabled`` so legacy runs
are unaffected. It only triggers on `livebench-math`. Caching uses the standard JsonlCache so
re-runs on the same pair are free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from rubric_gen.llm_client import LLMRouter
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


SOLVER_PROMPT_VERSION = "math_independent_solver_v1"

_SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem step by step and report your final answer.

Rules:
- Show your reasoning briefly (2-6 short steps).
- Do NOT speculate about other people's responses; you are solving the problem from scratch.
- End with a single line of the exact form: ``FINAL_ANSWER: <value>``
  - <value> should be the simplest exact form (integer, fraction in lowest terms, or short closed form).
  - For multiple-choice problems, <value> is just the option letter (A-E or whatever the problem uses).
  - For repeated-letter format problems, <value> is the repeated string (e.g., AAAAA).
- If the problem has multiple parts, give the answer to the LAST part requested.
- If you genuinely cannot solve it, end with: ``FINAL_ANSWER: UNKNOWN``"""


_FINAL_ANSWER_RE = re.compile(
    r"FINAL[_\s]?ANSWER\s*[:\-]\s*([^\n]+)",
    re.IGNORECASE,
)


_NUMERIC_TOKEN_RE = re.compile(r"-?\d+(?:[/.]\d+)?")
_REPEATED_LETTER_RE = re.compile(r"^([A-Za-z])\1{2,}$")
_REPEATED_LETTER_SCAN_RE = re.compile(r"\b([A-Za-z])\1{4,}\b")
_OPTION_LETTER_RE = re.compile(r"^[A-Za-z]$")
_BOXED_RE = re.compile(r"\\boxed\s*\{([^}]+)\}")
_LEADING_INSTRUCTION_RE = re.compile(
    r"^(?:thus|so|hence|therefore|the\s+answer\s+is|final\s+answer\s+is|"
    r"final\s+answer\s*[:\-]|answer\s*[:\-]|"
    r"thus\s+final\s+concluding\s+answer\s+is|repeat\s+the\s+answer\s+\w+\s+times[^:]*:)\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IndependentSolverResult:
    raw_answer: str
    parsed_answer: str
    canonical_answer: str
    cache_hit: bool
    parse_error: str = ""


@dataclass(frozen=True)
class MathIndependentSolverOutcome:
    triggered: bool
    recommended_decision: str
    confidence: str
    reason: str
    solver_canonical_answer: str
    a_canonical_answer: str
    b_canonical_answer: str
    a_match: bool
    b_match: bool
    schema: str = "math_independent_solver_outcome_v1"


def _strip_punctuation(value: str) -> str:
    return value.strip().strip(".,;:!?\"'`()[]{}\\$").strip()


def _canonicalize_answer(raw: str) -> str:
    """
    Reduce a free-form math answer string to a comparable canonical form.

    Order of preference:
    1. ``\\boxed{...}`` LaTeX content (very common in math responses).
    2. Repeated-letter MCQ outputs (``aaaaa``, ``ccccc``) anywhere in the string.
    3. Numeric values (with fraction-in-lowest-terms and trailing-zero normalization).
    4. Lowercased / trimmed string after stripping common leading-instruction prefixes.
    """
    if not raw:
        return ""
    text = str(raw)
    boxed = _BOXED_RE.search(text)
    if boxed:
        text = boxed.group(1)
    text = _LEADING_INSTRUCTION_RE.sub("", text.strip()).strip()
    cleaned = _strip_punctuation(text).lower()
    if not cleaned:
        return ""
    repeated_inline = _REPEATED_LETTER_SCAN_RE.search(cleaned)
    if repeated_inline:
        letter = repeated_inline.group(1)
        return letter * len(repeated_inline.group(0))
    if _REPEATED_LETTER_RE.match(cleaned):
        return cleaned
    if _OPTION_LETTER_RE.match(cleaned):
        return cleaned
    numerics = _NUMERIC_TOKEN_RE.findall(cleaned)
    if numerics:
        token = numerics[-1]
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            try:
                num_int = int(numerator)
                den_int = int(denominator)
                if den_int != 0:
                    from math import gcd

                    g = gcd(abs(num_int), abs(den_int))
                    return f"{num_int // g}/{den_int // g}"
            except Exception:
                pass
            return token
        try:
            value = float(token)
            if value == int(value):
                return str(int(value))
            return str(round(value, 8))
        except Exception:
            return token
    return cleaned


def _parse_final_answer_line(text: str) -> str:
    if not text:
        return ""
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return _strip_punctuation(match.group(1))
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return _strip_punctuation(lines[-1]) if lines else ""


def _extract_candidate_final_value(response_text: str) -> str:
    if not response_text:
        return ""
    match = _FINAL_ANSWER_RE.search(response_text)
    if match:
        return _strip_punctuation(match.group(1))
    boxed_matches = _BOXED_RE.findall(response_text)
    if boxed_matches:
        return _strip_punctuation(boxed_matches[-1])
    repeated_scan = list(_REPEATED_LETTER_SCAN_RE.finditer(response_text))
    if repeated_scan:
        last = repeated_scan[-1]
        return last.group(0)
    final_marker_re = re.compile(
        r"(?:final\s+answer|the\s+answer\s+is|final:|answer:)\s*[:\-]?\s*([^\n.]+)",
        re.IGNORECASE,
    )
    match = final_marker_re.search(response_text)
    if match:
        return _strip_punctuation(match.group(1))
    last_500 = response_text[-500:]
    all_numerics = _NUMERIC_TOKEN_RE.findall(last_500)
    if all_numerics:
        return all_numerics[-1]
    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return _strip_punctuation(lines[-1])


def _run_independent_solver_single(
    *,
    question: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    sample_index: int = 0,
    temperature: float = 0.0,
) -> IndependentSolverResult:
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
            return IndependentSolverResult(
                raw_answer="",
                parsed_answer="",
                canonical_answer="",
                cache_hit=False,
                parse_error=f"router_error:{type(exc).__name__}",
            )
        raw_text = response.raw_text or response.text or ""
        if cache and cache.enabled:
            cache.set(cache_key, {"raw_response": raw_text, "kind": "math_independent_solver"})
    parsed = _parse_final_answer_line(raw_text)
    canonical = _canonicalize_answer(parsed)
    parse_error = ""
    if not parsed:
        parse_error = "no_final_answer_parsed"
    elif canonical.lower() == "unknown":
        parse_error = "solver_returned_unknown"
    return IndependentSolverResult(
        raw_answer=raw_text,
        parsed_answer=parsed,
        canonical_answer=canonical,
        cache_hit=cache_hit,
        parse_error=parse_error,
    )


def run_independent_solver(
    *,
    question: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    samples: int = 1,
    temperature: float = 0.0,
) -> IndependentSolverResult:
    """
    Run the independent solver. When ``samples > 1`` runs N calls and majority-votes on the
    canonical answer; the first call uses ``temperature``, subsequent calls bump it slightly to
    encourage diversity. The first sample uses ``temperature=0.0`` for cache reuse.

    The returned ``IndependentSolverResult`` reflects the majority canonical answer; when the
    vote is split with no plurality, returns a ``parse_error == "majority_vote_inconclusive"`` so
    callers can fall through to a different signal.
    """
    samples = max(1, int(samples or 1))
    if samples == 1:
        return _run_independent_solver_single(
            question=question,
            model_spec=model_spec,
            router=router,
            cache=cache,
            sample_index=0,
            temperature=temperature,
        )
    results: List[IndependentSolverResult] = []
    for idx in range(samples):
        sample_temp = 0.0 if idx == 0 else max(temperature, 0.5)
        results.append(
            _run_independent_solver_single(
                question=question,
                model_spec=model_spec,
                router=router,
                cache=cache,
                sample_index=idx,
                temperature=sample_temp,
            )
        )
    canonical_votes: Dict[str, int] = {}
    for r in results:
        if r.parse_error:
            continue
        if not r.canonical_answer:
            continue
        canonical_votes[r.canonical_answer] = canonical_votes.get(r.canonical_answer, 0) + 1
    if not canonical_votes:
        primary = results[0]
        return IndependentSolverResult(
            raw_answer=primary.raw_answer,
            parsed_answer=primary.parsed_answer,
            canonical_answer="",
            cache_hit=all(r.cache_hit for r in results),
            parse_error="majority_vote_no_valid_answers",
        )
    sorted_votes = sorted(canonical_votes.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, winner_count = sorted_votes[0]
    runner_up_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    if winner_count == runner_up_count:
        primary = results[0]
        return IndependentSolverResult(
            raw_answer=primary.raw_answer,
            parsed_answer=primary.parsed_answer,
            canonical_answer="",
            cache_hit=all(r.cache_hit for r in results),
            parse_error="majority_vote_inconclusive",
        )
    matching = next(r for r in results if r.canonical_answer == winner)
    return IndependentSolverResult(
        raw_answer=matching.raw_answer,
        parsed_answer=matching.parsed_answer,
        canonical_answer=winner,
        cache_hit=all(r.cache_hit for r in results),
        parse_error="",
    )


def evaluate_math_independent_solver(
    *,
    question: str,
    response_a: str,
    response_b: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    samples: int = 1,
    temperature: float = 0.0,
    use_sympy: bool = False,
) -> MathIndependentSolverOutcome:
    solver = run_independent_solver(
        question=question,
        model_spec=model_spec,
        router=router,
        cache=cache,
        samples=samples,
        temperature=temperature,
    )
    if solver.parse_error or not solver.canonical_answer:
        return MathIndependentSolverOutcome(
            triggered=False,
            recommended_decision="",
            confidence="low",
            reason=solver.parse_error or "solver_no_answer",
            solver_canonical_answer="",
            a_canonical_answer=_canonicalize_answer(_extract_candidate_final_value(response_a)),
            b_canonical_answer=_canonicalize_answer(_extract_candidate_final_value(response_b)),
            a_match=False,
            b_match=False,
        )
    a_raw = _extract_candidate_final_value(response_a)
    b_raw = _extract_candidate_final_value(response_b)
    a_value = _canonicalize_answer(a_raw)
    b_value = _canonicalize_answer(b_raw)
    a_match = bool(a_value) and a_value == solver.canonical_answer
    b_match = bool(b_value) and b_value == solver.canonical_answer
    if use_sympy:
        from rubric_gen.compiled.sympy_math_verifier import equivalent_strings

        if not a_match and a_value and equivalent_strings(a_raw, solver.canonical_answer):
            a_match = True
        if not b_match and b_value and equivalent_strings(b_raw, solver.canonical_answer):
            b_match = True
    if a_match and not b_match:
        return MathIndependentSolverOutcome(
            triggered=True,
            recommended_decision="A>B",
            confidence="high",
            reason="a_matches_independent_solver",
            solver_canonical_answer=solver.canonical_answer,
            a_canonical_answer=a_value,
            b_canonical_answer=b_value,
            a_match=True,
            b_match=False,
        )
    if b_match and not a_match:
        return MathIndependentSolverOutcome(
            triggered=True,
            recommended_decision="B>A",
            confidence="high",
            reason="b_matches_independent_solver",
            solver_canonical_answer=solver.canonical_answer,
            a_canonical_answer=a_value,
            b_canonical_answer=b_value,
            a_match=False,
            b_match=True,
        )
    if a_match and b_match:
        return MathIndependentSolverOutcome(
            triggered=False,
            recommended_decision="",
            confidence="medium",
            reason="both_match_solver",
            solver_canonical_answer=solver.canonical_answer,
            a_canonical_answer=a_value,
            b_canonical_answer=b_value,
            a_match=True,
            b_match=True,
        )
    return MathIndependentSolverOutcome(
        triggered=False,
        recommended_decision="",
        confidence="low",
        reason="neither_matches_solver",
        solver_canonical_answer=solver.canonical_answer,
        a_canonical_answer=a_value,
        b_canonical_answer=b_value,
        a_match=False,
        b_match=False,
    )
