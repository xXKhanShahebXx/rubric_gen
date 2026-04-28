"""
Independent-solver verifier for ``livebench-reasoning`` JudgeBench examples.

Many livebench-reasoning failures look like this: candidate A and candidate B both produce
plausible step-by-step reasoning ending in a structured short answer ("yes, yes, no",
"square pyramid", "Liam, Max, Isabella"), and the existing exact_answer_verifier and
reasoning_process_verifier can't tell which is right because both candidates have similar
format quality.

This module mirrors the math independent solver and the MMLU independent answerer: make
ONE additional LLM call asking the model to solve the puzzle WITHOUT seeing either
candidate response, then compare the solver's structured answer against canonicalised
answers extracted from each candidate. When exactly one candidate matches, fire HIGH
confidence in that direction.

The verifier:

- Only fires when ``policy.reasoning_independent_solver_enabled`` is set and the example is
  ``livebench-reasoning``.
- Takes a ``reasoning_solver_model`` policy field so the solver can be Claude / Gemini etc.
  (the FINAL judge call remains GPT-4o; the solver is a tool that produces a candidate
  answer for the verifier to compare against).
- Caches via the standard ``JsonlCache``.
- Abstains when neither, both, or no candidates produce a recognisable answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from rubric_gen.llm_client import LLMRouter
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


SOLVER_PROMPT_VERSION = "reasoning_independent_solver_v1"


_SYSTEM_PROMPT = """You are an expert problem-solver tackling a logic / reasoning puzzle.

Rules:
- Read only the question. You will NOT see other candidate responses.
- Reason briefly (3-8 short steps) to deduce the answer.
- End with a single line of the exact form: ``FINAL_ANSWER: <answer>``
  - <answer> should be the SHORT canonical answer expected by the question.
  - For yes/no sequences, format as ``yes, no, yes`` (comma-separated, lowercase, no
    extra words).
  - For "Type your answer in alphabetical order" or single-word answers, give just the
    short string.
  - For "list the names" answers, format as ``Alice, Bob, Carol`` in the order requested.
  - For shape / object answers, give the simplest noun phrase ("square pyramid",
    "pentagon", "cube").
- If you genuinely cannot decide, end with: ``FINAL_ANSWER: UNKNOWN``"""


_FINAL_ANSWER_RE = re.compile(
    r"FINAL[_\s]?ANSWER\s*[:\-]\s*([^\n]+)",
    re.IGNORECASE,
)
_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*")
_TAIL_LINE_RE = re.compile(r"^[^\n]+$", re.MULTILINE)
_PUNCT_RE = re.compile(r"[\s\.\!\?\;\:]+")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ReasoningSolverResult:
    raw_answer: str
    parsed_answer: str
    canonical_answer: str
    cache_hit: bool
    parse_error: str = ""


@dataclass(frozen=True)
class ReasoningIndependentSolverOutcome:
    triggered: bool
    recommended_decision: str
    confidence: str
    reason: str
    solver_canonical_answer: str
    a_canonical_answer: str
    b_canonical_answer: str
    a_match: bool
    b_match: bool
    schema: str = "reasoning_independent_solver_outcome_v1"


def _canonicalize_answer(raw: str) -> str:
    """
    Normalize a free-form answer string for comparison.

    Steps:
    - Strip surrounding whitespace and ``**`` bold markers.
    - Lowercase.
    - Drop trailing punctuation (``.``, ``!``, ``?``).
    - Collapse internal whitespace.
    - Normalize ``yes,no, yes`` -> ``yes, no, yes`` (canonical comma-space).
    """
    if not raw:
        return ""
    s = raw.strip()
    s = s.replace("*", "")
    s = s.lower()
    s = s.strip()
    while s and s[-1] in ".,!?;:":
        s = s[:-1]
    s = _WS_RE.sub(" ", s).strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) > 1:
        s = ", ".join(parts)
    return s


def _parse_solver_answer(raw_text: str) -> str:
    if not raw_text:
        return ""
    m = _FINAL_ANSWER_RE.search(raw_text)
    if not m:
        return ""
    val = m.group(1).strip()
    if val.lower().startswith("unknown"):
        return ""
    return _canonicalize_answer(val)


def _extract_candidate_answer(response_text: str) -> str:
    """
    Extract a short canonical answer from a free-form reasoning response.

    Strategy:
    1. ``FINAL_ANSWER: ...`` marker (rare in candidates but possible).
    2. Last bolded text near the end of the response (``**yes, yes, no**``,
       ``**square pyramid**``).
    3. Last non-empty line.
    """
    if not response_text:
        return ""
    m = _FINAL_ANSWER_RE.search(response_text)
    if m:
        return _canonicalize_answer(m.group(1))
    bold_matches = _BOLD_RE.findall(response_text)
    if bold_matches:
        for candidate in reversed(bold_matches):
            cleaned = candidate.strip()
            if not cleaned:
                continue
            if any(
                cleaned.lower().startswith(prefix)
                for prefix in (
                    "answer",
                    "final answer",
                    "conclusion",
                    "bold conclusion",
                    "answers",
                )
            ):
                continue
            return _canonicalize_answer(cleaned)
    lines = [ln.strip() for ln in response_text.splitlines() if ln.strip()]
    if lines:
        return _canonicalize_answer(lines[-1])
    return ""


def _run_solver_single(
    *,
    question: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    sample_index: int = 0,
    temperature: float = 0.0,
) -> ReasoningSolverResult:
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
            return ReasoningSolverResult(
                raw_answer="",
                parsed_answer="",
                canonical_answer="",
                cache_hit=False,
                parse_error=f"router_error:{type(exc).__name__}",
            )
        raw_text = response.raw_text or response.text or ""
        if cache and cache.enabled:
            cache.set(
                cache_key,
                {"raw_response": raw_text, "kind": "reasoning_independent_solver"},
            )
    parsed = _parse_solver_answer(raw_text)
    canonical = parsed
    if not parsed:
        return ReasoningSolverResult(
            raw_answer=raw_text,
            parsed_answer="",
            canonical_answer="",
            cache_hit=cache_hit,
            parse_error="no_final_answer_marker",
        )
    return ReasoningSolverResult(
        raw_answer=raw_text,
        parsed_answer=parsed,
        canonical_answer=canonical,
        cache_hit=cache_hit,
        parse_error="",
    )


def run_reasoning_solver(
    *,
    question: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    samples: int = 1,
    temperature: float = 0.0,
) -> ReasoningSolverResult:
    samples = max(1, int(samples or 1))
    if samples == 1:
        return _run_solver_single(
            question=question,
            model_spec=model_spec,
            router=router,
            cache=cache,
            sample_index=0,
            temperature=temperature,
        )
    results: List[ReasoningSolverResult] = []
    for idx in range(samples):
        sample_temp = 0.0 if idx == 0 else max(temperature, 0.5)
        results.append(
            _run_solver_single(
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
        if r.canonical_answer:
            votes[r.canonical_answer] = votes.get(r.canonical_answer, 0) + 1
    if not votes:
        primary = results[0]
        return ReasoningSolverResult(
            raw_answer=primary.raw_answer,
            parsed_answer="",
            canonical_answer="",
            cache_hit=all(r.cache_hit for r in results),
            parse_error="majority_vote_no_answers",
        )
    sorted_votes = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, winner_count = sorted_votes[0]
    runner_up = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    if winner_count == runner_up:
        primary = results[0]
        return ReasoningSolverResult(
            raw_answer=primary.raw_answer,
            parsed_answer="",
            canonical_answer="",
            cache_hit=all(r.cache_hit for r in results),
            parse_error="majority_vote_inconclusive",
        )
    matching = next(r for r in results if r.canonical_answer == winner)
    return ReasoningSolverResult(
        raw_answer=matching.raw_answer,
        parsed_answer=matching.parsed_answer,
        canonical_answer=winner,
        cache_hit=all(r.cache_hit for r in results),
        parse_error="",
    )


def evaluate_reasoning_independent_solver(
    *,
    question: str,
    response_a: str,
    response_b: str,
    model_spec: ModelSpec,
    router: LLMRouter,
    cache: Optional[JsonlCache],
    samples: int = 1,
    temperature: float = 0.0,
) -> ReasoningIndependentSolverOutcome:
    solver = run_reasoning_solver(
        question=question,
        model_spec=model_spec,
        router=router,
        cache=cache,
        samples=samples,
        temperature=temperature,
    )
    a_canon = _extract_candidate_answer(response_a)
    b_canon = _extract_candidate_answer(response_b)
    if not solver.canonical_answer:
        return ReasoningIndependentSolverOutcome(
            triggered=False,
            recommended_decision="",
            confidence="low",
            reason=solver.parse_error or "solver_no_answer",
            solver_canonical_answer="",
            a_canonical_answer=a_canon,
            b_canonical_answer=b_canon,
            a_match=False,
            b_match=False,
        )
    a_match = bool(a_canon) and a_canon == solver.canonical_answer
    b_match = bool(b_canon) and b_canon == solver.canonical_answer
    if a_match and not b_match:
        return ReasoningIndependentSolverOutcome(
            triggered=True,
            recommended_decision="A>B",
            confidence="high",
            reason="a_matches_independent_solver",
            solver_canonical_answer=solver.canonical_answer,
            a_canonical_answer=a_canon,
            b_canonical_answer=b_canon,
            a_match=True,
            b_match=False,
        )
    if b_match and not a_match:
        return ReasoningIndependentSolverOutcome(
            triggered=True,
            recommended_decision="B>A",
            confidence="high",
            reason="b_matches_independent_solver",
            solver_canonical_answer=solver.canonical_answer,
            a_canonical_answer=a_canon,
            b_canonical_answer=b_canon,
            a_match=False,
            b_match=True,
        )
    if a_match and b_match:
        return ReasoningIndependentSolverOutcome(
            triggered=False,
            recommended_decision="",
            confidence="medium",
            reason="both_match_solver",
            solver_canonical_answer=solver.canonical_answer,
            a_canonical_answer=a_canon,
            b_canonical_answer=b_canon,
            a_match=True,
            b_match=True,
        )
    return ReasoningIndependentSolverOutcome(
        triggered=False,
        recommended_decision="",
        confidence="low",
        reason="neither_matches_solver",
        solver_canonical_answer=solver.canonical_answer,
        a_canonical_answer=a_canon,
        b_canonical_answer=b_canon,
        a_match=False,
        b_match=False,
    )
