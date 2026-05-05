"""Score-tie tiebreaker for pair-preference scoring.

This module supplies a direct LLM pair judge (GPT-4o by default) that the
post-hoc rescorer falls through to when the rubric pipeline assigns
identical aggregate scores to ``pair_response_a`` and ``pair_response_b``
under the headline method (typically ``rrd_whitened_uniform``).

Mirrors the spirit of ``rubric_gen/compiled/judgebench_eval.py``'s
``uniform_breaks_strict_tie`` cascade -- when the strict scorer ties,
fall back to a different signal source, in this case a fresh GPT-4o call
asking which response better answers the question. The judge prompt
mirrors ``scripts/relabel_pair_dataset.py``'s medical-quality criteria
so verdicts are directly comparable to the §2.5 Opus labels.

v2 additions (Tier C of the shard 0 v2 plan):

* ``format_prior_predict`` -- a pure-heuristic pre-judge step that
  resolves score-tied rows when the question makes an explicit format
  request (list / table / code / "answer in one sentence" / "explain in
  detail") and only one of the two responses honours it.  Free, no
  LLM calls.
* ``direct_judge_pair_anti_tie`` -- variant of ``direct_judge_pair`` with
  a strengthened system prompt that explicitly tells the model the
  rubric stage already tied and pushes it to find the smallest
  substantive medical difference rather than declaring another tie.  GPT-4o
  ties on 52.5% of cascade-judge rows under the v1 prompt; the anti-tie
  prompt aims to lower that.

Caching is required: every score-tied row triggers exactly one LLM call,
so a re-run on the same artifacts hits the cache instead of paying again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


TIEBREAKER_PROMPT_VERSION = "pair_tiebreaker_v1"
ANTI_TIE_PROMPT_VERSION = "pair_tiebreaker_anti_tie_v1"


_SYSTEM_PROMPT = """You are an expert medical evaluator comparing two candidate responses to a medical question. Decide which response is better on the following criteria, in priority order:

1. Medical correctness. Are the facts, diagnoses, mechanisms, dosages, and guidelines accurate per current evidence-based medicine?
2. Reasoning soundness. Does the response derive its answer from the right premises, without unjustified leaps or hand-waving?
3. Completeness on the asked question. Does it address what was asked, including relevant caveats and contraindications, without padding the answer with off-topic material?
4. Hallucination resistance. Does it avoid inventing facts, numbers, or guidelines that are not actually correct?

Do NOT reward purely stylistic features (markdown tables, bullet lists, length, formality) unless they materially help correctness or completeness on this specific question.

Output a single JSON object exactly:
{
  "verdict": "A" | "B" | "TIE",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<2-4 sentences citing the specific medical claims that differ>"
}

Use "TIE" only when both responses are equivalently correct and complete on every criterion above. Prefer A or B whenever you can identify a meaningful difference.

Output ONLY the JSON object. Do not include any other text."""


_ANTI_TIE_SYSTEM_PROMPT = """You are an expert medical evaluator acting as a TIEBREAKER between two candidate responses to a medical question. The rubric-based scorer already gave both responses identical aggregate scores -- your job is to find a substantive medical difference that the rubrics missed.

Decide on these criteria, in priority order:

1. Medical correctness on the central claim. Are the diagnoses, mechanisms, dosages, or guidelines factually accurate per current evidence-based medicine? Even one incorrect central claim should decide the verdict.
2. Reasoning soundness. Does each step of the reasoning follow from valid premises, with no unjustified leaps or hand-waving?
3. Completeness on the specific question asked, not on padding the answer with off-topic material.
4. Hallucination resistance. Inventing facts, numbers, or guidelines is never acceptable.

DO NOT reward style (markdown, headers, length, formality) UNLESS it materially improves clarity on the asked question.

VERY IMPORTANT: Both responses already passed the rubric stage with identical scores. That makes "TIE" suspicious -- usually one of them has a small but real medical advantage that the rubrics missed (an extra correct caveat, a more accurate dosage, fewer hand-waves). Look for the smallest concrete medical difference that justifies a verdict, and PICK A or B based on it.

Return TIE only if the two responses make literally the same medical claims with the same reasoning. Pure stylistic differences (length, headers, bullets) are NOT a tie -- in that case, pick the response that is more medically precise, even by a small margin.

Output a single JSON object exactly:
{
  "verdict": "A" | "B" | "TIE",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<2-4 sentences citing the specific medical claim that differs>"
}

Output ONLY the JSON object. Do not include any other text."""


def _build_user_prompt(question: str, response_a: str, response_b: str) -> str:
    return (
        f"<Question>\n{(question or '').strip()}\n</Question>\n\n"
        f"<Response A>\n{(response_a or '').strip()}\n</Response A>\n\n"
        f"<Response B>\n{(response_b or '').strip()}\n</Response B>\n\n"
        "Return your JSON now."
    )


@dataclass(frozen=True)
class TiebreakerOutcome:
    verdict: str  # "a" / "b" / "tie" (lowercase to match pair_correct_label)
    confidence: str
    reasoning: str
    cache_hit: bool
    raw_response: str
    parse_error: str = ""
    router_error: str = ""


def _normalize_verdict(raw: object) -> str:
    text = str(raw or "").strip().upper()
    if text in {"A", "REFERENCE_ANSWER_A", "RESPONSE_A", "ANSWER_A"}:
        return "a"
    if text in {"B", "REFERENCE_ANSWER_B", "RESPONSE_B", "ANSWER_B"}:
        return "b"
    if text in {"TIE", "EQUAL", "DRAW", "EQUIVALENT", "BOTH"}:
        return "tie"
    return ""


def _normalize_confidence(raw: object) -> str:
    text = str(raw or "").strip().lower()
    return text if text in {"high", "medium", "low"} else "medium"


def _cache_key(
    *,
    judge_model: ModelSpec,
    question: str,
    response_a: str,
    response_b: str,
    temperature: float,
    prompt_version: str = TIEBREAKER_PROMPT_VERSION,
    system_prompt: str = _SYSTEM_PROMPT,
) -> str:
    payload = {
        "prompt_version": prompt_version,
        "model": f"{judge_model.provider}:{judge_model.model}",
        "temperature": round(float(temperature), 4),
        "question_hash": stable_hash(question or ""),
        "response_a_hash": stable_hash(response_a or ""),
        "response_b_hash": stable_hash(response_b or ""),
        "system_prompt_hash": stable_hash(system_prompt),
    }
    return make_cache_key(prompt_version, payload)


def _direct_judge_pair_impl(
    *,
    question: str,
    response_a: str,
    response_b: str,
    judge_model: ModelSpec,
    router: Optional[LLMRouter],
    cache: JsonlCache,
    temperature: float,
    dry_run: bool,
    system_prompt: str,
    prompt_version: str,
) -> TiebreakerOutcome:
    """Shared core for direct_judge_pair / direct_judge_pair_anti_tie."""
    key = _cache_key(
        judge_model=judge_model,
        question=question,
        response_a=response_a,
        response_b=response_b,
        temperature=temperature,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
    )
    cached = cache.get(key) if cache.enabled else None
    if cached and isinstance(cached.get("raw_response"), str):
        verdict, confidence, reasoning, parse_error = _parse(cached["raw_response"])
        return TiebreakerOutcome(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            cache_hit=True,
            raw_response=cached["raw_response"],
            parse_error=parse_error,
        )
    if dry_run or router is None:
        return TiebreakerOutcome(
            verdict="",
            confidence="",
            reasoning="",
            cache_hit=False,
            raw_response="",
            parse_error="dry_run" if dry_run else "no_router",
        )
    try:
        response = router.generate(
            judge_model,
            system_prompt=system_prompt,
            user_prompt=_build_user_prompt(question, response_a, response_b),
            temperature=float(temperature),
        )
    except Exception as exc:  # pragma: no cover - exercised via mocks in tests
        return TiebreakerOutcome(
            verdict="",
            confidence="",
            reasoning="",
            cache_hit=False,
            raw_response="",
            parse_error="router_error",
            router_error=f"{type(exc).__name__}: {exc}",
        )
    raw_text = response.raw_text or response.text or ""
    verdict, confidence, reasoning, parse_error = _parse(raw_text)
    if cache.enabled:
        cache.set(
            key,
            {
                "kind": "pair_tiebreaker",
                "prompt_version": prompt_version,
                "raw_response": raw_text,
                "verdict": verdict,
                "confidence": confidence,
                "reasoning": reasoning,
                "parse_error": parse_error,
            },
        )
    return TiebreakerOutcome(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        cache_hit=False,
        raw_response=raw_text,
        parse_error=parse_error,
    )


def direct_judge_pair(
    *,
    question: str,
    response_a: str,
    response_b: str,
    judge_model: ModelSpec,
    router: Optional[LLMRouter],
    cache: JsonlCache,
    temperature: float = 0.0,
    dry_run: bool = False,
) -> TiebreakerOutcome:
    """Run a fresh LLM pair judge on ``(response_a, response_b)`` for ``question``.

    Uses the original v1 system prompt.  Returns a :class:`TiebreakerOutcome`
    with a normalised verdict (``"a"``, ``"b"``, ``"tie"``, or empty on parse
    / router error).  Cache hits short-circuit the LLM call.
    """
    return _direct_judge_pair_impl(
        question=question,
        response_a=response_a,
        response_b=response_b,
        judge_model=judge_model,
        router=router,
        cache=cache,
        temperature=temperature,
        dry_run=dry_run,
        system_prompt=_SYSTEM_PROMPT,
        prompt_version=TIEBREAKER_PROMPT_VERSION,
    )


def direct_judge_pair_anti_tie(
    *,
    question: str,
    response_a: str,
    response_b: str,
    judge_model: ModelSpec,
    router: Optional[LLMRouter],
    cache: JsonlCache,
    temperature: float = 0.0,
    dry_run: bool = False,
) -> TiebreakerOutcome:
    """Tier C2 v2: anti-tie variant of :func:`direct_judge_pair`.

    Uses a strengthened system prompt that explicitly tells the model the
    rubric stage already tied and pushes it to commit on the smallest
    substantive medical difference rather than declaring another tie.
    Caches under a separate key (``ANTI_TIE_PROMPT_VERSION``) so it does
    not collide with v1 cache entries.
    """
    return _direct_judge_pair_impl(
        question=question,
        response_a=response_a,
        response_b=response_b,
        judge_model=judge_model,
        router=router,
        cache=cache,
        temperature=temperature,
        dry_run=dry_run,
        system_prompt=_ANTI_TIE_SYSTEM_PROMPT,
        prompt_version=ANTI_TIE_PROMPT_VERSION,
    )


def _parse(raw_text: str) -> Tuple[str, str, str, str]:
    payload = extract_json_object(raw_text)
    if not isinstance(payload, dict):
        return "", "", "", "no_json_object"
    verdict = _normalize_verdict(payload.get("verdict") or payload.get("decision"))
    if not verdict:
        return "", "", "", "unrecognized_verdict"
    confidence = _normalize_confidence(payload.get("confidence"))
    reasoning = str(payload.get("reasoning") or payload.get("rationale") or "").strip()
    return verdict, confidence, reasoning, ""


def predict_via_score(score_a: Optional[float], score_b: Optional[float]) -> str:
    """Score-based prediction without tiebreak. Returns 'a' / 'b' / 'tie' / ''.

    Mirrors the post-tie-fix ``_pair_preference_outcome`` from
    ``rubric_gen/evaluation/reporting.py``.
    """
    if score_a is None or score_b is None:
        return ""
    if float(score_a) > float(score_b):
        return "a"
    if float(score_b) > float(score_a):
        return "b"
    return "tie"


# ---------------------------------------------------------------------------
# Tier C1 v2: format-prior heuristic tiebreaker (no LLM call)
# ---------------------------------------------------------------------------

# Format cues we look for in the question.  These map to features of the
# candidate responses that are easy to detect deterministically.
_FORMAT_CUE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "wants_terse": (
        r"\bin\s+one\s+sentence\b",
        r"\bin\s+a\s+sentence\b",
        r"\bone[-\s]liner\b",
        r"\b(very\s+)?(briefly|concisely|short(est)?)\b",
        r"\bshort\s+answer\b",
    ),
    "wants_list": (
        r"\b(list|enumerate)\b",
        r"\bbullet(?:ed|s|\s+points?)?\b",
    ),
    "wants_numbered": (
        r"\bstep[-\s]by[-\s]step\b",
        r"\bnumbered\s+list\b",
        r"\bin\s+order\b",
    ),
    "wants_table": (
        r"\btable\b",
        r"\btabular\b",
    ),
    "wants_code": (
        r"\bwrite\s+(?:a\s+)?(?:python|javascript|js|typescript|sql|bash|shell)\b",
        r"\bimplement\s+(?:a\s+)?function\b",
        r"\bcode\s+(?:snippet|example)\b",
    ),
    "wants_explain": (
        r"\bexplain\s+(?:in\s+detail|thoroughly|step\b)",
        r"\belaborate\b",
        r"\bdescribe\s+in\s+detail\b",
        r"\bin\s+detail\b",
    ),
}

_FORMAT_CUE_REGEXES: Dict[str, List[re.Pattern]] = {
    name: [re.compile(p, re.IGNORECASE) for p in patterns]
    for name, patterns in _FORMAT_CUE_PATTERNS.items()
}


def _detect_format_request(question: str) -> Dict[str, bool]:
    q = question or ""
    out: Dict[str, bool] = {}
    for name, patterns in _FORMAT_CUE_REGEXES.items():
        out[name] = any(p.search(q) for p in patterns)
    return out


_BULLET_LINE_RE = re.compile(r"^\s*([-*\u2022]|\d+[.)])\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*-{3,}.*\|.*$")
_CODE_FENCE_RE = re.compile(r"```")


def _format_features(text: str) -> Dict[str, float]:
    t = text or ""
    lines = t.split("\n")
    bullets = sum(1 for line in lines if _BULLET_LINE_RE.match(line))
    table_rows = sum(1 for line in lines if _TABLE_ROW_RE.match(line))
    has_table_separator = any(_TABLE_SEPARATOR_RE.match(line) for line in lines)
    code_blocks = len(_CODE_FENCE_RE.findall(t)) // 2
    words = len(t.split())
    return {
        "char_count": float(len(t)),
        "word_count": float(words),
        "bullet_count": float(bullets),
        "table_rows": float(table_rows),
        "has_table": float(table_rows >= 2 and has_table_separator),
        "code_blocks": float(code_blocks),
    }


@dataclass(frozen=True)
class FormatPriorOutcome:
    """Result of the format-prior tiebreaker (no LLM call)."""

    verdict: str  # 'a' / 'b' / 'tie'
    reason: str
    cues: Dict[str, bool]
    votes_a: int
    votes_b: int


def format_prior_predict(
    question: str,
    response_a: str,
    response_b: str,
    *,
    terse_word_ratio: float = 0.7,
    explain_word_ratio: float = 1.5,
) -> FormatPriorOutcome:
    """Heuristic format-prior tiebreaker for score-tied rows.

    Looks at the question for explicit format requests (list / table / code /
    "answer in one sentence" / "explain in detail") and votes for whichever
    response honours that request.  Returns ``"tie"`` whenever neither side
    clearly wins the format match -- the cascade falls through to the LLM
    judge in that case.  Free, no LLM calls; intended as a cheap pre-judge
    step in the cascade.
    """
    cues = _detect_format_request(question)
    fa = _format_features(response_a)
    fb = _format_features(response_b)

    votes_a = 0
    votes_b = 0
    reasons: List[str] = []

    if cues["wants_terse"] and fa["word_count"] > 0 and fb["word_count"] > 0:
        if fb["word_count"] == 0 or fa["word_count"] / max(1.0, fb["word_count"]) < terse_word_ratio:
            votes_a += 1
            reasons.append("terse:A")
        elif fa["word_count"] == 0 or fb["word_count"] / max(1.0, fa["word_count"]) < terse_word_ratio:
            votes_b += 1
            reasons.append("terse:B")

    if cues["wants_list"] or cues["wants_numbered"]:
        if fa["bullet_count"] >= 2 and fb["bullet_count"] < 2:
            votes_a += 1
            reasons.append("list:A")
        elif fb["bullet_count"] >= 2 and fa["bullet_count"] < 2:
            votes_b += 1
            reasons.append("list:B")

    if cues["wants_table"]:
        if fa["has_table"] and not fb["has_table"]:
            votes_a += 1
            reasons.append("table:A")
        elif fb["has_table"] and not fa["has_table"]:
            votes_b += 1
            reasons.append("table:B")

    if cues["wants_code"]:
        if fa["code_blocks"] > 0 and fb["code_blocks"] == 0:
            votes_a += 1
            reasons.append("code:A")
        elif fb["code_blocks"] > 0 and fa["code_blocks"] == 0:
            votes_b += 1
            reasons.append("code:B")

    if cues["wants_explain"] and fa["word_count"] > 0 and fb["word_count"] > 0:
        if fa["word_count"] / max(1.0, fb["word_count"]) > explain_word_ratio:
            votes_a += 1
            reasons.append("explain:A")
        elif fb["word_count"] / max(1.0, fa["word_count"]) > explain_word_ratio:
            votes_b += 1
            reasons.append("explain:B")

    if votes_a > votes_b:
        return FormatPriorOutcome(
            verdict="a",
            reason="format_prior:" + ",".join(reasons),
            cues=cues,
            votes_a=votes_a,
            votes_b=votes_b,
        )
    if votes_b > votes_a:
        return FormatPriorOutcome(
            verdict="b",
            reason="format_prior:" + ",".join(reasons),
            cues=cues,
            votes_a=votes_a,
            votes_b=votes_b,
        )
    if votes_a == 0 and votes_b == 0:
        return FormatPriorOutcome(
            verdict="tie",
            reason="no_format_signal",
            cues=cues,
            votes_a=0,
            votes_b=0,
        )
    return FormatPriorOutcome(
        verdict="tie",
        reason="format_prior_tied:" + ",".join(reasons),
        cues=cues,
        votes_a=votes_a,
        votes_b=votes_b,
    )
