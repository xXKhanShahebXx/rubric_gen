"""Regenerate one side (A or B) of a pair-preference dataset, then re-judge.

Per row of the input pair-preference JSONL (default
``data/medical_gpt41_answers_rl.jsonl``):

1. Calls ``--writer-model`` (default ``openai:gpt-5``) with the row's
   ``question`` to generate a fresh response. With ``--regen-side A`` (default)
   this replaces the original ``reference_answer_a``; with ``--regen-side B``
   it replaces ``reference_answer_b`` instead.
2. Calls ``--judge-model`` (default ``anthropic:claude-opus-4-7``) to judge
   the resulting (response_a, response_b) pair using explicit medical-quality
   criteria (mirrors ``scripts/relabel_pair_dataset.py``'s judge prompt).

Two independent JSONL caches:
- ``regen.jsonl`` for writer calls (keyed on writer model + question; safe to
  share across A/B runs since the same model + question produces the same
  text regardless of which slot it later lands in).
- ``pair_judge.jsonl`` for judge calls (keyed on judge model + question +
  both responses).

Both are keyed on stable hashes of their inputs, so a re-run of the same
samples is free, and a Ctrl+C mid-run is fully recoverable.

Output schema (the regenerated side gets ``original_*`` fields preserving the
old text plus ``*_model`` / ``*_cache_hit`` provenance):

- ``reference_answer_<a|b>`` overwritten with the new writer-model response.
- ``original_reference_answer_<a|b>`` preserves the old response.
- ``reference_answer_<a|b>_model``: e.g. ``"openai:gpt-5"``.
- ``reference_answer_<a|b>_cache_hit``: bool.
- ``correct_answer`` overwritten with the judge verdict
  (``reference_answer_a`` / ``reference_answer_b``); ties keep the original.
- ``original_correct_answer``: preserved.
- ``judge_*`` fields: model, verdict, confidence, reasoning, tie flag, cache
  hit, optional parse / router error.

Examples:

  # Regenerate A with gpt-5 (original behavior):
  python scripts/regenerate_a_and_judge.py \\
    --input data/medical_gpt41_answers_rl.jsonl \\
    --output data/medical_gpt5_a_vs_gpt41_b_opus_judged.jsonl \\
    --limit 500

  # Regenerate B with gpt-4o (intra-model gpt-4o noise-floor test):
  python scripts/regenerate_a_and_judge.py \\
    --regen-side B \\
    --writer-model openai:gpt-4o \\
    --input data/medical_gpt41_answers_rl.jsonl \\
    --output data/medical_gpt4o_b_regen_opus_judged.jsonl \\
    --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from rubric_gen.config import parse_model_spec
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


REGEN_PROMPT_VERSION = "regen_a_v1"  # cache-stable; "a" is historical, response is side-agnostic
JUDGE_PROMPT_VERSION = "pair_relabel_v1"  # match relabel_pair_dataset.py for diff-ability

DEFAULT_WRITER_MODEL = "openai:gpt-5"
DEFAULT_JUDGE_MODEL = "anthropic:claude-opus-4-7"
DEFAULT_REGEN_SIDE = "A"
ALLOWED_REGEN_SIDES = ("A", "B")

# How to map a TIE verdict to ``correct_answer`` in the output JSONL.
# - "original": preserve the row's original ``correct_answer`` (back-compat).
# - "a"       : ties go to reference_answer_a.
# - "b"       : ties go to reference_answer_b.
# Parse errors / empty verdicts always preserve the original (treated as
# "no information", not as a tie).
DEFAULT_TIE_POLICY = "original"
ALLOWED_TIE_POLICIES = ("original", "a", "b")

# Side-aware default output paths. Used only when --output is not provided.
DEFAULT_OUTPUT_BY_SIDE: Dict[str, Path] = {
    "A": Path("data/medical_gpt5_a_vs_gpt41_b_opus_judged.jsonl"),
    "B": Path("data/medical_gpt4o_b_regen_opus_judged.jsonl"),
}

VERDICT_A = "A"
VERDICT_B = "B"
VERDICT_TIE = "TIE"
ALLOWED_VERDICTS = (VERDICT_A, VERDICT_B, VERDICT_TIE)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
ALLOWED_CONFIDENCE = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)


_REGEN_SYSTEM_PROMPT = """You are an expert medical assistant answering a medical question for a clinician.

Produce a clinically faithful, well-reasoned answer to the question. Cover the relevant facts, mechanisms, guidelines, contraindications, and caveats that a knowledgeable clinician would expect, without padding the response with off-topic material. Avoid inventing facts, dosages, or guidelines that are not actually correct. Format the answer in whatever structure best serves the question (prose, bullet points, or short headers); do not over-format simple answers.

Output the answer text only -- no preamble, no commentary about your sources, no meta-discussion."""


# Mirrors scripts/relabel_pair_dataset.py so verdicts are directly comparable
# across the two relabel experiments.
_JUDGE_SYSTEM_PROMPT = """You are an expert medical evaluator comparing two candidate responses to a medical question. Decide which response is better on the following criteria, in priority order:

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

Use "TIE" only when both responses are equivalently correct and complete on every criterion above.

Output ONLY the JSON object. Do not include any other text."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegenOutcome:
    text: str
    cache_hit: bool
    error: str = ""


@dataclass(frozen=True)
class JudgeOutcome:
    verdict: str  # "A" / "B" / "TIE" / ""
    confidence: str
    reasoning: str
    raw_response: str
    cache_hit: bool
    parse_error: str = ""
    router_error: str = ""


@dataclass(frozen=True)
class RowResult:
    index: int
    row: Dict[str, Any]
    relabeled_row: Dict[str, Any]
    regen: RegenOutcome
    judge: JudgeOutcome


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/medical_gpt41_answers_rl.jsonl"),
        help="Source pair-preference JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination JSONL with regenerated side + judge fields. "
            "If omitted, defaults to "
            f"{DEFAULT_OUTPUT_BY_SIDE['A']} for --regen-side A or "
            f"{DEFAULT_OUTPUT_BY_SIDE['B']} for --regen-side B."
        ),
    )
    parser.add_argument(
        "--regen-side",
        type=str,
        choices=list(ALLOWED_REGEN_SIDES),
        default=DEFAULT_REGEN_SIDE,
        help=(
            "Which side of the pair to regenerate with --writer-model. "
            f"Default: {DEFAULT_REGEN_SIDE}. The other side is left as-is."
        ),
    )
    parser.add_argument(
        "--tie-policy",
        type=str,
        choices=list(ALLOWED_TIE_POLICIES),
        default=DEFAULT_TIE_POLICY,
        help=(
            "How to map a TIE verdict to the output 'correct_answer'. "
            f"Default: {DEFAULT_TIE_POLICY!r} (preserve the row's original label). "
            "Use 'a' to give ties to reference_answer_a, 'b' to give ties to "
            "reference_answer_b. Parse errors always preserve the original."
        ),
    )
    parser.add_argument(
        "--writer-model",
        type=str,
        default=DEFAULT_WRITER_MODEL,
        help=(
            f"provider:model for regenerating the chosen side "
            f"(default: {DEFAULT_WRITER_MODEL})."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=(
            f"provider:model for the Opus judge (default: {DEFAULT_JUDGE_MODEL}). "
            "Override if you don't have access to Opus 4.7."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/regen_a_judge_cache"),
        help=(
            "Directory for the regen + judge JSONL caches. "
            "Default kept as 'regen_a_judge_cache' for back-compat with the "
            "A-side run; safe to share across A/B runs since cache keys "
            "include the writer model + question."
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N rows of the input.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Process this many rows after --start (default: 500).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent (regen + judge) pipelines in flight (default: 8).",
    )
    parser.add_argument(
        "--writer-max-tokens",
        type=int,
        default=4096,
        help="Max output tokens for the writer regeneration call (default: 4096).",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature (default: 0.0). Ignored for Opus 4.7+ thinking models.",
    )
    parser.add_argument(
        "--writer-temperature",
        type=float,
        default=1.0,
        help=(
            "Writer sampling temperature (default: 1.0; the gpt-5 responses API "
            "ignores this anyway, but kept for symmetry)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all API calls. Useful to preview which rows would be processed.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress line every N completions.",
    )
    return parser


# ---------------------------------------------------------------------------
# gpt-5 regeneration
# ---------------------------------------------------------------------------


def _regen_cache_key(*, model: ModelSpec, question: str, max_tokens: int) -> str:
    payload = {
        "prompt_version": REGEN_PROMPT_VERSION,
        "model": f"{model.provider}:{model.model}",
        "question_hash": stable_hash(question or ""),
        "max_tokens": int(max_tokens),
        "system_prompt_hash": stable_hash(_REGEN_SYSTEM_PROMPT),
    }
    return make_cache_key(REGEN_PROMPT_VERSION, payload)


def _regen_response(
    *,
    writer_model: ModelSpec,
    router: Optional[LLMRouter],
    cache: JsonlCache,
    question: str,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
) -> RegenOutcome:
    """Generate a response from ``writer_model`` for ``question``.

    Side-agnostic: the caller decides which slot the result goes into.
    """
    cache_key = _regen_cache_key(model=writer_model, question=question, max_tokens=max_tokens)
    cached = cache.get(cache_key) if cache.enabled else None
    if cached and isinstance(cached.get("text"), str) and cached["text"]:
        return RegenOutcome(text=str(cached["text"]), cache_hit=True)
    if dry_run or router is None:
        return RegenOutcome(text="", cache_hit=False, error="dry_run" if dry_run else "no_router")
    try:
        response = router.generate(
            writer_model,
            system_prompt=_REGEN_SYSTEM_PROMPT,
            user_prompt=str(question or ""),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
    except Exception as exc:
        return RegenOutcome(text="", cache_hit=False, error=f"{type(exc).__name__}: {exc}")
    text = (response.text or response.raw_text or "").strip()
    if cache.enabled and text:
        cache.set(
            cache_key,
            {
                "kind": "regen",
                "model": f"{writer_model.provider}:{writer_model.model}",
                "text": text,
            },
        )
    return RegenOutcome(text=text, cache_hit=False, error="" if text else "empty_response")


# Back-compat alias: existing tests / callers may import _regen_a.
_regen_a = _regen_response


# ---------------------------------------------------------------------------
# Opus judge (mirrors scripts/relabel_pair_dataset.py for verdict comparability)
# ---------------------------------------------------------------------------


def _normalize_verdict(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if text in {"A", "REFERENCE_ANSWER_A", "RESPONSE_A", "ANSWER_A", "PAIR_RESPONSE_A"}:
        return VERDICT_A
    if text in {"B", "REFERENCE_ANSWER_B", "RESPONSE_B", "ANSWER_B", "PAIR_RESPONSE_B"}:
        return VERDICT_B
    if text in {"TIE", "EQUAL", "DRAW", "EQUIVALENT", "BOTH"}:
        return VERDICT_TIE
    return ""


def _normalize_confidence(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in ALLOWED_CONFIDENCE:
        return text
    return CONFIDENCE_MEDIUM


def _parse_judge_response(raw_text: str) -> Tuple[str, str, str, str]:
    payload = extract_json_object(raw_text)
    if not isinstance(payload, dict):
        return "", "", "", "no_json_object"
    verdict = _normalize_verdict(payload.get("verdict") or payload.get("decision") or payload.get("answer"))
    if not verdict:
        return "", "", "", "unrecognized_verdict"
    confidence = _normalize_confidence(payload.get("confidence"))
    reasoning = str(payload.get("reasoning") or payload.get("rationale") or "").strip()
    return verdict, confidence, reasoning, ""


def _judge_cache_key(
    *,
    judge_model: ModelSpec,
    question: str,
    response_a: str,
    response_b: str,
    temperature: float,
) -> str:
    payload = {
        "prompt_version": JUDGE_PROMPT_VERSION,
        "model": f"{judge_model.provider}:{judge_model.model}",
        "temperature": round(float(temperature), 4),
        "question_hash": stable_hash(question or ""),
        "response_a_hash": stable_hash(response_a or ""),
        "response_b_hash": stable_hash(response_b or ""),
    }
    return make_cache_key(JUDGE_PROMPT_VERSION, payload)


def _build_judge_user_prompt(question: str, response_a: str, response_b: str) -> str:
    return (
        f"<Question>\n{question.strip()}\n</Question>\n\n"
        f"<Response A>\n{response_a.strip()}\n</Response A>\n\n"
        f"<Response B>\n{response_b.strip()}\n</Response B>\n\n"
        "Return your JSON now."
    )


def _judge_pair(
    *,
    judge_model: ModelSpec,
    router: Optional[LLMRouter],
    cache: JsonlCache,
    question: str,
    response_a: str,
    response_b: str,
    temperature: float,
    dry_run: bool,
) -> JudgeOutcome:
    cache_key = _judge_cache_key(
        judge_model=judge_model,
        question=question,
        response_a=response_a,
        response_b=response_b,
        temperature=temperature,
    )
    cached = cache.get(cache_key) if cache.enabled else None
    if cached and isinstance(cached.get("raw_response"), str):
        verdict, confidence, reasoning, parse_error = _parse_judge_response(cached["raw_response"])
        return JudgeOutcome(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=cached["raw_response"],
            cache_hit=True,
            parse_error=parse_error,
        )
    if dry_run or router is None:
        return JudgeOutcome(
            verdict="",
            confidence="",
            reasoning="",
            raw_response="",
            cache_hit=False,
            parse_error="dry_run" if dry_run else "no_router",
        )
    try:
        response = router.generate(
            judge_model,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            user_prompt=_build_judge_user_prompt(question, response_a, response_b),
            temperature=float(temperature),
        )
    except Exception as exc:
        return JudgeOutcome(
            verdict="",
            confidence="",
            reasoning="",
            raw_response="",
            cache_hit=False,
            parse_error="router_error",
            router_error=f"{type(exc).__name__}: {exc}",
        )
    raw_text = response.raw_text or response.text or ""
    verdict, confidence, reasoning, parse_error = _parse_judge_response(raw_text)
    if cache.enabled:
        cache.set(
            cache_key,
            {
                "kind": "pair_relabel",
                "raw_response": raw_text,
                "verdict": verdict,
                "confidence": confidence,
                "reasoning": reasoning,
                "parse_error": parse_error,
            },
        )
    return JudgeOutcome(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        raw_response=raw_text,
        cache_hit=False,
        parse_error=parse_error,
    )


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------


def _verdict_to_correct_answer(
    verdict: str,
    original: str,
    tie_policy: str = DEFAULT_TIE_POLICY,
) -> str:
    """Map a judge verdict to the output ``correct_answer`` label.

    Parse errors / empty verdicts always preserve the original label
    (treated as "no information"); only TIE verdicts are routed through
    ``tie_policy``.
    """
    if verdict == VERDICT_A:
        return "reference_answer_a"
    if verdict == VERDICT_B:
        return "reference_answer_b"
    if verdict == VERDICT_TIE:
        if tie_policy == "a":
            return "reference_answer_a"
        if tie_policy == "b":
            return "reference_answer_b"
        return original  # "original"
    return original  # parse error / empty verdict


def _side_field_names(side: str) -> Tuple[str, str, str, str, str]:
    """Return field name tuple for the regenerated side.

    Returns (current, original, model, cache_hit, error) field names.
    """
    suffix = side.lower()
    return (
        f"reference_answer_{suffix}",
        f"original_reference_answer_{suffix}",
        f"reference_answer_{suffix}_model",
        f"reference_answer_{suffix}_cache_hit",
        f"reference_answer_{suffix}_error",
    )


def _build_relabeled_row(
    *,
    row: Mapping[str, Any],
    new_text: str,
    regen_side: str,
    regen: RegenOutcome,
    writer_model: ModelSpec,
    judge: JudgeOutcome,
    judge_model: ModelSpec,
    tie_policy: str = DEFAULT_TIE_POLICY,
) -> Dict[str, Any]:
    out = dict(row)
    cur_field, orig_field, model_field, cache_field, err_field = _side_field_names(regen_side)
    original_correct = str(row.get("correct_answer") or "")
    original_text = str(row.get(cur_field) or "")

    out[orig_field] = original_text
    if new_text:
        out[cur_field] = new_text
    out[model_field] = f"{writer_model.provider}:{writer_model.model}"
    out[cache_field] = bool(regen.cache_hit)
    if regen.error:
        out[err_field] = regen.error

    out["original_correct_answer"] = original_correct
    out["correct_answer"] = _verdict_to_correct_answer(
        judge.verdict, original_correct, tie_policy=tie_policy
    )
    out["judge_model"] = f"{judge_model.provider}:{judge_model.model}"
    out["judge_verdict"] = judge.verdict
    out["judge_confidence"] = judge.confidence
    out["judge_reasoning"] = judge.reasoning
    out["judge_tie"] = judge.verdict == VERDICT_TIE
    out["judge_cache_hit"] = bool(judge.cache_hit)
    out["regen_side"] = regen_side
    out["tie_policy"] = tie_policy
    if judge.parse_error:
        out["judge_parse_error"] = judge.parse_error
    if judge.router_error:
        out["judge_router_error"] = judge.router_error
    return out


def _process_row(
    *,
    idx: int,
    row: Dict[str, Any],
    regen_side: str,
    writer_model: ModelSpec,
    judge_model: ModelSpec,
    router: Optional[LLMRouter],
    regen_cache: JsonlCache,
    judge_cache: JsonlCache,
    writer_max_tokens: int,
    writer_temperature: float,
    judge_temperature: float,
    dry_run: bool,
    tie_policy: str = DEFAULT_TIE_POLICY,
) -> RowResult:
    question = str(row.get("question") or "")
    original_a = str(row.get("reference_answer_a") or "")
    original_b = str(row.get("reference_answer_b") or "")

    regen = _regen_response(
        writer_model=writer_model,
        router=router,
        cache=regen_cache,
        question=question,
        max_tokens=writer_max_tokens,
        temperature=writer_temperature,
        dry_run=dry_run,
    )

    if regen_side == "A":
        new_text = regen.text or original_a
        judge_a = new_text
        judge_b = original_b
    else:
        new_text = regen.text or original_b
        judge_a = original_a
        judge_b = new_text

    if regen.text:
        judge = _judge_pair(
            judge_model=judge_model,
            router=router,
            cache=judge_cache,
            question=question,
            response_a=judge_a,
            response_b=judge_b,
            temperature=judge_temperature,
            dry_run=dry_run,
        )
    else:
        # Skip judging if regen failed (otherwise we'd be judging the original
        # pair, which contaminates the experiment with stale data).
        judge = JudgeOutcome(
            verdict="",
            confidence="",
            reasoning="",
            raw_response="",
            cache_hit=False,
            parse_error="skipped_due_to_regen_failure" if regen.error else "",
        )

    relabeled = _build_relabeled_row(
        row=row,
        new_text=new_text,
        regen_side=regen_side,
        regen=regen,
        writer_model=writer_model,
        judge=judge,
        judge_model=judge_model,
        tie_policy=tie_policy,
    )
    return RowResult(index=idx, row=row, relabeled_row=relabeled, regen=regen, judge=judge)


# ---------------------------------------------------------------------------
# IO + summary
# ---------------------------------------------------------------------------


def _read_input_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_output_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(results: List[RowResult]) -> Dict[str, Any]:
    n = len(results)
    verdicts = {VERDICT_A: 0, VERDICT_B: 0, VERDICT_TIE: 0, "": 0}
    confidence = {CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 0, CONFIDENCE_LOW: 0, "": 0}
    regen_failures = 0
    regen_cache_hits = 0
    judge_parse_errors = 0
    judge_router_errors = 0
    judge_cache_hits = 0
    label_flips = 0
    final_label_counts = {"reference_answer_a": 0, "reference_answer_b": 0, "other": 0, "": 0}
    for r in results:
        verdicts[r.judge.verdict if r.judge.verdict in verdicts else ""] += 1
        confidence[r.judge.confidence if r.judge.confidence in confidence else ""] += 1
        if r.regen.error:
            regen_failures += 1
        if r.regen.cache_hit:
            regen_cache_hits += 1
        if r.judge.parse_error:
            judge_parse_errors += 1
        if r.judge.router_error:
            judge_router_errors += 1
        if r.judge.cache_hit:
            judge_cache_hits += 1
        original = str(r.row.get("correct_answer") or "")
        new = str(r.relabeled_row.get("correct_answer") or "")
        if original and new and original != new:
            label_flips += 1
        if new in final_label_counts:
            final_label_counts[new] += 1
        elif new:
            final_label_counts["other"] += 1
        else:
            final_label_counts[""] += 1
    decided = verdicts[VERDICT_A] + verdicts[VERDICT_B]
    return {
        "rows_processed": n,
        "verdict_counts": verdicts,
        "confidence_counts": confidence,
        "regen_failures": regen_failures,
        "regen_cache_hits": regen_cache_hits,
        "judge_parse_errors": judge_parse_errors,
        "judge_router_errors": judge_router_errors,
        "judge_cache_hits": judge_cache_hits,
        "label_flips": label_flips,
        "tie_rate": (verdicts[VERDICT_TIE] / n) if n else 0.0,
        "decided_rate": (decided / n) if n else 0.0,
        "label_flip_rate_among_decided": (label_flips / decided) if decided else 0.0,
        "a_win_rate_among_decided": (verdicts[VERDICT_A] / decided) if decided else 0.0,
        "b_win_rate_among_decided": (verdicts[VERDICT_B] / decided) if decided else 0.0,
        "final_label_counts": final_label_counts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    regen_side = str(args.regen_side or DEFAULT_REGEN_SIDE).upper()
    if regen_side not in ALLOWED_REGEN_SIDES:
        print(
            f"ERROR: --regen-side must be one of {ALLOWED_REGEN_SIDES}, got {regen_side!r}",
            file=sys.stderr,
        )
        return 1
    tie_policy = str(args.tie_policy or DEFAULT_TIE_POLICY).lower()
    if tie_policy not in ALLOWED_TIE_POLICIES:
        print(
            f"ERROR: --tie-policy must be one of {ALLOWED_TIE_POLICIES}, got {tie_policy!r}",
            file=sys.stderr,
        )
        return 1
    if args.output is None:
        args.output = DEFAULT_OUTPUT_BY_SIDE[regen_side]

    writer_model = parse_model_spec(args.writer_model, default_alias="regen_writer")
    judge_model = parse_model_spec(args.judge_model, default_alias="pair_relabel_judge")
    print(f"Writer (regenerates {regen_side}): {writer_model.provider}:{writer_model.model}")
    print(f"Judge  (A vs B)         : {judge_model.provider}:{judge_model.model}")
    print(f"Tie policy              : {tie_policy}")

    rows = _read_input_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    if args.start > 0:
        rows = rows[args.start :]
        print(f"After --start {args.start}: {len(rows)} rows.")
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"After --limit {args.limit}: {len(rows)} rows.")

    if not rows:
        print("Nothing to do.")
        return 0

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # Cache file kept as "regen_a.jsonl" for back-compat with the original
    # A-side run; entries are keyed on (writer_model, question, max_tokens) and
    # are valid regardless of which slot the regenerated text lands in.
    regen_cache = JsonlCache(args.cache_dir / "regen_a.jsonl", enabled=True)
    judge_cache = JsonlCache(args.cache_dir / "pair_judge.jsonl", enabled=True)
    regen_cache.load()
    judge_cache.load()
    print(f"Caches under {args.cache_dir}")

    router = None if args.dry_run else LLMRouter(max_retries=3)
    workers = max(1, int(args.workers))
    started = time.perf_counter()

    results: Dict[int, RowResult] = {}

    def _job(idx: int, row: Dict[str, Any]) -> RowResult:
        return _process_row(
            idx=idx,
            row=row,
            regen_side=regen_side,
            writer_model=writer_model,
            judge_model=judge_model,
            router=router,
            regen_cache=regen_cache,
            judge_cache=judge_cache,
            writer_max_tokens=int(args.writer_max_tokens),
            writer_temperature=float(args.writer_temperature),
            judge_temperature=float(args.judge_temperature),
            dry_run=args.dry_run,
            tie_policy=tie_policy,
        )

    if workers <= 1 or args.dry_run:
        for idx, row in enumerate(rows):
            result = _job(idx, row)
            results[idx] = result
            done = idx + 1
            if done % args.progress_every == 0 or done == len(rows):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                print(f"  progress: {done}/{len(rows)} ({rate:.2f} rows/s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_job, idx, row): idx for idx, row in enumerate(rows)}
            done = 0
            for fut in as_completed(futures):
                result = fut.result()
                results[result.index] = result
                done += 1
                if done % args.progress_every == 0 or done == len(rows):
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta_sec = (len(rows) - done) / rate if rate > 0 else 0.0
                    print(
                        f"  progress: {done}/{len(rows)} ({rate:.2f} rows/s, ETA {eta_sec/60:.1f} min)",
                        flush=True,
                    )

    ordered = [results[i] for i in sorted(results.keys())]
    output_rows = [r.relabeled_row for r in ordered]
    _write_output_rows(args.output, output_rows)
    print(f"Wrote {len(output_rows)} relabeled rows to {args.output}")

    summary = _summarize(ordered)
    summary["input"] = str(args.input)
    summary["output"] = str(args.output)
    summary["regen_side"] = regen_side
    summary["tie_policy"] = tie_policy
    summary["writer_model"] = f"{writer_model.provider}:{writer_model.model}"
    summary["judge_model"] = f"{judge_model.provider}:{judge_model.model}"
    summary["dry_run"] = bool(args.dry_run)
    summary["wall_clock_seconds"] = round(time.perf_counter() - started, 1)
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
