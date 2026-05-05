"""Relabel a pair-preference JSONL dataset using a strong Anthropic judge.

Input dataset must follow the medical_gpt41_answers_rl.jsonl schema:

    {"id": "...", "question": "...",
     "reference_answer_a": "...", "reference_answer_b": "...",
     "correct_answer": "reference_answer_a"}

The script sends each (question, response_a, response_b) triple to a strong
Anthropic judge (Claude Opus 4.7 by default) with explicit medical-quality
criteria. The judge returns ``A`` / ``B`` / ``TIE`` plus a confidence and a
short reasoning string. The original ``correct_answer`` is preserved as
``original_correct_answer`` and the new verdict is written into
``correct_answer`` (so the existing pipeline loaders pick it up unchanged).

Phased rollout: ``--restrict-to-run-dir <run-dir>`` only relabels rows whose
``id`` already has a per-example artifact under ``<run-dir>/examples/``. This
is the cheap "validate the validation" pass -- relabel just the ~hundreds of
examples already scored by the rubric pipeline, then use
``scripts/rescore_pair_artifacts.py`` to recompute the pair-preference
accuracy against the new labels in seconds.

Caching: every judge call is keyed via JsonlCache on the (prompt, responses,
model, strictness, prompt version) tuple, so re-runs are free and a Ctrl+C
mid-run is fully recoverable.

Usage examples:

  # Phase 1: relabel only the 356 examples already scored by the pipeline.
  python scripts/relabel_pair_dataset.py \\
    --restrict-to-run-dir artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt41 \\
    --output data/medical_gpt41_answers_rl_relabeled_phase1.jsonl

  # Phase 3a: relabel the full 4000-row dataset.
  python scripts/relabel_pair_dataset.py \\
    --output data/medical_gpt41_answers_rl_relabeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from rubric_gen.config import parse_model_spec
from rubric_gen.llm_client import LLMRouter, extract_json_object
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


PROMPT_VERSION = "pair_relabel_v1"
# Default to the latest Opus thinking model. LLMRouter detects the
# Opus-4-7+ family and omits the `temperature` parameter automatically
# (these models reject it). Override via --judge-model to pin a specific
# version (e.g. anthropic:claude-opus-4-1-20250805, anthropic:claude-opus-4-6).
DEFAULT_JUDGE_MODEL = "anthropic:claude-opus-4-7"

VERDICT_A = "A"
VERDICT_B = "B"
VERDICT_TIE = "TIE"
ALLOWED_VERDICTS = (VERDICT_A, VERDICT_B, VERDICT_TIE)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
ALLOWED_CONFIDENCE = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)


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

Use "TIE" only when both responses are equivalently correct and complete on every criterion above.

Output ONLY the JSON object. Do not include any other text."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeOutcome:
    """One judge call's result for a single pair row."""

    verdict: str  # "A" / "B" / "TIE" / "" (parse error)
    confidence: str  # "high" / "medium" / "low" / ""
    reasoning: str
    raw_response: str
    cache_hit: bool
    parse_error: str = ""
    router_error: str = ""


@dataclass(frozen=True)
class RowResult:
    """Output for one input row plus its judge outcome."""

    index: int
    row: Dict[str, Any]
    relabeled_row: Dict[str, Any]
    outcome: JudgeOutcome


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/medical_gpt41_answers_rl.jsonl"),
        help="Source pair-preference JSONL (default: data/medical_gpt41_answers_rl.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL. Same schema as input plus judge-* fields.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "provider:model for the judge (default: %(default)s). Override if your "
            "Anthropic account does not have access to Opus 4.7, e.g. "
            "anthropic:claude-opus-4-1-20250805 or anthropic:claude-opus-4-6."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/pair_relabel_cache"),
        help=(
            "Directory for the per-pair JSONL cache. Re-runs that include "
            "already-judged pairs are free."
        ),
    )
    parser.add_argument(
        "--restrict-to-run-dir",
        type=Path,
        default=None,
        help=(
            "If set, only relabel rows whose 'id' suffix appears as an example "
            "artifact under <run-dir>/examples/. Use this for the Phase 1 "
            "'relabel just the already-scored ~hundreds' shortcut."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent judge calls in flight (default: 8).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process this many rows (after restriction filtering).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip all API calls. Reports what would be relabeled (after filtering "
            "and cache lookup) without spending budget."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress line every N completions (default: 25).",
    )
    return parser


# ---------------------------------------------------------------------------
# Restriction
# ---------------------------------------------------------------------------


def _ids_from_run_dir(run_dir: Path) -> Set[str]:
    """Return the set of JSONL row ids whose example artifact lives under run_dir.

    Per-example filenames look like ``<task_profile>__<id>.json``. We split on
    the first ``__`` and take everything after it (in case the source field
    itself contains an underscore, the suffix is the original ``id``).
    """
    examples_dir = run_dir / "examples"
    if not examples_dir.is_dir():
        raise SystemExit(f"--restrict-to-run-dir: no examples/ under {run_dir}")
    ids: Set[str] = set()
    for path in examples_dir.glob("*.json"):
        stem = path.stem
        if "__" not in stem:
            continue
        _, _, suffix = stem.partition("__")
        if suffix:
            ids.add(suffix)
    return ids


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def _build_user_prompt(question: str, response_a: str, response_b: str) -> str:
    return (
        f"<Question>\n{question.strip()}\n</Question>\n\n"
        f"<Response A>\n{response_a.strip()}\n</Response A>\n\n"
        f"<Response B>\n{response_b.strip()}\n</Response B>\n\n"
        "Return your JSON now."
    )


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
    """Parse the judge's JSON response. Returns (verdict, confidence, reasoning, parse_error)."""
    payload = extract_json_object(raw_text)
    if not isinstance(payload, dict):
        return "", "", "", "no_json_object"
    verdict = _normalize_verdict(payload.get("verdict") or payload.get("decision") or payload.get("answer"))
    if not verdict:
        return "", "", "", "unrecognized_verdict"
    confidence = _normalize_confidence(payload.get("confidence"))
    reasoning = str(payload.get("reasoning") or payload.get("rationale") or "").strip()
    return verdict, confidence, reasoning, ""


def _build_cache_key(
    *,
    judge_model: ModelSpec,
    question: str,
    response_a: str,
    response_b: str,
    temperature: float,
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": f"{judge_model.provider}:{judge_model.model}",
        "temperature": round(float(temperature), 4),
        "question_hash": stable_hash(question or ""),
        "response_a_hash": stable_hash(response_a or ""),
        "response_b_hash": stable_hash(response_b or ""),
    }
    return make_cache_key(PROMPT_VERSION, payload)


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
    cache_key = _build_cache_key(
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
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(question, response_a, response_b),
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
# Row-level processing
# ---------------------------------------------------------------------------


def _verdict_to_correct_answer(verdict: str, original: str) -> str:
    if verdict == VERDICT_A:
        return "reference_answer_a"
    if verdict == VERDICT_B:
        return "reference_answer_b"
    return original  # TIE / unparseable -> keep original


def _relabel_row(row: Mapping[str, Any], outcome: JudgeOutcome, judge_model: ModelSpec) -> Dict[str, Any]:
    out = dict(row)
    original = str(row.get("correct_answer") or "")
    out["original_correct_answer"] = original
    out["correct_answer"] = _verdict_to_correct_answer(outcome.verdict, original)
    out["judge_model"] = f"{judge_model.provider}:{judge_model.model}"
    out["judge_verdict"] = outcome.verdict
    out["judge_confidence"] = outcome.confidence
    out["judge_reasoning"] = outcome.reasoning
    out["judge_tie"] = outcome.verdict == VERDICT_TIE
    out["judge_cache_hit"] = outcome.cache_hit
    if outcome.parse_error:
        out["judge_parse_error"] = outcome.parse_error
    if outcome.router_error:
        out["judge_router_error"] = outcome.router_error
    return out


def _read_input_rows(input_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fh:
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


def _write_output_rows(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _summarize(results: List[RowResult]) -> Dict[str, Any]:
    n = len(results)
    verdicts = {VERDICT_A: 0, VERDICT_B: 0, VERDICT_TIE: 0, "": 0}
    confidence = {CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 0, CONFIDENCE_LOW: 0, "": 0}
    parse_errors = 0
    router_errors = 0
    cache_hits = 0
    label_flips = 0
    for r in results:
        verdicts[r.outcome.verdict if r.outcome.verdict in verdicts else ""] += 1
        confidence[r.outcome.confidence if r.outcome.confidence in confidence else ""] += 1
        if r.outcome.parse_error:
            parse_errors += 1
        if r.outcome.router_error:
            router_errors += 1
        if r.outcome.cache_hit:
            cache_hits += 1
        original = str(r.row.get("correct_answer") or "")
        new = str(r.relabeled_row.get("correct_answer") or "")
        if original and new and original != new:
            label_flips += 1
    abstain = verdicts[""] + verdicts[VERDICT_TIE]
    decided = max(0, n - abstain)
    return {
        "rows_processed": n,
        "verdict_counts": verdicts,
        "confidence_counts": confidence,
        "parse_errors": parse_errors,
        "router_errors": router_errors,
        "cache_hits": cache_hits,
        "label_flips": label_flips,
        "tie_rate": (verdicts[VERDICT_TIE] / n) if n else 0.0,
        "decided_rate": (decided / n) if n else 0.0,
        "label_flip_rate_among_decided": (label_flips / decided) if decided else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    judge_model = parse_model_spec(args.judge_model, default_alias="pair_relabel_judge")
    if judge_model.provider != "anthropic":
        print(
            f"WARNING: judge model is {judge_model.provider}:{judge_model.model}; "
            "the prompt is tuned for an Anthropic Opus-class judge. Continuing anyway.",
            file=sys.stderr,
        )

    rows = _read_input_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    if args.restrict_to_run_dir is not None:
        allowed_ids = _ids_from_run_dir(args.restrict_to_run_dir)
        before = len(rows)
        rows = [row for row in rows if str(row.get("id") or "") in allowed_ids]
        print(
            f"Restricted to {len(rows)} rows whose id appears under "
            f"{args.restrict_to_run_dir / 'examples'} (was {before})."
        )

    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"Capped at --limit {args.limit} -> {len(rows)} rows.")

    if not rows:
        print("Nothing to do.")
        return 0

    cache_path = args.cache_dir / "pair_relabel.jsonl"
    cache = JsonlCache(cache_path, enabled=True)
    cache.load()
    print(f"Cache: {cache_path} (loaded)")

    router = None if args.dry_run else LLMRouter(max_retries=3)

    results: Dict[int, RowResult] = {}
    started = time.perf_counter()
    workers = max(1, int(args.workers))

    def _process_row(idx: int, row: Dict[str, Any]) -> RowResult:
        outcome = _judge_pair(
            judge_model=judge_model,
            router=router,
            cache=cache,
            question=str(row.get("question") or ""),
            response_a=str(row.get("reference_answer_a") or ""),
            response_b=str(row.get("reference_answer_b") or ""),
            temperature=float(args.temperature),
            dry_run=args.dry_run,
        )
        relabeled = _relabel_row(row, outcome, judge_model)
        return RowResult(index=idx, row=row, relabeled_row=relabeled, outcome=outcome)

    if workers <= 1 or args.dry_run:
        for idx, row in enumerate(rows):
            result = _process_row(idx, row)
            results[idx] = result
            if (idx + 1) % args.progress_every == 0 or (idx + 1) == len(rows):
                elapsed = time.perf_counter() - started
                rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  progress: {idx+1}/{len(rows)} ({rate:.2f} rows/s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_row, idx, row): idx for idx, row in enumerate(rows)}
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
    summary["judge_model"] = f"{judge_model.provider}:{judge_model.model}"
    summary["dry_run"] = bool(args.dry_run)
    summary["wall_clock_seconds"] = round(time.perf_counter() - started, 1)
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
