"""Regenerate ``reference_answer_b`` in a pair-preference JSONL using a chosen writer model.

Reads a pair-preference JSONL (e.g. data/medical_gpt41_answers_rl.jsonl) and, for
each row, re-prompts the writer model with the row's ``question`` to produce a
fresh ``reference_answer_b``. The original B text is preserved as
``original_reference_answer_b`` and the writer model id is recorded in
``reference_answer_b_model``. ``reference_answer_a`` is unchanged.

Designed to be a building block for "swap one side of a pair dataset" tasks --
e.g. swap gpt-4.1 B's for gpt-5 B's so a downstream Opus judge can score
gpt-4o vs gpt-5 instead of gpt-4o vs gpt-4.1.

Caching: every writer call is keyed via JsonlCache on
(model, question hash, system prompt hash, temperature, prompt version) so a
re-run is free.

Usage:

  # Default: regenerate B with GPT-5 on the first 500 rows.
  python scripts/regen_pair_response_b.py \\
    --input data/medical_gpt41_answers_rl.jsonl \\
    --output data/medical_gpt5_answers_rl.jsonl \\
    --writer-model openai:gpt-5 \\
    --limit 500

  # Pin a specific GPT-5 variant.
  python scripts/regen_pair_response_b.py ... --writer-model openai:gpt-5.4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from rubric_gen.config import parse_model_spec
from rubric_gen.llm_client import LLMRouter
from rubric_gen.storage import JsonlCache, make_cache_key, stable_hash
from rubric_gen.types import ModelSpec


PROMPT_VERSION = "regen_pair_response_b_v1"
DEFAULT_WRITER_MODEL = "openai:gpt-5"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert medical assistant answering a medical question for a "
    "clinician. Produce a clinically accurate, well-reasoned answer grounded in "
    "current evidence-based medicine. Cover the conclusion, the supporting "
    "reasoning, and any clinically relevant caveats. Do not invent facts, "
    "numbers, or guidelines. Output only the answer."
)


@dataclass(frozen=True)
class WriterOutcome:
    text: str
    cache_hit: bool
    router_error: str = ""


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
        required=True,
        help="Destination JSONL. Same schema as input plus writer-model metadata.",
    )
    parser.add_argument(
        "--writer-model",
        type=str,
        default=DEFAULT_WRITER_MODEL,
        help=(
            "provider:model for the new B response (default: %(default)s). "
            "GPT-5 family is auto-routed via LLMRouter's responses API."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the writer. Defaults to a neutral medical-expert template.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/regen_pair_b_cache"),
        help="Directory for the per-row JSONL cache.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent writer calls in flight (default: 8).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Writer temperature (ignored for GPT-5 thinking models).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for the writer's response.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N input rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process this many rows after --start.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; reports counts of what would be regenerated.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress line every N completions.",
    )
    return parser


# ---------------------------------------------------------------------------
# I/O helpers
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


# ---------------------------------------------------------------------------
# Writer call + cache
# ---------------------------------------------------------------------------


def _build_cache_key(
    *,
    writer_spec: ModelSpec,
    question: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": f"{writer_spec.provider}:{writer_spec.model}",
        "temperature": round(float(temperature), 4),
        "max_tokens": int(max_tokens),
        "system_prompt_hash": stable_hash(system_prompt or ""),
        "question_hash": stable_hash(question or ""),
    }
    return make_cache_key(PROMPT_VERSION, payload)


def _generate_response(
    *,
    writer_spec: ModelSpec,
    router: Optional[LLMRouter],
    cache: JsonlCache,
    question: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> WriterOutcome:
    cache_key = _build_cache_key(
        writer_spec=writer_spec,
        question=question,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cached = cache.get(cache_key) if cache.enabled else None
    if cached and isinstance(cached.get("text"), str):
        return WriterOutcome(text=str(cached["text"]), cache_hit=True)

    if dry_run or router is None:
        return WriterOutcome(text="", cache_hit=False, router_error="dry_run" if dry_run else "no_router")

    try:
        response = router.generate(
            writer_spec,
            system_prompt=system_prompt,
            user_prompt=str(question or ""),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
    except Exception as exc:
        return WriterOutcome(
            text="",
            cache_hit=False,
            router_error=f"{type(exc).__name__}: {exc}",
        )

    text = response.text or response.raw_text or ""
    if cache.enabled:
        cache.set(cache_key, {"kind": "regen_pair_b", "text": text})
    return WriterOutcome(text=text, cache_hit=False)


# ---------------------------------------------------------------------------
# Row-level processing
# ---------------------------------------------------------------------------


def _regen_row(row: Mapping[str, Any], outcome: WriterOutcome, writer_spec: ModelSpec) -> Dict[str, Any]:
    out = dict(row)
    out["original_reference_answer_b"] = str(row.get("reference_answer_b") or "")
    out["reference_answer_b"] = outcome.text
    out["reference_answer_b_model"] = f"{writer_spec.provider}:{writer_spec.model}"
    out["reference_answer_b_cache_hit"] = outcome.cache_hit
    if outcome.router_error:
        out["reference_answer_b_error"] = outcome.router_error
    return out


def _summarize(rows_after: List[Dict[str, Any]], wall_clock_s: float, writer_spec: ModelSpec) -> Dict[str, Any]:
    n = len(rows_after)
    successes = sum(1 for r in rows_after if r.get("reference_answer_b") and not r.get("reference_answer_b_error"))
    cache_hits = sum(1 for r in rows_after if r.get("reference_answer_b_cache_hit"))
    errors = n - successes
    avg_chars = (
        sum(len(str(r.get("reference_answer_b") or "")) for r in rows_after) / max(1, successes)
    )
    return {
        "rows_processed": n,
        "successes": successes,
        "errors": errors,
        "cache_hits": cache_hits,
        "average_response_chars": round(avg_chars, 1),
        "writer_model": f"{writer_spec.provider}:{writer_spec.model}",
        "wall_clock_seconds": round(wall_clock_s, 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    writer_spec = parse_model_spec(args.writer_model, default_alias="regen_pair_b_writer")

    all_rows = _read_input_rows(args.input)
    print(f"Loaded {len(all_rows)} rows from {args.input}")

    rows = all_rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"Processing {len(rows)} rows (start={args.start}, limit={args.limit}).")

    if not rows:
        print("Nothing to do.")
        return 0

    cache_path = args.cache_dir / "regen_pair_b.jsonl"
    cache = JsonlCache(cache_path, enabled=True)
    cache.load()
    print(f"Cache: {cache_path}")

    router = None if args.dry_run else LLMRouter(max_retries=3)

    started = time.perf_counter()
    workers = max(1, int(args.workers))

    def _process(idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
        outcome = _generate_response(
            writer_spec=writer_spec,
            router=router,
            cache=cache,
            question=str(row.get("question") or ""),
            system_prompt=args.system_prompt,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            dry_run=args.dry_run,
        )
        return _regen_row(row, outcome, writer_spec)

    results_by_index: Dict[int, Dict[str, Any]] = {}
    if workers <= 1 or args.dry_run:
        for idx, row in enumerate(rows):
            results_by_index[idx] = _process(idx, row)
            done = idx + 1
            if done % args.progress_every == 0 or done == len(rows):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(rows) - done) / rate if rate > 0 else 0.0
                print(f"  progress: {done}/{len(rows)} ({rate:.2f} rows/s, ETA {eta/60:.1f} min)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, idx, row): idx for idx, row in enumerate(rows)}
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                results_by_index[idx] = fut.result()
                done += 1
                if done % args.progress_every == 0 or done == len(rows):
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (len(rows) - done) / rate if rate > 0 else 0.0
                    print(
                        f"  progress: {done}/{len(rows)} ({rate:.2f} rows/s, ETA {eta/60:.1f} min)",
                        flush=True,
                    )

    out_rows = [results_by_index[i] for i in sorted(results_by_index.keys())]
    _write_output_rows(args.output, out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.output}")

    summary = _summarize(out_rows, time.perf_counter() - started, writer_spec)
    summary["input"] = str(args.input)
    summary["output"] = str(args.output)
    summary["dry_run"] = bool(args.dry_run)
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
