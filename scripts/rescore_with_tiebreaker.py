"""Post-hoc rescore of an existing pair-preference validation run.

Applies a JudgeBench-style tiebreaker cascade to the already-written
per-example artifacts under ``<run-dir>/examples/`` -- no need to re-run
the LLM pipeline, no re-generation of rubric banks. The tiebreaker
cascade for the headline method (default ``rrd_whitened_uniform``):

  1. Strict score: predicted = arg-max(score_a, score_b). If they differ,
     done.
  2. Score tie -> consult ``rrd_uniform`` on the same artifact. If
     uniform produces a strict winner, use it (``--strategy uniform``).
  3. Still tie -> direct GPT-4o pair judge call (``--strategy
     uniform_then_judge``, the default). Mirrors
     ``rubric_gen/compiled/judgebench_eval.py``'s
     ``uniform_breaks_strict_tie`` cascade plus a final LLM fallback.

Writes ``summary_v3.md``, ``summary_v3.json``, ``method_metrics_v3.csv``
side-by-side with the originals so the audit trail is preserved. Caches
LLM tiebreaker calls in ``<run-dir>/../cache/pair_tiebreaker.jsonl``
(or ``--cache-path``) so re-runs are free.

Cost: typically ~0.10-0.15 USD per 100 score-tied rows under
``openai:gpt-4o`` at temperature 0. For the 4k validation run,
~498 ties * 1 call ~= 0.50-2.50 USD total.

Usage::

  python scripts/rescore_with_tiebreaker.py \\
    --run-dir artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt5_b_4k \\
    --strategy uniform_then_judge --workers 8

  # Smoke on 200 rows -- only rescore rows 0..199 (still rewrites a fresh summary_v3.md)
  python scripts/rescore_with_tiebreaker.py --run-dir <run> --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rubric_gen.config import parse_model_spec
from rubric_gen.evaluation.pair_tiebreaker import (
    FormatPriorOutcome,
    direct_judge_pair,
    direct_judge_pair_anti_tie,
    format_prior_predict,
    predict_via_score,
    TiebreakerOutcome,
)
from rubric_gen.llm_client import LLMRouter
from rubric_gen.storage import JsonlCache
from rubric_gen.types import ModelSpec

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


PRIMARY_METHOD = "rrd_whitened_uniform"
TIEBREAK_METHOD = "rrd_uniform"
ALLOWED_STRATEGIES = (
    "none",
    "uniform",
    "judge",
    "uniform_then_judge",
    # v2 Tier C cascades that prepend the free format-prior heuristic and
    # optionally swap the LLM judge for the anti-tie variant.
    "format_then_judge",
    "format_then_uniform_then_judge",
    "format_then_uniform_then_anti_tie",
)


# ---------------------------------------------------------------------------
# Per-row resolution
# ---------------------------------------------------------------------------


@dataclass
class RowDecision:
    sid: str
    gold: str
    pred: str  # final pred after cascade: "a"/"b"/"tie"/""
    primary_pred: str  # pred from PRIMARY_METHOD score alone
    secondary_pred: str = ""
    judge_outcome: Optional[TiebreakerOutcome] = None
    format_outcome: Optional[FormatPriorOutcome] = None
    decision_policy: str = "primary"  # primary / uniform / judge / unresolved / format / format_then_*


def _pair_anchor(ex: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    cands = ex.get("candidates") or []
    pa = next((c for c in cands if c.get("source_label") == "pair_response_a"), None)
    pb = next((c for c in cands if c.get("source_label") == "pair_response_b"), None)
    return pa, pb


def _scores_for(method_obj: Dict[str, Any], cand_a: str, cand_b: str) -> Tuple[Optional[float], Optional[float]]:
    by_id = {r.get("candidate_id"): r for r in (method_obj.get("ranking") or [])}
    a_row = by_id.get(cand_a)
    b_row = by_id.get(cand_b)
    sa = a_row.get("score") if a_row else None
    sb = b_row.get("score") if b_row else None
    return (
        float(sa) if sa is not None else None,
        float(sb) if sb is not None else None,
    )


def _source_id(example_id: str) -> str:
    if "__" in example_id:
        return example_id.split("__", 1)[1]
    return example_id


def _resolve_one(
    *,
    ex: Dict[str, Any],
    strategy: str,
    judge_model: ModelSpec,
    router: Optional[LLMRouter],
    cache: JsonlCache,
    judge_temperature: float,
    dry_run: bool,
) -> RowDecision:
    example = ex.get("example", {}) or {}
    sid = _source_id(example.get("example_id", ""))
    gold = (example.get("pair_correct_label") or "").strip().lower()
    methods = ex.get("methods", {}) or {}
    primary = methods.get(PRIMARY_METHOD, {}) or {}
    secondary = methods.get(TIEBREAK_METHOD, {}) or {}

    pa, pb = _pair_anchor(ex)
    if not pa or not pb or gold not in {"a", "b"}:
        return RowDecision(sid=sid, gold=gold, pred="", primary_pred="", decision_policy="unevaluable")

    cand_a = pa["candidate_id"]
    cand_b = pb["candidate_id"]
    primary_a, primary_b = _scores_for(primary, cand_a, cand_b)
    primary_pred = predict_via_score(primary_a, primary_b)
    if primary_pred in {"a", "b"}:
        return RowDecision(sid=sid, gold=gold, pred=primary_pred, primary_pred=primary_pred, decision_policy="primary")
    if primary_pred == "" or strategy == "none":
        # No score on primary or strategy explicitly disabled.
        return RowDecision(
            sid=sid, gold=gold, pred=primary_pred, primary_pred=primary_pred,
            decision_policy="primary",
        )

    # v2 Tier C1: format-prior heuristic (no LLM call).  Run this first when
    # the strategy includes the ``format_*`` prefix, so we resolve the easy
    # format-mismatch ties for free before paying for an LLM call.
    format_outcome: Optional[FormatPriorOutcome] = None
    question = str(example.get("task_prompt") or "")
    response_a = str(example.get("pair_response_a") or pa.get("text") or "")
    response_b = str(example.get("pair_response_b") or pb.get("text") or "")
    if strategy.startswith("format_"):
        format_outcome = format_prior_predict(
            question=question,
            response_a=response_a,
            response_b=response_b,
        )
        if format_outcome.verdict in {"a", "b"}:
            return RowDecision(
                sid=sid,
                gold=gold,
                pred=format_outcome.verdict,
                primary_pred=primary_pred,
                format_outcome=format_outcome,
                decision_policy="format",
            )

    # Score tie. Try secondary aggregation if requested.
    secondary_pred = ""
    if strategy in {"uniform", "uniform_then_judge", "format_then_uniform_then_judge", "format_then_uniform_then_anti_tie"}:
        sec_a, sec_b = _scores_for(secondary, cand_a, cand_b)
        secondary_pred = predict_via_score(sec_a, sec_b)
        if secondary_pred in {"a", "b"}:
            return RowDecision(
                sid=sid, gold=gold, pred=secondary_pred, primary_pred=primary_pred,
                secondary_pred=secondary_pred, format_outcome=format_outcome,
                decision_policy="uniform",
            )

    # Still tied. Try direct LLM judge if requested.
    if strategy in {
        "judge",
        "uniform_then_judge",
        "format_then_judge",
        "format_then_uniform_then_judge",
        "format_then_uniform_then_anti_tie",
    }:
        # v2 Tier C2: anti-tie prompt variant on the new cascade.
        if strategy == "format_then_uniform_then_anti_tie":
            judge = direct_judge_pair_anti_tie(
                question=question,
                response_a=response_a,
                response_b=response_b,
                judge_model=judge_model,
                router=router,
                cache=cache,
                temperature=judge_temperature,
                dry_run=dry_run,
            )
            judge_policy = "judge_anti_tie"
            tied_policy = "judge_anti_tie_tied"
        else:
            judge = direct_judge_pair(
                question=question,
                response_a=response_a,
                response_b=response_b,
                judge_model=judge_model,
                router=router,
                cache=cache,
                temperature=judge_temperature,
                dry_run=dry_run,
            )
            judge_policy = "judge"
            tied_policy = "judge_tied"
        verdict = judge.verdict
        if verdict in {"a", "b"}:
            return RowDecision(
                sid=sid, gold=gold, pred=verdict, primary_pred=primary_pred,
                secondary_pred=secondary_pred, judge_outcome=judge,
                format_outcome=format_outcome,
                decision_policy=judge_policy,
            )
        return RowDecision(
            sid=sid, gold=gold, pred="tie", primary_pred=primary_pred,
            secondary_pred=secondary_pred, judge_outcome=judge,
            format_outcome=format_outcome,
            decision_policy=tied_policy,
        )

    # Strategy was "uniform" only and that also tied.
    return RowDecision(
        sid=sid, gold=gold, pred="tie", primary_pred=primary_pred,
        secondary_pred=secondary_pred, format_outcome=format_outcome,
        decision_policy="uniform_tied",
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class Summary:
    n: int = 0
    correct: int = 0
    wrong: int = 0
    tie: int = 0
    unevaluable: int = 0
    by_policy: Dict[str, int] = field(default_factory=dict)
    by_policy_correct: Dict[str, int] = field(default_factory=dict)
    judge_calls: int = 0
    judge_cache_hits: int = 0
    by_gold: Dict[str, Dict[str, int]] = field(default_factory=lambda: {"a": {"n": 0, "correct": 0}, "b": {"n": 0, "correct": 0}})


def _summarize(rows: Sequence[RowDecision]) -> Summary:
    s = Summary()
    for r in rows:
        if r.decision_policy == "unevaluable":
            s.unevaluable += 1
            continue
        s.n += 1
        s.by_policy[r.decision_policy] = s.by_policy.get(r.decision_policy, 0) + 1
        if r.gold in s.by_gold:
            s.by_gold[r.gold]["n"] += 1
        if r.pred == r.gold:
            s.correct += 1
            s.by_policy_correct[r.decision_policy] = s.by_policy_correct.get(r.decision_policy, 0) + 1
            if r.gold in s.by_gold:
                s.by_gold[r.gold]["correct"] += 1
        elif r.pred in {"tie", ""}:
            s.tie += 1
        else:
            s.wrong += 1
        if r.judge_outcome is not None:
            s.judge_calls += 1
            if r.judge_outcome.cache_hit:
                s.judge_cache_hits += 1
    return s


def _format_summary_md(s: Summary, *, run_dir: Path, strategy: str, judge_model: str) -> str:
    acc = (s.correct / s.n * 100) if s.n else 0.0
    lines = ["# Pair-preference rescore (post-hoc tiebreaker)", ""]
    lines.append(f"- Run dir       : `{run_dir}`")
    lines.append(f"- Strategy      : `{strategy}`")
    lines.append(f"- Primary       : `{PRIMARY_METHOD}`")
    lines.append(f"- Secondary     : `{TIEBREAK_METHOD}`")
    lines.append(f"- Judge model   : `{judge_model}`")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---:|")
    lines.append(f"| Evaluable rows | {s.n} |")
    lines.append(f"| Correct | {s.correct} |")
    lines.append(f"| Wrong (decided != gold) | {s.wrong} |")
    lines.append(f"| Tie (still ambiguous after cascade) | {s.tie} |")
    lines.append(f"| **Accuracy** | **{acc:.2f}%** |")
    lines.append(f"| Direct-judge calls made | {s.judge_calls} (cache hits {s.judge_cache_hits}) |")
    lines.append("")
    lines.append("## Decision policy breakdown (where the final pred came from)")
    lines.append("")
    lines.append("| Policy | N | Correct | Acc on policy |")
    lines.append("|---|---:|---:|---:|")
    for policy in sorted(s.by_policy.keys()):
        n = s.by_policy[policy]
        c = s.by_policy_correct.get(policy, 0)
        a = c / n * 100 if n else 0.0
        lines.append(f"| {policy} | {n} | {c} | {a:.2f}% |")
    lines.append("")
    lines.append("## Per-gold breakdown")
    lines.append("")
    lines.append("| Gold | N | Correct | Recall |")
    lines.append("|---|---:|---:|---:|")
    for gold in ("a", "b"):
        n = s.by_gold[gold]["n"]
        c = s.by_gold[gold]["correct"]
        r = c / n * 100 if n else 0.0
        lines.append(f"| {gold.upper()} | {n} | {c} | {r:.2f}% |")
    if s.unevaluable:
        lines.append("")
        lines.append(f"_{s.unevaluable} rows skipped as unevaluable (missing pair anchors / labels)._")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--strategy",
        type=str,
        choices=list(ALLOWED_STRATEGIES),
        default="uniform_then_judge",
        help="Cascade. Default 'uniform_then_judge': try rrd_uniform when rrd_whitened_uniform ties, then GPT-4o.",
    )
    p.add_argument("--judge-model", type=str, default="openai:gpt-4o")
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="JSONL cache for tiebreaker calls. Default: <run-dir>/../cache/pair_tiebreaker.jsonl",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Only rescore the first N example artifacts (sorted). 0 = all.")
    p.add_argument("--dry-run", action="store_true", help="Skip all LLM calls (for parse / smoke checks).")
    p.add_argument("--suffix", type=str, default="_v3", help="Suffix appended to summary filenames.")
    p.add_argument("--progress-every", type=int, default=50)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir: Path = args.run_dir
    examples_dir = run_dir / "examples"
    files = sorted(examples_dir.glob("*.json"))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    print(f"Run dir       : {run_dir}")
    print(f"Examples      : {len(files)}")
    print(f"Strategy      : {args.strategy}")

    judge_model = parse_model_spec(args.judge_model, default_alias="pair_tiebreaker_judge")
    print(f"Judge         : {judge_model.provider}:{judge_model.model}")

    cache_path = args.cache_path or (run_dir.parent / "cache" / "pair_tiebreaker.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = JsonlCache(cache_path, enabled=True)
    cache.load()
    print(f"Cache         : {cache_path}")

    router = None if args.dry_run else LLMRouter(max_retries=3)
    workers = max(1, int(args.workers))
    started = time.perf_counter()

    artifacts = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    decisions: List[RowDecision] = [None] * len(artifacts)  # type: ignore[list-item]

    def _job(idx: int) -> Tuple[int, RowDecision]:
        return idx, _resolve_one(
            ex=artifacts[idx],
            strategy=args.strategy,
            judge_model=judge_model,
            router=router,
            cache=cache,
            judge_temperature=float(args.judge_temperature),
            dry_run=args.dry_run,
        )

    if workers <= 1 or args.dry_run:
        for i in range(len(artifacts)):
            idx, dec = _job(i)
            decisions[idx] = dec
            if (i + 1) % args.progress_every == 0 or (i + 1) == len(artifacts):
                el = time.perf_counter() - started
                rate = (i + 1) / el if el > 0 else 0.0
                print(f"  progress: {i+1}/{len(artifacts)} ({rate:.2f} rows/s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, i) for i in range(len(artifacts))]
            done = 0
            for fut in as_completed(futures):
                idx, dec = fut.result()
                decisions[idx] = dec
                done += 1
                if done % args.progress_every == 0 or done == len(artifacts):
                    el = time.perf_counter() - started
                    rate = done / el if el > 0 else 0.0
                    eta = (len(artifacts) - done) / rate / 60 if rate > 0 else 0.0
                    print(f"  progress: {done}/{len(artifacts)} ({rate:.2f} rows/s, ETA {eta:.1f} min)", flush=True)

    s = _summarize(decisions)
    elapsed = time.perf_counter() - started

    md = _format_summary_md(s, run_dir=run_dir, strategy=args.strategy,
                             judge_model=f"{judge_model.provider}:{judge_model.model}")
    summary_dir = run_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    md_path = summary_dir / f"summary{args.suffix}.md"
    json_path = summary_dir / f"summary{args.suffix}.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "headline": {
                    "n": s.n, "correct": s.correct, "wrong": s.wrong, "tie": s.tie,
                    "accuracy": (s.correct / s.n) if s.n else 0.0,
                    "unevaluable": s.unevaluable,
                    "judge_calls": s.judge_calls,
                    "judge_cache_hits": s.judge_cache_hits,
                    "wall_clock_seconds": round(elapsed, 1),
                    "strategy": args.strategy,
                    "primary_method": PRIMARY_METHOD,
                    "tiebreak_method": TIEBREAK_METHOD,
                    "judge_model": f"{judge_model.provider}:{judge_model.model}",
                },
                "by_policy": s.by_policy,
                "by_policy_correct": s.by_policy_correct,
                "by_gold": s.by_gold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(md)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wall clock: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
