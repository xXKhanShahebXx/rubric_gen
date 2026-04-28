"""
Offline replay of the math independent-solver verifier against existing artifacts.

For each ``livebench-math`` artifact in a final-eval / OOF run directory, runs the independent
solver with one extra GPT-4o call (cached in a JSONL file so subsequent runs are free), compares
the solver's canonical answer against the candidate responses, and reports how many of the
math failures would be resolved by layering this verifier on top of the existing decisions.

Usage::

    python -m rubric_gen.compiled.math_solver_replay \
        --run-dir artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v2_full_seed29 \
        --judge-model openai:gpt-4o-2024-05-13

The replay also writes a structured JSON report under ``summaries/math_solver_replay.json``
with per-pair diagnostics so we can inspect *which* math failures are solver-resolvable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from rubric_gen.compiled.math_independent_solver_verifier import (
    _canonicalize_answer,
    _extract_candidate_final_value,
    evaluate_math_independent_solver,
)
from rubric_gen.config import parse_model_spec
from rubric_gen.llm_client import LLMRouter
from rubric_gen.storage import JsonlCache


def _flip_decision(decision: str) -> str:
    if decision == "A>B":
        return "B>A"
    if decision == "B>A":
        return "A>B"
    return decision


def _double_order_correct(label: str, decision_original: str, decision_reversed: str) -> bool:
    flip = _flip_decision(label)
    counter = 0
    for d in (decision_original, _flip_decision(decision_reversed)):
        if d == label:
            counter += 1
        elif d == flip:
            counter -= 1
    return counter > 0


@dataclass
class SolverReplayDiagnostic:
    pair_id: str
    label: str
    original_decision: str
    original_decision_reversed: str
    original_correct: bool
    new_decision: str
    new_decision_reversed: str
    new_correct: bool
    solver_canonical_answer: str
    a_canonical_answer: str
    b_canonical_answer: str
    solver_triggered: bool
    solver_reason: str
    response_a_text_preview: str = ""
    response_b_text_preview: str = ""
    question_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolverReplaySummary:
    run_dir: str
    math_pair_count: int
    original_correct: int
    new_correct: int
    solver_triggered_count: int
    solver_resolved_failures: int
    solver_caused_regressions: int
    by_pair_diagnostics: List[SolverReplayDiagnostic] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "compiled_judgebench_math_solver_replay_v1",
            "run_dir": self.run_dir,
            "math_pair_count": self.math_pair_count,
            "original_correct": self.original_correct,
            "new_correct": self.new_correct,
            "original_accuracy_pct": round(100.0 * self.original_correct / max(1, self.math_pair_count), 4),
            "new_accuracy_pct": round(100.0 * self.new_correct / max(1, self.math_pair_count), 4),
            "solver_triggered_count": self.solver_triggered_count,
            "solver_resolved_failures": self.solver_resolved_failures,
            "solver_caused_regressions": self.solver_caused_regressions,
            "diagnostics": [d.to_dict() for d in self.by_pair_diagnostics],
        }


def _load_artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    examples_dirs = list(run_dir.rglob("examples"))
    artifacts: List[Dict[str, Any]] = []
    seen_ids = set()
    for d in examples_dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pair_id = str(((payload.get("pair", {}) or {}).get("pair_id") or "")).strip()
            if not pair_id or pair_id in seen_ids:
                continue
            seen_ids.add(pair_id)
            artifacts.append(payload)
    return artifacts


def replay_math_solver(
    run_dir: Path,
    *,
    judge_model_spec: str,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
    limit: Optional[int] = None,
) -> SolverReplaySummary:
    artifacts = _load_artifacts(Path(run_dir))
    math_artifacts = [
        a for a in artifacts
        if str(((a.get("pair", {}) or {}).get("source_family") or "")).strip() == "livebench-math"
    ]
    if limit is not None and limit > 0:
        math_artifacts = math_artifacts[: int(limit)]

    cache: Optional[JsonlCache] = None
    if use_cache:
        cache_path = Path(cache_path) if cache_path else (Path(run_dir) / "cache" / "math_solver_replay.jsonl")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = JsonlCache(cache_path, enabled=True)

    model_spec = parse_model_spec(judge_model_spec, default_alias="math-solver")
    router = LLMRouter()

    diagnostics: List[SolverReplayDiagnostic] = []
    original_correct = 0
    new_correct = 0
    solver_triggered = 0
    resolved_failures = 0
    caused_regressions = 0

    for artifact in math_artifacts:
        pair = artifact.get("pair", {}) or {}
        pair_id = str(pair.get("pair_id") or "")
        label = str(pair.get("label") or "")
        question = str(pair.get("question") or "")
        response_a = str(pair.get("response_A") or "")
        response_b = str(pair.get("response_B") or "")
        wu_result = (artifact.get("scoring", {}) or {}).get("whitened_uniform", {}).get("result", {}) or {}
        original_decision = str(wu_result.get("decision") or "")
        original_decision_reversed = str(wu_result.get("decision_reversed") or "")
        original_ok = _double_order_correct(label, original_decision, original_decision_reversed)
        if original_ok:
            original_correct += 1

        outcome = evaluate_math_independent_solver(
            question=question,
            response_a=response_a,
            response_b=response_b,
            model_spec=model_spec,
            router=router,
            cache=cache,
        )
        if outcome.triggered:
            solver_triggered += 1
            new_decision = outcome.recommended_decision
            new_decision_reversed = _flip_decision(new_decision)
        else:
            new_decision = original_decision
            new_decision_reversed = original_decision_reversed
        new_ok = _double_order_correct(label, new_decision, new_decision_reversed)
        if new_ok:
            new_correct += 1
        if not original_ok and new_ok:
            resolved_failures += 1
        elif original_ok and not new_ok:
            caused_regressions += 1

        diagnostics.append(
            SolverReplayDiagnostic(
                pair_id=pair_id,
                label=label,
                original_decision=original_decision,
                original_decision_reversed=original_decision_reversed,
                original_correct=original_ok,
                new_decision=new_decision,
                new_decision_reversed=new_decision_reversed,
                new_correct=new_ok,
                solver_canonical_answer=outcome.solver_canonical_answer,
                a_canonical_answer=outcome.a_canonical_answer,
                b_canonical_answer=outcome.b_canonical_answer,
                solver_triggered=outcome.triggered,
                solver_reason=outcome.reason,
                question_preview=question[:200],
                response_a_text_preview=response_a[:200],
                response_b_text_preview=response_b[:200],
            )
        )

    return SolverReplaySummary(
        run_dir=str(run_dir),
        math_pair_count=len(math_artifacts),
        original_correct=original_correct,
        new_correct=new_correct,
        solver_triggered_count=solver_triggered,
        solver_resolved_failures=resolved_failures,
        solver_caused_regressions=caused_regressions,
        by_pair_diagnostics=diagnostics,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline replay of the math independent-solver verifier against existing artifacts."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge-model", type=str, default="openai:gpt-4o-2024-05-13")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw)
    summary = replay_math_solver(
        args.run_dir,
        judge_model_spec=args.judge_model,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
        limit=args.limit,
    )
    out_path = args.out_path or (Path(args.run_dir) / "summaries" / "math_solver_replay.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    headline = {
        "math_pair_count": summary.math_pair_count,
        "original_correct": summary.original_correct,
        "new_correct": summary.new_correct,
        "original_accuracy_pct": round(100.0 * summary.original_correct / max(1, summary.math_pair_count), 2),
        "new_accuracy_pct": round(100.0 * summary.new_correct / max(1, summary.math_pair_count), 2),
        "solver_triggered_count": summary.solver_triggered_count,
        "solver_resolved_failures": summary.solver_resolved_failures,
        "solver_caused_regressions": summary.solver_caused_regressions,
    }
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
