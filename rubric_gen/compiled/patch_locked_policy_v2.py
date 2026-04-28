"""
One-time data-fix to backport v2 policy fields into a previously written
``frozen_policy/locked_policy.json`` whose builder run predates the fix that threads ``v2_config``
into ``build_initial_frozen_policy`` for the locked policy.

This is needed for a single class of runs: ``jb_*_v2_full_*`` runs whose train-only command set
v2 flags; the per-fold policies received the v2 settings (so OOF metrics include v2) but the
locked policy used at final-eval time was built without the v2 fields. Patching the locked
policy + ``frozen_policy_hash`` lets the existing final-eval CLI run with the full v2 stack
without re-running the entire 320 OOF.

Mechanism hash is unaffected because ``_build_mechanism_spec`` does not include any v2 fields.

Usage::

    python -m rubric_gen.compiled.patch_locked_policy_v2 \
        --run-dir artifacts/compiled_judgebench_train_only_runs/jb_320_v2_full_seed29 \
        --rubric-library-path artifacts/rubric_library/v1/library.json \
        --library-retrieval-top-k 6 \
        --self-consistency-n 5 \
        --self-consistency-temperature 0.7 \
        --v2-wide-discriminator-gate \
        --holistic-judge \
        --enable-rrd-filters \
        --rrd-redundancy-threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rubric_gen.compiled.judgebench_eval import (
    _DEFAULT_SELF_CONSISTENCY_N,
    _DEFAULT_SELF_CONSISTENCY_TEMPERATURE,
    _policy_core_hash,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backport v2 policy fields into an existing locked_policy.json."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rubric-library-path", type=str, default="")
    parser.add_argument("--library-retrieval-top-k", type=int, default=0)
    parser.add_argument("--self-consistency-n", type=int, default=_DEFAULT_SELF_CONSISTENCY_N)
    parser.add_argument(
        "--self-consistency-temperature", type=float, default=_DEFAULT_SELF_CONSISTENCY_TEMPERATURE
    )
    parser.add_argument("--v2-wide-discriminator-gate", action="store_true")
    parser.add_argument("--holistic-judge", action="store_true")
    parser.add_argument("--enable-rrd-filters", action="store_true")
    parser.add_argument("--rrd-redundancy-threshold", type=float, default=0.9)
    parser.add_argument(
        "--library-retrieval-family-top-k",
        action="append",
        default=[],
        help="Per-family library top-k override in 'family=N' form (repeatable). "
        "Use 'livecodebench=0' to disable library on code, etc.",
    )
    parser.add_argument("--family-strict-library-mode", action="store_true")
    parser.add_argument("--math-independent-solver", action="store_true")
    parser.add_argument("--math-solver-samples", type=int, default=1)
    parser.add_argument("--math-solver-temperature", type=float, default=0.5)
    parser.add_argument("--code-execution-verifier", action="store_true")
    parser.add_argument("--code-execution-timeout-s", type=float, default=10.0)
    parser.add_argument("--code-execution-min-margin", type=float, default=0.34)
    parser.add_argument("--math-solver-use-sympy", action="store_true")
    parser.add_argument("--few-shot-train", action="store_true")
    parser.add_argument("--few-shot-top-k", type=int, default=3)
    parser.add_argument("--few-shot-train-dataset", type=str, default="")
    parser.add_argument("--few-shot-official-dataset", type=str, default="")
    parser.add_argument("--rubric-satisfaction-samples", type=int, default=1)
    parser.add_argument("--rubric-satisfaction-temperature", type=float, default=0.4)
    parser.add_argument("--discriminator-self-critique", action="store_true")
    parser.add_argument("--mmlu-independent-answerer", action="store_true")
    parser.add_argument("--mmlu-answerer-samples", type=int, default=1)
    parser.add_argument("--mmlu-answerer-temperature", type=float, default=0.5)
    parser.add_argument(
        "--mmlu-answerer-model",
        type=str,
        default="",
        help="Optional provider:model for the MMLU independent answerer (e.g. anthropic:claude-opus-4-20250514). "
        "Empty -> use the default GPT-4o scoring model.",
    )
    parser.add_argument(
        "--mmlu-answerer-secondary-model",
        type=str,
        default="",
        help="Optional second-opinion provider:model. When set, the verifier requires the secondary "
        "answerer to agree on the chosen letter before overriding the base verdict (dual consensus).",
    )
    parser.add_argument("--mmlu-answerer-secondary-samples", type=int, default=1)
    parser.add_argument("--mmlu-answerer-secondary-temperature", type=float, default=0.0)
    parser.add_argument(
        "--math-solver-model",
        type=str,
        default="",
        help="Optional provider:model for the math independent solver (e.g. anthropic:claude-opus-4-20250514). "
        "Empty -> use the default GPT-4o scoring model.",
    )
    parser.add_argument("--reasoning-independent-solver", action="store_true")
    parser.add_argument("--reasoning-solver-samples", type=int, default=1)
    parser.add_argument("--reasoning-solver-temperature", type=float, default=0.5)
    parser.add_argument(
        "--reasoning-solver-model",
        type=str,
        default="",
        help="Optional provider:model for the reasoning independent solver. Empty -> default GPT-4o.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the patched policy without writing.")
    return parser


def _parse_family_int_overrides(values: List[str]) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    for v in values or []:
        if "=" not in v:
            raise ValueError(f"Expected family=N, got {v!r}")
        family, raw = v.split("=", 1)
        family = family.strip()
        try:
            parsed[family] = max(0, min(16, int(raw.strip())))
        except ValueError as exc:
            raise ValueError(f"Bad family override {v!r}: {exc}") from exc
    return parsed


def _patch_policy(policy_path: Path, args: argparse.Namespace) -> dict:
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["self_consistency_n"] = max(1, min(9, int(args.self_consistency_n or 1)))
    payload["self_consistency_temperature"] = max(
        0.0, min(1.5, float(args.self_consistency_temperature or 0.0))
    )
    payload["v2_wide_discriminator_gate"] = bool(args.v2_wide_discriminator_gate)
    payload["holistic_judge_enabled"] = bool(args.holistic_judge)
    payload["library_retrieval_top_k"] = max(0, min(16, int(args.library_retrieval_top_k or 0)))
    payload["rubric_library_path"] = str(args.rubric_library_path or "").strip()
    payload["enable_rrd_filters"] = bool(args.enable_rrd_filters)
    payload["rrd_redundancy_threshold"] = max(0.0, min(1.0, float(args.rrd_redundancy_threshold)))
    payload["library_retrieval_top_k_by_family"] = _parse_family_int_overrides(
        list(getattr(args, "library_retrieval_family_top_k", []) or [])
    )
    payload["family_strict_library_mode"] = bool(getattr(args, "family_strict_library_mode", False))
    payload["math_independent_solver_enabled"] = bool(getattr(args, "math_independent_solver", False))
    payload["math_solver_samples"] = max(1, min(9, int(getattr(args, "math_solver_samples", 1) or 1)))
    payload["math_solver_temperature"] = max(
        0.0, min(1.5, float(getattr(args, "math_solver_temperature", 0.5) or 0.5))
    )
    payload["code_execution_verifier_enabled"] = bool(getattr(args, "code_execution_verifier", False))
    payload["code_execution_timeout_s"] = max(
        1.0, min(60.0, float(getattr(args, "code_execution_timeout_s", 10.0) or 10.0))
    )
    payload["code_execution_min_margin"] = max(
        0.0, min(1.0, float(getattr(args, "code_execution_min_margin", 0.34) or 0.34))
    )
    payload["math_solver_use_sympy"] = bool(getattr(args, "math_solver_use_sympy", False))
    payload["few_shot_train_enabled"] = bool(getattr(args, "few_shot_train", False))
    payload["few_shot_top_k"] = max(0, min(10, int(getattr(args, "few_shot_top_k", 3) or 3)))
    payload["few_shot_train_dataset_path"] = str(
        getattr(args, "few_shot_train_dataset", "") or ""
    )
    payload["few_shot_official_dataset_path"] = str(
        getattr(args, "few_shot_official_dataset", "") or ""
    )
    payload["rubric_satisfaction_samples"] = max(
        1, min(9, int(getattr(args, "rubric_satisfaction_samples", 1) or 1))
    )
    payload["rubric_satisfaction_temperature"] = max(
        0.0, min(1.5, float(getattr(args, "rubric_satisfaction_temperature", 0.4) or 0.4))
    )
    payload["discriminator_self_critique_enabled"] = bool(
        getattr(args, "discriminator_self_critique", False)
    )
    payload["mmlu_independent_answerer_enabled"] = bool(
        getattr(args, "mmlu_independent_answerer", False)
    )
    payload["mmlu_answerer_samples"] = max(
        1, min(9, int(getattr(args, "mmlu_answerer_samples", 1) or 1))
    )
    payload["mmlu_answerer_temperature"] = max(
        0.0, min(1.5, float(getattr(args, "mmlu_answerer_temperature", 0.5) or 0.5))
    )
    payload["mmlu_answerer_model"] = str(getattr(args, "mmlu_answerer_model", "") or "")
    payload["mmlu_answerer_secondary_model"] = str(
        getattr(args, "mmlu_answerer_secondary_model", "") or ""
    )
    payload["mmlu_answerer_secondary_samples"] = max(
        1, min(9, int(getattr(args, "mmlu_answerer_secondary_samples", 1) or 1))
    )
    payload["mmlu_answerer_secondary_temperature"] = max(
        0.0, min(1.5, float(getattr(args, "mmlu_answerer_secondary_temperature", 0.0) or 0.0))
    )
    payload["math_solver_model"] = str(getattr(args, "math_solver_model", "") or "")
    payload["reasoning_independent_solver_enabled"] = bool(
        getattr(args, "reasoning_independent_solver", False)
    )
    payload["reasoning_solver_samples"] = max(
        1, min(9, int(getattr(args, "reasoning_solver_samples", 1) or 1))
    )
    payload["reasoning_solver_temperature"] = max(
        0.0, min(1.5, float(getattr(args, "reasoning_solver_temperature", 0.5) or 0.5))
    )
    payload["reasoning_solver_model"] = str(getattr(args, "reasoning_solver_model", "") or "")

    new_core_hash = _policy_core_hash(payload)
    locking_metadata = dict(payload.get("locking_metadata", {}) or {})
    locking_metadata["frozen_policy_hash"] = new_core_hash
    locking_metadata["v2_backport_applied"] = True
    payload["locking_metadata"] = locking_metadata
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = _build_parser().parse_args(raw)
    run_dir = Path(args.run_dir)
    policy_path = run_dir / "frozen_policy" / "locked_policy.json"
    if not policy_path.exists():
        print(f"ERROR: locked policy not found at {policy_path}", file=sys.stderr)
        return 2

    patched = _patch_policy(policy_path, args)

    summary = {
        "rubric_library_path": patched.get("rubric_library_path"),
        "library_retrieval_top_k": patched.get("library_retrieval_top_k"),
        "v2_wide_discriminator_gate": patched.get("v2_wide_discriminator_gate"),
        "holistic_judge_enabled": patched.get("holistic_judge_enabled"),
        "self_consistency_n": patched.get("self_consistency_n"),
        "self_consistency_temperature": patched.get("self_consistency_temperature"),
        "enable_rrd_filters": patched.get("enable_rrd_filters"),
        "rrd_redundancy_threshold": patched.get("rrd_redundancy_threshold"),
        "new_frozen_policy_hash": patched.get("locking_metadata", {}).get("frozen_policy_hash"),
    }
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("(dry-run: not writing to disk)")
        return 0

    policy_path.write_text(json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote patched locked policy to {policy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
