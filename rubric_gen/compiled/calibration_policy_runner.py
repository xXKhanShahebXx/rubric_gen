"""
CLI for deriving per-profile calibration eligibility from held-out benchmark runs.

Run:
    python -m rubric_gen.compiled.calibration_policy_runner
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from rubric_gen.compiled.gold_refinement import derive_calibration_profile_policy
from rubric_gen.compiled.serialize import write_json


def _summary_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / "run_summary.json"
    return candidate


def _load_summary(path: Path) -> Dict[str, Any]:
    summary_path = _summary_path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Run summary at {summary_path} did not contain a JSON object.")
    payload = dict(payload)
    paths = dict(payload.get("paths", {}))
    paths["summary_source"] = str(summary_path)
    payload["paths"] = paths
    return payload


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a per-profile calibration eligibility policy by comparing a baseline benchmark run "
            "to a calibration-applied held-out benchmark run."
        )
    )
    parser.add_argument("--baseline-run", type=Path, required=True, help="Baseline run dir or run_summary.json path.")
    parser.add_argument("--apply-run", type=Path, required=True, help="Calibration-applied run dir or run_summary.json path.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for calibration_profile_policy.json.",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=5,
        help="Minimum held-out examples required before a profile can be enabled (default: 5).",
    )
    args = parser.parse_args(argv)

    baseline_summary = _load_summary(args.baseline_run)
    apply_summary = _load_summary(args.apply_run)
    policy = derive_calibration_profile_policy(
        baseline_summary=baseline_summary,
        apply_summary=apply_summary,
        min_examples=max(1, args.min_examples),
    )
    write_json(args.out, policy)
    print(f"Wrote calibration policy to: {args.out}")
    print(f"enabled_profiles={','.join(policy.get('enabled_profiles', [])) or '(none)'}")


if __name__ == "__main__":
    main()
