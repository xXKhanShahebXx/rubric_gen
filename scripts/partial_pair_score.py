"""Tiny diagnostic: print the in-progress pair-preference accuracy for a run dir.

Usage:
    python scripts/partial_pair_score.py artifacts/medical_rl_validation/runs/medical_v47_pair_validation_gpt41

Aggregates the metrics across however many per-example artifacts are on disk so
far. Useful while a long validation run is still in flight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rubric_gen.evaluation.reporting import aggregate_method_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    artifact_paths = sorted((args.run_dir / "examples").glob("*.json"))
    if not artifact_paths:
        print(f"No example artifacts under {args.run_dir / 'examples'}")
        return 1

    artifacts = []
    for path in artifact_paths:
        try:
            artifacts.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    rows = aggregate_method_metrics(artifacts)
    print(f"Aggregating across {len(artifacts)} completed examples (snapshot, run still in progress).\n")
    print(f"{'method':<32}  {'pair_acc':>9}  {'correct/n':>12}  {'strong/weak':>12}  {'mean_margin':>11}")
    print("-" * 86)
    for r in rows:
        pair_acc = float(r.get("pair_preference_accuracy", 0.0) or 0.0)
        pair_correct = int(r.get("pair_preference_correct", 0) or 0)
        pair_n = int(r.get("pair_preference_evaluable", 0) or 0)
        sw = float(r.get("strong_vs_weak_accuracy", 0.0) or 0.0)
        margin = float(r.get("mean_strong_weak_margin", 0.0) or 0.0)
        print(
            f"{r['method']:<32}  {pair_acc:>9.3f}  {str(pair_correct) + '/' + str(pair_n):>12}  "
            f"{sw:>12.3f}  {margin:>11.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
