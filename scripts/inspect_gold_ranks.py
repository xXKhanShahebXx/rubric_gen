"""Quick diagnostic: where does the gold reference rank under each method?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print gold-reference rank/score per example for each method.")
    parser.add_argument("run_dir", type=Path, help="Path to a pipeline run directory containing examples/.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "rrd_uniform",
            "rrd_whitened_uniform",
            "compressed_bank_uniform",
            "production_bank_uniform",
            "one_shot_uniform",
        ],
    )
    args = parser.parse_args(argv)

    examples_dir = args.run_dir / "examples"
    artifacts = sorted(examples_dir.glob("*.json"))
    if not artifacts:
        print(f"No example artifacts under {examples_dir}")
        return 1

    methods: List[str] = list(args.methods)
    gold_ranks: Dict[str, List[int]] = {m: [] for m in methods}
    gold_scores: Dict[str, List[float]] = {m: [] for m in methods}
    rows = []

    for path in artifacts:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        source_id = artifact["example"].get("source_id", path.stem)[:14]
        method_cells = []
        for method in methods:
            payload = artifact["methods"].get(method, {})
            ranking = payload.get("ranking", [])
            gold = next((r for r in ranking if r.get("source_label") == "reference_note"), None)
            if gold is None:
                method_cells.append("-")
                continue
            method_cells.append(f"r{gold['rank']}/{len(ranking)} s={gold['score']:.2f}")
            gold_ranks[method].append(int(gold["rank"]))
            gold_scores[method].append(float(gold["score"]))
        rows.append((source_id, method_cells))

    header = f"{'id':<14}  " + "  ".join(f"{m[:22]:>22}" for m in methods)
    print(header)
    print("-" * len(header))
    for source_id, cells in rows:
        print(f"{source_id:<14}  " + "  ".join(f"{c:>22}" for c in cells))

    print()
    print("aggregate (n =", len(artifacts), "):")
    print(f"  {'method':<32}  {'rank1_hits':>10}  {'mean_rank':>9}  {'mean_score':>10}")
    for method in methods:
        ranks = gold_ranks[method]
        scores = gold_scores[method]
        r1 = sum(1 for r in ranks if r == 1)
        mr = sum(ranks) / len(ranks) if ranks else 0.0
        ms = sum(scores) / len(scores) if scores else 0.0
        print(f"  {method:<32}  {r1:>10}  {mr:>9.2f}  {ms:>10.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
