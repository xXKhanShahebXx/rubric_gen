"""Print one sample judge reasoning per verdict from a relabeled JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--per-verdict", type=int, default=1)
    parser.add_argument("--confidence", default="high")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Total rows: {len(rows)}\n")
    for verdict in ("A", "B", "TIE"):
        matches = [
            r
            for r in rows
            if r.get("judge_verdict") == verdict
            and (not args.confidence or r.get("judge_confidence") == args.confidence)
        ]
        for sample in matches[: args.per_verdict]:
            q = sample.get("question", "")
            print(f"--- verdict={verdict} confidence={sample.get('judge_confidence')} ---")
            print(f"  id        : {sample.get('id')}")
            print(f"  question  : {q[:140]}{'...' if len(q) > 140 else ''}")
            print(f"  reasoning : {sample.get('judge_reasoning', '')}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
