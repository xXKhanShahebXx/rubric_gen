"""Inspect why discovery produced no per-pair rubrics for an RB2 pair."""

import json
import sys
from pathlib import Path


def main(run_dir_str: str) -> None:
    d = Path(run_dir_str)
    files = sorted(d.glob("*.json"))
    j = json.loads(files[0].read_text(encoding="utf-8"))
    p0 = (j.get("discovery", {}) or {}).get("pairs", [{}])[0]
    print(f"raw_proposals_total: {p0.get('raw_proposals_total')}")
    print(f"promoted_proposals_total: {p0.get('promoted_proposals_total')}")
    print(f"rejected_proposals_total: {p0.get('rejected_proposals_total')}")
    print(f"parse_error: {p0.get('parse_error')}")
    print(f"cache: {p0.get('cache')}")
    grounding = p0.get("grounding") or {}
    if isinstance(grounding, dict):
        print(f"grounding keys: {list(grounding.keys())}")
        print(json.dumps(grounding, indent=2)[:600])
    print()
    rejected = p0.get("rejected_proposals", []) or []
    raw = p0.get("raw_proposals", []) or []
    print(f"raw_proposals count: {len(raw)}")
    for r in raw[:5]:
        print(f"  - text: {r.get('text', '')[:120]}")
    print(f"\nrejected_proposals count: {len(rejected)}")
    for r in rejected[:5]:
        print(f"  - text: {r.get('text', '')[:120]}")
        print(f"    reason: {r.get('rejection_reason', 'none')}")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "artifacts/compiled_judgebench_final_eval_runs/rb2_smoke_n2/rb2_smoke_n2/final/examples"
    )
