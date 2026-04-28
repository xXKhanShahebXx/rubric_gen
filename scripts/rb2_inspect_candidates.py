"""Inspect candidates and discovery pairs for one RB2 artifact."""

import json
import sys
from pathlib import Path


def main(run_dir_str: str) -> None:
    d = Path(run_dir_str)
    j = json.loads(sorted(d.glob("*.json"))[0].read_text(encoding="utf-8"))
    cands = j.get("candidates", [])
    print(f"candidates: {len(cands)}")
    for c in cands:
        print(
            f"  id={c.get('candidate_id', '')[:60]}  "
            f"label={c.get('source_label')}  "
            f"bucket={c.get('quality_bucket')}  "
            f"origin={c.get('origin_kind')}  "
            f"text_len={len(c.get('text', ''))}"
        )
    print()
    pairs = (j.get("discovery", {}) or {}).get("pairs", [])
    print(f"discovery pairs: {len(pairs)}")
    for p in pairs:
        print(f"  pair_id: {p.get('pair_id')}")
        g = p.get("grounding", {}) or {}
        print(f"    mode: {g.get('mode')}  delta_mode: {g.get('delta_mode')}")
        print(f"    weak_mutation_id: {g.get('weak_mutation_id')}")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "artifacts/compiled_judgebench_final_eval_runs/rb2_smoke_n2/rb2_smoke_n2/final/examples"
    )
