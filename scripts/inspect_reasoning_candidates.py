"""Inspect the tail of reasoning candidate responses to see answer formats."""

import json
from pathlib import Path


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v38_claude_n3/validation_350/final/examples"
    )
    seen = 0
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair = j.get("pair", {}) or {}
        if pair.get("source_family") != "livebench-reasoning":
            continue
        label = pair.get("label")
        decision = (
            ((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}
        ).get("decision", "")
        if decision == label:
            continue
        print(f"PAIR {pair.get('pair_id', '')[:18]} label={label} dec={decision}")
        q = pair.get("question", "")
        print(f"  Q ({len(q)} chars): {q[:140]}")
        for letter in ("A", "B"):
            text = pair.get(f"response_{letter}", "") or ""
            tail = text[-280:]
            print(f"  --- {letter} tail ---")
            print(f"  {tail}")
        print()
        seen += 1
        if seen >= 4:
            break


if __name__ == "__main__":
    main()
