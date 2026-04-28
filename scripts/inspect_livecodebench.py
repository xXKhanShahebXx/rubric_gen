"""Inspect livecodebench question / candidate format."""

import json
from pathlib import Path


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v42_dual_mmlu/validation_350/final/examples"
    )
    seen = 0
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair = j.get("pair", {}) or {}
        if pair.get("source_family") != "livecodebench":
            continue
        decision = (((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get(
            "decision", ""
        )
        label = pair.get("label", "")
        if decision == label:
            continue
        print(f"PAIR {pair.get('pair_id', '')[:18]} label={label} dec={decision}")
        q = pair.get("question", "")
        print(f"  Q first 600: {q[:600]}")
        print(f"  Q last 300: {q[-300:]}")
        for letter in ("A", "B"):
            text = pair.get(f"response_{letter}", "") or ""
            print(f"  --- {letter} (len={len(text)}) head ---")
            print(f"  {text[:300]}")
        print()
        seen += 1
        if seen >= 2:
            break


if __name__ == "__main__":
    main()
