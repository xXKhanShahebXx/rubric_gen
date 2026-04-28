"""Look at mmlu-pro pairs where MMLU answerer's letter matches neither candidate."""

import json
from pathlib import Path


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v41_claude_opus41_n3/validation_350/final/examples"
    )
    seen = 0
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair = j.get("pair", {}) or {}
        if pair.get("source_family") != "mmlu-pro":
            continue
        pv = (j.get("scoring", {}) or {}).get("pair_verifier", {}) or {}
        mm = pv.get("mmlu_independent_answerer") or {}
        if not mm:
            continue
        if mm.get("reason") != "neither_matches_solver":
            continue
        decision = (((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get(
            "decision", ""
        )
        label = pair.get("label", "")
        if decision == label:
            continue
        # Wrong pair where solver had answer but candidate letters didn't match
        print(f"PAIR {pair.get('pair_id', '')[:18]} label={label} dec={decision}")
        print(f"  solver_letter={mm.get('solver_letter')}")
        print(f"  a_letter={mm.get('a_letter')} b_letter={mm.get('b_letter')}")
        print(f"  Q tail: ...{(pair.get('question') or '')[-300:]}")
        for letter in ("A", "B"):
            tail = (pair.get(f"response_{letter}") or "")[-220:]
            print(f"  {letter}: {tail}")
        print()
        seen += 1
        if seen >= 5:
            break


if __name__ == "__main__":
    main()
