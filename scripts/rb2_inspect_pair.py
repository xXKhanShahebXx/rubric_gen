"""Inspect one pair from a smoke run."""

import json
import sys
from pathlib import Path


def main(run_dir_str: str) -> None:
    d = Path(run_dir_str)
    files = sorted(d.glob("*.json"))
    if not files:
        print(f"no artifacts in {d}")
        return
    j = json.loads(files[0].read_text(encoding="utf-8"))
    pair = j.get("pair", {}) or {}
    wu = ((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}
    print(f"pair_id: {pair.get('pair_id')}")
    print(f"label:    {pair.get('label')}")
    print(f"decision: {wu.get('decision')}")
    print(f"score_A:  {wu.get('score_A')}")
    print(f"score_B:  {wu.get('score_B')}")
    print(f"satisfied_A: {wu.get('satisfied_A')}")
    print(f"satisfied_B: {wu.get('satisfied_B')}")
    print(f"tie_break_reason: {wu.get('tie_break_reason')}")
    print(f"decision_policy:  {wu.get('decision_policy')}")
    print()
    rubrics = j.get("rubrics", [])
    print(f"rubric count: {len(rubrics)}")
    for r in rubrics[:5]:
        text = r.get("text", "")
        print(f"  - {text[:140]}")
    print()
    evals = j.get("evaluations", [])
    print(f"evaluation count: {len(evals)}")
    for e in evals[:6]:
        rid = e.get("rubric_id", "")
        a = e.get("response_A_satisfied")
        b = e.get("response_B_satisfied")
        print(f"  rubric={rid[-25:]} A={a} B={b}")
    print()
    pv = (j.get("scoring", {}) or {}).get("pair_verifier") or {}
    print(f"pair_verifier triggered: {pv.get('triggered')}")
    print(f"  recommended_decision: {pv.get('recommended_decision')}")
    print(f"  decision_source: {pv.get('decision_source')}")
    print(f"  reason: {pv.get('reason')}")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "artifacts/compiled_judgebench_final_eval_runs/rb2_smoke_n2/rb2_smoke_n2/final/examples"
    )
