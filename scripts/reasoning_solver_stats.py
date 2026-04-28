"""Stats: how often did reasoning_independent_solver fire and what was its precision?"""

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v39_claude_reasoning/validation_350/final/examples"
    )
    n = 0
    triggered = 0
    overrode = 0
    overrode_correct = 0
    overrode_wrong = 0
    no_answer_count = 0
    neither_match = 0
    both_match = 0
    reasons = Counter()
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (j.get("pair", {}) or {}).get("source_family") != "livebench-reasoning":
            continue
        n += 1
        pv = (j.get("scoring", {}) or {}).get("pair_verifier", {}) or {}
        rs = pv.get("reasoning_independent_solver") or {}
        reason = rs.get("reason", "")
        if reason == "no_final_answer_marker" or reason == "solver_no_answer":
            no_answer_count += 1
        elif reason == "neither_matches_solver":
            neither_match += 1
        elif reason == "both_match_solver":
            both_match += 1
        reasons[reason] += 1
        if rs.get("triggered"):
            triggered += 1
        if pv.get("decision_source") == "reasoning_independent_solver":
            overrode += 1
            label = (j.get("pair", {}) or {}).get("label", "")
            decision = (((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get(
                "decision", ""
            )
            if decision == label:
                overrode_correct += 1
            else:
                overrode_wrong += 1
    print(f"reasoning_pairs={n}")
    print(f"  solver_triggered={triggered}")
    print(f"  solver_overrode_base={overrode} (correct={overrode_correct}, wrong={overrode_wrong})")
    print(f"  no_solver_answer={no_answer_count}")
    print(f"  neither_candidate_matches_solver={neither_match}")
    print(f"  both_candidates_match_solver={both_match}")
    print(f"  reason distribution:")
    for r, c in reasons.most_common():
        print(f"    {r}: {c}")


if __name__ == "__main__":
    main()
