"""Categorize the 32 wrong mmlu pairs in v4.5 by failure mode."""

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v45_no_reasoning_hp/validation_350/final/examples"
    )
    failure_modes = Counter()
    breakdown = []
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair = j.get("pair", {}) or {}
        if pair.get("source_family") != "mmlu-pro":
            continue
        label = pair.get("label", "")
        decision = (((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get(
            "decision", ""
        )
        if decision == label:
            continue
        pv = (j.get("scoring", {}) or {}).get("pair_verifier", {}) or {}
        mm = pv.get("mmlu_independent_answerer") or {}
        solver_letter = mm.get("solver_letter", "")
        a_letter = mm.get("a_letter", "")
        b_letter = mm.get("b_letter", "")
        reason = mm.get("reason", "")
        # decode failure mode
        if not solver_letter:
            mode = "solver_no_letter"
        elif reason == "neither_matches_solver":
            mode = "neither_matches_solver"
        elif reason == "both_match_solver":
            mode = "both_match"
        elif reason.startswith("primary_secondary_disagree"):
            mode = "dual_consensus_disagree"
        elif pv.get("decision_source") == "mmlu_independent_answerer":
            # solver fired and overrode but ended up wrong
            mode = "solver_overrode_to_wrong"
        else:
            mode = f"other_{reason}"
        failure_modes[mode] += 1
        breakdown.append((pair.get("pair_id", "")[:18], label, decision, solver_letter, a_letter, b_letter, mode))
    print(f"wrong mmlu pairs: {sum(failure_modes.values())}")
    for k, v in failure_modes.most_common():
        print(f"  {k}: {v}")
    print("\nfirst 8 examples:")
    for row in breakdown[:8]:
        print(f"  {row[0]} label={row[1]} dec={row[2]} solver={row[3]} A={row[4]} B={row[5]} mode={row[6]}")


if __name__ == "__main__":
    main()
