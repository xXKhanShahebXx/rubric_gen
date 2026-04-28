"""Quick smoke check: did the MMLU answerer fire on mmlu-pro pairs in a run?"""

import json
import sys
from pathlib import Path


def main(run_dir: Path) -> None:
    examples_dir = run_dir / "validation_350" / "final" / "examples"
    if not examples_dir.exists():
        print(f"no examples dir: {examples_dir}")
        return
    mmlu_pairs = 0
    fired = 0
    fired_decisions = []
    for f in sorted(examples_dir.glob("*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (j.get("pair", {}) or {}).get("source_family") != "mmlu-pro":
            continue
        mmlu_pairs += 1
        pv = (j.get("scoring", {}) or {}).get("pair_verifier", {}) or {}
        mm = pv.get("mmlu_independent_answerer") or {}
        if mm.get("triggered"):
            fired += 1
            fired_decisions.append(
                f"  {f.stem[:18]} solver={mm.get('solver_letter')} A={mm.get('a_letter')} "
                f"B={mm.get('b_letter')} -> {pv.get('decision_source')}={pv.get('recommended_decision')}"
            )
    print(f"mmlu_pairs_seen={mmlu_pairs} answerer_fired={fired}")
    for line in fired_decisions[:10]:
        print(line)


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v35_run_a"
    )
    main(run_dir)
