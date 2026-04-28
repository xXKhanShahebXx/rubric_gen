"""Inspect livebench-reasoning failures from the latest run."""

import json
import sys
from pathlib import Path
from collections import Counter


def main(run_dir: Path) -> None:
    examples_dir = run_dir / "validation_350" / "final" / "examples"
    failures = []
    for f in examples_dir.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (j.get("pair", {}) or {}).get("source_family") != "livebench-reasoning":
            continue
        label = (j.get("pair", {}) or {}).get("label", "")
        decision = (((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get(
            "decision", ""
        )
        if decision != label:
            pv = (j.get("scoring", {}) or {}).get("pair_verifier", {}) or {}
            rpv = pv.get("reasoning_process_verifier") or {}
            failures.append(
                {
                    "pair_id": (j.get("pair", {}) or {}).get("pair_id", "")[:18],
                    "label": label,
                    "decision": decision,
                    "verifier_source": pv.get("decision_source", ""),
                    "rpv_triggered": rpv.get("triggered", False),
                    "rpv_decision": rpv.get("recommended_decision", ""),
                    "rpv_reason": rpv.get("reason", ""),
                    "question_snippet": (j.get("pair", {}) or {}).get("question", "")[:100],
                }
            )
    print(f"livebench-reasoning failures: {len(failures)}/98")
    sources = Counter(f["verifier_source"] for f in failures)
    print(f"verifier_source distribution: {dict(sources)}")
    rpv_fired = sum(1 for f in failures if f["rpv_triggered"])
    print(f"RPV fired in {rpv_fired}/{len(failures)} failures")
    print("\nSample failures:")
    for f in failures[:5]:
        print(f"  {f['pair_id']} label={f['label']} dec={f['decision']} src={f['verifier_source']} rpv={f['rpv_decision']}/{f['rpv_reason']}")
        print(f"    Q: {f['question_snippet']}")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v38_claude_n3"
    )
    main(run_dir)
