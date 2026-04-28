"""Compare run-a (with MMLU answerer) to v3p5_critique baseline on mmlu-pro flips."""

import json
import sys
from pathlib import Path


def load_decisions(run_dir: Path) -> dict:
    examples_dir = run_dir / "validation_350" / "final" / "examples"
    out = {}
    for f in examples_dir.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair_id = (j.get("pair", {}) or {}).get("pair_id")
        if not pair_id:
            continue
        decision = (
            ((j.get("scoring", {}) or {}).get("whitened_uniform") or {})
            .get("result", {}) or {}
        ).get("decision", "")
        verifier = (j.get("scoring", {}) or {}).get("pair_verifier", {}) or {}
        out[pair_id] = {
            "decision": decision,
            "label": (j.get("pair", {}) or {}).get("label", ""),
            "source_family": (j.get("pair", {}) or {}).get("source_family", ""),
            "verifier_source": verifier.get("decision_source", ""),
            "verifier_decision": verifier.get("recommended_decision", ""),
            "mmlu_triggered": (verifier.get("mmlu_independent_answerer") or {}).get(
                "triggered", False
            ),
            "mmlu_decision": (verifier.get("mmlu_independent_answerer") or {}).get(
                "recommended_decision", ""
            ),
        }
    return out


def main() -> None:
    base = load_decisions(
        Path("artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v3p5_critique_seed29")
    )
    new = load_decisions(
        Path("artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v35_run_a")
    )
    common = set(base) & set(new)
    mmlu_pairs = [pid for pid in common if base[pid]["source_family"] == "mmlu-pro"]
    overrode = 0
    fired_total = 0
    flipped_to_correct = 0
    flipped_to_wrong = 0
    same = 0
    for pid in mmlu_pairs:
        b = base[pid]
        n = new[pid]
        if n["mmlu_triggered"]:
            fired_total += 1
        if n["verifier_source"] == "mmlu_independent_answerer":
            overrode += 1
        if b["decision"] != n["decision"]:
            label = b["label"]
            b_correct = b["decision"] == label
            n_correct = n["decision"] == label
            if not b_correct and n_correct:
                flipped_to_correct += 1
            elif b_correct and not n_correct:
                flipped_to_wrong += 1
        else:
            same += 1
    print(f"mmlu_pairs={len(mmlu_pairs)}")
    print(f"answerer_fired={fired_total}")
    print(f"answerer_overrode={overrode}")
    print(f"flipped_base_wrong->new_correct={flipped_to_correct}")
    print(f"flipped_base_correct->new_wrong={flipped_to_wrong}")
    print(f"same_decision={same}")
    base_correct = sum(1 for pid in mmlu_pairs if base[pid]["decision"] == base[pid]["label"])
    new_correct = sum(1 for pid in mmlu_pairs if new[pid]["decision"] == new[pid]["label"])
    print(f"base_mmlu_correct={base_correct}/{len(mmlu_pairs)} ({100*base_correct/len(mmlu_pairs):.2f}%)")
    print(f"new_mmlu_correct={new_correct}/{len(mmlu_pairs)} ({100*new_correct/len(mmlu_pairs):.2f}%)")


if __name__ == "__main__":
    main()
