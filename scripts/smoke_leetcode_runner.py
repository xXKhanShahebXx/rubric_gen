"""Smoke test: run leetcode runner on all 42 livecodebench pairs in v4.2 artifacts."""

import json
from pathlib import Path

from rubric_gen.compiled.leetcode_test_runner import evaluate_leetcode_pair_verifier


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v42_dual_mmlu/validation_350/final/examples"
    )
    fired = 0
    correct_overrides = 0
    wrong_overrides = 0
    no_method = 0
    no_examples = 0
    parse_errors = 0
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair = j.get("pair", {}) or {}
        if pair.get("source_family") != "livecodebench":
            continue
        outcome = evaluate_leetcode_pair_verifier(
            question=pair.get("question") or "",
            response_a=pair.get("response_A") or "",
            response_b=pair.get("response_B") or "",
            timeout_s=10.0,
        )
        reason = outcome.reason or ""
        if outcome.triggered:
            fired += 1
            label = pair.get("label", "")
            if outcome.recommended_decision == label:
                correct_overrides += 1
            else:
                wrong_overrides += 1
        elif reason == "no_method_signature":
            no_method += 1
        elif reason == "no_leetcode_examples":
            no_examples += 1
        elif "no_candidate_code" in reason:
            parse_errors += 1
    print(f"livecodebench=42")
    print(f"  triggered={fired} (correct={correct_overrides}, wrong={wrong_overrides})")
    print(f"  no_method={no_method}")
    print(f"  no_examples={no_examples}")
    print(f"  parse_errors_or_no_code={parse_errors}")


if __name__ == "__main__":
    main()
