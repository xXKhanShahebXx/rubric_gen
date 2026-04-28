"""Find code pairs where the LeetCode runner couldn't extract a method signature."""

import json
from pathlib import Path

from rubric_gen.compiled.code_execution_verifier import extract_candidate_code
from rubric_gen.compiled.leetcode_test_runner import (
    _parse_method_signature,
    parse_leetcode_examples,
    evaluate_leetcode_pair_verifier,
)


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v45_no_reasoning_hp/validation_350/final/examples"
    )
    no_method = []
    no_examples = []
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair = j.get("pair", {}) or {}
        if pair.get("source_family") != "livecodebench":
            continue
        q = pair.get("question") or ""
        ra = pair.get("response_A") or ""
        rb = pair.get("response_B") or ""
        ca = extract_candidate_code(ra)
        cb = extract_candidate_code(rb)
        method, params = _parse_method_signature(q, candidate_code=ca or cb)
        if not method:
            no_method.append((pair.get("pair_id", "")[:18], q[:200]))
            continue
        examples = parse_leetcode_examples(q, expected_params=params)
        if not examples:
            no_examples.append((pair.get("pair_id", "")[:18], q[:200]))
    print(f"no_method count: {len(no_method)}")
    for pid, q in no_method:
        print(f"  {pid}")
        print(f"    Q: {q}")
    print(f"\nno_examples count: {len(no_examples)}")
    for pid, q in no_examples:
        print(f"  {pid}")
        print(f"    Q: {q}")


if __name__ == "__main__":
    main()
