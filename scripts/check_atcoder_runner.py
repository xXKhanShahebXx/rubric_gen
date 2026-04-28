"""Run the existing code_execution_verifier (AtCoder stdin) on the 6 no_method pairs."""

import json
from pathlib import Path

from rubric_gen.compiled.code_execution_verifier import (
    evaluate_code_pair_verifier,
    extract_visible_io_pairs,
    extract_candidate_code,
)


TARGET_IDS = {
    "0ca7d4e7-aa30-589d",
    "100c98a6-0077-5ccd",
    "23913c97-eb0c-5740",
    "310930bc-03d8-5dc0",
    "5b5bd25b-3f2d-5ffc",
    "82e65bbd-1ecf-51e4",
}


def main() -> None:
    d = Path(
        "artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v45_no_reasoning_hp/validation_350/final/examples"
    )
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = (j.get("pair", {}) or {}).get("pair_id", "")
        short = pid[:18]
        if short not in TARGET_IDS:
            continue
        pair = j.get("pair", {}) or {}
        q = pair.get("question") or ""
        io = extract_visible_io_pairs(q)
        ra = pair.get("response_A") or ""
        rb = pair.get("response_B") or ""
        ca = extract_candidate_code(ra)
        cb = extract_candidate_code(rb)
        outcome = evaluate_code_pair_verifier(
            question=q,
            response_a=ra,
            response_b=rb,
            timeout_s=10.0,
        )
        label = pair.get("label", "")
        print(f"{short} label={label}")
        print(f"  io_pairs={len(io)} ca_len={len(ca)} cb_len={len(cb)}")
        print(f"  triggered={outcome.triggered} reason={outcome.reason!r}")
        if outcome.a_report:
            print(f"  a_pass_rate={outcome.a_report.pass_rate:.2f} ({outcome.a_report.pass_count}/{outcome.a_report.total_cases})")
        if outcome.b_report:
            print(f"  b_pass_rate={outcome.b_report.pass_rate:.2f} ({outcome.b_report.pass_count}/{outcome.b_report.total_cases})")
        print()


if __name__ == "__main__":
    main()
