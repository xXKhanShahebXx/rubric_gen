"""Compare v3.5 (GPT-4o answerer) and v3.6 (Claude Opus answerer)."""

import json
from pathlib import Path


def load(run_dir: Path) -> dict:
    examples_dir = run_dir / "validation_350" / "final" / "examples"
    out = {}
    for f in examples_dir.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = (j.get("pair", {}) or {}).get("pair_id")
        if not pid:
            continue
        out[pid] = j
    return out


def main() -> None:
    gpt = load(Path("artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v35_run_a"))
    claude = load(Path("artifacts/compiled_judgebench_final_eval_runs/jb_350_blind_v36_claude_run_a"))
    common = set(gpt) & set(claude)
    mmlu = [pid for pid in common if (gpt[pid].get("pair", {}) or {}).get("source_family") == "mmlu-pro"]

    gpt_correct = 0
    claude_correct = 0
    flips = {"gpt_correct_claude_wrong": 0, "gpt_wrong_claude_correct": 0, "both_correct": 0, "both_wrong": 0}
    claude_overrode = 0
    for pid in mmlu:
        g = gpt[pid]
        c = claude[pid]
        label = (g.get("pair", {}) or {}).get("label", "")
        g_dec = (((g.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get("decision", "")
        c_dec = (((c.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get("decision", "")
        c_pv = (c.get("scoring", {}) or {}).get("pair_verifier", {}) or {}
        if c_pv.get("decision_source") == "mmlu_independent_answerer":
            claude_overrode += 1
        gc = g_dec == label
        cc = c_dec == label
        if gc:
            gpt_correct += 1
        if cc:
            claude_correct += 1
        if gc and not cc:
            flips["gpt_correct_claude_wrong"] += 1
        elif not gc and cc:
            flips["gpt_wrong_claude_correct"] += 1
        elif gc and cc:
            flips["both_correct"] += 1
        else:
            flips["both_wrong"] += 1
    print(f"mmlu_pairs={len(mmlu)}")
    print(f"gpt_answerer_correct={gpt_correct}/{len(mmlu)} ({100*gpt_correct/len(mmlu):.2f}%)")
    print(f"claude_answerer_correct={claude_correct}/{len(mmlu)} ({100*claude_correct/len(mmlu):.2f}%)")
    print(f"claude_overrode_count={claude_overrode}")
    for k, v in flips.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
