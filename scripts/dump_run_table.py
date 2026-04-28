"""Dump per-run validation_350 scores + family breakdowns for the master report."""

import json
from collections import defaultdict
from pathlib import Path


def family_acc(run_dir: Path) -> dict:
    examples_dir = run_dir / "validation_350" / "final" / "examples"
    counts = defaultdict(lambda: [0, 0])
    if not examples_dir.exists():
        return {}
    for f in examples_dir.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        family = (j.get("pair", {}) or {}).get("source_family", "")
        label = (j.get("pair", {}) or {}).get("label", "")
        decision = (
            ((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}
        ).get("decision", "")
        counts[family][1] += 1
        if decision == label:
            counts[family][0] += 1
    return dict(counts)


def main() -> None:
    runs_dir = Path("artifacts/compiled_judgebench_final_eval_runs")
    target = [
        "jb_350_blind_v2_full_seed29",
        "jb_350_blind_v2_no_holistic_seed29",
        "jb_350_blind_v21_full_seed29",
        "jb_350_blind_v3p1_code_seed29",
        "jb_350_blind_v3p2_sympy_seed29",
        "jb_350_blind_v3p3_fewshot_seed29",
        "jb_350_blind_v3p4_rubric_sc_seed29",
        "jb_350_blind_v3p5_critique_seed29",
        "jb_350_blind_v35_run_a",
        "jb_350_blind_v35_run_b",
        "jb_350_blind_v35_run_c",
        "jb_350_blind_v36_claude_run_a",
        "jb_350_blind_v36_claude_shared",
        "jb_350_blind_v37_claude_both",
        "jb_350_blind_v38_claude_n3",
        "jb_350_blind_v39_claude_reasoning",
        "jb_350_blind_v40_gpt5_solvers",
        "jb_350_blind_v41_claude_opus41_n3",
        "jb_350_blind_v42_dual_mmlu",
        "jb_350_blind_v43_leetcode",
        "jb_350_blind_v44_high_precision_override",
        "jb_350_blind_v45_no_reasoning_hp",
        "jb_350_blind_v46_n5_solvers",
        "jb_350_blind_v47_agreement_aware",
    ]
    for name in target:
        run_dir = runs_dir / name
        if not run_dir.exists():
            print(f"{name}: MISSING")
            continue
        fams = family_acc(run_dir)
        total_c = sum(c for c, _ in fams.values())
        total_t = sum(t for _, t in fams.values())
        if total_t == 0:
            print(f"{name}: no artifacts")
            continue
        overall_pct = 100.0 * total_c / total_t
        parts = []
        for fam in ("mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench"):
            if fam in fams:
                c, t = fams[fam]
                parts.append(f"{fam}={c}/{t}({100*c/max(1,t):.2f})")
        print(f"{name}: overall={total_c}/{total_t}({overall_pct:.2f}%)  " + "  ".join(parts))


if __name__ == "__main__":
    main()
