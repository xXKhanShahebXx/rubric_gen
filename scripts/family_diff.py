"""Per-family accuracy comparison across runs."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def family_acc(run_dir: Path) -> dict:
    examples_dir = run_dir / "validation_350" / "final" / "examples"
    counts = defaultdict(lambda: [0, 0])
    for f in examples_dir.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        family = (j.get("pair", {}) or {}).get("source_family", "")
        label = (j.get("pair", {}) or {}).get("label", "")
        decision = (((j.get("scoring", {}) or {}).get("whitened_uniform") or {}).get("result", {}) or {}).get(
            "decision", ""
        )
        counts[family][1] += 1
        if decision == label:
            counts[family][0] += 1
    return dict(counts)


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        print("usage: family_diff.py run_a run_b ...")
        return
    rows = {r.name: family_acc(r) for r in runs}
    families = sorted({fam for r in rows.values() for fam in r})
    header = f"{'family':<24}" + "".join(f"{name[:30]:>32}" for name in rows)
    print(header)
    for fam in families:
        line = f"{fam:<24}"
        for name in rows:
            c, t = rows[name].get(fam, [0, 0])
            line += f"{c}/{t} ({100*c/max(1,t):>5.2f}%)".rjust(32)
        print(line)
    print()
    for name, fams in rows.items():
        total_c = sum(c for c, _ in fams.values())
        total_t = sum(t for _, t in fams.values())
        print(f"{name}: {total_c}/{total_t} = {100*total_c/max(1,total_t):.2f}%")


if __name__ == "__main__":
    main()
