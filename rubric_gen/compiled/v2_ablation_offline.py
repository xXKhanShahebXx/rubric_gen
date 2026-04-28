"""
Offline v2 ablation analysis.

Reads existing per-example artifacts from a final-eval run and computes JudgeBench
*double-order accuracy* under each of:

- ``v2_full``: the as-saved decisions (baseline, identical to summary.json)
- ``no_holistic``: revert any decision made by the holistic judge
- ``no_discriminator``: revert any decision made by the wide pair discriminator
- ``no_pair_verifier``: revert any decision made by the pair verifier (covers RPV cases too)

Each ablation walks the artifact's override chain and replaces the final decision with
``base_decision`` when the ablated component made it. The ablation does NOT re-run any LLM
calls — it only restages decisions already on disk. Library ablation requires a re-run and is
not handled here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


_FAMILIES = ("mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench")


def _flip_decision(decision: str) -> str:
    if decision == "A>B":
        return "B>A"
    if decision == "B>A":
        return "A>B"
    return decision


def _load_artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    examples_dir = (
        run_dir / "validation_350" / "final" / "examples"
    )
    if not examples_dir.exists():
        for candidate in run_dir.rglob("examples"):
            if candidate.is_dir():
                examples_dir = candidate
                break
    artifacts: List[Dict[str, Any]] = []
    for path in sorted(examples_dir.glob("*.json")):
        try:
            artifacts.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return artifacts


_ABLATION_POLICY_SETS: Dict[str, set] = {
    "no_holistic": {"holistic_judge"},
    "no_discriminator": {"blind_pair_discriminator", "pairwise_discriminator"},
    "no_pair_verifier": {"pair_verifier"},
}


def _ablate_decision(artifact: Mapping[str, Any], ablation: str) -> Tuple[str, str]:
    """
    Return ``(decision, decision_reversed)`` under the requested ablation.

    The artifact stores the *final* decision after every override, plus the immediate
    ``base_decision`` predecessor. This function reverts the override only when the
    ``decision_policy`` of the final decision matches one of the components being ablated.

    ``ablation`` may be one of the single-component names (``no_holistic``,
    ``no_discriminator``, ``no_pair_verifier``), the special ``v2_full`` (no ablation), or a
    combined ablation joined by ``+`` (e.g. ``no_holistic+no_pair_verifier``).
    """
    result = (artifact.get("scoring", {}) or {}).get("whitened_uniform", {}).get("result", {}) or {}
    final_decision = str(result.get("decision", "")).strip()
    final_reversed = str(result.get("decision_reversed", "")).strip()
    decision_policy = str(result.get("decision_policy", "")).strip()
    base_decision = str(result.get("base_decision", "")).strip()

    if ablation == "v2_full":
        return final_decision, final_reversed

    ablate_policies: set = set()
    for component in ablation.split("+"):
        ablate_policies.update(_ABLATION_POLICY_SETS.get(component.strip(), set()))

    if decision_policy in ablate_policies and base_decision in {"A>B", "B>A", "A=B"}:
        return base_decision, _flip_decision(base_decision)

    return final_decision, final_reversed


def _double_order_correct(
    label: str,
    decision_original: str,
    decision_reversed: str,
) -> bool:
    flip = _flip_decision(label)
    counter = 0
    for d in (decision_original, _flip_decision(decision_reversed)):
        if d == label:
            counter += 1
        elif d == flip:
            counter -= 1
    return counter > 0


def _accuracy(rows: Iterable[Tuple[str, str, str, str]]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    correct = sum(
        1
        for source_family, label, d1, d2 in rows
        if _double_order_correct(label, d1, d2)
    )
    return 100.0 * correct / len(rows)


def _per_family_accuracy(rows: Iterable[Tuple[str, str, str, str]]) -> Dict[str, float]:
    rows = list(rows)
    out: Dict[str, float] = {"overall": _accuracy(rows)}
    for fam in _FAMILIES:
        subset = [(sf, lbl, d1, d2) for sf, lbl, d1, d2 in rows if sf == fam]
        out[fam] = _accuracy(subset)
    return out


def run_ablations(run_dir: Path) -> Dict[str, Any]:
    artifacts = _load_artifacts(run_dir)
    ablations = [
        "v2_full",
        "no_holistic",
        "no_discriminator",
        "no_pair_verifier",
        "no_holistic+no_discriminator",
        "no_holistic+no_pair_verifier",
        "no_holistic+no_discriminator+no_pair_verifier",
    ]
    counts_by_ablation: Dict[str, List[Tuple[str, str, str, str]]] = {a: [] for a in ablations}
    decision_policy_distribution: Dict[str, int] = {}
    flip_counts: Dict[str, int] = {a: 0 for a in ablations}
    for artifact in artifacts:
        source_family = str(((artifact.get("pair", {}) or {}).get("source_family") or "")).strip()
        label = str(((artifact.get("pair", {}) or {}).get("label") or "")).strip()
        if source_family not in _FAMILIES or label not in {"A>B", "B>A", "A=B"}:
            continue
        wu_result = (artifact.get("scoring", {}) or {}).get("whitened_uniform", {}).get("result", {}) or {}
        decision_policy = str(wu_result.get("decision_policy", "")).strip()
        decision_policy_distribution[decision_policy] = (
            decision_policy_distribution.get(decision_policy, 0) + 1
        )
        for ablation in ablations:
            d1, d2 = _ablate_decision(artifact, ablation)
            if d1 != str(wu_result.get("decision", "")).strip():
                flip_counts[ablation] += 1
            counts_by_ablation[ablation].append((source_family, label, d1, d2))

    summary: Dict[str, Any] = {
        "schema": "compiled_judgebench_v2_ablation_offline_v1",
        "run_dir": str(run_dir),
        "artifact_count": len(artifacts),
        "decision_policy_distribution": decision_policy_distribution,
        "results": {},
        "flip_counts": flip_counts,
    }
    for ablation, rows in counts_by_ablation.items():
        summary["results"][ablation] = _per_family_accuracy(rows)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline ablation analysis of a JudgeBench v2 final-eval run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, default=None)
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw)
    summary = run_ablations(args.run_dir)
    out_path = args.out_path or (args.run_dir / "summaries" / "ablation_offline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
