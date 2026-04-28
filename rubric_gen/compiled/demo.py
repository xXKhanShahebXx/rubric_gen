"""
Demo entrypoint: load the sample dataset, compile starter artifacts for the first row, and write JSON.

Run: python -m rubric_gen.compiled.demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rubric_gen.dataio import load_examples
from rubric_gen.compiled.compiler import (
    build_note_family_spec,
    build_starter_ontology,
    compile_case_rubric,
    infer_note_family,
)
from rubric_gen.compiled.heuristic_judge import evaluate_note_against_rubric
from rubric_gen.compiled.serialize import write_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compiled rubric starter demo (scaffold).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to sample dataset JSON (default: data/sample_100_aci_400_agbonnet.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for example artifacts (default: docs/spec/examples/)",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    dataset = args.dataset or (root / "data" / "sample_100_aci_400_agbonnet.json")
    out_dir = args.out_dir or (root / "docs" / "spec" / "examples")

    examples = load_examples(dataset, start=0, limit=1)
    if not examples:
        raise SystemExit("No examples loaded; check dataset path and contents.")
    ex = examples[0]

    ontology = build_starter_ontology()
    note_family_id = infer_note_family(ex)
    note_family = build_note_family_spec(note_family_id, ontology)
    case_rubric = compile_case_rubric(ex, ontology, note_family)

    write_json(out_dir / "starter_rubric_ontology.json", ontology)
    write_json(out_dir / f"note_family_spec__{note_family_id}.json", note_family)
    safe_id = ex.example_id.replace("/", "_")
    write_json(out_dir / f"case_rubric__{safe_id}.json", case_rubric)

    ref_eval = evaluate_note_against_rubric(
        candidate_id="reference_note",
        note_text=ex.reference_note,
        dialogue=ex.conversation,
        case_rubric=case_rubric,
    )
    write_json(out_dir / f"case_evaluation__{safe_id}__reference_note.json", ref_eval)

    if ex.augmented_note.strip():
        aug_eval = evaluate_note_against_rubric(
            candidate_id="augmented_note",
            note_text=ex.augmented_note,
            dialogue=ex.conversation,
            case_rubric=case_rubric,
        )
        write_json(out_dir / f"case_evaluation__{safe_id}__augmented_note.json", aug_eval)


if __name__ == "__main__":
    main()
