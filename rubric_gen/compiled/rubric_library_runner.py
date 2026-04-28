"""
CLI runner for building the JudgeBench v2 rubric library.

Use the seed bootstrap (no network, deterministic, used by tests and CI):

.. code-block:: bash

    python -m rubric_gen.compiled.rubric_library_runner seed \
        --out artifacts/rubric_library/v1/library.json

A production build from external preference manifests is invoked as:

.. code-block:: bash

    python -m rubric_gen.compiled.rubric_library_runner build \
        --manifest path/to/manifest.json \
        --out artifacts/rubric_library/v1/library.json

Manifest loaders (HelpSteer3, UltraFeedback, PPE, Arena Hard) are registered in
``rubric_gen.compiled.rubric_library_external_loaders``. That module is wired lazily so that the
seed bootstrap works without the external deps installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rubric_gen.compiled.rubric_library import save_rubric_library
from rubric_gen.compiled.rubric_library_builder import (
    BuilderConfig,
    build_library_from_manifest,
    distill_library,
    merge_library_with_existing,
    read_library,
)
from rubric_gen.compiled.rubric_library_seed import (
    SEED_MANIFEST,
    build_seed_pair_set,
    seed_proposer,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_out_path() -> Path:
    return _repo_root() / "artifacts" / "rubric_library" / "v1" / "library.json"


def _add_build_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-total", type=int, default=60)
    parser.add_argument("--per-family-target", type=int, default=14)
    parser.add_argument("--redundancy-threshold", type=float, default=0.82)
    parser.add_argument("--min-direction-evidence", type=int, default=1)
    parser.add_argument("--max-per-dimension", type=int, default=6)
    parser.add_argument("--no-require-verification", action="store_true")


def _config_from_args(args: argparse.Namespace) -> BuilderConfig:
    return BuilderConfig(
        target_total=max(1, int(args.target_total)),
        per_family_target=max(1, int(args.per_family_target)),
        redundancy_threshold=max(0.0, min(1.0, float(args.redundancy_threshold))),
        min_direction_evidence=max(0, int(args.min_direction_evidence)),
        max_per_dimension=max(1, int(args.max_per_dimension)),
        require_verification=not bool(args.no_require_verification),
    )


def _build_seed_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the seed rubric library from the hand-curated corpus (no network)."
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--merge-into-existing", action="store_true")
    _add_build_config_args(parser)
    return parser


def _build_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the rubric library from an external preference manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--merge-into-existing", action="store_true")
    _add_build_config_args(parser)
    return parser


def _run_seed(argv: List[str]) -> None:
    parser = _build_seed_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    result = distill_library(build_seed_pair_set(), proposer=seed_proposer, config=cfg)
    library = result.library
    out_path = args.out or _default_out_path()
    if args.merge_into_existing and out_path.exists():
        existing = read_library(out_path)
        library = merge_library_with_existing(existing, library)
    save_rubric_library(library, out_path)
    summary = library.summarize()
    summary["accepted_count"] = result.accepted_count
    summary["rejected_misaligned"] = result.rejected_misaligned
    summary["rejected_redundant"] = result.rejected_redundant
    summary["proposals_seen"] = result.proposals_seen
    summary["out_path"] = str(out_path)
    print(json.dumps(summary, indent=2))


def _run_build(argv: List[str]) -> None:
    parser = _build_manifest_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    try:
        from rubric_gen.compiled.rubric_library_external_loaders import (
            REGISTERED_LOADERS,
            build_default_proposer,
            build_default_verifier,
        )

        loader_registry = REGISTERED_LOADERS
        proposer = build_default_proposer(manifest.get("proposer", {}) or {})
        verifier = build_default_verifier(manifest.get("verifier", {}) or {})
    except Exception as exc:
        print(f"Failed to initialise external loaders: {exc}. Falling back to seed proposer.", file=sys.stderr)
        loader_registry = {}
        proposer = seed_proposer
        verifier = None
    result = build_library_from_manifest(
        manifest,
        proposer=proposer,
        verifier=verifier,
        config=cfg,
        loader_registry=loader_registry,
    )
    library = result.library
    out_path = args.out or _default_out_path()
    if args.merge_into_existing and out_path.exists():
        existing = read_library(out_path)
        library = merge_library_with_existing(existing, library)
    save_rubric_library(library, out_path)
    summary = library.summarize()
    summary["accepted_count"] = result.accepted_count
    summary["rejected_misaligned"] = result.rejected_misaligned
    summary["rejected_redundant"] = result.rejected_redundant
    summary["proposals_seen"] = result.proposals_seen
    summary["out_path"] = str(out_path)
    print(json.dumps(summary, indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if not raw or raw[0] in {"-h", "--help"}:
        print("usage: rubric_library_runner {seed|build} [args]")
        print("  seed  : build library from the hand-curated seed corpus")
        print("  build : build library from an external preference manifest")
        return
    command = raw[0]
    if command == "seed":
        _run_seed(raw[1:])
        return
    if command == "build":
        _run_build(raw[1:])
        return
    print(f"Unknown command: {command}. Expected 'seed' or 'build'.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
