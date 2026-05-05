"""Project a regen+judge JSONL down to a minimal pair-preference schema.

Default output schema is exactly the 5 fields RL-style training expects:

    {id, question, reference_answer_a, reference_answer_b, correct_answer}

`correct_answer` is taken straight from the source row, which means it
already reflects whatever ``--tie-policy`` was used when the source
artifact was produced (e.g. ``tie-policy=a`` on the GPT-5 4k run keeps
ties routed to ``reference_answer_a``).

Source files preserved on disk; this script only writes a new file.

Usage:

    python scripts/slim_pair_dataset.py \\
        --input data/medical_gpt5_b_regen_4k_opus_judged.jsonl \\
        --output data/medical_gpt5_b_regen_4k_rl.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_FIELDS: Sequence[str] = (
    "id",
    "question",
    "reference_answer_a",
    "reference_answer_b",
    "correct_answer",
)
ALLOWED_LABELS = ("reference_answer_a", "reference_answer_b")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL (will be overwritten if it exists).",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help=(
            "Whitelist of fields to keep, in output order. "
            f"Default: {' '.join(DEFAULT_FIELDS)}."
        ),
    )
    parser.add_argument(
        "--require-label",
        action="store_true",
        default=True,
        help=(
            "Require correct_answer to be one of "
            f"{ALLOWED_LABELS}. Rows missing or with other values are dropped "
            "(and counted)."
        ),
    )
    parser.add_argument(
        "--allow-other-labels",
        dest="require_label",
        action="store_false",
        help="Disable the correct_answer label whitelist (keep all rows).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail with a non-zero exit code if any row is missing a required "
            "field. Default: drop the row and continue."
        ),
    )
    return parser


def _read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected object, got {type(obj).__name__}")
            yield obj


def _project_row(
    row: Dict[str, Any],
    fields: Sequence[str],
    *,
    require_label: bool,
) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for field in fields:
        if field not in row:
            return None
        out[field] = row[field]
    if require_label and "correct_answer" in fields:
        if out["correct_answer"] not in ALLOWED_LABELS:
            return None
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    fields = list(args.fields)
    rows_in = 0
    rows_out = 0
    dropped_missing_field: Dict[str, int] = {}
    dropped_bad_label = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_fh:
        for row in _read_rows(args.input):
            rows_in += 1
            projected = _project_row(row, fields, require_label=args.require_label)
            if projected is None:
                missing = [f for f in fields if f not in row]
                if missing:
                    if args.strict:
                        print(
                            f"ERROR: row {rows_in} missing fields: {missing}",
                            file=sys.stderr,
                        )
                        return 2
                    for f in missing:
                        dropped_missing_field[f] = dropped_missing_field.get(f, 0) + 1
                else:
                    dropped_bad_label += 1
                continue
            out_fh.write(json.dumps(projected, ensure_ascii=False) + "\n")
            rows_out += 1

    print(f"Read   : {rows_in:>5}  rows from {args.input}")
    print(f"Wrote  : {rows_out:>5}  rows to   {args.output}")
    if dropped_missing_field:
        print(f"Dropped (missing field): {dropped_missing_field}")
    if dropped_bad_label:
        print(
            f"Dropped (correct_answer not in {ALLOWED_LABELS}): {dropped_bad_label}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
