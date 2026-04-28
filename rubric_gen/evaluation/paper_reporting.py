from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from rubric_gen.evaluation.pairwise import build_proxy_pairwise_preferences, evaluate_pairwise_preferences
from rubric_gen.storage import write_json


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_paper_pairwise_rows(
    example_artifacts: Iterable[Dict[str, object]],
    label_mode: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for artifact in example_artifacts:
        example = artifact["example"]
        pairs = artifact.get("pairwise_labels")
        if pairs is None:
            pairs = build_proxy_pairwise_preferences(artifact["candidates"], label_mode=label_mode)
        for method_name, method_payload in artifact["methods"].items():
            ranking = method_payload.get("ranking", [])
            metrics = evaluate_pairwise_preferences(ranking, pairs)
            rows.append(
                {
                    "example_id": example["example_id"],
                    "source": example["source"],
                    "method": method_name,
                    "pair_count": int(metrics["pair_count"]),
                    "pairwise_accuracy": metrics["pairwise_accuracy"],
                    "mean_margin": metrics["mean_margin"],
                    "rubric_count": len(method_payload.get("rubrics", [])),
                    "label_mode": label_mode,
                }
            )
    return rows


def aggregate_paper_pairwise_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_method: Dict[str, Dict[str, object]] = {}
    for row in rows:
        entry = by_method.setdefault(
            str(row["method"]),
            {
                "method": row["method"],
                "examples": 0,
                "pair_count": 0,
                "pairwise_accuracy": [],
                "mean_margin": [],
                "rubric_count": [],
            },
        )
        entry["examples"] += 1
        entry["pair_count"] += int(row["pair_count"])
        entry["pairwise_accuracy"].append(float(row["pairwise_accuracy"]))
        entry["mean_margin"].append(float(row["mean_margin"]))
        entry["rubric_count"].append(float(row["rubric_count"]))

    aggregated: List[Dict[str, object]] = []
    for method_name, entry in sorted(by_method.items()):
        aggregated.append(
            {
                "method": method_name,
                "examples": int(entry["examples"]),
                "pair_count": int(entry["pair_count"]),
                "avg_pairwise_accuracy": mean(entry["pairwise_accuracy"]) if entry["pairwise_accuracy"] else 0.0,
                "avg_mean_margin": mean(entry["mean_margin"]) if entry["mean_margin"] else 0.0,
                "avg_rubric_count": mean(entry["rubric_count"]) if entry["rubric_count"] else 0.0,
            }
        )
    return aggregated


def _summary_markdown(rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Paper Mode Summary",
        "",
        "| Method | Examples | Pair Count | Avg Pairwise Accuracy | Avg Margin | Avg Rubrics |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {examples} | {pair_count} | {avg_pairwise_accuracy:.3f} | "
            "{avg_mean_margin:.3f} | {avg_rubric_count:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("Higher `avg_pairwise_accuracy` is better. These are proxy pairwise results, not human labels.")
    return "\n".join(lines)


def write_paper_reports(
    run_dir: Path,
    example_artifacts: List[Dict[str, object]],
    label_mode: str,
) -> Dict[str, Path]:
    pairwise_rows = build_paper_pairwise_rows(example_artifacts, label_mode=label_mode)
    aggregated = aggregate_paper_pairwise_rows(pairwise_rows)

    pairwise_results_path = run_dir / "reports" / "pairwise_results.csv"
    paper_summary_md_path = run_dir / "summaries" / "paper_summary.md"
    paper_summary_json_path = run_dir / "summaries" / "paper_summary.json"

    _write_csv(pairwise_results_path, pairwise_rows)
    paper_summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    paper_summary_md_path.write_text(_summary_markdown(aggregated), encoding="utf-8")
    write_json(
        paper_summary_json_path,
        {
            "aggregated": aggregated,
            "pairwise_rows": pairwise_rows,
            "label_mode": label_mode,
        },
    )

    return {
        "pairwise_results_csv": pairwise_results_path,
        "paper_summary_md": paper_summary_md_path,
        "paper_summary_json": paper_summary_json_path,
    }
