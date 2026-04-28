from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rubric_gen.storage import write_json


STRONG_BUCKETS = {"gold_like", "strong_anchor", "frontier_generated", "open_generated"}
WEAK_BUCKETS = {"synthetically_degraded"}


def _ranking_lookup(ranking: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {row["candidate_id"]: row for row in ranking}


def _reference_top1(example_artifact: Dict[str, object], ranking: List[Dict[str, object]]) -> Tuple[bool, bool]:
    candidates = example_artifact["candidates"]
    reference_candidates = [candidate for candidate in candidates if candidate["source_label"] == "reference_note"]
    if not reference_candidates:
        return False, False
    ranking_lookup = _ranking_lookup(ranking)
    for candidate in reference_candidates:
        ranked = ranking_lookup.get(candidate["candidate_id"])
        if ranked and int(ranked["rank"]) == 1:
            return True, True
    return True, False


def _strong_vs_weak_accuracy(example_artifact: Dict[str, object], ranking: List[Dict[str, object]]) -> Tuple[int, int, float]:
    ranking_lookup = _ranking_lookup(ranking)
    strong = [
        candidate for candidate in example_artifact["candidates"] if candidate["quality_bucket"] in STRONG_BUCKETS
    ]
    weak = [
        candidate for candidate in example_artifact["candidates"] if candidate["quality_bucket"] in WEAK_BUCKETS
    ]

    total = 0
    correct = 0
    margins: List[float] = []

    for strong_candidate in strong:
        for weak_candidate in weak:
            strong_rank = ranking_lookup.get(strong_candidate["candidate_id"], {}).get("rank")
            weak_rank = ranking_lookup.get(weak_candidate["candidate_id"], {}).get("rank")
            strong_score = ranking_lookup.get(strong_candidate["candidate_id"], {}).get("score", 0.0)
            weak_score = ranking_lookup.get(weak_candidate["candidate_id"], {}).get("score", 0.0)
            if strong_rank is None or weak_rank is None:
                continue
            total += 1
            if int(strong_rank) < int(weak_rank):
                correct += 1
            margins.append(float(strong_score) - float(weak_score))
    return total, correct, (sum(margins) / len(margins) if margins else 0.0)


def aggregate_method_metrics(example_artifacts: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    by_method: Dict[str, Dict[str, object]] = {}

    for artifact in example_artifacts:
        for method_name, method_payload in artifact["methods"].items():
            metrics = by_method.setdefault(
                method_name,
                {
                    "method": method_name,
                    "examples": 0,
                    "reference_examples": 0,
                    "reference_top1_hits": 0,
                    "pairwise_total": 0,
                    "pairwise_correct": 0,
                    "margin_sum": 0.0,
                    "rubric_count_sum": 0,
                    "compressed_rubric_count_sum": 0,
                    "production_rubric_count_sum": 0,
                    "rejected_sum": 0,
                    "superseded_sum": 0,
                },
            )

            ranking = method_payload.get("ranking", [])
            metrics["examples"] += 1

            has_reference, reference_top1 = _reference_top1(artifact, ranking)
            if has_reference:
                metrics["reference_examples"] += 1
                metrics["reference_top1_hits"] += int(reference_top1)

            pairwise_total, pairwise_correct, margin = _strong_vs_weak_accuracy(artifact, ranking)
            metrics["pairwise_total"] += pairwise_total
            metrics["pairwise_correct"] += pairwise_correct
            metrics["margin_sum"] += margin

            if "rubrics" in method_payload:
                metrics["rubric_count_sum"] += len(method_payload["rubrics"])
            if "compressed_bank" in method_payload:
                metrics["compressed_rubric_count_sum"] += len(method_payload["compressed_bank"])
            if "production_bank" in method_payload:
                metrics["production_rubric_count_sum"] += len(method_payload["production_bank"])
            if "artifact" in method_payload:
                artifact_stats = method_payload["artifact"]
                metrics["rejected_sum"] += len(artifact_stats.get("rejected", []))
                metrics["superseded_sum"] += len(artifact_stats.get("superseded", []))

    rows: List[Dict[str, object]] = []
    for method_name, metrics in sorted(by_method.items()):
        examples = max(1, int(metrics["examples"]))
        reference_examples = int(metrics["reference_examples"])
        pairwise_total = int(metrics["pairwise_total"])
        rows.append(
            {
                "method": method_name,
                "examples": int(metrics["examples"]),
                "reference_top1_rate": (
                    float(metrics["reference_top1_hits"]) / reference_examples if reference_examples else 0.0
                ),
                "strong_vs_weak_accuracy": (
                    float(metrics["pairwise_correct"]) / pairwise_total if pairwise_total else 0.0
                ),
                "mean_strong_weak_margin": float(metrics["margin_sum"]) / examples,
                "avg_rubric_count": float(metrics["rubric_count_sum"]) / examples,
                "avg_compressed_rubric_count": float(metrics["compressed_rubric_count_sum"]) / examples,
                "avg_production_rubric_count": float(metrics["production_rubric_count_sum"]) / examples,
                "avg_rejected_rubrics": float(metrics["rejected_sum"]) / examples,
                "avg_superseded_rubrics": float(metrics["superseded_sum"]) / examples,
            }
        )
    return rows


def flatten_rankings(example_artifacts: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for artifact in example_artifacts:
        example = artifact["example"]
        candidates = {candidate["candidate_id"]: candidate for candidate in artifact["candidates"]}
        for method_name, method_payload in artifact["methods"].items():
            for ranking_row in method_payload.get("ranking", []):
                candidate = candidates.get(ranking_row["candidate_id"], {})
                rows.append(
                    {
                        "example_id": example["example_id"],
                        "source": example["source"],
                        "method": method_name,
                        "candidate_id": ranking_row["candidate_id"],
                        "source_label": candidate.get("source_label"),
                        "quality_bucket": candidate.get("quality_bucket"),
                        "rank": ranking_row.get("rank"),
                        "score": ranking_row.get("score"),
                    }
                )
    return rows


def flatten_compressed_banks(example_artifacts: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for artifact in example_artifacts:
        example = artifact["example"]
        for method_name, method_payload in artifact["methods"].items():
            for entry in method_payload.get("compressed_bank", []):
                rows.append(
                    {
                        "example_id": example["example_id"],
                        "source": example["source"],
                        "method": method_name,
                        "category_id": entry.get("category_id"),
                        "label": entry.get("label"),
                        "family": entry.get("family"),
                        "canonical_text": entry.get("canonical_text"),
                        "representative_rubric_id": entry.get("representative_rubric_id"),
                        "representative_text": entry.get("representative_text"),
                        "member_count": entry.get("member_count"),
                        "max_coverage_count": entry.get("max_coverage_count"),
                    }
                )
    return rows


def flatten_production_banks(example_artifacts: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for artifact in example_artifacts:
        example = artifact["example"]
        for method_name, method_payload in artifact["methods"].items():
            utility = method_payload.get("production_bank_utility", {})
            for entry in method_payload.get("production_bank", []):
                rows.append(
                    {
                        "example_id": example["example_id"],
                        "source": example["source"],
                        "method": method_name,
                        "production_rubric_id": entry.get("production_rubric_id"),
                        "group_id": entry.get("group_id"),
                        "label": entry.get("label"),
                        "family": entry.get("family"),
                        "canonical_text": entry.get("canonical_text"),
                        "conditionality": entry.get("conditionality"),
                        "importance_tier": entry.get("importance_tier"),
                        "action_taken": entry.get("action_taken"),
                        "source_member_count": entry.get("source_member_count"),
                        "coverage_count": entry.get("coverage_count"),
                        "discrimination_score": entry.get("discrimination_score"),
                        "utility_overall": utility.get("overall_usefulness"),
                        "utility_reasoning": utility.get("brief_reasoning"),
                    }
                )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(metric_rows: List[Dict[str, object]]) -> str:
    lines = [
        "# RRD Rubric Pipeline Summary",
        "",
        "| Method | Reference Top-1 | Strong>Weak Accuracy | Mean Margin | Avg Rubrics | Avg Compressed | Avg Production | Avg Rejected | Avg Superseded |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            "| {method} | {reference_top1_rate:.3f} | {strong_vs_weak_accuracy:.3f} | "
            "{mean_strong_weak_margin:.3f} | {avg_rubric_count:.2f} | {avg_compressed_rubric_count:.2f} | "
            "{avg_production_rubric_count:.2f} | {avg_rejected_rubrics:.2f} | {avg_superseded_rubrics:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("Higher `reference_top1_rate` and `strong_vs_weak_accuracy` are better.")
    return "\n".join(lines)


def write_reports(run_dir: Path, example_artifacts: List[Dict[str, object]]) -> Dict[str, Path]:
    metric_rows = aggregate_method_metrics(example_artifacts)
    ranking_rows = flatten_rankings(example_artifacts)
    compressed_bank_rows = flatten_compressed_banks(example_artifacts)
    production_bank_rows = flatten_production_banks(example_artifacts)

    metric_path = run_dir / "reports" / "method_metrics.csv"
    rankings_path = run_dir / "reports" / "candidate_rankings.csv"
    compressed_path = run_dir / "reports" / "compressed_banks.csv"
    production_path = run_dir / "reports" / "production_banks.csv"
    summary_json_path = run_dir / "summaries" / "summary.json"
    summary_md_path = run_dir / "summaries" / "summary.md"

    _write_csv(metric_path, metric_rows)
    _write_csv(rankings_path, ranking_rows)
    _write_csv(compressed_path, compressed_bank_rows)
    _write_csv(production_path, production_bank_rows)
    write_json(
        summary_json_path,
        {
            "method_metrics": metric_rows,
            "compressed_banks": compressed_bank_rows,
            "production_banks": production_bank_rows,
            "examples": example_artifacts,
        },
    )
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text(_summary_markdown(metric_rows), encoding="utf-8")

    return {
        "method_metrics_csv": metric_path,
        "candidate_rankings_csv": rankings_path,
        "compressed_banks_csv": compressed_path,
        "production_banks_csv": production_path,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
    }
