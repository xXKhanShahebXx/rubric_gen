from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


STRONG_BUCKETS = {"gold_like", "strong_anchor", "frontier_generated", "open_generated"}
WEAK_BUCKETS = {"synthetically_degraded"}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ranking_lookup(ranking: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {row["candidate_id"]: row for row in ranking}


def _reference_top1(candidates: List[Dict[str, object]], ranking: List[Dict[str, object]]) -> float:
    reference_ids = {
        candidate["candidate_id"]
        for candidate in candidates
        if candidate.get("source_label") == "reference_note"
    }
    if not reference_ids:
        return 0.0
    ranking_lookup = _ranking_lookup(ranking)
    for candidate_id in reference_ids:
        ranked = ranking_lookup.get(candidate_id)
        if ranked and int(ranked.get("rank", 0)) == 1:
            return 1.0
    return 0.0


def _strong_vs_weak_accuracy(candidates: List[Dict[str, object]], ranking: List[Dict[str, object]]) -> Tuple[float, float]:
    ranking_lookup = _ranking_lookup(ranking)
    strong = [candidate for candidate in candidates if candidate.get("quality_bucket") in STRONG_BUCKETS]
    weak = [candidate for candidate in candidates if candidate.get("quality_bucket") in WEAK_BUCKETS]
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
    accuracy = float(correct) / total if total else 0.0
    margin = mean(margins) if margins else 0.0
    return accuracy, margin


def _stage_rows(example_artifacts: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for artifact in example_artifacts:
        example = artifact["example"]
        candidates = artifact["candidates"]
        for proposer_label, proposer_payload in artifact["comparison"].items():
            proposer_model = proposer_payload["proposer_model"]["model"]
            for stage_name in ("one_shot", "rrd"):
                stage_payload = proposer_payload[stage_name]
                bank = stage_payload["bank_judgment"]
                uniform_accuracy, uniform_margin = _strong_vs_weak_accuracy(candidates, stage_payload["uniform"]["ranking"])
                wu_accuracy, wu_margin = _strong_vs_weak_accuracy(candidates, stage_payload["whitened_uniform"]["ranking"])
                rows.append(
                    {
                        "example_id": example["example_id"],
                        "source": example["source"],
                        "proposer_label": proposer_label,
                        "proposer_model": proposer_model,
                        "stage": stage_name,
                        "raw_rubric_count": len(stage_payload["rubrics"]),
                        "compressed_rubric_count": len(stage_payload.get("compressed_bank", [])),
                        "coverage": bank["coverage"],
                        "atomicity": bank["atomicity"],
                        "redundancy": bank["redundancy"],
                        "directionality": bank["directionality"],
                        "executability": bank["executability"],
                        "overall_usefulness": bank["overall_usefulness"],
                        "uniform_reference_top1": _reference_top1(candidates, stage_payload["uniform"]["ranking"]),
                        "uniform_strong_vs_weak_accuracy": uniform_accuracy,
                        "uniform_mean_margin": uniform_margin,
                        "wu_reference_top1": _reference_top1(candidates, stage_payload["whitened_uniform"]["ranking"]),
                        "wu_strong_vs_weak_accuracy": wu_accuracy,
                        "wu_mean_margin": wu_margin,
                        "brief_reasoning": bank["brief_reasoning"],
                    }
                )
    return rows


def _aggregate_comparison_rows(stage_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_proposer: Dict[str, Dict[str, object]] = {}
    score_fields = [
        "coverage",
        "atomicity",
        "redundancy",
        "directionality",
        "executability",
        "overall_usefulness",
        "raw_rubric_count",
        "compressed_rubric_count",
        "uniform_reference_top1",
        "uniform_strong_vs_weak_accuracy",
        "uniform_mean_margin",
        "wu_reference_top1",
        "wu_strong_vs_weak_accuracy",
        "wu_mean_margin",
    ]

    for row in stage_rows:
        proposer = str(row["proposer_label"])
        proposer_entry = by_proposer.setdefault(
            proposer,
            {
                "proposer_label": proposer,
                "proposer_model": row["proposer_model"],
                "examples": set(),
                "one_shot": {field: [] for field in score_fields},
                "rrd": {field: [] for field in score_fields},
            },
        )
        proposer_entry["examples"].add(row["example_id"])
        stage_bucket = proposer_entry[str(row["stage"])]
        for field in score_fields:
            stage_bucket[field].append(float(row[field]))

    aggregated: List[Dict[str, object]] = []
    for proposer_label, proposer_entry in sorted(by_proposer.items()):
        row: Dict[str, object] = {
            "proposer_label": proposer_label,
            "proposer_model": proposer_entry["proposer_model"],
            "examples": len(proposer_entry["examples"]),
        }
        for field in score_fields:
            one_shot_mean = mean(proposer_entry["one_shot"][field]) if proposer_entry["one_shot"][field] else 0.0
            rrd_mean = mean(proposer_entry["rrd"][field]) if proposer_entry["rrd"][field] else 0.0
            row[f"one_shot_{field}"] = one_shot_mean
            row[f"rrd_{field}"] = rrd_mean
            row[f"delta_{field}"] = rrd_mean - one_shot_mean
        aggregated.append(row)
    return aggregated


def _summary_markdown(comparison_rows: List[Dict[str, object]]) -> str:
    lines = [
        "# RRD Proposer Comparison",
        "",
        "## Bank Judge Deltas",
        "",
        "| Proposer | Examples | Delta Overall | Delta Coverage | Delta Atomicity | Delta Redundancy | Delta Directionality | Delta Executability | Delta Raw Rubrics | Delta Compressed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            "| {proposer_label} | {examples} | {delta_overall_usefulness:.3f} | {delta_coverage:.3f} | "
            "{delta_atomicity:.3f} | {delta_redundancy:.3f} | {delta_directionality:.3f} | "
            "{delta_executability:.3f} | {delta_raw_rubric_count:.3f} | {delta_compressed_rubric_count:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Downstream Deltas",
            "",
            "| Proposer | Delta Uniform Strong>Weak | Delta WU Strong>Weak | Delta Uniform Margin | Delta WU Margin |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| {proposer_label} | {delta_uniform_strong_vs_weak_accuracy:.3f} | "
            "{delta_wu_strong_vs_weak_accuracy:.3f} | {delta_uniform_mean_margin:.3f} | "
            "{delta_wu_mean_margin:.3f} |".format(**row)
        )
    return "\n".join(lines)


def _flatten_production_banks(example_artifacts: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for artifact in example_artifacts:
        example = artifact["example"]
        for proposer_label, proposer_payload in artifact["comparison"].items():
            stage_payload = proposer_payload.get("rrd", {})
            utility = stage_payload.get("production_bank_utility", {})
            for entry in stage_payload.get("production_bank", []):
                rows.append(
                    {
                        "example_id": example["example_id"],
                        "source": example["source"],
                        "proposer_label": proposer_label,
                        "proposer_model": proposer_payload["proposer_model"]["model"],
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


def write_comparison_reports(run_dir: Path, example_artifacts: List[Dict[str, object]]) -> Dict[str, Path]:
    stage_rows = _stage_rows(example_artifacts)
    comparison_rows = _aggregate_comparison_rows(stage_rows)
    production_bank_rows = _flatten_production_banks(example_artifacts)

    bank_judgments_path = run_dir / "reports" / "rubric_bank_judgments.csv"
    comparison_path = run_dir / "reports" / "proposer_comparison.csv"
    production_bank_path = run_dir / "reports" / "production_banks.csv"
    summary_path = run_dir / "summaries" / "comparison_summary.md"

    _write_csv(bank_judgments_path, stage_rows)
    _write_csv(comparison_path, comparison_rows)
    _write_csv(production_bank_path, production_bank_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary_markdown(comparison_rows), encoding="utf-8")

    return {
        "rubric_bank_judgments_csv": bank_judgments_path,
        "proposer_comparison_csv": comparison_path,
        "production_banks_csv": production_bank_path,
        "comparison_summary_md": summary_path,
    }
