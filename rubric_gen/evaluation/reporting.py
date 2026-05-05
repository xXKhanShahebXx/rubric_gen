from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rubric_gen.storage import write_json


STRONG_BUCKETS = {"gold_like", "strong_anchor", "frontier_generated", "open_generated", "pair_anchor"}
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


def _pair_preference_outcome(
    example_artifact: Dict[str, object],
    ranking: List[Dict[str, object]],
) -> Tuple[bool, Optional[bool]]:
    """Compute the pair-preference outcome for one example under one method.

    Returns ``(evaluable, pipeline_correct)`` where:

    - ``evaluable`` is True when both pair anchors and a valid label are
      present and both anchors appear in the ranking.
    - ``pipeline_correct`` is True iff the higher-scoring pair anchor
      matches ``pair_correct_label``. A pipeline-side tie (equal raw
      scores) counts as wrong since the ground-truth label is strictly A
      or B and we treat ``A=B`` as "no preference expressed".
      ``None`` when not evaluable.

    Why scores instead of ranks: the upstream ranker assigns distinct
    ranks to score-tied candidates via a stable secondary sort
    (alphabetical ``candidate_id``). For pair anchors that secondary sort
    silently always picks A, hiding the model's "I have no preference"
    signal. Mirroring JudgeBench's ``compiled/judgebench_eval.py``
    behaviour, where a strict ``decision == "A=B"`` is honoured rather
    than being broken arbitrarily, we read raw scores and treat ties as
    wrong. This typically affects only a small fraction of rows (the
    ones where the rubric satisfaction calls happened to land on the
    same numeric score for both anchors).
    """
    candidates = example_artifact.get("candidates", []) or []
    pair_a = next((c for c in candidates if c.get("source_label") == "pair_response_a"), None)
    pair_b = next((c for c in candidates if c.get("source_label") == "pair_response_b"), None)
    if not (pair_a and pair_b):
        return False, None
    label = ""
    metadata_a = pair_a.get("metadata") or {}
    if isinstance(metadata_a, dict):
        label = str(metadata_a.get("pair_correct_label") or "").strip().lower()
    if label not in {"a", "b"}:
        return False, None
    lookup = _ranking_lookup(ranking)
    a_row = lookup.get(pair_a["candidate_id"])
    b_row = lookup.get(pair_b["candidate_id"])
    if not a_row or not b_row:
        return False, None
    a_score = a_row.get("score")
    b_score = b_row.get("score")
    if a_score is None or b_score is None:
        # Defensive fallback to ranks if a method ever omits the score
        # (current pipeline always emits it on the ranking entry).
        a_rank = a_row.get("rank")
        b_rank = b_row.get("rank")
        if a_rank is None or b_rank is None:
            return False, None
        if int(a_rank) == int(b_rank):
            return True, False
        pipeline_winner = "a" if int(a_rank) < int(b_rank) else "b"
        return True, pipeline_winner == label
    a_score_f = float(a_score)
    b_score_f = float(b_score)
    if a_score_f == b_score_f:
        # Pipeline-side tie: equal raw scores -> "A=B" decision -> wrong.
        return True, False
    pipeline_winner = "a" if a_score_f > b_score_f else "b"
    return True, pipeline_winner == label


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
                    "pair_preference_evaluable": 0,
                    "pair_preference_correct": 0,
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

            pair_evaluable, pair_correct = _pair_preference_outcome(artifact, ranking)
            if pair_evaluable:
                metrics["pair_preference_evaluable"] += 1
                metrics["pair_preference_correct"] += int(bool(pair_correct))

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
        pair_preference_evaluable = int(metrics["pair_preference_evaluable"])
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
                "pair_preference_accuracy": (
                    float(metrics["pair_preference_correct"]) / pair_preference_evaluable
                    if pair_preference_evaluable
                    else 0.0
                ),
                "pair_preference_correct": int(metrics["pair_preference_correct"]),
                "pair_preference_evaluable": pair_preference_evaluable,
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
    # Only show the pair-preference column when at least one example was
    # evaluable for it; otherwise the run is non-pair and the column would
    # be a sea of zeros that distracts from the real metrics.
    pair_evaluable_total = sum(int(row.get("pair_preference_evaluable", 0)) for row in metric_rows)
    show_pair = pair_evaluable_total > 0
    lines = ["# RRD Rubric Pipeline Summary", ""]
    if show_pair:
        lines.append(
            "| Method | Pair Pref Acc | Pair Pref (correct/n) | Reference Top-1 | "
            "Strong>Weak Accuracy | Mean Margin | Avg Rubrics | Avg Compressed | "
            "Avg Production | Avg Rejected | Avg Superseded |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in metric_rows:
            pair_acc = float(row.get("pair_preference_accuracy", 0.0) or 0.0)
            pair_correct = int(row.get("pair_preference_correct", 0) or 0)
            pair_n = int(row.get("pair_preference_evaluable", 0) or 0)
            lines.append(
                "| {method} | {pair_acc:.3f} | {pair_correct}/{pair_n} | "
                "{reference_top1_rate:.3f} | {strong_vs_weak_accuracy:.3f} | "
                "{mean_strong_weak_margin:.3f} | {avg_rubric_count:.2f} | "
                "{avg_compressed_rubric_count:.2f} | {avg_production_rubric_count:.2f} | "
                "{avg_rejected_rubrics:.2f} | {avg_superseded_rubrics:.2f} |".format(
                    pair_acc=pair_acc,
                    pair_correct=pair_correct,
                    pair_n=pair_n,
                    **row,
                )
            )
    else:
        lines.append(
            "| Method | Reference Top-1 | Strong>Weak Accuracy | Mean Margin | "
            "Avg Rubrics | Avg Compressed | Avg Production | Avg Rejected | Avg Superseded |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in metric_rows:
            lines.append(
                "| {method} | {reference_top1_rate:.3f} | {strong_vs_weak_accuracy:.3f} | "
                "{mean_strong_weak_margin:.3f} | {avg_rubric_count:.2f} | {avg_compressed_rubric_count:.2f} | "
                "{avg_production_rubric_count:.2f} | {avg_rejected_rubrics:.2f} | {avg_superseded_rubrics:.2f} |".format(**row)
            )
    lines.append("")
    if show_pair:
        lines.append(
            "Higher `pair_preference_accuracy` (the JudgeBench-style metric vs "
            "ground-truth labels), `reference_top1_rate`, and "
            "`strong_vs_weak_accuracy` are better."
        )
    else:
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
