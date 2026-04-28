from __future__ import annotations

from dataclasses import asdict, replace
from typing import Dict, List

from rubric_gen.candidate_generation import build_candidate_pool
from rubric_gen.config import PipelineConfig
from rubric_gen.dataio import load_examples
from rubric_gen.evaluation.bank_judge import judge_rubric_bank
from rubric_gen.evaluation.baselines import run_bank_method, run_one_shot_baseline
from rubric_gen.evaluation.comparison_reporting import write_comparison_reports
from rubric_gen.evaluation.reporting import write_reports
from rubric_gen.llm_client import LLMRouter
from rubric_gen.rrd.compression import compress_rubric_bank
from rubric_gen.rrd.engine import RRDEngine
from rubric_gen.rrd.production_bank import build_production_bank
from rubric_gen.storage import JsonlCache, read_json, write_json


def _expand_weighted_methods(
    result: Dict[str, object],
    prefix: str,
    proposer_label: str,
    proposer_model: str,
    stage: str,
) -> Dict[str, Dict[str, object]]:
    methods: Dict[str, Dict[str, object]] = {
        f"{prefix}_uniform": {
            "ranking": result["uniform"]["ranking"],
            "weights": result["uniform"]["weights"],
            "rubrics": result["rubrics"],
            "evaluations": result["evaluations"],
            "weighting": "uniform",
            "proposer_label": proposer_label,
            "proposer_model": proposer_model,
            "stage": stage,
        },
        f"{prefix}_whitened_uniform": {
            "ranking": result["whitened_uniform"]["ranking"],
            "weights": result["whitened_uniform"]["weights"],
            "weight_debug": result["whitened_uniform"].get("debug", {}),
            "rubrics": result["rubrics"],
            "evaluations": result["evaluations"],
            "weighting": "whitened_uniform",
            "proposer_label": proposer_label,
            "proposer_model": proposer_model,
            "stage": stage,
        },
    }
    if "rrd_artifact" in result:
        for payload in methods.values():
            payload["artifact"] = result["rrd_artifact"]
    if "compressed_bank" in result:
        for payload in methods.values():
            payload["compressed_bank"] = result["compressed_bank"]
            payload["compression_summary"] = result.get("compression_summary", {})
    if "production_bank" in result:
        for payload in methods.values():
            payload["production_bank"] = result["production_bank"]
            payload["production_bank_summary"] = result.get("production_bank_summary", {})
    if "bank_judgment" in result:
        for payload in methods.values():
            payload["bank_judgment"] = result["bank_judgment"]
    if "production_bank_utility" in result:
        for payload in methods.values():
            payload["production_bank_utility"] = result["production_bank_utility"]
    return methods


class ProposerComparisonPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.layout = config.artifact_layout()
        self.router = None if config.dry_run else LLMRouter(max_retries=3)
        cache_enabled = not config.no_cache
        self.candidate_generation_cache = JsonlCache(self.layout.cache_dir / "candidate_generation.jsonl", enabled=cache_enabled)
        self.rubric_proposal_cache = JsonlCache(self.layout.cache_dir / "rubric_proposals.jsonl", enabled=cache_enabled)
        self.rubric_filter_cache = JsonlCache(self.layout.cache_dir / "rubric_filters.jsonl", enabled=cache_enabled)
        self.rubric_satisfaction_cache = JsonlCache(self.layout.cache_dir / "rubric_satisfaction.jsonl", enabled=cache_enabled)
        self.rubric_bank_judge_cache = JsonlCache(self.layout.cache_dir / "rubric_bank_judgments.jsonl", enabled=cache_enabled)

    def _example_artifact_path(self, example_id: str):
        return self.layout.examples_dir / f"{example_id}.json"

    def _build_engine(self, proposer_model) -> RRDEngine:
        branch_config = replace(
            self.config,
            rubric_proposer=proposer_model,
            baseline_judge=self.config.downstream_note_judge or self.config.baseline_judge,
        )
        return RRDEngine(
            config=branch_config,
            router=self.router,
            proposal_cache=self.rubric_proposal_cache,
            filter_cache=self.rubric_filter_cache,
            satisfaction_cache=self.rubric_satisfaction_cache,
        )

    def _enrich_stage_result(
        self,
        example,
        candidates,
        proposer_label: str,
        stage: str,
        result: Dict[str, object],
    ) -> Dict[str, object]:
        compression = compress_rubric_bank(result["rubrics"])
        result["compressed_bank"] = compression["compressed_bank"]
        result["compression_summary"] = {
            "raw_rubric_count": compression["raw_rubric_count"],
            "compressed_rubric_count": compression["compressed_rubric_count"],
            "family_counts": compression["family_counts"],
            "unmapped_rubrics": compression["unmapped_rubrics"],
        }
        result["bank_judgment"] = judge_rubric_bank(
            config=self.config,
            router=self.router,
            cache=self.rubric_bank_judge_cache,
            example=example,
            candidates=candidates,
            rubrics=result["rubrics"],
            proposer_label=proposer_label,
            stage=stage,
        )
        return result

    def _run_example(self, example) -> Dict[str, object]:
        candidates = build_candidate_pool(
            example=example,
            config=self.config,
            router=self.router,
            generation_cache=self.candidate_generation_cache,
        )
        methods: Dict[str, Dict[str, object]] = {}
        comparison: Dict[str, Dict[str, object]] = {}

        for proposer_model in self.config.proposer_models:
            proposer_label = proposer_model.alias
            engine = self._build_engine(proposer_model)
            one_shot_result = self._enrich_stage_result(
                example,
                candidates,
                proposer_label,
                "one_shot",
                run_one_shot_baseline(engine, example, candidates),
            )
            rrd_result = self._enrich_stage_result(
                example,
                candidates,
                proposer_label,
                "rrd",
                engine.run_rrd(example, candidates),
            )
            production = build_production_bank(
                rubrics=rrd_result["rubrics"],
                evaluations=rrd_result["evaluations"],
                compressed_bank=rrd_result["compressed_bank"],
                candidates=[asdict(candidate) for candidate in candidates],
            )
            rrd_result["production_bank"] = production["production_bank"]
            rrd_result["production_bank_summary"] = production["production_bank_summary"]
            rrd_result["production_bank_utility"] = judge_rubric_bank(
                config=self.config,
                router=self.router,
                cache=self.rubric_bank_judge_cache,
                example=example,
                candidates=candidates,
                rubrics=rrd_result["production_bank"],
                proposer_label=proposer_label,
                stage="production_bank",
            )
            compressed_bank_result = run_bank_method(
                engine=engine,
                example=example,
                candidates=candidates,
                bank_entries=rrd_result["compressed_bank"],
                source_stage="compressed_bank",
                method_prefix="compressed_bank",
                baseline_name="compressed_bank",
                text_key="canonical_text",
            )
            production_bank_result = run_bank_method(
                engine=engine,
                example=example,
                candidates=candidates,
                bank_entries=rrd_result["production_bank"],
                source_stage="production_bank",
                method_prefix="production_bank",
                baseline_name="production_bank",
                text_key="canonical_text",
            )
            compressed_bank_result["bank_judgment"] = judge_rubric_bank(
                config=self.config,
                router=self.router,
                cache=self.rubric_bank_judge_cache,
                example=example,
                candidates=candidates,
                rubrics=compressed_bank_result["rubrics"],
                proposer_label=proposer_label,
                stage="compressed_bank",
            )
            production_bank_result["bank_judgment"] = rrd_result["production_bank_utility"]

            comparison[proposer_label] = {
                "proposer_model": asdict(proposer_model),
                "one_shot": one_shot_result,
                "rrd": rrd_result,
            }
            methods.update(
                _expand_weighted_methods(
                    one_shot_result,
                    prefix=f"{proposer_label}_one_shot",
                    proposer_label=proposer_label,
                    proposer_model=proposer_model.model,
                    stage="one_shot",
                )
            )
            methods.update(
                _expand_weighted_methods(
                    rrd_result,
                    prefix=f"{proposer_label}_rrd",
                    proposer_label=proposer_label,
                    proposer_model=proposer_model.model,
                    stage="rrd",
                )
            )
            methods.update(
                _expand_weighted_methods(
                    compressed_bank_result,
                    prefix=f"{proposer_label}_compressed_bank",
                    proposer_label=proposer_label,
                    proposer_model=proposer_model.model,
                    stage="compressed_bank",
                )
            )
            methods.update(
                _expand_weighted_methods(
                    production_bank_result,
                    prefix=f"{proposer_label}_production_bank",
                    proposer_label=proposer_label,
                    proposer_model=proposer_model.model,
                    stage="production_bank",
                )
            )

        return {
            "example": asdict(example),
            "candidates": [asdict(candidate) for candidate in candidates],
            "comparison": comparison,
            "methods": methods,
        }

    def run(self) -> Dict[str, object]:
        examples = load_examples(
            dataset_path=self.config.dataset_path,
            start=self.config.start,
            limit=self.config.limit,
            source_filter=self.config.source_filter,
        )
        write_json(
            self.layout.run_dir / "normalized_examples.json",
            {"examples": [asdict(example) for example in examples]},
        )
        write_json(
            self.layout.run_dir / "run_config.json",
            {
                "comparison_mode": self.config.comparison_mode,
                "no_cache": self.config.no_cache,
                "rrd_settings": {
                    "target_candidate_count": self.config.target_candidate_count,
                    "decomposition_threshold": self.config.decomposition_threshold,
                    "termination_rejections": self.config.termination_rejections,
                    "max_initial_rubrics": self.config.max_initial_rubrics,
                    "max_final_rubrics": self.config.max_final_rubrics,
                    "max_decomposition_depth": self.config.max_decomposition_depth,
                    "decomposition_min_recall": self.config.decomposition_min_recall,
                    "decomposition_max_extra_ratio": self.config.decomposition_max_extra_ratio,
                    "decomposition_min_discrimination_gain": self.config.decomposition_min_discrimination_gain,
                    "decomposition_max_pair_overlap": self.config.decomposition_max_pair_overlap,
                },
                "writer_models": [asdict(spec) for spec in self.config.writer_models],
                "proposer_models": [asdict(spec) for spec in self.config.proposer_models],
                "rubric_judge": asdict(self.config.rubric_judge) if self.config.rubric_judge else None,
                "rubric_bank_judge": asdict(self.config.rubric_bank_judge) if self.config.rubric_bank_judge else None,
                "downstream_note_judge": asdict(self.config.downstream_note_judge) if self.config.downstream_note_judge else None,
            },
        )

        example_artifacts: List[Dict[str, object]] = []
        for example in examples:
            artifact_path = self._example_artifact_path(example.example_id)
            if self.config.resume and artifact_path.exists():
                example_artifacts.append(read_json(artifact_path))
                continue

            example_artifact = self._run_example(example)
            write_json(artifact_path, example_artifact)
            example_artifacts.append(example_artifact)

        standard_report_paths = write_reports(self.layout.run_dir, example_artifacts)
        comparison_report_paths = write_comparison_reports(self.layout.run_dir, example_artifacts)
        report_paths = {**standard_report_paths, **comparison_report_paths}
        return {
            "run_name": self.config.run_name,
            "example_count": len(example_artifacts),
            "report_paths": {name: str(path) for name, path in report_paths.items()},
            "run_dir": str(self.layout.run_dir),
        }
