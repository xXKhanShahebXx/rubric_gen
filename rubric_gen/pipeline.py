from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from rubric_gen.candidate_generation import build_candidate_pool
from rubric_gen.config import PipelineConfig
from rubric_gen.dataio import load_examples
from rubric_gen.evaluation.bank_judge import judge_rubric_bank
from rubric_gen.evaluation.baselines import (
    run_direct_judge_baseline,
    run_bank_method,
    run_one_shot_baseline,
    run_static_healthcare_baseline,
)
from rubric_gen.evaluation.reporting import write_reports
from rubric_gen.llm_client import LLMRouter
from rubric_gen.rrd.compression import compress_rubric_bank
from rubric_gen.rrd.engine import RRDEngine
from rubric_gen.rrd.production_bank import build_production_bank
from rubric_gen.storage import JsonlCache, read_json, write_json


def _expand_weighted_methods(result: Dict[str, object], prefix: str) -> Dict[str, Dict[str, object]]:
    methods: Dict[str, Dict[str, object]] = {
        f"{prefix}_uniform": {
            "ranking": result["uniform"]["ranking"],
            "weights": result["uniform"]["weights"],
            "rubrics": result["rubrics"],
            "evaluations": result["evaluations"],
            "weighting": "uniform",
        },
        f"{prefix}_whitened_uniform": {
            "ranking": result["whitened_uniform"]["ranking"],
            "weights": result["whitened_uniform"]["weights"],
            "weight_debug": result["whitened_uniform"].get("debug", {}),
            "rubrics": result["rubrics"],
            "evaluations": result["evaluations"],
            "weighting": "whitened_uniform",
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
    if "baseline_name" in result:
        for payload in methods.values():
            payload["baseline_name"] = result["baseline_name"]
    return methods


class RubricPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.layout = config.artifact_layout()
        self.router = None if config.dry_run else LLMRouter(max_retries=3)
        cache_enabled = not config.no_cache
        self.candidate_generation_cache = JsonlCache(self.layout.cache_dir / "candidate_generation.jsonl", enabled=cache_enabled)
        self.rubric_proposal_cache = JsonlCache(self.layout.cache_dir / "rubric_proposals.jsonl", enabled=cache_enabled)
        self.rubric_filter_cache = JsonlCache(self.layout.cache_dir / "rubric_filters.jsonl", enabled=cache_enabled)
        self.rubric_satisfaction_cache = JsonlCache(self.layout.cache_dir / "rubric_satisfaction.jsonl", enabled=cache_enabled)
        self.direct_judge_cache = JsonlCache(self.layout.cache_dir / "direct_judge.jsonl", enabled=cache_enabled)
        self.rubric_bank_judge_cache = JsonlCache(self.layout.cache_dir / "rubric_bank_judgments.jsonl", enabled=cache_enabled)
        self.engine = RRDEngine(
            config=config,
            router=self.router,
            proposal_cache=self.rubric_proposal_cache,
            filter_cache=self.rubric_filter_cache,
            satisfaction_cache=self.rubric_satisfaction_cache,
        )

    def _example_artifact_path(self, example_id: str):
        return self.layout.examples_dir / f"{example_id}.json"

    def _run_example(self, example) -> Dict[str, object]:
        candidates = build_candidate_pool(
            example=example,
            config=self.config,
            router=self.router,
            generation_cache=self.candidate_generation_cache,
        )
        rrd_result = self.engine.run_rrd(example, candidates)
        compression = compress_rubric_bank(rrd_result["rubrics"])
        rrd_result["compressed_bank"] = compression["compressed_bank"]
        rrd_result["compression_summary"] = {
            "raw_rubric_count": compression["raw_rubric_count"],
            "compressed_rubric_count": compression["compressed_rubric_count"],
            "family_counts": compression["family_counts"],
            "unmapped_rubrics": compression["unmapped_rubrics"],
        }
        production = build_production_bank(
            rubrics=rrd_result["rubrics"],
            evaluations=rrd_result["evaluations"],
            compressed_bank=rrd_result["compressed_bank"],
            candidates=[asdict(candidate) for candidate in candidates],
        )
        rrd_result["production_bank"] = production["production_bank"]
        rrd_result["production_bank_summary"] = production["production_bank_summary"]
        if not self.config.skip_bank_utility:
            rrd_result["bank_judgment"] = judge_rubric_bank(
                config=self.config,
                router=self.router,
                cache=self.rubric_bank_judge_cache,
                example=example,
                candidates=candidates,
                rubrics=rrd_result["rubrics"],
                proposer_label=self.config.rubric_proposer.alias if self.config.rubric_proposer else "standard",
                stage="rrd",
            )
            rrd_result["production_bank_utility"] = judge_rubric_bank(
                config=self.config,
                router=self.router,
                cache=self.rubric_bank_judge_cache,
                example=example,
                candidates=candidates,
                rubrics=rrd_result["production_bank"],
                proposer_label=self.config.rubric_proposer.alias if self.config.rubric_proposer else "standard",
                stage="production_bank",
            )

        methods = {}
        methods.update(_expand_weighted_methods(rrd_result, "rrd"))

        if self.config.rubrics_only:
            one_shot_rubrics = self.engine.propose_initial_rubrics(example, candidates, include_responses=True)
            methods["one_shot_rubrics_only"] = {
                "rubrics": one_shot_rubrics,
                "rubric_count": len(one_shot_rubrics),
            }
            return {
                "example": asdict(example),
                "candidates": [asdict(candidate) for candidate in candidates],
                "methods": methods,
            }

        compressed_bank_result = run_bank_method(
            engine=self.engine,
            example=example,
            candidates=candidates,
            bank_entries=rrd_result["compressed_bank"],
            source_stage="compressed_bank",
            method_prefix="compressed_bank",
            baseline_name="compressed_bank",
            text_key="canonical_text",
        )
        production_bank_result = run_bank_method(
            engine=self.engine,
            example=example,
            candidates=candidates,
            bank_entries=rrd_result["production_bank"],
            source_stage="production_bank",
            method_prefix="production_bank",
            baseline_name="production_bank",
            text_key="canonical_text",
        )
        if not self.config.skip_bank_utility:
            compressed_bank_result["bank_judgment"] = judge_rubric_bank(
                config=self.config,
                router=self.router,
                cache=self.rubric_bank_judge_cache,
                example=example,
                candidates=candidates,
                rubrics=compressed_bank_result["rubrics"],
                proposer_label=self.config.rubric_proposer.alias if self.config.rubric_proposer else "standard",
                stage="compressed_bank",
            )
            production_bank_result["bank_judgment"] = rrd_result["production_bank_utility"]
        one_shot_result = run_one_shot_baseline(self.engine, example, candidates)
        static_result = run_static_healthcare_baseline(self.engine, example, candidates)
        direct_result = run_direct_judge_baseline(
            config=self.config,
            router=self.router,
            judge_cache=self.direct_judge_cache,
            example=example,
            candidates=candidates,
        )

        methods.update(_expand_weighted_methods(compressed_bank_result, "compressed_bank"))
        methods.update(_expand_weighted_methods(production_bank_result, "production_bank"))
        methods.update(_expand_weighted_methods(one_shot_result, "one_shot"))
        methods.update(_expand_weighted_methods(static_result, "static_healthcare"))
        methods["direct_judge"] = direct_result

        return {
            "example": asdict(example),
            "candidates": [asdict(candidate) for candidate in candidates],
            "methods": methods,
        }

    def run(self) -> Dict[str, object]:
        examples = load_examples(
            dataset_path=self.config.dataset_path,
            start=self.config.start,
            limit=self.config.limit,
            source_filter=self.config.source_filter,
            split=self.config.split,
            train_size=self.config.train_size,
            val_size=self.config.val_size,
            num_shards=self.config.num_shards,
            shard_index=self.config.shard_index,
            reference_field_overrides=self.config.reference_fields,
        )
        write_json(
            self.layout.run_dir / "normalized_examples.json",
            {"examples": [asdict(example) for example in examples]},
        )
        write_json(
            self.layout.run_dir / "split_manifest.json",
            {
                "dataset_path": str(self.config.dataset_path),
                "preset": self.config.preset,
                "split": self.config.split,
                "train_size": self.config.train_size,
                "val_size": self.config.val_size,
                "num_shards": self.config.num_shards,
                "shard_index": self.config.shard_index,
                "start_offset_within_shard": self.config.start,
                "limit_within_shard": self.config.limit,
                "source_filter": self.config.source_filter,
                "reference_fields": list(self.config.reference_fields or []),
                "loaded_example_count": len(examples),
                "first_source_id": examples[0].source_id if examples else None,
                "last_source_id": examples[-1].source_id if examples else None,
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

        report_paths = write_reports(self.layout.run_dir, example_artifacts)
        return {
            "run_name": self.config.run_name,
            "example_count": len(example_artifacts),
            "report_paths": {name: str(path) for name, path in report_paths.items()},
            "run_dir": str(self.layout.run_dir),
        }
