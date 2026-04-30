from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from rubric_gen.candidate_generation import build_paper_candidate_pool
from rubric_gen.config import PipelineConfig
from rubric_gen.dataio import load_examples
from rubric_gen.evaluation.baselines import (
    run_direct_judge_baseline,
    run_one_shot_baseline,
    run_prompt_only_baseline,
)
from rubric_gen.evaluation.pairwise import build_judge_pairwise_preferences, build_proxy_pairwise_preferences
from rubric_gen.evaluation.paper_reporting import write_paper_reports
from rubric_gen.llm_client import LLMRouter
from rubric_gen.rrd.engine import RRDEngine
from rubric_gen.storage import JsonlCache, read_json, write_json
from rubric_gen.types import CandidateNote


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
    if "baseline_name" in result:
        for payload in methods.values():
            payload["baseline_name"] = result["baseline_name"]
    return methods


class PaperModePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.layout = config.artifact_layout()
        self.router = None if config.dry_run else LLMRouter(max_retries=3)
        cache_enabled = not config.no_cache
        self.candidate_generation_cache = JsonlCache(self.layout.cache_dir / "paper_candidate_generation.jsonl", enabled=cache_enabled)
        self.rubric_proposal_cache = JsonlCache(self.layout.cache_dir / "paper_rubric_proposals.jsonl", enabled=cache_enabled)
        self.rubric_filter_cache = JsonlCache(self.layout.cache_dir / "paper_rubric_filters.jsonl", enabled=cache_enabled)
        self.rubric_satisfaction_cache = JsonlCache(self.layout.cache_dir / "paper_rubric_satisfaction.jsonl", enabled=cache_enabled)
        self.direct_judge_cache = JsonlCache(self.layout.cache_dir / "paper_direct_judge.jsonl", enabled=cache_enabled)
        self.paper_pairwise_label_cache = JsonlCache(self.layout.cache_dir / "paper_pairwise_labels.jsonl", enabled=cache_enabled)
        self.engine = RRDEngine(
            config=config,
            router=self.router,
            proposal_cache=self.rubric_proposal_cache,
            filter_cache=self.rubric_filter_cache,
            satisfaction_cache=self.rubric_satisfaction_cache,
        )

    def _example_artifact_path(self, example_id: str):
        return self.layout.examples_dir / f"{example_id}.json"

    def _paper_eval_candidates(self, example, proposal_candidates: List[CandidateNote]) -> List[CandidateNote]:
        candidates = list(proposal_candidates)
        if self.config.paper_include_reference_eval_anchor:
            if example.reference_note:
                candidates.append(
                    CandidateNote(
                        candidate_id=f"{example.example_id}__paper_reference_note",
                        example_id=example.example_id,
                        text=example.reference_note,
                        source_label="reference_note",
                        quality_bucket="gold_like",
                        origin_kind="anchor",
                    )
                )
            elif example.augmented_note:
                candidates.append(
                    CandidateNote(
                        candidate_id=f"{example.example_id}__paper_augmented_note",
                        example_id=example.example_id,
                        text=example.augmented_note,
                        source_label="augmented_note",
                        quality_bucket="strong_anchor",
                        origin_kind="anchor",
                    )
                )
        return candidates

    def _run_example(self, example) -> Dict[str, object]:
        proposal_candidates = build_paper_candidate_pool(
            example=example,
            config=self.config,
            router=self.router,
            generation_cache=self.candidate_generation_cache,
        )
        evaluation_candidates = self._paper_eval_candidates(example, proposal_candidates)

        prompt_only_result = run_prompt_only_baseline(
            engine=self.engine,
            example=example,
            proposal_candidates=proposal_candidates,
            evaluation_candidates=evaluation_candidates,
        )
        one_shot_result = run_one_shot_baseline(
            engine=self.engine,
            example=example,
            candidates=proposal_candidates,
            evaluation_candidates=evaluation_candidates,
            include_responses=True,
            method_prefix="one_shot",
            baseline_name="one_shot",
        )
        rrd_result = self.engine.run_rrd(
            example=example,
            candidates=proposal_candidates,
            evaluation_candidates=evaluation_candidates,
        )
        direct_result = run_direct_judge_baseline(
            config=self.config,
            router=self.router,
            judge_cache=self.direct_judge_cache,
            example=example,
            candidates=evaluation_candidates,
        )

        methods: Dict[str, Dict[str, object]] = {}
        methods["base_judge"] = direct_result
        methods.update(_expand_weighted_methods(prompt_only_result, "prompt_only"))
        methods.update(_expand_weighted_methods(one_shot_result, "one_shot"))
        methods.update(_expand_weighted_methods(rrd_result, "rrd"))

        pairwise_labels = (
            build_judge_pairwise_preferences(
                config=self.config,
                router=self.router,
                cache=self.paper_pairwise_label_cache,
                example=example,
                candidates=[asdict(candidate) for candidate in proposal_candidates],
            )
            if self.config.paper_pairwise_label_mode == "judge_proxy"
            else build_proxy_pairwise_preferences(
                [asdict(candidate) for candidate in evaluation_candidates],
                label_mode=self.config.paper_pairwise_label_mode,
            )
        )

        return {
            "example": asdict(example),
            "proposal_candidates": [asdict(candidate) for candidate in proposal_candidates],
            "candidates": [asdict(candidate) for candidate in evaluation_candidates],
            "pairwise_labels": pairwise_labels,
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
            self.layout.run_dir / "run_config.json",
            {
                "paper_mode": self.config.paper_mode,
                "no_cache": self.config.no_cache,
                "paper_settings": {
                    "target_candidate_count": self.config.target_candidate_count,
                    "include_reference_eval_anchor": self.config.paper_include_reference_eval_anchor,
                    "prompt_only_baseline": self.config.paper_prompt_only_baseline,
                    "response_only_judging": self.config.paper_response_only_judging,
                    "pairwise_label_mode": self.config.paper_pairwise_label_mode,
                    "decomposition_threshold": self.config.decomposition_threshold,
                    "termination_rejections": self.config.termination_rejections,
                },
                "writer_models": [asdict(spec) for spec in self.config.writer_models],
                "rubric_proposer": asdict(self.config.rubric_proposer) if self.config.rubric_proposer else None,
                "rubric_judge": asdict(self.config.rubric_judge) if self.config.rubric_judge else None,
                "paper_pairwise_judge": asdict(self.config.paper_pairwise_judge) if self.config.paper_pairwise_judge else None,
                "baseline_judge": asdict(self.config.baseline_judge) if self.config.baseline_judge else None,
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

        paper_report_paths = write_paper_reports(
            self.layout.run_dir,
            example_artifacts,
            label_mode=self.config.paper_pairwise_label_mode,
        )
        return {
            "run_name": self.config.run_name,
            "example_count": len(example_artifacts),
            "report_paths": {name: str(path) for name, path in paper_report_paths.items()},
            "run_dir": str(self.layout.run_dir),
        }
