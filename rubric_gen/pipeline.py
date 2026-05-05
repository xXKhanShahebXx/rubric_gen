from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from rubric_gen.candidate_generation import build_candidate_pool
from rubric_gen.compiled.medical_rubric_index import (
    DEFAULT_EMBEDDING_MODEL_SPEC as _MEDICAL_INDEX_EMBEDDING_SPEC,
    MedicalRubricIndex,
    OpenAIEmbedder,
)
from rubric_gen.compiled.relevance_filter import (
    RelevanceFilterConfig,
    filter_relevant_criteria,
    parse_strictness as _parse_relevance_filter_strictness,
)
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
        self.relevance_filter_cache = JsonlCache(
            self.layout.cache_dir / "rubric_relevance_filter.jsonl", enabled=cache_enabled
        )
        self.medical_query_embedding_cache = JsonlCache(
            self.layout.cache_dir / "medical_query_embeddings.jsonl", enabled=cache_enabled
        )
        self.engine = RRDEngine(
            config=config,
            router=self.router,
            proposal_cache=self.rubric_proposal_cache,
            filter_cache=self.rubric_filter_cache,
            satisfaction_cache=self.rubric_satisfaction_cache,
        )
        # Validation-time retrieval state. The index and embedder are lazy because
        # most invocations (training, dry-runs, paper mode) don't need them. The
        # ``_should_retrieve`` predicate gates the actual retrieval call below.
        self._medical_index: MedicalRubricIndex | None = None
        self._medical_embedder: OpenAIEmbedder | None = None
        self._medical_index_load_error: str = ""
        self._relevance_filter_config = self._build_relevance_filter_config()

    # ------------------------------------------------------------------
    # Validation-time retrieval helpers
    # ------------------------------------------------------------------

    def _build_relevance_filter_config(self) -> RelevanceFilterConfig:
        if not self.config.medical_rubric_filter_enabled:
            return RelevanceFilterConfig(enabled=False)
        try:
            strictness = _parse_relevance_filter_strictness(
                self.config.medical_rubric_filter_strictness
            )
        except ValueError:
            strictness = _parse_relevance_filter_strictness(None)
        model_spec = self.config.medical_rubric_filter_model
        if model_spec is None:
            from rubric_gen.compiled.relevance_filter import DEFAULT_FILTER_MODEL

            model_spec = DEFAULT_FILTER_MODEL
        return RelevanceFilterConfig(
            enabled=True,
            model_spec=model_spec,
            strictness=strictness,
        )

    def _maybe_load_medical_index(self) -> None:
        if self._medical_index is not None or self._medical_index_load_error:
            return
        index_path = self.config.medical_rubric_index_path
        if index_path is None:
            return
        try:
            self._medical_index = MedicalRubricIndex.load(Path(index_path))
        except Exception as exc:  # pragma: no cover - load failures surface in debug
            self._medical_index = None
            self._medical_index_load_error = f"{type(exc).__name__}: {exc}"

    def _maybe_create_embedder(self) -> OpenAIEmbedder | None:
        if self.config.dry_run:
            return None
        if self._medical_embedder is None:
            self._medical_embedder = OpenAIEmbedder(model_spec=_MEDICAL_INDEX_EMBEDDING_SPEC)
        return self._medical_embedder

    def _should_retrieve(self) -> bool:
        # Retrieval fires when (a) the medical rubric index is supplied,
        # (b) the run is not a dry-run, (c) the retrieval top-k is positive,
        # and (d) the split is `val` (the original validation flow) OR `all`
        # (a standalone validation dataset such as medical_gpt41_answers_rl
        # which isn't a split of the training corpus). Training (split=`train`)
        # never retrieves -- the per-example RRD discovery there is the source
        # the index is BUILT from, so seeding it with itself would be circular.
        return (
            self.config.medical_rubric_index_path is not None
            and self.config.split in ("val", "all")
            and not self.config.dry_run
            and self.config.medical_rubric_retrieval_top_k > 0
        )

    def _example_artifact_path(self, example_id: str) -> Path:
        return self.layout.examples_dir / f"{example_id}.json"

    @staticmethod
    def _collect_filter_candidate_texts(example) -> List[str]:
        """Pick the candidate texts shown to the relevance filter.

        v2 Tier B1 fix: pair-mode datasets (e.g. medical_gpt5_b_regen_4k_rl)
        do not populate ``reference_artifact`` / ``augmented_artifact`` because
        the loader's ``_REFERENCE_ARTIFACT_KEYS`` does not list
        ``reference_answer_a/b``.  Without this fix the filter saw an empty
        ``[""]`` for pair-only rows and dropped almost every retrieved
        rubric as IRRELEVANT (19.8% zero-seed rate at the original 4k run).

        Order of preference:
          1. Both pair anchors when the example carries ``has_pair_candidates``.
          2. ``augmented_artifact`` then ``reference_artifact`` (legacy single-
             response rows).
          3. ``[""]`` only as a last resort -- equivalent to the old behaviour.
        """
        if getattr(example, "has_pair_candidates", False):
            texts = []
            if getattr(example, "pair_response_a", ""):
                texts.append(str(example.pair_response_a).strip())
            if getattr(example, "pair_response_b", ""):
                texts.append(str(example.pair_response_b).strip())
            texts = [t for t in texts if t]
            if texts:
                return texts
        single = (
            (getattr(example, "augmented_artifact", "") or "")
            or (getattr(example, "reference_artifact", "") or "")
        ).strip()
        return [single] if single else [""]

    def _process_example_to_disk(self, example) -> Dict[str, object]:
        """Run a single example end-to-end and persist its artifact.

        Safe to call concurrently from multiple sample workers: the underlying
        caches use an RLock and `write_json` writes one file in one open/close
        block.
        """
        artifact = self._run_example(example)
        write_json(self._example_artifact_path(example.example_id), artifact)
        return artifact

    def _retrieve_seed_rubrics_for_example(
        self, example
    ) -> Tuple[List[str], Dict[str, object]]:
        """Embed the example's prompt, hit the medical rubric index, then filter.

        Returns ``(seed_rubric_texts, debug_payload)``. ``debug_payload`` records the
        retrieval source, K, the criterion ids that survived the relevance filter,
        and the filter's per-criterion verdicts so the per-example artifact can be
        audited later. On any failure (missing index, embed error) the helper
        degrades gracefully: returns no seeds and records the error in the debug
        payload, so the validation run still completes with pure RRD discovery.
        """
        debug: Dict[str, object] = {
            "index_path": str(self.config.medical_rubric_index_path)
            if self.config.medical_rubric_index_path
            else "",
            "top_k": int(self.config.medical_rubric_retrieval_top_k),
            "filter_enabled": bool(self._relevance_filter_config.enabled),
            "filter_strictness": self._relevance_filter_config.strictness,
        }
        self._maybe_load_medical_index()
        if self._medical_index is None:
            debug["error"] = self._medical_index_load_error or "index_not_loaded"
            return [], debug
        embedder = self._maybe_create_embedder()
        if embedder is None:
            debug["error"] = "no_embedder_router"
            return [], debug

        prompt_text = (example.task_prompt or "").strip()
        if not prompt_text:
            debug["error"] = "empty_prompt"
            return [], debug

        try:
            embedding = embedder.embed_text(prompt_text)
        except Exception as exc:
            debug["error"] = f"embed_error:{type(exc).__name__}"
            return [], debug
        if not embedding:
            debug["error"] = "empty_embedding"
            return [], debug

        try:
            scored = self._medical_index.retrieve_top_k(
                embedding,
                k=int(self.config.medical_rubric_retrieval_top_k),
            )
        except Exception as exc:
            debug["error"] = f"retrieve_error:{type(exc).__name__}"
            return [], debug
        retrieved_items = [item for item, _score in scored]
        debug["retrieved_count"] = len(retrieved_items)
        debug["retrieved"] = [
            {"rubric_id": item.rubric_id, "score": float(score)}
            for item, score in scored
        ]

        criteria = [item.to_criterion() for item in retrieved_items]
        if self._relevance_filter_config.enabled and criteria:
            # v2 Tier B1: pair-mode candidate-text fix.
            # The original code passed only ``example.augmented_artifact or
            # example.reference_artifact`` to the relevance filter.  For the
            # 4k pair JSONL (`reference_answer_a` / `reference_answer_b`),
            # the loader's ``_REFERENCE_ARTIFACT_KEYS`` does NOT include
            # those fields, so on pair-only rows the filter saw an empty
            # ``[""]`` candidate text and drove the Sonnet judge toward
            # IRRELEVANT for nearly everything (19.8% of 4k rows ended up
            # with zero seed inputs).  When the example has pair candidates,
            # pass both anchor responses so the filter sees real text.
            candidate_texts = self._collect_filter_candidate_texts(example)
            kept_criteria, filter_debug = filter_relevant_criteria(
                criteria,
                prompt_text=prompt_text,
                candidate_texts=candidate_texts,
                config=self._relevance_filter_config,
                router=self.router,
                cache=self.relevance_filter_cache,
            )
            debug["filter"] = filter_debug
            criteria = kept_criteria

        seed_texts = [criterion.requirement for criterion in criteria if criterion.requirement]
        debug["seed_count"] = len(seed_texts)
        return seed_texts, debug

    def _run_example(self, example) -> Dict[str, object]:
        candidates = build_candidate_pool(
            example=example,
            config=self.config,
            router=self.router,
            generation_cache=self.candidate_generation_cache,
        )

        seed_initial_rubrics: List[str] = []
        retrieval_debug: Dict[str, object] = {}
        if self._should_retrieve():
            seed_initial_rubrics, retrieval_debug = self._retrieve_seed_rubrics_for_example(
                example
            )

        rrd_result = self.engine.run_rrd(
            example,
            candidates,
            seed_initial_rubrics=seed_initial_rubrics,
        )
        if retrieval_debug:
            rrd_result["retrieval_debug"] = retrieval_debug
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

        sample_workers = max(1, int(self.config.sample_workers or 1))

        # Resumable bookkeeping: split into already-finished (read from disk) and
        # pending (need processing) so we never re-submit a finished example to a
        # worker. Per-example artifact writes happen inline inside
        # `_process_example_to_disk`; both that write and the underlying JSONL
        # caches are thread-safe (storage.JsonlCache holds an RLock around set()),
        # so cross-sample parallelism does not need additional synchronisation.
        pending: List[Tuple[int, object]] = []
        results_by_index: Dict[int, Dict[str, object]] = {}
        for index, example in enumerate(examples):
            artifact_path = self._example_artifact_path(example.example_id)
            if self.config.resume and artifact_path.exists():
                results_by_index[index] = read_json(artifact_path)
            else:
                pending.append((index, example))

        if sample_workers <= 1:
            for index, example in pending:
                results_by_index[index] = self._process_example_to_disk(example)
        else:
            with ThreadPoolExecutor(max_workers=sample_workers) as pool:
                futures = {
                    pool.submit(self._process_example_to_disk, example): index
                    for index, example in pending
                }
                for future in as_completed(futures):
                    index = futures[future]
                    results_by_index[index] = future.result()

        # Restore dataset order so reports remain deterministic regardless of
        # which sample worker happened to finish first.
        example_artifacts: List[Dict[str, object]] = [
            results_by_index[index] for index in range(len(examples))
        ]

        report_paths = write_reports(self.layout.run_dir, example_artifacts)
        return {
            "run_name": self.config.run_name,
            "example_count": len(example_artifacts),
            "report_paths": {name: str(path) for name, path in report_paths.items()},
            "run_dir": str(self.layout.run_dir),
        }
