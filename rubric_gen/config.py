from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from rubric_gen.types import ModelSpec

load_dotenv(override=True)


DEFAULT_DATASET_PATH = Path("data/sample_30_aci_70_agbonnet.json")
DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_TARGET_CANDIDATES = 8
DEFAULT_DECOMPOSITION_THRESHOLD = 4
DEFAULT_TERMINATION_REJECTIONS = 15
DEFAULT_MAX_INITIAL_RUBRICS = 8
DEFAULT_MAX_FINAL_RUBRICS = 24
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_WORKERS = 4
DEFAULT_DECOMPOSITION_MIN_RECALL = 0.85
DEFAULT_DECOMPOSITION_MAX_EXTRA_RATIO = 0.15
DEFAULT_DECOMPOSITION_MIN_DISCRIMINATION_GAIN = 0.03
DEFAULT_DECOMPOSITION_MAX_PAIR_OVERLAP = 0.75


# Default 3000/2000 split aligns with the medical_rl_prompts team workflow.
DEFAULT_TRAIN_SIZE = 3000
DEFAULT_VAL_SIZE = 2000
DEFAULT_NUM_SHARDS = 3
DEFAULT_SHARD_INDEX = 0
ALLOWED_SPLITS = ("all", "train", "val")


# Presets that pin a complete configuration in one flag.
#
# `judgebench-v47-medical` locks in the rubric-judge core that JudgeBench v4.7
# (`jb_350_blind_v47_agreement_aware`, 80.57 % single-order on val_350) was
# built on, applied to single-response medical Q&A:
#   * GPT-4o for both rubric proposal and rubric-satisfaction judging, since
#     v4.7 keeps GPT-4o as the base judge for every rubric / discriminator
#     call.
#   * Multi-model writers (GPT-4o-mini + Claude Sonnet 4.x when keys are
#     available) so candidate diversity matches the multi-model RRD proposer
#     setup from v2.
#   * Default split / shard knobs sized for the 5,000-row medical_rl_prompts
#     workflow (3,000 train / 2,000 val / 3 shards of 1,000 rows each).
#
# The JudgeBench-specific HP-override verifiers (mmlu_independent_answerer,
# math_independent_solver, reasoning_independent_solver, code_execution
# verifier, leetcode_test_runner) are *not* part of this preset because they
# only fire on JudgeBench source families (mmlu-pro, livebench-{math,
# reasoning}, livecodebench) and have nothing to verify on medical Q&A.
PRESET_JUDGEBENCH_V47_MEDICAL = "judgebench-v47-medical"
ALLOWED_PRESETS = (PRESET_JUDGEBENCH_V47_MEDICAL,)


_V47_RUBRIC_MODEL = "openai:gpt-4o"
_V47_JUDGE_MODEL = "openai:gpt-4o"
_V47_WRITER_MODELS_BY_PROVIDER = {
    "openai": "openai:gpt-4o-mini",
    "anthropic": "anthropic:claude-sonnet-4-5-20250929",
}
# medical_rl_prompts ships its existing answer in the `response` column and we
# treat it as the gold answer (rank-1 target for `reference_top1_rate`). The
# preset promotes `response` into the reference slot so the loader does not
# also build a duplicate `augmented_note` anchor with the same text.
_V47_REFERENCE_FIELDS: List[str] = ["response"]


@dataclass
class ArtifactLayout:
    root_dir: Path
    cache_dir: Path
    run_dir: Path
    examples_dir: Path
    reports_dir: Path
    summaries_dir: Path

    def ensure(self) -> None:
        for path in [
            self.root_dir,
            self.cache_dir,
            self.run_dir,
            self.examples_dir,
            self.reports_dir,
            self.summaries_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    dataset_path: Path
    output_dir: Path
    run_name: str
    comparison_mode: bool = False
    paper_mode: bool = False
    no_cache: bool = False
    rubrics_only: bool = False
    skip_bank_utility: bool = False
    start: int = 0
    limit: int = 0
    source_filter: Optional[str] = None
    resume: bool = False
    dry_run: bool = False
    max_workers: int = DEFAULT_MAX_WORKERS
    sample_workers: int = 1
    target_candidate_count: int = DEFAULT_TARGET_CANDIDATES
    split: str = "all"
    train_size: int = 0
    val_size: int = 0
    num_shards: int = 1
    shard_index: int = 0
    preset: Optional[str] = None
    reference_fields: List[str] = field(default_factory=list)
    paper_include_reference_eval_anchor: bool = False
    paper_prompt_only_baseline: bool = True
    paper_response_only_judging: bool = False
    paper_pairwise_label_mode: str = "judge_proxy"
    decomposition_threshold: int = DEFAULT_DECOMPOSITION_THRESHOLD
    termination_rejections: int = DEFAULT_TERMINATION_REJECTIONS
    max_initial_rubrics: int = DEFAULT_MAX_INITIAL_RUBRICS
    max_final_rubrics: int = DEFAULT_MAX_FINAL_RUBRICS
    max_decomposition_depth: int = DEFAULT_MAX_DEPTH
    decomposition_min_recall: float = DEFAULT_DECOMPOSITION_MIN_RECALL
    decomposition_max_extra_ratio: float = DEFAULT_DECOMPOSITION_MAX_EXTRA_RATIO
    decomposition_min_discrimination_gain: float = DEFAULT_DECOMPOSITION_MIN_DISCRIMINATION_GAIN
    decomposition_max_pair_overlap: float = DEFAULT_DECOMPOSITION_MAX_PAIR_OVERLAP
    covariance_ridge: float = 1e-3
    writer_models: List[ModelSpec] = field(default_factory=list)
    rubric_proposer: Optional[ModelSpec] = None
    proposer_models: List[ModelSpec] = field(default_factory=list)
    rubric_judge: Optional[ModelSpec] = None
    rubric_bank_judge: Optional[ModelSpec] = None
    downstream_note_judge: Optional[ModelSpec] = None
    paper_pairwise_judge: Optional[ModelSpec] = None
    baseline_judge: Optional[ModelSpec] = None

    def artifact_layout(self) -> ArtifactLayout:
        root_dir = self.output_dir
        run_dir = root_dir / "runs" / self.run_name
        return ArtifactLayout(
            root_dir=root_dir,
            cache_dir=root_dir / "cache",
            run_dir=run_dir,
            examples_dir=run_dir / "examples",
            reports_dir=run_dir / "reports",
            summaries_dir=run_dir / "summaries",
        )


def default_run_name() -> str:
    return datetime.utcnow().strftime("rrd_%Y%m%d_%H%M%S")


def _env_model_name(name: str, fallback: str) -> str:
    return os.getenv(name, fallback)


def _sanitize_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _default_max_tokens(provider: str, model: str) -> int:
    lowered_model = model.lower()
    if lowered_model.startswith("gpt-5.4"):
        return 4096
    if lowered_model.startswith("gpt-5"):
        return 4096
    if provider.lower().strip() == "anthropic" and "opus-4-6" in lowered_model:
        return 4096
    return 2048


def _provider_to_spec(provider: str, model: str, alias: str) -> ModelSpec:
    normalized = provider.lower().strip()
    if normalized == "openai":
        return ModelSpec(
            alias=alias,
            provider="openai",
            model=model,
            api_key_env="OPENAI_API_KEY",
            max_tokens=_default_max_tokens(normalized, model),
        )
    if normalized == "anthropic":
        return ModelSpec(
            alias=alias,
            provider="anthropic",
            model=model,
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=_default_max_tokens(normalized, model),
        )
    if normalized in {"together", "openai_compatible"}:
        base_url = os.getenv(
            "TOGETHER_BASE_URL" if normalized == "together" else "OPENAI_COMPAT_BASE_URL",
            "https://api.together.xyz/v1" if normalized == "together" else None,
        )
        return ModelSpec(
            alias=alias,
            provider="openai_compatible",
            model=model,
            api_key_env="TOGETHER_API_KEY" if normalized == "together" else "OPENAI_COMPAT_API_KEY",
            base_url=base_url,
            max_tokens=_default_max_tokens(normalized, model),
        )
    raise ValueError(f"Unsupported provider '{provider}'.")


def parse_model_spec(
    raw: str,
    alias_prefix: str = "",
    index: int = 0,
    default_alias: Optional[str] = None,
) -> ModelSpec:
    explicit_alias: Optional[str] = None
    value = raw.strip()
    if "=" in value:
        left, right = value.split("=", 1)
        if ":" in right and left.strip():
            explicit_alias = left.strip()
            value = right.strip()

    if ":" not in value:
        raise ValueError(
            "Model spec must look like 'provider:model-name', for example "
            "'openai:gpt-4.1-mini' or 'together:meta-llama/Llama-3.3-70B-Instruct-Turbo'."
        )
    provider, model = value.split(":", 1)
    if explicit_alias:
        alias = _sanitize_alias(explicit_alias)
    elif default_alias:
        alias = _sanitize_alias(default_alias)
    elif alias_prefix:
        alias = f"{alias_prefix}-{index}"
    else:
        alias = _sanitize_alias(f"{provider}_{model}")
    return _provider_to_spec(provider=provider, model=model, alias=alias)


def discover_default_writer_models() -> List[ModelSpec]:
    models: List[ModelSpec] = []
    if os.getenv("OPENAI_API_KEY"):
        models.append(
            _provider_to_spec(
                "openai",
                _env_model_name("RUBRIC_GEN_OPENAI_WRITER_MODEL", "gpt-4.1-mini"),
                alias="openai-writer",
            )
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append(
            _provider_to_spec(
                "anthropic",
                _env_model_name("RUBRIC_GEN_ANTHROPIC_WRITER_MODEL", "claude-sonnet-4-20250514"),
                alias="anthropic-writer",
            )
        )
    if os.getenv("TOGETHER_API_KEY"):
        models.append(
            _provider_to_spec(
                "together",
                _env_model_name(
                    "RUBRIC_GEN_TOGETHER_WRITER_MODEL",
                    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                ),
                alias="together-writer",
            )
        )
    return models


def discover_default_rubric_model() -> Optional[ModelSpec]:
    if os.getenv("OPENAI_API_KEY"):
        return _provider_to_spec(
            "openai",
            _env_model_name("RUBRIC_GEN_RUBRIC_MODEL", "gpt-4.1"),
            alias="openai-rubric",
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        return _provider_to_spec(
            "anthropic",
            _env_model_name("RUBRIC_GEN_RUBRIC_MODEL", "claude-sonnet-4-20250514"),
            alias="anthropic-rubric",
        )
    if os.getenv("TOGETHER_API_KEY"):
        return _provider_to_spec(
            "together",
            _env_model_name(
                "RUBRIC_GEN_RUBRIC_MODEL",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            ),
            alias="together-rubric",
        )
    return None


def discover_default_comparison_proposers() -> List[ModelSpec]:
    proposer_models: List[ModelSpec] = []
    if os.getenv("OPENAI_API_KEY"):
        proposer_models.append(
            _provider_to_spec(
                "openai",
                _env_model_name("RUBRIC_GEN_OPENAI_PROPOSER_MODEL", "gpt-5"),
                alias="proposer_openai_gpt5",
            )
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        proposer_models.append(
            _provider_to_spec(
                "anthropic",
                _env_model_name("RUBRIC_GEN_ANTHROPIC_PROPOSER_MODEL", "claude-opus-4-6"),
                alias="proposer_anthropic_strongest",
            )
        )
    return proposer_models


def discover_default_judge_model() -> Optional[ModelSpec]:
    if os.getenv("OPENAI_API_KEY"):
        return _provider_to_spec(
            "openai",
            _env_model_name("RUBRIC_GEN_JUDGE_MODEL", "gpt-4.1-mini"),
            alias="openai-judge",
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        return _provider_to_spec(
            "anthropic",
            _env_model_name("RUBRIC_GEN_JUDGE_MODEL", "claude-3-5-haiku-latest"),
            alias="anthropic-judge",
        )
    if os.getenv("TOGETHER_API_KEY"):
        return _provider_to_spec(
            "together",
            _env_model_name(
                "RUBRIC_GEN_JUDGE_MODEL",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            ),
            alias="together-judge",
        )
    return None


def discover_compiled_llm_judge_model() -> Optional[ModelSpec]:
    """Model for `rubric_gen.compiled.llm_judge` — explicit env overrides default judge discovery."""
    raw = os.getenv("RUBRIC_GEN_COMPILED_JUDGE_MODEL", "").strip()
    if raw:
        return parse_model_spec(raw, default_alias="compiled-llm-judge")
    return discover_default_judge_model()


def discover_default_comparison_judge_model() -> Optional[ModelSpec]:
    if os.getenv("OPENAI_API_KEY"):
        return _provider_to_spec(
            "openai",
            _env_model_name("RUBRIC_GEN_COMPARISON_JUDGE_MODEL", "gpt-5.4"),
            alias="comparison_judge_gpt54",
        )
    return discover_default_judge_model()


def discover_default_bank_judge_model() -> Optional[ModelSpec]:
    if os.getenv("OPENAI_API_KEY"):
        return _provider_to_spec(
            "openai",
            _env_model_name("RUBRIC_GEN_BANK_JUDGE_MODEL", "gpt-5.4"),
            alias="bank_judge_gpt54",
        )
    return None


def discover_default_paper_pairwise_judge_model() -> Optional[ModelSpec]:
    if os.getenv("OPENAI_API_KEY"):
        return _provider_to_spec(
            "openai",
            _env_model_name("RUBRIC_GEN_PAPER_PAIRWISE_JUDGE_MODEL", "gpt-5.4"),
            alias="paper_pairwise_judge_gpt54",
        )
    return None


def discover_default_downstream_note_judge_model() -> Optional[ModelSpec]:
    if os.getenv("OPENAI_API_KEY"):
        return _provider_to_spec(
            "openai",
            _env_model_name("RUBRIC_GEN_DOWNSTREAM_JUDGE_MODEL", "gpt-5.4"),
            alias="downstream_note_judge",
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        return _provider_to_spec(
            "anthropic",
            _env_model_name("RUBRIC_GEN_DOWNSTREAM_JUDGE_MODEL", "claude-3-5-haiku-latest"),
            alias="downstream_note_judge",
        )
    if os.getenv("TOGETHER_API_KEY"):
        return _provider_to_spec(
            "together",
            _env_model_name(
                "RUBRIC_GEN_DOWNSTREAM_JUDGE_MODEL",
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            ),
            alias="downstream_note_judge",
        )
    return None


def _apply_preset_defaults(
    preset: Optional[str],
    *,
    rubric_model: Optional[str],
    judge_model: Optional[str],
    writer_models: Optional[List[str]],
    split: Optional[str],
    train_size: Optional[int],
    val_size: Optional[int],
    num_shards: Optional[int],
    reference_fields: Optional[List[str]],
) -> Dict[str, Any]:
    """Resolve preset defaults that user-supplied flags can still override.

    Returns a dict of effective values for the keys that presets care about.
    Each value is the user override when present, otherwise the preset value,
    otherwise the original default that `build_config` was already going to use.
    """
    if preset is not None and preset not in ALLOWED_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Allowed: {', '.join(ALLOWED_PRESETS)}."
        )

    effective: Dict[str, Any] = {
        "rubric_model": rubric_model,
        "judge_model": judge_model,
        "writer_models": writer_models,
        "split": split,
        "train_size": train_size,
        "val_size": val_size,
        "num_shards": num_shards,
        "reference_fields": reference_fields,
    }

    if preset == PRESET_JUDGEBENCH_V47_MEDICAL:
        if effective["rubric_model"] is None:
            effective["rubric_model"] = _V47_RUBRIC_MODEL
        if effective["judge_model"] is None:
            effective["judge_model"] = _V47_JUDGE_MODEL
        if not effective["writer_models"]:
            preset_writers: List[str] = []
            if os.getenv("OPENAI_API_KEY"):
                preset_writers.append(_V47_WRITER_MODELS_BY_PROVIDER["openai"])
            if os.getenv("ANTHROPIC_API_KEY"):
                preset_writers.append(_V47_WRITER_MODELS_BY_PROVIDER["anthropic"])
            if preset_writers:
                effective["writer_models"] = preset_writers
        if effective["split"] is None:
            effective["split"] = "train"
        if effective["train_size"] is None:
            effective["train_size"] = DEFAULT_TRAIN_SIZE
        if effective["val_size"] is None:
            effective["val_size"] = DEFAULT_VAL_SIZE
        if effective["num_shards"] is None:
            effective["num_shards"] = DEFAULT_NUM_SHARDS
        if not effective["reference_fields"]:
            effective["reference_fields"] = list(_V47_REFERENCE_FIELDS)

    return effective


def build_config(
    dataset_path: Optional[str],
    output_dir: Optional[str],
    run_name: Optional[str],
    comparison_mode: bool,
    paper_mode: bool,
    no_cache: bool,
    rubrics_only: bool,
    skip_bank_utility: bool,
    start: int,
    limit: int,
    source_filter: Optional[str],
    resume: bool,
    dry_run: bool,
    max_workers: int,
    sample_workers: int = 1,
    target_candidate_count: Optional[int] = None,
    decomposition_threshold: Optional[int] = None,
    max_initial_rubrics: Optional[int] = None,
    max_final_rubrics: Optional[int] = None,
    max_decomposition_depth: Optional[int] = None,
    writer_models: Optional[List[str]] = None,
    proposer_models: Optional[List[str]] = None,
    rubric_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    bank_judge_model: Optional[str] = None,
    downstream_judge_model: Optional[str] = None,
    paper_pairwise_label_mode: Optional[str] = None,
    paper_pairwise_judge_model: Optional[str] = None,
    split: Optional[str] = None,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    num_shards: Optional[int] = None,
    shard_index: int = 0,
    preset: Optional[str] = None,
    reference_fields: Optional[List[str]] = None,
) -> PipelineConfig:
    preset_resolved = _apply_preset_defaults(
        preset,
        rubric_model=rubric_model,
        judge_model=judge_model,
        writer_models=writer_models,
        split=split,
        train_size=train_size,
        val_size=val_size,
        num_shards=num_shards,
        reference_fields=reference_fields,
    )
    rubric_model = preset_resolved["rubric_model"]
    judge_model = preset_resolved["judge_model"]
    writer_models = preset_resolved["writer_models"]
    effective_split = preset_resolved["split"] if preset_resolved["split"] is not None else "all"
    effective_train_size = preset_resolved["train_size"] if preset_resolved["train_size"] is not None else 0
    effective_val_size = preset_resolved["val_size"] if preset_resolved["val_size"] is not None else 0
    effective_num_shards = preset_resolved["num_shards"] if preset_resolved["num_shards"] is not None else 1
    effective_reference_fields = list(preset_resolved["reference_fields"] or [])

    if effective_split not in ALLOWED_SPLITS:
        raise ValueError(
            f"--split must be one of {', '.join(ALLOWED_SPLITS)}; got '{effective_split}'."
        )
    if effective_num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if shard_index < 0 or shard_index >= effective_num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {effective_num_shards}); got {shard_index}."
        )
    if effective_split == "train" and effective_train_size > 0 and effective_num_shards > 1:
        if effective_train_size % effective_num_shards != 0:
            raise ValueError(
                f"--train-size ({effective_train_size}) is not evenly divisible by "
                f"--num-shards ({effective_num_shards}). Pick a divisible value to "
                "guarantee identical shard sizes across team members."
            )

    configured_writer_models = (
        [parse_model_spec(spec, alias_prefix="writer", index=i) for i, spec in enumerate(writer_models or [])]
        if writer_models
        else discover_default_writer_models()
    )

    configured_rubric_model = (
        parse_model_spec(rubric_model, alias_prefix="rubric")
        if rubric_model
        else discover_default_rubric_model()
    )
    configured_proposer_models = (
        [
            parse_model_spec(
                spec,
                alias_prefix="proposer",
                index=i,
            )
            for i, spec in enumerate(proposer_models or [])
        ]
        if proposer_models
        else discover_default_comparison_proposers()
    )
    configured_judge_model = (
        parse_model_spec(judge_model, alias_prefix="judge")
        if judge_model
        else (
            discover_default_comparison_judge_model()
            if comparison_mode
            else discover_default_judge_model()
        )
    )
    configured_bank_judge_model = (
        parse_model_spec(bank_judge_model, alias_prefix="bank-judge")
        if bank_judge_model
        else discover_default_bank_judge_model()
    )
    configured_downstream_judge_model = (
        parse_model_spec(downstream_judge_model, alias_prefix="downstream-judge")
        if downstream_judge_model
        else discover_default_downstream_note_judge_model()
    )
    configured_paper_pairwise_judge_model = (
        parse_model_spec(paper_pairwise_judge_model, alias_prefix="paper-pairwise-judge")
        if paper_pairwise_judge_model
        else discover_default_paper_pairwise_judge_model()
    )

    if not dry_run and not comparison_mode and configured_rubric_model is None:
        raise ValueError(
            "No rubric proposer model is configured. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / "
            "TOGETHER_API_KEY or pass --rubric-model."
        )
    if not dry_run and configured_judge_model is None:
        raise ValueError(
            "No rubric judge model is configured. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / "
            "TOGETHER_API_KEY or pass --judge-model."
        )
    if not dry_run and comparison_mode and len(configured_proposer_models) < 2:
        raise ValueError(
            "Comparison mode requires at least two proposer models. Pass --proposer-model twice "
            "or configure both OpenAI and Anthropic proposer defaults."
        )
    if not dry_run and comparison_mode and configured_bank_judge_model is None:
        raise ValueError(
            "Comparison mode requires a rubric-bank judge model. Set OPENAI_API_KEY or pass "
            "--bank-judge-model."
        )
    effective_paper_pairwise_label_mode = (
        paper_pairwise_label_mode or ("judge_proxy" if paper_mode else "reference_proxy")
    )
    if (
        not dry_run
        and paper_mode
        and effective_paper_pairwise_label_mode == "judge_proxy"
        and configured_paper_pairwise_judge_model is None
    ):
        raise ValueError(
            "Paper mode with judge_proxy labels requires a pairwise judge model. Set OPENAI_API_KEY or "
            "pass --paper-pairwise-judge-model."
        )

    config = PipelineConfig(
        dataset_path=Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH,
        output_dir=Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR,
        run_name=run_name or default_run_name(),
        comparison_mode=comparison_mode,
        paper_mode=paper_mode,
        no_cache=no_cache,
        rubrics_only=rubrics_only,
        skip_bank_utility=skip_bank_utility,
        start=start,
        limit=limit,
        source_filter=source_filter,
        resume=resume,
        dry_run=dry_run,
        max_workers=max(1, max_workers),
        sample_workers=max(1, sample_workers),
        target_candidate_count=target_candidate_count or DEFAULT_TARGET_CANDIDATES,
        split=effective_split,
        train_size=effective_train_size,
        val_size=effective_val_size,
        num_shards=effective_num_shards,
        shard_index=shard_index,
        preset=preset,
        reference_fields=effective_reference_fields,
        writer_models=configured_writer_models,
        paper_pairwise_label_mode=effective_paper_pairwise_label_mode,
        paper_response_only_judging=paper_mode,
        rubric_proposer=configured_rubric_model or (configured_proposer_models[0] if configured_proposer_models else None),
        proposer_models=configured_proposer_models,
        rubric_judge=configured_judge_model,
        rubric_bank_judge=configured_bank_judge_model,
        downstream_note_judge=configured_downstream_judge_model or configured_judge_model,
        paper_pairwise_judge=configured_paper_pairwise_judge_model,
        baseline_judge=configured_downstream_judge_model or configured_judge_model,
    )
    if decomposition_threshold is not None:
        config.decomposition_threshold = decomposition_threshold
    if max_initial_rubrics is not None:
        config.max_initial_rubrics = max_initial_rubrics
    if max_final_rubrics is not None:
        config.max_final_rubrics = max_final_rubrics
    if max_decomposition_depth is not None:
        config.max_decomposition_depth = max_decomposition_depth
    config.artifact_layout().ensure()
    return config
