"""Tests for the v2 multi-sample majority-vote rubric satisfaction (Tier A4)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List

from rubric_gen.config import PipelineConfig
from rubric_gen.rrd.engine import RRDEngine
from rubric_gen.storage import JsonlCache
from rubric_gen.types import (
    CandidateNote,
    ExampleRecord,
    LLMTextResponse,
    ModelSpec,
    RubricCriterion,
)


class _ScriptedRouter:
    """Returns a sequence of canned responses, one per ``generate`` call."""

    def __init__(self, raw_texts: List[str]) -> None:
        self.raw_texts = list(raw_texts)
        self.calls: List[dict] = []

    def generate(self, *args: Any, **kwargs: Any) -> LLMTextResponse:
        if not self.raw_texts:
            raise RuntimeError("ScriptedRouter exhausted")
        raw = self.raw_texts.pop(0)
        self.calls.append({"args": args, "kwargs": kwargs})
        return LLMTextResponse(
            text=raw,
            raw_text=raw,
            latency_s=0.0,
            model_alias="judge",
            provider="openai",
        )


def _yes() -> str:
    return json.dumps({"verdict": "YES", "reasoning": "."})


def _no() -> str:
    return json.dumps({"verdict": "NO", "reasoning": "."})


def _spec() -> ModelSpec:
    return ModelSpec(alias="judge", provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")


def _make_engine(samples: int, router: Any, *, satisfaction_temp: float = 0.4) -> RRDEngine:
    config = PipelineConfig(
        dataset_path=Path("/tmp/none"),
        output_dir=Path("/tmp/none"),
        run_name="multi_sample_unit",
        rubric_judge=_spec(),
        rubric_satisfaction_samples=samples,
        rubric_satisfaction_temperature=satisfaction_temp,
    )
    tmp = Path("ignored")  # unused -- caches are disabled
    return RRDEngine(
        config=config,
        router=router,
        proposal_cache=JsonlCache(tmp, enabled=False),
        filter_cache=JsonlCache(tmp, enabled=False),
        satisfaction_cache=JsonlCache(tmp, enabled=False),
    )


def _example() -> ExampleRecord:
    return ExampleRecord(
        example_id="ex_1",
        source="medical",
        source_id="ex_1",
        dataset_subset="",
        conversation="",
        task_prompt="What is X?",
    )


def _candidate() -> CandidateNote:
    return CandidateNote(
        candidate_id="ex_1__c1",
        example_id="ex_1",
        text="X is the answer.",
        source_label="generated_direct",
        quality_bucket="frontier_generated",
        origin_kind="generated",
    )


def _rubric() -> RubricCriterion:
    return RubricCriterion(
        rubric_id="ex_1__rubric_0_0_0",
        text="The note answers the question.",
        source_stage="initial",
        depth=0,
        round_index=0,
    )


class MultiSampleSatisfactionTests(unittest.TestCase):
    def test_samples_1_calls_router_once(self) -> None:
        router = _ScriptedRouter([_yes()])
        engine = _make_engine(samples=1, router=router)
        ev = engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
        self.assertTrue(ev.satisfied)
        self.assertEqual(len(router.calls), 1)
        # Sample 0 should be temperature 0.0
        self.assertAlmostEqual(router.calls[0]["kwargs"].get("temperature", 0.0), 0.0)

    def test_samples_3_majority_yes_wins(self) -> None:
        router = _ScriptedRouter([_yes(), _yes(), _no()])
        engine = _make_engine(samples=3, router=router)
        ev = engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
        self.assertTrue(ev.satisfied)
        self.assertEqual(len(router.calls), 3)
        self.assertEqual(ev.metadata.get("yes_votes"), 2)
        self.assertEqual(ev.metadata.get("no_votes"), 1)
        self.assertEqual(ev.metadata.get("samples"), 3)

    def test_samples_3_majority_no_wins(self) -> None:
        router = _ScriptedRouter([_no(), _yes(), _no()])
        engine = _make_engine(samples=3, router=router)
        ev = engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
        self.assertFalse(ev.satisfied)
        self.assertEqual(ev.metadata.get("yes_votes"), 1)
        self.assertEqual(ev.metadata.get("no_votes"), 2)

    def test_tie_votes_resolve_to_no(self) -> None:
        # 2-yes / 2-no with samples=4 -> per spec "yes > no" so NO wins.
        router = _ScriptedRouter([_yes(), _no(), _yes(), _no()])
        engine = _make_engine(samples=4, router=router)
        ev = engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
        self.assertFalse(ev.satisfied)
        self.assertEqual(ev.metadata.get("yes_votes"), 2)
        self.assertEqual(ev.metadata.get("no_votes"), 2)

    def test_sample_temperatures_match_config(self) -> None:
        router = _ScriptedRouter([_yes(), _yes(), _yes()])
        engine = _make_engine(samples=3, router=router, satisfaction_temp=0.5)
        engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
        # Sample 0 always 0.0; samples 1..N-1 use the configured temperature
        self.assertAlmostEqual(router.calls[0]["kwargs"]["temperature"], 0.0)
        self.assertAlmostEqual(router.calls[1]["kwargs"]["temperature"], 0.5)
        self.assertAlmostEqual(router.calls[2]["kwargs"]["temperature"], 0.5)

    def test_sample_history_recorded(self) -> None:
        router = _ScriptedRouter([_yes(), _no(), _yes()])
        engine = _make_engine(samples=3, router=router)
        ev = engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
        history = ev.metadata.get("sample_history") or []
        self.assertEqual(len(history), 3)
        self.assertEqual([h["satisfied"] for h in history], [True, False, True])

    def test_aggregated_cache_hit_on_second_call(self) -> None:
        with TemporaryDirectory() as td:
            cache = JsonlCache(Path(td) / "sat.jsonl", enabled=True)
            cache.load()
            router = _ScriptedRouter([_yes(), _yes(), _no()])
            config = PipelineConfig(
                dataset_path=Path("/tmp/none"),
                output_dir=Path("/tmp/none"),
                run_name="multi_sample_cache_unit",
                rubric_judge=_spec(),
                rubric_satisfaction_samples=3,
                rubric_satisfaction_temperature=0.4,
            )
            engine = RRDEngine(
                config=config,
                router=router,
                proposal_cache=JsonlCache(Path(td) / "p.jsonl", enabled=False),
                filter_cache=JsonlCache(Path(td) / "f.jsonl", enabled=False),
                satisfaction_cache=cache,
            )
            ev1 = engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
            self.assertEqual(len(router.calls), 3)
            # Second call hits cache: aggregated key resolves first.
            ev2 = engine.evaluate_rubric_on_candidate(_example(), _candidate(), _rubric())
            self.assertEqual(len(router.calls), 3)  # no new calls
            self.assertEqual(ev1.satisfied, ev2.satisfied)
            self.assertEqual(ev1.metadata.get("yes_votes"), ev2.metadata.get("yes_votes"))


if __name__ == "__main__":
    unittest.main()
