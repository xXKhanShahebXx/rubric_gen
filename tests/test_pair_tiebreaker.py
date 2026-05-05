from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from rubric_gen.evaluation.pair_tiebreaker import (
    ANTI_TIE_PROMPT_VERSION,
    TIEBREAKER_PROMPT_VERSION,
    TiebreakerOutcome,
    _cache_key,
    _normalize_confidence,
    _normalize_verdict,
    _parse,
    direct_judge_pair,
    direct_judge_pair_anti_tie,
    predict_via_score,
)
from rubric_gen.storage import JsonlCache
from rubric_gen.types import LLMTextResponse, ModelSpec


def _spec(provider: str = "openai", model: str = "gpt-4o") -> ModelSpec:
    return ModelSpec(alias="judge", provider=provider, model=model, api_key_env="OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Score-based prediction
# ---------------------------------------------------------------------------


class PredictViaScoreTests(unittest.TestCase):
    def test_a_wins(self) -> None:
        self.assertEqual(predict_via_score(0.9, 0.4), "a")

    def test_b_wins(self) -> None:
        self.assertEqual(predict_via_score(0.1, 0.7), "b")

    def test_equal_returns_tie(self) -> None:
        self.assertEqual(predict_via_score(0.5, 0.5), "tie")

    def test_none_returns_empty(self) -> None:
        self.assertEqual(predict_via_score(None, 0.5), "")
        self.assertEqual(predict_via_score(0.5, None), "")
        self.assertEqual(predict_via_score(None, None), "")


# ---------------------------------------------------------------------------
# Verdict + confidence parsing
# ---------------------------------------------------------------------------


class NormalizeTests(unittest.TestCase):
    def test_verdict_a_aliases(self) -> None:
        for raw in ("A", " a ", "RESPONSE_A", "reference_answer_a", "Answer_A"):
            self.assertEqual(_normalize_verdict(raw), "a")

    def test_verdict_b_aliases(self) -> None:
        for raw in ("B", "RESPONSE_B", "reference_answer_b"):
            self.assertEqual(_normalize_verdict(raw), "b")

    def test_verdict_tie_aliases(self) -> None:
        for raw in ("TIE", "tie", "EQUAL", "DRAW", "BOTH"):
            self.assertEqual(_normalize_verdict(raw), "tie")

    def test_verdict_unrecognized(self) -> None:
        for raw in ("", None, "MAYBE", "C"):
            self.assertEqual(_normalize_verdict(raw), "")

    def test_confidence_known(self) -> None:
        for raw in ("HIGH", "low", " medium "):
            self.assertEqual(_normalize_confidence(raw), raw.strip().lower())

    def test_confidence_default_medium(self) -> None:
        self.assertEqual(_normalize_confidence("foo"), "medium")
        self.assertEqual(_normalize_confidence(None), "medium")


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------


class ParseTests(unittest.TestCase):
    def test_parse_valid_json(self) -> None:
        v, c, r, err = _parse(json.dumps({"verdict": "A", "confidence": "high", "reasoning": "Because X"}))
        self.assertEqual(v, "a")
        self.assertEqual(c, "high")
        self.assertEqual(r, "Because X")
        self.assertEqual(err, "")

    def test_parse_no_json(self) -> None:
        v, c, r, err = _parse("not json at all")
        self.assertEqual(v, "")
        self.assertEqual(err, "no_json_object")

    def test_parse_invalid_verdict(self) -> None:
        v, c, r, err = _parse(json.dumps({"verdict": "MAYBE", "confidence": "high"}))
        self.assertEqual(v, "")
        self.assertEqual(err, "unrecognized_verdict")

    def test_parse_alternative_key_decision(self) -> None:
        v, c, r, err = _parse(json.dumps({"decision": "B", "confidence": "low", "rationale": "..."}))
        self.assertEqual(v, "b")
        self.assertEqual(c, "low")


# ---------------------------------------------------------------------------
# Cache key stability
# ---------------------------------------------------------------------------


class CacheKeyTests(unittest.TestCase):
    def test_same_inputs_same_key(self) -> None:
        k1 = _cache_key(judge_model=_spec(), question="Q?", response_a="A", response_b="B", temperature=0.0)
        k2 = _cache_key(judge_model=_spec(), question="Q?", response_a="A", response_b="B", temperature=0.0)
        self.assertEqual(k1, k2)

    def test_swapping_responses_changes_key(self) -> None:
        k1 = _cache_key(judge_model=_spec(), question="Q?", response_a="A", response_b="B", temperature=0.0)
        k2 = _cache_key(judge_model=_spec(), question="Q?", response_a="B", response_b="A", temperature=0.0)
        self.assertNotEqual(k1, k2)

    def test_different_model_changes_key(self) -> None:
        k1 = _cache_key(judge_model=_spec(model="gpt-4o"), question="Q", response_a="A", response_b="B", temperature=0.0)
        k2 = _cache_key(judge_model=_spec(model="gpt-4o-mini"), question="Q", response_a="A", response_b="B", temperature=0.0)
        self.assertNotEqual(k1, k2)


# ---------------------------------------------------------------------------
# direct_judge_pair (with fake router + cache)
# ---------------------------------------------------------------------------


class _FakeRouter:
    def __init__(self, raw: str = json.dumps({"verdict": "A", "confidence": "high", "reasoning": "."})):
        self.raw = raw
        self.calls = 0

    def generate(self, *args: Any, **kwargs: Any) -> LLMTextResponse:
        self.calls += 1
        return LLMTextResponse(
            text=self.raw,
            raw_text=self.raw,
            latency_s=0.0,
            model_alias="judge",
            provider="openai",
        )


class _RaisingRouter:
    def generate(self, *args: Any, **kwargs: Any) -> LLMTextResponse:
        raise RuntimeError("boom")


class DirectJudgeTests(unittest.TestCase):
    def test_dry_run_returns_empty_verdict(self) -> None:
        with TemporaryDirectory() as td:
            cache = JsonlCache(Path(td) / "c.jsonl", enabled=True)
            cache.load()
            out = direct_judge_pair(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=None, cache=cache,
                temperature=0.0, dry_run=True,
            )
            self.assertEqual(out.verdict, "")
            self.assertEqual(out.parse_error, "dry_run")
            self.assertFalse(out.cache_hit)

    def test_writes_cache_then_hits_on_second_call(self) -> None:
        with TemporaryDirectory() as td:
            cache = JsonlCache(Path(td) / "c.jsonl", enabled=True)
            cache.load()
            router = _FakeRouter()
            out1 = direct_judge_pair(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=router, cache=cache, temperature=0.0,
            )
            self.assertEqual(out1.verdict, "a")
            self.assertFalse(out1.cache_hit)
            self.assertEqual(router.calls, 1)
            out2 = direct_judge_pair(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=router, cache=cache, temperature=0.0,
            )
            self.assertEqual(out2.verdict, "a")
            self.assertTrue(out2.cache_hit)
            self.assertEqual(router.calls, 1)  # no second LLM call

    def test_router_error_recorded_not_raised(self) -> None:
        with TemporaryDirectory() as td:
            cache = JsonlCache(Path(td) / "c.jsonl", enabled=True)
            cache.load()
            out = direct_judge_pair(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=_RaisingRouter(), cache=cache,
                temperature=0.0,
            )
            self.assertEqual(out.verdict, "")
            self.assertEqual(out.parse_error, "router_error")
            self.assertIn("RuntimeError", out.router_error)

    def test_unparseable_response_records_parse_error(self) -> None:
        with TemporaryDirectory() as td:
            cache = JsonlCache(Path(td) / "c.jsonl", enabled=True)
            cache.load()
            router = _FakeRouter(raw="totally not json")
            out = direct_judge_pair(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=router, cache=cache, temperature=0.0,
            )
            self.assertEqual(out.verdict, "")
            self.assertEqual(out.parse_error, "no_json_object")


# ---------------------------------------------------------------------------
# rescore_with_tiebreaker._resolve_one
# ---------------------------------------------------------------------------


def _load_rescore_module():
    import importlib.util
    import sys as _sys
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "rescore_with_tiebreaker.py"
    spec = importlib.util.spec_from_file_location("rescore_with_tiebreaker", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass introspection works (the dataclass
    # decorator looks the class's module up via sys.modules).
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_rescore = _load_rescore_module()


def _make_pair_artifact(*, source_id: str, gold: str, primary_a: float, primary_b: float, secondary_a: float, secondary_b: float) -> Dict[str, Any]:
    cand_a = f"{source_id}__pair_a"
    cand_b = f"{source_id}__pair_b"
    return {
        "example": {
            "example_id": f"med__{source_id}",
            "task_prompt": "What is the diagnosis?",
            "pair_response_a": "A explanation...",
            "pair_response_b": "B explanation...",
            "pair_correct_label": gold,
        },
        "candidates": [
            {"candidate_id": cand_a, "source_label": "pair_response_a", "text": "A text"},
            {"candidate_id": cand_b, "source_label": "pair_response_b", "text": "B text"},
        ],
        "methods": {
            "rrd_whitened_uniform": {
                "ranking": [
                    {"candidate_id": cand_a, "rank": 1 if primary_a >= primary_b else 2, "score": primary_a},
                    {"candidate_id": cand_b, "rank": 2 if primary_a >= primary_b else 1, "score": primary_b},
                ],
            },
            "rrd_uniform": {
                "ranking": [
                    {"candidate_id": cand_a, "rank": 1 if secondary_a >= secondary_b else 2, "score": secondary_a},
                    {"candidate_id": cand_b, "rank": 2 if secondary_a >= secondary_b else 1, "score": secondary_b},
                ],
            },
        },
    }


class ResolveOneCascadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.cache = JsonlCache(Path(self._td.name) / "c.jsonl", enabled=True)
        self.cache.load()

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_primary_strict_winner_no_cascade(self) -> None:
        ex = _make_pair_artifact(source_id="x1", gold="a", primary_a=0.7, primary_b=0.4, secondary_a=0.5, secondary_b=0.5)
        d = _rescore._resolve_one(ex=ex, strategy="uniform_then_judge", judge_model=_spec(), router=None, cache=self.cache, judge_temperature=0.0, dry_run=True)
        self.assertEqual(d.pred, "a")
        self.assertEqual(d.decision_policy, "primary")

    def test_primary_tie_then_uniform_breaks(self) -> None:
        ex = _make_pair_artifact(source_id="x2", gold="b", primary_a=0.5, primary_b=0.5, secondary_a=0.3, secondary_b=0.6)
        d = _rescore._resolve_one(ex=ex, strategy="uniform_then_judge", judge_model=_spec(), router=None, cache=self.cache, judge_temperature=0.0, dry_run=True)
        self.assertEqual(d.pred, "b")
        self.assertEqual(d.decision_policy, "uniform")
        self.assertEqual(d.primary_pred, "tie")
        self.assertEqual(d.secondary_pred, "b")

    def test_primary_and_secondary_tie_then_judge(self) -> None:
        ex = _make_pair_artifact(source_id="x3", gold="a", primary_a=0.5, primary_b=0.5, secondary_a=0.4, secondary_b=0.4)
        router = _FakeRouter(raw=json.dumps({"verdict": "A", "confidence": "high", "reasoning": "."}))
        d = _rescore._resolve_one(ex=ex, strategy="uniform_then_judge", judge_model=_spec(), router=router, cache=self.cache, judge_temperature=0.0, dry_run=False)
        self.assertEqual(d.pred, "a")
        self.assertEqual(d.decision_policy, "judge")
        self.assertIsNotNone(d.judge_outcome)

    def test_strategy_none_keeps_tie(self) -> None:
        ex = _make_pair_artifact(source_id="x4", gold="a", primary_a=0.5, primary_b=0.5, secondary_a=0.3, secondary_b=0.6)
        d = _rescore._resolve_one(ex=ex, strategy="none", judge_model=_spec(), router=None, cache=self.cache, judge_temperature=0.0, dry_run=True)
        self.assertEqual(d.pred, "tie")
        self.assertEqual(d.decision_policy, "primary")

    def test_strategy_uniform_only_no_judge(self) -> None:
        ex = _make_pair_artifact(source_id="x5", gold="a", primary_a=0.5, primary_b=0.5, secondary_a=0.3, secondary_b=0.3)
        d = _rescore._resolve_one(ex=ex, strategy="uniform", judge_model=_spec(), router=_RaisingRouter(), cache=self.cache, judge_temperature=0.0, dry_run=True)
        self.assertEqual(d.pred, "tie")  # secondary tied too, no judge
        self.assertEqual(d.decision_policy, "uniform_tied")

    def test_format_then_judge_resolves_via_format_prior(self) -> None:
        # Tier C1: format prior should pick a winner without an LLM call when
        # the question explicitly asks for terse output and one side is much
        # shorter than the other.
        ex = _make_pair_artifact(source_id="x6", gold="a", primary_a=0.5, primary_b=0.5, secondary_a=0.5, secondary_b=0.5)
        ex["example"]["task_prompt"] = "Answer in one sentence."
        ex["example"]["pair_response_a"] = "D."
        ex["example"]["pair_response_b"] = "The correct answer is D because " + ("of the mechanism " * 60)
        # Use a router that would raise if called -- we expect the format prior
        # to short-circuit before the LLM judge.
        d = _rescore._resolve_one(
            ex=ex, strategy="format_then_judge", judge_model=_spec(),
            router=_RaisingRouter(), cache=self.cache, judge_temperature=0.0, dry_run=False,
        )
        self.assertEqual(d.pred, "a")
        self.assertEqual(d.decision_policy, "format")
        self.assertIsNotNone(d.format_outcome)

    def test_format_then_uniform_then_judge_falls_through_when_no_format_signal(self) -> None:
        # Q has no format cue -> format prior returns 'tie' -> falls through
        # to uniform (also tied) -> falls through to judge.
        ex = _make_pair_artifact(source_id="x7", gold="b", primary_a=0.5, primary_b=0.5, secondary_a=0.4, secondary_b=0.4)
        ex["example"]["task_prompt"] = "What is the diagnosis?"
        ex["example"]["pair_response_a"] = "MI."
        ex["example"]["pair_response_b"] = "MI."
        router = _FakeRouter(raw=json.dumps({"verdict": "B", "confidence": "high", "reasoning": "."}))
        d = _rescore._resolve_one(
            ex=ex, strategy="format_then_uniform_then_judge", judge_model=_spec(),
            router=router, cache=self.cache, judge_temperature=0.0, dry_run=False,
        )
        self.assertEqual(d.pred, "b")
        self.assertEqual(d.decision_policy, "judge")

    def test_format_then_uniform_then_anti_tie_uses_anti_tie_prompt(self) -> None:
        ex = _make_pair_artifact(source_id="x8", gold="a", primary_a=0.5, primary_b=0.5, secondary_a=0.4, secondary_b=0.4)
        ex["example"]["task_prompt"] = "What is the diagnosis?"
        router = _FakeRouter(raw=json.dumps({"verdict": "A", "confidence": "medium", "reasoning": "."}))
        d = _rescore._resolve_one(
            ex=ex, strategy="format_then_uniform_then_anti_tie", judge_model=_spec(),
            router=router, cache=self.cache, judge_temperature=0.0, dry_run=False,
        )
        self.assertEqual(d.pred, "a")
        self.assertEqual(d.decision_policy, "judge_anti_tie")


# ---------------------------------------------------------------------------
# Anti-tie variant + cache namespace
# ---------------------------------------------------------------------------


class AntiTieJudgeTests(unittest.TestCase):
    def test_anti_tie_uses_independent_cache_key(self) -> None:
        with TemporaryDirectory() as td:
            cache = JsonlCache(Path(td) / "c.jsonl", enabled=True)
            cache.load()
            router_v1 = _FakeRouter(raw=json.dumps({"verdict": "A", "confidence": "high", "reasoning": "v1."}))
            router_anti = _FakeRouter(raw=json.dumps({"verdict": "B", "confidence": "high", "reasoning": "anti."}))
            v1 = direct_judge_pair(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=router_v1, cache=cache, temperature=0.0,
            )
            anti = direct_judge_pair_anti_tie(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=router_anti, cache=cache, temperature=0.0,
            )
            # Independent prompts -> independent cache entries -> both LLMs called.
            self.assertEqual(v1.verdict, "a")
            self.assertEqual(anti.verdict, "b")
            self.assertEqual(router_v1.calls, 1)
            self.assertEqual(router_anti.calls, 1)

    def test_anti_tie_caches_on_second_call(self) -> None:
        with TemporaryDirectory() as td:
            cache = JsonlCache(Path(td) / "c.jsonl", enabled=True)
            cache.load()
            router = _FakeRouter(raw=json.dumps({"verdict": "B", "confidence": "high", "reasoning": "."}))
            out1 = direct_judge_pair_anti_tie(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=router, cache=cache, temperature=0.0,
            )
            out2 = direct_judge_pair_anti_tie(
                question="Q", response_a="A", response_b="B",
                judge_model=_spec(), router=router, cache=cache, temperature=0.0,
            )
            self.assertEqual(out1.verdict, "b")
            self.assertEqual(out2.verdict, "b")
            self.assertTrue(out2.cache_hit)
            self.assertEqual(router.calls, 1)

    def test_prompt_versions_distinct(self) -> None:
        self.assertNotEqual(TIEBREAKER_PROMPT_VERSION, ANTI_TIE_PROMPT_VERSION)


class SummarizeTests(unittest.TestCase):
    def test_aggregates_correct_counts(self) -> None:
        rows = [
            _rescore.RowDecision(sid="a", gold="a", pred="a", primary_pred="a", decision_policy="primary"),
            _rescore.RowDecision(sid="b", gold="b", pred="b", primary_pred="b", decision_policy="primary"),
            _rescore.RowDecision(sid="c", gold="a", pred="b", primary_pred="b", decision_policy="primary"),
            _rescore.RowDecision(sid="d", gold="b", pred="tie", primary_pred="tie", decision_policy="judge_tied"),
            _rescore.RowDecision(sid="e", gold="", pred="", primary_pred="", decision_policy="unevaluable"),
        ]
        s = _rescore._summarize(rows)
        self.assertEqual(s.n, 4)
        self.assertEqual(s.correct, 2)
        self.assertEqual(s.wrong, 1)
        self.assertEqual(s.tie, 1)
        self.assertEqual(s.unevaluable, 1)
        self.assertEqual(s.by_gold["a"]["n"], 2)
        self.assertEqual(s.by_gold["a"]["correct"], 1)
        self.assertEqual(s.by_gold["b"]["n"], 2)
        self.assertEqual(s.by_gold["b"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
