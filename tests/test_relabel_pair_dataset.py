from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from rubric_gen.llm_client import _anthropic_rejects_temperature
from rubric_gen.types import LLMTextResponse, ModelSpec


def _load_relabel_module():
    """Import scripts/relabel_pair_dataset.py as an ad-hoc module for testing."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "relabel_pair_dataset.py"
    spec = importlib.util.spec_from_file_location("relabel_pair_dataset", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_relabel = _load_relabel_module()


def _spec(model: str = "claude-opus-4-7") -> ModelSpec:
    return ModelSpec(
        alias="test_judge",
        provider="anthropic",
        model=model,
        api_key_env="ANTHROPIC_API_KEY",
    )


# ---------------------------------------------------------------------------
# Cache key stability
# ---------------------------------------------------------------------------


class CacheKeyTests(unittest.TestCase):
    def test_same_inputs_same_key(self) -> None:
        k1 = _relabel._build_cache_key(
            judge_model=_spec(),
            question="What causes diabetes?",
            response_a="Insulin resistance.",
            response_b="Genetic factors.",
            temperature=0.0,
        )
        k2 = _relabel._build_cache_key(
            judge_model=_spec(),
            question="What causes diabetes?",
            response_a="Insulin resistance.",
            response_b="Genetic factors.",
            temperature=0.0,
        )
        self.assertEqual(k1, k2)

    def test_different_model_different_key(self) -> None:
        k1 = _relabel._build_cache_key(
            judge_model=_spec("claude-opus-4-6"),
            question="Q",
            response_a="A",
            response_b="B",
            temperature=0.0,
        )
        k2 = _relabel._build_cache_key(
            judge_model=_spec("claude-opus-4-7"),
            question="Q",
            response_a="A",
            response_b="B",
            temperature=0.0,
        )
        self.assertNotEqual(k1, k2)

    def test_temperature_change_busts_key(self) -> None:
        k1 = _relabel._build_cache_key(
            judge_model=_spec(), question="Q", response_a="A", response_b="B", temperature=0.0
        )
        k2 = _relabel._build_cache_key(
            judge_model=_spec(), question="Q", response_a="A", response_b="B", temperature=0.5
        )
        self.assertNotEqual(k1, k2)

    def test_response_swap_busts_key(self) -> None:
        # Order matters: swapping A and B is a different judge call.
        k1 = _relabel._build_cache_key(
            judge_model=_spec(), question="Q", response_a="X", response_b="Y", temperature=0.0
        )
        k2 = _relabel._build_cache_key(
            judge_model=_spec(), question="Q", response_a="Y", response_b="X", temperature=0.0
        )
        self.assertNotEqual(k1, k2)


# ---------------------------------------------------------------------------
# Verdict / confidence normalisation + parse
# ---------------------------------------------------------------------------


class VerdictParsingTests(unittest.TestCase):
    def test_normalize_verdict_aliases(self) -> None:
        self.assertEqual(_relabel._normalize_verdict("A"), "A")
        self.assertEqual(_relabel._normalize_verdict("a"), "A")
        self.assertEqual(_relabel._normalize_verdict("reference_answer_a"), "A")
        self.assertEqual(_relabel._normalize_verdict("response_a"), "A")
        self.assertEqual(_relabel._normalize_verdict("B"), "B")
        self.assertEqual(_relabel._normalize_verdict("reference_answer_b"), "B")
        self.assertEqual(_relabel._normalize_verdict("TIE"), "TIE")
        self.assertEqual(_relabel._normalize_verdict("tie"), "TIE")
        self.assertEqual(_relabel._normalize_verdict("draw"), "TIE")
        self.assertEqual(_relabel._normalize_verdict(None), "")
        self.assertEqual(_relabel._normalize_verdict("maybe"), "")

    def test_normalize_confidence_defaults(self) -> None:
        self.assertEqual(_relabel._normalize_confidence("HIGH"), "high")
        self.assertEqual(_relabel._normalize_confidence("low"), "low")
        self.assertEqual(_relabel._normalize_confidence("garbage"), "medium")
        self.assertEqual(_relabel._normalize_confidence(None), "medium")

    def test_parse_judge_response_valid_json(self) -> None:
        verdict, conf, reasoning, err = _relabel._parse_judge_response(
            '{"verdict": "B", "confidence": "high", "reasoning": "B cites the right guideline."}'
        )
        self.assertEqual(verdict, "B")
        self.assertEqual(conf, "high")
        self.assertEqual(reasoning, "B cites the right guideline.")
        self.assertEqual(err, "")

    def test_parse_judge_response_no_json(self) -> None:
        verdict, conf, reasoning, err = _relabel._parse_judge_response("Just some prose.")
        self.assertEqual(verdict, "")
        self.assertEqual(err, "no_json_object")

    def test_parse_judge_response_unknown_verdict(self) -> None:
        verdict, conf, reasoning, err = _relabel._parse_judge_response(
            '{"verdict": "MAYBE_LATER", "confidence": "low"}'
        )
        self.assertEqual(verdict, "")
        self.assertEqual(err, "unrecognized_verdict")

    def test_parse_judge_response_extra_prose_around_json(self) -> None:
        # extract_json_object handles prefix/suffix prose around the JSON object.
        verdict, conf, reasoning, err = _relabel._parse_judge_response(
            'Here is my verdict: {"verdict": "A", "confidence": "medium", "reasoning": "ok"} done.'
        )
        self.assertEqual(verdict, "A")
        self.assertEqual(conf, "medium")


# ---------------------------------------------------------------------------
# _verdict_to_correct_answer
# ---------------------------------------------------------------------------


class VerdictToCorrectAnswerTests(unittest.TestCase):
    def test_a_b_map_back(self) -> None:
        self.assertEqual(
            _relabel._verdict_to_correct_answer("A", "reference_answer_a"),
            "reference_answer_a",
        )
        self.assertEqual(
            _relabel._verdict_to_correct_answer("B", "reference_answer_a"),
            "reference_answer_b",
        )

    def test_tie_keeps_original(self) -> None:
        self.assertEqual(
            _relabel._verdict_to_correct_answer("TIE", "reference_answer_a"),
            "reference_answer_a",
        )

    def test_empty_verdict_keeps_original(self) -> None:
        self.assertEqual(
            _relabel._verdict_to_correct_answer("", "reference_answer_a"),
            "reference_answer_a",
        )


# ---------------------------------------------------------------------------
# _ids_from_run_dir
# ---------------------------------------------------------------------------


class IdsFromRunDirTests(unittest.TestCase):
    def test_extracts_id_suffix_after_double_underscore(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "examples").mkdir(parents=True)
            for name in [
                "general_instruction_following__0006000-abc.json",
                "agentic_workflows__0006001-def.json",
                "clinical_decision_support__0006002-ghi.json",
                # One file without the __ separator -> ignored.
                "weird_file.json",
            ]:
                (run_dir / "examples" / name).write_text("{}", encoding="utf-8")

            ids = _relabel._ids_from_run_dir(run_dir)

        self.assertEqual(
            ids,
            {"0006000-abc", "0006001-def", "0006002-ghi"},
        )

    def test_missing_examples_dir_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                _relabel._ids_from_run_dir(Path(tmp) / "missing_run")


# ---------------------------------------------------------------------------
# _judge_pair caching + outcomes
# ---------------------------------------------------------------------------


class _ScriptedRouter:
    def __init__(self, response_text: str = "", *, raise_exc: Optional[Exception] = None):
        self.response_text = response_text
        self.raise_exc = raise_exc
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        spec: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMTextResponse:
        self.calls.append({"system": system_prompt, "user": user_prompt, "temperature": temperature})
        if self.raise_exc is not None:
            raise self.raise_exc
        return LLMTextResponse(
            text=self.response_text,
            raw_text=self.response_text,
            latency_s=0.0,
            model_alias=spec.alias,
            provider=spec.provider,
            metadata={"model": spec.model},
        )


class JudgePairCacheTests(unittest.TestCase):
    def test_cache_hit_avoids_router_call(self) -> None:
        from rubric_gen.storage import JsonlCache

        router = _ScriptedRouter(
            response_text='{"verdict": "B", "confidence": "high", "reasoning": "b is better"}'
        )
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "rl_pair_cache.jsonl", enabled=True)

            # First call -> router fires once and writes to cache.
            outcome1 = _relabel._judge_pair(
                judge_model=_spec(),
                router=router,
                cache=cache,
                question="Q?",
                response_a="A txt",
                response_b="B txt",
                temperature=0.0,
                dry_run=False,
            )
            self.assertEqual(outcome1.verdict, "B")
            self.assertEqual(len(router.calls), 1)
            self.assertFalse(outcome1.cache_hit)

            # Second call with a fresh router that would explode if used.
            second_router = _ScriptedRouter(raise_exc=RuntimeError("should not be called"))
            outcome2 = _relabel._judge_pair(
                judge_model=_spec(),
                router=second_router,
                cache=cache,
                question="Q?",
                response_a="A txt",
                response_b="B txt",
                temperature=0.0,
                dry_run=False,
            )
            self.assertEqual(outcome2.verdict, "B")
            self.assertEqual(second_router.calls, [])
            self.assertTrue(outcome2.cache_hit)

    def test_router_error_returns_outcome_without_caching(self) -> None:
        from rubric_gen.storage import JsonlCache

        router = _ScriptedRouter(raise_exc=RuntimeError("network down"))
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "rl_pair_cache.jsonl", enabled=True)
            outcome = _relabel._judge_pair(
                judge_model=_spec(),
                router=router,
                cache=cache,
                question="Q?",
                response_a="A",
                response_b="B",
                temperature=0.0,
                dry_run=False,
            )
        self.assertEqual(outcome.verdict, "")
        self.assertEqual(outcome.parse_error, "router_error")
        self.assertIn("network down", outcome.router_error)
        # Cache file might not exist yet -- verify by re-running and confirming
        # the router would be called again.
        router2 = _ScriptedRouter(raise_exc=RuntimeError("network still down"))
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "rl_pair_cache.jsonl", enabled=True)
            _relabel._judge_pair(
                judge_model=_spec(),
                router=router2,
                cache=cache,
                question="Q?",
                response_a="A",
                response_b="B",
                temperature=0.0,
                dry_run=False,
            )
        self.assertEqual(len(router2.calls), 1)

    def test_dry_run_skips_call(self) -> None:
        from rubric_gen.storage import JsonlCache

        router = _ScriptedRouter(raise_exc=RuntimeError("should not be called"))
        with TemporaryDirectory() as tmp:
            cache = JsonlCache(Path(tmp) / "rl_pair_cache.jsonl", enabled=True)
            outcome = _relabel._judge_pair(
                judge_model=_spec(),
                router=router,
                cache=cache,
                question="Q?",
                response_a="A",
                response_b="B",
                temperature=0.0,
                dry_run=True,
            )
        self.assertEqual(outcome.verdict, "")
        self.assertEqual(outcome.parse_error, "dry_run")
        self.assertEqual(router.calls, [])


# ---------------------------------------------------------------------------
# _relabel_row output schema
# ---------------------------------------------------------------------------


class RelabelRowSchemaTests(unittest.TestCase):
    def test_relabel_row_preserves_input_and_adds_judge_fields(self) -> None:
        row = {
            "id": "0006000-abc",
            "question": "Q?",
            "reference_answer_a": "A",
            "reference_answer_b": "B",
            "correct_answer": "reference_answer_a",
            "extra_field": "passthrough",
        }
        outcome = _relabel.JudgeOutcome(
            verdict="B",
            confidence="high",
            reasoning="B is medically more accurate.",
            raw_response='{"verdict":"B"}',
            cache_hit=False,
        )
        out = _relabel._relabel_row(row, outcome, _spec("claude-opus-4-7"))

        # Original fields preserved
        self.assertEqual(out["id"], "0006000-abc")
        self.assertEqual(out["question"], "Q?")
        self.assertEqual(out["reference_answer_a"], "A")
        self.assertEqual(out["reference_answer_b"], "B")
        self.assertEqual(out["extra_field"], "passthrough")
        # Original label preserved under new key
        self.assertEqual(out["original_correct_answer"], "reference_answer_a")
        # Correct answer now reflects judge verdict
        self.assertEqual(out["correct_answer"], "reference_answer_b")
        # Judge fields surfaced
        self.assertEqual(out["judge_verdict"], "B")
        self.assertEqual(out["judge_confidence"], "high")
        self.assertEqual(out["judge_reasoning"], "B is medically more accurate.")
        self.assertEqual(out["judge_model"], "anthropic:claude-opus-4-7")
        self.assertFalse(out["judge_tie"])
        self.assertNotIn("judge_parse_error", out)
        self.assertNotIn("judge_router_error", out)

    def test_relabel_row_tie_keeps_original_label_and_flags_tie(self) -> None:
        row = {
            "id": "x",
            "correct_answer": "reference_answer_a",
            "question": "?",
            "reference_answer_a": "A",
            "reference_answer_b": "B",
        }
        outcome = _relabel.JudgeOutcome(
            verdict="TIE",
            confidence="medium",
            reasoning="Equally good.",
            raw_response="",
            cache_hit=False,
        )
        out = _relabel._relabel_row(row, outcome, _spec())
        self.assertEqual(out["correct_answer"], "reference_answer_a")
        self.assertTrue(out["judge_tie"])

    def test_relabel_row_includes_error_fields_on_failure(self) -> None:
        row = {
            "id": "x",
            "correct_answer": "reference_answer_a",
            "question": "?",
            "reference_answer_a": "A",
            "reference_answer_b": "B",
        }
        outcome = _relabel.JudgeOutcome(
            verdict="",
            confidence="",
            reasoning="",
            raw_response="",
            cache_hit=False,
            parse_error="no_json_object",
            router_error="ConnectError: timeout",
        )
        out = _relabel._relabel_row(row, outcome, _spec())
        self.assertEqual(out["judge_parse_error"], "no_json_object")
        self.assertEqual(out["judge_router_error"], "ConnectError: timeout")
        # Verdict empty -> correct_answer falls back to original.
        self.assertEqual(out["correct_answer"], "reference_answer_a")


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------


class SummarizeTests(unittest.TestCase):
    def _result(self, verdict: str, original: str = "reference_answer_a") -> Any:
        outcome = _relabel.JudgeOutcome(
            verdict=verdict,
            confidence="high",
            reasoning="",
            raw_response="",
            cache_hit=False,
        )
        row = {"correct_answer": original}
        relabeled = _relabel._relabel_row(row, outcome, _spec())
        return _relabel.RowResult(index=0, row=row, relabeled_row=relabeled, outcome=outcome)

    def test_summarize_counts_and_rates(self) -> None:
        results = [
            self._result("A"),  # agree
            self._result("A"),  # agree
            self._result("B"),  # flip
            self._result("B"),  # flip
            self._result("TIE"),
            self._result(""),  # parse error
        ]
        summary = _relabel._summarize(results)
        self.assertEqual(summary["rows_processed"], 6)
        self.assertEqual(summary["verdict_counts"]["A"], 2)
        self.assertEqual(summary["verdict_counts"]["B"], 2)
        self.assertEqual(summary["verdict_counts"]["TIE"], 1)
        self.assertEqual(summary["verdict_counts"][""], 1)
        self.assertEqual(summary["label_flips"], 2)
        # Decided = 6 - (TIE + empty) = 4. label_flip_rate = 2/4.
        self.assertAlmostEqual(summary["label_flip_rate_among_decided"], 0.5)
        # Tie rate over all rows.
        self.assertAlmostEqual(summary["tie_rate"], 1 / 6)


class AnthropicTemperatureGuardTests(unittest.TestCase):
    """LLMRouter must skip the `temperature` parameter for Opus 4.7+ thinking models."""

    def test_opus_4_7_rejects_temperature(self) -> None:
        self.assertTrue(_anthropic_rejects_temperature("claude-opus-4-7"))
        self.assertTrue(_anthropic_rejects_temperature("Claude-Opus-4-7"))
        # Future placeholders:
        self.assertTrue(_anthropic_rejects_temperature("claude-opus-4-8"))
        self.assertTrue(_anthropic_rejects_temperature("claude-opus-5"))

    def test_opus_4_1_and_4_6_still_accept_temperature(self) -> None:
        self.assertFalse(_anthropic_rejects_temperature("claude-opus-4-1-20250805"))
        self.assertFalse(_anthropic_rejects_temperature("claude-opus-4-6"))
        self.assertFalse(_anthropic_rejects_temperature("claude-sonnet-4-5-20250929"))
        self.assertFalse(_anthropic_rejects_temperature(""))


if __name__ == "__main__":
    unittest.main()
