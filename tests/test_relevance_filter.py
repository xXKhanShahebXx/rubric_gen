from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from rubric_gen.compiled.relevance_filter import (
    AGGRESSIVE,
    CONSERVATIVE,
    DEFAULT_FILTER_MODEL,
    RELEVANCE_FILTER_PROMPT_VERSION,
    RelevanceFilterConfig,
    VERDICT_APPLICABLE,
    VERDICT_IRRELEVANT,
    VERDICT_UNCERTAIN,
    build_default_config,
    filter_relevant_criteria,
    parse_strictness,
)
from rubric_gen.compiled.rubric_library import RubricLibraryCriterion
from rubric_gen.storage import JsonlCache
from rubric_gen.types import LLMTextResponse


class _FakeRouter:
    """Stand-in for :class:`LLMRouter` that hands back queued responses verbatim."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.raise_on_call = False

    def generate(
        self,
        spec,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMTextResponse:
        self.calls.append(
            {
                "model": f"{spec.provider}:{spec.model}",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
            }
        )
        if self.raise_on_call:
            raise RuntimeError("simulated provider outage")
        if not self._responses:
            raise RuntimeError("FakeRouter has no remaining responses")
        text = self._responses.pop(0)
        return LLMTextResponse(
            text=text,
            raw_text=text,
            latency_s=0.0,
            model_alias=spec.alias,
            provider=spec.provider,
            metadata={"model": spec.model},
        )


def _criterion(idx: int, family: str = "generic") -> RubricLibraryCriterion:
    return RubricLibraryCriterion(
        criterion_id=f"lib_test_{idx}",
        dimension=f"dim_{idx}",
        label=f"Criterion {idx}",
        requirement=f"Requirement number {idx}",
        severity_tier="medium",
        applicable_families=(family,),
        source_tag="test",
    )


def _enabled_config(strictness: str = CONSERVATIVE) -> RelevanceFilterConfig:
    return RelevanceFilterConfig(
        enabled=True,
        model_spec=DEFAULT_FILTER_MODEL,
        strictness=strictness,
        max_criteria_per_call=16,
    )


class RelevanceFilterDisabledTests(unittest.TestCase):
    def test_disabled_short_circuits_with_no_router_calls(self) -> None:
        config = RelevanceFilterConfig(enabled=False)
        criteria = [_criterion(i) for i in range(3)]
        router = _FakeRouter(responses=[])

        kept, debug = filter_relevant_criteria(
            criteria,
            prompt_text="anything",
            candidate_texts=["response"],
            config=config,
            router=router,
        )

        self.assertEqual(len(kept), 3)
        self.assertEqual(kept, criteria)
        self.assertEqual(router.calls, [])
        self.assertFalse(debug["enabled"])
        self.assertEqual(debug["dropped_count"], 0)
        self.assertEqual(debug["kept_count"], 3)

    def test_empty_input_is_a_no_op_even_when_enabled(self) -> None:
        config = _enabled_config()
        router = _FakeRouter(responses=[])

        kept, debug = filter_relevant_criteria(
            [],
            prompt_text="prompt",
            candidate_texts=["response"],
            config=config,
            router=router,
        )

        self.assertEqual(kept, [])
        self.assertEqual(router.calls, [])
        self.assertEqual(debug["input_count"], 0)
        self.assertEqual(debug["kept_count"], 0)
        self.assertEqual(debug["dropped_count"], 0)


class RelevanceFilterGateTests(unittest.TestCase):
    def test_conservative_keeps_applicable_and_uncertain_drops_irrelevant(self) -> None:
        criteria = [_criterion(i) for i in range(3)]
        router = _FakeRouter(
            responses=[
                """{
                    "0": {"verdict": "APPLICABLE", "reason": "on-topic"},
                    "1": {"verdict": "IRRELEVANT", "reason": "different cancer"},
                    "2": {"verdict": "UNCERTAIN", "reason": "borderline"}
                }"""
            ]
        )

        kept, debug = filter_relevant_criteria(
            criteria,
            prompt_text="prompt about disease X",
            candidate_texts=["response"],
            config=_enabled_config(CONSERVATIVE),
            router=router,
        )

        self.assertEqual([c.criterion_id for c in kept], ["lib_test_0", "lib_test_2"])
        self.assertEqual(debug["dropped_count"], 1)
        self.assertEqual(debug["verdict_counts"][VERDICT_APPLICABLE], 1)
        self.assertEqual(debug["verdict_counts"][VERDICT_IRRELEVANT], 1)
        self.assertEqual(debug["verdict_counts"][VERDICT_UNCERTAIN], 1)
        kept_decision_ids = {d["criterion_id"]: d["kept"] for d in debug["decisions"]}
        self.assertTrue(kept_decision_ids["lib_test_0"])
        self.assertFalse(kept_decision_ids["lib_test_1"])
        self.assertTrue(kept_decision_ids["lib_test_2"])
        self.assertEqual(len(router.calls), 1)

    def test_aggressive_keeps_only_applicable(self) -> None:
        criteria = [_criterion(i) for i in range(3)]
        router = _FakeRouter(
            responses=[
                """{
                    "0": {"verdict": "APPLICABLE", "reason": "on-topic"},
                    "1": {"verdict": "IRRELEVANT", "reason": "different cancer"},
                    "2": {"verdict": "UNCERTAIN", "reason": "borderline"}
                }"""
            ]
        )

        kept, debug = filter_relevant_criteria(
            criteria,
            prompt_text="prompt about disease X",
            candidate_texts=["response"],
            config=_enabled_config(AGGRESSIVE),
            router=router,
        )

        self.assertEqual([c.criterion_id for c in kept], ["lib_test_0"])
        self.assertEqual(debug["dropped_count"], 2)
        self.assertEqual(debug["strictness"], AGGRESSIVE)

    def test_router_failure_falls_back_to_keeping_everything_under_conservative(self) -> None:
        criteria = [_criterion(i) for i in range(2)]
        router = _FakeRouter(responses=[])
        router.raise_on_call = True

        kept, debug = filter_relevant_criteria(
            criteria,
            prompt_text="prompt",
            candidate_texts=["response"],
            config=_enabled_config(CONSERVATIVE),
            router=router,
        )

        # Conservative gate keeps UNCERTAIN, and the router-error fallback marks
        # everything UNCERTAIN, so all criteria survive.
        self.assertEqual(kept, criteria)
        batch = debug["batches"][0]
        self.assertIn("router_error", batch)
        self.assertNotEqual(batch["router_error"], "")

    def test_unrecognized_verdict_falls_back_to_uncertain(self) -> None:
        criteria = [_criterion(0)]
        router = _FakeRouter(
            responses=[
                """{"0": {"verdict": "MAYBE_LATER", "reason": "model invented a label"}}"""
            ]
        )
        kept, debug = filter_relevant_criteria(
            criteria,
            prompt_text="prompt",
            candidate_texts=["response"],
            config=_enabled_config(CONSERVATIVE),
            router=router,
        )
        self.assertEqual(kept, criteria)
        self.assertEqual(debug["decisions"][0]["verdict"], VERDICT_UNCERTAIN)
        self.assertIn("missing_or_unrecognized", debug["batches"][0]["parse_error"])


class RelevanceFilterCacheTests(unittest.TestCase):
    def test_cache_hit_avoids_second_router_call(self) -> None:
        criteria = [_criterion(i) for i in range(2)]
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "rubric_relevance_filter.jsonl"
            cache = JsonlCache(cache_path, enabled=True)
            router = _FakeRouter(
                responses=[
                    """{
                        "0": {"verdict": "APPLICABLE", "reason": "ok"},
                        "1": {"verdict": "IRRELEVANT", "reason": "off-topic"}
                    }"""
                ]
            )
            config = _enabled_config(CONSERVATIVE)

            first_kept, first_debug = filter_relevant_criteria(
                criteria,
                prompt_text="prompt",
                candidate_texts=["response"],
                config=config,
                router=router,
                cache=cache,
            )
            self.assertEqual([c.criterion_id for c in first_kept], ["lib_test_0"])
            self.assertEqual(len(router.calls), 1)
            self.assertFalse(first_debug["batches"][0]["cache_hit"])

            # Re-build a fresh cache instance from the same file to simulate a new run
            # process. The second call must hit the cache, so the empty-response router
            # would error out on a real generate() call -- if anything fires, the
            # assertion below catches it.
            second_router = _FakeRouter(responses=[])
            second_cache = JsonlCache(cache_path, enabled=True)
            second_kept, second_debug = filter_relevant_criteria(
                criteria,
                prompt_text="prompt",
                candidate_texts=["response"],
                config=config,
                router=second_router,
                cache=second_cache,
            )
            self.assertEqual([c.criterion_id for c in second_kept], ["lib_test_0"])
            self.assertEqual(second_router.calls, [])
            self.assertTrue(second_debug["batches"][0]["cache_hit"])

    def test_strictness_change_busts_the_cache(self) -> None:
        criteria = [_criterion(i) for i in range(2)]
        responses = [
            """{
                "0": {"verdict": "APPLICABLE", "reason": "ok"},
                "1": {"verdict": "UNCERTAIN", "reason": "borderline"}
            }""",
            """{
                "0": {"verdict": "APPLICABLE", "reason": "ok"},
                "1": {"verdict": "UNCERTAIN", "reason": "borderline"}
            }""",
        ]
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "rubric_relevance_filter.jsonl"
            cache = JsonlCache(cache_path, enabled=True)
            router = _FakeRouter(responses=responses)

            kept_conservative, _ = filter_relevant_criteria(
                criteria,
                prompt_text="prompt",
                candidate_texts=["response"],
                config=_enabled_config(CONSERVATIVE),
                router=router,
                cache=cache,
            )
            kept_aggressive, _ = filter_relevant_criteria(
                criteria,
                prompt_text="prompt",
                candidate_texts=["response"],
                config=_enabled_config(AGGRESSIVE),
                router=router,
                cache=cache,
            )

            # Conservative keeps both, aggressive keeps only the APPLICABLE one.
            self.assertEqual(len(kept_conservative), 2)
            self.assertEqual([c.criterion_id for c in kept_aggressive], ["lib_test_0"])
            # Two LLM calls were issued because the strictness flip changes the cache key.
            self.assertEqual(len(router.calls), 2)


class RelevanceFilterBatchingTests(unittest.TestCase):
    def test_large_input_is_processed_in_multiple_batches(self) -> None:
        criteria = [_criterion(i) for i in range(5)]
        # max_criteria_per_call=2 forces 3 batches (2+2+1)
        config = RelevanceFilterConfig(
            enabled=True,
            model_spec=DEFAULT_FILTER_MODEL,
            strictness=CONSERVATIVE,
            max_criteria_per_call=2,
        )
        router = _FakeRouter(
            responses=[
                """{"0":{"verdict":"APPLICABLE","reason":"a"},"1":{"verdict":"APPLICABLE","reason":"b"}}""",
                """{"0":{"verdict":"IRRELEVANT","reason":"c"},"1":{"verdict":"APPLICABLE","reason":"d"}}""",
                """{"0":{"verdict":"APPLICABLE","reason":"e"}}""",
            ]
        )

        kept, debug = filter_relevant_criteria(
            criteria,
            prompt_text="prompt",
            candidate_texts=["response"],
            config=config,
            router=router,
        )

        # Dropped: lib_test_2 (irrelevant). Survivors keep dataset order.
        self.assertEqual(
            [c.criterion_id for c in kept],
            ["lib_test_0", "lib_test_1", "lib_test_3", "lib_test_4"],
        )
        self.assertEqual(debug["dropped_count"], 1)
        self.assertEqual(len(debug["batches"]), 3)
        self.assertEqual(debug["batches"][0]["size"], 2)
        self.assertEqual(debug["batches"][1]["size"], 2)
        self.assertEqual(debug["batches"][2]["size"], 1)


class RelevanceFilterHelperTests(unittest.TestCase):
    def test_parse_strictness_round_trips(self) -> None:
        self.assertEqual(parse_strictness(CONSERVATIVE), CONSERVATIVE)
        self.assertEqual(parse_strictness(AGGRESSIVE), AGGRESSIVE)
        self.assertEqual(parse_strictness("CONSERVATIVE"), CONSERVATIVE)
        self.assertEqual(parse_strictness(None), CONSERVATIVE)
        self.assertEqual(parse_strictness(""), CONSERVATIVE)

    def test_parse_strictness_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            parse_strictness("loose")

    def test_build_default_config_uses_default_model(self) -> None:
        config = build_default_config(enabled=True)
        self.assertTrue(config.enabled)
        self.assertEqual(config.model_spec.alias, DEFAULT_FILTER_MODEL.alias)
        self.assertEqual(config.strictness, CONSERVATIVE)
        self.assertEqual(config.prompt_version, RELEVANCE_FILTER_PROMPT_VERSION)

    def test_invalid_strictness_in_config_raises(self) -> None:
        with self.assertRaises(ValueError):
            RelevanceFilterConfig(enabled=True, strictness="loose")


class RelevanceFilterRouterRequiredTests(unittest.TestCase):
    def test_enabled_filter_without_router_raises(self) -> None:
        with self.assertRaises(ValueError):
            filter_relevant_criteria(
                [_criterion(0)],
                prompt_text="prompt",
                candidate_texts=["response"],
                config=_enabled_config(),
                router=None,
            )


if __name__ == "__main__":
    unittest.main()
