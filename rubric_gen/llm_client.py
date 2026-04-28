from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from rubric_gen.types import LLMTextResponse, ModelSpec


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
TAG_VALUE_RE = re.compile(r"<(?P<tag>[A-Z_]+)>\s*(?P<value>.*?)\s*</(?P=tag)>", re.DOTALL)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_code_fences(text)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(cleaned)
    if not match:
        return None

    candidate = match.group(0)
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        return None
    return None


def extract_tagged_value(text: str, tag_name: str) -> Optional[str]:
    for match in TAG_VALUE_RE.finditer(text):
        if match.group("tag").upper() == tag_name.upper():
            return match.group("value").strip()
    return None


def parse_yes_no(text: str) -> Tuple[Optional[bool], str]:
    payload = extract_json_object(text)
    if payload:
        for key in ("verdict", "evaluation", "label", "answer"):
            raw = payload.get(key)
            if isinstance(raw, str):
                normalized = raw.strip().upper()
                if normalized in {"YES", "Y", "TRUE", "PASS"}:
                    return True, str(payload.get("reasoning", payload.get("rationale", "")))
                if normalized in {"NO", "N", "FALSE", "FAIL"}:
                    return False, str(payload.get("reasoning", payload.get("rationale", "")))

    tagged = extract_tagged_value(text, "EVALUATION")
    if tagged:
        normalized = tagged.strip().upper()
        if normalized in {"YES", "Y", "TRUE", "PASS"}:
            return True, extract_tagged_value(text, "REASONING") or ""
        if normalized in {"NO", "N", "FALSE", "FAIL"}:
            return False, extract_tagged_value(text, "REASONING") or ""

    normalized = strip_code_fences(text).strip().upper()
    first_token = normalized.split()[0] if normalized else ""
    if first_token in {"YES", "Y", "TRUE", "PASS"}:
        return True, ""
    if first_token in {"NO", "N", "FALSE", "FAIL"}:
        return False, ""
    return None, ""


class LLMRouter:
    def __init__(self, max_retries: int = 3, base_sleep_s: float = 1.0):
        self.max_retries = max(1, max_retries)
        self.base_sleep_s = max(0.1, base_sleep_s)
        self._clients: Dict[str, Any] = {}

    def _client_key(self, spec: ModelSpec) -> str:
        return f"{spec.provider}:{spec.model}:{spec.base_url or ''}"

    def _api_key(self, spec: ModelSpec) -> str:
        api_key = os.getenv(spec.api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"Environment variable '{spec.api_key_env}' is required for {spec.alias}."
            )
        return api_key

    def _get_client(self, spec: ModelSpec) -> Any:
        key = self._client_key(spec)
        if key in self._clients:
            return self._clients[key]

        if spec.provider in {"openai", "openai_compatible"}:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key(spec), base_url=spec.base_url)
        elif spec.provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=self._api_key(spec))
        else:
            raise ValueError(f"Unsupported provider '{spec.provider}'.")

        self._clients[key] = client
        return client

    def generate(
        self,
        spec: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMTextResponse:
        last_error: Optional[Exception] = None
        effective_max_tokens = max_tokens or spec.max_tokens

        for attempt in range(self.max_retries):
            started_at = time.perf_counter()
            try:
                client = self._get_client(spec)
                if spec.provider in {"openai", "openai_compatible"}:
                    is_gpt5 = spec.model.lower().startswith("gpt-5")
                    if is_gpt5:
                        reasoning_effort = "none" if spec.model.lower().startswith("gpt-5.4") else "minimal"
                        response = client.responses.create(
                            model=spec.model,
                            instructions=system_prompt,
                            input=user_prompt,
                            max_output_tokens=effective_max_tokens,
                            reasoning={"effort": reasoning_effort},
                        )
                        text = response.output_text or ""
                    else:
                        response = client.chat.completions.create(
                            model=spec.model,
                            temperature=temperature,
                            max_tokens=effective_max_tokens,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        )
                        text = response.choices[0].message.content or ""
                else:
                    response = client.messages.create(
                        model=spec.model,
                        system=system_prompt,
                        temperature=temperature,
                        max_tokens=effective_max_tokens,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    text = "".join(
                        block.text
                        for block in response.content
                        if getattr(block, "type", "") == "text"
                    )

                elapsed = time.perf_counter() - started_at
                return LLMTextResponse(
                    text=strip_code_fences(text),
                    raw_text=text,
                    latency_s=elapsed,
                    model_alias=spec.alias,
                    provider=spec.provider,
                    metadata={"model": spec.model},
                )
            except Exception as exc:  # pragma: no cover - provider exceptions vary
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.base_sleep_s * (2**attempt))

        assert last_error is not None
        raise RuntimeError(
            f"LLM call failed for {spec.alias} after {self.max_retries} attempts: {last_error}"
        ) from last_error
