"""
Build a strict-disjoint JudgeBench-style train split from upstream datasets.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from rubric_gen.compiled.contrast_strategies import mutation_function_for_id

_BUILDER_VERSION = "strict_disjoint_v1"
_CURRENT_VAL_A = Path("data/judgebench_80_human.json")
_CURRENT_VAL_B = Path("data/judgebench_270_generated.json")
_OUTPUT_ROOT = Path("artifacts/generated_judgebench_splits")
_NONCODE_MODEL = "gpt-4o-mini"
_CODE_MODEL = "gpt-4o"
_NONCODE_MUTATION_ID = "corrupt_final_answer"
_CODE_MUTATION_IDS = (
    "code_off_by_one_loop",
    "code_flip_condition_branch",
    "code_corrupt_input_parsing",
    "code_drop_negative_directions",
    "code_remove_zero_guard",
    "code_drop_reset_logic",
)
_OPENAI_REQUEST_TIMEOUT_S = 120
_OPENAI_REQUEST_RETRIES = 3
_MMLU_OPTION_LABELS = tuple("ABCDEFGHIJ")
_WHITESPACE_RE = re.compile(r"\s+")
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_LINE_RE = re.compile(r"(?im)^\s*final answer\s*:\s*.+$")


@dataclass(frozen=True)
class StrictSplitArtifacts:
    output_dir: Path
    train_dataset_path: Path
    validation_dataset_path: Path
    official_dataset_path: Path
    manifest_path: Path


@dataclass
class CodeAttemptResult:
    code: str
    passed: bool
    failures: List[Dict[str, str]]
    attempt_count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip().lower()).strip("-") or "unknown"


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip()).lower()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array at {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _pair_id_for(family: str, source_id: str) -> str:
    namespace = f"judgebench:{_BUILDER_VERSION}:{family}:{source_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, namespace))


def _question_answer_format(question: str, ground_truth: str) -> str:
    answer = (ground_truth or "").strip()
    lowered = question.lower()
    if "***" in question and not answer.startswith("***"):
        return f"***{answer.strip('* ')}***"
    if "**bold**" in lowered or "put your answer in **bold**" in lowered:
        return f"**{answer.strip('* ')}**"
    return answer


def _build_mmlu_prompt(question: str, options: Sequence[str]) -> str:
    lines = [question.strip()]
    for label, option in zip(_MMLU_OPTION_LABELS, options):
        lines.append(f"({label}) {str(option).strip()}")
    lines.append(
        "If you cannot determine the correct multiple-choice answer, take your best guess. "
        "Once you have your answer, please duplicate that letter five times in a single string. "
        "For example, if the answer is K, then write KKKKK."
    )
    lines.append("Let's think step by step.")
    return "\n".join(lines).strip()


def _load_current_validation_context(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set[int], set[str]]:
    current_a = _load_json_array(root / _CURRENT_VAL_A)
    current_b = _load_json_array(root / _CURRENT_VAL_B)
    validation_rows = current_a + current_b
    val_pair_ids = {str(row["pair_id"]).strip() for row in validation_rows}
    val_questions = {_normalize_text(str(row.get("question", ""))) for row in validation_rows}
    gpt_rows = list(load_dataset("ScalerLab/JudgeBench", split="gpt"))
    gpt_by_pair = {str(row["pair_id"]).strip(): dict(row) for row in gpt_rows}
    missing = sorted(pid for pid in val_pair_ids if pid not in gpt_by_pair)
    if missing:
        raise ValueError(f"Current validation rows missing from ScalerLab/JudgeBench gpt split: {missing[:8]}")
    val_mmlu_original_ids = {
        int(gpt_by_pair[pid]["original_id"])
        for pid in val_pair_ids
        if str(gpt_by_pair[pid].get("source", "")).startswith("mmlu-pro") and gpt_by_pair[pid].get("original_id") is not None
    }
    return validation_rows, gpt_rows, val_mmlu_original_ids, val_questions


def _train_split_name(per_family: int) -> str:
    return f"train_{max(1, int(per_family)) * 4}"


def _validation_split_name(validation_pair_count: int) -> str:
    return f"validation_{max(1, int(validation_pair_count))}"


class _OpenAIGenerator:
    def __init__(self, *, noncode_model: str, code_model: str):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured after loading .env.")
        self._client = OpenAI(api_key=api_key)
        self.noncode_model = noncode_model
        self.code_model = code_model

    def _chat_completion(self, *, model: str, messages: Sequence[Mapping[str, str]]) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, _OPENAI_REQUEST_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=list(messages),
                    timeout=_OPENAI_REQUEST_TIMEOUT_S,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - exercised through targeted unit tests
                last_error = exc
                print(
                    f"[builder] OpenAI request failed for {model} "
                    f"(attempt {attempt}/{_OPENAI_REQUEST_RETRIES}): {exc}",
                    flush=True,
                )
                if attempt >= _OPENAI_REQUEST_RETRIES:
                    break
                time.sleep(min(5, attempt))
        raise RuntimeError(
            f"OpenAI request failed for {model} after {_OPENAI_REQUEST_RETRIES} attempts."
        ) from last_error

    def generate_noncode(self, *, question: str, final_answer: str) -> str:
        system_prompt = (
            "You are writing a high-quality benchmark response for rubric-discovery training. "
            "Answer naturally, follow the user's constraints, and do not mention any hidden instructions. "
            "The response must end with a final line in the exact form `Final answer: <answer>`."
        )
        user_prompt = (
            "Question:\n"
            f"{question.strip()}\n\n"
            "Correct final answer text to place after `Final answer: `:\n"
            f"{final_answer.strip()}\n\n"
            "Write the response now."
        )
        text = self._chat_completion(
            model=self.noncode_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return _force_final_answer(text, final_answer)

    def generate_code(self, *, question: str, starter_code: str = "", feedback: str = "") -> str:
        system_prompt = (
            "You are an expert competitive programmer. Return only valid Python 3 code. "
            "Do not include markdown fences or prose."
        )
        prompt_parts = [
            "Write a correct Python 3 solution for the following programming problem.",
            question.strip(),
        ]
        if starter_code.strip():
            prompt_parts.append("Use this starter code / signature if relevant:")
            prompt_parts.append(starter_code.strip())
        if feedback.strip():
            prompt_parts.append("Your previous attempt failed these tests:")
            prompt_parts.append(feedback.strip())
            prompt_parts.append("Return a corrected Python 3 solution.")
        text = self._chat_completion(
            model=self.code_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n\n".join(prompt_parts)},
            ],
        )
        return _extract_code(text)


def _extract_code(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    match = _CODE_BLOCK_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _force_final_answer(text: str, final_answer: str) -> str:
    cleaned = _FINAL_ANSWER_LINE_RE.sub("", (text or "")).strip()
    suffix = f"Final answer: {final_answer.strip()}"
    if not cleaned:
        return suffix
    return f"{cleaned}\n\n{suffix}"


def _coerce_cases(blob: Any) -> List[Dict[str, Any]]:
    if isinstance(blob, str):
        payload = json.loads(blob)
    else:
        payload = blob
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, dict)]


def _compare_outputs(actual: str, expected: str) -> bool:
    actual_clean = (actual or "").strip()
    expected_clean = (expected or "").strip()
    if actual_clean == expected_clean:
        return True
    try:
        return ast.literal_eval(actual_clean) == ast.literal_eval(expected_clean)
    except Exception:
        return _normalize_text(actual_clean) == _normalize_text(expected_clean)


def _run_python_script(code: str, *, stdin: str = "", timeout_s: int = 8) -> Tuple[str, str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return "", "__TIMEOUT__", 124
        return proc.stdout, proc.stderr, proc.returncode


def _run_functional_case(code: str, expression: str, *, timeout_s: int = 8) -> Tuple[str, str, int]:
    harness = "\n\n".join(
        [
            code.strip(),
            "import builtins",
            f"_expr = {expression!r}",
            "result = eval(_expr, globals(), locals())",
            "print(repr(result))",
        ]
    )
    return _run_python_script(harness, timeout_s=timeout_s)


def _evaluate_code(code: str, row: Mapping[str, Any], *, max_failures: int = 3) -> Tuple[bool, List[Dict[str, str]]]:
    failures: List[Dict[str, str]] = []
    test_cases = _coerce_cases(row.get("public_test_cases")) + _coerce_cases(row.get("private_test_cases"))
    for case in test_cases:
        testtype = str(case.get("testtype", "")).strip().lower()
        expected = str(case.get("output", ""))
        if testtype == "stdin":
            actual_stdout, stderr, returncode = _run_python_script(code, stdin=str(case.get("input", "")))
        elif testtype == "functional":
            actual_stdout, stderr, returncode = _run_functional_case(code, str(case.get("input", "")))
        else:
            failures.append({"testtype": testtype or "unknown", "error": "unsupported_testtype"})
            break
        if returncode != 0 or not _compare_outputs(actual_stdout, expected):
            failures.append(
                {
                    "testtype": testtype,
                    "input": str(case.get("input", ""))[:500],
                    "expected": expected[:500],
                    "actual": (actual_stdout or stderr or "").strip()[:500],
                }
            )
            if len(failures) >= max_failures:
                break
    return not failures, failures


def _summarize_failures(failures: Sequence[Mapping[str, str]]) -> str:
    lines: List[str] = []
    for failure in failures[:3]:
        lines.append(f"Test type: {failure.get('testtype', '')}")
        if failure.get("input"):
            lines.append(f"Input: {failure['input']}")
        if failure.get("expected"):
            lines.append(f"Expected: {failure['expected']}")
        if failure.get("actual"):
            lines.append(f"Actual: {failure['actual']}")
        lines.append("")
    return "\n".join(lines).strip()


def _generate_code_solution(
    generator: _OpenAIGenerator,
    row: Mapping[str, Any],
    *,
    max_attempts: int,
) -> CodeAttemptResult:
    feedback = ""
    last_code = ""
    last_failures: List[Dict[str, str]] = []
    for attempt in range(1, max_attempts + 1):
        last_code = generator.generate_code(
            question=str(row.get("question_content", "")).strip(),
            starter_code=str(row.get("starter_code", "")).strip(),
            feedback=feedback,
        )
        passed, failures = _evaluate_code(last_code, row)
        if passed:
            return CodeAttemptResult(code=last_code, passed=True, failures=[], attempt_count=attempt)
        last_failures = failures
        feedback = _summarize_failures(failures)
    return CodeAttemptResult(code=last_code, passed=False, failures=last_failures, attempt_count=max_attempts)


def _mutate_noncode_response(text: str) -> Tuple[str, str]:
    fn = mutation_function_for_id(_NONCODE_MUTATION_ID)
    mutated = fn(text) if fn is not None else text
    if mutated != text:
        return mutated, _NONCODE_MUTATION_ID
    fallback = (text.rstrip() + "\nFinal answer: UNKNOWN").strip()
    return fallback, "fallback_unknown_final_answer"


def _mutate_code_response(code: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    for mutation_id in _CODE_MUTATION_IDS:
        fn = mutation_function_for_id(mutation_id)
        if fn is None:
            continue
        mutated = fn(code)
        if not mutated.strip() or mutated == code:
            continue
        passed, _ = _evaluate_code(mutated, row)
        if not passed:
            return mutated, mutation_id
    fallback = (code.rstrip() + "\nraise RuntimeError('intentional failure')\n").strip()
    return fallback, "fallback_runtime_failure"


def _build_pair_rows(
    *,
    pair_id: str,
    original_id: Any,
    source: str,
    question: str,
    strong_response: str,
    weak_response: str,
    response_model: str,
    rng: random.Random,
) -> Dict[str, Dict[str, Any]]:
    if rng.random() < 0.5:
        response_a, response_b, label = strong_response, weak_response, "A>B"
    else:
        response_a, response_b, label = weak_response, strong_response, "B>A"
    local_row = {
        "pair_id": pair_id,
        "source": source,
        "question": question,
        "reference_answer": strong_response,
        "candidate_models": [response_model, "synthetic_weak_response"],
        "verifier_model": _BUILDER_VERSION,
    }
    official_row = {
        "pair_id": pair_id,
        "original_id": original_id,
        "source": source,
        "question": question,
        "response_model": response_model,
        "response_A": response_a,
        "response_B": response_b,
        "label": label,
    }
    return {"local": local_row, "official": official_row}


def _sample_train_rows(
    *,
    rng: random.Random,
    generator: _OpenAIGenerator,
    validation_questions: set[str],
    validation_mmlu_original_ids: set[int],
    per_family: int,
    max_code_attempts: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_local: List[Dict[str, Any]] = []
    train_official: List[Dict[str, Any]] = []
    selection_manifest: Dict[str, Any] = {"families": {}}

    mmlu = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
    reasoning = list(load_dataset("livebench/reasoning", split="test"))
    math = list(load_dataset("livebench/math", split="test"))
    code = list(load_dataset("livecodebench/code_generation", split="test"))

    eligible_mmlu = [row for row in mmlu if int(row["question_id"]) not in validation_mmlu_original_ids]
    eligible_reasoning = [row for row in reasoning if _normalize_text(str(row["turns"][0])) not in validation_questions]
    eligible_math = [row for row in math if _normalize_text(str(row["turns"][0])) not in validation_questions]
    eligible_code = [row for row in code if _normalize_text(str(row["question_content"])) not in validation_questions]

    if len(eligible_mmlu) < per_family or len(eligible_reasoning) < per_family or len(eligible_math) < per_family:
        raise ValueError("Insufficient strict-disjoint upstream tasks to satisfy the requested family counts.")
    print(
        "[builder] eligible counts:",
        {
            "mmlu-pro": len(eligible_mmlu),
            "livebench-reasoning": len(eligible_reasoning),
            "livebench-math": len(eligible_math),
            "livecodebench": len(eligible_code),
        },
        flush=True,
    )

    sampled_mmlu = rng.sample(eligible_mmlu, per_family)
    sampled_reasoning = rng.sample(eligible_reasoning, per_family)
    sampled_math = rng.sample(eligible_math, per_family)

    def add_noncode_family(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        records: List[Dict[str, Any]] = []
        for row in rows:
            if name == "mmlu-pro":
                question = _build_mmlu_prompt(str(row["question"]).strip(), list(row["options"]))
                answer_text = str(row["answer"]).strip().upper() * 5
                source = f"mmlu-pro-{_slugify(str(row['category']))}"
                source_id = str(row["question_id"])
            else:
                question = str(row["turns"][0]).strip()
                answer_text = _question_answer_format(question, str(row["ground_truth"]).strip())
                source = name
                source_id = str(row["question_id"])
            strong_response = generator.generate_noncode(question=question, final_answer=answer_text)
            weak_response, mutation_id = _mutate_noncode_response(strong_response)
            pair_id = _pair_id_for(name, source_id)
            built = _build_pair_rows(
                pair_id=pair_id,
                original_id=source_id,
                source=source,
                question=question,
                strong_response=strong_response,
                weak_response=weak_response,
                response_model=f"{generator.noncode_model}+synthetic",
                rng=rng,
            )
            built["local"].update(
                {
                    "upstream_question_id": source_id,
                    "builder_family": name,
                    "weak_mutation_id": mutation_id,
                    "builder_version": _BUILDER_VERSION,
                }
            )
            train_local.append(built["local"])
            train_official.append(built["official"])
            records.append(
                {
                    "pair_id": pair_id,
                    "upstream_question_id": source_id,
                    "source": source,
                    "weak_mutation_id": mutation_id,
                }
            )
        selection_manifest["families"][name] = records
        print(f"[builder] completed {name}: {len(records)}/{per_family}", flush=True)

    add_noncode_family("mmlu-pro", sampled_mmlu)
    add_noncode_family("livebench-reasoning", sampled_reasoning)
    add_noncode_family("livebench-math", sampled_math)

    rng.shuffle(eligible_code)
    code_records: List[Dict[str, Any]] = []
    for row in eligible_code:
        if len(code_records) >= per_family:
            break
        question_id = str(row["question_id"]).strip()
        question = str(row["question_content"]).strip()
        result = _generate_code_solution(generator, row, max_attempts=max_code_attempts)
        if not result.passed:
            continue
        weak_response, mutation_id = _mutate_code_response(result.code, row)
        pair_id = _pair_id_for("livecodebench", question_id)
        built = _build_pair_rows(
            pair_id=pair_id,
            original_id=question_id,
            source="livecodebench",
            question=question,
            strong_response=result.code,
            weak_response=weak_response,
            response_model=f"{generator.code_model}+synthetic",
            rng=rng,
        )
        built["local"].update(
            {
                "upstream_question_id": question_id,
                "builder_family": "livecodebench",
                "platform": str(row.get("platform", "")).strip(),
                "difficulty": str(row.get("difficulty", "")).strip(),
                "weak_mutation_id": mutation_id,
                "code_generation_attempts": result.attempt_count,
                "builder_version": _BUILDER_VERSION,
            }
        )
        train_local.append(built["local"])
        train_official.append(built["official"])
        code_records.append(
            {
                "pair_id": pair_id,
                "upstream_question_id": question_id,
                "source": "livecodebench",
                "platform": str(row.get("platform", "")).strip(),
                "difficulty": str(row.get("difficulty", "")).strip(),
                "weak_mutation_id": mutation_id,
                "code_generation_attempts": result.attempt_count,
            }
        )
        print(
            f"[builder] livecodebench accepted {len(code_records)}/{per_family}: {question_id} "
            f"({str(row.get('difficulty', '')).strip() or 'unknown'})",
            flush=True,
        )
    if len(code_records) < per_family:
        raise ValueError(f"Only generated {len(code_records)} passing strict-disjoint livecodebench examples; need {per_family}.")
    selection_manifest["families"]["livecodebench"] = code_records
    return train_local, train_official, selection_manifest


def build_strict_disjoint_split(
    *,
    output_dir: Path,
    seed: int,
    per_family: int,
    noncode_model: str,
    code_model: str,
    max_code_attempts: int,
) -> StrictSplitArtifacts:
    root = _repo_root()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows, official_gpt_rows, val_mmlu_original_ids, validation_questions = _load_current_validation_context(root)
    generator = _OpenAIGenerator(noncode_model=noncode_model, code_model=code_model)
    rng = random.Random(seed)

    train_local, train_official, selection_manifest = _sample_train_rows(
        rng=rng,
        generator=generator,
        validation_questions=validation_questions,
        validation_mmlu_original_ids=val_mmlu_original_ids,
        per_family=per_family,
        max_code_attempts=max_code_attempts,
    )

    train_split_name = _train_split_name(per_family)
    validation_split_name = _validation_split_name(len(validation_rows))
    family_pair_counts: Dict[str, int] = {}
    for row in train_local:
        family = (
            str(row.get("builder_family", "")).strip()
            or str(row.get("source_family", "")).strip()
            or str(row.get("source", "")).split("-", 1)[0].strip()
            or "unknown"
        )
        family_pair_counts[family] = family_pair_counts.get(family, 0) + 1

    validation_dataset_path = output_dir / f"{validation_split_name}.json"
    train_dataset_path = output_dir / f"{train_split_name}_strict.json"
    official_dataset_path = output_dir / f"official_{train_split_name}_{validation_split_name}.jsonl"
    manifest_path = output_dir / "manifest.json"

    combined_official_rows = [dict(row) for row in official_gpt_rows] + train_official
    _write_json(train_dataset_path, train_local)
    _write_json(validation_dataset_path, validation_rows)
    _write_jsonl(official_dataset_path, combined_official_rows)
    _write_json(
        manifest_path,
        {
            "builder_version": _BUILDER_VERSION,
            "seed": seed,
            "per_family": per_family,
            "noncode_model": noncode_model,
            "code_model": code_model,
            "max_code_attempts": max_code_attempts,
            "train_split_name": train_split_name,
            "validation_split_name": validation_split_name,
            "train_pair_count": len(train_local),
            "validation_pair_count": len(validation_rows),
            "official_pair_count": len(combined_official_rows),
            "train_dataset_filename": train_dataset_path.name,
            "validation_dataset_filename": validation_dataset_path.name,
            "official_dataset_filename": official_dataset_path.name,
            "current_validation_inputs": [
                str((root / _CURRENT_VAL_A).resolve()),
                str((root / _CURRENT_VAL_B).resolve()),
            ],
            "family_pair_counts": dict(sorted(family_pair_counts.items())),
            "selection": selection_manifest,
        },
    )
    return StrictSplitArtifacts(
        output_dir=output_dir,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        official_dataset_path=official_dataset_path,
        manifest_path=manifest_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a strict-disjoint JudgeBench-style train/validation split.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the generated split bundle.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for strict-disjoint sampling.")
    parser.add_argument(
        "--per-family",
        type=int,
        default=20,
        help="Examples per source family for the train split (default: 20, yielding 80 total).",
    )
    parser.add_argument(
        "--noncode-model",
        type=str,
        default=_NONCODE_MODEL,
        help=f"OpenAI model used to draft strong non-code responses (default: {_NONCODE_MODEL}).",
    )
    parser.add_argument(
        "--code-model",
        type=str,
        default=_CODE_MODEL,
        help=f"OpenAI model used to solve strict-disjoint code tasks (default: {_CODE_MODEL}).",
    )
    parser.add_argument(
        "--max-code-attempts",
        type=int,
        default=3,
        help="Maximum code-generation attempts per livecodebench task (default: 3).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    output_dir = args.output_dir or (
        root / _OUTPUT_ROOT / f"{_BUILDER_VERSION}_{_train_split_name(max(1, int(args.per_family)))}_validation350"
    )
    artifacts = build_strict_disjoint_split(
        output_dir=output_dir,
        seed=args.seed,
        per_family=max(1, int(args.per_family)),
        noncode_model=args.noncode_model,
        code_model=args.code_model,
        max_code_attempts=max(1, int(args.max_code_attempts)),
    )
    print(f"Built strict-disjoint JudgeBench split at: {artifacts.output_dir}")
    print(f"train_dataset={artifacts.train_dataset_path}")
    print(f"validation_dataset={artifacts.validation_dataset_path}")
    print(f"official_dataset={artifacts.official_dataset_path}")


if __name__ == "__main__":
    main()
