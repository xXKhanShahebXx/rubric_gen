"""
LeetCode-style test harness for ``livecodebench`` JudgeBench examples.

The existing :mod:`code_execution_verifier` runs candidate code with stdin/stdout (AtCoder
format) and never triggered on LiveCodeBench because LiveCodeBench questions follow the
LeetCode pattern: ``class Solution: def methodName(self, args): ...``, with examples
formatted as::

    Example 1:
    Input: nums = [3,2,1,4,5]
    Output: 2

This module:

1. Parses the method signature from the question (the ``class Solution`` lines).
2. Parses ``Input: kw = literal, kw = literal, ...`` example blocks.
3. For each example, builds a Python harness that imports the candidate's class
   definition, calls ``Solution().method_name(**kwargs)``, and prints the return value.
4. Executes the harness in the existing isolated subprocess sandbox.
5. Compares ``stdout`` against the example's ``Output:`` value.

A candidate's pass-rate is the fraction of examples it passes; when the difference exceeds
``min_margin`` (default 0.34) the verifier fires high confidence in favour of the
higher-passing candidate.

Sandboxing: reuses :func:`code_execution_verifier.run_in_sandbox` (subprocess with
``python -I``, per-case timeout, no network restriction).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rubric_gen.compiled.code_execution_verifier import (
    CandidatePassReport,
    CodeExecutionVerifierOutcome,
    ExecutionResult,
    extract_candidate_code,
    run_in_sandbox,
)


_CLASS_METHOD_SIG_RE = re.compile(
    r"class\s+Solution\s*:\s*\n(?:[^\n]*\n)*?\s*def\s+(\w+)\s*\(\s*self\s*(?:,\s*([^)]*))?\)",
    re.MULTILINE,
)
_DEF_METHOD_SIG_RE = re.compile(
    r"^\s*def\s+(\w+)\s*\(\s*self\s*(?:,\s*([^)]*))?\)\s*(?:->\s*[^:]+)?:",
    re.MULTILINE,
)
_EXAMPLE_INPUT_RE = re.compile(
    r"(?:^|\n)\s*Input\s*:\s*(.+?)(?=\n\s*Output\s*:)",
    re.IGNORECASE | re.DOTALL,
)
_EXAMPLE_OUTPUT_RE = re.compile(
    r"(?:^|\n)\s*Output\s*:\s*(.+?)(?=\n\s*(?:Explanation|Example\s*\d|Note|Constraints?|$))",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class LeetcodeExample:
    """One LeetCode-style Input / Output pair, parsed for callable invocation."""

    raw_input: str
    raw_output: str
    parsed_kwargs: Dict[str, Any] = field(default_factory=dict)
    parse_error: str = ""


def _parse_method_signature(question: str, *, candidate_code: str = "") -> Tuple[str, List[str]]:
    """
    Find the ``method_name`` and parameter names from either the question or candidate code.

    LeetCode questions sometimes embed the signature in the question body as:
    ``Implement the following ... class Solution: def maxOperations(self, nums: List[int]) -> int``
    """
    for source in (question or "", candidate_code or ""):
        for regex in (_CLASS_METHOD_SIG_RE, _DEF_METHOD_SIG_RE):
            m = regex.search(source)
            if not m:
                continue
            method = m.group(1).strip()
            params_text = (m.group(2) or "").strip()
            params: List[str] = []
            for raw in params_text.split(","):
                raw = raw.strip()
                if not raw:
                    continue
                name = raw.split(":")[0].split("=")[0].strip()
                if name:
                    params.append(name)
            return method, params
    return "", []


def _parse_value(literal_text: str) -> Tuple[Any, str]:
    """
    Parse a Python literal expression.

    Returns ``(value, "")`` on success or ``(None, error_string)`` on failure.
    Wraps ``ast.literal_eval`` with light pre-processing (smart quotes, ``true``/``false``,
    ``null``, trailing commas / whitespace).
    """
    if literal_text is None:
        return None, "empty"
    s = literal_text.strip()
    if not s:
        return None, "empty"
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    if s.lower() == "true":
        return True, ""
    if s.lower() == "false":
        return False, ""
    if s.lower() in ("null", "none"):
        return None, ""
    if s.endswith(","):
        s = s[:-1].rstrip()
    try:
        return ast.literal_eval(s), ""
    except Exception as exc:
        return None, f"literal_eval_error:{type(exc).__name__}"


def _parse_input_block(block: str, expected_params: List[str]) -> Tuple[Dict[str, Any], str]:
    """
    Parse an ``Input:`` block of the form ``name1 = literal, name2 = literal``.

    Strategy:
    - Greedy split on ``,\\s*\\w+\\s*=`` only when the right-hand side is balanced (so we don't
      split inside lists / strings). We use a small parser that respects bracket depth.
    """
    if not block:
        return {}, "empty"
    cleaned = block.strip().rstrip("\n")
    # Strip trailing prose like "...such that the sum equals i"
    parts: List[Tuple[str, str]] = []
    i = 0
    text = cleaned
    n = len(text)
    while i < n:
        m = re.match(r"\s*(\w+)\s*=\s*", text[i:])
        if not m:
            break
        name = m.group(1)
        i += m.end()
        depth = 0
        in_string: Optional[str] = None
        start = i
        while i < n:
            c = text[i]
            if in_string:
                if c == "\\":
                    i += 2
                    continue
                if c == in_string:
                    in_string = None
                i += 1
                continue
            if c in ("'", '"'):
                in_string = c
                i += 1
                continue
            if c in "[({":
                depth += 1
            elif c in "])}":
                depth -= 1
            elif c == "," and depth == 0:
                break
            i += 1
        value_text = text[start:i].strip()
        parts.append((name, value_text))
        if i < n and text[i] == ",":
            i += 1
    if not parts:
        return {}, "no_param_split"
    parsed: Dict[str, Any] = {}
    for name, raw in parts:
        value, err = _parse_value(raw)
        if err:
            return {}, f"param_{name}_{err}"
        parsed[name] = value
    if expected_params:
        for ep in expected_params:
            if ep not in parsed:
                return {}, f"missing_param_{ep}"
    return parsed, ""


def parse_leetcode_examples(
    question: str,
    *,
    expected_params: List[str],
    max_pairs: int = 8,
) -> List[LeetcodeExample]:
    """
    Parse LeetCode-style ``Example k:`` blocks from the question.

    Returns a list of :class:`LeetcodeExample`. Examples that fail to parse are still included
    with ``parse_error`` set so callers can count parsing failures.
    """
    if not question:
        return []
    inputs = list(_EXAMPLE_INPUT_RE.finditer(question))
    outputs = list(_EXAMPLE_OUTPUT_RE.finditer(question))
    n = min(len(inputs), len(outputs), max_pairs)
    examples: List[LeetcodeExample] = []
    for k in range(n):
        in_block = inputs[k].group(1).strip()
        out_block = outputs[k].group(1).strip()
        # Strip trailing prose from output (sometimes "Output: 6\n\nExplanation: ..." sneaks in)
        out_block = out_block.split("\n")[0].strip()
        parsed_kwargs, err = _parse_input_block(in_block, expected_params)
        examples.append(
            LeetcodeExample(
                raw_input=in_block,
                raw_output=out_block,
                parsed_kwargs=parsed_kwargs,
                parse_error=err,
            )
        )
    return examples


def _build_test_harness(candidate_code: str, method_name: str, kwargs: Dict[str, Any]) -> str:
    """
    Wrap candidate code with a ``Solution().method(**kwargs)`` invocation that prints the result.

    We import ``typing`` symbols up front because LiveCodeBench candidates often type-hint
    parameters with ``List[int]`` / ``Optional[str]`` etc. but don't import them.
    """
    args_repr = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
    return (
        "from typing import List, Optional, Dict, Tuple, Set, Any\n"
        "import sys\n"
        "import json\n"
        "import math\n"
        "import collections\n"
        "from collections import Counter, defaultdict, deque\n\n"
        f"{candidate_code.rstrip()}\n\n"
        "_sol = Solution()\n"
        f"_result = _sol.{method_name}({args_repr})\n"
        "if isinstance(_result, (list, tuple, set, dict)):\n"
        "    print(json.dumps(_result, sort_keys=True))\n"
        "else:\n"
        "    print(_result)\n"
    )


def _normalize_expected(text: str) -> str:
    if not text:
        return ""
    return text.strip().rstrip(",.").strip()


def _normalize_actual(text: str) -> str:
    if not text:
        return ""
    return text.strip()


def _outputs_match(actual: str, expected: str) -> bool:
    a = _normalize_actual(actual)
    e = _normalize_expected(expected)
    if a == e:
        return True
    # Try parsing both as JSON / Python literal
    actual_val, actual_err = _parse_value(a)
    expected_val, expected_err = _parse_value(e)
    if not actual_err and not expected_err:
        if actual_val == expected_val:
            return True
        try:
            if isinstance(actual_val, list) and isinstance(expected_val, list):
                if len(actual_val) == len(expected_val) and sorted(actual_val) == sorted(expected_val):
                    pass
        except TypeError:
            pass
    return False


def evaluate_leetcode_candidate(
    candidate_code: str,
    examples: List[LeetcodeExample],
    method_name: str,
    *,
    timeout_s: float = 10.0,
    python_executable: Optional[str] = None,
) -> CandidatePassReport:
    if not candidate_code or not method_name or not examples:
        return CandidatePassReport(
            code_extracted=bool(candidate_code),
            code_text=candidate_code or "",
            pass_count=0,
            total_cases=len(examples),
            case_details=tuple(),
        )
    case_details: List[Dict[str, Any]] = []
    pass_count = 0
    runnable_count = 0
    for idx, ex in enumerate(examples):
        if ex.parse_error:
            case_details.append(
                {
                    "case_index": idx,
                    "passed": False,
                    "timed_out": False,
                    "return_code": -1,
                    "stderr_preview": f"parse:{ex.parse_error}",
                    "stdout_preview": "",
                }
            )
            continue
        runnable_count += 1
        harness = _build_test_harness(candidate_code, method_name, ex.parsed_kwargs)
        result = run_in_sandbox(
            harness,
            stdin="",
            timeout_s=timeout_s,
            python_executable=python_executable,
        )
        passed = result.success and _outputs_match(result.stdout, ex.raw_output)
        case_details.append(
            {
                "case_index": idx,
                "passed": bool(passed),
                "timed_out": bool(result.timed_out),
                "return_code": int(result.return_code),
                "stderr_preview": (result.stderr or "")[:200],
                "stdout_preview": (result.stdout or "")[:200],
            }
        )
        if passed:
            pass_count += 1
    return CandidatePassReport(
        code_extracted=True,
        code_text=candidate_code,
        pass_count=pass_count,
        total_cases=runnable_count if runnable_count else len(examples),
        case_details=tuple(case_details),
    )


def evaluate_leetcode_pair_verifier(
    *,
    question: str,
    response_a: str,
    response_b: str,
    timeout_s: float = 10.0,
    min_margin: float = 0.34,
) -> CodeExecutionVerifierOutcome:
    code_a = extract_candidate_code(response_a)
    code_b = extract_candidate_code(response_b)
    method_name, expected_params = _parse_method_signature(question, candidate_code=code_a or code_b)
    if not method_name:
        return CodeExecutionVerifierOutcome(
            triggered=False,
            recommended_decision="",
            confidence="",
            reason="no_method_signature",
            a_report=None,
            b_report=None,
            io_pair_count=0,
        )
    examples = parse_leetcode_examples(question, expected_params=expected_params)
    if not examples:
        return CodeExecutionVerifierOutcome(
            triggered=False,
            recommended_decision="",
            confidence="",
            reason="no_leetcode_examples",
            a_report=None,
            b_report=None,
            io_pair_count=0,
        )
    if not code_a and not code_b:
        return CodeExecutionVerifierOutcome(
            triggered=False,
            recommended_decision="",
            confidence="",
            reason="no_candidate_code",
            a_report=None,
            b_report=None,
            io_pair_count=len(examples),
        )
    report_a = evaluate_leetcode_candidate(code_a, examples, method_name, timeout_s=timeout_s)
    report_b = evaluate_leetcode_candidate(code_b, examples, method_name, timeout_s=timeout_s)
    margin = report_a.pass_rate - report_b.pass_rate
    if abs(margin) < min_margin:
        return CodeExecutionVerifierOutcome(
            triggered=False,
            recommended_decision="",
            confidence="low",
            reason=f"insufficient_margin_{report_a.pass_count}_{report_b.pass_count}",
            a_report=report_a,
            b_report=report_b,
            io_pair_count=len(examples),
        )
    if margin > 0:
        decision = "A>B"
    else:
        decision = "B>A"
    return CodeExecutionVerifierOutcome(
        triggered=True,
        recommended_decision=decision,
        confidence="high",
        reason=f"leetcode_pass_rate_a={report_a.pass_count}/{report_a.total_cases}_b={report_b.pass_count}/{report_b.total_cases}",
        a_report=report_a,
        b_report=report_b,
        io_pair_count=len(examples),
    )
