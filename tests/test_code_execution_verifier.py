from __future__ import annotations

import unittest

from rubric_gen.compiled.code_execution_verifier import (
    IOExample,
    evaluate_candidate_code,
    evaluate_code_pair_verifier,
    extract_candidate_code,
    extract_visible_io_pairs,
    run_in_sandbox,
)


_LCB_PROMPT = """\
You are given an integer n. Print n+1.

Example 1:

Input:
3

Output:
4

Example 2:

Input:
0

Output:
1
"""


_LCB_PROMPT_NO_HEADERS = "Just a description with no example block."


_ATCODER_PROMPT = """\
Takahashi played N games. Print N+1.

Sample Input 1

3

Sample Output 1

4

Sample Input 2

0

Sample Output 2

1
"""


_GOOD_CODE = """\
n = int(input())
print(n + 1)
"""

_BAD_CODE = """\
n = int(input())
print(n + 2)
"""

_TIMEOUT_CODE = """\
while True:
    pass
"""

_SYNTAX_ERROR_CODE = """\
def foo(:
    pass
"""


class ExtractVisibleIOPairsTests(unittest.TestCase):
    def test_parses_example_blocks(self) -> None:
        pairs = extract_visible_io_pairs(_LCB_PROMPT)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].inputs.strip(), "3")
        self.assertEqual(pairs[0].outputs.strip(), "4")
        self.assertEqual(pairs[1].inputs.strip(), "0")
        self.assertEqual(pairs[1].outputs.strip(), "1")

    def test_no_examples_returns_empty(self) -> None:
        self.assertEqual(extract_visible_io_pairs(_LCB_PROMPT_NO_HEADERS), [])

    def test_parses_atcoder_sample_input_output_format(self) -> None:
        pairs = extract_visible_io_pairs(_ATCODER_PROMPT)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].inputs.strip(), "3")
        self.assertEqual(pairs[0].outputs.strip(), "4")
        self.assertEqual(pairs[1].inputs.strip(), "0")
        self.assertEqual(pairs[1].outputs.strip(), "1")

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(extract_visible_io_pairs(""), [])


class ExtractCandidateCodeTests(unittest.TestCase):
    def test_prefers_python_fence(self) -> None:
        text = "Here is my code:\n\n```python\nprint('hello')\n```\n\nDone."
        self.assertEqual(extract_candidate_code(text), "print('hello')")

    def test_prefers_largest_fence(self) -> None:
        text = (
            "First try:\n```python\nprint(1)\n```\n\n"
            "Better:\n```python\nfor i in range(3):\n    print(i)\n```"
        )
        result = extract_candidate_code(text)
        self.assertIn("for i in range(3):", result)

    def test_fallback_to_indented_block(self) -> None:
        text = (
            "Solution:\n\n"
            "def solve():\n    n = int(input())\n    print(n)\n\n"
            "And then we call solve.\n"
        )
        result = extract_candidate_code(text)
        self.assertIn("def solve():", result)

    def test_empty_response(self) -> None:
        self.assertEqual(extract_candidate_code(""), "")


class RunInSandboxTests(unittest.TestCase):
    def test_correct_code_runs(self) -> None:
        result = run_in_sandbox(_GOOD_CODE, stdin="3\n")
        self.assertTrue(result.success)
        self.assertEqual(result.stdout.strip(), "4")
        self.assertFalse(result.timed_out)

    def test_timeout_aborts_long_loop(self) -> None:
        result = run_in_sandbox(_TIMEOUT_CODE, stdin="", timeout_s=1.0)
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)

    def test_syntax_error_reports_failure(self) -> None:
        result = run_in_sandbox(_SYNTAX_ERROR_CODE)
        self.assertFalse(result.success)
        self.assertFalse(result.timed_out)
        self.assertNotEqual(result.return_code, 0)

    def test_empty_code_returns_failure(self) -> None:
        result = run_in_sandbox("")
        self.assertFalse(result.success)


class EvaluateCandidateCodeTests(unittest.TestCase):
    def test_passes_all_io_cases(self) -> None:
        io_pairs = [IOExample(inputs="3", outputs="4"), IOExample(inputs="0", outputs="1")]
        report = evaluate_candidate_code(_GOOD_CODE, io_pairs)
        self.assertEqual(report.pass_count, 2)
        self.assertEqual(report.total_cases, 2)
        self.assertEqual(report.pass_rate, 1.0)
        self.assertTrue(report.code_extracted)

    def test_fails_some_cases(self) -> None:
        io_pairs = [IOExample(inputs="3", outputs="4"), IOExample(inputs="0", outputs="1")]
        report = evaluate_candidate_code(_BAD_CODE, io_pairs)
        self.assertEqual(report.pass_count, 0)

    def test_no_code_returns_zero_pass(self) -> None:
        report = evaluate_candidate_code("", [IOExample(inputs="x", outputs="y")])
        self.assertEqual(report.pass_count, 0)
        self.assertFalse(report.code_extracted)

    def test_no_io_pairs_returns_zero(self) -> None:
        report = evaluate_candidate_code(_GOOD_CODE, [])
        self.assertEqual(report.total_cases, 0)
        self.assertEqual(report.pass_rate, 0.0)


class EvaluateCodePairVerifierTests(unittest.TestCase):
    def test_a_passes_more_cases_triggers_high_confidence(self) -> None:
        outcome = evaluate_code_pair_verifier(
            question=_LCB_PROMPT,
            response_a=f"```python\n{_GOOD_CODE}\n```",
            response_b=f"```python\n{_BAD_CODE}\n```",
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "A>B")
        self.assertEqual(outcome.confidence, "high")
        self.assertEqual(outcome.io_pair_count, 2)

    def test_b_better_returns_b_decision(self) -> None:
        outcome = evaluate_code_pair_verifier(
            question=_LCB_PROMPT,
            response_a=f"```python\n{_BAD_CODE}\n```",
            response_b=f"```python\n{_GOOD_CODE}\n```",
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "B>A")

    def test_no_visible_io_returns_no_trigger(self) -> None:
        outcome = evaluate_code_pair_verifier(
            question=_LCB_PROMPT_NO_HEADERS,
            response_a=f"```python\n{_GOOD_CODE}\n```",
            response_b=f"```python\n{_BAD_CODE}\n```",
        )
        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.reason, "no_visible_io_pairs")

    def test_missing_candidate_code_returns_no_trigger(self) -> None:
        outcome = evaluate_code_pair_verifier(
            question=_LCB_PROMPT,
            response_a="No code provided here.",
            response_b=f"```python\n{_GOOD_CODE}\n```",
        )
        self.assertFalse(outcome.triggered)
        self.assertEqual(outcome.reason, "missing_candidate_code")


if __name__ == "__main__":
    unittest.main()
