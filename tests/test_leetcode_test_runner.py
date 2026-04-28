from __future__ import annotations

import unittest

from rubric_gen.compiled.leetcode_test_runner import (
    _parse_input_block,
    _parse_method_signature,
    evaluate_leetcode_pair_verifier,
    parse_leetcode_examples,
)


class ParseMethodSignatureTests(unittest.TestCase):
    def test_extracts_from_class_def(self) -> None:
        q = """class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        pass"""
        m, params = _parse_method_signature(q)
        self.assertEqual(m, "maxOperations")
        self.assertEqual(params, ["nums"])

    def test_extracts_multi_param(self) -> None:
        q = "class Solution:\n    def doIt(self, nums: List[int], k: int, target: str) -> bool:\n        pass"
        m, params = _parse_method_signature(q)
        self.assertEqual(m, "doIt")
        self.assertEqual(params, ["nums", "k", "target"])

    def test_falls_back_to_candidate_code(self) -> None:
        code = "class Solution:\n    def helper(self, x: int) -> int:\n        return x"
        m, params = _parse_method_signature("just question prose", candidate_code=code)
        self.assertEqual(m, "helper")
        self.assertEqual(params, ["x"])


class ParseInputBlockTests(unittest.TestCase):
    def test_single_list_param(self) -> None:
        kwargs, err = _parse_input_block("nums = [3,2,1,4,5]", ["nums"])
        self.assertEqual(err, "")
        self.assertEqual(kwargs, {"nums": [3, 2, 1, 4, 5]})

    def test_multi_param(self) -> None:
        kwargs, err = _parse_input_block("nums = [1,2,3], k = 2", ["nums", "k"])
        self.assertEqual(err, "")
        self.assertEqual(kwargs, {"nums": [1, 2, 3], "k": 2})

    def test_string_param(self) -> None:
        kwargs, err = _parse_input_block('s = "abc"', ["s"])
        self.assertEqual(err, "")
        self.assertEqual(kwargs, {"s": "abc"})

    def test_missing_expected_param(self) -> None:
        kwargs, err = _parse_input_block("nums = [1,2]", ["nums", "k"])
        self.assertNotEqual(err, "")


class ParseExamplesTests(unittest.TestCase):
    def test_extracts_examples(self) -> None:
        q = """Some prose.

Example 1:

Input: nums = [3,2,1,4,5]
Output: 2
Explanation: blah.

Example 2:

Input: nums = [1,2,3]
Output: 1
Explanation: more.

Constraints:
2 <= nums.length"""
        examples = parse_leetcode_examples(q, expected_params=["nums"])
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].parsed_kwargs, {"nums": [3, 2, 1, 4, 5]})
        self.assertEqual(examples[0].raw_output, "2")
        self.assertEqual(examples[1].parsed_kwargs, {"nums": [1, 2, 3]})


class EvaluateLeetcodePairTests(unittest.TestCase):
    def _question(self) -> str:
        return """class Solution:
    def addOne(self, n: int) -> int:
        pass

Example 1:

Input: n = 5
Output: 6
Explanation: 5 + 1 = 6.

Example 2:

Input: n = 10
Output: 11
"""

    def test_correct_candidate_wins(self) -> None:
        correct_code = """class Solution:
    def addOne(self, n: int) -> int:
        return n + 1"""
        wrong_code = """class Solution:
    def addOne(self, n: int) -> int:
        return n + 2"""
        outcome = evaluate_leetcode_pair_verifier(
            question=self._question(),
            response_a=f"```python\n{correct_code}\n```",
            response_b=f"```python\n{wrong_code}\n```",
            timeout_s=10.0,
            min_margin=0.34,
        )
        self.assertTrue(outcome.triggered)
        self.assertEqual(outcome.recommended_decision, "A>B")

    def test_no_examples_does_not_trigger(self) -> None:
        outcome = evaluate_leetcode_pair_verifier(
            question="class Solution:\n    def f(self, x): pass\n\nNo examples here.",
            response_a="```python\nclass Solution:\n    def f(self, x): return x\n```",
            response_b="```python\nclass Solution:\n    def f(self, x): return x+1\n```",
            timeout_s=10.0,
        )
        self.assertFalse(outcome.triggered)


if __name__ == "__main__":
    unittest.main()
