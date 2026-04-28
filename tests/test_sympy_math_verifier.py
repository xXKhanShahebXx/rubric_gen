from __future__ import annotations

import unittest

from rubric_gen.compiled.sympy_math_verifier import (
    equivalent,
    equivalent_strings,
    try_sympify,
    verify_arithmetic_chain,
)


class TrySympifyTests(unittest.TestCase):
    def test_simple_integer(self) -> None:
        self.assertIsNotNone(try_sympify("42"))

    def test_fraction(self) -> None:
        expr = try_sympify("4/8")
        self.assertIsNotNone(expr)

    def test_latex_frac(self) -> None:
        expr = try_sympify("\\frac{1}{2}")
        self.assertIsNotNone(expr)

    def test_latex_boxed(self) -> None:
        expr = try_sympify("$\\boxed{42}$")
        self.assertIsNotNone(expr)

    def test_latex_sqrt(self) -> None:
        expr = try_sympify("\\sqrt{2}")
        self.assertIsNotNone(expr)

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(try_sympify(""))

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(try_sympify(":::not_math:::"))


class EquivalentStringsTests(unittest.TestCase):
    def test_string_equal_short_circuits(self) -> None:
        self.assertTrue(equivalent_strings("42", "42"))

    def test_fraction_equivalence(self) -> None:
        self.assertTrue(equivalent_strings("4/8", "1/2"))

    def test_decimal_equivalence(self) -> None:
        self.assertTrue(equivalent_strings("0.5", "1/2"))

    def test_latex_fraction_equivalence(self) -> None:
        self.assertTrue(equivalent_strings("\\frac{1}{2}", "0.5"))

    def test_unrelated_values_not_equivalent(self) -> None:
        self.assertFalse(equivalent_strings("42", "43"))

    def test_empty_returns_false(self) -> None:
        self.assertFalse(equivalent_strings("", "42"))
        self.assertFalse(equivalent_strings("42", ""))


class VerifyArithmeticChainTests(unittest.TestCase):
    def test_correct_arithmetic_steps(self) -> None:
        text = "Step 1: 2 + 3 = 5\nStep 2: 5 * 4 = 20"
        correct, total = verify_arithmetic_chain(text)
        self.assertGreaterEqual(total, 1)
        self.assertEqual(correct, total)

    def test_one_wrong_step(self) -> None:
        text = "Step: 2 + 3 = 6"
        correct, total = verify_arithmetic_chain(text)
        self.assertEqual(total, 1)
        self.assertEqual(correct, 0)

    def test_no_arithmetic_returns_zero(self) -> None:
        correct, total = verify_arithmetic_chain("This response has no arithmetic.")
        self.assertEqual(total, 0)
        self.assertEqual(correct, 0)


if __name__ == "__main__":
    unittest.main()
