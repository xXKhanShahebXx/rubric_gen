"""
Sympy-based symbolic math equivalence and arithmetic-chain checks.

The existing :mod:`rubric_gen.compiled.math_independent_solver_verifier` compares the LLM-derived
canonical answer against candidate-extracted canonical strings. Pure string equality misses
canonical-form equivalences such as ``4/8`` vs ``1/2`` vs ``0.5`` vs ``\\frac{1}{2}``.

This module provides:

- :func:`try_sympify`: tolerant parsing of free-form math strings, with LaTeX-aware
  preprocessing for the most common patterns (``\\frac{a}{b}``, ``\\sqrt{x}``, ``\\boxed{x}``).
- :func:`equivalent`: checks if two sympy expressions are equal under simplification.
- :func:`equivalent_strings`: convenience that composes ``try_sympify`` + ``equivalent``.
- :func:`verify_arithmetic_chain`: walks ``LHS = RHS`` patterns in a response, checks each
  with sympy, returns a (correct, total) tuple.

All sympy calls are wrapped in ``try/except`` and a 1.5s wall-clock guard so a malformed
expression cannot stall the verifier.
"""

from __future__ import annotations

import re
import signal
import threading
from typing import Optional, Tuple

try:
    import sympy as _sp
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    _SYMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _SYMPY_AVAILABLE = False


_LATEX_FRAC_RE = re.compile(r"\\(?:d?frac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_LATEX_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_LATEX_TEXT_RE = re.compile(r"\\text\s*\{[^{}]*\}")
_DOLLAR_RE = re.compile(r"\$+")


def _preprocess_latex(text: str) -> str:
    if not text:
        return ""
    out = text.replace("\r\n", "\n").strip()
    boxed = _LATEX_BOXED_RE.search(out)
    if boxed:
        out = boxed.group(1).strip()
    out = _DOLLAR_RE.sub(" ", out)
    out = _LATEX_TEXT_RE.sub(" ", out)

    def _replace_frac(match: "re.Match[str]") -> str:
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f"(({num})/({den}))"

    out = _LATEX_FRAC_RE.sub(_replace_frac, out)
    out = _LATEX_SQRT_RE.sub(lambda m: f"sqrt({m.group(1).strip()})", out)
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("\\pi", "pi")
    out = out.replace("\\left", "").replace("\\right", "")
    out = re.sub(r"\\[A-Za-z]+", " ", out)
    out = out.replace("{", "(").replace("}", ")")
    out = re.sub(r"\^", "**", out)
    out = re.sub(r"[A-Za-z]\(", lambda m: m.group(0)[:-1] + "(", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _run_with_timeout(func, args=(), kwargs=None, timeout_s: float = 1.5):
    """Run ``func`` in a daemon thread with a wall-clock timeout. Returns ``(ok, result)``."""
    kwargs = kwargs or {}
    result_holder = {"ok": False, "result": None, "error": None}

    def runner():
        try:
            result_holder["result"] = func(*args, **kwargs)
            result_holder["ok"] = True
        except Exception as exc:
            result_holder["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout_s))
    if thread.is_alive():
        return False, None
    return bool(result_holder["ok"]), result_holder["result"]


def try_sympify(value: str, *, timeout_s: float = 1.5):
    """
    Best-effort parse of a free-form math string into a sympy expression.

    Returns ``None`` if sympy is unavailable, the input is empty, or parsing fails / times out.
    """
    if not _SYMPY_AVAILABLE or not value:
        return None
    cleaned = _preprocess_latex(value)
    if not cleaned:
        return None

    def _do_parse():
        return parse_expr(
            cleaned,
            transformations=standard_transformations + (implicit_multiplication_application,),
        )

    ok, result = _run_with_timeout(_do_parse, timeout_s=timeout_s)
    if not ok:
        return None
    return result


def equivalent(a, b, *, timeout_s: float = 1.5) -> bool:
    """Return True if ``a`` and ``b`` simplify to the same value."""
    if not _SYMPY_AVAILABLE or a is None or b is None:
        return False

    def _do_check():
        diff = _sp.simplify(a - b)
        return bool(diff == 0) or bool(getattr(diff, "is_zero", False))

    ok, result = _run_with_timeout(_do_check, timeout_s=timeout_s)
    return bool(ok and result)


def equivalent_strings(a: str, b: str, *, timeout_s: float = 1.5) -> bool:
    """Convenience: parse both strings then check equivalence."""
    if not a or not b:
        return False
    if a.strip().lower() == b.strip().lower():
        return True
    expr_a = try_sympify(a, timeout_s=timeout_s)
    expr_b = try_sympify(b, timeout_s=timeout_s)
    if expr_a is None or expr_b is None:
        return False
    return equivalent(expr_a, expr_b, timeout_s=timeout_s)


_ARITH_LINE_RE = re.compile(
    r"([0-9A-Za-z\.\(\)\s\+\-\*\/\^\\\{\}_]+?)\s*(?:=|\\\\=|\\equiv|\\approx)\s*"
    r"([0-9A-Za-z\.\(\)\s\+\-\*\/\^\\\{\}_]+?)(?=\s*(?:[\.,;\n]|$))"
)


def verify_arithmetic_chain(response_text: str, *, max_pairs: int = 24) -> Tuple[int, int]:
    """
    Walk the response for ``LHS = RHS`` patterns and check each with sympy. Returns
    ``(correct_count, total_count)``. Only counts pairs where both sides parse with sympy.
    """
    if not _SYMPY_AVAILABLE or not response_text:
        return 0, 0
    text = response_text.replace("\\$", "")
    found = _ARITH_LINE_RE.findall(text)
    correct = 0
    total = 0
    for raw_lhs, raw_rhs in found[:max_pairs]:
        lhs = raw_lhs.strip()
        rhs = raw_rhs.strip()
        if not lhs or not rhs:
            continue
        expr_l = try_sympify(lhs, timeout_s=1.0)
        expr_r = try_sympify(rhs, timeout_s=1.0)
        if expr_l is None or expr_r is None:
            continue
        total += 1
        if equivalent(expr_l, expr_r, timeout_s=1.0):
            correct += 1
    return correct, total
