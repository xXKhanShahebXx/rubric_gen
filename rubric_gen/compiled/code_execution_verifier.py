"""
Subprocess-based code-execution verifier for ``livecodebench`` JudgeBench examples.

The rubric judge cannot tell whether candidate code is *correct* — it only sees text. LiveCodeBench
problems include visible example I/O blocks in the prompt. This verifier extracts those examples,
extracts the candidate's Python code, runs it in an isolated Python subprocess against each
example, and produces a pass-rate per response. When one candidate's pass-rate exceeds the
other's by a meaningful margin, the verifier fires with high confidence.

The verifier never returns a guess: if either candidate has no extractable code, or no visible
I/O pairs are found in the prompt, the verifier abstains.

Sandboxing notes:

- ``python -I`` runs in *isolated* mode (no PYTHON* env vars, no user site-packages). That
  prevents accidentally importing user-installed mutable packages.
- The candidate code is written to a fresh ``NamedTemporaryFile`` and deleted on exit.
- ``subprocess.run`` is called with ``timeout`` per case (default 10s) and ``capture_output``;
  the test stdin (if any) is fed via ``input=``.
- We do NOT attempt to constrain network access at the OS level. JudgeBench candidate code is not
  adversarial; the worst realistic case is a slow / non-terminating program, which the timeout
  handles. If we ever apply this to untrusted code, swap to a Docker / nsjail layer.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class IOExample:
    inputs: str
    outputs: str


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    timed_out: bool
    return_code: int = 0


@dataclass(frozen=True)
class CandidatePassReport:
    code_extracted: bool
    code_text: str
    pass_count: int
    total_cases: int
    case_details: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def pass_rate(self) -> float:
        if self.total_cases <= 0:
            return 0.0
        return self.pass_count / float(self.total_cases)


@dataclass(frozen=True)
class CodeExecutionVerifierOutcome:
    triggered: bool
    recommended_decision: str
    confidence: str
    reason: str
    a_report: Optional[CandidatePassReport]
    b_report: Optional[CandidatePassReport]
    io_pair_count: int
    schema: str = "code_execution_verifier_outcome_v1"


_INPUT_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:Sample\s+Input|Example\s+Input|Test\s+Input|Input)\s*\d*\s*[:.\-]?\s*\n",
    re.IGNORECASE,
)
_OUTPUT_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:Sample\s+Output|Example\s+Output|Expected\s+Output|Test\s+Output|Output)\s*\d*\s*[:.\-]?\s*\n",
    re.IGNORECASE,
)
_TRAILER_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:Explanation|Note|Constraints?|Example|Sample|Test|Input|Output)\s*\d*\s*[:.\-]?\s*\n",
    re.IGNORECASE,
)


def _strip_blockquote(value: str) -> str:
    if not value:
        return ""
    text = value.strip()
    text = re.sub(r"^[`>\s]+", "", text)
    text = re.sub(r"[`>\s]+$", "", text)
    return text.strip()


def _find_header_match(text: str, regex: "re.Pattern[str]", start: int = 0):
    """Search for the next header match at or after ``start``, returning the match or None."""
    if not text:
        return None
    return regex.search(text, start)


def extract_visible_io_pairs(question: str, *, max_pairs: int = 8) -> List[IOExample]:
    """
    Extract visible Input/Output example pairs from a LiveCodeBench-style question prompt.

    Walks the prompt left-to-right, alternately finding Input and Output header matches. The
    block between an Input header and its following Output header becomes ``inputs``; the block
    between that Output header and the next Input/trailer header becomes ``outputs``. Handles
    both the LeetCode-style ``Example 1: Input: ... Output: ...`` format and the AtCoder-style
    ``Sample Input 1\\n...\\nSample Output 1\\n...`` format.
    """
    if not question:
        return []
    text = question
    pairs: List[IOExample] = []
    cursor = 0
    while len(pairs) < max_pairs:
        in_match = _find_header_match(text, _INPUT_HEADER_RE, cursor)
        if not in_match:
            break
        out_match = _find_header_match(text, _OUTPUT_HEADER_RE, in_match.end())
        if not out_match:
            break
        inputs_block = text[in_match.end():out_match.start()]
        next_in_match = _find_header_match(text, _INPUT_HEADER_RE, out_match.end())
        next_trailer = _find_header_match(text, _TRAILER_HEADER_RE, out_match.end())
        end_idx = len(text)
        candidate_ends = [m.start() for m in (next_in_match, next_trailer) if m is not None]
        if candidate_ends:
            end_idx = min(candidate_ends)
        outputs_block = text[out_match.end():end_idx]
        inputs_clean = _strip_blockquote(inputs_block)
        outputs_clean = _strip_blockquote(outputs_block)
        if inputs_clean or outputs_clean:
            pairs.append(IOExample(inputs=inputs_clean, outputs=outputs_clean))
        cursor = out_match.end()
    return pairs


def extract_candidate_code(response_text: str) -> str:
    """
    Pull Python code out of a candidate response. Prefer triple-backtick code fences; fall back
    to indented blocks following ``def``/``class`` lines; otherwise return empty string.
    """
    if not response_text:
        return ""
    fences = _CODE_FENCE_RE.findall(response_text)
    if fences:
        return max(fences, key=len).strip()
    lines = response_text.splitlines()
    blocks: List[List[str]] = []
    current: List[str] = []
    in_block = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "import ", "from ")) and (
            line == stripped or len(line) - len(stripped) == 0
        ):
            if current:
                blocks.append(current)
            current = [line]
            in_block = True
            continue
        if in_block:
            if line.strip() == "" or line.startswith((" ", "\t")):
                current.append(line)
            else:
                blocks.append(current)
                current = []
                in_block = False
    if current:
        blocks.append(current)
    if not blocks:
        return ""
    longest = max(blocks, key=lambda b: sum(len(line) for line in b))
    return "\n".join(longest).strip()


def run_in_sandbox(
    code: str,
    *,
    stdin: str = "",
    timeout_s: float = 10.0,
    python_executable: Optional[str] = None,
) -> ExecutionResult:
    """
    Run the supplied code as a fresh isolated Python subprocess.

    The code is written to a temp file (deleted on exit). We capture stdout/stderr and bounded
    by ``timeout_s``. Network is not blocked at OS level — see module docstring.
    """
    if not code or not code.strip():
        return ExecutionResult(success=False, stdout="", stderr="no_code", timed_out=False)
    executable = python_executable or sys.executable
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as fh:
            fh.write(code)
            tmp_path = Path(fh.name)
        try:
            result = subprocess.run(
                [executable, "-I", "-X", "utf8", str(tmp_path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, stdout="", stderr="timeout", timed_out=True)
        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            timed_out=False,
            return_code=int(result.returncode),
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _normalize_output(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned


def _outputs_match(actual: str, expected: str) -> bool:
    a = _normalize_output(actual)
    b = _normalize_output(expected)
    if a == b:
        return True
    a_tokens = a.split()
    b_tokens = b.split()
    if a_tokens == b_tokens:
        return True
    if not b_tokens:
        return False
    return a_tokens[: len(b_tokens)] == b_tokens


def evaluate_candidate_code(
    code: str,
    io_pairs: List[IOExample],
    *,
    timeout_s: float = 10.0,
    python_executable: Optional[str] = None,
) -> CandidatePassReport:
    if not code or not io_pairs:
        return CandidatePassReport(
            code_extracted=bool(code),
            code_text=code or "",
            pass_count=0,
            total_cases=len(io_pairs),
            case_details=tuple(),
        )
    case_details: List[Dict[str, Any]] = []
    pass_count = 0
    for idx, io_pair in enumerate(io_pairs):
        result = run_in_sandbox(
            code,
            stdin=io_pair.inputs,
            timeout_s=timeout_s,
            python_executable=python_executable,
        )
        passed = result.success and _outputs_match(result.stdout, io_pair.outputs)
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
        code_text=code,
        pass_count=pass_count,
        total_cases=len(io_pairs),
        case_details=tuple(case_details),
    )


def evaluate_code_pair_verifier(
    *,
    question: str,
    response_a: str,
    response_b: str,
    timeout_s: float = 10.0,
    python_executable: Optional[str] = None,
    min_margin: float = 0.34,
) -> CodeExecutionVerifierOutcome:
    """
    Pair-level code-execution verifier. Returns a triggered / not triggered outcome with a
    recommended A>B / B>A decision when one candidate clearly out-performs the other on the
    visible test cases.
    """
    io_pairs = extract_visible_io_pairs(question)
    if not io_pairs:
        return CodeExecutionVerifierOutcome(
            triggered=False,
            recommended_decision="",
            confidence="",
            reason="no_visible_io_pairs",
            a_report=None,
            b_report=None,
            io_pair_count=0,
        )
    code_a = extract_candidate_code(response_a)
    code_b = extract_candidate_code(response_b)
    if not code_a or not code_b:
        return CodeExecutionVerifierOutcome(
            triggered=False,
            recommended_decision="",
            confidence="",
            reason="missing_candidate_code",
            a_report=None,
            b_report=None,
            io_pair_count=len(io_pairs),
        )
    report_a = evaluate_candidate_code(
        code_a,
        io_pairs,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )
    report_b = evaluate_candidate_code(
        code_b,
        io_pairs,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )
    margin = abs(report_a.pass_rate - report_b.pass_rate)
    if margin < float(min_margin):
        return CodeExecutionVerifierOutcome(
            triggered=False,
            recommended_decision="",
            confidence="low",
            reason="insufficient_pass_rate_margin",
            a_report=report_a,
            b_report=report_b,
            io_pair_count=len(io_pairs),
        )
    decision = "A>B" if report_a.pass_rate > report_b.pass_rate else "B>A"
    confidence = "high" if margin >= 0.5 else "medium"
    reason = (
        "a_passes_more_visible_cases"
        if decision == "A>B"
        else "b_passes_more_visible_cases"
    )
    return CodeExecutionVerifierOutcome(
        triggered=True,
        recommended_decision=decision,
        confidence=confidence,
        reason=reason,
        a_report=report_a,
        b_report=report_b,
        io_pair_count=len(io_pairs),
    )
