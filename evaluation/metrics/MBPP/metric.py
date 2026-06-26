"""MBPP metric: pass@1 via subprocess code execution.

For each sample the metric:
1. Extracts the Python code block from the model's response.
2. Runs `python -c "<code>\\n<test_cases>"` in a subprocess with a timeout.
3. Reports pass@1 = fraction of problems where all tests pass.
"""
import json
import os
import re
import sys
import tempfile
import subprocess
import pandas as pd

from evaluation.metrics.base_metric import BaseMetric


# ──────────────────────────────────────────────
# Code extraction helpers
# ──────────────────────────────────────────────

def _unwrap_response(response: str) -> str:
    """If response is a JSON-wrapped assistant message, extract the text content."""
    if not response or not response.strip().startswith("{"):
        return response
    try:
        obj = json.loads(response)
        # {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        content = obj.get("content", [])
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            return "\n".join(parts)
        if isinstance(content, str):
            return content
    except (json.JSONDecodeError, AttributeError):
        pass
    return response


def _extract_code(response: str) -> str:
    """Extract Python code from model response.

    Handles (in priority order):
    1. JSON-wrapped assistant message  ->  unwrap first
    2. Fenced markdown:  ```python ... ``` or ``` ... ```
    3. [BEGIN] ... [DONE] / [END] delimiters
    4. Fallback: first def/import/class line onwards
    """
    if not isinstance(response, str):
        return ""

    # Unwrap JSON-wrapped responses before any pattern matching
    response = _unwrap_response(response)

    # 1. Markdown fenced block
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. [BEGIN] / [DONE|END] delimiters
    m = re.search(r"\[BEGIN\](.*?)(?:\[DONE\]|\[END\])", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. Fallback: truncate at first [DONE]/[END] (may appear mid-string when the
    # model continues generating after the delimiter), then extract from first
    # def/import/class line onwards.
    response = re.split(r"\[(?:DONE|END)\]", response)[0]

    lines = response.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("def ") or line.strip().startswith("import ") or line.strip().startswith("class "):
            return "\n".join(lines[i:]).strip()

    return response.strip()


# ──────────────────────────────────────────────
# Code execution helper
# ──────────────────────────────────────────────

def _run_tests(code: str, test_lines: list[str], timeout: int = 10) -> bool:
    """Execute code + test assertions in a subprocess.

    Returns True iff the subprocess exits with code 0.
    """
    if not code:
        return False

    full_source = code + "\n\n" + "\n".join(test_lines)

    # Write to a temp file so we get proper tracebacks
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_source)
        fname = f.name

    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


# ──────────────────────────────────────────────
# Metric class
# ──────────────────────────────────────────────

class MBPPMetric(BaseMetric):
    """MBPP pass@1: exact execution pass rate on 427 sanitized problems."""

    def __init__(self, dataset_name="mbpp", execution_timeout: int = 10, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.execution_timeout = execution_timeout
        self.results = []

    def load_model(self, logger=None):
        pass

    def process(self, answers, labels, subjects=None, **kwargs):
        """
        Args:
            answers:  list of model response strings.
            labels:   list of test-case strings (newline-joined assert lines).
            subjects: list of task_id strings.
        """
        if answers is None:
            return self.results
        subjects = subjects or ["mbpp"] * len(answers)
        for response, label_str, task_id in zip(answers, labels, subjects):
            test_lines = [l for l in label_str.splitlines() if l.strip()]
            code = _extract_code(response)
            passed = _run_tests(code, test_lines, timeout=self.execution_timeout)
            self.results.append({
                "task_id":  task_id,
                "response": response,
                "code":     code,
                "passed":   passed,
            })
        return self.results

    def compute_metrics(self, results):
        df = pd.DataFrame(results)
        total    = len(df)
        n_passed = df["passed"].sum()
        pass_at_1 = n_passed / total if total > 0 else 0.0
        return {
            "pass@1": (round(float(pass_at_1), 6), total),
        }
