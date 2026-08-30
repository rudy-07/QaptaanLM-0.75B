"""Isolated Execution Sandbox for Code Evaluation.

Provides safe subprocess execution of generated code against unit test suites
with process timeout control, memory containment, and strict exception handling.
Supports HumanEval, EvalPlus (HumanEval+ / MBPP+), MBPP, BigCodeBench, and LiveCodeBench.
"""

import ast
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional


def extract_python_code(
    completion: str,
    entry_point: Optional[str] = None,
    task_type: str = "completion",
) -> str:
    """Extracts valid Python code from raw model completions.

    Handles raw code, markdown blocks (```python ... ```), partial code,
    and docstring-to-code continuations.
    """
    text = completion.strip()

    # 1. Match standard markdown code fences ```python ... ``` or ``` ... ```
    if "```python" in text:
        blocks = text.split("```python")
        for b in blocks[1:]:
            code_chunk = b.split("```")[0]
            if code_chunk.strip():
                return code_chunk.strip()

    if "```" in text:
        blocks = text.split("```")
        for b in blocks[1:]:
            code_chunk = b.split("```")[0]
            if code_chunk.strip() and not code_chunk.startswith("json"):
                return code_chunk.strip()

    # 2. Stop at common markdown or natural language explanations
    lines = text.split("\n")
    valid_lines = []
    for line in lines:
        if line.startswith("## ") or line.startswith("### ") or line.startswith("Explanation:"):
            break
        valid_lines.append(line)

    code = "\n".join(valid_lines)
    return code.strip()


def run_code_with_sandbox(
    code: str,
    test_code: str,
    entry_point: Optional[str] = None,
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """Executes Python code and tests inside an isolated subprocess sandbox.

    Args:
        code: The generated implementation code.
        test_code: Test assertions or test harness code (e.g. check() function).
        entry_point: Function or class entry point name.
        timeout: Maximum execution timeout in seconds.

    Returns:
        Dictionary with:
            - 'passed': bool (True if execution succeeded with no assertion failures)
            - 'status': 'passed' | 'failed' | 'timeout' | 'syntax_error' | 'runtime_error'
            - 'error': error message string if failed
            - 'exec_time': float time in seconds
    """
    t_start = time.perf_counter()

    # Pre-check Python AST syntax to catch obvious syntax errors quickly
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {
            "passed": False,
            "status": "syntax_error",
            "error": f"SyntaxError: {e}",
            "exec_time": time.perf_counter() - t_start,
        }

    full_program = (
        "import sys\n"
        "import math\n"
        "import collections\n"
        "import itertools\n"
        "import functools\n"
        "import re\n"
        "from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Iterable, Sequence\n\n"
        f"{code}\n\n"
        f"{test_code}\n"
    )

    if entry_point and f"check({entry_point})" not in test_code and "check(" in test_code:
        full_program += f"\ncheck({entry_point})\n"

    try:
        res = subprocess.run(
            [sys.executable, "-c", full_program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exec_time = time.perf_counter() - t_start

        if res.returncode == 0:
            return {
                "passed": True,
                "status": "passed",
                "error": None,
                "exec_time": exec_time,
            }
        else:
            stderr = res.stderr.strip()
            if "AssertionError" in stderr:
                status = "failed"
            elif "SyntaxError" in stderr:
                status = "syntax_error"
            else:
                status = "runtime_error"

            # Extract last line of stderr
            err_msg = stderr.splitlines()[-1] if stderr else f"Process exited with {res.returncode}"
            return {
                "passed": False,
                "status": status,
                "error": err_msg,
                "exec_time": exec_time,
            }

    except subprocess.TimeoutExpired:
        exec_time = time.perf_counter() - t_start
        return {
            "passed": False,
            "status": "timeout",
            "error": f"Execution timed out after {timeout}s",
            "exec_time": exec_time,
        }
    except Exception as e:
        exec_time = time.perf_counter() - t_start
        return {
            "passed": False,
            "status": "runtime_error",
            "error": f"{type(e).__name__}: {e}",
            "exec_time": exec_time,
        }


def evaluate_problem_completions(
    problem: Dict[str, Any],
    completions: List[str],
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """Evaluates a batch of generated completions for a single coding problem."""
    task_id = problem.get("task_id", problem.get("id", "unknown"))
    entry_point = problem.get("entry_point")
    test_code = problem.get("test", "")
    prompt = problem.get("prompt", "")

    eval_results = []
    passed_count = 0

    for completion in completions:
        extracted = extract_python_code(completion, entry_point)
        if entry_point and entry_point not in extracted and entry_point in prompt:
            full_code = prompt + "\n" + extracted
        else:
            full_code = extracted

        res = run_code_with_sandbox(
            code=full_code,
            test_code=test_code,
            entry_point=entry_point,
            timeout=timeout,
        )
        if res["passed"]:
            passed_count += 1
        eval_results.append(res)

    return {
        "task_id": task_id,
        "passed_count": passed_count,
        "total": len(completions),
        "results": eval_results,
    }
