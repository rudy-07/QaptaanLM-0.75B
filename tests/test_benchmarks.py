"""Unit tests for Comprehensive Benchmark Suite."""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.core.execution_sandbox import (
    extract_python_code,
    run_code_with_sandbox,
)
from benchmarks.core.metrics_calculator import (
    estimate_pass_at_k,
    extract_gsm8k_answer,
    extract_latex_boxed,
    extract_mcq_answer,
    is_math_answer_equal,
    normalize_math_answer,
)
from benchmarks.core.prompt_templates import (
    GSM8K_COT_PROMPT,
    MATH_COT_PROMPT,
    MBPP_3SHOT_PROMPT,
    format_mcq_prompt,
    format_mmlu_pro_prompt,
)
from benchmarks.core.report_generator import (
    format_delta,
    generate_markdown_report,
)
from benchmarks.datasets.loaders import (
    load_arc_challenge,
    load_gsm8k,
    load_hellaswag,
    load_humaneval,
    load_mbpp,
    load_mmlu,
)


class TestExecutionSandbox(unittest.TestCase):
    """Tests for safe code execution sandbox and Python extraction."""

    def test_extract_python_code(self):
        markdown_block = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nHope this helps!"
        extracted = extract_python_code(markdown_block)
        self.assertEqual(extracted, "def add(a, b):\n    return a + b")

    def test_sandbox_passing_code(self):
        code = "def add(a, b):\n    return a + b"
        test_code = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        result = run_code_with_sandbox(code, test_code, timeout=2.0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "passed")

    def test_sandbox_assertion_failure(self):
        code = "def add(a, b):\n    return a - b"
        test_code = "assert add(2, 3) == 5"
        result = run_code_with_sandbox(code, test_code, timeout=2.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "failed")

    def test_sandbox_syntax_error(self):
        code = "def invalid_syntax(:"
        test_code = "assert True"
        result = run_code_with_sandbox(code, test_code, timeout=2.0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "syntax_error")

    def test_sandbox_timeout(self):
        code = "import time\ndef infinite_loop():\n    while True:\n        time.sleep(0.1)"
        test_code = "infinite_loop()"
        result = run_code_with_sandbox(code, test_code, timeout=0.5)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "timeout")


class TestMetricsCalculator(unittest.TestCase):
    """Tests for metric calculations, answer extraction, and math normalization."""

    def test_pass_at_1_greedy(self):
        # 1 sample per problem, 8 correct out of 10
        score = estimate_pass_at_k(num_samples=[1]*10, num_correct=[1]*8 + [0]*2, k=1)
        self.assertAlmostEqual(score, 0.8)

    def test_gsm8k_answer_extraction(self):
        text1 = "Natalia sold 48 + 24 = 72 clips. #### 72"
        self.assertEqual(extract_gsm8k_answer(text1), "72")

        text2 = "Therefore, the total cost is $1,250.00."
        self.assertEqual(extract_gsm8k_answer(text2), "1250.00")

    def test_latex_boxed_extraction(self):
        text = "The solution simplifies to \\boxed{\\frac{3}{4}}."
        self.assertEqual(extract_latex_boxed(text), "\\frac{3}{4}")

    def test_math_answer_equality(self):
        self.assertTrue(is_math_answer_equal("72", "72"))
        self.assertTrue(is_math_answer_equal("72.0", "72"))
        self.assertTrue(is_math_answer_equal("$150", "150"))
        self.assertFalse(is_math_answer_equal("40", "50"))

    def test_mcq_answer_extraction(self):
        self.assertEqual(extract_mcq_answer("The answer is (B)"), "B")
        self.assertEqual(extract_mcq_answer("Answer: C. Because of physics"), "C")
        self.assertEqual(extract_mcq_answer("D"), "D")


class TestPromptTemplates(unittest.TestCase):
    """Tests for standard benchmark prompt formatting."""

    def test_mcq_prompt_formatting(self):
        prompt = format_mcq_prompt(
            question="What is 2+2?",
            choices=["3", "4", "5"],
            subject="elementary_math",
        )
        self.assertIn("elementary math", prompt)
        self.assertIn("(A) 3", prompt)
        self.assertIn("(B) 4", prompt)

    def test_mmlu_pro_formatting(self):
        prompt = format_mmlu_pro_prompt(
            question="Which layer is layer 4?",
            options=["Transport", "Network", "Physical"],
        )
        self.assertIn("(A) Transport", prompt)
        self.assertIn("Thought:", prompt)


class TestReportGenerator(unittest.TestCase):
    """Tests for report generator."""

    def test_format_delta(self):
        self.assertEqual(format_delta(40.0, 45.0), "+5.00%")
        self.assertEqual(format_delta(50.0, 45.0), "-5.00%")
        self.assertEqual(format_delta(40.0, 40.0), "0.00%")

    def test_markdown_report_generation(self):
        eval_results = {
            "BaseModel": {
                "tasks": {
                    "HumanEval": {"score": 30.0, "category": "Code", "metric": "pass@1"},
                    "GSM8K": {"score": 40.0, "category": "Math", "metric": "accuracy"},
                }
            },
            "CPTModel": {
                "tasks": {
                    "HumanEval": {"score": 35.0, "category": "Code", "metric": "pass@1"},
                    "GSM8K": {"score": 42.5, "category": "Math", "metric": "accuracy"},
                }
            },
        }
        report = generate_markdown_report(eval_results, baseline_model="BaseModel")
        self.assertIn("BaseModel", report)
        self.assertIn("CPTModel", report)
        self.assertIn("+5.00%", report)
        self.assertIn("+2.50%", report)


class TestDatasetLoaders(unittest.TestCase):
    """Tests fallback dataset loaders for zero-network testing."""

    def test_humaneval_loader(self):
        items = load_humaneval(limit=2, fallback_only=True)
        self.assertGreaterEqual(len(items), 1)
        self.assertIn("task_id", items[0])
        self.assertIn("prompt", items[0])
        self.assertIn("test", items[0])

    def test_mbpp_loader(self):
        items = load_mbpp(limit=2, fallback_only=True)
        self.assertGreaterEqual(len(items), 1)
        self.assertIn("task_id", items[0])
        self.assertIn("prompt", items[0])

    def test_gsm8k_loader(self):
        items = load_gsm8k(limit=2, fallback_only=True)
        self.assertGreaterEqual(len(items), 1)
        self.assertIn("question", items[0])
        self.assertIn("answer", items[0])


if __name__ == "__main__":
    unittest.main()
