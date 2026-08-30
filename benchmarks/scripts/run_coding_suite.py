"""Dedicated CLI for Running Coding Benchmarks.

Evaluates models on:
- HumanEval (OpenAI pass@1)
- HumanEval+ (EvalPlus amplified test suite)
- MBPP (Mostly Basic Python Problems)
- MBPP+ (EvalPlus amplified test suite)
- LiveCodeBench (Contest code generation)
- BigCodeBench (Complex multi-library tasks)

Usage:
    python -m benchmarks.scripts.run_coding_suite --model kaptaan45/QaptaanLM-0.75B
    python -m benchmarks.scripts.run_coding_suite --model Qwen/Qwen2.5-Coder-0.5B --limit 50
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.scripts.run_all_benchmarks import main as run_benchmarks_main


def parse_args():
    parser = argparse.ArgumentParser(description="Run Coding Benchmark Suite")
    parser.add_argument(
        "--model",
        type=str,
        default="kaptaan45/QaptaanLM-0.75B",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Baseline model ID for comparative delta reporting",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="humaneval,humaneval_plus,mbpp,mbpp_plus",
        help="Comma-separated coding tasks (humaneval, humaneval_plus, mbpp, mbpp_plus)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test samples per task",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda, cpu, cuda:0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type precision (bfloat16, float16, float32, auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metrics/results/coding",
        help="Directory to save raw benchmark results and summary reports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Forward arguments to main benchmark runner
    sys.argv = [
        sys.argv[0],
        "--suite", "coding",
    ] + sys.argv[1:]
    run_benchmarks_main()
