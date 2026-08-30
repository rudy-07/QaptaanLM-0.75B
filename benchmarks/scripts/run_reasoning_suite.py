"""Dedicated CLI for Running Reasoning, Math & Intelligence Benchmarks.

Evaluates models on:
- MMLU (Massive Multitask Language Understanding)
- MMLU-Pro (10-option complex reasoning)
- MMLU-Redux (Cleaned benchmark)
- ARC-Challenge (Advanced reasoning science questions)
- HellaSwag & Winogrande (Commonsense reasoning)
- TruthfulQA (Factuality & safety)
- BBH (BIG-Bench Hard 3-shot CoT)
- GPQA (Graduate PhD-level reasoning)
- GSM8K (Math word problems 5-shot CoT)
- MATH (Hendrycks competition math 4-shot CoT)

Usage:
    python -m benchmarks.scripts.run_reasoning_suite --model kaptaan45/QaptaanLM-0.75B
    python -m benchmarks.scripts.run_reasoning_suite --model Qwen/Qwen2.5-Coder-0.5B --tasks mmlu,gsm8k,math
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
    parser = argparse.ArgumentParser(description="Run Reasoning & Intelligence Benchmark Suite")
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
        default="mmlu,arc_challenge,hellaswag,winogrande,truthfulqa,gsm8k,math",
        help="Comma-separated tasks",
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
        default="metrics/results/reasoning",
        help="Directory to save raw benchmark results and summary reports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Forward arguments to main benchmark runner
    sys.argv = [
        sys.argv[0],
        "--suite", "reasoning",
    ] + sys.argv[1:]
    run_benchmarks_main()
