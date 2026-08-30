"""Master CLI Entrypoint for Comprehensive LLM Benchmarks.

Runs standardized evaluation across Coding and Intelligence/Reasoning suites:
- Coding: HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench, BigCodeBench
- Reasoning: MMLU, MMLU-Pro, MMLU-Redux, ARC-Challenge, HellaSwag, Winogrande, TruthfulQA, BBH, GPQA, GSM8K, MATH

Usage:
    python -m benchmarks.scripts.run_all_benchmarks --model kaptaan45/QaptaanLM-0.75B --suite all
    python -m benchmarks.scripts.run_all_benchmarks --model Qwen/Qwen2.5-Coder-0.5B --suite coding --limit 20
    python -m benchmarks.scripts.run_all_benchmarks --model kaptaan45/QaptaanLM-0.75B --tasks humaneval,gsm8k,mmlu
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.core.model_loader import BenchmarkModelWrapper
from benchmarks.core.report_generator import (
    generate_html_report,
    generate_markdown_report,
)
from benchmarks.runners.standalone_runner import StandaloneBenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_all_benchmarks")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Comprehensive LLM Benchmark Suite")
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
        "--suite",
        type=str,
        choices=["all", "coding", "reasoning", "quick"],
        default="all",
        help="Benchmark suite category to evaluate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated explicit list of tasks (e.g. humaneval,mbpp,gsm8k,mmlu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test samples per task (useful for smoke tests)",
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
        default="metrics/results",
        help="Directory to save raw benchmark results and summary reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting Benchmark Suite for: {args.model}")
    logger.info(f"Suite: {args.suite} | Tasks: {args.tasks} | Limit: {args.limit}")

    # Determine tasks to run
    if args.tasks:
        target_tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    elif args.suite == "coding":
        target_tasks = ["humaneval", "humaneval_plus", "mbpp", "mbpp_plus"]
    elif args.suite == "reasoning":
        target_tasks = ["mmlu", "arc_challenge", "hellaswag", "winogrande", "truthfulqa", "gsm8k", "math"]
    elif args.suite == "quick":
        target_tasks = ["humaneval", "mbpp", "gsm8k", "mmlu", "arc_challenge", "hellaswag"]
        if args.limit is None:
            args.limit = 10
    else:  # all
        target_tasks = [
            "humaneval",
            "humaneval_plus",
            "mbpp",
            "mbpp_plus",
            "mmlu",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "truthfulqa",
            "gsm8k",
            "math",
        ]

    # Load Model Wrapper
    wrapper = BenchmarkModelWrapper(
        model_path_or_id=args.model,
        device=args.device,
        dtype=args.dtype,
    )
    runner = StandaloneBenchmarkRunner(wrapper)

    task_results = {}

    try:
        for task in target_tasks:
            logger.info(f"\n========================================================")
            logger.info(f"Executing: {task.upper()}")
            logger.info(f"========================================================")

            if task == "humaneval":
                res = runner.run_humaneval(plus=False, limit=args.limit)
            elif task == "humaneval_plus":
                res = runner.run_humaneval(plus=True, limit=args.limit)
            elif task == "mbpp":
                res = runner.run_mbpp(plus=False, sanitized=True, limit=args.limit)
            elif task == "mbpp_plus":
                res = runner.run_mbpp(plus=True, limit=args.limit)
            elif task == "gsm8k":
                res = runner.run_gsm8k(limit=args.limit)
            elif task == "math":
                res = runner.run_math(limit=args.limit)
            elif task == "mmlu":
                res = runner.run_mmlu(limit=args.limit)
            elif task == "arc_challenge":
                res = runner.run_arc_challenge(limit=args.limit)
            elif task == "hellaswag":
                res = runner.run_hellaswag(limit=args.limit)
            elif task == "winogrande":
                res = runner.run_winogrande(limit=args.limit)
            elif task == "truthfulqa":
                res = runner.run_truthfulqa(limit=args.limit)
            else:
                logger.warning(f"Unknown task: {task}. Skipping.")
                continue

            task_results[res["task"]] = res
            logger.info(f"✅ {res['task']} Completed: {res['score']:.2f}% ({res['passed']}/{res['total']}) in {res['elapsed_sec']}s")

    finally:
        wrapper.unload()

    # Load reference baselines
    ref_baselines_path = Path(__file__).resolve().parent.parent / "reference_baselines" / "published_scores.json"
    ref_baselines = {}
    if ref_baselines_path.exists():
        with open(ref_baselines_path, "r", encoding="utf-8") as f:
            ref_baselines = json.load(f).get("models", {})

    # Save output artifacts
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = args.model.replace("/", "_")

    payload = {
        "model": args.model,
        "timestamp": timestamp,
        "suite": args.suite,
        "limit": args.limit,
        "tasks": task_results,
    }

    # 1. JSON
    json_path = out_dir / f"benchmark_{clean_model_name}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved benchmark JSON: {json_path}")

    # 2. Markdown & HTML
    eval_dict = {args.model: {"tasks": task_results}}
    md_content = generate_markdown_report(
        eval_results=eval_dict,
        baseline_model=args.baseline_model if args.baseline_model != args.model else None,
        reference_baselines=ref_baselines,
        title=f"Benchmark Evaluation: {args.model}",
    )
    md_path = out_dir / f"report_{clean_model_name}_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved benchmark Markdown report: {md_path}")

    html_path = out_dir / f"report_{clean_model_name}_{timestamp}.html"
    generate_html_report(
        eval_results=eval_dict,
        output_path=html_path,
        baseline_model=args.baseline_model,
        reference_baselines=ref_baselines,
    )
    logger.info(f"Saved benchmark HTML report: {html_path}")

    print("\n" + md_content)


if __name__ == "__main__":
    main()
