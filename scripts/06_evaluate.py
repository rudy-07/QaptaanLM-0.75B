"""Evaluation script for baseline or CPT model.

Usage:
    python scripts/06_evaluate.py --model-path models/Qwen3.5-0.8B-Base
    python scripts/06_evaluate.py --compare --base models/Qwen3.5-0.8B-Base --cpt checkpoints/cpt/final
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.benchmarks import (
    HUMANEVAL_SAMPLE_PROMPTS,
    MATH_REASONING_PROMPTS,
    evaluate_generation,
    compute_perplexity,
)
from src.evaluation.compare import compare_models
from src.training.utils import load_model_for_training, detect_hardware
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 Models")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3.5-0.8B-Base", help="Model path/ID to evaluate")
    parser.add_argument("--compare", action="store_true", help="Run comparison mode between base and CPT model")
    parser.add_argument("--base", type=str, default="Qwen/Qwen3.5-0.8B-Base", help="Base model path for comparison")
    parser.add_argument("--cpt", type=str, default=None, help="CPT model path for comparison")
    parser.add_argument("--output", type=str, default="logs/eval_report.json", help="Path to save evaluation report")
    args = parser.parse_args()

    setup_logging(log_dir=str(project_root / "logs"), log_name="evaluation")
    hw = detect_hardware()
    device = "cuda" if hw.get("device") == "cuda" else "cpu"

    if args.compare:
        if not args.cpt:
            logger.error("--cpt path is required when using --compare")
            return
        logger.info(f"Running model comparison: Base={args.base} vs CPT={args.cpt}")
        report = compare_models(args.base, args.cpt, output_path=args.output, device=device)
        print(f"Comparison report written to {args.output}")
        return

    logger.info(f"Evaluating model: {args.model_path} on {device}")
    model, tokenizer = load_model_for_training(
        args.model_path,
        dtype="float32" if device == "cpu" else "bfloat16",
        gradient_checkpointing=False,
        strip_vision=True,
    )
    model = model.to(device)

    # 1. Evaluate coding prompts
    logger.info("Evaluating coding prompts...")
    code_results = evaluate_generation(model, tokenizer, HUMANEVAL_SAMPLE_PROMPTS, device=device)

    # 2. Evaluate math prompts
    logger.info("Evaluating math prompts...")
    math_results = evaluate_generation(model, tokenizer, MATH_REASONING_PROMPTS, device=device)

    report = {
        "model": args.model_path,
        "device": device,
        "code_eval": code_results,
        "math_eval": math_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("Sample Coding Output:")
    for r in code_results[:2]:
        logger.info(f"\n--- Task {r.get('task_id')} ---\n{r.get('prompt')}{r.get('completion')}\n")

    logger.info(f"Full evaluation report saved to {args.output}")


if __name__ == "__main__":
    main()
