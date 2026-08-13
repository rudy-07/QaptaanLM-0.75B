"""Model comparison utility: compare base model vs fine-tuned CPT model.

Runs identical prompts through both models and outputs side-by-side
comparisons of responses, token speed, and perplexity.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from src.evaluation.benchmarks import (
    HUMANEVAL_SAMPLE_PROMPTS,
    MATH_REASONING_PROMPTS,
    evaluate_generation,
    compute_perplexity,
)
from src.training.utils import load_model_for_training

logger = logging.getLogger(__name__)


def compare_models(
    base_model_path: str,
    cpt_model_path: str,
    output_path: Optional[str] = "logs/comparison_report.json",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run comparative evaluation on Base vs CPT model.

    Args:
        base_model_path: Path/ID to base model.
        cpt_model_path: Path/ID to CPT model.
        output_path: Where to save comparison report JSON.
        device: "cuda" or "cpu".

    Returns:
        Summary dictionary with comparison results.
    """
    logger.info("=== Loading Base Model ===")
    base_model, base_tok = load_model_for_training(
        base_model_path,
        dtype="float32" if device == "cpu" else "bfloat16",
        gradient_checkpointing=False,
        strip_vision=True,
    )
    base_model = base_model.to(device)

    logger.info("Evaluating Base model on coding prompts...")
    base_code_res = evaluate_generation(base_model, base_tok, HUMANEVAL_SAMPLE_PROMPTS, device=device)

    logger.info("Evaluating Base model on math prompts...")
    base_math_res = evaluate_generation(base_model, base_tok, MATH_REASONING_PROMPTS, device=device)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=== Loading CPT Model ===")
    cpt_model, cpt_tok = load_model_for_training(
        cpt_model_path,
        dtype="float32" if device == "cpu" else "bfloat16",
        gradient_checkpointing=False,
        strip_vision=True,
    )
    cpt_model = cpt_model.to(device)

    logger.info("Evaluating CPT model on coding prompts...")
    cpt_code_res = evaluate_generation(cpt_model, cpt_tok, HUMANEVAL_SAMPLE_PROMPTS, device=device)

    logger.info("Evaluating CPT model on math prompts...")
    cpt_math_res = evaluate_generation(cpt_model, cpt_tok, MATH_REASONING_PROMPTS, device=device)

    del cpt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compile side-by-side comparison
    code_comparisons = []
    for b, c in zip(base_code_res, cpt_code_res):
        code_comparisons.append({
            "task_id": b.get("task_id"),
            "prompt": b.get("prompt"),
            "base_completion": b.get("completion"),
            "cpt_completion": c.get("completion"),
            "base_elapsed": b.get("elapsed_seconds"),
            "cpt_elapsed": c.get("elapsed_seconds"),
        })

    math_comparisons = []
    for b, c in zip(base_math_res, cpt_math_res):
        math_comparisons.append({
            "id": b.get("id"),
            "prompt": b.get("prompt"),
            "expected": b.get("expected"),
            "base_completion": b.get("completion"),
            "cpt_completion": c.get("completion"),
        })

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": base_model_path,
        "cpt_model": cpt_model_path,
        "code_comparisons": code_comparisons,
        "math_comparisons": math_comparisons,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Comparison report saved to {output_path}")

    return report
