"""Orchestrator for EleutherAI LM Evaluation Harness (lm-eval).

Standard industry tool used for:
- MMLU (5-shot)
- MMLU-Pro (5-shot CoT)
- MMLU-Redux
- ARC-Challenge (25-shot)
- HellaSwag (10-shot)
- Winogrande (5-shot)
- TruthfulQA (MC2)
- BBH (Big-Bench Hard 3-shot CoT)
- GPQA (Diamond / Main)
- GSM8K (5-shot CoT)
- MATH (Hendrycks 4-shot CoT)
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Mapping from standard benchmark name to official lm-eval task identifier
LM_EVAL_TASK_MAP = {
    "mmlu": "mmlu",
    "mmlu_pro": "mmlu_pro",
    "mmlu_redux": "mmlu_redux",
    "arc_challenge": "arc_challenge",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "truthfulqa": "truthfulqa_mc2",
    "bbh": "bbh_cot_fewshot",
    "gpqa": "gpqa_diamond_zeroshot",
    "gsm8k": "gsm8k_cot",
    "math": "hendrycks_math",
}


def build_lm_eval_command(
    model_path_or_id: str,
    tasks: List[str],
    batch_size: Union[int, str] = "auto",
    device: str = "cuda:0",
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    dtype: str = "bfloat16",
) -> List[str]:
    """Builds the official lm-eval CLI command line."""
    task_keys = [LM_EVAL_TASK_MAP.get(t.lower(), t) for t in tasks]
    tasks_str = ",".join(task_keys)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path_or_id},dtype={dtype},trust_remote_code=True",
        "--tasks", tasks_str,
        "--batch_size", str(batch_size),
        "--device", device,
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if output_path is not None:
        cmd.extend(["--output_path", output_path])

    return cmd


def run_lm_eval_programmatic(
    model_path_or_id: str,
    tasks: List[str],
    limit: Optional[int] = None,
    batch_size: Union[int, str] = "auto",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> Dict[str, Any]:
    """Runs lm-eval programmatically via Python API if installed, or CLI fallback."""
    task_keys = [LM_EVAL_TASK_MAP.get(t.lower(), t) for t in tasks]

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        logger.info(f"Running programmatic lm-eval on {model_path_or_id} for tasks: {task_keys}")
        hflm = HFLM(
            pretrained=model_path_or_id,
            dtype=dtype,
            trust_remote_code=True,
            device=device,
            batch_size=batch_size,
        )

        eval_results = lm_eval.simple_evaluate(
            model=hflm,
            tasks=task_keys,
            limit=limit,
        )

        return parse_lm_eval_results(eval_results)
    except ImportError:
        logger.info("lm_eval Python package not directly importable. Falling back to subprocess CLI.")
        cmd = build_lm_eval_command(
            model_path_or_id=model_path_or_id,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            limit=limit,
            dtype=dtype,
        )
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"lm-eval failed: {res.stderr}")
        return {"raw_output": res.stdout}


def parse_lm_eval_results(results_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts clean scores from raw lm-eval output dictionary."""
    parsed = {}
    raw_results = results_dict.get("results", {})

    for task_name, metrics in raw_results.items():
        # Look for accuracy or acc_norm
        score = None
        for key in ["acc,none", "acc_norm,none", "exact_match,none", "acc", "exact_match"]:
            if key in metrics:
                score = metrics[key] * 100.0 if metrics[key] <= 1.0 else metrics[key]
                break

        parsed[task_name] = {
            "score": round(score, 2) if score is not None else None,
            "raw_metrics": metrics,
        }

    return parsed
