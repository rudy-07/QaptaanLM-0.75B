"""Stage 2 Post-SFT Evaluation & Qualitative Assessment Suite.

Evaluates instruction following, code generation, reasoning, and FIM preservation
across base, CPT, and post-SFT checkpoints.

Usage:
    # Evaluate SFT model directly
    python scripts/08_evaluate_sft.py --model checkpoints/jax_sft_hf
    
    # Compare CPT vs SFT side-by-side
    python scripts/08_evaluate_sft.py --model checkpoints/jax_sft_hf --baseline kaptaan45/QaptaanLM-0.75B
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Prevent broken torchvision binary mismatch crashes
try:
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    _iu.is_torchvision_available = lambda: False
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_sft")


EVAL_PROMPTS = [
    {
        "category": "Code Generation (Python)",
        "prompt": "Write a Python function `lru_cache_custom(maxsize: int)` that implements an LRU cache decorator from scratch using a doubly linked list and a hash map. Include type hints and docstrings.",
    },
    {
        "category": "Code Generation (TypeScript)",
        "prompt": "Write a generic TypeScript class `PriorityQueue<T>` with `push(item: T, priority: number): void`, `pop(): T | undefined`, `peek(): T | undefined`, and `size(): number` methods.",
    },
    {
        "category": "SQL Query",
        "prompt": "Given a table `Transactions(transaction_id, user_id, amount, transaction_time)`, write a SQL query to calculate the 7-day rolling average spend per user for each transaction date.",
    },
    {
        "category": "Debugging & Code Repair",
        "prompt": "Find and fix the bugs in this Python function:\n\n```python\ndef binary_search(arr, target):\n    low = 0\n    high = len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\n```\nExplain what went wrong and provide the corrected code.",
    },
    {
        "category": "Mathematical Reasoning (CoT)",
        "prompt": "A train travels from Station A to Station B at an average speed of 60 km/h, and returns from B to A at 40 km/h. What is the average speed of the train for the entire round trip? Show your step-by-step reasoning.",
    },
    {
        "category": "Technical Architecture QA",
        "prompt": "Explain the architectural difference between Grouped-Query Attention (GQA) and standard Multi-Head Attention (MHA). Why does GQA reduce memory bandwidth during autoregressive decoding?",
    },
    {
        "category": "Fill-in-the-Middle (FIM)",
        "prompt": "FIM",
        "fim_prefix": "def calculate_bmi(weight_kg: float, height_m: float) -> float:\n    \"\"\"Calculate Body Mass Index.\"\"\"\n    if height_m <= 0 or weight_kg <= 0:\n        raise ValueError(\"Height and weight must be positive\")\n",
        "fim_suffix": "\n    return round(bmi, 2)\n",
    },
]


def load_eval_model(model_path: str):
    """Load model and tokenizer with automatic precision."""
    logger.info(f"Loading model from {model_path}...")
    if (model_path.startswith("/") or model_path.startswith(".") or "\\" in model_path) and not Path(model_path).exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}. Ensure training has completed and saved checkpoint.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt_info: Dict) -> str:
    """Generate response for standard or FIM prompts."""
    if prompt_info.get("prompt") == "FIM":
        # Format FIM prompt
        prefix = prompt_info["fim_prefix"]
        suffix = prompt_info["fim_suffix"]
        fim_text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        inputs = tokenizer(fim_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|fim_middle|>") or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    messages = [
        {"role": "system", "content": "You are QaptaanLM, an expert programming and reasoning assistant."},
        {"role": "user", "content": prompt_info["prompt"]},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def run_evaluation(model_path: str, baseline_path: Optional[str] = None, output_file: Optional[str] = None):
    """Run full evaluation suite and display rich results."""
    model, tokenizer = load_eval_model(model_path)
    
    baseline_model = None
    baseline_tokenizer = None
    if baseline_path:
        baseline_model, baseline_tokenizer = load_eval_model(baseline_path)

    results = []

    console.print("\n[bold cyan]═══ QaptaanLM SFT Evaluation Suite ═══[/bold cyan]\n")

    for i, p_info in enumerate(EVAL_PROMPTS, 1):
        cat = p_info["category"]
        p_text = p_info["prompt"] if p_info["prompt"] != "FIM" else f"Prefix: {p_info['fim_prefix'][:40]}... Suffix: {p_info['fim_suffix'][:30]}..."
        
        console.print(f"[bold yellow]Test {i}/{len(EVAL_PROMPTS)}: [{cat}][/bold yellow]")
        console.print(f"[dim]{p_text}[/dim]\n")

        t0 = time.time()
        sft_response = generate_response(model, tokenizer, p_info)
        sft_time = time.time() - t0

        baseline_response = None
        if baseline_model and baseline_tokenizer:
            baseline_response = generate_response(baseline_model, baseline_tokenizer, p_info)

        record = {
            "test_num": i,
            "category": cat,
            "prompt": p_text,
            "sft_model": model_path,
            "sft_response": sft_response,
            "latency_sec": round(sft_time, 2),
        }
        if baseline_response:
            record["baseline_model"] = baseline_path
            record["baseline_response"] = baseline_response

        results.append(record)

        console.print(Panel(sft_response, title=f"SFT Output ({sft_time:.2f}s)", border_style="green"))
        if baseline_response:
            console.print(Panel(baseline_response, title="Baseline (CPT) Output", border_style="blue"))
        console.print("-" * 75)

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n✓ Saved evaluation results to [bold green]{output_file}[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="QaptaanLM SFT Evaluation Suite")
    parser.add_argument("--model", type=str, default="checkpoints/jax_sft_hf", help="Path to SFT model")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline model to compare")
    parser.add_argument("--output", type=str, default="reports/sft_evaluation_report.json", help="Path to save report JSON")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        baseline_path=args.baseline,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
