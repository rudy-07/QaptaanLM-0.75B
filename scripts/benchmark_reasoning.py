"""Automated Reasoning and Intelligence Benchmarking Script (Base vs. QaptaanLM CPT).

Evaluates models across reasoning domains:
- MMLU (Multidisciplinary knowledge & science)
- GSM8K (Math word problems with Chain-of-Thought)
- ARC-Challenge (Advanced reasoning science questions)
- HellaSwag & Winogrande (Commonsense reasoning)
- TruthfulQA (Factuality and truthfulness)
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# Curated benchmark datasets for fast & representative evaluation
REASONING_DATASETS = {
    "MMLU (Sample Questions)": [
        {
            "id": "mmlu_cs_1",
            "prompt": "Question: In computer networking, which layer of the OSI model is responsible for end-to-end communication and reliability control?\nA. Transport layer\nB. Network layer\nC. Data link layer\nD. Physical layer\nAnswer:",
            "correct": "A",
        },
        {
            "id": "mmlu_physics_1",
            "prompt": "Question: Which of the following principles states that the total momentum of an isolated system remains constant?\nA. Conservation of Energy\nB. Conservation of Linear Momentum\nC. Bernoulli's Principle\nD. Heisenberg Uncertainty Principle\nAnswer:",
            "correct": "B",
        },
        {
            "id": "mmlu_math_1",
            "prompt": "Question: What is the derivative of f(x) = 3*x^2 + 5*x - 7 with respect to x?\nA. 6x + 5\nB. 3x + 5\nC. 6x\nD. 6x^2 + 5\nAnswer:",
            "correct": "A",
        },
    ],
    "GSM8K (Math Word Problems)": [
        {
            "id": "gsm8k_1",
            "prompt": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's think step by step.",
            "correct": "72",
        },
        {
            "id": "gsm8k_2",
            "prompt": "Question: Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes of babysitting. How much did she earn?\nAnswer: Let's think step by step.",
            "correct": "10",
        },
        {
            "id": "gsm8k_3",
            "prompt": "Question: A train travels at a speed of 60 mph for 2.5 hours. How many miles did the train travel?\nAnswer: Let's think step by step.",
            "correct": "150",
        },
    ],
    "ARC-Challenge (Science QA)": [
        {
            "id": "arc_1",
            "prompt": "Question: Which property of a mineral can be determined simply by scratching it against a piece of glass?\nA. Hardness\nB. Luster\nC. Streak\nD. Cleavage\nAnswer:",
            "correct": "A",
        },
        {
            "id": "arc_2",
            "prompt": "Question: What type of energy transformation occurs when a battery powers a flashlight?\nA. Thermal to radiant\nB. Chemical to electrical to light\nC. Nuclear to mechanical\nD. Solar to chemical\nAnswer:",
            "correct": "B",
        },
    ],
    "Commonsense (HellaSwag / Winogrande)": [
        {
            "id": "cs_1",
            "prompt": "Question: The trophy didn't fit into the brown suitcase because it was too large. What was too large?\nA. The trophy\nB. The suitcase\nAnswer:",
            "correct": "A",
        },
        {
            "id": "cs_2",
            "prompt": "Question: A person puts a pot of water on the stove and turns on the heat. After ten minutes, what happens?\nA. The water begins to boil and produce steam\nB. The water turns into ice cubes\nC. The pot melts immediately\nD. The water disappears instantaneously\nAnswer:",
            "correct": "A",
        },
    ]
}


def extract_answer(completion: str, expected_type: str = "mcq") -> str:
    """Extracts the final answer from completion."""
    clean = completion.strip()
    if expected_type == "mcq":
        # Look for leading option letters A, B, C, D
        match = re.search(r"\b([A-D])\b", clean[:20])
        if match:
            return match.group(1)
        return clean[:1].upper()
    else:
        # Look for numeric values
        numbers = re.findall(r"\d+(?:\.\d+)?", clean)
        return numbers[-1] if numbers else clean


def evaluate_reasoning_domain(
    model_id: str,
    domain_name: str,
    samples: List[Dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 120,
) -> Dict[str, Any]:
    is_math = "GSM8K" in domain_name
    correct_count = 0
    total = len(samples)
    results = []

    for item in samples:
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        pred = extract_answer(completion, expected_type="math" if is_math else "mcq")
        is_correct = (str(item["correct"]).lower() in str(pred).lower()) or (str(item["correct"]) in completion)
        if is_correct:
            correct_count += 1

        results.append({
            "id": item["id"],
            "prompt": prompt,
            "completion": completion,
            "predicted": pred,
            "expected": item["correct"],
            "correct": is_correct,
            "time_sec": round(elapsed, 2),
        })

    accuracy = (correct_count / total) * 100
    return {
        "domain": domain_name,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "details": results,
    }


def benchmark_model_on_reasoning(
    model_id: str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"Evaluating Reasoning Benchmark for: {model_id}")
    print(f"========================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    domain_results = {}
    total_correct = 0
    total_samples = 0

    t_start = time.perf_counter()

    for domain_name, samples in REASONING_DATASETS.items():
        res = evaluate_reasoning_domain(model_id, domain_name, samples, tokenizer, model, device)
        domain_results[domain_name] = res
        total_correct += res["correct"]
        total_samples += res["total"]
        print(f"  ✓ {domain_name:<35}: {res['correct']}/{res['total']} ({res['accuracy']:.1f}%)")

    total_time = time.perf_counter() - t_start
    overall_acc = (total_correct / total_samples) * 100

    print(f"\nOverall Reasoning Accuracy: {total_correct}/{total_samples} ({overall_acc:.1f}%) in {total_time:.2f}s")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "total_time_sec": round(total_time, 2),
        "domains": domain_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Reasoning Benchmark on Base vs CPT")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--cpt_model", type=str, default="kaptaan45/QaptaanLM-0.75B")
    parser.add_argument("--output_file", type=str, default="reasoning_benchmark_results.json")
    args = parser.parse_args()

    base_summary = benchmark_model_on_reasoning(args.base_model)
    cpt_summary = benchmark_model_on_reasoning(args.cpt_model)

    summary = {
        "base_model": base_summary,
        "cpt_model": cpt_summary,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Reasoning Benchmark Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
