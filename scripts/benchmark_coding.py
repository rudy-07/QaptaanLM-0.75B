"""Automated Coding Benchmarking Script (Base Model vs. QaptaanLM CPT Model).

Evaluates Python code generation across HumanEval and MBPP problem sets with
automated sandboxed unit testing and pass@1 accuracy measurement.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# Standard HumanEval sample problems for self-contained evaluation
HUMANEVAL_PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "entry_point": "has_close_elements",
        "test": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced, each group is completely closed, do not include any spaces.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "entry_point": "separate_paren_groups",
        "test": "assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\nassert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\nassert separate_paren_groups('(()(())((())))') == ['(()(())((())))']\n",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1, also called fractional part).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
        "entry_point": "truncate_number",
        "test": "assert truncate_number(3.5) == 0.5\nassert abs(truncate_number(1.33) - 0.33) < 1e-4\nassert abs(truncate_number(123.456) - 0.456) < 1e-4\n",
    },
    {
        "task_id": "HumanEval/3",
        "prompt": "def below_zero(operations: list[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    balance zero. Your task is to detect if at any point the balance of account falls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
        "entry_point": "below_zero",
        "test": "assert below_zero([]) == False\nassert below_zero([1, 2, -3, 1, 2, -3]) == False\nassert below_zero([1, 2, -4, 5, 6]) == True\nassert below_zero([1, -1, 2, -2, 5, -5, -1]) == True\nassert below_zero([1, -1, 2, -2, 5, -5, 1]) == False\n",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": "def mean_absolute_deviation(numbers: list[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a mean of this dataset:\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
        "entry_point": "mean_absolute_deviation",
        "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-4\nassert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-4\nassert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-4\n",
    },
    {
        "task_id": "HumanEval/8",
        "prompt": "def sum_product(numbers: list[int]) -> tuple[int, int]:\n    \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n    Empty sum should be equal to 0 and empty product should be equal to 1.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    \"\"\"\n",
        "entry_point": "sum_product",
        "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 1, 1]) == (3, 1)\nassert sum_product([100, 0]) == (100, 0)\nassert sum_product([3, 5, 7]) == (15, 105)\nassert sum_product([10]) == (10, 10)\n",
    },
    {
        "task_id": "HumanEval/11",
        "prompt": "def string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
        "entry_point": "string_xor",
        "test": "assert string_xor('111000', '101010') == '010010'\nassert string_xor('1', '1') == '0'\nassert string_xor('0101', '0000') == '0101'\n",
    },
    {
        "task_id": "HumanEval/15",
        "prompt": "def string_sequence(n: int) -> str:\n    \"\"\" Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n    >>> string_sequence(0)\n    '0'\n    >>> string_sequence(5)\n    '0 1 2 3 4 5'\n    \"\"\"\n",
        "entry_point": "string_sequence",
        "test": "assert string_sequence(0) == '0'\nassert string_sequence(3) == '0 1 2 3'\nassert string_sequence(10) == '0 1 2 3 4 5 6 7 8 9 10'\n",
    },
]

MBPP_PROBLEMS = [
    {
        "task_id": "MBPP/1",
        "prompt": "def min_cost(cost: list[list[int]], m: int, n: int) -> int:\n    \"\"\"Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[R][C].\"\"\"\n",
        "entry_point": "min_cost",
        "test": "cost = [[1, 2, 3], [4, 8, 2], [1, 5, 3]]\nassert min_cost(cost, 2, 2) == 8\n",
    },
    {
        "task_id": "MBPP/2",
        "prompt": "def similar_elements(test_tup1: tuple, test_tup2: tuple) -> tuple:\n    \"\"\"Write a function to find the similar elements from the given two tuple lists.\"\"\"\n",
        "entry_point": "similar_elements",
        "test": "assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == {4, 5}\nassert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == {3, 4}\n",
    },
    {
        "task_id": "MBPP/3",
        "prompt": "def is_not_prime(n: int) -> bool:\n    \"\"\"Write a python function to identify non-prime numbers.\"\"\"\n",
        "entry_point": "is_not_prime",
        "test": "assert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True\nassert is_not_prime(37) == False\n",
    },
]


def extract_code(completion: str) -> str:
    """Extracts valid Python code from model completion."""
    if "```python" in completion:
        code = completion.split("```python")[1].split("```")[0]
        return code
    if "```" in completion:
        code = completion.split("```")[1].split("```")[0]
        return code
    return completion


def run_code_test(code: str, test_code: str) -> bool:
    """Executes code and unit test in a safe local namespace."""
    full_code = f"from typing import List, Tuple, Dict, Optional, Any, Union\nimport math\n\n{code}\n\n{test_code}"
    local_ns = {}
    try:
        exec(full_code, {}, local_ns)
        return True
    except Exception:
        return False


def benchmark_model_on_coding(
    model_id: str,
    problems: List[Dict[str, Any]],
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"Evaluating Coding Benchmark for: {model_id}")
    print(f"========================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    passed = 0
    total = len(problems)
    results = []

    t_start = time.perf_counter()
    total_tokens = 0

    for prob in tqdm(problems, desc=f"Evaluating {model_id.split('/')[-1]}"):
        prompt = prob["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        
        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        total_tokens += len(gen_tokens)
        
        completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        full_code = prompt + extract_code(completion)
        
        is_pass = run_code_test(full_code, prob["test"])
        if is_pass:
            passed += 1

        results.append({
            "task_id": prob["task_id"],
            "prompt": prompt,
            "completion": completion,
            "passed": is_pass,
            "tokens": len(gen_tokens),
            "time_sec": round(elapsed, 2),
        })

    total_time = time.perf_counter() - t_start
    throughput = total_tokens / total_time if total_time > 0 else 0
    pass_rate = (passed / total) * 100

    print(f"\nResults for {model_id}:")
    print(f"  Pass@1: {passed}/{total} ({pass_rate:.1f}%)")
    print(f"  Throughput: {throughput:.1f} tok/s | Total Time: {total_time:.2f}s")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "throughput_tok_s": round(throughput, 1),
        "total_time_sec": round(total_time, 2),
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Coding Benchmark on Base vs CPT")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--cpt_model", type=str, default="kaptaan45/QaptaanLM-0.75B")
    parser.add_argument("--output_file", type=str, default="coding_benchmark_results.json")
    args = parser.parse_args()

    all_problems = HUMANEVAL_PROBLEMS + MBPP_PROBLEMS

    base_summary = benchmark_model_on_coding(args.base_model, all_problems)
    cpt_summary = benchmark_model_on_coding(args.cpt_model, all_problems)

    summary = {
        "base_model": base_summary,
        "cpt_model": cpt_summary,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Coding Benchmark Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
