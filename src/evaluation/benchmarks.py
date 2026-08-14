"""Evaluation suite for base and fine-tuned models.

Implements lightweight benchmark evaluation for:
- HumanEval / Python coding problems
- Basic reasoning and perplexity
- Math reasoning (GSM8K style prompts)
- Generation quality comparison
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Sample coding prompts for quick benchmark evaluation
HUMANEVAL_SAMPLE_PROMPTS = [
    {
        "task_id": "Python/0",
        "prompt": "def has_close_elements(numbers: list, threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "Python/1",
        "prompt": "def separate_paren_groups(paren_string: str) -> list:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced, each group is completely closed, do not include any spaces.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "Python/2",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1, also called fractional part).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
        "entry_point": "truncate_number",
    },
    {
        "task_id": "Python/3",
        "prompt": "def below_zero(operations: list) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    balance zero. Your task is to detect if at any point the balance of account falls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
        "entry_point": "below_zero",
    },
    {
        "task_id": "Python/4",
        "prompt": "def mean_absolute_deviation(numbers: list) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a mean of this dataset:\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
        "entry_point": "mean_absolute_deviation",
    },
]

MATH_REASONING_PROMPTS = [
    {
        "id": "math_0",
        "prompt": "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA: Let's think step by step.\n",
        "expected": "72",
    },
    {
        "id": "math_1",
        "prompt": "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nA: Let's think step by step.\n",
        "expected": "10",
    },
    {
        "id": "math_2",
        "prompt": "Q: A train travels at a speed of 60 mph for 2.5 hours. How far did the train travel?\nA: Let's think step by step.\n",
        "expected": "150",
    },
]


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    prompts: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run generation on a list of benchmark prompts.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Tokenizer.
        prompts: List of prompt dictionaries with 'prompt' key.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        device: Device to run evaluation on.

    Returns:
        List of results with prompt, completion, and generation time.
    """
    model.eval()
    results = []

    for item in tqdm(prompts, desc="Generating benchmark completions", unit="prompt"):
        prompt_text = item["prompt"]
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        start_t = time.time()
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_t

        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        completion = gen_text[len(prompt_text):]

        result = dict(item)
        result["completion"] = completion
        result["full_text"] = gen_text
        result["tokens_generated"] = len(gen_tokens[0]) - inputs["input_ids"].shape[1]
        result["elapsed_seconds"] = round(elapsed, 3)
        results.append(result)

    return results


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    eval_texts: List[str],
    max_length: int = 2048,
    device: str = "cpu",
) -> float:
    """Compute perplexity on a list of evaluation texts.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Tokenizer.
        eval_texts: List of evaluation text strings.
        max_length: Maximum sequence length.
        device: Device to run on.

    Returns:
        Perplexity score as a float.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for text in tqdm(eval_texts, desc="Computing perplexity", unit="doc"):
            if not text.strip():
                continue
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            input_ids = encodings.input_ids
            if input_ids.shape[1] < 2:
                continue

            target_ids = input_ids.clone()
            outputs = model(input_ids)
            logits = outputs.logits

            # Shift logits and targets for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = target_ids[..., 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
            )

            num_tokens = shift_targets.numel()
            total_loss += loss.item()
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity
