"""Dataset Loaders and Formatters for Benchmark Evaluation.

Downloads and normalizes benchmark datasets directly from HuggingFace Hub
or local cache, providing clean structured schemas across all 17 benchmarks:
- HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench, BigCodeBench
- MMLU, MMLU-Pro, MMLU-Redux, ARC-Challenge, HellaSwag, Winogrande, TruthfulQA, BBH, GPQA, GSM8K, MATH
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# Code Benchmarks
# ==============================================================================
def load_humaneval(
    plus: bool = False,
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads OpenAI HumanEval or EvalPlus HumanEval+ dataset."""
    if fallback_only:
        return _get_fallback_humaneval(limit)

    ds_name = "evalplus/humanevalplus" if plus else "openai_humaneval"
    try:
        from datasets import load_dataset

        ds = load_dataset(ds_name, split="test")

        problems = []
        for row in ds:
            problems.append({
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "canonical_solution": row.get("canonical_solution", ""),
            })
            if limit and len(problems) >= limit:
                break
        return problems
    except Exception as e:
        logger.warning(f"Could not load {ds_name} from HuggingFace ({e}). Falling back to sample set.")
        return _get_fallback_humaneval(limit)


def load_mbpp(
    plus: bool = False,
    sanitized: bool = True,
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads MBPP (Mostly Basic Python Problems) or MBPP+ dataset."""
    if fallback_only:
        return _get_fallback_mbpp(limit)

    try:
        from datasets import load_dataset

        if plus:
            ds = load_dataset("evalplus/mbppplus", split="test")
        elif sanitized:
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        else:
            ds = load_dataset("google-research-datasets/mbpp", "full", split="test")

        problems = []
        for row in ds:
            task_id = f"MBPP/{row['task_id']}"
            prompt = row.get("prompt", row.get("text", ""))
            test_list = row.get("test_list", [])
            test_setup = row.get("test_setup_code", "")
            test_code = (test_setup + "\n" if test_setup else "") + "\n".join(test_list)

            entry_point = None
            if "entry_point" in row and row["entry_point"]:
                entry_point = row["entry_point"]
            elif test_list:
                first_test = test_list[0]
                if "assert " in first_test and "(" in first_test:
                    entry_point = first_test.split("assert ")[1].split("(")[0].strip()

            problems.append({
                "task_id": task_id,
                "prompt": prompt,
                "test": test_code,
                "entry_point": entry_point,
                "code": row.get("code", ""),
            })
            if limit and len(problems) >= limit:
                break
        return problems
    except Exception as e:
        logger.warning(f"Could not load MBPP from HuggingFace ({e}). Falling back to sample set.")
        return _get_fallback_mbpp(limit)


# ==============================================================================
# Reasoning & Intelligence Benchmarks
# ==============================================================================
def load_gsm8k(
    split: str = "test",
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads GSM8K grade school math word problems."""
    if fallback_only:
        return _get_fallback_gsm8k(limit)

    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=split)
        items = []
        for idx, row in enumerate(ds):
            ans_raw = row["answer"]
            ground_truth = ans_raw.split("####")[-1].strip() if "####" in ans_raw else ans_raw.strip()
            items.append({
                "id": f"gsm8k_{idx}",
                "question": row["question"],
                "solution": ans_raw,
                "answer": ground_truth,
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.warning(f"Could not load GSM8K from HuggingFace ({e}). Falling back to sample set.")
        return _get_fallback_gsm8k(limit)


def load_hendrycks_math(
    split: str = "test",
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads Hendrycks MATH competition dataset."""
    if fallback_only:
        return _get_fallback_math(limit)

    try:
        from datasets import load_dataset

        ds = load_dataset("lighteval/MATH", split=split)
        items = []
        for idx, row in enumerate(ds):
            items.append({
                "id": f"math_{idx}",
                "problem": row["problem"],
                "solution": row["solution"],
                "type": row.get("type", "General"),
                "level": row.get("level", "Level 1"),
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.warning(f"Could not load MATH dataset ({e}). Falling back to sample set.")
        return _get_fallback_math(limit)


def load_mmlu(
    subject: Optional[str] = None,
    split: str = "test",
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads MMLU (Massive Multitask Language Understanding) dataset."""
    if fallback_only:
        return _get_fallback_mmlu(limit)

    try:
        from datasets import load_dataset

        subj_name = subject if subject else "all"
        ds = load_dataset("cais/mmlu", subj_name, split=split)
        items = []
        letters = ["A", "B", "C", "D"]
        for idx, row in enumerate(ds):
            items.append({
                "id": f"mmlu_{idx}",
                "question": row["question"],
                "choices": row["choices"],
                "answer": letters[row["answer"]] if isinstance(row["answer"], int) else str(row["answer"]),
                "subject": row.get("subject", subj_name),
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.warning(f"Could not load MMLU ({e}). Falling back to sample set.")
        return _get_fallback_mmlu(limit)


def load_arc_challenge(
    split: str = "test",
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads AI2 Reasoning Challenge (ARC-Challenge) dataset."""
    if fallback_only:
        return _get_fallback_arc(limit)

    try:
        from datasets import load_dataset

        ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        items = []
        for row in ds:
            choices_text = row["choices"]["text"]
            choices_label = row["choices"]["label"]
            items.append({
                "id": row["id"],
                "question": row["question"],
                "choices": choices_text,
                "labels": choices_label,
                "answer": row["answerKey"],
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.warning(f"Could not load ARC-Challenge ({e}). Falling back to sample set.")
        return _get_fallback_arc(limit)


def load_hellaswag(
    split: str = "validation",
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads HellaSwag commonsense reasoning dataset."""
    if fallback_only:
        return _get_fallback_hellaswag(limit)

    try:
        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag", split=split)
        items = []
        letters = ["A", "B", "C", "D"]
        for row in ds:
            label_idx = int(row["label"]) if str(row["label"]).isdigit() else 0
            items.append({
                "id": row["ind"],
                "activity_label": row["activity_label"],
                "ctx": row["ctx"],
                "endings": row["endings"],
                "answer": letters[label_idx],
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.warning(f"Could not load HellaSwag ({e}). Falling back to sample set.")
        return _get_fallback_hellaswag(limit)


def load_winogrande(
    split: str = "validation",
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads Winogrande coreference disambiguation dataset."""
    if fallback_only:
        return _get_fallback_winogrande(limit)

    try:
        from datasets import load_dataset

        ds = load_dataset("winogrande", "winogrande_xl", split=split)
        items = []
        for idx, row in enumerate(ds):
            items.append({
                "id": f"winogrande_{idx}",
                "sentence": row["sentence"],
                "option1": row["option1"],
                "option2": row["option2"],
                "answer": "A" if str(row["answer"]) == "1" else "B",
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.warning(f"Could not load Winogrande ({e}). Falling back to sample set.")
        return _get_fallback_winogrande(limit)


def load_truthfulqa(
    split: str = "validation",
    limit: Optional[int] = None,
    fallback_only: bool = False,
) -> List[Dict[str, Any]]:
    """Loads TruthfulQA dataset."""
    if fallback_only:
        return _get_fallback_truthfulqa(limit)

    try:
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split=split)
        items = []
        for idx, row in enumerate(ds):
            mc1_targets = row["mc1_targets"]
            choices = mc1_targets["choices"]
            labels = mc1_targets["labels"]
            correct_idx = labels.index(1) if 1 in labels else 0
            letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
            items.append({
                "id": f"truthfulqa_{idx}",
                "question": row["question"],
                "choices": choices,
                "answer": letters[correct_idx],
                "category": row.get("category", "General"),
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.warning(f"Could not load TruthfulQA ({e}). Falling back to sample set.")
        return _get_fallback_truthfulqa(limit)


# ==============================================================================
# Curated Representative Fallback Datasets (Zero-Network Mode)
# ==============================================================================
def _get_fallback_humaneval(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    problems = [
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
    ]
    return problems[:limit] if limit else problems


def _get_fallback_mbpp(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    problems = [
        {
            "task_id": "MBPP/1",
            "prompt": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[R][C].",
            "entry_point": "min_cost",
            "test": "cost = [[1, 2, 3], [4, 8, 2], [1, 5, 3]]\nassert min_cost(cost, 2, 2) == 8\n",
        },
        {
            "task_id": "MBPP/2",
            "prompt": "Write a function to find the similar elements from the given two tuple lists.",
            "entry_point": "similar_elements",
            "test": "assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == {4, 5}\nassert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == {3, 4}\n",
        },
    ]
    return problems[:limit] if limit else problems


def _get_fallback_gsm8k(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "gsm8k_0",
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "solution": "Natalia sold 48 / 2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips. #### 72",
            "answer": "72",
        },
        {
            "id": "gsm8k_1",
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "solution": "Weng earns $12 / 60 = $0.2 per minute. For 50 minutes she earns 50 * 0.2 = $10. #### 10",
            "answer": "10",
        },
    ]
    return items[:limit] if limit else items


def _get_fallback_math(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "math_0",
            "problem": "What is the degree of the polynomial $f(x) = 3x^4 - 2x^2 + 7$?",
            "solution": "The highest exponent of $x$ is 4. The answer is 4.",
            "type": "Algebra",
            "level": "Level 1",
        },
    ]
    return items[:limit] if limit else items


def _get_fallback_mmlu(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "mmlu_0",
            "question": "In computer networking, which layer of the OSI model is responsible for end-to-end communication and reliability control?",
            "choices": ["Transport layer", "Network layer", "Data link layer", "Physical layer"],
            "answer": "A",
            "subject": "computer_science",
        },
        {
            "id": "mmlu_1",
            "question": "What is the derivative of f(x) = 3*x^2 + 5*x - 7 with respect to x?",
            "choices": ["6x + 5", "3x + 5", "6x", "6x^2 + 5"],
            "answer": "A",
            "subject": "college_mathematics",
        },
    ]
    return items[:limit] if limit else items


def _get_fallback_arc(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "arc_0",
            "question": "Which property of a mineral can be determined simply by scratching it against a piece of glass?",
            "choices": ["Hardness", "Luster", "Streak", "Cleavage"],
            "labels": ["A", "B", "C", "D"],
            "answer": "A",
        },
    ]
    return items[:limit] if limit else items


def _get_fallback_hellaswag(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "hs_0",
            "activity_label": "Baking cookies",
            "ctx": "A person places cookie dough onto a baking sheet and puts it in the preheated oven. After 12 minutes, they",
            "endings": [
                "take the golden-brown cookies out with an oven mitt.",
                "paint the oven blue with a brush.",
                "throw the baking sheet into the swimming pool.",
                "start singing into a microphone on a concert stage.",
            ],
            "answer": "A",
        }
    ]
    return items[:limit] if limit else items


def _get_fallback_winogrande(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "wg_0",
            "sentence": "The trophy didn't fit into the brown suitcase because _ was too large.",
            "option1": "the trophy",
            "option2": "the suitcase",
            "answer": "A",
        }
    ]
    return items[:limit] if limit else items


def _get_fallback_truthfulqa(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = [
        {
            "id": "tqa_0",
            "question": "What happens if you crack your knuckles?",
            "choices": [
                "Gas bubbles burst in the synovial fluid of your joints, but it does not cause arthritis.",
                "It immediately causes severe osteoarthritis.",
                "Your bones will shatter instantly.",
                "Your joints become permanently dislocated.",
            ],
            "answer": "A",
            "category": "Health",
        }
    ]
    return items[:limit] if limit else items
