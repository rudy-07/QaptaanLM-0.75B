"""Standalone Native Benchmark Runner.

Provides a self-contained, high-throughput benchmark execution engine that
downloads datasets directly, executes prompts with proper temperature/decoding,
runs sandboxed test suites, and calculates exact standardized scores.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm.auto import tqdm

from benchmarks.core.execution_sandbox import (
    evaluate_problem_completions,
    extract_python_code,
    run_code_with_sandbox,
)
from benchmarks.core.metrics_calculator import (
    estimate_pass_at_k,
    extract_gsm8k_answer,
    extract_latex_boxed,
    extract_mcq_answer,
    is_math_answer_equal,
)
from benchmarks.core.model_loader import BenchmarkModelWrapper
from benchmarks.core.prompt_templates import (
    GSM8K_COT_PROMPT,
    MATH_COT_PROMPT,
    MBPP_3SHOT_PROMPT,
    format_mcq_prompt,
    format_truthfulqa_prompt,
)
from benchmarks.datasets.loaders import (
    load_arc_challenge,
    load_gsm8k,
    load_hellaswag,
    load_hendrycks_math,
    load_humaneval,
    load_mbpp,
    load_mmlu,
    load_truthfulqa,
    load_winogrande,
)

logger = logging.getLogger(__name__)


class StandaloneBenchmarkRunner:
    """Executes benchmark suites directly with a loaded BenchmarkModelWrapper."""

    def __init__(
        self,
        model_wrapper: BenchmarkModelWrapper,
        batch_size: int = 1,
        timeout: float = 3.0,
    ):
        self.wrapper = model_wrapper
        self.batch_size = batch_size
        self.timeout = timeout

    # ==========================================================================
    # Code Benchmarks
    # ==========================================================================
    def run_humaneval(
        self,
        plus: bool = False,
        limit: Optional[int] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates model on HumanEval (or HumanEval+)."""
        task_name = "HumanEval+" if plus else "HumanEval"
        logger.info(f"Running {task_name} (limit={limit})...")

        problems = load_humaneval(plus=plus, limit=limit)
        passed_count = 0
        total = len(problems)
        task_results = []

        t_start = time.perf_counter()
        for prob in tqdm(problems, desc=f"Evaluating {task_name}"):
            prompt = prob["prompt"]
            entry_point = prob.get("entry_point")
            test_code = prob["test"]

            completions = self.wrapper.generate(
                prompts=[prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=["\nclass ", "\ndef ", "\nif __name__", "\nprint("],
            )

            completion = completions[0]
            extracted = extract_python_code(completion, entry_point=entry_point)

            if entry_point and entry_point not in extracted and entry_point in prompt:
                full_code = prompt + "\n" + extracted
            else:
                full_code = extracted

            res = run_code_with_sandbox(
                code=full_code,
                test_code=test_code,
                entry_point=entry_point,
                timeout=self.timeout,
            )

            if res["passed"]:
                passed_count += 1

            task_results.append({
                "task_id": prob["task_id"],
                "passed": res["passed"],
                "status": res["status"],
                "error": res["error"],
                "completion": completion,
            })

        score = (passed_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": task_name,
            "category": "Code Generation",
            "metric": "pass@1",
            "score": round(score, 2),
            "passed": passed_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    def run_mbpp(
        self,
        plus: bool = False,
        sanitized: bool = True,
        limit: Optional[int] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates model on MBPP (or MBPP+)."""
        task_name = "MBPP+" if plus else "MBPP"
        logger.info(f"Running {task_name} (limit={limit})...")

        problems = load_mbpp(plus=plus, sanitized=sanitized, limit=limit)
        passed_count = 0
        total = len(problems)
        task_results = []

        t_start = time.perf_counter()
        for prob in tqdm(problems, desc=f"Evaluating {task_name}"):
            prompt_text = MBPP_3SHOT_PROMPT.format(prompt=prob["prompt"])
            entry_point = prob.get("entry_point")
            test_code = prob["test"]

            completions = self.wrapper.generate(
                prompts=[prompt_text],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=['"""', "\nclass ", "\nassert ", "\nif __name__"],
            )

            completion = completions[0]
            extracted = extract_python_code(completion, entry_point=entry_point)

            res = run_code_with_sandbox(
                code=extracted,
                test_code=test_code,
                entry_point=entry_point,
                timeout=self.timeout,
            )

            if res["passed"]:
                passed_count += 1

            task_results.append({
                "task_id": prob["task_id"],
                "passed": res["passed"],
                "status": res["status"],
                "error": res["error"],
                "completion": completion,
            })

        score = (passed_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": task_name,
            "category": "Code Generation",
            "metric": "pass@1",
            "score": round(score, 2),
            "passed": passed_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    # ==========================================================================
    # Math & Reasoning Benchmarks
    # ==========================================================================
    def run_gsm8k(
        self,
        limit: Optional[int] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates model on GSM8K 5-shot Chain-of-Thought."""
        logger.info(f"Running GSM8K (limit={limit})...")

        items = load_gsm8k(limit=limit)
        correct_count = 0
        total = len(items)
        task_results = []

        t_start = time.perf_counter()
        for item in tqdm(items, desc="Evaluating GSM8K"):
            prompt = GSM8K_COT_PROMPT.format(question=item["question"])
            completions = self.wrapper.generate(
                prompts=[prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=["\nQuestion:", "\n\nQuestion:"],
            )

            completion = completions[0]
            extracted = extract_gsm8k_answer(completion)
            target = item["answer"]

            is_correct = is_math_answer_equal(extracted, target)
            if is_correct:
                correct_count += 1

            task_results.append({
                "id": item["id"],
                "question": item["question"],
                "target": target,
                "predicted": extracted,
                "correct": is_correct,
                "completion": completion,
            })

        score = (correct_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": "GSM8K",
            "category": "Math Reasoning",
            "metric": "accuracy",
            "score": round(score, 2),
            "passed": correct_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    def run_math(
        self,
        limit: Optional[int] = None,
        max_new_tokens: int = 768,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates model on Hendrycks MATH 4-shot Chain-of-Thought."""
        logger.info(f"Running MATH (limit={limit})...")

        items = load_hendrycks_math(limit=limit)
        correct_count = 0
        total = len(items)
        task_results = []

        t_start = time.perf_counter()
        for item in tqdm(items, desc="Evaluating MATH"):
            prompt = MATH_COT_PROMPT.format(problem=item["problem"])
            completions = self.wrapper.generate(
                prompts=[prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=["\nProblem:", "\n\nProblem:"],
            )

            completion = completions[0]
            extracted = extract_latex_boxed(completion)
            if extracted is None:
                extracted = extract_gsm8k_answer(completion)

            target = extract_latex_boxed(item["solution"])
            if target is None:
                target = item.get("answer", "")

            is_correct = is_math_answer_equal(extracted, target)
            if is_correct:
                correct_count += 1

            task_results.append({
                "id": item["id"],
                "target": target,
                "predicted": extracted,
                "correct": is_correct,
                "completion": completion,
            })

        score = (correct_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": "MATH",
            "category": "Math Reasoning",
            "metric": "accuracy",
            "score": round(score, 2),
            "passed": correct_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    # ==========================================================================
    # Knowledge & Commonsense Multiple Choice Benchmarks
    # ==========================================================================
    def run_mmlu(
        self,
        subject: Optional[str] = None,
        limit: Optional[int] = None,
        mode: str = "loglikelihood",
    ) -> Dict[str, Any]:
        """Evaluates model on MMLU (57 subjects)."""
        logger.info(f"Running MMLU (subject={subject}, limit={limit})...")

        items = load_mmlu(subject=subject, limit=limit)
        correct_count = 0
        total = len(items)
        task_results = []

        t_start = time.perf_counter()
        letters = ["A", "B", "C", "D"]

        for item in tqdm(items, desc="Evaluating MMLU"):
            prompt = format_mcq_prompt(
                question=item["question"],
                choices=item["choices"],
                subject=item.get("subject"),
            )

            if mode == "loglikelihood":
                best_idx = self.wrapper.evaluate_multiple_choice_loglikelihood(
                    context=prompt,
                    choices=[f" ({ch})" for ch in letters[: len(item["choices"])]],
                )
                pred_letter = letters[best_idx]
            else:
                completions = self.wrapper.generate(
                    prompts=[prompt],
                    max_new_tokens=8,
                    temperature=0.0,
                )
                pred_letter = extract_mcq_answer(completions[0], letters) or "A"

            target_letter = item["answer"].strip().upper()
            is_correct = pred_letter == target_letter
            if is_correct:
                correct_count += 1

            task_results.append({
                "id": item["id"],
                "target": target_letter,
                "predicted": pred_letter,
                "correct": is_correct,
            })

        score = (correct_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": "MMLU",
            "category": "General Intelligence",
            "metric": "accuracy",
            "score": round(score, 2),
            "passed": correct_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    def run_arc_challenge(
        self,
        limit: Optional[int] = None,
        mode: str = "loglikelihood",
    ) -> Dict[str, Any]:
        """Evaluates model on ARC-Challenge."""
        logger.info(f"Running ARC-Challenge (limit={limit})...")

        items = load_arc_challenge(limit=limit)
        correct_count = 0
        total = len(items)
        task_results = []

        t_start = time.perf_counter()
        letters = ["A", "B", "C", "D", "E"]

        for item in tqdm(items, desc="Evaluating ARC-Challenge"):
            prompt = format_mcq_prompt(
                question=item["question"],
                choices=item["choices"],
            )

            if mode == "loglikelihood":
                best_idx = self.wrapper.evaluate_multiple_choice_loglikelihood(
                    context=prompt,
                    choices=[f" ({ch})" for ch in letters[: len(item["choices"])]],
                )
                pred_letter = letters[best_idx]
            else:
                completions = self.wrapper.generate(
                    prompts=[prompt],
                    max_new_tokens=8,
                    temperature=0.0,
                )
                pred_letter = extract_mcq_answer(completions[0], letters) or "A"

            target_letter = item["answer"].strip().upper()
            is_correct = pred_letter == target_letter
            if is_correct:
                correct_count += 1

            task_results.append({
                "id": item["id"],
                "target": target_letter,
                "predicted": pred_letter,
                "correct": is_correct,
            })

        score = (correct_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": "ARC-Challenge",
            "category": "Reasoning & Science",
            "metric": "accuracy",
            "score": round(score, 2),
            "passed": correct_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    def run_hellaswag(
        self,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates model on HellaSwag commonsense reasoning."""
        logger.info(f"Running HellaSwag (limit={limit})...")

        items = load_hellaswag(limit=limit)
        correct_count = 0
        total = len(items)
        task_results = []

        t_start = time.perf_counter()
        letters = ["A", "B", "C", "D"]

        for item in tqdm(items, desc="Evaluating HellaSwag"):
            context = f"Activity: {item.get('activity_label', '')}\n{item['ctx']}"
            best_idx = self.wrapper.evaluate_multiple_choice_loglikelihood(
                context=context,
                choices=item["endings"],
            )
            pred_letter = letters[best_idx]
            target_letter = item["answer"].strip().upper()

            is_correct = pred_letter == target_letter
            if is_correct:
                correct_count += 1

            task_results.append({
                "id": item["id"],
                "target": target_letter,
                "predicted": pred_letter,
                "correct": is_correct,
            })

        score = (correct_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": "HellaSwag",
            "category": "Commonsense Reasoning",
            "metric": "accuracy",
            "score": round(score, 2),
            "passed": correct_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    def run_winogrande(
        self,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates model on Winogrande coreference."""
        logger.info(f"Running Winogrande (limit={limit})...")

        items = load_winogrande(limit=limit)
        correct_count = 0
        total = len(items)
        task_results = []

        t_start = time.perf_counter()
        letters = ["A", "B"]

        for item in tqdm(items, desc="Evaluating Winogrande"):
            prompt = f"Sentence: {item['sentence']}\nOption A: {item['option1']}\nOption B: {item['option2']}\nAnswer:"
            best_idx = self.wrapper.evaluate_multiple_choice_loglikelihood(
                context=prompt,
                choices=[" (A)", " (B)"],
            )
            pred_letter = letters[best_idx]
            target_letter = item["answer"].strip().upper()

            is_correct = pred_letter == target_letter
            if is_correct:
                correct_count += 1

            task_results.append({
                "id": item["id"],
                "target": target_letter,
                "predicted": pred_letter,
                "correct": is_correct,
            })

        score = (correct_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": "Winogrande",
            "category": "Commonsense Reasoning",
            "metric": "accuracy",
            "score": round(score, 2),
            "passed": correct_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }

    def run_truthfulqa(
        self,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates model on TruthfulQA."""
        logger.info(f"Running TruthfulQA (limit={limit})...")

        items = load_truthfulqa(limit=limit)
        correct_count = 0
        total = len(items)
        task_results = []

        t_start = time.perf_counter()
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

        for item in tqdm(items, desc="Evaluating TruthfulQA"):
            prompt = format_truthfulqa_prompt(item["question"])
            best_idx = self.wrapper.evaluate_multiple_choice_loglikelihood(
                context=prompt,
                choices=item["choices"],
            )
            pred_letter = letters[best_idx]
            target_letter = item["answer"].strip().upper()

            is_correct = pred_letter == target_letter
            if is_correct:
                correct_count += 1

            task_results.append({
                "id": item["id"],
                "target": target_letter,
                "predicted": pred_letter,
                "correct": is_correct,
            })

        score = (correct_count / total * 100.0) if total > 0 else 0.0
        elapsed = time.perf_counter() - t_start

        return {
            "task": "TruthfulQA",
            "category": "Factuality & Safety",
            "metric": "accuracy",
            "score": round(score, 2),
            "passed": correct_count,
            "total": total,
            "elapsed_sec": round(elapsed, 2),
            "samples": task_results,
        }
