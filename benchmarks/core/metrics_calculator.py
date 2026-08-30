"""Standardized Metric Calculators for Industry Benchmarks.

Implements:
- Pass@k (unbiased estimator for code generation)
- Mathematical expression & numerical equivalence (GSM8K, MATH)
- Multiple-choice answer extraction & accuracy (MMLU, MMLU-Pro, ARC, HellaSwag, GPQA, Winogrande)
- Chain-of-Thought final answer extraction
"""

import math
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[int, List[int], np.ndarray],
    k: int,
) -> float:
    """Estimates pass@k using the standard unbiased estimator from Chen et al. (HumanEval).

    Formula:
        pass@k = 1 - C(n - c, k) / C(n, k) = 1 - prod_{i=1}^k (1 - c / (n - i + 1))
    """
    if isinstance(num_samples, int):
        num_samples = [num_samples]
        num_correct = [num_correct]

    num_samples = np.array(num_samples)
    num_correct = np.array(num_correct)

    if (num_samples < k).any():
        # Fallback to simple ratio if fewer samples than k
        return float(np.mean(num_correct / np.maximum(num_samples, 1)))

    def _pass_at_k(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - float(np.prod(1.0 - c / np.arange(n - k + 1, n + 1)))

    scores = [_pass_at_k(n, c, k) for n, c in zip(num_samples, num_correct)]
    return float(np.mean(scores))


# ==============================================================================
# Math Answer Extraction & Equivalence (GSM8K & MATH)
# ==============================================================================
def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extracts numeric answer from GSM8K completion."""
    # 1. Look for '#### <number>' standard tag
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.replace(",", "").replace("$", "").strip()
        return ans

    # 2. Look for 'The answer is <number>'
    match = re.search(r"[Tt]he answer is:?\s*([+-]?\$?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").replace("$", "").strip()

    # 3. Look for LaTeX boxed expression '\boxed{...}'
    boxed = extract_latex_boxed(text)
    if boxed is not None:
        return boxed.replace(",", "").replace("$", "").strip()

    # 4. Fallback to extracting the last number in the text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1].strip()

    return None


def extract_latex_boxed(text: str) -> Optional[str]:
    """Extracts content inside LaTeX \\boxed{...} with balanced brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed ")
        if idx == -1:
            return None
        return text[idx + 7 :].split()[0].strip()

    start = idx + 7
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def normalize_math_answer(ans: Optional[str]) -> str:
    """Normalizes math answer string for strict and relaxed comparison."""
    if ans is None:
        return ""
    clean = ans.strip()
    clean = clean.replace("$", "").replace("%", "").replace(",", "")
    clean = re.sub(r"\s+", "", clean)
    clean = clean.rstrip(".")
    # Remove leading +
    if clean.startswith("+"):
        clean = clean[1:]
    return clean


def is_math_answer_equal(pred: Optional[str], target: str) -> bool:
    """Checks if predicted math answer matches target answer numerically or symbolically."""
    if pred is None or not target:
        return False

    norm_pred = normalize_math_answer(pred)
    norm_target = normalize_math_answer(target)

    # 1. Exact string match
    if norm_pred.lower() == norm_target.lower():
        return True

    # 2. Numerical float match
    try:
        f_pred = float(norm_pred)
        f_target = float(norm_target)
        if math.isclose(f_pred, f_target, rel_tol=1e-4, abs_tol=1e-4):
            return True
    except (ValueError, OverflowError):
        pass

    # 3. Try SymPy symbolic comparison if available
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr

        p_expr = parse_expr(norm_pred.replace("^", "**"))
        t_expr = parse_expr(norm_target.replace("^", "**"))
        if sympy.simplify(p_expr - t_expr) == 0:
            return True
    except Exception:
        pass

    return False


# ==============================================================================
# Multiple Choice Answer Extraction (MMLU, MMLU-Pro, ARC, etc.)
# ==============================================================================
def extract_mcq_answer(
    text: str,
    valid_options: Optional[List[str]] = None,
) -> Optional[str]:
    """Extracts multiple choice letter (A, B, C, D, ...) from completion text."""
    if valid_options is None:
        valid_options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    valid_pattern = "|".join(valid_options)
    clean = text.strip()

    # 1. Look for 'The answer is (X)' or 'The correct answer is X'
    match = re.search(
        rf"[Tt]he (?:correct )?answer is:?\s*\(?([{valid_pattern}])\)?", clean
    )
    if match:
        return match.group(1).upper()

    # 2. Look for '\boxed{X}'
    boxed = extract_latex_boxed(clean)
    if boxed and boxed.upper() in valid_options:
        return boxed.upper()

    # 3. Look for leading letter: e.g. '(A)', 'A.', 'A)'
    match = re.match(rf"^\(?([{valid_pattern}])[\.\:\)]", clean)
    if match:
        return match.group(1).upper()

    # 4. Search for any standalone option letter near the start or end
    matches = re.findall(rf"\b([{valid_pattern}])\b", clean)
    if matches:
        return matches[-1].upper()

    # 5. Check first character
    if clean and clean[0].upper() in valid_options:
        return clean[0].upper()

    return None
