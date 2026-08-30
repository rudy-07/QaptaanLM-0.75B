"""Standard Prompt Templates and Few-Shot Exemplars for LLM Benchmarking.

Follows official implementations from:
- OpenAI HumanEval
- MBPP (Austin et al.)
- EleutherAI LM Evaluation Harness (MMLU, ARC, HellaSwag, Winogrande, TruthfulQA, BBH, GPQA)
- GSM8K (Cobbe et al.) 5-shot Chain-of-Thought
- Hendrycks MATH 4-shot Chain-of-Thought
- TIGER-Lab MMLU-Pro 5-shot CoT
"""

from typing import Any, Dict, List, Optional

# ==============================================================================
# GSM8K (5-shot Chain-of-Thought exemplars)
# ==============================================================================
GSM8K_COT_PROMPT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then he got 2 * 2 = 4 toys. 5 + 4 = 9. The answer is 9.

Question: {question}
Answer:"""

# ==============================================================================
# Hendrycks MATH (4-shot Chain-of-Thought exemplars)
# ==============================================================================
MATH_COT_PROMPT = """Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.
Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5-x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to 0, so $5-x>0$, which means $x<5$. Therefore, the domain of the expression is $[2,5)$. The answer is [2,5).

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
Solution: We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = 24.$ The answer is 24.

Problem: If $f(x) = \\frac{1}{x-1}$, find $f(f(x))$.
Solution: We have $f(f(x)) = f\\left(\\frac{1}{x-1}\\right) = \\frac{1}{\\frac{1}{x-1}-1} = \\frac{1}{\\frac{1-(x-1)}{x-1}} = \\frac{x-1}{2-x}.$ The answer is \\frac{x-1}{2-x}.

Problem: What is the sum of the digits of the integer equal to $10^{15} - 25$?
Solution: $10^{15} - 25 = 999999999999975.$ There are thirteen 9s, one 7, and one 5, so the sum of the digits is $13 \\times 9 + 7 + 5 = 117 + 12 = 129.$ The answer is 129.

Problem: {problem}
Solution:"""

# ==============================================================================
# MBPP (3-shot standard Python prompt)
# ==============================================================================
MBPP_3SHOT_PROMPT = '''"""
Write a function to find the shared elements from the given two lists.
"""
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res)

"""
Write a python function to identify non-prime numbers.
"""
def is_not_prime(n):
    result = False
    for i in range(2,int(n**0.5)+1):
        if n % i == 0:
            result = True
            break
    return result

"""
Write a function to find the largest sum of a contiguous subarray in the given array.
"""
def max_sub_array_sum(a, size):
    max_so_far = a[0]
    curr_max = a[0]
    for i in range(1, size):
        curr_max = max(a[i], curr_max + a[i])
        max_so_far = max(max_so_far, curr_max)
    return max_so_far

"""
{prompt}
"""
'''

# ==============================================================================
# BBH (BIG-Bench Hard 3-shot Chain-of-Thought general template)
# ==============================================================================
BBH_COT_PREFIX = """Follow the given examples and answer the question step by step. End your answer with 'So the answer is [answer]'.\n\n"""

# ==============================================================================
# GPQA (Google-Proof Q&A CoT template)
# ==============================================================================
GPQA_PROMPT_TEMPLATE = """Answer the following multiple choice question. Think step by step before giving your final answer.

Question: {question}

Choices:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Explanation: Let's think step by step:"""

# ==============================================================================
# MMLU / MMLU-Redux / ARC-Challenge standard multiple choice format
# ==============================================================================
def format_mcq_prompt(
    question: str,
    choices: List[str],
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    subject: Optional[str] = None,
) -> str:
    """Formats standard multiple choice prompt with optional subject and few-shot examples."""
    prompt = ""
    if subject:
        clean_subj = subject.replace("_", " ")
        prompt += f"The following are multiple choice questions (with answers) about {clean_subj}.\n\n"

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            for i, ch in enumerate(ex["choices"]):
                prompt += f"({letters[i]}) {ch}\n"
            prompt += f"Answer: ({ex['answer']})\n\n"

    prompt += f"Question: {question}\n"
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for i, ch in enumerate(choices):
        prompt += f"({letters[i]}) {ch}\n"
    prompt += "Answer:"
    return prompt


# ==============================================================================
# MMLU-Pro 5-shot Chain-of-Thought Format
# ==============================================================================
def format_mmlu_pro_prompt(
    question: str,
    options: List[str],
    cot_examples: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """Formats MMLU-Pro prompt with up to 10 options and Chain-of-Thought format."""
    header = ""
    if category:
        header = f"The following are multiple choice questions with step-by-step reasoning about {category}.\n\n"

    prefix = cot_examples if cot_examples else header
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    q_block = f"Question: {question}\nOptions:\n"
    for i, opt in enumerate(options):
        q_block += f"({letters[i]}) {opt}\n"
    q_block += "\nThought:\n"

    return prefix + q_block


# ==============================================================================
# HellaSwag & Winogrande formatting
# ==============================================================================
def format_hellaswag_prompt(activity_label: str, ctx: str, endings: List[str]) -> Dict[str, Any]:
    """Formats HellaSwag context and candidate continuations."""
    context = f"Activity: {activity_label}\n{ctx}"
    return {
        "context": context,
        "continuations": endings,
    }


def format_winogrande_prompt(sentence: str, option1: str, option2: str) -> Dict[str, Any]:
    """Formats Winogrande sentence with fill-in-the-blank '_' replacement."""
    cand1 = sentence.replace("_", option1)
    cand2 = sentence.replace("_", option2)
    return {
        "sentence": sentence,
        "candidates": [cand1, cand2],
    }


# ==============================================================================
# TruthfulQA formatting
# ==============================================================================
def format_truthfulqa_prompt(question: str) -> str:
    """Formats TruthfulQA single question prompt."""
    return f"Q: {question}\nA:"
