# 🏆 QaptaanLM Comprehensive Benchmark Suite

An industry-standard, reproducible benchmarking framework for evaluating **QaptaanLM-0.75B**, **Qwen2.5-Coder-0.5B**, and other language and code models across **17 official benchmarks** spanning **Coding**, **Reasoning**, **Mathematics**, and **General Intelligence**.

---

## 🎯 Supported Benchmarks

### 1. Coding Benchmarks
| Benchmark | Scope | Metric | Evaluation Method |
| :--- | :--- | :--- | :--- |
| **HumanEval** | 164 Python problems | `pass@1` | OpenAI standard prompt & unit test assertions |
| **HumanEval+** | 164 problems (EvalPlus) | `pass@1` | 80x test amplification with contract validation |
| **MBPP** | 257 sanitized / 500 tasks | `pass@1` | 3-shot Python docstring-to-code execution |
| **MBPP+** | 378 problems (EvalPlus) | `pass@1` | Amplified test coverage via EvalPlus |
| **LiveCodeBench** | Contest problems (LeetCode/AtCoder) | `pass@1` | Live uncontaminated programming challenges |
| **BigCodeBench** | 1,140 complex tasks | `pass@1` | Multi-library (Numpy/Pandas/PyTorch) sandbox tests |

### 2. Intelligence, Reasoning & Knowledge Benchmarks
| Benchmark | Scope / Domain | Metric | Standard Setup |
| :--- | :--- | :--- | :--- |
| **MMLU** | 57 subjects (STEM, Humanities, Social Sci) | `accuracy` | 5-shot MC log-likelihood / greedy |
| **MMLU-Pro** | 14 domains (10 choices, reasoning) | `accuracy` | 5-shot Chain-of-Thought (CoT) |
| **MMLU-Redux** | 57 subjects (Cleaned annotations) | `accuracy` | 5-shot MC log-likelihood |
| **ARC-Challenge** | Grade-school science QA | `accuracy` | 25-shot / 0-shot multiple choice |
| **HellaSwag** | Commonsense situational continuation | `accuracy` | 10-shot / 0-shot log-likelihood |
| **Winogrande** | Adversarial coreference disambiguation | `accuracy` | 5-shot / 0-shot log-likelihood |
| **TruthfulQA** | Factuality & anti-hallucination | `accuracy` | 0-shot multiple choice |
| **BBH** | BIG-Bench Hard (23 reasoning tasks) | `exact_match` | 3-shot Chain-of-Thought |
| **GPQA** | Graduate-level PhD-level domain QA | `accuracy` | 5-shot Chain-of-Thought |
| **GSM8K** | 8,500 grade school math word problems | `accuracy` | 5-shot Chain-of-Thought numeric extraction |
| **MATH** | Competition math (Prealgebra to Calculus) | `accuracy` | 4-shot Chain-of-Thought + SymPy normalization |

---

## ⚡ Quickstart

### 1. Installation
```bash
pip install -r benchmarks/requirements-benchmarks.txt
```

### 2. Run All Benchmarks
```bash
# Evaluate QaptaanLM across all tasks
python -m benchmarks.scripts.run_all_benchmarks \
    --model kaptaan45/QaptaanLM-0.75B \
    --suite all

# Fast Smoke Test (10 samples per task)
python -m benchmarks.scripts.run_all_benchmarks \
    --model kaptaan45/QaptaanLM-0.75B \
    --suite quick \
    --limit 10
```

### 3. Run Specific Suites
```bash
# Run Coding Benchmarks Only (HumanEval, MBPP, etc.)
python -m benchmarks.scripts.run_coding_suite \
    --model kaptaan45/QaptaanLM-0.75B

# Run Reasoning & Math Benchmarks Only (MMLU, GSM8K, MATH, ARC, etc.)
python -m benchmarks.scripts.run_reasoning_suite \
    --model kaptaan45/QaptaanLM-0.75B \
    --tasks mmlu,gsm8k,math,arc_challenge
```

### 4. Compare Models Head-to-Head
```bash
python -m benchmarks.scripts.compare_models \
    --results metrics/results/benchmark_Qwen_Qwen2.5-Coder-0.5B_*.json metrics/results/benchmark_kaptaan45_QaptaanLM-0.75B_*.json \
    --baseline Qwen/Qwen2.5-Coder-0.5B \
    --output-dir reports/comparison
```

---

## 📊 Reference Published Baselines

Bundled in `benchmarks/reference_baselines/published_scores.json`:

| Model | Parameters | HumanEval | MBPP | MMLU | GSM8K | MATH | ARC-C |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-Coder-0.5B** | 0.49B | 31.1% | 52.5% | 45.4% | 40.2% | 21.6% | 38.6% |
| **Qwen2.5-Coder-1.5B** | 1.54B | 48.2% | 68.9% | 56.7% | 60.5% | 33.8% | 47.9% |
| **Qwen2.5-0.5B (Base)** | 0.49B | 18.9% | 38.1% | 46.8% | 36.4% | 18.5% | 40.2% |
| **SmolLM2-360M** | 0.36B | 14.6% | 28.4% | 38.2% | 20.8% | 7.5% | 33.1% |
| **SmolLM2-1.7B** | 1.71B | 28.7% | 51.4% | 52.3% | 48.5% | 21.0% | 46.2% |
| **Llama-3.2-1B** | 1.23B | 26.2% | 44.0% | 49.3% | 44.7% | 18.2% | 42.4% |

---

## 📁 Architecture Overview

```
benchmarks/
├── core/
│   ├── model_loader.py          # Unified CausalLM / KV-cache / Loglikelihood loader
│   ├── prompt_templates.py      # Standard 5-shot CoT and few-shot formatting
│   ├── execution_sandbox.py     # Multiprocessing timeout sandbox for code execution
│   ├── metrics_calculator.py    # Unbiased pass@k, SymPy math & MCQ extractors
│   └── report_generator.py      # Markdown, JSON, and standalone HTML leaderboard
├── datasets/
│   └── loaders.py               # HuggingFace dataset downloader & fallback schemas
├── runners/
│   ├── standalone_runner.py     # High-speed native benchmark runner
│   ├── lm_eval_runner.py        # EleutherAI lm-eval integration
│   ├── evalplus_runner.py       # EvalPlus (HumanEval+ / MBPP+) integration
│   ├── bigcodebench_runner.py   # BigCodeBench integration
│   └── livecodebench_runner.py  # LiveCodeBench integration
├── configs/                     # YAML configs for all, coding, reasoning, quick eval
├── reference_baselines/         # Published baseline scores from official papers
├── scripts/                     # CLI entrypoints (run_all, coding, reasoning, compare)
└── notebooks/
    └── comprehensive_benchmark_kaggle.ipynb  # Ready-to-run 1-click Kaggle GPU notebook
```
