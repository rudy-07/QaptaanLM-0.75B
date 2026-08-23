---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B-Base
language:
- en
- code
tags:
- code
- causal-lm
- qwen
- qwen3.5
- hybrid-attention
- deltanet
- gqa
- continued-pretraining
- fill-in-the-middle
- instruction-tuning
- kapcode
- kapinstruct
- text-generation
datasets:
- kaptaan45/KapCode-1B
- kaptaan45/KapInstruct-100M
pipeline_tag: text-generation
library_name: transformers
---

# QaptaanLM-0.75B: Efficient Hybrid-Attention Foundation Language Model

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-752M%20(Text--Only)-blue.svg)](#model-specification)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20DeltaNet%20%2B%20GQA-purple.svg)](#architecture)
[![Context Length](https://img.shields.io/badge/Context-256K%20Native-orange.svg)](#model-specification)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)
[![CPT Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20CPT%20Dataset-kaptaan45%2FKapCode--1B-yellow.svg)](https://huggingface.co/datasets/kaptaan45/KapCode-1B)
[![SFT Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20SFT%20Dataset-kaptaan45%2FKapInstruct--100M-orange.svg)](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-kapcode--1b-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/datasets/kaptaan45/kapcode-1b)

**QaptaanLM-0.75B** is a compact, high-efficiency hybrid-attention foundation language model optimized for source code synthesis, technical reasoning, and long-context code comprehension. 

Engineered by stripping the visual transformer from `Qwen/Qwen3.5-0.8B-Base` down to **752M dense parameters**, QaptaanLM achieves state-of-the-art computational and memory efficiency on consumer GPUs and edge accelerators. It couples linear-complexity recurrence layers with dense multi-head attention and undergoes a systematic two-stage training curriculum:
1. **Stage 1: Continued Pre-Training (CPT)** on **[KapCode-1B](https://huggingface.co/datasets/kaptaan45/KapCode-1B)** (1-billion-token curated code, doc, and STEM corpus with 50% Fill-in-the-Middle infilling).
2. **Stage 2: Supervised Fine-Tuning (SFT)** on **[KapInstruct-100M](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)** (100-million-token multi-source instruction mixture with strict assistant-only loss masking).

---

## Key Highlights

- **Text-Only Parameter Density**: Stripped vision transformer components from the base model, reducing parameter count from ~870M to **752M parameters**, dedicating 100% of capacity and VRAM to text and code synthesis.
- **3:1 Hybrid Attention Backbone**: Interleaves **3 Gated DeltaNet linear attention layers** with **1 Gated Grouped-Query Attention (GQA) layer** every 4 layers across 24 decoder layers. This maintains linear $O(N)$ computational and memory complexity for long sequences while preserving associative recall and multi-hop reasoning.
- **256K Context Window**: Native context length of 262,144 tokens powered by interleaved M-RoPE ($\theta = 10,000,000$).
- **Fill-in-the-Middle (FIM) Native**: Trained on prefix-suffix-middle code transformations for seamless IDE inline code completion.
- **Assistant-Only Loss Masking**: Aligned with Qwen ChatML using loss masking on system and user prompts to optimize gradient utilization.

---

## Model Specification

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Model Name** | QaptaanLM-0.75B | Text-only causal language model |
| **Base Architecture** | `Qwen/Qwen3.5-0.8B-Base` | Stripped vision encoder via `Qwen3_5ForCausalLM` |
| **Total Parameters** | **752,382,976 (752M)** | All 752M parameters trainable |
| **Hidden Size ($d_{model}$)** | 1024 | Base hidden dimension |
| **Intermediate Size ($d_{ffn}$)** | 3584 | SwiGLU non-linear activation |
| **Total Layers** | 24 | 18 Linear Attention + 6 Full Attention layers |
| **Full Attention Heads** | 8 Query / 2 Key-Value | Grouped-Query Attention (4:1 query-to-KV ratio) |
| **Full Attention Head Dim** | 256 | Query head dimension |
| **Linear Attention Heads** | 16 QK / 16 V | Gated DeltaNet (128 head dim, conv kernel dim 4) |
| **Max Context Length** | 262,144 tokens (256K native) | Tested with packed 4096-token sequences |
| **Vocabulary Size** | 248,320 tokens | Tied input and output word embeddings |
| **Rotary Position Embedding** | Interleaved M-RoPE | $\theta = 10,000,000$, partial rotary factor 0.25 |
| **Normalization** | RMSNorm ($\epsilon = 1\text{e-}6$) | Pre-layer normalization |
| **Precision Support** | `bfloat16`, `fp16`, `float32` | Native BF16 on modern GPUs & TPUs |

---

## Architecture

The backbone consists of 24 decoder layers structured into 6 repeating macro-blocks:

```
[Input Token IDs (vocab=248,320)]
               │
               ▼
   [Tied Embedding Layer (dim=1024)]
               │
               ▼
┌───────────────────────────────────────────────────────────┐
│       Repeating Macro-Block (Repeated 6x for 24 Layers)   │
│                                                           │
│  Layer 1: Gated DeltaNet (Linear Attention) + SwiGLU FFN  │
│  Layer 2: Gated DeltaNet (Linear Attention) + SwiGLU FFN  │
│  Layer 3: Gated DeltaNet (Linear Attention) + SwiGLU FFN  │
│  Layer 4: Gated Attention (Full GQA)        + SwiGLU FFN  │
└───────────────────────────────────────────────────────────┘
               │
               ▼
       [RMSNorm (eps=1e-6)]
               │
               ▼
     [Tied Output LM Head]
               │
               ▼
      [Next-Token Logits]
```

---

## Quickstart & Usage

### 1. Standard Autoregressive Generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "kaptaan45/QaptaanLM-0.75B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = "def quicksort(arr: list[int]) -> list[int]:\n    \"\"\"Sort list using divide-and-conquer quicksort.\"\"\"\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. ChatML Dialogue Inference

```python
messages = [
    {"role": "system", "content": "You are QaptaanLM, an expert programming and reasoning assistant."},
    {"role": "user", "content": "Write a Python function to compute the Levenshtein distance between two strings."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### 3. Fill-in-the-Middle (FIM) Code Completion

```python
prefix = "def calculate_circle_area(radius: float) -> float:\n    \"\"\"Compute area of circle.\"\"\"\n    if radius < 0:\n        raise ValueError('Radius cannot be negative')\n"
suffix = "\n    return area\n"

# Construct FIM prompt using special tokens
fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

inputs = tokenizer(fim_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.1,
        do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|fim_middle|>"),
    )

infilled = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Infilled code:\n", infilled)
```

---

## Training Data & Methodology

### Stage 1: Continued Pre-Training on [KapCode-1B](https://huggingface.co/datasets/kaptaan45/KapCode-1B) (1B Tokens)

KapCode-1B unifies 5 high-signal data domains with exact SHA-256 deduplication, language filtering via FastText, and 50% FIM formatting:

| Partition | Upstream Source | Share | Tokens | Key Characteristics |
| :--- | :--- | :---:| :---:| :--- |
| **Source Code** | `HuggingFaceCode/stack-v3-train` | **35%** | 350M | 13 programming languages filtered for high quality |
| **Technical Documentation** | `HuggingFaceCode/stack-v3-train` | **20%** | 200M | READMEs, Markdown guides, API references, architecture docs |
| **Function-Level Code** | `Fsoft-AIC/the-vault-function` | **20%** | 200M | Functions with docstrings, type annotations, and return types |
| **High-Quality Web** | `epfml/FineWeb-HQ` | **15%** | 150M | Top-tier STEM and educational English web documents |
| **Mathematical Reasoning** | `open-web-math/open-web-math` | **10%** | 100M | LaTeX equations, proofs, and formal STEM literature |

### Stage 2: Supervised Fine-Tuning on [KapInstruct-100M](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M) (100M Tokens)

KapInstruct-100M is composed of 12 balanced instruction datasets spanning coding synthesis, debugging, multi-turn reasoning, and STEM QA formatted with **Qwen ChatML** and **assistant-only loss masking**:
- **Code Generation & Programming [31%]**: Magicoder-Evol (13%), Magicoder-OSS (8%), Self-OSS-Instruct (5%), Smol-Constraints (3%).
- **General Dialogue & Reasoning [27%]**: Smol-Magpie-Ultra (18%), OpenHermes-2.5 (9%).
- **Mathematical Reasoning (CoT) [17%]**: OpenMathInstruct-2 (11%), NuminaMath-CoT (6%).
- **Debugging & Error Repair [10%]**: CodeFeedback-Filtered (10%).
- **STEM & Scientific QA [11%]**: OpenThoughts-114k (7%), WebInstructSub (4%).
- **Strict Constraint Adherence [4%]**: Tulu-3-SFT (6%), Smol-Constraints (3%).

---

## Hardware & Training Infrastructure

- **Triton Liger Kernel**: Fuses cross-entropy computation directly with output projection, saving 40%–60% VRAM during backward passes.
- **PyTorch SDPA & FlashAttention**: High-throughput memory-efficient scaled dot-product attention on GPU.
- **Google TPU v5e-8 PJRT Runtime**: Distributed BF16 execution across 8 TPU cores (128GB HBM) with static shape compilation.

---

## Related Repositories and Datasets

- **GitHub Repository**: [rudy-07/QaptaanLM-0.75B](https://github.com/rudy-07/QaptaanLM-0.75B)
- **CPT Dataset (GitHub)**: [rudy-07/KapCode-1B](https://github.com/rudy-07/KapCode-1B)
- **CPT Dataset (Hugging Face)**: [kaptaan45/KapCode-1B](https://huggingface.co/datasets/kaptaan45/KapCode-1B)
- **SFT Dataset (GitHub)**: [rudy-07/KapInstruct-100M](https://github.com/rudy-07/KapInstruct-100M)
- **SFT Dataset (Hugging Face)**: [kaptaan45/KapInstruct-100M](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)
- **Kaggle Dataset (KapCode-1B)**: [kaptaan45/kapcode-1b](https://www.kaggle.com/datasets/kaptaan45/kapcode-1b)
- **Kaggle Dataset (KapInstruct-100M)**: [kaptaan45/kapinstruct-100m](https://www.kaggle.com/datasets/kaptaan45/kapinstruct-100m)

---

## Citation

```bibtex
@misc{qaptaanlm2026,
  title   = {{QaptaanLM-0.75B}: Efficient Hybrid-Attention Language Model for Code and Technical Reasoning},
  author  = {Rudy and Contributors},
  year    = {2026},
  publisher = {Hugging Face},
  url     = {https://huggingface.co/kaptaan45/QaptaanLM-0.75B}
}
```

```bibtex
@misc{kapcode1b2026,
  title   = {{KapCode-1B}: A Curated 1-Billion Token Dataset for Compact Code Models},
  author  = {Rudy and Contributors},
  year    = {2026},
  publisher = {Hugging Face},
  url     = {https://huggingface.co/datasets/kaptaan45/KapCode-1B}
}
```

```bibtex
@misc{kapinstruct100m2026,
  title   = {{KapInstruct-100M}: A Curated 100-Million Token Multi-Source Instruction Tuning Dataset for Compact Models},
  author  = {Rudy and Contributors},
  year    = {2026},
  publisher = {Hugging Face},
  url     = {https://huggingface.co/datasets/kaptaan45/KapInstruct-100M}
}
```

---

## License

This model and its code are released under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0). Upstream base model weights and architecture are adapted from `Qwen/Qwen3.5-0.8B-Base` by the Qwen Team (Alibaba Cloud).
