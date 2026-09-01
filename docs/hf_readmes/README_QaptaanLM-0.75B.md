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
- kapcode
- text-generation
datasets:
- kaptaan45/KapCode-1B
pipeline_tag: text-generation
library_name: transformers
---

# QaptaanLM-0.75B: Efficient Hybrid-Attention Foundation Base Model

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-752M%20(Text--Only)-blue.svg)](#model-specification)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20DeltaNet%20%2B%20GQA-purple.svg)](#architecture)
[![Context Length](https://img.shields.io/badge/Context-256K%20Native-orange.svg)](#model-specification)
[![Instruct SFT Model](https://img.shields.io/badge/%F0%9F%A4%97%20Instruct%20Model-kaptaan45%2FQaptaanLM--0.75B--Instruct-purple.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)
[![CPT Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20CPT%20Dataset-kaptaan45%2FKapCode--1B-yellow.svg)](https://huggingface.co/datasets/kaptaan45/KapCode-1B)

**QaptaanLM-0.75B** is a compact, high-efficiency hybrid-attention foundation language model optimized for source code synthesis, technical reasoning, and long-context code comprehension. 

Engineered by stripping the visual transformer from `Qwen/Qwen3.5-0.8B-Base` down to **752M dense parameters**, QaptaanLM achieves state-of-the-art computational and memory efficiency on consumer GPUs and edge accelerators. It couples linear-complexity recurrence layers with dense multi-head attention and was trained on **[KapCode-1B](https://huggingface.co/datasets/kaptaan45/KapCode-1B)** (1-billion-token curated code, doc, and STEM corpus with 50% Fill-in-the-Middle infilling on Google TPU v5e-8).

---

## 🌐 Model Ecosystem & Formats

| Format / Variant | Repository | Target Use-Case |
| :--- | :--- | :--- |
| **SFT Instruct Model (ChatML)** | [`kaptaan45/QaptaanLM-0.75B-Instruct`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct) | Conversational programming assistant, debugging, ChatML instruction following |
| **GGUF Base (13 Quants)** | [`kaptaan45/QaptaanLM-0.75B-GGUF`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-GGUF) | Local CPU / llama.cpp code completion |
| **GGUF Instruct (13 Quants + Modelfiles)** | [`kaptaan45/QaptaanLM-0.75B-Instruct-GGUF`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct-GGUF) | Desktop / edge chat via Ollama & llama.cpp |
| **BitsAndBytes Base (4-bit & 8-bit)** | [`kaptaan45/QaptaanLM-0.75B-BnB`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-BnB) | Low-VRAM CUDA code completion (~730 MB VRAM) |
| **BitsAndBytes Instruct (4-bit & 8-bit)** | [`kaptaan45/QaptaanLM-0.75B-Instruct-BnB`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct-BnB) | Low-VRAM CUDA instruction serving (~730 MB VRAM) |
| **ONNX Runtime Base** | [`kaptaan45/QaptaanLM-0.75B-ONNX`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-ONNX) | In-browser IDE autocomplete, WebGPU, edge runtimes |
| **ONNX Runtime Instruct** | [`kaptaan45/QaptaanLM-0.75B-Instruct-ONNX`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct-ONNX) | Client-side WebGPU chat, Transformers.js |

---

## Model Specification

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Model Name** | QaptaanLM-0.75B | Text-only causal language model |
| **Base Architecture** | `Qwen/Qwen3.5-0.8B-Base` | Vision transformer stripped via `Qwen3_5ForCausalLM` |
| **Total Parameters** | **752,382,976 (752M)** | Tied input and output word embeddings (`tie_word_embeddings=True`) |
| **Trainable Parameters** | 752,382,976 | Full-parameter Continued Pre-Training (no LoRA) |
| **Hidden Size ($d_{model}$)** | 1024 | Base hidden dimension |
| **Intermediate Size ($d_{ffn}$)** | 3584 | SwiGLU activation function |
| **Total Layers** | 24 | 18 Linear Attention + 6 Full Attention layers (3:1 ratio) |
| **Full Attention Heads** | 8 Query / 2 Key-Value | Grouped-Query Attention (4:1 query-to-KV ratio) |
| **Full Attention Head Dim** | 256 | Query head dimension |
| **Linear Attention Heads** | 16 QK / 16 V | Gated DeltaNet (128 head dim, conv kernel dim 4) |
| **Native Context Length** | 262,144 tokens (256K native) | Interleaved M-RoPE ($\theta = 10,000,000$, partial rotary factor 0.25) |
| **Vocabulary Size** | 248,320 tokens | Tied word embeddings |
| **Normalization** | RMSNorm ($\epsilon = 1\text{e-}6$) | Pre-layer normalization |
| **Precision Support** | `bfloat16`, `fp16`, `float32` | Native BF16 execution on modern GPUs & TPUs |

---

## Recommended Generation Parameters (CPT Base)

```python
generation_config = {
    "do_sample": False,              # Greedy decoding for exact deterministic code completion
    "temperature": 0.15,             # Low temperature when sampling
    "top_p": 0.90,
    "top_k": 40,
    "repetition_penalty": 1.10,      # Prevents repetition loops
    "eos_token_id": [248044, 248046],# <|endoftext|> and <|im_end|>
    "pad_token_id": 248044
}
```

---

## Quickstart & Usage

### 1. Standard Code Prefix Completion

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

prompt = 'def binary_search(arr: list[int], target: int) -> int:\n    """Return index of target in sorted arr, or -1 if not found."""\n    '
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.10,
        eos_token_id=[248044, 248046],
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Fill-in-the-Middle (FIM) Code Infilling

```python
prefix = "def calculate_circle_area(radius: float) -> float:\n    \"\"\"Compute area of circle.\"\"\"\n    if radius < 0:\n        raise ValueError('Radius cannot be negative')\n"
suffix = "\n    return area\n"

# Format: <|fim_prefix|> Prefix <|fim_suffix|> Suffix <|fim_middle|>
fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
inputs = tokenizer(fim_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|fim_middle|>"),
    )

infilled = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Infilled Code:\n", infilled)
```

---

## Dataset Attribution

Trained on **[KapCode-1B](https://huggingface.co/datasets/kaptaan45/KapCode-1B)** (1,000,013,824 tokens) across 5 curated domains:
- **Source Code (35%)**: Multi-language code from The Stack v3.
- **Technical Documentation (20%)**: READMEs, Markdown guides, API references.
- **Function-Level Code (20%)**: Annotated functions from The Vault.
- **High-Quality Web (15%)**: STEM educational web documents from FineWeb-HQ.
- **Mathematical Reasoning (10%)**: LaTeX equations and proofs from OpenWebMath.

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**. Upstream base model weights and architecture adapted from `Qwen/Qwen3.5-0.8B-Base` by the Qwen Team (Alibaba Cloud).
