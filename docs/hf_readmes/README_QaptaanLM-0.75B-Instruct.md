---
license: apache-2.0
base_model: kaptaan45/QaptaanLM-0.75B
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
- instruction-tuning
- chatml
- kapinstruct
- text-generation
datasets:
- kaptaan45/KapInstruct-100M
pipeline_tag: text-generation
library_name: transformers
---

# QaptaanLM-0.75B-Instruct: Compact Hybrid-Attention AI Programming Assistant

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-752M%20(Text--Only)-blue.svg)](#model-specification)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20DeltaNet%20%2B%20GQA-purple.svg)](#architecture)
[![Context Length](https://img.shields.io/badge/Context-256K%20Native-orange.svg)](#model-specification)
[![Base CPT Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base%20Model-kaptaan45%2FQaptaanLM--0.75B-blue.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)
[![SFT Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20SFT%20Dataset-kaptaan45%2FKapInstruct--100M-orange.svg)](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)

**QaptaanLM-0.75B-Instruct** is the instruction-aligned programming and technical reasoning assistant built on `QaptaanLM-0.75B`. Fine-tuned on **[KapInstruct-100M](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)** using **Qwen ChatML** formatting and **assistant-only loss masking**, the model excels at multi-language code generation, algorithmic reasoning, debugging, and constraint-based instruction following.

---

## 🌐 Model Ecosystem & Deployment Formats

| Format / Variant | Repository | Target Use-Case |
| :--- | :--- | :--- |
| **Base CPT Model (Safetensors)** | [`kaptaan45/QaptaanLM-0.75B`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B) | Raw foundation base, code completion, FIM infilling |
| **GGUF Instruct (13 Quants + Modelfiles)** | [`kaptaan45/QaptaanLM-0.75B-Instruct-GGUF`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct-GGUF) | Desktop / edge chat via Ollama & llama.cpp |
| **GGUF Base (13 Quants)** | [`kaptaan45/QaptaanLM-0.75B-GGUF`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-GGUF) | Local CPU / llama.cpp code completion |
| **BitsAndBytes Instruct (4-bit & 8-bit)** | [`kaptaan45/QaptaanLM-0.75B-Instruct-BnB`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct-BnB) | Low-VRAM CUDA instruction serving (~730 MB VRAM) |
| **BitsAndBytes Base (4-bit & 8-bit)** | [`kaptaan45/QaptaanLM-0.75B-BnB`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-BnB) | Low-VRAM CUDA code completion (~730 MB VRAM) |
| **ONNX Runtime Instruct** | [`kaptaan45/QaptaanLM-0.75B-Instruct-ONNX`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct-ONNX) | Client-side WebGPU chat, Transformers.js |
| **ONNX Runtime Base** | [`kaptaan45/QaptaanLM-0.75B-ONNX`](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-ONNX) | In-browser IDE autocomplete, WebGPU, edge runtimes |

---

## Model Specification

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Model Name** | QaptaanLM-0.75B-Instruct | Instruction-tuned text-only causal language model |
| **Base Model** | `kaptaan45/QaptaanLM-0.75B` | 752M dense parameter foundation model |
| **Total Parameters** | **752,382,976 (752M)** | Tied input/output word embeddings (`tie_word_embeddings=True`) |
| **Hidden Size ($d_{model}$)** | 1024 | Base hidden dimension |
| **Intermediate Size ($d_{ffn}$)** | 3584 | SwiGLU activation function |
| **Total Layers** | 24 | 18 Linear Attention + 6 Full Attention layers (3:1 ratio) |
| **Full Attention Heads** | 8 Query / 2 Key-Value | Grouped-Query Attention (4:1 query-to-KV ratio) |
| **Linear Attention Heads** | 16 QK / 16 V | Gated DeltaNet (128 head dim, conv kernel dim 4) |
| **Native Context Length** | 262,144 tokens (256K native) | Interleaved M-RoPE ($\theta = 10,000,000$) |
| **Prompt Template** | Qwen ChatML | `<|im_start|>system/user/assistant`, `<|im_end|>` |
| **Precision Support** | `bfloat16`, `fp16`, `float32` | Native BF16 execution on modern GPUs & TPUs |

---

## Recommended Generation Parameters (SFT Instruct)

```python
generation_config = {
    "do_sample": True,
    "temperature": 0.20,              # Optimal balance of determinism & creative reasoning
    "top_p": 0.90,                    # Nucleus sampling threshold
    "top_k": 40,                      # Restricts to top-40 candidate tokens
    "repetition_penalty": 1.12,       # Prevents repetitive code generation loops
    "eos_token_id": [248044, 248046], # <|endoftext|> (248044) and <|im_end|> (248046)
    "pad_token_id": 248044
}
```

---

## Quickstart & Usage

### 1. ChatML Dialogue Inference via `transformers`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "kaptaan45/QaptaanLM-0.75B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are QaptaanLM, an expert AI programming assistant."},
    {"role": "user", "content": "Write a Python function `def is_palindrome(s: str) -> bool:` with docstring and examples."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.20,
        top_p=0.90,
        repetition_penalty=1.12,
        eos_token_id=[248044, 248046],
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Dataset Attribution

Fine-tuned on **[KapInstruct-100M](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)** across 12 diverse instruction domains:
- **Code Generation & Programming [31%]**: Magicoder-Evol (13%), Magicoder-OSS (8%), Self-OSS-Instruct (5%), Smol-Constraints (3%).
- **General Dialogue & Reasoning [27%]**: Smol-Magpie-Ultra (18%), OpenHermes-2.5 (9%).
- **Mathematical Reasoning (CoT) [17%]**: OpenMathInstruct-2 (11%), NuminaMath-CoT (6%).
- **Debugging & Error Repair [10%]**: CodeFeedback-Filtered (10%).
- **STEM & Scientific QA [11%]**: OpenThoughts-114k (7%), WebInstructSub (4%).
- **Strict Constraint Adherence [4%]**: Tulu-3-SFT (6%), Smol-Constraints (3%).

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**.
