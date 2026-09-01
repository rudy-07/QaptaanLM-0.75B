---
license: apache-2.0
base_model: kaptaan45/QaptaanLM-0.75B-Instruct
language:
- en
- code
tags:
- bitsandbytes
- 4bit
- 8bit
- nf4
- int8
- code
- causal-lm
- qwen
- qwen3.5
- hybrid-attention
- instruction-tuning
pipeline_tag: text-generation
---

# QaptaanLM-0.75B-Instruct-BnB

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Instruct Model](https://img.shields.io/badge/%F0%9F%A4%97%20Instruct%20Model-kaptaan45%2FQaptaanLM--0.75B--Instruct-purple.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)

This repository contains pre-quantized **BitsAndBytes (4-Bit NF4 and 8-Bit Int8)** weights for **[QaptaanLM-0.75B-Instruct](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct)**, enabling low-latency GPU serving on graphics cards with **< 1 GB VRAM**.

---

## 📦 Directory Structure

```text
kaptaan45/QaptaanLM-0.75B-Instruct-BnB/
├── 4bit/
│   ├── model.safetensors       # 731 MB (4-bit NF4 quantized MLP projections + BF16 attention/norms)
│   ├── config.json             # 4-bit model configuration
│   └── generation_config.json  # temperature=0.20, repetition_penalty=1.12
└── 8bit/
    ├── model.safetensors       # 962 MB (8-bit Int8 quantized projections + SCB scales)
    ├── config.json             # 8-bit model configuration
    └── generation_config.json  # temperature=0.20, repetition_penalty=1.12
```

---

## ⚡ Quickstart: Loading with `transformers` & `bitsandbytes`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure 4-bit NF4 quantization on CUDA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "kaptaan45/QaptaanLM-0.75B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are QaptaanLM, an expert AI programming assistant."},
    {"role": "user", "content": "Write a Python function `def is_palindrome(s: str) -> bool:` with docstring."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.20,
        top_p=0.90,
        repetition_penalty=1.12,
        eos_token_id=[248044, 248046],
    )

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**.
