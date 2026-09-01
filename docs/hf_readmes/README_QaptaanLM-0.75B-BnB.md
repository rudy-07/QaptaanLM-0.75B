---
license: apache-2.0
base_model: kaptaan45/QaptaanLM-0.75B
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
- continued-pretraining
pipeline_tag: text-generation
---

# QaptaanLM-0.75B-BnB (CPT Base)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base%20Model-kaptaan45%2FQaptaanLM--0.75B-blue.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)

This repository contains pre-quantized **BitsAndBytes (4-Bit NF4 and 8-Bit Int8)** weights for **[QaptaanLM-0.75B](https://huggingface.co/kaptaan45/QaptaanLM-0.75B)** (Base CPT foundation model), enabling low-VRAM code completion on consumer GPUs with **< 1 GB VRAM**.

---

## 📦 Directory Structure

```text
kaptaan45/QaptaanLM-0.75B-BnB/
├── 4bit/
│   ├── model.safetensors       # 731 MB (4-bit NF4 quantized weights)
│   ├── config.json             # 4-bit model configuration
│   └── generation_config.json  # temperature=0.15, repetition_penalty=1.10
└── 8bit/
    ├── model.safetensors       # 962 MB (8-bit Int8 quantized weights)
    ├── config.json             # 8-bit model configuration
    └── generation_config.json  # temperature=0.15, repetition_penalty=1.10
```

---

## ⚡ Quickstart: Loading with `transformers` & `bitsandbytes`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "kaptaan45/QaptaanLM-0.75B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

prompt = 'def is_palindrome(s: str) -> bool:\n    """Return True if s is a palindrome."""\n    '
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=False,
        repetition_penalty=1.10,
        eos_token_id=[248044, 248046],
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**.
