---
license: apache-2.0
base_model: kaptaan45/QaptaanLM-0.75B-Instruct
language:
- en
- code
tags:
- onnx
- onnxruntime
- webgpu
- browser
- code
- causal-lm
- qwen
- qwen3.5
- hybrid-attention
- instruction-tuning
pipeline_tag: text-generation
---

# QaptaanLM-0.75B-Instruct-ONNX

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Instruct Model](https://img.shields.io/badge/%F0%9F%A4%97%20Instruct%20Model-kaptaan45%2FQaptaanLM--0.75B--Instruct-purple.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)

This repository contains exported **ONNX Runtime graph weights** for **[QaptaanLM-0.75B-Instruct](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct)**, optimized for in-browser client-side execution via WebGPU, Node.js, and cross-platform desktop ONNX runtimes.

---

## 🌐 Quickstart: Python `onnxruntime`

```python
import os
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_id = "kaptaan45/QaptaanLM-0.75B-Instruct-ONNX"

# Download the complete model directory (model.onnx + tensor shards)
model_dir = snapshot_download(repo_id=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Initialize ONNX Runtime session (supports CUDAExecutionProvider or CPUExecutionProvider)
session = ort.InferenceSession(
    os.path.join(model_dir, "model.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

prompt = "<|im_start|>system\\nYou are QaptaanLM, an expert AI programming assistant.<|im_end|>\\n<|im_start|>user\\nWrite a Python function `def is_palindrome(s: str) -> bool:` with docstring.<|im_end|>\\n<|im_start|>assistant\\n"
input_ids = tokenizer.encode(prompt, return_tensors="np")

outputs = session.run(None, {"input_ids": input_ids})
logits = outputs[0]  # shape: (batch_size, sequence_length, 248320)
next_token = int(np.argmax(logits[0, -1, :]))
print("Next Token ID:", next_token, "Decoded:", repr(tokenizer.decode([next_token])))
```

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**.
