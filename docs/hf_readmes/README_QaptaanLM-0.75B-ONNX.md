---
license: apache-2.0
base_model: kaptaan45/QaptaanLM-0.75B
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
- continued-pretraining
pipeline_tag: text-generation
---

# QaptaanLM-0.75B-ONNX (CPT Base)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base%20Model-kaptaan45%2FQaptaanLM--0.75B-blue.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)

This repository contains exported **ONNX Runtime graph weights** for **[QaptaanLM-0.75B](https://huggingface.co/kaptaan45/QaptaanLM-0.75B)** (Base CPT foundation model), designed for lightweight in-browser IDE autocompletion and edge device code inference.

---

## 🌐 Quickstart: Python `onnxruntime`

```python
import os
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_id = "kaptaan45/QaptaanLM-0.75B-ONNX"

model_dir = snapshot_download(repo_id=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

session = ort.InferenceSession(
    os.path.join(model_dir, "model.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

prompt = 'def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s is a palindrome.\"\"\"\n    '
input_ids = tokenizer.encode(prompt, return_tensors="np")

outputs = session.run(None, {"input_ids": input_ids})
logits = outputs[0]
next_token = int(np.argmax(logits[0, -1, :]))
print("Next Token ID:", next_token, "Decoded:", repr(tokenizer.decode([next_token])))
```

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**.
