---
license: apache-2.0
base_model: kaptaan45/QaptaanLM-0.75B
language:
- en
- code
tags:
- gguf
- ollama
- llama.cpp
- quantized
- code
- causal-lm
- qwen
- qwen3.5
- hybrid-attention
- continued-pretraining
pipeline_tag: text-generation
---

# QaptaanLM-0.75B-GGUF (CPT Base)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base%20Model-kaptaan45%2FQaptaanLM--0.75B-blue.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B)
[![Instruct GGUF](https://img.shields.io/badge/%F0%9F%A4%97%20Instruct%20GGUF-kaptaan45%2FQaptaanLM--0.75B--Instruct--GGUF-purple.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct-GGUF)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)

This repository contains official **GGUF quantized formats** for **[QaptaanLM-0.75B](https://huggingface.co/kaptaan45/QaptaanLM-0.75B)** (Base CPT foundation model), optimized for fast code continuation and Fill-in-the-Middle (FIM) infilling in local IDEs and edge environments.

---

## 📦 Quantization Matrix & Download Sizes

| # | Quantization | File Name | Size | Memory Required | Quality & Description |
|---|:---|:---|:---:|:---:|:---|
| 1 | **FP16** | `qaptaan-0.75b-cpt-f16.gguf` | **1.45 GB** | ~2.0 GB | Full 16-bit float unquantized baseline master. |
| 2 | **BF16** | `qaptaan-0.75b-cpt-bf16.gguf` | **1.45 GB** | ~2.0 GB | 16-bit BFloat precision GGUF. |
| 3 | **Q8_0** | `qaptaan-0.75b-cpt-q8_0.gguf` | **774 MB** | ~1.1 GB | 8-bit integer quantization (99.9% quality retention). |
| 4 | **Q6_K** | `qaptaan-0.75b-cpt-q6_k.gguf` | **600 MB** | ~900 MB | 6-bit K-quantization for high-fidelity code completion. |
| 5 | **Q5_K_M** | `qaptaan-0.75b-cpt-q5_k_m.gguf` | **551 MB** | ~800 MB | 5-bit K-quantization (Medium tensor split). |
| 6 | **Q5_K_S** | `qaptaan-0.75b-cpt-q5_k_s.gguf` | **537 MB** | ~780 MB | 5-bit K-quantization (Small tensor split). |
| 7 | **Q5_0** | `qaptaan-0.75b-cpt-q5_0.gguf` | **537 MB** | ~780 MB | 5-bit standard legacy quantization. |
| 8 | **Q4_K_M** ⭐ | `qaptaan-0.75b-cpt-q4_k_m.gguf` | **504 MB** | ~720 MB | **Recommended**. Best balance of speed, size, and completion accuracy. |
| 9 | **Q4_K_S** | `qaptaan-0.75b-cpt-q4_k_s.gguf` | **481 MB** | ~700 MB | 4-bit K-quantization (Small). |
| 10 | **Q4_0** | `qaptaan-0.75b-cpt-q4_0.gguf` | **478 MB** | ~700 MB | 4-bit legacy quantization. |
| 11 | **Q3_K_M** | `qaptaan-0.75b-cpt-q3_k_m.gguf` | **444 MB** | ~650 MB | 3-bit K-quantization (Medium). |
| 12 | **Q3_K_S** | `qaptaan-0.75b-cpt-q3_k_s.gguf` | **415 MB** | ~600 MB | 3-bit K-quantization (Small). |
| 13 | **Q2_K** | `qaptaan-0.75b-cpt-q2_k.gguf` | **402 MB** | ~550 MB | 2-bit extreme compression. |

---

## 🛠️ Usage with `llama.cpp`

### Code Prefix Completion:
```bash
./llama-cli -m qaptaan-0.75b-cpt-q4_k_m.gguf \
  -p "def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s is a palindrome.\"\"\"\n    " \
  -n 64 --temp 0.15 --repeat-penalty 1.10
```

### Fill-in-the-Middle (FIM) Completion:
```bash
./llama-cli -m qaptaan-0.75b-cpt-q4_k_m.gguf \
  -p "<|fim_prefix|>def compute_area(r: float) -> float:\n<|fim_suffix|>\n    return area<|fim_middle|>" \
  -n 32 --temp 0.10
```

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**.
