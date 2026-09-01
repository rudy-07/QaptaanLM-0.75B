---
license: apache-2.0
base_model: kaptaan45/QaptaanLM-0.75B-Instruct
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
- instruction-tuning
pipeline_tag: text-generation
---

# QaptaanLM-0.75B-Instruct-GGUF

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Instruct%20Model-kaptaan45%2FQaptaanLM--0.75B--Instruct-purple.svg)](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)

This repository contains official **GGUF quantized formats** and turnkey **Ollama Modelfiles** for **[QaptaanLM-0.75B-Instruct](https://huggingface.co/kaptaan45/QaptaanLM-0.75B-Instruct)**, an efficient 752M hybrid linear attention foundation programming model.

---

## 📦 Quantization Matrix & Download Sizes

| # | Quantization | File Name | Size | Memory Required | Quality & Description |
|---|:---|:---|:---:|:---:|:---|
| 1 | **FP16** | `qaptaan-0.75b-instruct-f16.gguf` | **1.45 GB** | ~2.0 GB | Full 16-bit float unquantized baseline master. |
| 2 | **BF16** | `qaptaan-0.75b-instruct-bf16.gguf` | **1.45 GB** | ~2.0 GB | 16-bit BFloat precision GGUF. |
| 3 | **Q8_0** | `qaptaan-0.75b-instruct-q8_0.gguf` | **774 MB** | ~1.1 GB | 8-bit integer quantization (99.9% quality retention). |
| 4 | **Q6_K** | `qaptaan-0.75b-instruct-q6_k.gguf` | **600 MB** | ~900 MB | 6-bit K-quantization for high-fidelity code synthesis. |
| 5 | **Q5_K_M** | `qaptaan-0.75b-instruct-q5_k_m.gguf` | **551 MB** | ~800 MB | 5-bit K-quantization (Medium tensor split - Sweet Spot). |
| 6 | **Q5_K_S** | `qaptaan-0.75b-instruct-q5_k_s.gguf` | **537 MB** | ~780 MB | 5-bit K-quantization (Small tensor split). |
| 7 | **Q5_0** | `qaptaan-0.75b-instruct-q5_0.gguf` | **537 MB** | ~780 MB | 5-bit standard legacy quantization. |
| 8 | **Q4_K_M** ⭐ | `qaptaan-0.75b-instruct-q4_k_m.gguf` | **504 MB** | ~720 MB | **Recommended**. Best balance of speed, size, and reasoning quality. |
| 9 | **Q4_K_S** | `qaptaan-0.75b-instruct-q4_k_s.gguf` | **481 MB** | ~700 MB | 4-bit K-quantization (Small). |
| 10 | **Q4_0** | `qaptaan-0.75b-instruct-q4_0.gguf` | **478 MB** | ~700 MB | 4-bit legacy quantization. |
| 11 | **Q3_K_M** | `qaptaan-0.75b-instruct-q3_k_m.gguf` | **444 MB** | ~650 MB | 3-bit K-quantization (Medium). |
| 12 | **Q3_K_S** | `qaptaan-0.75b-instruct-q3_k_s.gguf` | **415 MB** | ~600 MB | 3-bit K-quantization (Small). |
| 13 | **Q2_K** | `qaptaan-0.75b-instruct-q2_k.gguf` | **402 MB** | ~550 MB | 2-bit extreme compression for ultra low-memory micro-devices. |

---

## 🚀 Quickstart: Running with Ollama

Every quantization format has a corresponding pre-configured `Modelfile` (e.g., `Modelfile-Q4_K_M`, `Modelfile-Q8_0`):

```bash
# 1. Download the desired GGUF weights and Modelfile
huggingface-cli download kaptaan45/QaptaanLM-0.75B-Instruct-GGUF qaptaan-0.75b-instruct-q4_k_m.gguf Modelfile-Q4_K_M --local-dir ./qaptaan

# 2. Create the Ollama model
ollama create qaptaan-instruct -f ./qaptaan/Modelfile-Q4_K_M

# 3. Chat with QaptaanLM
ollama run qaptaan-instruct "Write a Python function to compute the Fibonacci sequence using dynamic programming."
```

---

## 🛠️ Quickstart: Running with `llama.cpp`

### Interactive CLI Chat:
```bash
./llama-cli -m qaptaan-0.75b-instruct-q4_k_m.gguf \
  -p "<|im_start|>system\nYou are QaptaanLM, an expert AI programming assistant.<|im_end|>\n<|im_start|>user\nWrite a Rust function for binary search.<|im_end|>\n<|im_start|>assistant\n" \
  -n 256 --temp 0.20 --repeat-penalty 1.12 -r "<|im_end|>"
```

### Local OpenAI-Compatible Server:
```bash
./llama-server -m qaptaan-0.75b-instruct-q4_k_m.gguf --port 8080 -c 4096
```

---

## License

Released under the **[Apache 2.0 License](https://opensource.org/licenses/Apache-2.0)**.
