---
license: other
task_categories:
- question-answering
- text-generation
language:
- en
- code
tags:
- instruction-tuning
- sft
- chatml
- code
- python
- typescript
- javascript
- cpp
- csharp
- java
- rust
- go
- math
- reasoning
- cot
- debugging
- qwen
- assistant-only
- kapinstruct
- smoltalk
- magicoder
- openmathinstruct
- numinamath
- openthoughts
- openhermes
- tulu-3
- starcoder
- webinstruct
- codefeedback
size_categories:
- 100M-1B
configs:
- config_name: default
  data_files:
  - split: train
    path: "*.arrow"
dataset_info:
  features:
  - name: input_ids
    sequence: int32
  - name: attention_mask
    sequence: int8
  - name: labels
    sequence: int32
  splits:
  - name: train
    num_bytes: 429496729
    num_examples: 24414
  download_size: 214748364
  dataset_size: 429496729
---

# KapInstruct-100M: Curated 100-Million Token Instruction Tuning Dataset

<p align="center">
  <img src="https://huggingface.co/datasets/kaptaan45/KapInstruct-100M/resolve/main/kapinstruct_cover_image.jpg" width="100%" alt="KapInstruct-100M Banner">
</p>

[![License](https://img.shields.io/badge/License-Source--Specific%20(Open)-green.svg)](#licensing-and-provenance)
[![Tokens](https://img.shields.io/badge/Usable%20Tokens-100%20Million-blue.svg)](#dataset-composition)
[![Dialogue Format](https://img.shields.io/badge/Format-ChatML%20%7C%20Qwen-orange.svg)](#chatml-formatting--loss-masking)
[![Loss Policy](https://img.shields.io/badge/Loss%20Masking-Assistant--Only-red.svg)](#chatml-formatting--loss-masking)
[![GitHub](https://img.shields.io/badge/GitHub-KapInstruct--100M-181717.svg?logo=github)](https://github.com/rudy-07/KapInstruct-100M)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kaptaan45%2FKapInstruct--100M-orange.svg)](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-kaptaan45%2Fkapinstruct--100m-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/datasets/kaptaan45/kapinstruct-100m)
[![Starter Notebook](https://img.shields.io/badge/Kaggle-Quickstart%20Notebook-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/code/kaptaan45/kapinstruct-100m-dataset-exploration-quickstart)
[![Builder Notebook](https://img.shields.io/badge/Kaggle-Builder%20Notebook-blueviolet.svg?logo=kaggle)](https://www.kaggle.com/code/kaptaan45/kapinstruct-100m-dataset-builder-hf-publisher)
[![Associated Model](https://img.shields.io/badge/Model-QaptaanLM--0.75B-purple.svg)](https://github.com/rudy-07/QaptaanLM-0.75B)

**KapInstruct-100M** is a high-fidelity, 100-million-token instruction-tuning dataset engineered for **Supervised Fine-Tuning (SFT)** and alignment of compact language models (under 1 billion parameters). Formatted with the **Qwen ChatML** chat template and tokenized using `Qwen/Qwen3.5-0.8B-Base`, the dataset enforces strict **assistant-only loss masking** (masking user prompts and structural delimiters to `-100`) to maximize training efficiency.

KapInstruct-100M unifies 12 balanced, high-signal instruction sources spanning programming synthesis, step-by-step mathematical reasoning (Chain-of-Thought), technical STEM QA, multi-turn dialogue, strict constraint following, and interactive code debugging/repair.

---

## Dataset Overview

- **Hugging Face Repository**: [`kaptaan45/KapInstruct-100M`](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)
- **Kaggle Dataset**: [`kaptaan45/kapinstruct-100m`](https://www.kaggle.com/datasets/kaptaan45/kapinstruct-100m)
- **Total Usable Tokens**: **100,000,000 tokens** post-filtering, normalization, and deduplication
- **Packed Sequence Length**: 4096 tokens per packed sequence
- **Tokenizer**: `Qwen/Qwen3.5-0.8B-Base` (248,044 BPE vocabulary)
- **Loss Masking Policy**: `assistant_only` (prompts, system messages, and `<|im_start|>` headers have `labels = -100`; loss is computed strictly on assistant response spans)
- **Primary Storage Formats**: Memory-mapped Apache Arrow IPC (`.arrow`) and Apache Parquet (`.parquet`)
- **Primary Use Case**: SFT / Instruction Tuning for compact code and reasoning models such as [QaptaanLM-0.75B](https://github.com/rudy-07/QaptaanLM-0.75B).

---

## Motivation & Design Principles

Supervised fine-tuning of compact models (0.5B to 1.5B parameters) is highly sensitive to data quality and token loss allocation:

1. **Assistant-Only Loss Masking**: Standard causal LM training over unmasked instruction data wastes gradient updates predicting user prompts and static system headers. By masking all non-assistant tokens to `-100`, 100% of gradient updates focus on assistant reasoning, syntax accuracy, and answer generation.
2. **Deficit-Weighted Balanced Scheduling**: Rather than concatenating disparate dumps, KapInstruct-100M uses a deficit-driven sampling scheduler that measures exact tokenizer tokens post-filtering, guaranteeing precise representation across all 12 domains.
3. **Cross-Source Global Deduplication**: Full multi-turn dialogues are canonicalized and indexed via SHA-256 to eliminate prompt leaks, dataset overlaps, and synthetic duplicates across independent upstream sources.
4. **Rich Reasoning Traces (Chain-of-Thought)**: Mathematics and STEM partitions retain detailed step-by-step reasoning solutions, empowering compact models to learn structured problem breakdown.

---

## Dataset Composition & Source Mixture

KapInstruct-100M is composed of 12 verified upstream sources sampled according to strict token budgets:

| Source | Domain / Category | Share | Tokens | License |
| :--- | :--- | :---:| :---:| :--- |
| **Smol-Magpie-Ultra** | General reasoning & conversation | **18%** | 18,000,000 | Apache-2.0 |
| **Magicoder-Evol** | Complex programming instructions | **13%** | 13,000,000 | Apache-2.0 |
| **OpenMathInstruct-2** | Math problem solving & synthesis | **11%** | 11,000,000 | CC-BY-4.0 |
| **CodeFeedback-Filtered** | Bug fixing & code repair | **10%** | 10,000,000 | Apache-2.0 |
| **OpenHermes-2.5** | Broad conversational QA | **9%** | 9,000,000 | MIT |
| **Magicoder-OSS** | Open-source code generation | **8%** | 8,000,000 | MIT |
| **OpenThoughts-114k** | General & STEM reasoning | **7%** | 7,000,000 | Apache-2.0 |
| **NuminaMath-CoT** | Competition math reasoning | **6%** | 6,000,000 | Apache-2.0 |
| **Tulu-3 SFT** | High-fidelity instruction following | **6%** | 6,000,000 | ODC-By |
| **Self-OSS StarCoder2** | Execution-validated code | **5%** | 5,000,000 | ODC-By |
| **WebInstructSub** | Science & technical QA | **4%** | 4,000,000 | Apache-2.0 |
| **Smol-Constraints** | Strict constraint adherence | **3%** | 3,000,000 | Apache-2.0 |
| **Total** | | **100%** | **100,000,000** | |

---

## Domain Allocation Breakdown

```text
+-------------------------------------------------------------+
|             KapInstruct-100M Domain Allocation              |
+-------------------------------------------------------------+
|  [================]  Code Generation (31% - 31M tokens)     |
|  [==============]    General Reasoning (27% - 27M tokens)   |
|  [=========]         Mathematics CoT (17% - 17M tokens)     |
|  [======]            STEM QA & Science (11% - 11M tokens)   |
|  [=====]             Debugging & Repair (10% - 10M tokens)  |
|  [==]                Constraint Adherence (4% - 4M tokens)  |
+-------------------------------------------------------------+
```

---

## ChatML Formatting & Loss Masking

Each conversation is formatted strictly following the Qwen ChatML schema:

```
<|im_start|>system
You are a helpful and harmless assistant.<|im_end|>
<|im_start|>user
Write a function in Python to compute the Levenshtein distance.<|im_end|>
<|im_start|>assistant
def levenshtein_distance(s1: str, s2: str) -> int:
    ...<|im_end|>
```

### Token-Level Alignment & Masking Verification

| Turn Component | Rendered Token Span | Loss Label (`labels`) | Masking Status |
| :--- | :--- | :--- | :--- |
| **System Turn** | `<|im_start|>system\n...<|im_end|>\n` | `[-100, -100, ...]` | **Masked** |
| **User Turn** | `<|im_start|>user\n...<|im_end|>\n` | `[-100, -100, ...]` | **Masked** |
| **Assistant Header** | `<|im_start|>assistant\n` | `[-100, -100, ...]` | **Masked** |
| **Assistant Content** | `response text...<|im_end|>\n` | `[id_0, id_1, id_2, ...]` | **TRAINABLE** |
| **Sequence Padding** | `<|endoftext|>` infilling | `[-100, -100, ...]` | **Masked** |

In multi-turn dialogues `[User 1 -> Assistant 1 -> User 2 -> Assistant 2]`, loss is computed strictly across `Assistant 1` and `Assistant 2` response spans.

---

## Quality Filtering & Deduplication

1. **Natural Language Filtering**: FastText language identification and English confidence scoring (`min_confidence = 0.65`), with automatic preservation of code-mixed technical dialogues.
2. **Programming Language Normalization**: Canonical alias mapping across 16 core languages (Python, TypeScript, JavaScript, C++, C, C#, Java, Rust, Go, Ruby, PHP, SQL, Shell, HTML, CSS, Dockerfile).
3. **Secret & Key Stripping**: Regex scanning and complete rejection of leaked API keys (OpenAI `sk-`, AWS `AKIA`, GitHub `ghp_`, Slack, JWTs, and private RSA/SSH keys).
4. **LaTeX & Math Integrity**: Rejection of unbalanced LaTeX delimiters (`$$`, `\begin{...}`) and OCR noise artifacts.
5. **Prompt Injection & Repetition Removal**: Scanning and removal of injection jailbreaks, infinite loops, and degenerative repetition.
6. **Global Cross-Source Deduplication**: Exact SHA-256 fingerprinting on normalized dialogue turns across all 12 constituent datasets.

---

## Dataset Loading and Usage

### 1. Zero-Copy Memory-Mapped PyArrow Loading (Fastest)

```python
import glob
import pyarrow as pa
from datasets import load_dataset

# Load Arrow shards directly
shard_files = sorted(glob.glob("data/kapinstruct/*.arrow"))
dataset = load_dataset("arrow", data_files=shard_files, split="train", keep_in_memory=False)

print(f"Total packed sequences: {len(dataset):,}")
sample = dataset[0]
print(f"Sequence length: {len(sample['input_ids'])} tokens")
print(f"Trainable tokens: {sum(1 for l in sample['labels'] if l != -100)}")
```

### 2. Hugging Face Datasets Streaming

```python
from datasets import load_dataset

dataset = load_dataset("kaptaan45/KapInstruct-100M", split="train", streaming=True)
sample = next(iter(dataset))
print("Loaded sequence keys:", list(sample.keys()))
```

### 3. PyTorch Training Loop Integration

```python
import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
for batch in loader:
    # Forward pass computes cross-entropy loss ONLY on labels != -100
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    loss = outputs.loss
    loss.backward()
    break
```

---

## Licensing and Provenance

KapInstruct-100M is a curated composite dataset. Each constituent subset retains its upstream license terms as documented in [`licenses.json`](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M/blob/main/licenses.json):

| Subset / Source | Upstream License | Attribution & Commercial Use |
| :--- | :--- | :--- |
| `smol_magpie_ultra`, `smol_constraints` | Apache-2.0 / Open | HuggingFaceTB / SmolTalk |
| `magicoder_evol`, `code_debugging` | Apache-2.0 | ISE UIUC / M-A-P |
| `magicoder_oss`, `openhermes_2_5` | MIT | ISE UIUC / Teknium |
| `openmathinstruct2` | CC-BY-4.0 | NVIDIA Corporation |
| `numinamath_cot`, `openthoughts_reasoning` | Apache-2.0 | AI-MO / Open-Thoughts |
| `tulu3_sft`, `self_oss_starcoder2` | ODC-By | Allen AI / BigCode Project |
| `stem_qa` (`WebInstructSub`) | Apache-2.0 | TIGER-Lab |

Users and researchers must comply with the individual licenses of each constituent source.

---

## Citation

To cite the **KapInstruct-100M** dataset in research:

```bibtex
@misc{kapinstruct100m2026,
  title   = {{KapInstruct-100M}: A Curated 100-Million Token Multi-Source Instruction Tuning Dataset for Compact Models},
  author  = {Kaptaan, Rudy and Contributors},
  year    = {2026},
  publisher = {Hugging Face},
  url     = {https://huggingface.co/datasets/kaptaan45/KapInstruct-100M}
}
```

```bibtex
@misc{qaptaanlm2026,
  title   = {{QaptaanLM-0.75B}: Efficient Hybrid-Attention Foundation Language Model},
  author  = {Kaptaan, Rudy and Contributors},
  year    = {2026},
  publisher = {GitHub},
  url     = {https://github.com/rudy-07/QaptaanLM-0.75B}
}
```
