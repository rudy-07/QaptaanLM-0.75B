# Qwen3.5-0.8B Coding + Instruct Fine-Tuning Pipeline

A robust, portable, and production-ready two-phase fine-tuning pipeline for **Qwen3.5-0.8B-Base** targeting state-of-the-art code generation, technical reasoning, and instruction following.

---

## 🚀 Key Highlights & Architectural Features

- **Base Model**: `Qwen3.5-0.8B-Base` (752M text parameters).
- **Hybrid Attention Architecture**: 3:1 ratio of Gated DeltaNet (linear attention) to Gated Attention (GQA), with native 256K context support and RMSNorm.
- **Text-Only Optimization**: Vision encoder automatically stripped during CPT via `Qwen3_5ForCausalLM` to maximize throughput and minimize GPU memory footprint.
- **Fill-in-the-Middle (FIM)**: 50% of code samples formatted with `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` tokens for code infilling and completion.
- **Document Packing**: Efficient multi-document sequence packing (4096 tokens) with EOS delimiters and attention masking to eliminate padding waste.
- **Multi-Environment Portability**: One-click training on Google Colab (GPU/TPU) and Kaggle (Dual T4/P100) with automatic checkpoint syncing to Google Drive and HuggingFace Hub.

---

## 📊 Dataset Mixture (~1 Billion Target Tokens)

| Dataset | Proportion | Target Tokens | Description |
|---|---|---|---|
| `HuggingFaceCode/stack-v3-train` (Code) | **35%** | ~350M | Multi-language source code filtered for quality & licenses |
| `HuggingFaceCode/stack-v3-train` (Docs) | **20%** | ~200M | Technical documentation, READMEs, API guides (`.md`, `.rst`) |
| `Fsoft-AIC/the-vault-function` | **20%** | ~200M | Function-level code with docstrings and identifiers |
| `epfml/FineWeb-HQ` | **15%** | ~150M | Top 10% educational & high-quality web text |
| `open-web-math/open-web-math` | **10%** | ~100M | Mathematical reasoning, LaTeX equations, and technical web |
| **Total** | **100%** | **~1 Billion** | |

### 🎯 Target Language Distribution (Code Subset)

| Language / Category | Target Proportion |
|---|---:|
| **Python** | **25%** |
| **TypeScript** | **13%** |
| **JavaScript** | **10%** |
| **SQL** | **9%** |
| **C++** | **7%** |
| **Shell / Bash** | **6%** |
| **C** | **5%** |
| **Java** | **5%** |
| **HTML** | **5%** |
| **Rust** | **4%** |
| **Go** | **4%** |
| **CSS** | **4%** |
| **Docker / CI-CD / IaC** | **3%** |
| **Total** | **100%** |

---

## 📁 Repository Structure

```
.
├── configs/
│   ├── cpt_config.yaml          # Phase 1 Continued Pre-Training configuration
│   ├── dataset_config.yaml      # Dataset mixture, filtering thresholds & dedup
│   └── eval_config.yaml         # Benchmark evaluation settings
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Unified streaming dataset loader (all 5 datasets)
│   │   ├── filters.py           # Multi-signal quality, language, and boilerplate filters
│   │   ├── dedup.py             # Exact SHA-256 and MinHash near-deduplication
│   │   ├── tokenize_and_pack.py # Qwen2Tokenizer packing, EOS boundaries & FIM
│   │   ├── mixture.py           # Deficit-based weighted dataset stream mixer
│   │   └── sharding.py          # Arrow/Parquet chunked shard writer & manifests
│   ├── training/
│   │   ├── trainer.py           # Full-parameter CPT Trainer with token-count stopping
│   │   ├── callbacks.py         # Progress logging, ETA, GDrive & HF Hub auto-upload
│   │   └── utils.py             # Hardware detection & automatic batch sizing
│   ├── evaluation/
│   │   ├── benchmarks.py        # HumanEval coding, math reasoning & perplexity suite
│   │   └── compare.py           # Base vs CPT side-by-side evaluation
│   └── utils/
│       ├── config.py            # Environment-aware YAML configuration loader
│       ├── logging_utils.py     # Structured console and file logging
│       └── storage.py           # Google Drive and Hugging Face Hub sync
│
├── scripts/
│   ├── 01_verify_model.py       # Base model inspection, text-only load & generation
│   ├── 02_verify_datasets.py    # Verify streaming connectivity for all 5 datasets
│   ├── 03_process_data.py       # End-to-end data filtering, mixing & sharding
│   ├── 04_smoke_test.py         # Complete 3-step training & checkpoint reload test
│   ├── 05_train_cpt.py          # Launch full CPT training run
│   └── 06_evaluate.py           # Benchmark evaluation and model comparison
│
├── notebooks/
│   ├── colab_cpt.ipynb          # Google Colab ready-to-run GPU/Drive workflow
│   └── kaggle_cpt.ipynb         # Kaggle GPU ready-to-run workflow
│
├── requirements.txt             # Python dependencies
└── PROJECT_SPEC.md              # Full original specification & guidelines
```

---

## 🛠️ Quick Start

### 1. Verify Model
```bash
python scripts/01_verify_model.py --model-path "models/Qwen3.5-0.8B-Base"
```

### 2. Run End-to-End Smoke Test
```bash
python scripts/04_smoke_test.py --model-path "models/Qwen3.5-0.8B-Base" --num-steps 3
```

### 3. Generate Packed Dataset Shards
```bash
# Full dataset mixture (~1B tokens)
python scripts/03_process_data.py --output-dir data/processed

# Or test on a small sample (e.g. 1000 samples per dataset)
python scripts/03_process_data.py --max-samples 1000 --output-dir data/processed
```

### 4. Launch CPT Training
```bash
python scripts/05_train_cpt.py --data-dir data/processed
```

### 5. Benchmark & Compare Models
```bash
python scripts/06_evaluate.py --compare --base Qwen/Qwen3.5-0.8B-Base --cpt checkpoints/cpt/final
```

---

## ☁️ Remote Training (Google Colab / Kaggle)

1. **Google Colab**: Open `notebooks/colab_cpt.ipynb`, connect an NVIDIA T4/L4/A100 GPU runtime, mount Google Drive, and run the cells. Checkpoints and dataset shards automatically sync to Google Drive.
2. **Kaggle**: Open `notebooks/kaggle_cpt.ipynb`, set accelerator to GPU T4 x2 or P100, and execute the training pipeline.
