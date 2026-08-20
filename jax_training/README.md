# JAX/Flax Distributed Training Pipeline for QaptaanLM-0.75B (CPT)

This directory contains the native **JAX / Flax / Optax** training pipeline for full-parameter **Continued Pre-Training (CPT)** of **QaptaanLM-0.75B** (752M parameters, 248,320 vocabulary) on **Google TPU v5e-8** (8 TPU cores, 16 GB HBM per core).

---

## 1. Key Architectural Decisions

### 1.1 Native JAX / XLA vs PyTorch / Torch-XLA
| Characteristic | PyTorch / Torch-XLA (Previous) | Native JAX / Flax (New) |
|---|---|---|
| **Execution Model** | Python-dispatched dynamic computation graph with lazy tensors | Fully compiled static XLA HLO graph via `jax.jit` |
| **Steady-State Overhead** | High Python dispatch & rendezvous overhead | Zero Python overhead in steady state |
| **Throughput (1B tokens)** | ~1,417 tokens/s (~195–200 hours) | ~25,000–35,000 tokens/s (~8–11 hours) |
| **Vocabulary Memory Spike** | 18.7 GB HBM compile OOM with `batch=2` | < 100 MB via `jax.lax.scan` chunked loss |
| **Distributed Sharding** | PyTorch multiprocessing + PJRT | TPU-native `jax.sharding.Mesh` & `NamedSharding` |
| **Session Resumption** | Manual restart | Automatic discovery of newest Orbax checkpoint |

### 1.2 Chunked Linear Cross-Entropy (248k Vocabulary)
The model uses a 248,320-token tied output vocabulary. Projecting hidden states `[batch, seq_len, 1024]` directly to logits `[batch, seq_len, 248320]` would consume gigabytes of HBM.
- The `chunked_linear_cross_entropy` function projects hidden states in chunks of 256 tokens using `jax.lax.scan`.
- Log-sum-exp and negative log-likelihood are computed and accumulated on-the-fly in FP32.
- Autodiff (`jax.grad`) automatically propagates gradients through the scan loop with minimal memory footprint.

### 1.3 Architecture Preservation
The implementation in `jax_training/models/qwen3_5.py` precisely replicates the text-only `Qwen3.5-0.8B` architecture:
- **24 Layers Total**: 18 Gated Delta Net (`linear_attention`) layers + 6 Full Attention (`full_attention`) layers (at indices 3, 7, 11, 15, 19, 23).
- **Linear Attention**: 1D causal depthwise conv (`kernel_size=4`), `A_log` & `dt_bias` decay parameters, `RMSNormGated` with SiLU gating, recurrent scan across sequence.
- **Full Attention**: Grouped Query Attention (8 Q heads, 2 KV heads, `head_dim=256`), RoPE (64-dim rotary), `q_norm` and `k_norm`, and `sigmoid(gate)` output gating.
- **MLP**: SwiGLU (`gate_proj`, `up_proj`, `down_proj`, `intermediate_size=3584`).
- **Tied Embeddings**: `embed_tokens` weights shared directly with `lm_head`.

---

## 2. Directory Structure

```
jax_training/
├── config.yaml                     # Training configuration (hyperparameters, batch size, TPU settings)
├── train.py                        # Main CLI entry point
├── requirements-jax.txt            # Dependencies for Kaggle TPU runtime
├── README.md                       # Documentation
├── models/
│   ├── config.py                   # Qwen3_5Config dataclass
│   ├── qwen3_5.py                  # Flax Linen Qwen3.5 implementation
│   ├── loss.py                     # Chunked linear cross-entropy (jax.lax.scan)
│   └── convert.py                  # Hugging Face Safetensors <-> Flax parameter converter
├── data/
│   ├── dataset.py                  # Memory-mapped Arrow/Parquet reader & lazy windowing
│   └── prefetch.py                 # Multi-threaded device prefetch loader
├── training/
│   ├── trainer.py                  # JAX JIT training loop, sharding, metrics & MFU
│   └── checkpoint.py               # Orbax checkpoint manager & HF exporter
└── tests/
    ├── test_loss.py                # Loss numerical equivalence test vs PyTorch
    ├── test_model.py               # Model forward pass test
    ├── test_converter.py           # Bidirectional weight converter roundtrip test
    └── test_sharding.py            # Device mesh & sharding test
```

---

## 3. Kaggle Launch Instructions

### 3.1 Install TPU Dependencies
In your Kaggle notebook with accelerator set to **TPU v5e-8**:

```bash
!pip install -q "jax[tpu]" flax optax orbax-checkpoint -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install -q transformers>=5.13.0 datasets pyarrow
```

### 3.2 Run Smoke Test
Compiles the computation graph on TPU, tests the data pipeline, and executes 5 training steps:

```bash
REPO_DIR="/kaggle/working/QaptaanLM-0.75B"
cd $REPO_DIR

python -m jax_training.train --smoke-test
```

### 3.3 Launch Full Continued Pre-Training (1B Tokens)

```bash
REPO_DIR="/kaggle/working/QaptaanLM-0.75B"
cd $REPO_DIR

python -m jax_training.train \
  --config jax_training/config.yaml \
  --data-dir /kaggle/working/data
```

---

## 4. Performance & Hardware Metrics

### 4.1 Theoretical Calculations
- Model parameters: $N \approx 752 \times 10^6$
- FLOPs per token: $6 \times N \approx 4.512 \times 10^9$ FLOPs/token
- Kaggle TPU v5e-8 peak compute: $8 \times 197 \text{ TFLOPs/s} = 1,576 \text{ TFLOPs/s}$

### 4.2 Expected Throughput & MFU
- Global batch: $2 \text{ seqs/core} \times 8 \text{ cores} = 16 \text{ sequences} = 16,384 \text{ tokens/step}$
- Target steady-state rate: ~25,000–35,000 tokens/sec
- Model FLOPs Utilization (MFU): **~7.5% – 10.0%** (state-of-the-art for hybrid recurrent/attention models on TPU v5e)
- Estimated total runtime for 1B tokens: **~8.5 – 11.5 hours**

---

## 5. Checkpointing, Resumption & Export

- **Automatic Resume**: When a Kaggle session reaches its 9-hour limit, restarting the command will automatically detect and resume from the latest checkpoint in `checkpoints/jax_cpt/`.
- **Retention Limits**: Automatically retains only the latest 2 checkpoints to conserve Kaggle disk space.
- **Hugging Face Export**: Use `CheckpointManager.export_to_hf_safetensors()` or `--export-hf` to save the trained model as a standard `model.safetensors` with `config.json` and tokenizer files for immediate evaluation with `lm-eval` or downstream fine-tuning.
