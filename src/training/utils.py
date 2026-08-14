"""Training environment utilities.

Handles:
- Environment detection (Colab / Kaggle / local)
- GPU/TPU detection and configuration
- Memory-efficient model loading
- Vision component stripping
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware (GPU, TPU, CPU).

    Returns:
        Dict with hardware info: device, gpu_name, gpu_memory, etc.
    """
    info = {
        "device": "cpu",
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "bf16_support": False,
        "tpu_available": False,
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
        info["bf16_support"] = torch.cuda.is_bf16_supported()

        logger.info(
            f"GPU detected: {info['gpu_name']} "
            f"({info['gpu_memory_gb']}GB, bf16={info['bf16_support']})"
        )
    else:
        logger.info("No GPU detected, using CPU")

    # Check for TPU
    try:
        import torch_xla.core.xla_model as xm

        info["tpu_available"] = True
        info["device"] = "xla"
        logger.info("TPU detected")
    except ImportError:
        pass

    return info


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def auto_configure_batch_size(
    gpu_memory_gb: float,
    seq_length: int = 4096,
    model_params_b: float = 0.8,
    gpu_count: int = 1,
    vocab_size: int = 248320,
) -> Tuple[int, int]:
    """Automatically determine safe micro batch size and gradient accumulation.

    Accounts for model parameters, optimizer states (AdamW), gradient checkpointing,
    and the large vocabulary (248k) logits tensor in float32 during loss calculation.

    Args:
        gpu_memory_gb: Available GPU memory in GB.
        seq_length: Sequence length.
        model_params_b: Model parameters in billions.
        gpu_count: Number of GPUs available.
        vocab_size: Model vocabulary size (default 248,320 for Qwen3.5).

    Returns:
        Tuple of (micro_batch_size, gradient_accumulation_steps).
    """
    # Base memory: model (2B/param) + grads (2B/param) + AdamW states (8B/param) + CUDA/NCCL overhead (~1.5GB)
    base_memory_gb = (model_params_b * 12.0) + 1.5  # ~11.1 GB for 0.8B

    # Per-sample peak memory at loss computation:
    # - Logits in fp32: seq_length * vocab_size * 4 bytes
    # - Activations with gradient checkpointing: ~0.5 GB
    logits_gb = (seq_length * vocab_size * 4) / (1024**3)
    activation_gb = (seq_length / 4096) * 0.5
    mem_per_sample_gb = logits_gb + activation_gb

    # Strict safety cap based on total GPU VRAM
    if gpu_memory_gb <= 16.5:
        # 16GB GPUs (T4, P100, V100-16GB): Must use micro_batch=1 due to 3.8GB logits tensor
        micro_batch = 1
    elif gpu_memory_gb <= 24.5:
        # 24GB GPUs (L4, RTX 3090, RTX 4090, A10G): max micro_batch=2
        available = max(0.0, gpu_memory_gb - base_memory_gb)
        micro_batch = max(1, min(2, int(available / mem_per_sample_gb)))
    elif gpu_memory_gb <= 48.0:
        # 40GB/48GB GPUs (A100-40GB, A6000): max micro_batch=4
        available = max(0.0, gpu_memory_gb - base_memory_gb)
        micro_batch = max(1, min(4, int(available / mem_per_sample_gb)))
    else:
        # 80GB GPUs (A100-80GB, H100):
        available = max(0.0, gpu_memory_gb - base_memory_gb)
        micro_batch = max(1, int(available / mem_per_sample_gb))

    # Target effective global batch size of ~64 to 128 sequences (256k-512k tokens per step)
    target_effective_sequences = 64
    total_micro_batch = micro_batch * max(1, gpu_count)
    grad_accum = max(1, target_effective_sequences // total_micro_batch)

    logger.info(
        f"Auto-configured: micro_batch={micro_batch}, "
        f"grad_accum={grad_accum} "
        f"(effective_batch_size={micro_batch * grad_accum * max(1, gpu_count)} sequences across {max(1, gpu_count)} GPU(s))"
    )

    return micro_batch, grad_accum


def load_model_for_training(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    strip_vision: bool = True,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """Load model and tokenizer for CPT training.

    Handles:
    - Loading as text-only CausalLM (strips vision encoder)
    - Setting up gradient checkpointing
    - Proper dtype configuration

    Args:
        model_name_or_path: HF model ID or local path.
        dtype: Torch dtype string.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        strip_vision: Whether to load text-only model (strip vision).
        trust_remote_code: Whether to allow remote code from HF Hub.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Load model
    if strip_vision:
        # Use Qwen3_5ForCausalLM which loads only the language model
        try:
            from transformers import Qwen3_5ForCausalLM

            logger.info("Loading model as Qwen3_5ForCausalLM (text-only)...")
            model = Qwen3_5ForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            logger.info(
                f"✓ Loaded text-only model. "
                f"Params: {sum(p.numel() for p in model.parameters()):,}"
            )
        except Exception as e:
            logger.warning(
                f"Qwen3_5ForCausalLM failed ({e}), "
                "falling back to AutoModelForCausalLM"
            )
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    # Ensure all parameters are trainable (full fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model ready: {total_params:,} total params, "
        f"{trainable_params:,} trainable"
    )

    return model, tokenizer
