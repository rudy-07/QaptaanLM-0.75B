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
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
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


def auto_configure_batch_size(
    gpu_memory_gb: float,
    seq_length: int = 4096,
    model_params_b: float = 0.8,
) -> Tuple[int, int]:
    """Automatically determine batch size and gradient accumulation.

    Args:
        gpu_memory_gb: Available GPU memory in GB.
        seq_length: Sequence length.
        model_params_b: Model parameters in billions.

    Returns:
        Tuple of (micro_batch_size, gradient_accumulation_steps).
    """
    # Rough memory estimates for full-parameter training with gradient checkpointing:
    # Model weights: ~2 bytes per param (bf16) = 1.6 GB for 0.8B
    # Optimizer states (AdamW): ~8 bytes per param = 6.4 GB
    # Gradients: ~2 bytes per param = 1.6 GB
    # Activations with grad checkpointing: ~2-4 GB depending on batch/seq
    # Total base: ~12 GB for 0.8B model

    base_memory_gb = model_params_b * 15  # Rough estimate

    available_for_batch = gpu_memory_gb - base_memory_gb
    # Each batch element ≈ seq_length * hidden_size * num_layers * 2 bytes
    # For 0.8B: roughly 0.5-1 GB per batch element at seq_len=4096
    mem_per_sample_gb = seq_length / 4096 * 0.8

    if available_for_batch <= 0:
        micro_batch = 1
    else:
        micro_batch = max(1, int(available_for_batch / mem_per_sample_gb))

    # Target effective batch of ~128 sequences
    target_effective = 128
    grad_accum = max(1, target_effective // micro_batch)

    logger.info(
        f"Auto-configured: micro_batch={micro_batch}, "
        f"grad_accum={grad_accum} "
        f"(effective={micro_batch * grad_accum})"
    )

    return micro_batch, grad_accum


def load_model_for_training(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    strip_vision: bool = True,
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

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
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
                trust_remote_code=False,
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
                trust_remote_code=False,
            )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
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
