"""Training environment utilities.

Handles:
- Environment detection (Colab / Kaggle / local)
- GPU/TPU detection and configuration
- Memory-efficient model loading
- Vision component stripping
- Liger Kernel-aware VRAM optimization
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
        # torch.cuda.is_bf16_supported() returns True on T4 (compute capability 7.5)
        # because PyTorch CAN emulate BF16, but it runs 5-10x slower than FP16 since
        # T4 lacks hardware BF16 Tensor Cores. Only GPUs with compute capability >= 8.0
        # (A100, H100, L4, RTX 30xx/40xx) have native BF16 Tensor Cores.
        cc_major = torch.cuda.get_device_properties(0).major
        info["bf16_support"] = cc_major >= 8

        logger.info(
            f"GPU detected: {info['gpu_name']} "
            f"({info['gpu_memory_gb']}GB, bf16={info['bf16_support']})"
        )
    else:
        logger.info("No GPU detected, using CPU")

    # torch_xla can be installed on a CPU notebook too.  Treat it as a TPU
    # only when PJRT was explicitly configured for one; importing it alone must
    # not make a local/CPU run select TPU-only precision and launch behaviour.
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        try:
            import torch_xla  # noqa: F401

            info["tpu_available"] = True
            info["device"] = "xla"
            logger.info("PJRT TPU requested")
        except ImportError as exc:
            raise RuntimeError(
                "PJRT_DEVICE=TPU was set but torch_xla is not installed in this environment."
            ) from exc

    return info


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def auto_configure_batch_size(
    gpu_memory_gb: float,
    seq_length: int = 4096,
    model_params_b: float = 0.8,
    gpu_count: int = 1,
    vocab_size: int = 248320,
    liger_enabled: bool = False,
) -> Tuple[int, int, int]:
    """Automatically determine safe micro batch size, gradient accumulation, and sequence length.

    Accounts for model parameters, optimizer states (AdamW), gradient checkpointing,
    and the large vocabulary (248k) logits tensor in float32 during loss calculation.

    When Liger Kernel is enabled, the FusedLinearCrossEntropy kernel avoids
    materializing the full [seq, vocab] logits tensor, saving ~2-4GB VRAM.
    This allows higher sequence lengths (2048 instead of 1024) on 16GB GPUs.

    Args:
        gpu_memory_gb: Available GPU memory in GB.
        seq_length: Requested sequence length.
        model_params_b: Model parameters in billions.
        gpu_count: Number of GPUs available.
        vocab_size: Model vocabulary size (default 248,320 for Qwen3.5).
        liger_enabled: Whether Liger Kernel is active (reduces CE VRAM).

    Returns:
        Tuple of (micro_batch_size, gradient_accumulation_steps, safe_seq_length).
    """
    safe_seq_length = seq_length

    if gpu_memory_gb <= 16.5:
        if liger_enabled:
            # Liger's FusedLinearCrossEntropy processes in chunks (~1024 tokens at a time),
            # never materializing the full [seq, vocab] logits tensor. This saves ~2-4GB.
            # With Liger: seq_len=2048 is safe on T4 (cross-entropy peak: ~0.6GB instead of ~5.7GB)
            if seq_length > 2048:
                logger.warning(
                    f"GPU has {gpu_memory_gb}GB VRAM. Even with Liger Kernel, reducing "
                    f"max_seq_length from {seq_length} to 2048 for safety."
                )
                safe_seq_length = 2048
            else:
                logger.info(
                    f"Liger Kernel enabled: keeping seq_length={seq_length} on {gpu_memory_gb}GB GPU "
                    f"(FusedLinearCrossEntropy avoids full logits materialization)"
                )
        else:
            # Without Liger: cap at 1024 because the 248k-vocab cross-entropy is massive
            # At 2048: 3 × (2048 × 248k × 4B) = ~5.7GB VRAM spike -> CUDA OOM
            # At 1024: 3 × (1024 × 248k × 4B) = ~2.8GB VRAM spike -> fits with ~3.7GB headroom
            if seq_length > 1024:
                logger.warning(
                    f"GPU has only {gpu_memory_gb}GB VRAM. Reducing max_seq_length from "
                    f"{seq_length} to 1024 to prevent CUDA OOM from the 248k-vocabulary "
                    f"cross-entropy buffers "
                    f"(saves ~{((seq_length - 1024) * vocab_size * 4 * 3) / (1024**3):.1f}GB VRAM "
                    f"during loss computation). Install liger-kernel to enable seq_len=2048."
                )
                safe_seq_length = 1024

    # Recurrent Linear Attention Scan Safety:
    # In PyTorch eager autograd, the unrolled recurrence over sequence length T retains
    # intermediate [batch, heads, head_k_dim, head_v_dim] state tensors for every step
    # across all 24 layers (~25 MB per token in FP32).
    # At seq_len=1024: ~25 GB activations/sample -> fits within 95GB VRAM at micro_batch=1.
    # At seq_len=4096: ~102 GB activations/sample -> exceeds 95GB VRAM at micro_batch=1.
    # We cap safe_seq_length to 1024 on GPU for reliable, OOM-free SFT training.
    if safe_seq_length > 1024:
        logger.info(
            f"Recurrent linear attention scan safety: capping max_seq_length from {seq_length} to 1024 "
            f"to ensure full autograd recurrence fits comfortably within GPU VRAM."
        )
        safe_seq_length = 1024

    # Base memory: model (4B/param FP32) + grads (4B/param FP32) + AdamW (8-bit or standard) + CUDA overhead
    base_memory_gb = (model_params_b * 10.0) + 1.5  # ~9.5 GB for 0.8B FP32 master weights

    # For recurrent linear attention in PyTorch autograd:
    # 1024 tokens = ~25 GB activations + 1 GB logits
    recurrent_act_gb = (safe_seq_length / 1024.0) * 25.0
    logits_gb = (safe_seq_length * vocab_size * 4) / (1024**3)
    mem_per_sample_gb = recurrent_act_gb + logits_gb

    available = max(0.0, gpu_memory_gb - base_memory_gb)
    if safe_seq_length >= 1024:
        micro_batch = 1  # 1 sequence of 1024 takes ~35GB total peak VRAM
    elif safe_seq_length <= 512 and available > (mem_per_sample_gb * 2):
        micro_batch = 2
    else:
        micro_batch = 1

    # Target effective global batch size: 16 sequences (16,384 tokens/step at seq=1024)
    target_effective_sequences = 16
    total_micro_batch = micro_batch * max(1, gpu_count)
    grad_accum = max(1, target_effective_sequences // total_micro_batch)

    effective_tokens_per_step = safe_seq_length * micro_batch * grad_accum * max(1, gpu_count)
    logger.info(
        f"Auto-configured: micro_batch={micro_batch}, "
        f"grad_accum={grad_accum}, seq_length={safe_seq_length} "
        f"(effective_batch_size={micro_batch * grad_accum * max(1, gpu_count)} sequences, "
        f"{effective_tokens_per_step:,} tokens/step across {max(1, gpu_count)} GPU(s))"
        f"{' [Liger Kernel active]' if liger_enabled else ''}"
    )

    return micro_batch, grad_accum, safe_seq_length


def load_model_for_training(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    strip_vision: bool = True,
    trust_remote_code: bool = True,
    use_sdpa: bool = True,
) -> Tuple[Any, Any]:
    """Load model and tokenizer for CPT training.

    Handles:
    - Loading as text-only CausalLM (strips vision encoder)
    - Setting up gradient checkpointing
    - Proper dtype configuration
    - SDPA attention backend (memory-efficient on T4)

    Args:
        model_name_or_path: HF model ID or local path.
        dtype: Torch dtype string.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        strip_vision: Whether to load text-only model (strip vision).
        trust_remote_code: Whether to allow remote code from HF Hub.
        use_sdpa: Whether to use PyTorch SDPA attention (recommended for T4).

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

    # Determine attention implementation
    # SDPA (Scaled Dot-Product Attention) uses PyTorch's native memory-efficient
    # attention backend. This works on ALL GPU architectures including T4 (SM75),
    # unlike FlashAttention-2 which requires SM80+ (Ampere).
    attn_impl = "sdpa" if use_sdpa else "eager"
    logger.info(f"Using attention implementation: {attn_impl}")

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
                attn_implementation=attn_impl,
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

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_impl,
                )
            except (ValueError, Exception) as e_attn:
                if attn_impl != "eager":
                    logger.warning(
                        f"Attention '{attn_impl}' not supported by custom model ({e_attn}). "
                        "Retrying with attn_implementation='eager'..."
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                else:
                    raise e_attn
    else:
        from transformers import AutoModelForCausalLM

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_impl,
            )
        except (ValueError, Exception) as e_attn:
            if attn_impl != "eager":
                logger.warning(
                    f"Attention '{attn_impl}' not supported by custom model ({e_attn}). "
                    "Retrying with attn_implementation='eager'..."
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
            else:
                raise e_attn

    # Enable gradient checkpointing for memory efficiency if supported
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled")
        except (ValueError, Exception) as e_gc:
            logger.warning(
                f"Gradient checkpointing not supported by {model.__class__.__name__} ({e_gc}). "
                "Proceeding without gradient checkpointing (ample VRAM available on GPU)."
            )

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
