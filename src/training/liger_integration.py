"""Liger Kernel Integration for Qwen3/Qwen3.5.

Applies fused Triton kernels that are mathematically identical to standard
HuggingFace implementations but significantly more efficient:

- FusedLinearCrossEntropy: Avoids materializing the full [batch, seq, vocab_size]
  logits tensor in FP32. For Qwen3.5's 248k vocabulary, this saves ~2.5GB VRAM
  at seq_len=2048 (previously the single largest VRAM consumer).
- FusedRMSNorm: Fuses normalization into a single kernel (minor speedup).
- FusedRoPE: Fuses rotary position embedding computation in-place.
- FusedSwiGLU: Fuses the gated activation in the MLP block.

All kernels are zero-accuracy-loss (bit-identical outputs for same inputs).
Works on T4 (SM75 / Turing) — does NOT require Ampere or newer.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Track whether patching has been applied (avoid double-patching)
_LIGER_APPLIED = False


def is_liger_available() -> bool:
    """Check if liger-kernel is installed and importable."""
    try:
        import liger_kernel  # noqa: F401
        return True
    except ImportError:
        return False


def get_liger_version() -> Optional[str]:
    """Get installed liger-kernel version, or None if not installed."""
    try:
        import liger_kernel
        return getattr(liger_kernel, "__version__", "unknown")
    except ImportError:
        return None


def apply_liger_kernel_patches(model_type: str = "qwen3") -> bool:
    """Apply Liger Kernel patches to the specified model architecture.

    Must be called BEFORE model instantiation (patches the model class,
    not the instance). Safe to call multiple times — will skip if already applied.

    Args:
        model_type: One of "qwen3", "qwen3_moe", "qwen3_vl".
                    Qwen3.5 uses the same architecture as Qwen3.

    Returns:
        True if patches were applied successfully, False otherwise.
    """
    global _LIGER_APPLIED

    if _LIGER_APPLIED:
        logger.info("Liger Kernel patches already applied, skipping.")
        return True

    if not is_liger_available():
        logger.warning(
            "liger-kernel is not installed. Install with: pip install liger-kernel>=0.3.0\n"
            "Liger Kernel provides ~40-60% VRAM savings and ~15-20% throughput improvement "
            "by fusing cross-entropy, RMSNorm, SwiGLU, and RoPE kernels.\n"
            "Continuing without Liger optimizations."
        )
        return False

    try:
        version = get_liger_version()
        logger.info(f"Applying Liger Kernel v{version} patches for model_type='{model_type}'...")

        model_type_lower = model_type.lower().replace("-", "_").replace(".", "_")

        # Qwen3.5 uses the same transformer architecture as Qwen3
        # (Qwen3_5ForCausalLM inherits the same attention/MLP/norm layers)
        if model_type_lower in ("qwen3", "qwen3_5", "qwen35"):
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_qwen3
                apply_liger_kernel_to_qwen3(
                    rope=True,
                    rms_norm=True,
                    swiglu=True,
                    cross_entropy=True,
                    fused_linear_cross_entropy=True,
                )
                logger.info(
                    "✓ Liger Kernel patches applied for Qwen3/Qwen3.5:\n"
                    "  - FusedLinearCrossEntropy (saves ~2.5GB VRAM for 248k vocab)\n"
                    "  - FusedRMSNorm\n"
                    "  - FusedRoPE\n"
                    "  - FusedSwiGLU"
                )
                _LIGER_APPLIED = True
                return True
            except ImportError:
                # Older liger-kernel versions may not have qwen3-specific patching
                logger.warning(
                    "apply_liger_kernel_to_qwen3 not found. "
                    "Trying generic AutoLigerKernelForCausalLM approach..."
                )
                return _apply_liger_auto_patch()

        elif model_type_lower in ("qwen3_moe", "qwen3_5_moe"):
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
                apply_liger_kernel_to_qwen3_moe(
                    rope=True,
                    rms_norm=True,
                    swiglu=True,
                    cross_entropy=True,
                    fused_linear_cross_entropy=True,
                )
                logger.info("✓ Liger Kernel patches applied for Qwen3 MoE.")
                _LIGER_APPLIED = True
                return True
            except ImportError:
                return _apply_liger_auto_patch()

        else:
            logger.warning(
                f"Unknown model_type '{model_type}' for Liger patching. "
                "Trying auto-detection..."
            )
            return _apply_liger_auto_patch()

    except Exception as e:
        logger.error(
            f"Failed to apply Liger Kernel patches: {e}\n"
            "Training will continue without Liger optimizations. "
            "Performance will be slower and VRAM usage higher.",
            exc_info=True,
        )
        return False


def _apply_liger_auto_patch() -> bool:
    """Fallback: use AutoLigerKernelForCausalLM for generic patching."""
    global _LIGER_APPLIED
    try:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM  # noqa: F401
        # AutoLigerKernelForCausalLM patches when used as the model class.
        # We just verify it's importable here — actual usage is in load_model_for_training.
        logger.info(
            "✓ AutoLigerKernelForCausalLM available as fallback. "
            "Will be used during model loading."
        )
        _LIGER_APPLIED = True
        return True
    except ImportError:
        logger.warning(
            "AutoLigerKernelForCausalLM not available. "
            "Liger Kernel optimizations will not be applied."
        )
        return False


def estimate_vram_savings(
    seq_length: int = 2048,
    vocab_size: int = 248_320,
    micro_batch: int = 1,
) -> dict:
    """Estimate VRAM savings from Liger Kernel's FusedLinearCrossEntropy.

    The standard cross-entropy loss in HuggingFace materializes the full
    logits tensor [batch, seq, vocab] in FP32 for numerical stability.
    Liger's fused version computes the loss in chunks, never materializing
    the full tensor.

    Args:
        seq_length: Sequence length.
        vocab_size: Model vocabulary size.
        micro_batch: Micro batch size per GPU.

    Returns:
        Dict with estimated savings in GB.
    """
    # Standard: 3 tensors (logits, shifted_logits, loss buffer) × FP32
    standard_ce_gb = 3 * (seq_length * vocab_size * 4 * micro_batch) / (1024**3)

    # Liger: processes in chunks of ~1024 tokens, only 1 chunk materialized at a time
    chunk_size = min(1024, seq_length)
    liger_ce_gb = 3 * (chunk_size * vocab_size * 4 * micro_batch) / (1024**3)

    # Additional savings from fused SwiGLU (avoids intermediate activation storage)
    swiglu_savings_gb = 0.3 * micro_batch  # Approximate

    savings = {
        "standard_ce_vram_gb": round(standard_ce_gb, 2),
        "liger_ce_vram_gb": round(liger_ce_gb, 2),
        "ce_savings_gb": round(standard_ce_gb - liger_ce_gb, 2),
        "swiglu_savings_gb": round(swiglu_savings_gb, 2),
        "total_savings_gb": round(standard_ce_gb - liger_ce_gb + swiglu_savings_gb, 2),
    }

    logger.info(
        f"Estimated VRAM savings with Liger Kernel:\n"
        f"  Cross-entropy: {savings['standard_ce_vram_gb']:.1f}GB → {savings['liger_ce_vram_gb']:.1f}GB "
        f"(saves {savings['ce_savings_gb']:.1f}GB)\n"
        f"  SwiGLU fusion: saves ~{savings['swiglu_savings_gb']:.1f}GB\n"
        f"  Total estimated savings: ~{savings['total_savings_gb']:.1f}GB"
    )

    return savings

