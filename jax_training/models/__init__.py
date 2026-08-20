"""JAX/Flax Models for Qwen3.5."""

from jax_training.models.config import Qwen3_5Config
from jax_training.models.qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5RotaryEmbedding,
    Qwen3_5MLP,
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5ForCausalLM,
)
from jax_training.models.loss import chunked_linear_cross_entropy
from jax_training.models.convert import convert_pytorch_to_flax_params

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5RMSNorm",
    "Qwen3_5RMSNormGated",
    "Qwen3_5RotaryEmbedding",
    "Qwen3_5MLP",
    "Qwen3_5Attention",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5DecoderLayer",
    "Qwen3_5Model",
    "Qwen3_5ForCausalLM",
    "chunked_linear_cross_entropy",
    "convert_pytorch_to_flax_params",
]
