"""Configuration for Qwen3.5 Flax Model."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Qwen3_5Config:
    """Configuration class for Qwen3.5 text-only causal language model."""

    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    attn_output_gate: bool = True
    full_attention_interval: int = 4
    
    # Linear Attention (Gated Delta Net) hyperparameters
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_conv_kernel_dim: int = 4
    
    # Precision & Execution
    dtype: str = "bfloat16"
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.layer_types is None:
            # By default: every 4th layer (3, 7, 11, 15, 19, 23) is full_attention, others are linear_attention
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen3_5Config":
        """Create config from dictionary (e.g. text_config from HuggingFace config.json)."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
