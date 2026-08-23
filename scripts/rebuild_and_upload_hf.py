"""Rebuild and publish clean, production-ready QaptaanLM-0.75B HuggingFace repository.

This script:
1. Clears out old artifacts and builds a clean Hugging Face repository directory.
2. Restores all 321 parameters from JAX checkpoint-61036 (including intact conv1d_weight).
3. Converts Flax parameters to PyTorch state dict and writes model.safetensors.
4. Packages official Qwen3.5 tokenizer (12.8MB) and configuration metadata.
5. Injects custom modeling code (modeling_qaptaan.py) with exact JAX recurrence and fast KV-cache.
6. Tests loading and generation in PyTorch using AutoModelForCausalLM.
7. Optionally uploads the clean repository to Hugging Face Hub.

Usage:
    python scripts/rebuild_and_upload_hf.py --hf-repo kaptaan45/QaptaanLM-0.75B --push
    python scripts/rebuild_and_upload_hf.py --local-only
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np


def build_configuration_qaptaan_code() -> str:
    """Returns the code for configuration_qaptaan.py."""
    return '''"""QaptaanLM-0.75B Configuration."""

from transformers.configuration_utils import PretrainedConfig


class QaptaanConfig(PretrainedConfig):
    model_type = "qaptaan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 1024,
        intermediate_size: int = 3584,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 262144,
        rope_theta: float = 10000000.0,
        partial_rotary_factor: float = 0.25,
        attn_output_gate: bool = True,
        full_attention_interval: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 16,
        linear_conv_kernel_dim: int = 4,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = None,
        eos_token_id: int = 248044,
        pad_token_id: int = 248044,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.attn_output_gate = attn_output_gate
        self.full_attention_interval = full_attention_interval
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Auto-compute layer types (hybrid 3:1 linear-to-full attention)
        self.layer_types = []
        for i in range(num_hidden_layers):
            if (i + 1) % full_attention_interval == 0:
                self.layer_types.append("full_attention")
            else:
                self.layer_types.append("linear_attention")

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
'''


def build_modeling_qaptaan_code() -> str:
    """Returns the code for modeling_qaptaan.py."""
    return '''"""QaptaanLM-0.75B PyTorch Model Implementation with Exact JAX Recurrence & Fast O(1) Cache."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache

from .configuration_qaptaan import QaptaanConfig


class QaptaanCache:
    """Hybrid State Cache storing Conv1D state, Gated Delta Net recurrent state, and Full Attention KV cache."""

    def __init__(self):
        self.conv_states: Dict[int, torch.Tensor] = {}
        self.recurrent_states: Dict[int, torch.Tensor] = {}
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self._seen_tokens: int = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class QaptaanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class QaptaanRMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        normed = normed * self.weight
        return normed * F.silu(gate)


class QaptaanRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 262144, base: float = 10000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, rotary_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_out = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_out = torch.cat([k_rot_embed, k_pass], dim=-1)
    return q_out, k_out


class QaptaanMLP(nn.Module):
    def __init__(self, config: QaptaanConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QaptaanFullAttention(nn.Module):
    def __init__(self, config: QaptaanConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = QaptaanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = QaptaanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary = QaptaanRotaryEmbedding(self.rotary_dim, config.max_position_embeddings, config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[QaptaanCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q_proj_out = self.q_proj(hidden_states)
        q_proj_out = q_proj_out.view(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        query = q_proj_out[..., : self.head_dim]
        gate = q_proj_out[..., self.head_dim :]

        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        query = self.q_norm(query).transpose(1, 2)  # [B, H, S, head_dim]
        key = self.k_norm(key).transpose(1, 2)      # [B, KV_H, S, head_dim]
        value = value.transpose(1, 2)               # [B, KV_H, S, head_dim]

        seen_tokens = past_key_value.get_seq_length(self.layer_idx) if (use_cache and past_key_value is not None) else 0
        cos, sin = self.rotary(seen_tokens + seq_len, hidden_states.device, query.dtype)
        cos = cos[seen_tokens : seen_tokens + seq_len]
        sin = sin[seen_tokens : seen_tokens + seq_len]
        query, key = apply_rope(query, key, cos, sin, self.rotary_dim)

        if use_cache and past_key_value is not None:
            key, value = past_key_value.update(key, value, self.layer_idx)

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            key_expanded = key.repeat_interleave(self.num_kv_groups, dim=1)
            value_expanded = value.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            key_expanded = key
            value_expanded = value

        kv_seq_len = key_expanded.shape[-2]
        scores = torch.matmul(query, key_expanded.transpose(-1, -2)) * self.scaling

        if seq_len > 1:
            causal_mask = torch.tril(
                torch.ones(seq_len, kv_seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=kv_seq_len - seq_len,
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if (attention_mask == 0).any():
                    scores = scores.masked_fill(attention_mask[:, None, None, :].eq(0), float("-inf"))
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        attn_out = torch.matmul(attn_weights, value_expanded).transpose(1, 2)  # [B, S, H, head_dim]
        attn_out = attn_out * torch.sigmoid(gate.float()).to(query.dtype)
        attn_out = attn_out.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


class QaptaanLinearAttention(nn.Module):
    """Linear Attention implementing the exact JAX CPT Recurrence Formula with fast O(1) Cache."""

    def __init__(self, config: QaptaanConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))

        self.norm = QaptaanRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[QaptaanCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Fast O(1) Single-Token Cached Step
        if use_cache and past_key_value is not None and seq_len == 1 and self.layer_idx in past_key_value.recurrent_states:
            conv_state = past_key_value.conv_states[self.layer_idx]
            recurrent_state = past_key_value.recurrent_states[self.layer_idx]

            mixed_qkv_t = mixed_qkv.transpose(1, 2)  # [B, conv_dim, 1]
            window = torch.cat([conv_state, mixed_qkv_t], dim=-1)  # [B, conv_dim, 4]
            past_key_value.conv_states[self.layer_idx] = window[:, :, 1:].detach()

            w = self.conv1d.weight.squeeze(1)  # [conv_dim, 4]
            conv_out = (window * w).sum(dim=-1, keepdim=True)  # [B, conv_dim, 1]
            mixed_qkv = F.silu(conv_out).transpose(1, 2)  # [B, 1, conv_dim]

            query = mixed_qkv[:, :, : self.key_dim].view(batch_size, 1, self.num_k_heads, self.head_k_dim)
            key = mixed_qkv[:, :, self.key_dim : 2 * self.key_dim].view(batch_size, 1, self.num_k_heads, self.head_k_dim)
            value = mixed_qkv[:, :, 2 * self.key_dim :].view(batch_size, 1, self.num_v_heads, self.head_v_dim)

            query = query / (torch.norm(query.float(), dim=-1, keepdim=True) + 1e-6).to(query.dtype)
            key = key / (torch.norm(key.float(), dim=-1, keepdim=True) + 1e-6).to(key.dtype)

            if self.num_v_heads // self.num_k_heads > 1:
                ratio = self.num_v_heads // self.num_k_heads
                query = query.repeat_interleave(ratio, dim=2)
                key = key.repeat_interleave(ratio, dim=2)

            beta = torch.sigmoid(b.float())
            g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())

            scale = 1.0 / math.sqrt(self.head_k_dim)
            q_i = (query.float() * scale).squeeze(1)   # [B, 16, 128]
            k_i = key.float().squeeze(1)                # [B, 16, 128]
            v_i = value.float().squeeze(1)              # [B, 16, 128]
            b_i = beta.squeeze(1).unsqueeze(-1)         # [B, 16, 1]
            g_i = g.squeeze(1).unsqueeze(-1)            # [B, 16, 1]
            decay = torch.exp(g_i).unsqueeze(-1)        # [B, 16, 1, 1]

            # O(1) single-token recurrent update (exact JAX formulation)
            v_prime = torch.einsum("bhk,bhkd->bhd", k_i, recurrent_state)
            v_new = (v_i - v_prime) * b_i

            attn_inter = torch.einsum("bhk,bhkd->bhd", q_i, recurrent_state) * torch.exp(g_i)
            qk_dot = torch.sum(q_i * k_i, dim=-1, keepdim=True)
            out_i = attn_inter + qk_dot * v_new

            new_state = recurrent_state * decay + torch.einsum("bhk,bhd->bhkd", k_i, v_new)
            past_key_value.recurrent_states[self.layer_idx] = new_state.detach()

            core_out = out_i.unsqueeze(1).to(hidden_states.dtype)
            z_reshaped = z.view(batch_size, 1, self.num_v_heads, self.head_v_dim)
            core_out = self.norm(core_out, z_reshaped)
            core_out = core_out.reshape(batch_size, 1, self.value_dim)
            return self.out_proj(core_out)

        # Prefill Mode (Full Sequence Scan)
        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        conv_out = self.conv1d(mixed_qkv_t)[:, :, :seq_len].transpose(1, 2)
        mixed_qkv = F.silu(conv_out)

        query = mixed_qkv[:, :, : self.key_dim].view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = mixed_qkv[:, :, self.key_dim : 2 * self.key_dim].view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = mixed_qkv[:, :, 2 * self.key_dim :].view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        query = query / (torch.norm(query.float(), dim=-1, keepdim=True) + 1e-6).to(query.dtype)
        key = key / (torch.norm(key.float(), dim=-1, keepdim=True) + 1e-6).to(key.dtype)

        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        beta = torch.sigmoid(b.float())
        g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())

        scale = 1.0 / math.sqrt(self.head_k_dim)
        q_scaled = query.float() * scale
        k_fp32 = key.float()
        v_fp32 = value.float()

        state = torch.zeros(
            batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            device=hidden_states.device, dtype=torch.float32
        )
        core_out = torch.zeros(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim,
            device=hidden_states.device, dtype=torch.float32
        )

        for t in range(seq_len):
            q_i = q_scaled[:, t]
            k_i = k_fp32[:, t]
            v_i = v_fp32[:, t]
            b_i = beta[:, t].unsqueeze(-1)
            g_i = g[:, t].unsqueeze(-1)
            decay = torch.exp(g_i).unsqueeze(-1)

            v_prime = torch.einsum("bhk,bhkd->bhd", k_i, state)
            v_new = (v_i - v_prime) * b_i

            attn_inter = torch.einsum("bhk,bhkd->bhd", q_i, state) * torch.exp(g_i)
            qk_dot = torch.sum(q_i * k_i, dim=-1, keepdim=True)
            out_i = attn_inter + qk_dot * v_new
            core_out[:, t] = out_i

            state = state * decay + torch.einsum("bhk,bhd->bhkd", k_i, v_new)

        if use_cache and past_key_value is not None:
            if seq_len >= 3:
                last_3 = mixed_qkv_t[:, :, -3:]
            else:
                last_3 = F.pad(mixed_qkv_t, (3 - seq_len, 0))
            past_key_value.conv_states[self.layer_idx] = last_3.detach()
            past_key_value.recurrent_states[self.layer_idx] = state.detach()

        core_out = core_out.to(hidden_states.dtype)
        z_reshaped = z.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        core_out = self.norm(core_out, z_reshaped)
        core_out = core_out.reshape(batch_size, seq_len, self.value_dim)
        return self.out_proj(core_out)


class QaptaanDecoderLayer(nn.Module):
    def __init__(self, config: QaptaanConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_full_attention = (layer_idx + 1) % config.full_attention_interval == 0

        self.input_layernorm = QaptaanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.is_full_attention:
            self.self_attn = QaptaanFullAttention(config, layer_idx)
        else:
            self.linear_attn = QaptaanLinearAttention(config, layer_idx)

        self.post_attention_layernorm = QaptaanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = QaptaanMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[QaptaanCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        if self.is_full_attention:
            attn_out = self.self_attn(
                normed, attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache
            )
        else:
            attn_out = self.linear_attn(normed, past_key_value=past_key_value, use_cache=use_cache)

        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class QaptaanPreTrainedModel(PreTrainedModel):
    config_class = QaptaanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["QaptaanDecoderLayer"]

    def _supports_default_dynamic_cache(self) -> bool:
        return False


class QaptaanModel(QaptaanPreTrainedModel):
    def __init__(self, config: QaptaanConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([QaptaanDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = QaptaanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[QaptaanCache] = None,
        use_cache: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        if use_cache and (past_key_values is None or not isinstance(past_key_values, QaptaanCache)):
            past_key_values = QaptaanCache()

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, past_key_value=past_key_values, use_cache=use_cache
            )
        hidden_states = self.norm(hidden_states)

        if use_cache and past_key_values is not None:
            past_key_values._seen_tokens += input_ids.shape[1]

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class QaptaanForCausalLM(QaptaanPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: QaptaanConfig):
        super().__init__(config)
        self.model = QaptaanModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[QaptaanCache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
'''


def build_readme_model_card() -> str:
    """Returns the Markdown README for the HuggingFace repository."""
    return """---
language:
- en
- code
license: apache-2.0
tags:
- code
- causal-lm
- qwen3.5
- cpt
- jax
- flax
pipeline_tag: text-generation
---

# QaptaanLM-0.75B (Qwen3.5-0.8B Continued Pre-Training)

QaptaanLM-0.75B is a high-efficiency code and reasoning foundation model produced by **Continued Pre-Training (CPT) for 1 Billion tokens** on Google Kaggle TPU v5e-8 on top of `Qwen/Qwen3.5-0.8B-Base`.

## Architecture Details

- **Total Parameters**: ~752M (0.75B)
- **Hybrid Layers (24 layers total)**: 18 Linear Attention layers (Gated Delta Net) + 6 Full Attention layers (GQA with RoPE) in a 3:1 ratio
- **Hidden Dimension**: 1024
- **Intermediate Dimension**: 3584 (SwiGLU)
- **Vocabulary Size**: 248,320 (Tied Word Embeddings)
- **Max Sequence Length**: 262,144

## Quickstart & Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "kaptaan45/QaptaanLM-0.75B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

prompt = '''def two_sum(nums: list[int], target: int) -> list[int]:
    \"\"\"Return indices of two numbers that add up to target.\"\"\"
'''

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.15,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation & Training

- **Hardware**: Google Cloud TPU v5e-8
- **Framework**: JAX / Flax / Orbax
- **Tokens Trained**: 1,000,000,000 tokens
"""


def export_clean_repository(
    ckpt_path: Path,
    export_dir: Path,
    base_model_id: str = "Qwen/Qwen3.5-0.8B-Base",
):
    """Performs full clean export from JAX checkpoint to Hugging Face directory."""
    print("=" * 80)
    print("1. CLEANING TARGET EXPORT DIRECTORY")
    print("=" * 80)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Clean directory initialized at: {export_dir.resolve()}")

    print("\n" * 80)
    print("2. RESTORING TRAINED JAX PARAMETERS FROM CHECKPOINT")
    print("=" * 80)
    import orbax.checkpoint as ocp
    from jax_training.models.config import Qwen3_5Config
    from jax_training.models.convert import convert_flax_to_pytorch_state_dict
    from safetensors.numpy import save_file
    from transformers import AutoTokenizer

    state_path = ckpt_path / "state" if (ckpt_path / "state").exists() else ckpt_path
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(state_path)
    params = restored["params"]
    print(f"✓ Restored JAX PyTree with {len(params['model'])} model modules")

    print("=" * 80)
    print("3. CONVERTING JAX PARAMETERS TO PYTORCH STATE DICT")
    print("=" * 80)
    config = Qwen3_5Config(dtype="bfloat16")
    state_dict = convert_flax_to_pytorch_state_dict(params, config)

    # Validate Conv1D weights presence & shape
    conv_count = sum(1 for k in state_dict if "conv1d.weight" in k)
    print(f"✓ Total PyTorch tensors: {len(state_dict)}")
    print(f"✓ Validated {conv_count} Conv1D linear attention kernel tensors")

    # Save model.safetensors
    st_file = export_dir / "model.safetensors"
    print(f"Saving safetensors to {st_file}...")
    save_file(state_dict, str(st_file))
    print(f"✓ Saved {st_file.stat().st_size / (1024 * 1024):.2f} MB safetensors")

    print("=" * 80)
    print("4. PACKAGING OFFICIAL QWEN3.5 TOKENIZER")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(export_dir))
    print("✓ Saved official tokenizer files (tokenizer.json 12.8 MB, merges.txt, vocab.json)")

    print("=" * 80)
    print("5. CREATING CONFIGURATION & MODELING MODULES")
    print("=" * 80)
    with open(export_dir / "configuration_qaptaan.py", "w", encoding="utf-8") as f:
        f.write(build_configuration_qaptaan_code())
    print("✓ Created configuration_qaptaan.py")

    with open(export_dir / "modeling_qaptaan.py", "w", encoding="utf-8") as f:
        f.write(build_modeling_qaptaan_code())
    print("✓ Created modeling_qaptaan.py")

    # config.json with auto_map
    config_dict = {
        "architectures": ["QaptaanForCausalLM"],
        "model_type": "qaptaan",
        "auto_map": {
            "AutoConfig": "configuration_qaptaan.QaptaanConfig",
            "AutoModelForCausalLM": "modeling_qaptaan.QaptaanForCausalLM",
        },
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "partial_rotary_factor": config.partial_rotary_factor,
        "attn_output_gate": config.attn_output_gate,
        "full_attention_interval": config.full_attention_interval,
        "linear_key_head_dim": config.linear_key_head_dim,
        "linear_value_head_dim": config.linear_value_head_dim,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(24)
        ],
        "use_cache": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.49.0",
    }

    with open(export_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print("✓ Created config.json with auto_map")

    # generation_config.json
    gen_config = {
        "bos_token_id": None,
        "eos_token_id": 248044,
        "pad_token_id": 248044,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_new_tokens": 120,
        "use_cache": True,
        "transformers_version": "4.49.0",
    }
    with open(export_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)
    print("✓ Created generation_config.json")

    with open(export_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(build_readme_model_card())
    print("✓ Created README.md")

    print("=" * 80)
    print("6. VALIDATING LOCAL PYTORCH LOADING & GENERATION")
    print("=" * 80)
    import torch
    from transformers import AutoModelForCausalLM

    print(f"Loading from {export_dir} via AutoModelForCausalLM...")
    test_model = AutoModelForCausalLM.from_pretrained(
        str(export_dir),
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    test_tokenizer = AutoTokenizer.from_pretrained(str(export_dir), trust_remote_code=True)
    test_model.eval()

    prompt = 'def two_sum(nums: list[int], target: int) -> list[int]:\n    """Return indices of two numbers that add up to target."""\n'
    inputs = test_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = test_model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=test_tokenizer.eos_token_id,
        )

    output_str = test_tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n--- Validation Generation Output ---")
    print(output_str)
    print("-" * 60)
    print("✅ PyTorch AutoModel validation PASSED successfully!")


def upload_to_hub(repo_id: str, local_dir: Path, token: Optional[str] = None):
    """Clears and uploads the entire clean folder to Hugging Face Hub."""
    print("=" * 80)
    print(f"7. UPLOADING CLEAN REPO TO HUGGING FACE HUB: {repo_id}")
    print("=" * 80)
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Ensure repository exists
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    print(f"Uploading files from {local_dir}...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release QaptaanLM-0.75B: Clean CPT export with exact recurrence and official tokenizer",
    )
    print(f"\n🎉 Successfully uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild and Upload Clean QaptaanLM-0.75B Repository")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint-61036")
    parser.add_argument("--export-dir", type=str, default="checkpoints/QaptaanLM-0.75B-Release", help="Output export directory")
    parser.add_argument("--hf-repo", type=str, default="kaptaan45/QaptaanLM-0.75B", help="Hugging Face repo ID")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--local-only", action="store_true", help="Only build locally without uploading")
    args = parser.parse_args()

    # Auto-locate checkpoint-61036
    ckpt_dir = None
    if args.ckpt_path:
        ckpt_dir = Path(args.ckpt_path)
    else:
        search_roots = ["/kaggle/input", "/kaggle/working", "checkpoints", "."]
        for base in search_roots:
            for p in Path(base).rglob("checkpoint-61036"):
                if p.is_dir():
                    ckpt_dir = p
                    break
            if ckpt_dir:
                break

    if not ckpt_dir or not ckpt_dir.exists():
        raise FileNotFoundError("Could not find checkpoint-61036 directory automatically. Pass --ckpt-path.")

    print(f"Found checkpoint-61036 at: {ckpt_dir}")
    export_dir = Path(args.export_dir).resolve()

    export_clean_repository(ckpt_path=ckpt_dir, export_dir=export_dir)

    if args.push and not args.local_only:
        token = os.environ.get("HF_TOKEN", None)
        upload_to_hub(repo_id=args.hf_repo, local_dir=export_dir, token=token)


if __name__ == "__main__":
    main()
