"""Qwen3.5 Text Model Implementation in Flax/JAX.

Supports hybrid Linear Attention (Gated Delta Net) and Full Attention (GQA with RoPE),
SwiGLU MLP, and RMSNorm with output gating.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_training.models.config import Qwen3_5Config


def _get_dtype(dtype_str: str) -> jnp.dtype:
    mapping = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }
    return mapping.get(dtype_str, jnp.bfloat16)


class Qwen3_5RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    dim: int
    eps: float = 1e-6
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        compute_dtype = _get_dtype(self.dtype)
        weight = self.param("weight", nn.initializers.ones, (self.dim,), compute_dtype)
        x_fp32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True)
        normed = x_fp32 * jax.lax.rsqrt(variance + self.eps)
        return (normed.astype(compute_dtype) * weight).astype(compute_dtype)


class Qwen3_5RMSNormGated(nn.Module):
    """Gated RMSNorm used in Gated Delta Net (Linear Attention)."""

    dim: int
    eps: float = 1e-6
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
        compute_dtype = _get_dtype(self.dtype)
        weight = self.param("weight", nn.initializers.ones, (self.dim,), compute_dtype)
        x_fp32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True)
        normed = x_fp32 * jax.lax.rsqrt(variance + self.eps)
        normed = (normed.astype(compute_dtype) * weight).astype(compute_dtype)
        gated = normed * jax.nn.silu(gate.astype(jnp.float32)).astype(compute_dtype)
        return gated


class Qwen3_5RotaryEmbedding:
    """Rotary Position Embedding (RoPE) for Qwen3.5."""

    def __init__(self, dim: int, max_position_embeddings: int = 262144, base: float = 10000000.0):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Inv freq for rotary dimensions: [dim // 2]
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))

    def get_rotary_emb(self, seq_len: int, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # [seq_len, dim]
        cos = jnp.cos(emb).astype(dtype)
        sin = jnp.sin(emb).astype(dtype)
        return cos, sin

    @staticmethod
    def rotate_half(x: jax.Array) -> jax.Array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return jnp.concatenate([-x2, x1], axis=-1)

    @classmethod
    def apply_rope(
        cls, query: jax.Array, key: jax.Array, cos: jax.Array, sin: jax.Array, rotary_dim: int
    ) -> Tuple[jax.Array, jax.Array]:
        """Apply RoPE to partial rotary dimensions of query and key.

        query: [batch, seq_len, num_heads, head_dim]
        key:   [batch, seq_len, num_kv_heads, head_dim]
        cos, sin: [seq_len, rotary_dim]
        """
        # Split into rotary and pass-through portions
        q_rot = query[..., :rotary_dim]
        q_pass = query[..., rotary_dim:]
        k_rot = key[..., :rotary_dim]
        k_pass = key[..., rotary_dim:]

        # Broadcast cos, sin: [1, seq_len, 1, rotary_dim]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        q_rot_embed = (q_rot * cos) + (cls.rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (cls.rotate_half(k_rot) * sin)

        q_out = jnp.concatenate([q_rot_embed, q_pass], axis=-1)
        k_out = jnp.concatenate([k_rot_embed, k_pass], axis=-1)
        return q_out, k_out


class Qwen3_5MLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron."""

    config: Qwen3_5Config

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        compute_dtype = _get_dtype(self.config.dtype)
        gate_proj = nn.Dense(self.config.intermediate_size, use_bias=False, dtype=compute_dtype, name="gate_proj")
        up_proj = nn.Dense(self.config.intermediate_size, use_bias=False, dtype=compute_dtype, name="up_proj")
        down_proj = nn.Dense(self.config.hidden_size, use_bias=False, dtype=compute_dtype, name="down_proj")

        gate = jax.nn.silu(gate_proj(x))
        up = up_proj(x)
        return down_proj(gate * up)


class Qwen3_5Attention(nn.Module):
    """Full Attention Layer with Grouped-Query Attention (GQA), RoPE, and Output Gating."""

    config: Qwen3_5Config
    layer_idx: int

    def setup(self):
        self.compute_dtype = _get_dtype(self.config.dtype)
        self.head_dim = self.config.head_dim
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        # q_proj outputs (2 * num_heads * head_dim) for query and gate
        self.q_proj = nn.Dense(self.num_heads * self.head_dim * 2, use_bias=False, dtype=self.compute_dtype, name="q_proj")
        self.k_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.compute_dtype, name="k_proj")
        self.v_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, dtype=self.compute_dtype, name="v_proj")
        self.o_proj = nn.Dense(self.config.hidden_size, use_bias=False, dtype=self.compute_dtype, name="o_proj")

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=self.config.rms_norm_eps, dtype=self.config.dtype, name="q_norm")
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=self.config.rms_norm_eps, dtype=self.config.dtype, name="k_norm")
        self.rotary = Qwen3_5RotaryEmbedding(self.config.rotary_dim, self.config.max_position_embeddings, self.config.rope_theta)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        batch_size, seq_len, _ = hidden_states.shape

        q_proj_out = self.q_proj(hidden_states)
        q_proj_out = q_proj_out.reshape(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        query_states = q_proj_out[..., : self.head_dim]
        gate = q_proj_out[..., self.head_dim :]

        key_states = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply head normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE
        cos, sin = self.rotary.get_rotary_emb(seq_len, dtype=self.compute_dtype)
        query_states, key_states = Qwen3_5RotaryEmbedding.apply_rope(
            query_states, key_states, cos, sin, self.config.rotary_dim
        )

        # Expand KV heads for GQA: [B, S, num_heads, head_dim]
        if self.num_kv_groups > 1:
            key_states = jnp.repeat(key_states, self.num_kv_groups, axis=2)
            value_states = jnp.repeat(value_states, self.num_kv_groups, axis=2)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = jnp.transpose(query_states, (0, 2, 1, 3))
        k = jnp.transpose(key_states, (0, 2, 1, 3))
        v = jnp.transpose(value_states, (0, 2, 1, 3))

        # Hardware-accelerated FlashAttention
        if attention_mask is not None:
            attn_out = jax.nn.dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                mask=attention_mask,
                is_causal=True,
            )
        else:
            attn_out = jax.nn.dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                is_causal=True,
            )

        # Transpose back: [B, S, H, head_dim]
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))

        # Apply output gate: gate is [B, S, H, head_dim]
        attn_out = attn_out * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(self.compute_dtype)
        attn_out = attn_out.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_out)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Linear Attention Layer (Gated Delta Net with 1D Causal Conv & Recurrent Scan)."""

    config: Qwen3_5Config
    layer_idx: int

    def setup(self):
        self.compute_dtype = _get_dtype(self.config.dtype)
        self.num_k_heads = self.config.linear_num_key_heads
        self.num_v_heads = self.config.linear_num_value_heads
        self.head_k_dim = self.config.linear_key_head_dim
        self.head_v_dim = self.config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = self.config.linear_conv_kernel_dim

        self.in_proj_qkv = nn.Dense(
            self.key_dim * 2 + self.value_dim, use_bias=False, dtype=self.compute_dtype, name="in_proj_qkv"
        )
        self.in_proj_z = nn.Dense(self.value_dim, use_bias=False, dtype=self.compute_dtype, name="in_proj_z")
        self.in_proj_b = nn.Dense(self.num_v_heads, use_bias=False, dtype=self.compute_dtype, name="in_proj_b")
        self.in_proj_a = nn.Dense(self.num_v_heads, use_bias=False, dtype=self.compute_dtype, name="in_proj_a")

        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, eps=self.config.rms_norm_eps, dtype=self.config.dtype, name="norm"
        )
        self.out_proj = nn.Dense(self.config.hidden_size, use_bias=False, dtype=self.compute_dtype, name="out_proj")

    @nn.compact
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = hidden_states.shape
        compute_dtype = self.compute_dtype

        # Decay parameters
        dt_bias = self.param("dt_bias", nn.initializers.ones, (self.num_v_heads,), compute_dtype)
        A_log = self.param("A_log", nn.initializers.uniform(scale=2.0), (self.num_v_heads,), compute_dtype)

        # 1D depthwise causal convolution weight: [conv_dim, 1, kernel_size] in PyTorch -> [kernel_size, conv_dim] in JAX
        conv_dim = self.key_dim * 2 + self.value_dim
        conv_weight = self.param(
            "conv1d_weight",
            nn.initializers.normal(stddev=0.02),
            (conv_dim, 1, self.conv_kernel_size),
            compute_dtype,
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, S, conv_dim]
        z = self.in_proj_z(hidden_states)            # [B, S, value_dim]
        b = self.in_proj_b(hidden_states)            # [B, S, num_v_heads]
        a = self.in_proj_a(hidden_states)            # [B, S, num_v_heads]

        # 1D Causal Convolution: pad left with (kernel_size - 1) zeros
        # conv_weight: [conv_dim, 1, K]
        w_2d = conv_weight.squeeze(1).T  # [K, conv_dim]
        pad_size = self.conv_kernel_size - 1
        padded_qkv = jnp.pad(mixed_qkv, ((0, 0), (pad_size, 0), (0, 0)), mode="constant")
        
        # Depthwise 1D conv via unrolled sliding windows
        conv_out = jnp.zeros_like(mixed_qkv)
        for k in range(self.conv_kernel_size):
            conv_out = conv_out + padded_qkv[:, k : k + seq_len, :] * w_2d[k : k + 1, :]

        mixed_qkv = jax.nn.silu(conv_out)

        query = mixed_qkv[:, :, : self.key_dim].reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = mixed_qkv[:, :, self.key_dim : 2 * self.key_dim].reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = mixed_qkv[:, :, 2 * self.key_dim :].reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # L2-normalization on query and key
        q_norm = jnp.linalg.norm(query.astype(jnp.float32), axis=-1, keepdims=True) + 1e-6
        k_norm = jnp.linalg.norm(key.astype(jnp.float32), axis=-1, keepdims=True) + 1e-6
        query = (query / q_norm).astype(compute_dtype)
        key = (key / k_norm).astype(compute_dtype)

        # Expand K-heads if V-heads > K-heads
        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            query = jnp.repeat(query, ratio, axis=2)
            key = jnp.repeat(key, ratio, axis=2)

        beta = jax.nn.sigmoid(b.astype(jnp.float32))  # [B, S, num_v_heads]
        g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias.astype(jnp.float32))  # [B, S, num_v_heads]

        # Recurrent Gated Delta Net scan across sequence
        scale = 1.0 / (self.head_k_dim ** 0.5)
        q_scaled = (query.astype(jnp.float32) * scale)  # [B, S, H, K_dim]
        k_fp32 = key.astype(jnp.float32)                # [B, S, H, K_dim]
        v_fp32 = value.astype(jnp.float32)              # [B, S, H, V_dim]

        # Transpose to [seq_len, batch_size, num_heads, dim] for jax.lax.scan
        q_t = jnp.transpose(q_scaled, (1, 0, 2, 3))
        k_t = jnp.transpose(k_fp32, (1, 0, 2, 3))
        v_t = jnp.transpose(v_fp32, (1, 0, 2, 3))
        beta_t = jnp.transpose(beta, (1, 0, 2))  # [S, B, H]
        g_t = jnp.transpose(g, (1, 0, 2))        # [S, B, H]

        init_state = jnp.zeros((batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=jnp.float32)

        @jax.checkpoint
        def _step_fn(state, step_inputs):
            q_i, k_i, v_i, b_i, g_i = step_inputs
            decay = jnp.exp(g_i)[:, :, None, None]  # [B, H, 1, 1]
            b_i_exp = b_i[:, :, None]               # [B, H, 1]

            # Projected value: v_prime = k_i @ state -> [B, H, V_dim]
            v_prime = jnp.einsum("bhk,bhkd->bhd", k_i, state)
            v_new = (v_i - v_prime) * b_i_exp

            # Attention output: attn_inter = q_i @ state * decay + (q_i @ k_i) * v_new
            attn_inter = jnp.einsum("bhk,bhkd->bhd", q_i, state) * decay.squeeze(-1)
            qk_dot = jnp.sum(q_i * k_i, axis=-1, keepdims=True)  # [B, H, 1]
            out_i = attn_inter + qk_dot * v_new

            # Update state: state = state * decay + k_i.T @ v_new
            new_state = state * decay + jnp.einsum("bhk,bhd->bhkd", k_i, v_new)
            return new_state, out_i

        _, core_out_t = jax.lax.scan(_step_fn, init_state, (q_t, k_t, v_t, beta_t, g_t), unroll=16)
        core_attn_out = jnp.transpose(core_out_t, (1, 0, 2, 3)).astype(compute_dtype)  # [B, S, H, V_dim]

        # RMSNormGated with gate z: [B, S, H, V_dim]
        z_reshaped = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_reshaped)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)

        return self.out_proj(core_attn_out)


class Qwen3_5DecoderLayer(nn.Module):
    """Qwen3.5 Transformer Decoder Layer (Hybrid Linear/Full Attention + SwiGLU MLP)."""

    config: Qwen3_5Config
    layer_idx: int

    def setup(self):
        is_full_attn = (self.layer_idx + 1) % self.config.full_attention_interval == 0
        if is_full_attn:
            self.attn = Qwen3_5Attention(self.config, self.layer_idx, name="self_attn")
        else:
            self.attn = Qwen3_5GatedDeltaNet(self.config, self.layer_idx, name="linear_attn")

        self.mlp = Qwen3_5MLP(self.config, name="mlp")
        self.input_layernorm = Qwen3_5RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.config.dtype, name="input_layernorm"
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype,
            name="post_attention_layernorm",
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        # Pre-norm Attention + residual
        normed = self.input_layernorm(hidden_states)
        is_full_attn = (self.layer_idx + 1) % self.config.full_attention_interval == 0
        if is_full_attn:
            attn_output = self.attn(normed, attention_mask=attention_mask)
        else:
            attn_output = self.attn(normed)
        hidden_states = hidden_states + attn_output

        # Pre-norm MLP + residual
        normed = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed)
        hidden_states = hidden_states + mlp_output

        return hidden_states


class Qwen3_5Model(nn.Module):
    """Qwen3.5 Base Model (Embeddings + 24 Decoder Layers + Final RMSNorm)."""

    config: Qwen3_5Config

    def setup(self):
        compute_dtype = _get_dtype(self.config.dtype)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=compute_dtype,
            name="embed_tokens",
        )
        RematDecoderLayer = nn.remat(Qwen3_5DecoderLayer)
        self.layers = [
            RematDecoderLayer(self.config, i, name=f"layers_{i}")
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = Qwen3_5RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.config.dtype, name="norm"
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return self.norm(hidden_states)


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 Causal Language Model with Tied Word Embeddings."""

    config: Qwen3_5Config

    def setup(self):
        self.model = Qwen3_5Model(self.config, name="model")

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Forward pass returning the last hidden states before vocabulary projection."""
        return self.model(input_ids, attention_mask=attention_mask)

    def get_embedding_weights(self, params: Dict[str, Any]) -> jax.Array:
        """Extract tied embedding weights [vocab_size, hidden_size]."""
        return params["model"]["embed_tokens"]["embedding"]
