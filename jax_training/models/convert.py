"""Bidirectional parameter conversion between HuggingFace PyTorch and Flax/JAX."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from jax_training.models.config import Qwen3_5Config


def convert_pytorch_to_flax_params(
    pt_state_dict: Dict[str, Any],
    config: Qwen3_5Config,
    target_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    """Convert a PyTorch Qwen3.5 state dict to a Flax Linen parameter dictionary."""
    jnp_dtype = jnp.bfloat16 if target_dtype == "bfloat16" else (jnp.float16 if target_dtype == "float16" else jnp.float32)

    def to_jax(tensor):
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        return jnp.array(tensor, dtype=jnp_dtype)

    params: Dict[str, Any] = {"model": {}}
    model_params = params["model"]

    # Embed tokens
    if "model.embed_tokens.weight" in pt_state_dict:
        model_params["embed_tokens"] = {
            "embedding": to_jax(pt_state_dict["model.embed_tokens.weight"])
        }

    # Final norm
    if "model.norm.weight" in pt_state_dict:
        model_params["norm"] = {
            "weight": to_jax(pt_state_dict["model.norm.weight"])
        }

    # Layers
    for i in range(config.num_hidden_layers):
        layer_name = f"layers_{i}"
        layer_dict: Dict[str, Any] = {}
        prefix = f"model.layers.{i}."

        # Norms
        if f"{prefix}input_layernorm.weight" in pt_state_dict:
            layer_dict["input_layernorm"] = {
                "weight": to_jax(pt_state_dict[f"{prefix}input_layernorm.weight"])
            }
        if f"{prefix}post_attention_layernorm.weight" in pt_state_dict:
            layer_dict["post_attention_layernorm"] = {
                "weight": to_jax(pt_state_dict[f"{prefix}post_attention_layernorm.weight"])
            }

        # MLP: linear weights in Flax Dense have shape [in_features, out_features] (transposed vs PyTorch)
        mlp_dict: Dict[str, Any] = {}
        if f"{prefix}mlp.gate_proj.weight" in pt_state_dict:
            mlp_dict["gate_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}mlp.gate_proj.weight"].T)}
        if f"{prefix}mlp.up_proj.weight" in pt_state_dict:
            mlp_dict["up_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}mlp.up_proj.weight"].T)}
        if f"{prefix}mlp.down_proj.weight" in pt_state_dict:
            mlp_dict["down_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}mlp.down_proj.weight"].T)}
        layer_dict["mlp"] = mlp_dict

        # Attention: Full Attention vs Linear Attention
        is_full_attn = (i + 1) % config.full_attention_interval == 0
        if is_full_attn:
            attn_dict: Dict[str, Any] = {}
            if f"{prefix}self_attn.q_proj.weight" in pt_state_dict:
                attn_dict["q_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}self_attn.q_proj.weight"].T)}
            if f"{prefix}self_attn.k_proj.weight" in pt_state_dict:
                attn_dict["k_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}self_attn.k_proj.weight"].T)}
            if f"{prefix}self_attn.v_proj.weight" in pt_state_dict:
                attn_dict["v_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}self_attn.v_proj.weight"].T)}
            if f"{prefix}self_attn.o_proj.weight" in pt_state_dict:
                attn_dict["o_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}self_attn.o_proj.weight"].T)}
            if f"{prefix}self_attn.q_norm.weight" in pt_state_dict:
                attn_dict["q_norm"] = {"weight": to_jax(pt_state_dict[f"{prefix}self_attn.q_norm.weight"])}
            if f"{prefix}self_attn.k_norm.weight" in pt_state_dict:
                attn_dict["k_norm"] = {"weight": to_jax(pt_state_dict[f"{prefix}self_attn.k_norm.weight"])}
            layer_dict["self_attn"] = attn_dict
        else:
            linear_dict: Dict[str, Any] = {}
            if f"{prefix}linear_attn.in_proj_qkv.weight" in pt_state_dict:
                linear_dict["in_proj_qkv"] = {"kernel": to_jax(pt_state_dict[f"{prefix}linear_attn.in_proj_qkv.weight"].T)}
            if f"{prefix}linear_attn.in_proj_z.weight" in pt_state_dict:
                linear_dict["in_proj_z"] = {"kernel": to_jax(pt_state_dict[f"{prefix}linear_attn.in_proj_z.weight"].T)}
            if f"{prefix}linear_attn.in_proj_b.weight" in pt_state_dict:
                linear_dict["in_proj_b"] = {"kernel": to_jax(pt_state_dict[f"{prefix}linear_attn.in_proj_b.weight"].T)}
            if f"{prefix}linear_attn.in_proj_a.weight" in pt_state_dict:
                linear_dict["in_proj_a"] = {"kernel": to_jax(pt_state_dict[f"{prefix}linear_attn.in_proj_a.weight"].T)}
            if f"{prefix}linear_attn.out_proj.weight" in pt_state_dict:
                linear_dict["out_proj"] = {"kernel": to_jax(pt_state_dict[f"{prefix}linear_attn.out_proj.weight"].T)}
            if f"{prefix}linear_attn.norm.weight" in pt_state_dict:
                linear_dict["norm"] = {"weight": to_jax(pt_state_dict[f"{prefix}linear_attn.norm.weight"])}
            if f"{prefix}linear_attn.dt_bias" in pt_state_dict:
                linear_dict["dt_bias"] = to_jax(pt_state_dict[f"{prefix}linear_attn.dt_bias"])
            if f"{prefix}linear_attn.A_log" in pt_state_dict:
                linear_dict["A_log"] = to_jax(pt_state_dict[f"{prefix}linear_attn.A_log"])
            if f"{prefix}linear_attn.conv1d.weight" in pt_state_dict:
                linear_dict["conv1d_weight"] = to_jax(pt_state_dict[f"{prefix}linear_attn.conv1d.weight"])
            layer_dict["linear_attn"] = linear_dict

        model_params[layer_name] = layer_dict

    return freeze(params)


def convert_flax_to_pytorch_state_dict(
    flax_params: Dict[str, Any],
    config: Qwen3_5Config,
) -> Dict[str, np.ndarray]:
    """Convert Flax Linen parameter dictionary back to PyTorch state dict (numpy arrays)."""
    flax_params = unfreeze(flax_params)
    model_params = flax_params.get("params", flax_params).get("model", {})
    state_dict: Dict[str, np.ndarray] = {}

    def to_np(arr):
        return np.array(arr)

    # Embed tokens & norm
    if "embed_tokens" in model_params:
        state_dict["model.embed_tokens.weight"] = to_np(model_params["embed_tokens"]["embedding"])
        # Tied lm_head
        if config.tie_word_embeddings:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    if "norm" in model_params:
        state_dict["model.norm.weight"] = to_np(model_params["norm"]["weight"])

    # Layers
    for i in range(config.num_hidden_layers):
        layer_name = f"layers_{i}"
        if layer_name not in model_params:
            continue
        ld = model_params[layer_name]
        prefix = f"model.layers.{i}."

        if "input_layernorm" in ld:
            state_dict[f"{prefix}input_layernorm.weight"] = to_np(ld["input_layernorm"]["weight"])
        if "post_attention_layernorm" in ld:
            state_dict[f"{prefix}post_attention_layernorm.weight"] = to_np(ld["post_attention_layernorm"]["weight"])

        if "mlp" in ld:
            state_dict[f"{prefix}mlp.gate_proj.weight"] = to_np(ld["mlp"]["gate_proj"]["kernel"].T)
            state_dict[f"{prefix}mlp.up_proj.weight"] = to_np(ld["mlp"]["up_proj"]["kernel"].T)
            state_dict[f"{prefix}mlp.down_proj.weight"] = to_np(ld["mlp"]["down_proj"]["kernel"].T)

        if "self_attn" in ld:
            sa = ld["self_attn"]
            state_dict[f"{prefix}self_attn.q_proj.weight"] = to_np(sa["q_proj"]["kernel"].T)
            state_dict[f"{prefix}self_attn.k_proj.weight"] = to_np(sa["k_proj"]["kernel"].T)
            state_dict[f"{prefix}self_attn.v_proj.weight"] = to_np(sa["v_proj"]["kernel"].T)
            state_dict[f"{prefix}self_attn.o_proj.weight"] = to_np(sa["o_proj"]["kernel"].T)
            state_dict[f"{prefix}self_attn.q_norm.weight"] = to_np(sa["q_norm"]["weight"])
            state_dict[f"{prefix}self_attn.k_norm.weight"] = to_np(sa["k_norm"]["weight"])

        if "linear_attn" in ld:
            la = ld["linear_attn"]
            state_dict[f"{prefix}linear_attn.in_proj_qkv.weight"] = to_np(la["in_proj_qkv"]["kernel"].T)
            state_dict[f"{prefix}linear_attn.in_proj_z.weight"] = to_np(la["in_proj_z"]["kernel"].T)
            state_dict[f"{prefix}linear_attn.in_proj_b.weight"] = to_np(la["in_proj_b"]["kernel"].T)
            state_dict[f"{prefix}linear_attn.in_proj_a.weight"] = to_np(la["in_proj_a"]["kernel"].T)
            state_dict[f"{prefix}linear_attn.out_proj.weight"] = to_np(la["out_proj"]["kernel"].T)
            state_dict[f"{prefix}linear_attn.norm.weight"] = to_np(la["norm"]["weight"])
            state_dict[f"{prefix}linear_attn.dt_bias"] = to_np(la["dt_bias"])
            state_dict[f"{prefix}linear_attn.A_log"] = to_np(la["A_log"])
            state_dict[f"{prefix}linear_attn.conv1d.weight"] = to_np(la["conv1d_weight"])

    return state_dict


def load_hf_weights_into_flax(
    model_name_or_path: str,
    config: Qwen3_5Config,
    target_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    """Load weights from Hugging Face repository or local path into Flax parameter dict."""
    try:
        from safetensors.torch import load_file as load_safetensors
        import torch

        # Check local safetensors
        local_path = Path(model_name_or_path)
        pt_state_dict = {}
        if local_path.exists() and local_path.is_dir():
            st_files = list(local_path.glob("*.safetensors"))
            if st_files:
                for st in st_files:
                    pt_state_dict.update(load_safetensors(str(st)))
            else:
                bin_files = list(local_path.glob("*.bin"))
                for b in bin_files:
                    pt_state_dict.update(torch.load(str(b), map_location="cpu"))
        else:
            # Load via transformers AutoModelForCausalLM / Qwen3_5ForCausalLM
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            pt_state_dict = hf_model.state_dict()

        return convert_pytorch_to_flax_params(pt_state_dict, config, target_dtype=target_dtype)

    except Exception as e:
        raise RuntimeError(f"Failed to load weights from {model_name_or_path}: {e}") from e
