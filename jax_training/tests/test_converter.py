"""Unit tests for bidirectional weight conversion between PyTorch and Flax."""

import unittest
import numpy as np
import torch

from jax_training.models.config import Qwen3_5Config
from jax_training.models.convert import (
    convert_pytorch_to_flax_params,
    convert_flax_to_pytorch_state_dict,
)


class TestWeightConverter(unittest.TestCase):
    """Test PyTorch <-> Flax weight conversion roundtrip."""

    def setUp(self):
        self.config = Qwen3_5Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            full_attention_interval=2,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            dtype="float32",
        )

    def test_roundtrip_conversion(self):
        # Create synthetic PyTorch state dict matching full text model
        pt_state_dict = {
            "model.embed_tokens.weight": torch.randn(self.config.vocab_size, self.config.hidden_size),
            "model.norm.weight": torch.ones(self.config.hidden_size),
            # Layer 0: linear_attention
            "model.layers.0.input_layernorm.weight": torch.ones(self.config.hidden_size),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(self.config.hidden_size),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(self.config.intermediate_size, self.config.hidden_size),
            "model.layers.0.mlp.up_proj.weight": torch.randn(self.config.intermediate_size, self.config.hidden_size),
            "model.layers.0.mlp.down_proj.weight": torch.randn(self.config.hidden_size, self.config.intermediate_size),
            "model.layers.0.linear_attn.in_proj_qkv.weight": torch.randn(2 * 32 + 32, self.config.hidden_size),
            "model.layers.0.linear_attn.in_proj_z.weight": torch.randn(32, self.config.hidden_size),
            "model.layers.0.linear_attn.in_proj_b.weight": torch.randn(2, self.config.hidden_size),
            "model.layers.0.linear_attn.in_proj_a.weight": torch.randn(2, self.config.hidden_size),
            "model.layers.0.linear_attn.out_proj.weight": torch.randn(self.config.hidden_size, 32),
            "model.layers.0.linear_attn.norm.weight": torch.ones(16),
            "model.layers.0.linear_attn.dt_bias": torch.ones(2),
            "model.layers.0.linear_attn.A_log": torch.randn(2),
            "model.layers.0.linear_attn.conv1d.weight": torch.randn(96, 1, 4),
            # Layer 1: full_attention
            "model.layers.1.input_layernorm.weight": torch.ones(self.config.hidden_size),
            "model.layers.1.post_attention_layernorm.weight": torch.ones(self.config.hidden_size),
            "model.layers.1.mlp.gate_proj.weight": torch.randn(self.config.intermediate_size, self.config.hidden_size),
            "model.layers.1.mlp.up_proj.weight": torch.randn(self.config.intermediate_size, self.config.hidden_size),
            "model.layers.1.mlp.down_proj.weight": torch.randn(self.config.hidden_size, self.config.intermediate_size),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(2 * 2 * 32, self.config.hidden_size),
            "model.layers.1.self_attn.k_proj.weight": torch.randn(1 * 32, self.config.hidden_size),
            "model.layers.1.self_attn.v_proj.weight": torch.randn(1 * 32, self.config.hidden_size),
            "model.layers.1.self_attn.o_proj.weight": torch.randn(self.config.hidden_size, 2 * 32),
            "model.layers.1.self_attn.q_norm.weight": torch.ones(32),
            "model.layers.1.self_attn.k_norm.weight": torch.ones(32),
        }

        # Convert PyTorch -> Flax
        flax_params = convert_pytorch_to_flax_params(pt_state_dict, self.config, target_dtype="float32")

        # Convert Flax -> PyTorch
        restored_pt = convert_flax_to_pytorch_state_dict(flax_params, self.config)

        # Verify all tensors match exactly
        for k, v in pt_state_dict.items():
            self.assertIn(k, restored_pt, f"Missing key {k} in restored state dict")
            orig_np = v.numpy()
            restored_np = restored_pt[k]
            np.testing.assert_allclose(
                orig_np,
                restored_np,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Tensor mismatch for key {k}",
            )


if __name__ == "__main__":
    unittest.main()
