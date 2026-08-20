"""Unit tests for Flax Qwen3.5 model architecture and forward pass."""

import unittest
import jax
import jax.numpy as jnp

from jax_training.models.config import Qwen3_5Config
from jax_training.models.qwen3_5 import Qwen3_5ForCausalLM


class TestQwen3_5Model(unittest.TestCase):
    """Test model initialization and forward pass on small synthetic model config."""

    def setUp(self):
        self.config = Qwen3_5Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            full_attention_interval=2,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            dtype="float32",
        )
        self.model = Qwen3_5ForCausalLM(self.config)

    def test_forward_pass_shape(self):
        batch_size = 2
        seq_len = 16
        rng = jax.random.PRNGKey(42)

        input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, self.config.vocab_size)
        variables = self.model.init(rng, input_ids)
        params = variables["params"]

        # Forward pass returning hidden states
        hidden_states = self.model.apply({"params": params}, input_ids)

        self.assertEqual(hidden_states.shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertFalse(jnp.any(jnp.isnan(hidden_states)))

        # Test tied embedding extraction
        tied_w = self.model.get_embedding_weights(params)
        self.assertEqual(tied_w.shape, (self.config.vocab_size, self.config.hidden_size))


if __name__ == "__main__":
    unittest.main()
