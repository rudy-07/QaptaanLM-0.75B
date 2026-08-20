"""Unit test verifying chunked linear cross-entropy numerical correctness."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F

from jax_training.models.loss import chunked_linear_cross_entropy


class TestChunkedCrossEntropy(unittest.TestCase):
    """Test chunked cross-entropy against standard PyTorch cross-entropy."""

    def test_numerical_equivalence(self):
        batch_size = 2
        seq_len = 32
        vocab_size = 1000
        hidden_dim = 128
        chunk_size = 8

        rng = np.random.default_rng(42)
        h_np = rng.normal(size=(batch_size, seq_len, hidden_dim)).astype(np.float32)
        w_np = rng.normal(size=(vocab_size, hidden_dim)).astype(np.float32)
        y_np = rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
        # Add some ignored labels (-100)
        y_np[0, 5] = -100
        y_np[1, 10] = -100

        # --- 1. PyTorch Standard Loss ---
        h_torch = torch.tensor(h_np)
        w_torch = torch.tensor(w_np)
        y_torch = torch.tensor(y_np, dtype=torch.long)

        # Predict labels[:, 1:] from hidden_states[:, :-1, :]
        flat_h_pt = h_torch[:, :-1, :].reshape(-1, hidden_dim)
        flat_y_pt = y_torch[:, 1:].reshape(-1)

        full_logits_pt = flat_h_pt @ w_torch.T
        pt_loss = F.cross_entropy(full_logits_pt, flat_y_pt, ignore_index=-100, reduction="mean").item()

        # --- 2. JAX Chunked Loss ---
        h_jax = jnp.array(h_np)
        w_jax = jnp.array(w_np)
        y_jax = jnp.array(y_np)

        jax_loss, valid_tokens = chunked_linear_cross_entropy(
            hidden_states=h_jax,
            labels=y_jax,
            embedding_weights=w_jax,
            chunk_size=chunk_size,
            ignore_index=-100,
        )
        jax_loss_val = float(jax_loss)

        print(f"\nPyTorch standard loss: {pt_loss:.6f}")
        print(f"JAX chunked loss:     {jax_loss_val:.6f}")
        print(f"Valid tokens:         {int(valid_tokens)}")

        self.assertAlmostEqual(pt_loss, jax_loss_val, places=4)

    def test_gradient_computation(self):
        """Verify gradients can be computed through chunked loss via jax.grad."""
        batch_size = 2
        seq_len = 16
        vocab_size = 500
        hidden_dim = 64

        h_jax = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
        w_jax = jnp.ones((vocab_size, hidden_dim), dtype=jnp.float32)
        y_jax = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

        def loss_fn(h, w):
            loss, _ = chunked_linear_cross_entropy(h, y_jax, w, chunk_size=8)
            return loss

        grad_h, grad_w = jax.grad(loss_fn, argnums=(0, 1))(h_jax, w_jax)
        self.assertEqual(grad_h.shape, h_jax.shape)
        self.assertEqual(grad_w.shape, w_jax.shape)
        self.assertFalse(jnp.any(jnp.isnan(grad_h)))
        self.assertFalse(jnp.any(jnp.isnan(grad_w)))


if __name__ == "__main__":
    unittest.main()
