"""Numerically stable chunked linear cross-entropy loss in JAX.

Avoids materializing the full [batch, sequence, vocab_size] logits tensor
(which requires ~18 GB for 248k vocabulary on TPU), computing loss and gradients
in sequential chunks via jax.lax.scan.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp


def chunked_linear_cross_entropy(
    hidden_states: jax.Array,
    labels: jax.Array,
    embedding_weights: jax.Array,
    chunk_size: int = 256,
    ignore_index: int = -100,
) -> Tuple[jax.Array, jax.Array]:
    """Compute chunked causal LM cross-entropy loss with tied embedding weights.

    Predicts `labels[:, 1:]` from `hidden_states[:, :-1, :]`.

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim] tensor.
        labels: [batch_size, seq_len] token IDs (with ignore_index for masked tokens).
        embedding_weights: [vocab_size, hidden_dim] embedding/head weight matrix.
        chunk_size: Number of tokens to project and score per chunk (default: 256).
        ignore_index: Label value to ignore in loss calculation (default: -100).

    Returns:
        Tuple of (scalar_loss, num_valid_tokens).
    """
    # Shift for causal LM: predict token t+1 from hidden state t
    flat_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])
    flat_labels = labels[:, 1:].reshape(-1)

    total_tokens = flat_labels.shape[0]
    remainder = total_tokens % chunk_size
    pad_len = (chunk_size - remainder) % chunk_size

    if pad_len > 0:
        flat_hidden = jnp.pad(flat_hidden, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
        flat_labels = jnp.pad(flat_labels, ((0, pad_len),), mode="constant", constant_values=ignore_index)

    num_chunks = (total_tokens + pad_len) // chunk_size
    hidden_chunks = flat_hidden.reshape(num_chunks, chunk_size, -1)
    label_chunks = flat_labels.reshape(num_chunks, chunk_size)

    @jax.checkpoint
    def _chunk_step(carry, chunk_inputs):
        total_loss, total_count = carry
        h_chunk, y_chunk = chunk_inputs

        # Compute logits for this chunk: [chunk_size, vocab_size] in FP32
        logits = jnp.dot(h_chunk.astype(jnp.float32), embedding_weights.astype(jnp.float32).T)

        # Numerically stable log_softmax
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        exp_logits = jnp.exp(logits - max_logits)
        sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
        log_sum_exp = jnp.log(jnp.maximum(sum_exp, 1e-20)) + max_logits

        valid_mask = (y_chunk != ignore_index) & (y_chunk >= 0) & (y_chunk < embedding_weights.shape[0])
        safe_labels = jnp.where(valid_mask, y_chunk, 0)

        # Extract logit corresponding to the true target label
        target_logits = jnp.take_along_axis(logits, safe_labels[:, None], axis=-1).squeeze(-1)
        lse = log_sum_exp.squeeze(-1)

        # Cross entropy = log_sum_exp - target_logit
        chunk_losses = jnp.where(valid_mask, lse - target_logits, 0.0)

        chunk_loss_sum = jnp.sum(chunk_losses)
        chunk_valid_count = jnp.sum(valid_mask.astype(jnp.float32))

        new_total_loss = total_loss + chunk_loss_sum
        new_total_count = total_count + chunk_valid_count
        return (new_total_loss, new_total_count), None

    init_carry = (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32))
    (sum_loss, valid_tokens), _ = jax.lax.scan(_chunk_step, init_carry, (hidden_chunks, label_chunks))

    mean_loss = sum_loss / jnp.maximum(valid_tokens, 1.0)
    return mean_loss, valid_tokens
