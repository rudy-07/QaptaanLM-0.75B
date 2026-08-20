"""PyTorch/XLA-specific model helpers.

The Qwen3.5 text model has a 248,320-token tied output vocabulary.  The stock
causal-LM forward creates the complete ``[batch, sequence, vocab]`` logits
tensor before applying cross entropy.  On a 16GB v5e core that unnecessarily
limits the per-core batch size.  The helper below keeps the same loss while
computing the projection and cross entropy in fixed token chunks.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


def enable_chunked_causal_lm_loss(model: Any, chunk_size: int = 512) -> bool:
    """Patch a Qwen-style causal LM to compute its loss in token chunks.

    The patch is instance-local and is only used for the current training
    process.  ``save_pretrained`` still serializes the original model class;
    reloaded checkpoints use the normal forward path for generation/evaluation.
    Returns ``False`` when the model does not expose the expected Qwen-style
    ``model`` and ``lm_head`` members.
    """

    if chunk_size <= 0 or not hasattr(model, "model") or not hasattr(model, "lm_head"):
        return False
    if getattr(model, "_qwen_chunked_loss_enabled", False):
        return True

    original_forward = model.forward

    def forward_with_chunked_loss(self, *args, **kwargs):
        labels = kwargs.get("labels")
        if labels is None:
            return original_forward(*args, **kwargs)

        # Remove Trainer's bookkeeping scalar before forwarding to the base
        # text model.  It is consumed only by the loss calculation below.
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        use_cache = kwargs.pop("use_cache", None)
        kwargs.pop("labels", None)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # Causal LM loss predicts labels[..., 1:] from hidden_states[..., :-1].
        flat_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])
        flat_labels = labels[..., 1:].reshape(-1).to(device=flat_hidden.device)

        loss_sum = flat_hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, flat_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, flat_hidden.shape[0])
            logits = self.lm_head(flat_hidden[start:end]).float()
            loss_sum = loss_sum + F.cross_entropy(
                logits,
                flat_labels[start:end],
                ignore_index=-100,
                reduction="sum",
            )

        if num_items_in_batch is None:
            denominator = (flat_labels != -100).sum().clamp_min(1)
        else:
            denominator = num_items_in_batch
            if not torch.is_tensor(denominator):
                denominator = torch.tensor(denominator, device=loss_sum.device)
            denominator = denominator.to(device=loss_sum.device).clamp_min(1)
        loss = loss_sum / denominator

        # During training Trainer consumes only ``loss``.  Omitting the full
        # logits tensor is the memory saving that makes batch=2 practical.
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = types.MethodType(forward_with_chunked_loss, model)
    model._qwen_chunked_loss_enabled = True
    model._qwen_original_forward = original_forward
    return True

