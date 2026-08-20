"""Orbax Checkpoint Manager with Auto-Resume, Pruning, and HF Export."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import jax
from flax.training import train_state

from jax_training.models.config import Qwen3_5Config
from jax_training.models.convert import convert_flax_to_pytorch_state_dict

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages Orbax / Flax training state checkpointing and automatic resumption."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        max_to_keep: int = 2,
        save_interval_steps: int = 250,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.max_to_keep = max(1, max_to_keep)
        self.save_interval_steps = save_interval_steps
        self.is_primary = jax.process_index() == 0

        if self.is_primary:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._checkpointer = None
        self._init_orbax()

    def _init_orbax(self):
        try:
            import orbax.checkpoint as ocp
            # Use PyTreeCheckpointer or StandardCheckpointer
            if hasattr(ocp, "StandardCheckpointer"):
                self._checkpointer = ocp.StandardCheckpointer()
            else:
                self._checkpointer = ocp.PyTreeCheckpointer()
        except Exception as e:
            logger.warning(f"Could not initialize Orbax checkpointer: {e}. Using native Flax checkpointer.")

    def save(
        self,
        step: int,
        state: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save a training checkpoint with step and metadata."""
        if not self.is_primary:
            return None

        step_dir = self.output_dir / f"checkpoint-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self._checkpointer is not None:
                # Orbax save
                import orbax.checkpoint as ocp
                target_state = jax.device_get(state)
                self._checkpointer.save(step_dir / "state", target_state)
                if hasattr(self._checkpointer, "wait_until_finished"):
                    self._checkpointer.wait_until_finished()
            else:
                from flax.training import checkpoints
                checkpoints.save_checkpoint(
                    ckpt_dir=str(self.output_dir),
                    target=jax.device_get(state),
                    step=step,
                    keep=self.max_to_keep,
                    overwrite=True,
                )

            # Save metadata
            meta = {
                "step": step,
                "metadata": metadata or {},
            }
            with open(step_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"✓ Saved checkpoint to {step_dir}")
            self._prune_old_checkpoints()
            return step_dir

        except Exception as e:
            logger.exception(f"Failed to save checkpoint at step {step}: {e}")
            return None

    def wait_until_finished(self):
        """Wait for any background async checkpoint saving to complete."""
        if self._checkpointer is not None and hasattr(self._checkpointer, "wait_until_finished"):
            try:
                self._checkpointer.wait_until_finished()
            except Exception as e:
                logger.warning(f"Error waiting for checkpoint completion: {e}")

    def restore(self, checkpoint_path: Union[str, Path], target_state: Any) -> Any:
        """Restore training state from a specific checkpoint directory."""
        ckpt_path = Path(checkpoint_path).resolve()
        logger.info(f"Restoring checkpoint from {ckpt_path}...")

        state_dir = (ckpt_path / "state").resolve() if (ckpt_path / "state").exists() else ckpt_path.resolve()

        try:
            if self._checkpointer is not None:
                try:
                    restored = self._checkpointer.restore(state_dir, target=target_state)
                except TypeError:
                    restored = self._checkpointer.restore(state_dir, item=target_state)
                logger.info("✓ Restored state using Orbax")
                return restored
            else:
                from flax.training import checkpoints
                restored = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_path), target=target_state)
                logger.info("✓ Restored state using Flax checkpoints")
                return restored
        except Exception as e:
            logger.exception(f"Failed to restore checkpoint from {ckpt_path}: {e}")
            raise

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest valid checkpoint directory in output_dir."""
        if not self.output_dir.exists():
            return None

        candidates = []
        for p in self.output_dir.glob("checkpoint-*"):
            if p.is_dir():
                try:
                    step_num = int(p.name.split("-")[-1])
                    candidates.append((step_num, p))
                except ValueError:
                    continue

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        # Check if the newest checkpoint has a valid state
        for _, ckpt_dir in candidates:
            if (ckpt_dir / "metadata.json").exists() or (ckpt_dir / "state").exists():
                return ckpt_dir

        return candidates[0][1]

    def _prune_old_checkpoints(self):
        """Prune older checkpoints exceeding max_to_keep."""
        if not self.is_primary:
            return

        candidates = []
        for p in self.output_dir.glob("checkpoint-*"):
            if p.is_dir():
                try:
                    step_num = int(p.name.split("-")[-1])
                    candidates.append((step_num, p))
                except ValueError:
                    continue

        if len(candidates) <= self.max_to_keep:
            return

        candidates.sort(key=lambda x: x[0])
        to_delete = candidates[: -self.max_to_keep]
        for _, path in to_delete:
            try:
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Pruned old checkpoint: {path}")
            except Exception as e:
                logger.warning(f"Could not prune {path}: {e}")

    def export_to_hf_safetensors(
        self,
        flax_params: Dict[str, Any],
        config: Qwen3_5Config,
        export_dir: Union[str, Path],
        tokenizer_name_or_path: Optional[str] = None,
    ):
        """Export Flax parameters to standard Hugging Face safetensors format."""
        if not self.is_primary:
            return

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        try:
            from safetensors.numpy import save_file
            pt_state_dict = convert_flax_to_pytorch_state_dict(flax_params, config)

            st_path = export_path / "model.safetensors"
            save_file(pt_state_dict, str(st_path))
            logger.info(f"✓ Saved Hugging Face safetensors to {st_path}")

            # Save config.json
            config_dict = {
                "architectures": ["Qwen3_5ForCausalLM"],
                "model_type": "qwen3_5",
                "text_config": {
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
                },
                "tie_word_embeddings": config.tie_word_embeddings,
            }
            with open(export_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)

            # Copy tokenizer files if available
            if tokenizer_name_or_path:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
                tokenizer.save_pretrained(str(export_path))
                logger.info("✓ Saved tokenizer files to export directory")

        except Exception as e:
            logger.exception(f"Failed to export HF checkpoint: {e}")
