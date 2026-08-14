"""Custom training callbacks for CPT/SFT.

Provides callbacks for:
- Periodic checkpoint upload to HF Hub / GDrive
- Token-count-based logging and stopping
- Mixture distribution monitoring
- Learning rate logging
"""

import logging
import time
from typing import Any, Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class TokenCountCallback(TrainerCallback):
    """Track and log total tokens processed during training.

    Useful because we train for a target number of tokens,
    not epochs. Logs estimated tokens based on steps and batch config.
    """

    def __init__(
        self,
        target_tokens: int = 1_000_000_000,
        seq_length: int = 4096,
        effective_batch_size: int = 32,
    ):
        """Initialize the callback.

        Args:
            target_tokens: Total target tokens for training.
            seq_length: Sequence length.
            effective_batch_size: micro_batch * grad_accum * num_gpus.
        """
        self.target_tokens = target_tokens
        self.tokens_per_step = seq_length * effective_batch_size
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        total_steps = self.target_tokens // self.tokens_per_step
        if state.is_world_process_zero:
            logger.info(
                f"Training started. Target: {self.target_tokens:,} tokens, "
                f"~{total_steps:,} steps ({self.tokens_per_step:,} tokens/step)"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step > 0:
            tokens_so_far = state.global_step * self.tokens_per_step
            progress = tokens_so_far / self.target_tokens
            elapsed = time.time() - self.start_time if self.start_time else 0
            tokens_per_sec = tokens_so_far / max(elapsed, 1)

            remaining_tokens = self.target_tokens - tokens_so_far
            eta_sec = remaining_tokens / max(tokens_per_sec, 1)
            eta_hrs = eta_sec / 3600

            if logs is not None:
                logs["tokens_processed"] = tokens_so_far
                logs["token_progress"] = f"{progress:.1%}"
                logs["tokens_per_second"] = f"{tokens_per_sec:,.0f}"
                logs["eta_hours"] = f"{eta_hrs:.1f}"

    def on_step_end(self, args, state, control, **kwargs):
        tokens_so_far = state.global_step * self.tokens_per_step
        if tokens_so_far >= self.target_tokens:
            logger.info(
                f"Target tokens reached ({tokens_so_far:,} >= "
                f"{self.target_tokens:,}). Stopping training."
            )
            control.should_training_stop = True
        return control


class CheckpointUploadCallback(TrainerCallback):
    """Upload checkpoints to HF Hub or GDrive after saving."""

    def __init__(
        self,
        hf_repo_id: Optional[str] = None,
        gdrive_dir: Optional[str] = None,
        upload_every_n_saves: int = 1,
    ):
        """Initialize checkpoint upload callback.

        Args:
            hf_repo_id: HF Hub repo for upload. None to skip.
            gdrive_dir: GDrive path for backup. None to skip.
            upload_every_n_saves: Upload every N saves (skip some for speed).
        """
        self.hf_repo_id = hf_repo_id
        self.gdrive_dir = gdrive_dir
        self.upload_every_n_saves = upload_every_n_saves
        self._save_count = 0

    def on_save(self, args, state, control, **kwargs):
        # In DDP multi-GPU, only rank 0 handles saving / uploading
        if not state.is_world_process_zero:
            return

        self._save_count += 1

        if self._save_count % self.upload_every_n_saves != 0:
            return

        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"

        # Upload to HF Hub
        if self.hf_repo_id:
            try:
                from src.utils.storage import upload_to_hub

                upload_to_hub(
                    checkpoint_dir,
                    self.hf_repo_id,
                    commit_message=f"Checkpoint step {state.global_step}",
                )
            except Exception as e:
                logger.error(f"HF Hub upload failed: {e}")

        # Copy to GDrive
        if self.gdrive_dir:
            try:
                from src.utils.storage import copy_checkpoint

                copy_checkpoint(checkpoint_dir, self.gdrive_dir)
            except Exception as e:
                logger.error(f"GDrive backup failed: {e}")


class DetailedLoggingCallback(TrainerCallback):
    """Enhanced logging with per-step details."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        # Add learning rate to logs if available
        if "learning_rate" in logs:
            loss_val = logs.get("loss")
            loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val or "N/A")
            lr_val = logs.get("learning_rate")
            lr_str = f"{lr_val:.2e}" if isinstance(lr_val, (int, float)) else str(lr_val or "0")
            logger.info(
                f"Step {state.global_step}: "
                f"loss={loss_str}, "
                f"lr={lr_str}, "
                f"tokens={logs.get('tokens_processed', 'N/A')}, "
                f"progress={logs.get('token_progress', 'N/A')}, "
                f"ETA={logs.get('eta_hours', 'N/A')}h"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            logger.info(f"Evaluation at step {state.global_step}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
