"""CPT and SFT Training Pipeline.

Main training loop built on HuggingFace Trainer with:
- Full-parameter training (no LoRA for CPT)
- Token-count-based stopping
- Cosine LR schedule with warmup
- Gradient checkpointing
- Checkpoint resumption
- Multi-environment support (Colab, Kaggle, local)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from src.training.utils import (
    detect_hardware,
    auto_configure_batch_size,
    load_model_for_training,
)
from src.training.callbacks import (
    TokenCountCallback,
    CheckpointUploadCallback,
    DetailedLoggingCallback,
)
from src.utils.config import load_config, detect_environment

logger = logging.getLogger(__name__)


class CPTTrainer:
    """Continued Pre-Training (CPT) trainer.

    Wraps HuggingFace Trainer with CPT-specific configuration,
    token-count-based training, and environment-aware setup.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CPT trainer.

        Args:
            config: Training configuration. If None, loads from
                   configs/cpt_config.yaml.
        """
        self.config = config or load_config("cpt_config.yaml")
        self.env = detect_environment()
        self.hardware = detect_hardware()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self) -> None:
        """Set up model, tokenizer, and training arguments."""
        model_cfg = self.config["model"]
        train_cfg = self.config["training"]

        # Auto-configure batch size if GPU detected
        if self.hardware["gpu_memory_gb"]:
            micro_batch, grad_accum = auto_configure_batch_size(
                gpu_memory_gb=self.hardware["gpu_memory_gb"],
                seq_length=train_cfg["max_seq_length"],
            )
            # Use auto-configured values unless explicitly set in config
            if self.env != "local":
                train_cfg["per_device_train_batch_size"] = micro_batch
                train_cfg["gradient_accumulation_steps"] = grad_accum

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_for_training(
            model_name_or_path=model_cfg["name_or_path"],
            dtype=model_cfg.get("dtype", "bfloat16"),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            strip_vision=True,
        )

        logger.info(f"Setup complete in {self.env} environment")

    def _build_training_args(self) -> TrainingArguments:
        """Build TrainingArguments from config."""
        train_cfg = self.config["training"]

        # Compute max_steps from target tokens if not explicitly set
        max_steps = train_cfg.get("max_steps", -1)
        if max_steps == -1:
            seq_len = train_cfg["max_seq_length"]
            batch_size = train_cfg["per_device_train_batch_size"]
            grad_accum = train_cfg["gradient_accumulation_steps"]
            n_gpus = max(self.hardware.get("gpu_count", 1), 1)
            tokens_per_step = seq_len * batch_size * grad_accum * n_gpus
            target_tokens = train_cfg.get("target_tokens", 1_000_000_000)
            max_steps = target_tokens // tokens_per_step
            logger.info(
                f"Computed max_steps={max_steps:,} from "
                f"target_tokens={target_tokens:,}, "
                f"tokens_per_step={tokens_per_step:,}"
            )

        args = TrainingArguments(
            output_dir=train_cfg["output_dir"],
            max_steps=max_steps,
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg.get(
                "per_device_eval_batch_size", 2
            ),
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.02),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            bf16=train_cfg.get("bf16", True),
            fp16=train_cfg.get("fp16", False),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            optim=train_cfg.get("optim", "adamw_torch"),
            adam_beta1=train_cfg.get("adam_beta1", 0.9),
            adam_beta2=train_cfg.get("adam_beta2", 0.95),
            adam_epsilon=train_cfg.get("adam_epsilon", 1e-8),
            logging_steps=train_cfg.get("logging_steps", 10),
            logging_first_step=train_cfg.get("logging_first_step", True),
            save_strategy=train_cfg.get("save_strategy", "steps"),
            save_steps=train_cfg.get("save_steps", 200),
            save_total_limit=train_cfg.get("save_total_limit", 5),
            eval_strategy=train_cfg.get("eval_strategy", "steps"),
            eval_steps=train_cfg.get("eval_steps", 200),
            dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
            dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
            seed=train_cfg.get("seed", 42),
            data_seed=train_cfg.get("data_seed", 42),
            report_to=train_cfg.get("report_to", "none"),
            run_name=train_cfg.get("run_name", "qwen35-0.8b-cpt"),
            resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"),
            remove_unused_columns=False,
        )

        return args

    def _build_callbacks(self) -> List:
        """Build training callbacks."""
        train_cfg = self.config["training"]
        storage_cfg = self.config.get("storage", {})

        seq_len = train_cfg["max_seq_length"]
        batch_size = train_cfg["per_device_train_batch_size"]
        grad_accum = train_cfg["gradient_accumulation_steps"]
        n_gpus = max(self.hardware.get("gpu_count", 1), 1)
        effective_batch = batch_size * grad_accum * n_gpus

        callbacks = [
            TokenCountCallback(
                target_tokens=train_cfg.get("target_tokens", 1_000_000_000),
                seq_length=seq_len,
                effective_batch_size=effective_batch,
            ),
            DetailedLoggingCallback(),
        ]

        # Checkpoint upload
        hf_cfg = storage_cfg.get("hf_hub", {})
        gdrive_cfg = storage_cfg.get("gdrive", {})

        if hf_cfg.get("push_checkpoints") and hf_cfg.get("repo_id"):
            callbacks.append(
                CheckpointUploadCallback(
                    hf_repo_id=hf_cfg["repo_id"],
                )
            )

        if self.env == "colab" and gdrive_cfg.get("checkpoint_dir"):
            mount_path = gdrive_cfg.get("mount_path", "/content/drive")
            callbacks.append(
                CheckpointUploadCallback(
                    gdrive_dir=os.path.join(
                        mount_path, gdrive_cfg["checkpoint_dir"]
                    ),
                )
            )

        return callbacks

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """Run CPT training.

        Args:
            train_dataset: Tokenized and packed training dataset.
            eval_dataset: Optional evaluation dataset.
        """
        if self.model is None:
            self.setup()

        training_args = self._build_training_args()
        callbacks = self._build_callbacks()

        # Data collator (simple — data is already tokenized and packed)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Resume from checkpoint if specified
        resume_path = self.config["training"].get("resume_from_checkpoint")
        if resume_path and Path(resume_path).exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")

        logger.info("Starting CPT training...")
        self.trainer.train(resume_from_checkpoint=resume_path)

        # Save final model
        final_dir = os.path.join(training_args.output_dir, "final")
        self.trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")

    def save_model(self, output_dir: str) -> None:
        """Save model and tokenizer to a directory.

        Args:
            output_dir: Directory to save to.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup() first.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
