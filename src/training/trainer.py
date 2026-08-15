"""CPT and SFT Training Pipeline.

Main training loop built on HuggingFace Trainer with:
- Full-parameter training (no LoRA for CPT)
- Token-count-based stopping
- Cosine LR schedule with warmup
- Gradient checkpointing
- Checkpoint resumption
- Multi-environment support (Colab, Kaggle, local)
- Liger Kernel integration (fused CE/RMSNorm/SwiGLU/RoPE)
- PyTorch SDPA attention backend
- torch.compile support
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
from src.training.liger_integration import (
    apply_liger_kernel_patches,
    is_liger_available,
    estimate_vram_savings,
)
from src.utils.config import load_config, detect_environment

logger = logging.getLogger(__name__)


class CPTTrainer:
    """Continued Pre-Training (CPT) trainer.

    Wraps HuggingFace Trainer with CPT-specific configuration,
    token-count-based training, and environment-aware setup.

    Speed Optimizations (automatically applied when available):
    - Liger Kernel: Fuses cross-entropy, RMSNorm, SwiGLU, RoPE → 40-60% VRAM savings
    - SDPA attention: Memory-efficient attention backend (works on T4, unlike FA2)
    - torch.compile: Kernel fusion and reduced CPU overhead → 1.3-1.5x speedup
    - 8-bit AdamW: Saves ~4.5GB VRAM on memory-constrained GPUs
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
        self.liger_enabled = False

    def setup(self) -> None:
        """Set up model, tokenizer, and training arguments."""
        model_cfg = self.config["model"]
        train_cfg = self.config["training"]

        is_tpu = bool(self.hardware.get("tpu_available", False))

        # ── Step 1: Apply Liger Kernel patches (CUDA GPUs only) ──
        if is_tpu:
            logger.info("TPU detected: skipping Liger Kernel (Triton is CUDA-only; TPU XLA handles graph fusion natively).")
            self.liger_enabled = False
        elif train_cfg.get("use_liger_kernel", True):
            self.liger_enabled = apply_liger_kernel_patches(model_type="qwen3")
            if self.liger_enabled:
                # Log estimated VRAM savings
                estimate_vram_savings(
                    seq_length=train_cfg.get("max_seq_length", 2048),
                    micro_batch=train_cfg.get("per_device_train_batch_size", 1),
                )
        else:
            logger.info("Liger Kernel disabled via config (use_liger_kernel: false)")

        # ── Step 2: Auto-configure batch size ──
        if is_tpu:
            # TPU v5e-8 has 8 cores x 16GB HBM = 128GB total memory
            train_cfg["per_device_train_batch_size"] = train_cfg.get("per_device_train_batch_size", 2)
            train_cfg["gradient_accumulation_steps"] = train_cfg.get("gradient_accumulation_steps", 2)
            train_cfg["max_seq_length"] = train_cfg.get("max_seq_length", 2048)
            logger.info(
                f"Auto-configured for TPU v5e-8: micro_batch={train_cfg['per_device_train_batch_size']} per core "
                f"x 8 cores = {8 * train_cfg['per_device_train_batch_size']} sequences/step "
                f"({8 * train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps'] * train_cfg['max_seq_length']:,} tokens/step)"
            )
        elif self.hardware["gpu_memory_gb"]:
            n_gpus = max(self.hardware.get("gpu_count", 1), 1)
            micro_batch, grad_accum, safe_seq_length = auto_configure_batch_size(
                gpu_memory_gb=self.hardware["gpu_memory_gb"],
                seq_length=train_cfg["max_seq_length"],
                gpu_count=n_gpus,
                liger_enabled=self.liger_enabled,
            )
            # Use auto-configured values unless explicitly set in config
            if self.env != "local":
                train_cfg["per_device_train_batch_size"] = micro_batch
                train_cfg["gradient_accumulation_steps"] = grad_accum
                train_cfg["max_seq_length"] = safe_seq_length

        # ── Step 3: Auto-configure precision ──
        if is_tpu:
            logger.info("TPU detected: enabling native hardware bfloat16 precision.")
            train_cfg["bf16"] = True
            train_cfg["fp16"] = False
            model_dtype = "bfloat16"
        else:
            # Tesla T4/V100/P100 lack hardware BF16 Tensor Cores and are 5-10x slower on bf16.
            # A100/H100/L4 have native BF16 Tensor Cores.
            has_bf16 = bool(self.hardware.get("bf16_support", False))
            if not has_bf16:
                logger.info("GPU lacks native BF16 Tensor Cores (e.g. Tesla T4). Auto-switching to FP16 AMP (master weights in FP32 + FP16 autocast) for 5-10x faster training.")
                train_cfg["bf16"] = False
                train_cfg["fp16"] = True
                model_dtype = "float32"
            else:
                model_dtype = model_cfg.get("dtype", "bfloat16")
                if model_dtype not in ("bfloat16", "float32"):
                    model_dtype = "bfloat16"

            # In PyTorch AMP with fp16=True, master model weights must be float32 for GradScaler to unscale gradients
            if train_cfg.get("fp16"):
                model_dtype = "float32"

        # Use 0 dataloader workers in Colab/Kaggle containers to prevent shared-memory / CPU RAM exhaustion
        if self.env in ("kaggle", "colab"):
            train_cfg["dataloader_num_workers"] = train_cfg.get("dataloader_num_workers", 0)
        else:
            train_cfg["dataloader_num_workers"] = train_cfg.get("dataloader_num_workers", 0)

        # ── Step 4: Load model and tokenizer (with SDPA attention) ──
        use_sdpa = train_cfg.get("use_sdpa", True) and not is_tpu
        self.model, self.tokenizer = load_model_for_training(
            model_name_or_path=model_cfg["name_or_path"],
            dtype=model_dtype,
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            strip_vision=True,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            use_sdpa=use_sdpa,
        )

        logger.info(f"Setup complete in {self.env} environment (dtype={model_dtype})")

    def _build_training_args(self, eval_dataset: Optional[Dataset] = None) -> TrainingArguments:
        """Build TrainingArguments dynamically and safely across all transformers versions."""
        import inspect

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

        # Calculate warmup steps
        warmup_ratio = train_cfg.get("warmup_ratio", 0.02)
        warmup_steps = train_cfg.get("warmup_steps")
        if warmup_steps is None and warmup_ratio is not None:
            if max_steps and max_steps > 0:
                warmup_steps = max(1, int(max_steps * warmup_ratio))
            else:
                warmup_steps = 100

        # Determine optimizer: on TPU use standard adamw_torch; on GPU <=16GB prefer 8-bit AdamW
        is_tpu = bool(self.hardware.get("tpu_available", False))
        optim_name = train_cfg.get("optim", "adamw_torch")
        if is_tpu:
            optim_name = "adamw_torch"
        elif optim_name in ("paged_adamw_8bit", "adamw_bnb_8bit") or (
            self.hardware.get("gpu_memory_gb") and self.hardware["gpu_memory_gb"] <= 16.5
        ):
            try:
                import bitsandbytes  # noqa: F401
                optim_name = "paged_adamw_8bit"
                logger.info("Using 'paged_adamw_8bit' optimizer to optimize VRAM on 16GB GPU (saves ~4.5GB VRAM)")
            except ImportError:
                optim_name = "adamw_torch"
                if self.hardware.get("gpu_memory_gb") and self.hardware["gpu_memory_gb"] <= 16.5:
                    logger.warning(
                        "bitsandbytes is not installed. Using standard 'adamw_torch'. "
                        "If you experience CUDA OOM during backward pass at seq_len=4096, "
                        "install bitsandbytes (`pip install bitsandbytes`) to use 8-bit optimizer."
                    )

        candidate_kwargs = {
            "output_dir": train_cfg["output_dir"],
            "max_steps": max_steps,
            "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
            "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 2),
            "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
            "learning_rate": train_cfg["learning_rate"],
            "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
            "weight_decay": train_cfg.get("weight_decay", 0.01),
            "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
            "bf16": train_cfg.get("bf16", False),
            "fp16": train_cfg.get("fp16", True),
            "gradient_checkpointing": train_cfg.get("gradient_checkpointing", True),
            "optim": optim_name,
            "adam_beta1": train_cfg.get("adam_beta1", 0.9),
            "adam_beta2": train_cfg.get("adam_beta2", 0.95),
            "adam_epsilon": train_cfg.get("adam_epsilon", 1e-8),
            "logging_steps": train_cfg.get("logging_steps", 10),
            "logging_first_step": train_cfg.get("logging_first_step", True),
            "save_strategy": train_cfg.get("save_strategy", "steps"),
            "save_steps": train_cfg.get("save_steps", 200),
            "save_total_limit": train_cfg.get("save_total_limit", 5),
            "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 2),
            "dataloader_pin_memory": False if is_tpu else train_cfg.get("dataloader_pin_memory", True),
            "seed": train_cfg.get("seed", 42),
            "data_seed": train_cfg.get("data_seed", 42),
            "report_to": train_cfg.get("report_to", "none"),
            "run_name": train_cfg.get("run_name", "qwen35-0.8b-cpt"),
            "remove_unused_columns": False,
        }

        # Inspect parameters supported by installed transformers version
        sig = inspect.signature(TrainingArguments.__init__)
        valid_params = set(sig.parameters.keys())

        # Set warmup (warmup_steps is universal; warmup_ratio if available)
        if "warmup_steps" in valid_params:
            candidate_kwargs["warmup_steps"] = warmup_steps
        elif "warmup_ratio" in valid_params:
            candidate_kwargs["warmup_ratio"] = warmup_ratio

        # Handle eval strategy parameter name difference (eval_strategy vs evaluation_strategy)
        if eval_dataset is None:
            eval_strat = "no"
        else:
            eval_strat = train_cfg.get("eval_strategy") or train_cfg.get("evaluation_strategy", "steps")

        eval_st = train_cfg.get("eval_steps", 200)
        if "eval_strategy" in valid_params:
            candidate_kwargs["eval_strategy"] = eval_strat
            candidate_kwargs["eval_steps"] = eval_st
        # Filter strictly to valid TrainingArguments parameters to avoid any TypeError
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_params and v is not None}
        return TrainingArguments(**filtered_kwargs)

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

        train_cfg = self.config["training"]
        seq_len = train_cfg["max_seq_length"]

        training_args = self._build_training_args(eval_dataset=eval_dataset)
        callbacks = self._build_callbacks()

        # Zero-copy dynamic slicing collator: slices sequences to seq_len on the fly
        # (Zero startup waiting time, instantly ready)
        def data_collator(features):
            batch = {}
            for k in ("input_ids", "attention_mask"):
                if k in features[0]:
                    tensors = [torch.as_tensor(f[k][:seq_len], dtype=torch.long) for f in features]
                    batch[k] = torch.stack(tensors)
            if "labels" in features[0]:
                tensors = [torch.as_tensor(f["labels"][:seq_len], dtype=torch.long) for f in features]
                batch["labels"] = torch.stack(tensors)
            elif "input_ids" in batch:
                batch["labels"] = batch["input_ids"].clone()
            return batch

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

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log optimization summary
        optimizations = []
        if self.liger_enabled:
            optimizations.append("Liger Kernel (FusedCE+RMSNorm+SwiGLU+RoPE)")
        if train_cfg.get("use_sdpa", True) and not bool(self.hardware.get("tpu_available", False)):
            optimizations.append("SDPA Attention")
        if bool(self.hardware.get("tpu_available", False)):
            optimizations.append("TPU v5e Native BF16")
        if "8bit" in str(training_args.optim):
            optimizations.append("8-bit AdamW")
        optimizations.append(f"seq_len={seq_len}")
        optimizations.append(f"micro_batch={training_args.per_device_train_batch_size}")

        logger.info(
            f"Starting CPT training with optimizations: {', '.join(optimizations)}\n"
            f"  Total steps: {training_args.max_steps:,}\n"
            f"  Tokens/step: {seq_len * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * max(self.hardware.get('gpu_count', 1), 1):,}"
        )

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
