"""Main CLI entry point for JAX/Flax CPT and SFT Training on Kaggle TPU v5e-8.

Usage:
    # Continued Pre-Training (CPT)
    python -m jax_training.train --mode cpt --config jax_training/config.yaml --data-dir /kaggle/working/data
    
    # Supervised Fine-Tuning (SFT)
    python -m jax_training.train --mode sft --config configs/sft_config.yaml --data-dir /kaggle/working/data
    
    # Smoke Test (5-step verification)
    python -m jax_training.train --mode sft --smoke-test
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import numpy as np

# Ensure UTF-8 output
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import jax
from jax_training.models.config import Qwen3_5Config
from jax_training.training.trainer import JAXTrainer, TrainConfig
from jax_training.data.dataset import load_sharded_dataset, split_dataset, ArrowShardDataset


def setup_logging(log_dir: str = "logs", log_name: str = "jax_training"):
    """Set up structured logging."""
    is_primary = jax.process_index() == 0
    handlers = [logging.StreamHandler(sys.stdout)] if is_primary else []

    if is_primary:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{log_name}.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def create_synthetic_dataset(num_samples: int = 100, seq_len: int = 4096, vocab_size: int = 248320, is_sft: bool = False):
    """Create synthetic packed dataset in memory for smoke testing."""
    from datasets import Dataset

    rng = np.random.default_rng(42)
    input_ids = rng.integers(1, vocab_size, size=(num_samples, seq_len), dtype=np.int32)
    labels = input_ids.copy()
    
    if is_sft:
        # Simulate assistant-only loss masking (~50% of tokens masked to -100)
        mask = rng.random(size=(num_samples, seq_len)) < 0.5
        labels[mask] = -100

    attention_mask = np.ones((num_samples, seq_len), dtype=np.int8)
    return Dataset.from_dict({
        "input_ids": list(input_ids),
        "labels": list(labels),
        "attention_mask": list(attention_mask),
    })


def _find_local_model_path() -> Optional[str]:
    """Auto-detect local CPT model weights in /kaggle/input/ or local directories."""
    if Path("/kaggle/input").exists():
        candidates = list(Path("/kaggle/input").rglob("model.safetensors"))
        for m in candidates:
            parent = m.parent
            if (parent / "config.json").exists():
                return str(parent)
    return None


def main():
    parser = argparse.ArgumentParser(description="JAX/Flax CPT and SFT Training on Kaggle TPU / GPU")
    parser.add_argument("--mode", type=str, choices=["cpt", "sft"], default=None, help="Training mode ('cpt' or 'sft')")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--model-path", type=str, default=None, help="Local directory containing model weights")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing Arrow/Parquet shards")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 5-step smoke test")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override maximum training steps")
    parser.add_argument("--export-hf", action="store_true", help="Export latest checkpoint to Hugging Face format")
    args = parser.parse_args()

    # Determine default config and mode
    if args.config is not None:
        config_path = Path(args.config)
        mode = args.mode or ("sft" if "sft" in config_path.name.lower() else "cpt")
    elif args.mode == "sft":
        config_path = Path("configs/sft_config.yaml")
        mode = "sft"
    else:
        config_path = Path("jax_training/config.yaml")
        mode = "cpt"

    setup_logging(log_name=f"jax_{mode}_training")
    logger = logging.getLogger(__name__)

    # Load YAML config
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using default TrainConfig.")
        raw_config = {}
    else:
        logger.info(f"Loading configuration from {config_path} [mode={mode}]")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

    model_raw = raw_config.get("model", {})
    train_raw = raw_config.get("training", {})
    storage_raw = raw_config.get("storage", {})

    default_model = "kaptaan45/QaptaanLM-0.75B" if mode == "sft" else "Qwen/Qwen3.5-0.8B-Base"
    default_output_dir = f"checkpoints/jax_{mode}"
    default_tokens = 100_000_000 if mode == "sft" else 1_000_000_000
    default_lr = 5e-6 if mode == "sft" else 2e-5
    default_min_lr_ratio = 0.0 if mode == "sft" else 0.1
    default_warmup_ratio = 0.03 if mode == "sft" else 0.02

    # Resolve model path (CLI > auto-detected local > config > default)
    resolved_model_path = args.model_path or model_raw.get("name_or_path")
    if not resolved_model_path or not Path(resolved_model_path).exists():
        detected_path = _find_local_model_path()
        if detected_path:
            logger.info(f"Auto-detected local CPT model weights at: {detected_path}")
            resolved_model_path = detected_path
        else:
            resolved_model_path = resolved_model_path or default_model

    model_config = Qwen3_5Config(
        vocab_size=model_raw.get("vocab_size", 248320),
        hidden_size=model_raw.get("hidden_size", 1024),
        intermediate_size=model_raw.get("intermediate_size", 3584),
        num_hidden_layers=model_raw.get("num_hidden_layers", 24),
        num_attention_heads=model_raw.get("num_attention_heads", 8),
        num_key_value_heads=model_raw.get("num_key_value_heads", 2),
        head_dim=model_raw.get("head_dim", 256),
        rms_norm_eps=model_raw.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=model_raw.get("tie_word_embeddings", True),
        full_attention_interval=model_raw.get("full_attention_interval", 4),
        linear_conv_kernel_dim=model_raw.get("linear_conv_kernel_dim", 4),
        linear_num_key_heads=model_raw.get("linear_num_key_heads", 16),
        linear_num_value_heads=model_raw.get("linear_num_value_heads", 16),
        linear_key_head_dim=model_raw.get("linear_key_head_dim", 128),
        linear_value_head_dim=model_raw.get("linear_value_head_dim", 128),
        dtype=model_raw.get("dtype", "bfloat16"),
    )

    train_config = TrainConfig(
        model_name_or_path=resolved_model_path,
        output_dir=train_raw.get("output_dir", default_output_dir),
        mode=mode,
        max_seq_length=train_raw.get("max_seq_length", 1024),
        dataset_packed_seq_length=train_raw.get("dataset_packed_seq_length", 4096),
        target_tokens=train_raw.get("target_tokens", default_tokens),
        per_device_batch_size=train_raw.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_raw.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_raw.get("gradient_accumulation_steps", 1),
        loss_chunk_size=train_raw.get("loss_chunk_size", 256),
        learning_rate=float(train_raw.get("learning_rate", default_lr)),
        min_lr_ratio=float(train_raw.get("min_lr_ratio", default_min_lr_ratio)),
        warmup_ratio=float(train_raw.get("warmup_ratio", default_warmup_ratio)),
        warmup_steps=train_raw.get("warmup_steps", None),
        weight_decay=float(train_raw.get("weight_decay", 0.01)),
        max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
        adam_beta1=float(train_raw.get("adam_beta1", 0.9)),
        adam_beta2=float(train_raw.get("adam_beta2", 0.95)),
        adam_epsilon=float(train_raw.get("adam_epsilon", 1e-8)),
        dtype=train_raw.get("dtype", "bfloat16"),
        seed=train_raw.get("seed", 42),
        eval_split_ratio=float(train_raw.get("eval_split_ratio", 0.015 if mode == "sft" else 0.0)),
        eval_steps=int(train_raw.get("eval_steps", 250)),
        max_steps=args.max_steps if args.max_steps > 0 else train_raw.get("max_steps", -1),
        logging_steps=train_raw.get("logging_steps", 25 if mode == "sft" else 50),
        save_steps=train_raw.get("save_steps", 500 if mode == "sft" else 2500),
        save_total_limit=train_raw.get("save_total_limit", 2),
        resume_from_checkpoint=args.resume or train_raw.get("resume_from_checkpoint", None),
        smoke_test=args.smoke_test,
    )

    data_dir = args.data_dir or storage_raw.get("data_dir", None)

    try:
        # Load dataset
        train_dataset = None
        eval_dataset = None

        if not args.smoke_test and data_dir:
            full_dataset = load_sharded_dataset(data_dir)
            if full_dataset is not None:
                if train_config.eval_split_ratio > 0:
                    train_dataset, eval_dataset = split_dataset(
                        full_dataset,
                        eval_ratio=train_config.eval_split_ratio,
                        seed=train_config.seed,
                    )
                    logger.info(
                        f"Split dataset ({len(full_dataset):,} packed records): "
                        f"Train={len(train_dataset):,} records, Val={len(eval_dataset):,} records "
                        f"({train_config.eval_split_ratio:.1%})"
                    )
                else:
                    train_dataset = full_dataset

        if train_dataset is None:
            if args.smoke_test:
                logger.info(f"Generating synthetic packed dataset for [{mode.upper()}] smoke test...")
                synthetic_full = create_synthetic_dataset(
                    num_samples=50,
                    seq_len=4096,
                    vocab_size=model_config.vocab_size,
                    is_sft=(mode == "sft"),
                )
                if train_config.eval_split_ratio > 0:
                    train_dataset, eval_dataset = split_dataset(synthetic_full, eval_ratio=0.2, seed=42)
                else:
                    train_dataset = synthetic_full
            else:
                logger.warning(
                    f"No dataset found at {data_dir}. Falling back to synthetic dataset. "
                    "Provide --data-dir <path> pointing to processed Arrow/Parquet shards."
                )

        if args.smoke_test and jax.devices()[0].platform == "cpu":
            logger.info("Local CPU smoke-test detected: using 2 decoder layers, seq_len=256 for instant verification.")
            model_config.num_hidden_layers = 2
            model_config.layer_types = ["linear_attention", "full_attention"]
            train_config.model_name_or_path = ""  # Initialize randomly for 2-layer smoke test
            train_config.max_seq_length = 256
            train_config.per_device_batch_size = 1
            train_config.per_device_eval_batch_size = 1

        # Initialize trainer and train
        trainer = JAXTrainer(config=train_config, model_config=model_config)
        final_state = trainer.train(train_dataset=train_dataset, eval_dataset=eval_dataset)

        # Export to Hugging Face safetensors format if requested and training completed
        should_export = args.export_hf or storage_raw.get("export_hf", True)
        if should_export and jax.process_index() == 0 and final_state is not None:
            default_export = f"checkpoints/jax_{mode}_hf"
            hf_export_dir = storage_raw.get("hf_export_dir", default_export)
            logger.info(f"Exporting final {mode.upper()} model to Hugging Face format at {hf_export_dir}...")
            trainer.checkpoint_manager.export_to_hf_safetensors(
                flax_params=final_state.params,
                config=model_config,
                export_dir=hf_export_dir,
                tokenizer_name_or_path=train_config.model_name_or_path or "Qwen/Qwen3.5-0.8B-Base",
            )

    except Exception as e:
        process_idx = jax.process_index()
        log_dir = Path("/kaggle/working/logs") if Path("/kaggle/working").exists() else (project_root / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        failure_file = log_dir / f"jax_{mode}_rank_{process_idx}_failure.log"
        with open(failure_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- JAX Rank {process_idx} Failure ---\n")
            f.write(traceback.format_exc())
        logger.exception(f"Training failed on rank {process_idx}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
