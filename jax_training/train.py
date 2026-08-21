"""Main CLI entry point for JAX/Flax CPT Training on Kaggle TPU v5e-8.

Usage:
    python -m jax_training.train --config jax_training/config.yaml --data-dir /kaggle/working/data
    python -m jax_training.train --smoke-test
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
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
from jax_training.data.dataset import load_sharded_dataset, ArrowShardDataset


def setup_logging(log_dir: str = "logs", log_name: str = "jax_cpt_training"):
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


def create_synthetic_dataset(num_samples: int = 100, seq_len: int = 4096, vocab_size: int = 248320):
    """Create synthetic packed dataset in memory for smoke testing."""
    from datasets import Dataset

    rng = np.random.default_rng(42)
    input_ids = rng.integers(1, vocab_size, size=(num_samples, seq_len), dtype=np.int32)
    labels = input_ids.copy()
    return Dataset.from_dict({"input_ids": list(input_ids), "labels": list(labels)})


def main():
    parser = argparse.ArgumentParser(description="JAX/Flax CPT Training on Kaggle TPU v5e-8")
    parser.add_argument("--config", type=str, default="jax_training/config.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing Arrow/Parquet shards")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 5-step smoke test")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override maximum training steps")
    parser.add_argument("--export-hf", action="store_true", help="Export latest checkpoint to Hugging Face format")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load YAML config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found. Using default TrainConfig.")
        raw_config = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

    model_raw = raw_config.get("model", {})
    train_raw = raw_config.get("training", {})
    storage_raw = raw_config.get("storage", {})

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
        model_name_or_path=model_raw.get("name_or_path", "Qwen/Qwen3.5-0.8B-Base"),
        output_dir=train_raw.get("output_dir", "checkpoints/jax_cpt"),
        max_seq_length=train_raw.get("max_seq_length", 1024),
        dataset_packed_seq_length=train_raw.get("dataset_packed_seq_length", 4096),
        target_tokens=train_raw.get("target_tokens", 1_000_000_000),
        per_device_batch_size=train_raw.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_raw.get("gradient_accumulation_steps", 1),
        loss_chunk_size=train_raw.get("loss_chunk_size", 256),
        learning_rate=float(train_raw.get("learning_rate", 2e-5)),
        min_lr_ratio=float(train_raw.get("min_lr_ratio", 0.1)),
        warmup_ratio=float(train_raw.get("warmup_ratio", 0.02)),
        warmup_steps=train_raw.get("warmup_steps", None),
        weight_decay=float(train_raw.get("weight_decay", 0.01)),
        max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
        adam_beta1=float(train_raw.get("adam_beta1", 0.9)),
        adam_beta2=float(train_raw.get("adam_beta2", 0.95)),
        adam_epsilon=float(train_raw.get("adam_epsilon", 1e-8)),
        dtype=train_raw.get("dtype", "bfloat16"),
        seed=train_raw.get("seed", 42),
        max_steps=args.max_steps if args.max_steps > 0 else train_raw.get("max_steps", -1),
        logging_steps=train_raw.get("logging_steps", 50),
        save_steps=train_raw.get("save_steps", 2500),
        save_total_limit=train_raw.get("save_total_limit", 2),
        resume_from_checkpoint=args.resume or train_raw.get("resume_from_checkpoint", None),
        smoke_test=args.smoke_test,
    )

    data_dir = args.data_dir or storage_raw.get("data_dir", None)

    try:
        # Load dataset
        dataset = None
        if not args.smoke_test and data_dir:
            dataset = load_sharded_dataset(data_dir)

        if dataset is None:
            if args.smoke_test:
                logger.info("Generating synthetic packed dataset for smoke test...")
                dataset = create_synthetic_dataset(num_samples=50, seq_len=4096, vocab_size=model_config.vocab_size)
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

        # Initialize trainer and train
        trainer = JAXTrainer(config=train_config, model_config=model_config)
        trainer.train(dataset)

    except Exception as e:
        process_idx = jax.process_index()
        failure_file = project_root / "logs" / f"jax_cpt_rank_{process_idx}_failure.log"
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- JAX Rank {process_idx} Failure ---\n")
            f.write(traceback.format_exc())
        logger.exception(f"Training failed on rank {process_idx}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
