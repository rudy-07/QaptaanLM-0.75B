"""CPT Training Launch Script.

Loads processed dataset shards (or generates small local batch)
and executes Continued Pre-Training on Qwen3.5-0.8B.

Usage:
    python scripts/05_train_cpt.py
    python scripts/05_train_cpt.py --config configs/cpt_config.yaml --data-dir data/processed
"""

import argparse
import logging
import os
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset, Dataset
from src.training.trainer import CPTTrainer
from src.utils.config import load_config, detect_environment
from src.utils.logging_utils import setup_logging

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Launch CPT Training")
    parser.add_argument("--config", type=str, default="cpt_config.yaml", help="Path to training config")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing processed Arrow/Parquet shards")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    env = detect_environment()

    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume

    # Set up logging
    log_dir = str(project_root / "logs")
    setup_logging(log_dir=log_dir, log_name="cpt_training")

    logger.info("=" * 60)
    logger.info(f"Starting Qwen3.5-0.8B CPT Training in [{env}] environment")
    logger.info("=" * 60)

    # Determine dataset path
    data_dir = args.data_dir or config["storage"]["processed_data"]["path"]
    data_path = Path(data_dir)

    if data_path.exists() and (any(data_path.glob("shard_*.arrow")) or any(data_path.glob("shard_*.parquet"))):
        logger.info(f"Discovering processed shards from {data_path}...")
        arrow_files = sorted(list(data_path.glob("shard_*.arrow")))
        parquet_files = sorted(list(data_path.glob("shard_*.parquet")))

        files_to_load = [str(f) for f in (arrow_files or parquet_files)]
        file_type = "arrow" if arrow_files else "parquet"
        logger.info(f"Found {len(files_to_load)} {file_type} shard files. Loading...")

        for f in tqdm(files_to_load, desc="Validating data shards", unit="shard"):
            pass

        train_dataset = load_dataset(file_type, data_files=files_to_load, split="train")
        logger.info(f"✓ Loaded {len(train_dataset):,} packed training sequences.")
    else:
        logger.warning(
            f"No processed shards found in {data_path}. "
            "Please run scripts/03_process_data.py first to generate training shards."
        )
        return

    # Initialize trainer
    trainer = CPTTrainer(config)
    trainer.setup()

    # Launch training
    trainer.train(train_dataset=train_dataset)
    logger.info("CPT Training successfully completed!")


if __name__ == "__main__":
    main()
