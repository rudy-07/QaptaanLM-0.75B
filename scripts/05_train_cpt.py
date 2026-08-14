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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Prevent broken torchvision binary mismatch crashes in Colab/Kaggle
try:
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    _iu.is_torchvision_available = lambda: False
except Exception:
    pass

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

    # Determine dataset path or Hugging Face Hub repo ID
    data_dir = args.data_dir or config["storage"]["processed_data"]["path"]
    
    train_dataset = None
    data_path = Path(data_dir) if data_dir else None

    # If the specified path does not exist, look across all /kaggle/input/ mounts
    if (not data_path or not data_path.exists()) and os.path.exists("/kaggle/input"):
        kaggle_root = Path("/kaggle/input")
        # Check if there are any arrow or parquet files anywhere under /kaggle/input
        kaggle_shards = list(kaggle_root.rglob("*.arrow")) or list(kaggle_root.rglob("*.parquet"))
        if kaggle_shards:
            # Pick the deepest common parent or the parent directory of shards
            data_path = kaggle_shards[0].parent
            logger.info(f"Auto-detected Kaggle dataset shards at: {data_path}")

    if data_path and data_path.exists():
        # 1. Try loading arrow/parquet files recursively
        arrow_files = sorted([str(f) for f in data_path.rglob("*.arrow") if not f.name.startswith(".")])
        parquet_files = sorted([str(f) for f in data_path.rglob("*.parquet") if not f.name.startswith(".")])

        if arrow_files or parquet_files:
            file_type = "arrow" if arrow_files else "parquet"
            files_to_load = arrow_files if arrow_files else parquet_files
            logger.info(f"Found {len(files_to_load)} {file_type} shard files in {data_path}. Loading...")
            train_dataset = load_dataset(file_type, data_files=files_to_load, split="train")
            logger.info(f"✓ Loaded {len(train_dataset):,} packed training sequences.")
        else:
            # 2. Try load_from_disk (if saved with save_to_disk)
            try:
                from datasets import load_from_disk
                train_dataset = load_from_disk(str(data_path))
                if isinstance(train_dataset, dict):
                    train_dataset = train_dataset.get("train", next(iter(train_dataset.values())))
                logger.info(f"✓ Loaded {len(train_dataset):,} packed training sequences via load_from_disk.")
            except Exception as e:
                logger.warning(f"Could not load via load_from_disk: {e}")

    if train_dataset is None:
        # Only treat as HF repo if it's in 'owner/dataset' format and NOT a filesystem path
        if data_dir and "/" in str(data_dir) and not str(data_dir).startswith("/"):
            # Load directly from Hugging Face dataset repository
            hf_token = os.environ.get("HF_TOKEN")
            logger.info(f"Loading dataset directly from Hugging Face Hub: {data_dir}...")
            train_dataset = load_dataset(str(data_dir), split="train", token=hf_token)
            logger.info(f"✓ Loaded {len(train_dataset):,} packed training sequences from HF Hub.")
        else:
            logger.error(
                f"No dataset found at '{data_dir}' or in /kaggle/input. "
                "Please verify the directory or provide a valid Kaggle path / HF Hub repo."
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
