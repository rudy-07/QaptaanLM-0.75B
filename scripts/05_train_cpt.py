"""CPT Training Launch Script.

Loads processed dataset shards (or generates small local batch)
and executes Continued Pre-Training on Qwen3.5-0.8B.

Usage:
    python scripts/05_train_cpt.py
    python scripts/05_train_cpt.py --config configs/cpt_config.yaml --data-dir data/processed

Kaggle TPU (PJRT):
    PJRT_DEVICE=TPU XLA_USE_BF16=1 python scripts/05_train_cpt.py --data-dir /kaggle/working/data

Do not wrap this command in ``torchrun`` or the removed ``xla_spawn`` launcher.
``torch_xla.launch`` below owns process creation and selects the correct number
of processes for the TPU topology exposed by the Kaggle runtime.
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


def _sanitize_kaggle_pjrt_environment():
    """Remove Kaggle's stale single-address XRT hint before PJRT starts.

    Kaggle currently exposes ``TPU_PROCESS_ADDRESSES=local`` in some TPU
    kernels.  PJRT interprets that as a one-worker topology, while the v5e-8
    runtime reports eight workers, producing ``Expected 8 worker addresses,
    got 1`` during initialization.  A real multi-host TPU setup may provide a
    comma-separated address list, so only the known Kaggle sentinel is removed.
    """
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return
    process_addresses = os.environ.get("TPU_PROCESS_ADDRESSES", "").strip().lower()
    if process_addresses in {"local", "localhost"}:
        os.environ.pop("TPU_PROCESS_ADDRESSES", None)


_sanitize_kaggle_pjrt_environment()

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Prevent broken torchvision binary mismatch crashes in Colab/Kaggle
try:
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    _iu.is_torchvision_available = lambda: False
except Exception:
    pass

# Prevent multithreading IPC deadlocks and memory thrashing in multi-worker TPU runs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from datasets import load_dataset, Dataset
from src.training.trainer import CPTTrainer
from src.utils.config import load_config, detect_environment
from src.utils.logging_utils import setup_logging

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _load_dataset(config, env):
    """Load the training dataset. Shared by single-process and multi-process paths."""
    import gc
    data_dir = None
    # Try CLI --data-dir first (stored in config by main)
    data_dir = config.get("_cli_data_dir")
    if not data_dir:
        data_dir = config["storage"]["processed_data"]["path"]

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
            # keep_in_memory=False ensures memory mapping from disk without duplicating full tables into RAM
            train_dataset = load_dataset(file_type, data_files=files_to_load, split="train", keep_in_memory=False)
            logger.info(f"✓ Loaded {len(train_dataset):,} packed training sequences.")
        else:
            # 2. Try load_from_disk (if saved with save_to_disk)
            try:
                from datasets import load_from_disk
                train_dataset = load_from_disk(str(data_path), keep_in_memory=False)
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
            train_dataset = load_dataset(str(data_dir), split="train", token=hf_token, keep_in_memory=False)
            logger.info(f"✓ Loaded {len(train_dataset):,} packed training sequences from HF Hub.")
        else:
            logger.error(
                f"No dataset found at '{data_dir}' or in /kaggle/input. "
                "Please verify the directory or provide a valid Kaggle path / HF Hub repo."
            )
            return None

    gc.collect()
    return train_dataset


def _is_tpu_requested():
    """Return whether this process was explicitly launched for a PJRT TPU."""
    try:
        import torch_xla  # noqa: F401
        return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
    except ImportError:
        return False


def _train_worker(config, train_dataset):
    """Training worker — runs on each device/process."""
    trainer = CPTTrainer(config)
    trainer.setup()
    trainer.train(train_dataset=train_dataset)
    logger.info("CPT Training successfully completed!")


def _tpu_worker(index, config, env):
    """Run once per process created by ``torch_xla.launch``.

    PJRT decides how many processes are required for the actual TPU topology.
    In particular, this must not be hard-coded to eight: a TPU chip can expose
    more than one core and the correct process count varies by runtime version.
    """
    import torch_xla
    import torch_xla.runtime as xr

    world_size = xr.world_size()
    global_ordinal = xr.global_ordinal()
    is_primary = global_ordinal == 0

    # Avoid eight workers concurrently truncating the same log file. Trainer
    # itself emits distributed-aware progress logs from the world process zero.
    setup_logging(
        log_dir=str(project_root / "logs") if is_primary else None,
        log_name="cpt_training",
        console=is_primary,
    )
    logger.info(
        "TPU PJRT worker initialized "
        f"(launch_index={index}, global_ordinal={global_ordinal}, "
        f"world_size={world_size}, device={torch_xla.device()})"
    )

    train_dataset = _load_dataset(config, env)
    if train_dataset is None:
        raise RuntimeError(f"[TPU worker {global_ordinal}] Dataset could not be loaded.")

    _train_worker(config, train_dataset)


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

    # Stash CLI data-dir into config so workers can access it
    if args.data_dir:
        config["_cli_data_dir"] = args.data_dir

    # Do not open a shared log file before the PJRT launcher forks/spawns.  Each
    # TPU worker configures logging in _tpu_worker, with only ordinal zero
    # writing files and the notebook console.
    log_dir = str(project_root / "logs")
    setup_logging(log_dir=None if _is_tpu_requested() else log_dir, log_name="cpt_training")

    logger.info("=" * 60)
    logger.info(f"Starting Qwen3.5-0.8B CPT Training in [{env}] environment")
    logger.info("=" * 60)

    if _is_tpu_requested():
        # xla_spawn/xmp.spawn and xrt_world_size belong to the retired XRT
        # workflow.  PJRT's supported entry point determines the topology and
        # does not need --num_cores or a hand-written nprocs value.
        import torch_xla

        logger.info("PJRT TPU requested. Launching across the runtime-discovered TPU topology...")
        torch_xla.launch(_tpu_worker, args=(config, env))
    else:
        # GPU / CPU path.
        train_dataset = _load_dataset(config, env)
        if train_dataset is not None:
            _train_worker(config, train_dataset)


if __name__ == "__main__":
    main()
