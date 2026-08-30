"""Stage 2: Supervised Fine-Tuning (SFT) Launch Script (PyTorch / Accelerate).

Executes full-parameter instruction fine-tuning on QaptaanLM-0.75B using the
KapInstruct-100M dataset with assistant-only loss masking.

Usage:
    python scripts/07_train_sft.py
    python scripts/07_train_sft.py --config configs/sft_config.yaml --data-dir data/kapinstruct
    
Multi-GPU (DDP):
    torchrun --nproc_per_node=2 scripts/07_train_sft.py --data-dir /kaggle/working/data
"""

import argparse
import glob
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Prevent broken torchvision binary mismatch crashes in Colab/Kaggle
try:
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    _iu.is_torchvision_available = lambda: False
except Exception:
    pass

import yaml
from datasets import load_dataset, Dataset
from src.training.trainer import CPTTrainer
from src.utils.config import load_config, detect_environment
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("train_sft")


def _load_sft_dataset(data_dir: str = None):
    """Auto-discover and load KapInstruct-100M Arrow/Parquet shards or load from HF."""
    search_dirs = []
    if data_dir:
        search_dirs.append(Path(data_dir))
    if os.path.exists("/kaggle/input"):
        search_dirs.append(Path("/kaggle/input"))
    if os.path.exists("data/kapinstruct"):
        search_dirs.append(Path("data/kapinstruct"))
    if os.path.exists("data/processed"):
        search_dirs.append(Path("data/processed"))

    files_to_load = []
    file_type = None

    for d in search_dirs:
        if not d.exists():
            continue
        arrow_files = sorted([str(f) for f in d.rglob("*.arrow") if not f.name.startswith(".")])
        parquet_files = sorted([str(f) for f in d.rglob("*.parquet") if not f.name.startswith(".")])
        if arrow_files:
            file_type = "arrow"
            files_to_load = arrow_files
            break
        elif parquet_files:
            file_type = "parquet"
            files_to_load = parquet_files
            break

    if files_to_load:
        logger.info(f"Loading {len(files_to_load)} {file_type.upper()} shards from disk...")
        return load_dataset(file_type, data_files=files_to_load, split="train", keep_in_memory=False)

    logger.info("No local shards found. Streaming KapInstruct-100M from Hugging Face Hub...")
    return load_dataset("kaptaan45/KapInstruct-100M", split="train")


def create_synthetic_sft_dataset(num_samples: int = 50, seq_len: int = 512) -> Dataset:
    """Generate synthetic packed instruction records for quick smoke testing."""
    import numpy as np
    rng = np.random.default_rng(42)
    input_ids = rng.integers(1, 248320, size=(num_samples, seq_len), dtype=np.int32)
    labels = input_ids.copy()
    
    # Apply assistant-only masking: 50% of tokens masked with -100
    mask = rng.random(size=(num_samples, seq_len)) < 0.5
    labels[mask] = -100
    
    return Dataset.from_dict({
        "input_ids": list(input_ids),
        "labels": list(labels),
        "attention_mask": list(np.ones((num_samples, seq_len), dtype=np.int8)),
    })


def _find_local_model_path() -> str:
    """Auto-detect local CPT model weights in /kaggle/input/ or local directories."""
    candidates = glob.glob("/kaggle/input/**/model.safetensors", recursive=True) + \
                 glob.glob("/kaggle/input/**/config.json", recursive=True)
    for m in candidates:
        parent = os.path.dirname(m)
        if os.path.exists(os.path.join(parent, "config.json")):
            return parent
    return None


def main():
    parser = argparse.ArgumentParser(description="QaptaanLM-0.75B PyTorch SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_gpu_config.yaml" if Path("configs/sft_gpu_config.yaml").exists() else "configs/sft_config.yaml",
        help="Path to SFT YAML config",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to base CPT model weights")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing KapInstruct Arrow shards")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5-step smoke test")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--jax", action="store_true", help="Use high-performance JAX/Flax training engine with XLA compilation")
    args = parser.parse_args()

    if args.jax or os.environ.get("FRAMEWORK", "").lower() == "jax":
        from jax_training.train import main as jax_main
        cfg_file = "configs/sft_config.yaml" if (project_root / "configs" / "sft_config.yaml").exists() else str(args.config)
        sys.argv = ["train.py", "--mode", "sft", "--config", cfg_file]
        if args.data_dir:
            sys.argv.extend(["--data-dir", args.data_dir])
        if args.model_path:
            sys.argv.extend(["--model-path", args.model_path])
        if args.smoke_test:
            sys.argv.append("--smoke-test")
        return jax_main()

    log_dir = "/kaggle/working/logs" if os.path.exists("/kaggle/working") else "logs"
    setup_logging(log_dir=log_dir, log_name="train_sft")
    logger.info("=" * 65)
    logger.info("QaptaanLM-0.75B Stage 2: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 65)

    config_path = Path(args.config)
    if not config_path.exists():
        candidate = project_root / args.config
        if candidate.exists():
            config_path = candidate
        elif (project_root / "configs" / "sft_gpu_config.yaml").exists():
            config_path = project_root / "configs" / "sft_gpu_config.yaml"

    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config {config_path} not found. Using defaults.")
        config = load_config("cpt_config.yaml")

    # Auto-detect local model weights if running offline
    model_cfg = config.setdefault("model", {})
    if args.model_path:
        model_cfg["name_or_path"] = args.model_path
    elif not os.path.exists(str(model_cfg.get("name_or_path", ""))):
        local_model = _find_local_model_path()
        if local_model:
            logger.info(f"Auto-detected local CPT model weights at: {local_model}")
            model_cfg["name_or_path"] = local_model

    # Map SFT config keys for CPTTrainer compatibility
    train_cfg = config.get("training", {})
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir
    elif os.path.exists("/kaggle/working"):
        train_cfg["output_dir"] = "/kaggle/working/checkpoints/sft"
    elif "output_dir" not in train_cfg or "cpt" in train_cfg["output_dir"]:
        train_cfg["output_dir"] = "checkpoints/sft"

    train_cfg["run_name"] = "qaptaanlm-0.75b-sft"

    if args.smoke_test:
        train_cfg["max_steps"] = 5
        train_cfg["logging_steps"] = 1
        train_cfg["save_steps"] = 5
        train_cfg["max_seq_length"] = 256
        dataset = create_synthetic_sft_dataset(num_samples=20, seq_len=256)
    else:
        dataset = _load_sft_dataset(args.data_dir)

    # Train / Val split if eval ratio specified
    eval_dataset = None
    eval_ratio = float(train_cfg.get("eval_split_ratio", 0.015))
    if not args.smoke_test and eval_ratio > 0 and len(dataset) > 100:
        split = dataset.train_test_split(test_size=eval_ratio, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Split KapInstruct: {len(dataset):,} train records, {len(eval_dataset):,} val records")

    trainer = CPTTrainer(config=config)
    trainer.setup()
    trainer.train(train_dataset=dataset, eval_dataset=eval_dataset)
    logger.info("✓ SFT Training complete!")


if __name__ == "__main__":
    main()
