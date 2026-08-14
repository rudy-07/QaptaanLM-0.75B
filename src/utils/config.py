"""Configuration loading and validation utilities.

Handles loading YAML configs, environment detection, and
merging environment-specific overrides.
"""

import os
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


def detect_environment() -> str:
    """Detect whether we're running on Colab, Kaggle, or local."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.path.exists("/content"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def get_project_root() -> Path:
    """Get the project root directory."""
    # Walk up from this file to find configs/
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists():
            return parent
    # Fallback: use CWD
    return Path.cwd()


def resolve_paths(config: Dict[str, Any], env: str) -> Dict[str, Any]:
    """Resolve storage paths based on detected environment."""
    storage = config.get("storage", {})

    if env == "colab":
        gdrive = storage.get("gdrive", {})
        mount = gdrive.get("mount_path", "/content/drive")
        config["training"]["output_dir"] = os.path.join(
            mount, gdrive.get("checkpoint_dir", "MyDrive/qwen-coder/checkpoints")
        )
        data_dir = os.path.join(
            mount, gdrive.get("data_dir", "MyDrive/qwen-coder/data")
        )
        if not config["storage"]["processed_data"].get("path"):
            config["storage"]["processed_data"]["path"] = data_dir

    elif env == "kaggle":
        kaggle_cfg = storage.get("kaggle", {})
        config["training"]["output_dir"] = os.path.join(
            kaggle_cfg.get("output_dir", "/kaggle/working"), "checkpoints"
        )
        if not config["storage"]["processed_data"].get("path"):
            config["storage"]["processed_data"]["path"] = os.path.join(
                kaggle_cfg.get("output_dir", "/kaggle/working"), "data"
            )

    else:  # local
        root = get_project_root()
        config["training"]["output_dir"] = str(root / "checkpoints" / "cpt")
        if not config["storage"]["processed_data"].get("path"):
            config["storage"]["processed_data"]["path"] = str(root / "data" / "processed")

    return config


def load_config(config_name: str = "cpt_config.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_name: Name of the config file in configs/, relative path,
                     or an absolute path.

    Returns:
        Merged configuration dictionary with environment overrides applied.
    """
    candidates = [
        Path(config_name),
        get_project_root() / config_name,
        get_project_root() / "configs" / config_name,
        get_project_root() / "configs" / Path(config_name).name,
    ]
    config_path = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            config_path = cand
            break

    if config_path is None:
        raise FileNotFoundError(
            f"Config file not found: {config_name} (searched: {[str(c) for c in candidates]})"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Detect environment and apply overrides
    env = detect_environment()
    env_cfg = config.get("environment", {})

    if env_cfg.get("type", "auto") != "auto":
        env = env_cfg["type"]

    # Apply environment-specific overrides to training config
    overrides = env_cfg.get(env, {})
    if overrides:
        training = config.get("training", {})
        for key, value in overrides.items():
            training[key] = value
        config["training"] = training

    # Resolve paths
    config = resolve_paths(config, env)
    config["_environment"] = env

    return config


def load_dataset_config(config_name: str = "dataset_config.yaml") -> Dict[str, Any]:
    """Load the dataset configuration file.

    Args:
        config_name: Name of the dataset config file.

    Returns:
        Dataset configuration dictionary.
    """
    candidates = [
        Path(config_name),
        get_project_root() / config_name,
        get_project_root() / "configs" / config_name,
        get_project_root() / "configs" / Path(config_name).name,
    ]
    config_path = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            config_path = cand
            break

    if config_path is None:
        raise FileNotFoundError(
            f"Dataset config not found: {config_name} (searched: {[str(c) for c in candidates]})"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate proportions sum to 1.0
    datasets = config.get("datasets", {})
    total_prop = sum(ds.get("target_proportion", 0) for ds in datasets.values())
    if abs(total_prop - 1.0) > 0.01:
        prop_details = {k: v.get("target_proportion", 0) for k, v in datasets.items()}
        raise ValueError(
            f"Dataset proportions sum to {total_prop:.3f}, expected 1.0. "
            f"Proportions: {prop_details}"
        )

    return config


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a human-readable summary of the configuration."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        # Fallback without rich
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        print(f"Environment: {config.get('_environment', 'unknown')}")
        training = config.get("training", {})
        print(f"Model: {config.get('model', {}).get('name_or_path', 'unknown')}")
        print(f"Output dir: {training.get('output_dir', 'unknown')}")
        print(f"Max seq length: {training.get('max_seq_length', 'unknown')}")
        print(f"Batch size: {training.get('per_device_train_batch_size', 'unknown')}")
        print(f"Grad accum: {training.get('gradient_accumulation_steps', 'unknown')}")
        print(f"LR: {training.get('learning_rate', 'unknown')}")
        print(f"BF16: {training.get('bf16', False)}")
        print(f"Grad checkpoint: {training.get('gradient_checkpointing', False)}")
        print("=" * 60)
        return

    console.print("\n[bold cyan]Configuration Summary[/bold cyan]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Parameter", style="dim")
    table.add_column("Value")

    training = config.get("training", {})
    model = config.get("model", {})

    table.add_row("Environment", config.get("_environment", "unknown"))
    table.add_row("Model", model.get("name_or_path", "unknown"))
    table.add_row("Model class", model.get("model_class", "auto"))
    table.add_row("Output dir", training.get("output_dir", "unknown"))
    table.add_row("Max seq length", str(training.get("max_seq_length", "?")))
    table.add_row(
        "Batch size (per device)",
        str(training.get("per_device_train_batch_size", "?")),
    )
    table.add_row(
        "Grad accumulation", str(training.get("gradient_accumulation_steps", "?"))
    )
    table.add_row("Learning rate", str(training.get("learning_rate", "?")))
    table.add_row("Warmup ratio", str(training.get("warmup_ratio", "?")))
    table.add_row("BF16", str(training.get("bf16", False)))
    table.add_row(
        "Gradient checkpointing", str(training.get("gradient_checkpointing", False))
    )
    table.add_row("Optimizer", training.get("optim", "unknown"))
    table.add_row("Save steps", str(training.get("save_steps", "?")))

    console.print(table)
