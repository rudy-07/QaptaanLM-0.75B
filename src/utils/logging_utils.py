"""Structured logging utilities for the training pipeline.

Provides consistent logging format across all pipeline stages,
with support for file and console output.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    log_dir: Optional[str] = None,
    log_name: str = "pipeline",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Set up structured logging with file and console handlers.

    Args:
        log_dir: Directory for log files. If None, logs to console only.
        log_name: Name prefix for the log file.
        level: Logging level.
        console: Whether to add console handler.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handlers: write to timestamped log AND stable static log
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Timestamped log
        file_handler = logging.FileHandler(
            log_path / f"{log_name}_{timestamp}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Static log (e.g. data_processing.log) for convenient tailing/cat
        static_file_handler = logging.FileHandler(
            log_path / f"{log_name}.log",
            mode="w",
            encoding="utf-8",
        )
        static_file_handler.setLevel(level)
        static_file_handler.setFormatter(formatter)
        logger.addHandler(static_file_handler)

    # Silence noisy third-party streaming and network loggers
    for noisy in ["httpx", "httpcore", "urllib3", "fsspec", "datasets", "filelock"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def log_dict(logger: logging.Logger, title: str, data: Dict[str, Any]) -> None:
    """Log a dictionary as a formatted block."""
    logger.info(f"--- {title} ---")
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k2, v2 in value.items():
                logger.info(f"    {k2}: {v2}")
        else:
            logger.info(f"  {key}: {value}")


def save_json_log(
    data: Dict[str, Any],
    path: str,
    append: bool = False,
) -> None:
    """Save structured data as JSON for later analysis.

    Args:
        data: Data to save.
        path: Output file path.
        append: Whether to append as JSONL instead of overwriting.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if append:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
