"""Publish KapInstruct-100M pipeline notebook to Kaggle.

Uses Kaggle API via environment variables (KAGGLE_USERNAME, KAGGLE_KEY / KAGGLE_API_TOKEN)
with ZERO hardcoded secrets.
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_kaggle_kapinstruct")

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "kaptaan45")
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY", "")


def create_kernel_metadata(kernel_dir: Path, title: str = "KapInstruct-100M Dataset Builder & HF Publisher"):
    """Generate kernel-metadata.json for Kaggle API."""
    slug = "kapinstruct-100m-builder"
    meta = {
        "id": f"{KAGGLE_USERNAME}/{slug}",
        "title": title,
        "code_file": "kaggle_kapinstruct_build_and_publish.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    meta_path = kernel_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Generated kernel metadata at {meta_path}")
    return meta_path


def push_to_kaggle(notebook_path: str = "notebooks/kaggle_kapinstruct_build_and_publish.ipynb"):
    """Push notebook to Kaggle using kaggle CLI."""
    nb_file = Path(notebook_path)
    if not nb_file.exists():
        logger.error(f"Notebook file not found at {nb_file}")
        return False

    kernel_dir = nb_file.parent
    create_kernel_metadata(kernel_dir)

    # Check for credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        logger.warning(
            "No Kaggle credentials found in ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY env vars.\n"
            "Please set KAGGLE_USERNAME and KAGGLE_KEY before pushing."
        )
        return False

    import subprocess
    cmd = ["kaggle", "kernels", "push", "-p", str(kernel_dir)]
    logger.info(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    return res.returncode == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Publish KapInstruct notebook to Kaggle")
    parser.add_argument("--notebook", type=str, default="notebooks/kaggle_kapinstruct.ipynb")
    args = parser.parse_args()

    push_to_kaggle(args.notebook)
