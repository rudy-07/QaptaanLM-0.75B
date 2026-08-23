"""Sync all dataset files from Hugging Face Hub (kaptaan45/KapInstruct-100M)
directly to Kaggle Dataset (kaptaan45/kapinstruct-100m).

This downloads all 25 Arrow IPC shards, manifests, checksums, and reports from HF
and pushes them as a new dataset version to Kaggle.

Usage:
    python scripts/sync_hf_to_kaggle_dataset.py
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sync_hf_to_kaggle")

HF_REPO = "kaptaan45/KapInstruct-100M"
KAGGLE_DATASET = "kaptaan45/kapinstruct-100m"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
KAGGLE_TOKEN = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")


def sync_hf_to_kaggle(download_dir: str = "data/hf_kapinstruct_download"):
    """Download HF dataset and upload to Kaggle."""
    target_path = Path(download_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"1. Downloading complete dataset from Hugging Face: {HF_REPO}...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(target_path),
        token=HF_TOKEN,
        max_workers=4,
    )

    logger.info(f"Downloaded files to {target_path.resolve()}")
    files = list(target_path.glob("*"))
    arrow_files = list(target_path.glob("*.arrow"))
    logger.info(f"Found {len(files)} total files, including {len(arrow_files)} Arrow shards.")

    # 2. Write dataset-metadata.json for Kaggle
    meta = {
        "title": "KapInstruct-100M",
        "id": KAGGLE_DATASET,
        "licenses": [{"name": "other"}]
    }
    meta_path = target_path / "dataset-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Wrote Kaggle dataset metadata to {meta_path}")

    # 3. Setup Kaggle authentication if needed
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        with open(kaggle_json, "w", encoding="utf-8") as f:
            json.dump({"username": "kaptaan45", "key": KAGGLE_TOKEN}, f)

    # 4. Upload to Kaggle using KaggleApi
    logger.info(f"2. Pushing new version to Kaggle dataset: {KAGGLE_DATASET}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        res = api.dataset_create_version(
            folder=str(target_path),
            version_notes="Upload complete 25 Arrow shards, manifests, checksums, and reports from HF",
            quiet=False,
            dir_mode="zip"
        )
        logger.info(f"Kaggle upload response: {res}")
        logger.info(f"Successfully uploaded dataset to: https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    except Exception as e:
        logger.error(f"Kaggle API upload error: {e}")
        logger.info("Alternative: Run the sync cell directly inside a Kaggle notebook.")


if __name__ == "__main__":
    sync_hf_to_kaggle()
