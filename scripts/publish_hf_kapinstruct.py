"""Publish KapInstruct-100M dataset to Hugging Face Hub.

Uploads generated Arrow IPC shards, manifests, checksums, licenses, mixture reports,
and Markdown documentation directly to Hugging Face datasets.

Usage:
    python scripts/publish_hf_kapinstruct.py --data-dir data/kapinstruct --hf-repo kaptaan45/KapInstruct-100M
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import HfApi, login

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_hf_kapinstruct")


def publish_dataset_to_hf(
    data_dir: str = "data/kapinstruct",
    hf_repo_id: str = "kaptaan45/KapInstruct-100M",
    hf_token: Optional[str] = None,
    private: bool = False,
):
    """Publish dataset folder to Hugging Face Hub."""
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret("HF_TOKEN")
        except Exception:
            pass

    if token:
        login(token=token)
        logger.info("Successfully logged into Hugging Face Hub.")
    else:
        logger.warning(
            "No Hugging Face token found in HF_TOKEN env var or Kaggle Secrets. "
            "Proceeding with existing cached credentials if available."
        )

    folder_path = Path(data_dir)
    if not folder_path.exists():
        raise FileNotFoundError(f"Data directory not found at: {folder_path.resolve()}")

    manifest_file = folder_path / "manifest.json"
    if not manifest_file.exists():
        logger.warning(f"manifest.json not found in {folder_path}. Make sure build has completed.")

    api = HfApi()
    logger.info(f"Creating / ensuring Hugging Face dataset repo: {hf_repo_id}...")
    api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=private, exist_ok=True)

    logger.info(f"Uploading files from {folder_path} to {hf_repo_id}...")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=hf_repo_id,
        repo_type="dataset",
        commit_message="Upload KapInstruct-100M dataset shards and reports",
    )

    logger.info("=" * 75)
    logger.info(f"Successfully published KapInstruct-100M to Hugging Face!")
    logger.info(f"URL: https://huggingface.co/datasets/{hf_repo_id}")
    logger.info("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish KapInstruct-100M to Hugging Face Hub")
    parser.add_argument("--data-dir", type=str, default="data/kapinstruct")
    parser.add_argument("--hf-repo", type=str, default="kaptaan45/KapInstruct-100M")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    publish_dataset_to_hf(
        data_dir=args.data_dir,
        hf_repo_id=args.hf_repo,
        hf_token=args.hf_token,
        private=args.private,
    )
