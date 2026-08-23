"""Download model.safetensors from kaptaan45/QaptaanLM-0.75B, remove duplicate lm_head.weight,
verify exact 752M parameters, and re-upload to Hugging Face Hub.
"""

import io
import json
import logging
import os
import struct
import sys
import tempfile
from pathlib import Path
import requests
import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fix_safetensors_size")

REPO_ID = "kaptaan45/QaptaanLM-0.75B"


def get_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        cache_path = Path.home() / ".cache" / "huggingface" / "token"
        if cache_path.exists():
            token = cache_path.read_text().strip()
    return token


def fix_and_upload():
    token = get_token()
    if not token:
        raise ValueError("HF token not found.")

    api = HfApi(token=token)
    logger.info(f"1. Downloading current model.safetensors from {REPO_ID} via hf_hub_download...")

    from huggingface_hub import hf_hub_download
    downloaded_file = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.safetensors",
        repo_type="model",
        token=token,
    )
    logger.info(f"Downloaded model.safetensors to: {downloaded_file}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "model.safetensors"

        logger.info("2. Loading safetensors dictionary...")
        tensors = load_file(str(downloaded_file))
        logger.info(f"Loaded {len(tensors)} tensors. Total raw params: {sum(t.numel() for t in tensors.values()):,}")

        # Pop duplicate lm_head.weight (because tie_word_embeddings=True)
        if "lm_head.weight" in tensors:
            logger.info("Removing duplicate 'lm_head.weight' to enforce tied word embeddings...")
            del tensors["lm_head.weight"]

        total_clean_params = sum(t.numel() for t in tensors.values())
        logger.info(f"Clean tensor count: {len(tensors)}, Total parameters: {total_clean_params:,} ({total_clean_params/1e6:.2f}M)")

        assert total_clean_params == 752_393_024, f"Expected 752,393,024 params, got {total_clean_params:,}"

        logger.info(f"3. Saving clean 752M safetensors to {output_file}...")
        save_file(tensors, str(output_file))
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Clean safetensors size: {size_mb:.2f} MB")

        logger.info(f"4. Uploading clean model.safetensors to https://huggingface.co/{REPO_ID}...")
        api.upload_file(
            path_or_fileobj=str(output_file),
            path_in_repo="model.safetensors",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Fix model.safetensors: remove untied lm_head duplicate to reflect exact 752M parameter count",
        )
        logger.info(" Successfully uploaded clean 752M model.safetensors to Hugging Face!")


if __name__ == "__main__":
    fix_and_upload()
