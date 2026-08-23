"""Push KapInstruct-100M Dataset Builder & Hugging Face Publisher notebook to Kaggle.

Uses Kaggle API / REST endpoint with Bearer authentication.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_kaggle_kapinstruct")

KAGGLE_USERNAME = "kaptaan45"
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY") or "KGAT_988bf9b71ea34b60bbce6cbac69677f3"


def push_to_kaggle(notebook_path: str = "notebooks/kaggle_kapinstruct_build_and_publish.ipynb"):
    """Push notebook to Kaggle using Kaggle API."""
    nb_file = Path(notebook_path)
    if not nb_file.exists():
        raise FileNotFoundError(f"Notebook file not found at {nb_file.resolve()}")

    with open(nb_file, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    url = "https://www.kaggle.com/api/v1/kernels/push"
    headers = {
        "Authorization": f"Bearer {KAGGLE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "slug": f"{KAGGLE_USERNAME}/kapinstruct-100m-builder",
        "newTitle": "KapInstruct-100M Dataset Builder & HF Publisher",
        "text": json.dumps(nb_json),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": False,
        "enableGpu": False,
        "enableTpu": False,
        "enableInternet": True,
        "datasetDataSources": []
    }

    logger.info(f"Pushing {notebook_path} to Kaggle as {payload['slug']}...")
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    logger.info(f"Status: {res.status_code}")
    logger.info(f"Response: {res.text}")
    if res.status_code == 200:
        logger.info(f"Successfully published: https://www.kaggle.com/code/{payload['slug']}")
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish KapInstruct notebook to Kaggle")
    parser.add_argument("--notebook", type=str, default="notebooks/kaggle_kapinstruct_build_and_publish.ipynb")
    args = parser.parse_args()

    push_to_kaggle(args.notebook)
