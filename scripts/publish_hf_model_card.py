"""Upload MODEL_CARD.md as README.md to Hugging Face Model repo kaptaan45/QaptaanLM-0.75B.
"""

import logging
import os
import sys
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_model_card")

MODEL_ID = "kaptaan45/QaptaanLM-0.75B"


def get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        cache_token_path = Path.home() / ".cache" / "huggingface" / "token"
        if cache_token_path.exists():
            token = cache_token_path.read_text().strip()
    return token


def upload_model_card(card_path: str = "MODEL_CARD.md", repo_id: str = MODEL_ID):
    token = get_hf_token()
    if not token:
        raise ValueError("HF_TOKEN not found in environment or ~/.cache/huggingface/token.")

    card_file = Path(card_path)
    if not card_file.exists():
        raise FileNotFoundError(f"Card file not found: {card_file.resolve()}")

    content = card_file.read_text(encoding="utf-8")

    logger.info(f"Uploading {card_file} to https://huggingface.co/{repo_id} as README.md...")

    # Method 1: Try HfApi if available
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(card_file),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update comprehensive model card for QaptaanLM-0.75B",
        )
        logger.info(f"Successfully uploaded model card to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        logger.warning(f"HfApi upload failed: {e}. Falling back to Hugging Face REST commit API...")

    # Method 2: Direct REST commit API fallback via requests
    # Hugging Face Commit API: POST /api/models/{repo_id}/commit/main
    import base64
    url = f"https://huggingface.co/api/models/{repo_id}/commit/main"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "summary": "Update comprehensive model card for QaptaanLM-0.75B",
        "operations": [
            {
                "operation": "file",
                "path": "README.md",
                "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                "encoding": "base64"
            }
        ]
    }

    res = requests.post(url, headers=headers, json=payload, timeout=30)
    logger.info(f"REST API Response Status: {res.status_code}")
    if res.status_code in [200, 201]:
        logger.info(f"Successfully uploaded model card to https://huggingface.co/{repo_id}")
        return True
    else:
        logger.error(f"Failed to upload: {res.status_code} - {res.text}")
        sys.exit(1)


if __name__ == "__main__":
    upload_model_card()
