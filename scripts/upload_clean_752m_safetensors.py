import logging
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_752m")

token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
api = HfApi(token=token)

src_file = Path(r"d:\Projects\mySphere projects\Qwen-Coder\models\model_752m.safetensors")
if not src_file.exists():
    raise FileNotFoundError(f"File not found: {src_file}")

logger.info(f"Uploading {src_file} ({src_file.stat().st_size / (1024*1024):.1f} MB) to kaptaan45/QaptaanLM-0.75B as model.safetensors...")

max_retries = 5
for attempt in range(1, max_retries + 1):
    try:
        logger.info(f"Attempt {attempt}/{max_retries}...")
        api.upload_file(
            path_or_fileobj=str(src_file),
            path_in_repo="model.safetensors",
            repo_id="kaptaan45/QaptaanLM-0.75B",
            repo_type="model",
            commit_message="Update model.safetensors: enforce tied embeddings (752M parameters)",
        )
        logger.info(" Successfully uploaded clean 752M model.safetensors!")
        break
    except Exception as e:
        logger.warning(f"Attempt {attempt} failed: {e}")
        if attempt < max_retries:
            time.sleep(3)
        else:
            raise
