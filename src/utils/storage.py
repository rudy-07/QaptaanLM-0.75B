"""Storage utilities for model/data upload/download.

Handles interactions with:
- Google Drive (Colab)
- Hugging Face Hub
- Kaggle output directories
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def mount_gdrive(mount_path: str = "/content/drive") -> bool:
    """Mount Google Drive in Colab.

    Returns:
        True if mounted successfully, False otherwise.
    """
    try:
        from google.colab import drive

        if not os.path.ismount(mount_path):
            drive.mount(mount_path)
        logger.info(f"Google Drive mounted at {mount_path}")
        return True
    except ImportError:
        logger.info("Not running in Colab, Google Drive not available")
        return False
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        return False


def upload_to_hub(
    local_path: str,
    repo_id: str,
    repo_type: str = "model",
    commit_message: str = "Upload checkpoint",
    token: Optional[str] = None,
) -> None:
    """Upload a directory to Hugging Face Hub.

    Args:
        local_path: Local directory path to upload.
        repo_id: HuggingFace repo ID (e.g., 'user/repo').
        repo_type: "model" or "dataset".
        commit_message: Git commit message.
        token: HF token. If None, uses cached token.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    try:
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        logger.info(f"Uploaded {local_path} to {repo_id}")
    except Exception as e:
        logger.error(f"Upload to Hub failed: {e}")
        raise


def copy_checkpoint(
    src: str,
    dst: str,
    max_checkpoints: int = 3,
) -> None:
    """Copy a checkpoint directory to backup storage.

    Args:
        src: Source checkpoint directory.
        dst: Destination directory.
        max_checkpoints: Maximum number of checkpoints to keep.
    """
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(src).name
    target = dst_path / checkpoint_name

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(src, target)
    logger.info(f"Copied checkpoint to {target}")

    # Clean old checkpoints
    existing = sorted(
        [d for d in dst_path.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    while len(existing) > max_checkpoints:
        old = existing.pop(0)
        shutil.rmtree(old)
        logger.info(f"Removed old checkpoint: {old.name}")


def get_storage_info() -> dict:
    """Get storage information for the current environment."""
    import shutil

    info = {}

    # Check common mount points
    for path_name, path in [
        ("cwd", os.getcwd()),
        ("/content", "/content"),
        ("/kaggle/working", "/kaggle/working"),
    ]:
        if os.path.exists(path):
            try:
                usage = shutil.disk_usage(path)
                info[path_name] = {
                    "total_gb": f"{usage.total / (1024**3):.1f}",
                    "used_gb": f"{usage.used / (1024**3):.1f}",
                    "free_gb": f"{usage.free / (1024**3):.1f}",
                }
            except Exception:
                pass

    return info
