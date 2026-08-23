"""Script to create and update Kaggle dataset metadata and description for kaptaan45/kapinstruct-100m.

Uses Kaggle API with token authentication (KGAT_... or KAGGLE_API_TOKEN).
"""

import json
import os
import sys
from pathlib import Path
import requests

TOKEN = os.environ.get("KAGGLE_API_TOKEN") or "KGAT_988bf9b71ea34b60bbce6cbac69677f3"
DATASET_SLUG = "kapinstruct-100m"
OWNER_SLUG = "kaptaan45"
URL = f"https://www.kaggle.com/api/v1/datasets/metadata/{OWNER_SLUG}/{DATASET_SLUG}"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


def prepare_kaggle_description() -> str:
    """Read DATASET_CARD_KAPINSTRUCT.md and format for Kaggle."""
    card_path = Path(__file__).resolve().parent.parent / "DATASET_CARD_KAPINSTRUCT.md"
    with open(card_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Strip YAML frontmatter
    if raw_content.startswith("---"):
        parts = raw_content.split("---", 2)
        body_markdown = parts[2].strip() if len(parts) >= 3 else raw_content.strip()
    else:
        body_markdown = raw_content.strip()

    kaggle_header = """[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kaptaan45%2FKapInstruct--100M-orange.svg)](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)
[![Starter Notebook](https://img.shields.io/badge/Kaggle-Starter%20Notebook-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/code/kaptaan45/kapinstruct-100m-builder)
[![Tokens: 100 Million](https://img.shields.io/badge/Tokens-100%20Million-blue.svg)](#dataset-composition--source-mixture)
[![Loss Masking: Assistant Only](https://img.shields.io/badge/Loss-Assistant%20Only-red.svg)](#chatml-formatting--loss-masking)

"""

    kaggle_quickstart = """
### Kaggle Notebook Quickstart (Zero-Copy Arrow Loading)

```python
import glob
from datasets import load_dataset

# Load pre-tokenized memory-mapped shards from Kaggle input
shard_files = sorted(glob.glob("/kaggle/input/kapinstruct-100m/*.arrow"))
if not shard_files:
    shard_files = sorted(glob.glob("/kaggle/input/kapinstruct*/**/*.arrow", recursive=True))

print(f"Found {len(shard_files)} Arrow shards")

# Memory-map the dataset (instant loading, zero RAM overhead)
dataset = load_dataset("arrow", data_files=shard_files, split="train", keep_in_memory=False)
print(f"Total packed sequences: {len(dataset):,}")
print(f"Sequence length: {len(dataset[0]['input_ids'])} tokens")
print(f"Trainable tokens in seq 0: {sum(1 for l in dataset[0]['labels'] if l != -100)}")
```
"""
    enhanced = kaggle_header + body_markdown.replace(
        "## Dataset Loading and Usage",
        "## Dataset Loading and Usage\n" + kaggle_quickstart
    )
    return enhanced


def build_dataset_metadata(upload_dir: Path):
    """Generate dataset-metadata.json inside the upload directory."""
    enhanced_desc = prepare_kaggle_description()

    files_meta = [
        {
            "name": "manifest.json",
            "description": "Global manifest indexing all 100M tokens, shard bounds, sequence counts, packing statistics, and SHA-256 checksums.",
            "totalBytes": 12500,
            "columns": []
        },
        {
            "name": "mixture_report.json",
            "description": "Token accounting report detailing exact rendered and trainable token quotas across all 12 instruction sources.",
            "totalBytes": 3800,
            "columns": []
        },
        {
            "name": "filter_report.json",
            "description": "Diagnostics and rejection statistics per source (English LID, code language, secrets, LaTeX, prompt injection).",
            "totalBytes": 2100,
            "columns": []
        },
        {
            "name": "licenses.json",
            "description": "Source-specific license registry mapping each of the 12 constituent datasets to their upstream license terms.",
            "totalBytes": 2000,
            "columns": []
        },
        {
            "name": "source_registry.json",
            "description": "Registry of upstream dataset repositories, configurations, splits, and pinned commit SHAs.",
            "totalBytes": 6400,
            "columns": []
        },
        {
            "name": "DATASET_CARD.md",
            "description": "Comprehensive dataset documentation and markdown card.",
            "totalBytes": 12000,
            "columns": []
        }
    ]

    for i in range(10):
        shard_name = f"shard_{i:05d}.arrow"
        files_meta.append({
            "name": shard_name,
            "description": f"Pre-tokenized memory-mapped Arrow shard {i} (packed 4096 tokens/sequence, ChatML format, assistant-only loss masking).",
            "totalBytes": 50000000,
            "columns": [
                {"name": "input_ids", "description": "BPE token IDs packed to length 4096 using Qwen3.5 tokenizer", "type": "integer"},
                {"name": "attention_mask", "description": "Binary attention mask vector of length 4096", "type": "integer"},
                {"name": "labels", "description": "Target token labels (-100 for masked prompt/system tokens, token_id for trainable assistant response)", "type": "integer"}
            ]
        })

    meta = {
        "title": "KapInstruct-100M",
        "id": f"{OWNER_SLUG}/{DATASET_SLUG}",
        "subtitle": "Curated 100-Million Token Instruction Tuning Dataset for Compact Models",
        "description": enhanced_desc,
        "isPrivate": False,
        "licenses": [{"name": "other"}],
        "keywords": ["nlp", "deep learning", "computer science", "artificial intelligence"],
        "data": files_meta
    }

    meta_file = upload_dir / "dataset-metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated {meta_file}")
    return meta


def update_kaggle_metadata():
    """Update dataset metadata directly via Kaggle API."""
    enhanced_desc = prepare_kaggle_description()
    payload = {
        "title": "KapInstruct-100M",
        "subtitle": "Curated 100-Million Token Instruction Tuning Dataset for Compact Models",
        "description": enhanced_desc,
        "isPrivate": False,
        "userSpecifiedSources": "https://huggingface.co/datasets/kaptaan45/KapInstruct-100M, HuggingFaceTB/smoltalk, ise-uiuc/Magicoder-Evol-Instruct-110K, nvidia/OpenMathInstruct-2, teknium/OpenHermes-2.5, m-a-p/CodeFeedback-Filtered-Instruction",
        "expectedUpdateFrequency": "never",
        "licenses": [{"name": "other"}],
        "keywords": ["nlp", "deep learning", "computer science", "artificial intelligence"]
    }

    print(f"Updating Kaggle dataset metadata for {OWNER_SLUG}/{DATASET_SLUG}...")
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
        print("Status code:", response.status_code)
        if response.status_code in (200, 201):
            print("Successfully updated Kaggle dataset metadata!")
        else:
            print("Response:", response.text)
    except Exception as e:
        print("Error connecting to Kaggle API:", e)


if __name__ == "__main__":
    update_kaggle_metadata()
