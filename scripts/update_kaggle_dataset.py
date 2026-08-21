"""
Script to update Kaggle dataset metadata and description for kaptaan45/kapcode-1b
"""
import requests
import json
import os

TOKEN = "KGAT_611f927b3ce3c16efd90315ff16b02e7"
DATASET_SLUG = "kapcode-1b"
OWNER_SLUG = "kaptaan45"
URL = f"https://www.kaggle.com/api/v1/datasets/metadata/{OWNER_SLUG}/{DATASET_SLUG}"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 1. Read DATASET_CARD.md
dataset_card_path = os.path.join(os.path.dirname(__file__), "..", "DATASET_CARD.md")
with open(dataset_card_path, "r", encoding="utf-8") as f:
    raw_content = f.read()

# Strip YAML frontmatter if present
if raw_content.startswith("---"):
    parts = raw_content.split("---", 2)
    if len(parts) >= 3:
        body_markdown = parts[2].strip()
    else:
        body_markdown = raw_content.strip()
else:
    body_markdown = raw_content.strip()

# Custom Kaggle description header with Kaggle badges and quick links
kaggle_header = """[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kaptaan45%2FKapCode--1B-yellow.svg)](https://huggingface.co/datasets/kaptaan45/KapCode-1B)
[![GitHub](https://img.shields.io/badge/GitHub-QaptaanLM--0.75B-181717.svg?logo=github)](https://github.com/rudy-07/QaptaanLM-0.75B)
[![Starter Notebook](https://img.shields.io/badge/Kaggle-Starter%20Notebook-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/code/kaptaan45/kapcode-1b-dataset-quickstart)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tokens: 1 Billion](https://img.shields.io/badge/Tokens-1%20Billion-blue.svg)](#dataset-composition)

"""

# Add Kaggle-specific loading section into markdown
kaggle_quickstart = """
### Kaggle Notebook Quickstart (Zero-Copy Arrow Loading)

```python
import glob
from datasets import load_dataset

# Load pre-tokenized memory-mapped shards from Kaggle input
shard_files = sorted(glob.glob("/kaggle/input/kapcode-1b/*.arrow"))
if not shard_files:
    # Fallback if mounted under different folder name
    shard_files = sorted(glob.glob("/kaggle/input/kapcode*/**/*.arrow", recursive=True))

print(f"Found {len(shard_files)} Arrow shards")

# Memory-map the dataset (instant loading, zero RAM overhead)
dataset = load_dataset("arrow", data_files=shard_files, split="train", keep_in_memory=False)
print(f"Total packed sequences: {len(dataset):,}")
print(f"Sequence length: {len(dataset[0]['input_ids'])} tokens")
```
"""

# Enhance description with Kaggle quickstart
enhanced_description = kaggle_header + body_markdown.replace(
    "## Dataset Loading and Usage",
    "## Dataset Loading and Usage\n" + kaggle_quickstart
)

# 2. Prepare file descriptions
files_meta = [
    {
        "name": ".gitattributes",
        "description": "Git LFS configuration defining Large File Storage attributes for binary Arrow shards.",
        "totalBytes": 2504,
        "columns": []
    },
    {
        "name": "manifest.json",
        "description": "Global manifest indexing all 1B tokens, shard bounds, sequence counts, and schema mappings.",
        "totalBytes": 16150,
        "columns": []
    },
    {
        "name": "processing_report.json",
        "description": "Curation diagnostics and audit log detailing upstream ingestion, FastText LID scores, and FIM distribution.",
        "totalBytes": 1594,
        "columns": []
    }
]

for i in range(17):
    shard_name = f"shard_{i:05d}.arrow"
    files_meta.append({
        "name": shard_name,
        "description": f"Pre-tokenized memory-mapped Arrow shard {i} (packed 4096 tokens/sequence) containing input_ids, attention_mask, and labels.",
        "totalBytes": 104904394,
        "columns": [
            {"name": "input_ids", "description": "BPE token IDs packed to length 4096 using Qwen3.5 tokenizer", "type": "integer"},
            {"name": "attention_mask", "description": "Binary attention mask vector of length 4096", "type": "integer"},
            {"name": "labels", "description": "Target token labels for causal LM loss computation", "type": "integer"}
        ]
    })

# 3. Complete unified payload
payload = {
    "title": "KapCode-1B",
    "subtitle": "Curated 1-Billion Token Dataset for Compact Code Models",
    "description": enhanced_description,
    "isPrivate": False,
    "userSpecifiedSources": "https://huggingface.co/datasets/kaptaan45/KapCode-1B, HuggingFaceCode/stack-v3-train, Fsoft-AIC/the-vault-function, epfml/FineWeb-HQ, open-web-math/open-web-math",
    "expectedUpdateFrequency": "never",
    "licenses": [{"name": "other"}],
    "keywords": ["computer science", "nlp", "deep learning"],
    "data": files_meta
}

print(f"Updating Kaggle dataset {OWNER_SLUG}/{DATASET_SLUG} with full metadata...")
response = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
print("Status code:", response.status_code)
try:
    res_json = response.json()
    print("Response:", json.dumps(res_json, indent=2))
    if res_json.get("errors"):
        print("Encountered errors:", res_json["errors"])
    else:
        print("Successfully updated dataset metadata, description, and file schemas!")
except Exception as e:
    print("Response text:", response.text)
