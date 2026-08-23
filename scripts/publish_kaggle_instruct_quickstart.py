"""Script to create and push the KapInstruct-100M Dataset Quickstart notebook to Kaggle.
"""

import json
import os
import requests

TOKEN = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
URL = "https://www.kaggle.com/api/v1/kernels/push"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}" if TOKEN else "",
    "Content-Type": "application/json"
}

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# KapInstruct-100M: Dataset Exploration & Quickstart\n",
            "\n",
            "This notebook demonstrates how to load, inspect, decode, and verify pre-tokenized 4096-token shards from the **[KapInstruct-100M](https://www.kaggle.com/datasets/kaptaan45/kapinstruct-100m)** dataset with **assistant-only loss masking**.\n",
            "\n",
            "### Links & Resources\n",
            "- **Kaggle Dataset**: [kaptaan45/kapinstruct-100m](https://www.kaggle.com/datasets/kaptaan45/kapinstruct-100m)\n",
            "- **Hugging Face Dataset**: [kaptaan45/KapInstruct-100M](https://huggingface.co/datasets/kaptaan45/KapInstruct-100M)\n",
            "- **Associated Foundation Model**: [QaptaanLM-0.75B on GitHub](https://github.com/rudy-07/QaptaanLM-0.75B)\n",
            "- **Dataset Builder Notebook**: [KapInstruct-100M Builder](https://www.kaggle.com/code/kaptaan45/kapinstruct-100m-builder)\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Locate Dataset Shards & Manifest\n",
            "We scan `/kaggle/input` for Apache Arrow shard files (`.arrow`) and the curation manifest."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import glob\n",
            "import json\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "from datasets import load_dataset\n",
            "from transformers import AutoTokenizer\n",
            "\n",
            "# 1. Locate all Arrow shards in /kaggle/input (recursive search)\n",
            "shard_files = sorted(glob.glob('/kaggle/input/**/shard_*.arrow', recursive=True))\n",
            "if not shard_files:\n",
            "    shard_files = sorted(glob.glob('/kaggle/input/**/*.arrow', recursive=True))\n",
            "if not shard_files:\n",
            "    shard_files = sorted(glob.glob('./**/*.arrow', recursive=True))\n",
            "\n",
            "print(f\"Found {len(shard_files)} Arrow shard files:\")\n",
            "for f in shard_files[:5]:\n",
            "    print(f\"  - {f}\")\n",
            "if len(shard_files) > 5:\n",
            "    print(f\"  ... and {len(shard_files) - 5} more shards\")\n",
            "\n",
            "# Locate manifest.json if present\n",
            "manifest_files = glob.glob('/kaggle/input/**/manifest*.json', recursive=True)\n",
            "if manifest_files:\n",
            "    with open(manifest_files[0], 'r') as f:\n",
            "        manifest = json.load(f)\n",
            "    print(\"\\nManifest Metadata:\")\n",
            "    for k, v in list(manifest.items())[:8]:\n",
            "        print(f\"  {k}: {v}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Zero-Copy Memory-Mapped Loading (PyArrow)\n",
            "If local shards are attached, we memory-map them using PyArrow. If running standalone, we stream directly from Hugging Face."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if shard_files:\n",
            "    print(\"Loading local Arrow shards via zero-copy memory-mapping...\")\n",
            "    dataset = load_dataset('arrow', data_files=shard_files, split='train', keep_in_memory=False)\n",
            "    print(f\"Total Packed Sequences (4096 tokens each): {len(dataset):,}\")\n",
            "    print(f\"Dataset features: {dataset.features}\")\n",
            "else:\n",
            "    print(\"No local shards found; loading in streaming mode from Hugging Face...\")\n",
            "    dataset = load_dataset('kaptaan45/KapInstruct-100M', split='train', streaming=True)\n",
            "    print(\"Hugging Face streaming dataset connected!\")\n",
            "\n",
            "# Retrieve sample sequence\n",
            "sample = next(iter(dataset)) if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__') else dataset[0]\n",
            "input_ids = sample['input_ids']\n",
            "labels = sample['labels']\n",
            "attention_mask = sample['attention_mask']\n",
            "\n",
            "print(f\"Sample sequence length: {len(input_ids)} tokens\")\n",
            "print(f\"Trainable tokens (labels != -100): {sum(1 for l in labels if l != -100)}\")\n",
            "print(f\"Masked tokens (labels == -100):    {sum(1 for l in labels if l == -100)}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Decode Sample Dialogue & Verify Assistant-Only Loss Masking\n",
            "We decode the sequence using `Qwen/Qwen3.5-0.8B-Base` tokenizer and inspect the exact token-level loss masking alignment."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Loading Qwen tokenizer...\")\n",
            "tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B-Base', trust_remote_code=True)\n",
            "\n",
            "print(\"=\" * 75)\n",
            "print(\"DECODED CONVERSATION SEQUENCE (First 400 Tokens):\")\n",
            "print(\"=\" * 75)\n",
            "print(tokenizer.decode(input_ids[:400], skip_special_tokens=False))\n",
            "print(\"=\" * 75)\n",
            "\n",
            "# Token-level loss label alignment preview\n",
            "print(\"\\nFirst 20 Token Alignment [Token -> Label]:\")\n",
            "for idx in range(min(20, len(input_ids))):\n",
            "    tok_str = tokenizer.decode([input_ids[idx]])\n",
            "    lbl = labels[idx]\n",
            "    status = 'TRAINABLE' if lbl != -100 else 'MASKED (-100)'\n",
            "    print(f\"  {idx:02d}: {repr(tok_str):<20} -> {lbl:<8} ({status})\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. PyTorch SFT Training Loop / DataLoader Example\n",
            "Demonstrates standard PyTorch DataLoader integration for Supervised Fine-Tuning (SFT) with assistant-only cross-entropy loss."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "from torch.utils.data import DataLoader\n",
            "\n",
            "class KapInstructTorchDataset(torch.utils.data.Dataset):\n",
            "    def __init__(self, hf_dataset):\n",
            "        self.dataset = hf_dataset\n",
            "        \n",
            "    def __len__(self):\n",
            "        return len(self.dataset)\n",
            "        \n",
            "    def __getitem__(self, idx):\n",
            "        item = self.dataset[idx]\n",
            "        return {\n",
            "            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),\n",
            "            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),\n",
            "            'labels': torch.tensor(item['labels'], dtype=torch.long)\n",
            "        }\n",
            "\n",
            "if hasattr(dataset, '__len__'):\n",
            "    torch_dataset = KapInstructTorchDataset(dataset)\n",
            "    loader = DataLoader(torch_dataset, batch_size=2, shuffle=True)\n",
            "    batch = next(iter(loader))\n",
            "    print(\"DataLoader Batch Shapes:\")\n",
            "    print(f\"  input_ids:      {batch['input_ids'].shape}\")\n",
            "    print(f\"  attention_mask: {batch['attention_mask'].shape}\")\n",
            "    print(f\"  labels:         {batch['labels'].shape}\")\n",
            "    print(f\"  trainable batch tokens: {(batch['labels'] != -100).sum().item()}\")\n"
        ]
    }
]

notebook_json = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

payload = {
    "slug": "kaptaan45/kapinstruct-100m-dataset-exploration-quickstart",
    "newTitle": "KapInstruct-100M Dataset Exploration & Quickstart",
    "text": json.dumps(notebook_json),
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": False,
    "enableGpu": False,
    "enableTpu": False,
    "enableInternet": True,
    "datasetDataSources": ["kaptaan45/kapinstruct-100m"]
}

# Write notebook locally
with open("notebooks/kaggle_kapinstruct_quickstart.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)
print("Wrote notebooks/kaggle_kapinstruct_quickstart.ipynb")

print("Pushing quickstart notebook to Kaggle...")
res = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
print("Status:", res.status_code)
print("Response:", res.text)
