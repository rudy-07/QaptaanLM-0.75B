"""
Script to create and push an updated, bulletproof starter notebook for KapCode-1B on Kaggle
"""
import requests
import json
import os

TOKEN = "KGAT_611f927b3ce3c16efd90315ff16b02e7"
URL = "https://www.kaggle.com/api/v1/kernels/push"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# KapCode-1B: Dataset Exploration & Quickstart\n",
            "\n",
            "This notebook demonstrates how to load, inspect, and decode pre-tokenized 4096-token shards from the **[KapCode-1B](https://www.kaggle.com/datasets/kaptaan45/kapcode-1b)** dataset.\n",
            "\n",
            "### Links & Resources\n",
            "- **Kaggle Dataset**: [kaptaan45/kapcode-1b](https://www.kaggle.com/datasets/kaptaan45/kapcode-1b)\n",
            "- **Hugging Face Dataset**: [kaptaan45/KapCode-1B](https://huggingface.co/datasets/kaptaan45/KapCode-1B)\n",
            "- **Associated Model Repository**: [QaptaanLM-0.75B on GitHub](https://github.com/rudy-07/QaptaanLM-0.75B)\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Locate Dataset Shards & Manifest\n",
            "We recursively scan `/kaggle/input` for Apache Arrow shard files (`.arrow`) and the curation manifest."
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
            "shard_files = sorted(glob.glob('/kaggle/input/**/*.arrow', recursive=True))\n",
            "if not shard_files:\n",
            "    # Fallback to local workspace search\n",
            "    shard_files = sorted(glob.glob('./**/*.arrow', recursive=True))\n",
            "\n",
            "print(f\"Found {len(shard_files)} Arrow shard files:\")\n",
            "for f in shard_files[:5]:\n",
            "    print(f\"  - {f}\")\n",
            "if len(shard_files) > 5:\n",
            "    print(f\"  ... and {len(shard_files) - 5} more shards\")\n",
            "\n",
            "# Locate manifest.json if present\n",
            "manifest_files = glob.glob('/kaggle/input/**/manifest.json', recursive=True)\n",
            "if manifest_files:\n",
            "    with open(manifest_files[0], 'r') as f:\n",
            "        manifest = json.load(f)\n",
            "    print(\"\\nManifest Metadata:\")\n",
            "    for k, v in list(manifest.items())[:6]:\n",
            "        print(f\"  {k}: {v}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Memory-Mapped Loading (Zero-Copy & RAM Efficient)\n",
            "If local shards are attached, we memory-map them using PyArrow. If running standalone, we stream directly from Hugging Face."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load dataset (local Arrow shards or Hugging Face streaming fallback)\n",
            "if shard_files:\n",
            "    print(\"Loading local Arrow shards via memory-mapping...\")\n",
            "    dataset = load_dataset('arrow', data_files=shard_files, split='train', keep_in_memory=False)\n",
            "    print(f\"Total Packed Sequences (4096 tokens each): {len(dataset):,}\")\n",
            "    print(f\"Dataset features: {dataset.features}\")\n",
            "else:\n",
            "    print(\"No local shards found; loading in streaming mode from Hugging Face...\")\n",
            "    dataset = load_dataset('kaptaan45/KapCode-1B', split='train', streaming=True)\n",
            "    print(\"Hugging Face streaming dataset connected!\")\n",
            "\n",
            "# Retrieve the first packed sequence\n",
            "first_sample = next(iter(dataset)) if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__') else dataset[0]\n",
            "input_ids = first_sample['input_ids']\n",
            "print(f\"Sample sequence #0 length: {len(input_ids)} tokens\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Decode Sample Token Sequence\n",
            "Decode the tokenized IDs back into readable code and mathematical proofs using the Qwen tokenizer."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Loading Qwen tokenizer for decoding...\")\n",
            "try:\n",
            "    # Load public Qwen tokenizer\n",
            "    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', trust_remote_code=True)\n",
            "    \n",
            "    # Decode first 350 tokens of sequence #0\n",
            "    decoded_snippet = tokenizer.decode(input_ids[:350], skip_special_tokens=False)\n",
            "    \n",
            "    print(\"=\" * 60)\n",
            "    print(\"Decoded Token Sample (First 350 Tokens):\")\n",
            "    print(\"=\" * 60)\n",
            "    print(decoded_snippet)\n",
            "    print(\"=\" * 60)\n",
            "except Exception as e:\n",
            "    print(f\"Tokenizer load note: {e}\")\n",
            "    print(f\"Raw token IDs (first 30 tokens): {input_ids[:30]}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. PyTorch / JAX Fast DataLoader Example\n",
            "Example pattern for training compact language models (like [QaptaanLM-0.75B](https://github.com/rudy-07/QaptaanLM-0.75B)) on KapCode-1B."
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
            "class PackedCodeDataset(torch.utils.data.Dataset):\n",
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
            "    torch_dataset = PackedCodeDataset(dataset)\n",
            "    loader = DataLoader(torch_dataset, batch_size=2, shuffle=False)\n",
            "    batch = next(iter(loader))\n",
            "    print(\"DataLoader Batch Shapes:\")\n",
            "    print(f\"  input_ids: {batch['input_ids'].shape}\")\n",
            "    print(f\"  attention_mask: {batch['attention_mask'].shape}\")\n",
            "    print(f\"  labels: {batch['labels'].shape}\")\n"
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
    "slug": "kaptaan45/kapcode-1b-dataset-quickstart",
    "newTitle": "KapCode-1B Dataset Exploration & Quickstart",
    "text": json.dumps(notebook_json),
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": False,
    "enableGpu": False,
    "enableTpu": False,
    "enableInternet": True,
    "datasetDataSources": ["kaptaan45/kapcode-1b"]
}

print("Pushing updated notebook to Kaggle...")
res = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
print("Status:", res.status_code)
print("Response:", res.text)
