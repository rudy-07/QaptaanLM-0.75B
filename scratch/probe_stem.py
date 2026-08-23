from datasets import load_dataset
from transformers import AutoTokenizer
import urllib.request
import json

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B-Base')

# Check SciQ
print('--- Checking SciQ ---')
try:
    ds = load_dataset('allenai/sciq', split='train')
    print('SciQ train rows:', len(ds))
    total_tokens = 0
    for r in ds:
        text = f"Question: {r['question']}\nContext: {r['support']}\nAnswer: {r['correct_answer']}"
        total_tokens += len(tok.encode(text, add_special_tokens=False))
    print(f'SciQ Total Tokens: {total_tokens:,} tokens ({total_tokens/1e6:.2f}M tokens)')
except Exception as e:
    print('SciQ failed:', e)

# Test candidate STEM QA datasets that have >4M tokens:
# Let's check potential STEM QA datasets on HuggingFace:
stem_candidates = [
    ("camel-ai/physics", None, "train"),
    ("camel-ai/chemistry", None, "train"),
    ("camel-ai/biology", None, "train"),
    ("camel-ai/math", None, "train"),
    ("ArtifactsMM/ScienceQA", None, "train"),
    ("allenai/ai2_arc", "ARC-Challenge", "train"),
    ("allenai/ai2_arc", "ARC-Easy", "train"),
    ("camel-ai/stem", None, "train"),
    ("Open-Orca/OpenOrca", None, "train"),
    ("TIGER-Lab/WebInstructSub", None, "train"),
    ("huggingface/synthetic-technical-qa", None, "train"),
    ("Idavidrein/gpqa", None, "train"),
    ("cais/mmlu", "all", "train"),
]

for name, cfg, spl in stem_candidates:
    print(f"\nChecking candidate: {name} (cfg={cfg})...")
    url = f"https://huggingface.co/api/datasets/{name}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"  Repo exists! SHA: {data.get('sha')}, License: {(data.get('cardData') or {}).get('license')}")
            tags = [t for t in data.get('tags', []) if 'size' in t or 'license' in t]
            print(f"  Tags: {tags}")
    except Exception as e:
        print(f"  HF API check failed: {e}")
