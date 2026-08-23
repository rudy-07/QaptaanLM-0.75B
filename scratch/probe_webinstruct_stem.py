import sys
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B-Base')

print("Probing WebInstructSub sources and STEM filtering...", flush=True)
ds = load_dataset("TIGER-Lab/WebInstructSub", split="train", streaming=True)

sources_seen = Counter()
stem_samples = 0
stem_tokens = 0
total_samples = 0

# STEM keyword/source indicators
STEM_KEYWORDS = [
    "physics", "chemistry", "biology", "science", "math", "engineering", 
    "computer", "algorithm", "mechanics", "astronomy", "geology", "thermodynamics",
    "quantum", "circuit", "biochemistry", "calculus", "algebra", "geometry",
    "electromagnetism", "optics", "molecule", "reaction", "genetics", "cellular"
]

t0 = time.time()
for r in ds:
    total_samples += 1
    src = r.get("source", "")
    sources_seen[src] += 1
    
    q = (r.get("question") or r.get("orig_question") or "").lower()
    a = (r.get("answer") or r.get("orig_answer") or "").lower()
    full = q + " " + a
    
    # Check if STEM
    is_stem = any(kw in full for kw in STEM_KEYWORDS)
    if is_stem:
        stem_samples += 1
        tokens = len(tok.encode(q + "\n" + a, add_special_tokens=False))
        stem_tokens += tokens
        
    if total_samples >= 5000:
        break

elapsed = time.time() - t0
print(f"Sampled {total_samples} rows in {elapsed:.2f}s:", flush=True)
print(f"Top sources in WebInstructSub: {sources_seen.most_common(10)}", flush=True)
print(f"STEM samples in sample: {stem_samples}/{total_samples} ({stem_samples/total_samples:.1%})", flush=True)
print(f"STEM tokens in sample: {stem_tokens:,} (Avg {stem_tokens/max(stem_samples,1):.1f} tokens/sample)", flush=True)
# Total rows in WebInstructSub is 2.3M+!
est_total_stem_tokens = (2_335_000 * (stem_samples/total_samples)) * (stem_tokens/max(stem_samples,1))
print(f"Estimated Total STEM Tokens across 2.3M dataset: {est_total_stem_tokens/1e6:.1f}M tokens!", flush=True)
