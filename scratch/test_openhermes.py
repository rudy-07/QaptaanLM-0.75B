from datasets import load_dataset
import time

print("Testing teknium/OpenHermes-2.5 loading...")
t0 = time.time()
try:
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    sample = next(iter(ds))
    print(f"SUCCESS in {time.time()-t0:.2f}s! Keys: {list(sample.keys())}")
    print("conversations len:", len(sample.get("conversations", [])))
    print("first conversation turn:", sample.get("conversations", [])[:1])
except Exception as e:
    print(f"FAILED: {e}")
