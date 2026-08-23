from datasets import load_dataset
import time

print("Testing direct parquet streaming for OpenHermes-2.5...")
t0 = time.time()
try:
    parquet_url = "https://huggingface.co/datasets/teknium/OpenHermes-2.5/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    ds = load_dataset("parquet", data_files=parquet_url, split="train", streaming=True)
    sample = next(iter(ds))
    print(f"SUCCESS in {time.time()-t0:.2f}s! Keys: {list(sample.keys())}")
    print("Sample:", sample.get("conversations", [])[:1])
except Exception as e:
    print(f"FAILED: {e}")
