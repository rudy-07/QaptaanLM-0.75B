from datasets import load_dataset
import time

def test_ds(name, cfg=None, split="train"):
    print(f"\nTesting: {name} (cfg={cfg}, split={split})")
    try:
        t0 = time.time()
        kw = {"streaming": True, "split": split}
        if cfg: kw["name"] = cfg
        ds = load_dataset(name, **kw)
        row = next(iter(ds))
        print(f"  SUCCESS in {time.time()-t0:.2f}s! Keys: {list(row.keys())}")
        for k in list(row.keys())[:4]:
            print(f"    {k}: {repr(row[k])[:100]}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

test_ds("TIGER-Lab/WebInstructSub", split="train")
test_ds("cais/mmlu", cfg="all", split="train")
