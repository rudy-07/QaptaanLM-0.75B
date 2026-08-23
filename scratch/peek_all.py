import sys
import time

def log(msg):
    print(msg, flush=True)

try:
    from datasets import load_dataset
except Exception as e:
    log(f"Datasets import error: {e}")
    sys.exit(1)

sources = [
    ("HuggingFaceTB/smoltalk", "openhermes-100k", "train"),
    ("HuggingFaceTB/smoltalk", "self-oss-instruct", "train"),
    ("open-thoughts/OpenThoughts-114k", None, "train"),
    ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train"),
    ("theblackcat102/evol-codealpaca-v1", None, "train"),
    ("allenai/sciq", None, "train"),
]

for name, cfg, spl in sources:
    log(f"\n==========================================")
    log(f"Inspecting: {name} (cfg={cfg}, spl={spl})")
    t0 = time.time()
    try:
        kwargs = {"streaming": True, "split": spl}
        if cfg:
            kwargs["name"] = cfg
        ds = load_dataset(name, **kwargs)
        row = next(iter(ds))
        elapsed = time.time() - t0
        log(f"SUCCESS in {elapsed:.2f}s! Keys: {list(row.keys())}")
        for k in list(row.keys())[:5]:
            v = row[k]
            if isinstance(v, str):
                s = repr(v[:120])
            elif isinstance(v, list):
                s = f"list(len={len(v)}) -> {repr(v[:1])[:100]}"
            elif isinstance(v, dict):
                s = f"dict keys={list(v.keys())}"
            else:
                s = repr(v)[:100]
            log(f"  {k}: {s}")
    except Exception as e:
        log(f"FAILED in {time.time()-t0:.2f}s: {e}")
