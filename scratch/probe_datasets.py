import sys
import json
import time

def log(msg):
    print(msg, flush=True)

try:
    from datasets import load_dataset
except ImportError as e:
    log(f"ImportError: {e}")
    sys.exit(1)

def test_one(ds_id, config=None, split='train'):
    log(f"\n--- Checking: {ds_id} (config={config}, split={split}) ---")
    t0 = time.time()
    try:
        kwargs = {"streaming": True, "split": split}
        if config:
            kwargs["name"] = config
        ds = load_dataset(ds_id, **kwargs)
        sample = next(iter(ds))
        elapsed = time.time() - t0
        log(f"  SUCCESS in {elapsed:.2f}s! Keys: {list(sample.keys())}")
        for k in list(sample.keys())[:4]:
            val = sample[k]
            if isinstance(val, str):
                s = val[:100].replace('\n', ' ')
            elif isinstance(val, list):
                s = f"list(len={len(val)}) -> {str(val[:1])[:80]}"
            else:
                s = str(val)[:80]
            log(f"    {k}: {s}")
        return True
    except Exception as e:
        log(f"  FAILED: {e}")
        return False

if __name__ == "__main__":
    sources = [
        ("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train"),
        ("ise-uiuc/Magicoder-Evol-Instruct-110K", None, "train"),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train"),
        ("nvidia/OpenMathInstruct-2", "default", "train_5M"),
        ("nvidia/OpenMathInstruct-2", None, "train_5M"),
        ("nvidia/OpenMathInstruct-2", None, "train"),
        ("AI-MO/NuminaMath-CoT", None, "train"),
        ("open-thoughts/OpenThoughts-114k", None, "train"),
        ("teknium/OpenHermes-2.5", None, "train"),
        ("allenai/tulu-3-sft-mixture", None, "train"),
        ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None, "train"),
        ("HuggingFaceTB/smoltalk", "smol-constraints", "train"),
        # STEM candidates:
        ("allenai/sciq", None, "train"),
        ("camel-ai/physics", None, "train"),
        ("camel-ai/chemistry", None, "train"),
        ("camel-ai/biology", None, "train"),
        ("FreedomIntelligence/STEM-Question-Answering", None, "train"),
        # Debugging candidates:
        ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train"),
        ("theblackcat102/evol-codealpaca-v1", None, "train"),
        ("bigcode/commitpackft", "python", "train"),
    ]

    for ds_id, cfg, spl in sources:
        test_one(ds_id, cfg, spl)
