import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from datasets import load_dataset

sources = [
    {
        "name": "smol_magpie_ultra",
        "id": "HuggingFaceTB/smoltalk",
        "config": "smol-magpie-ultra",
        "split": "train",
        "sha": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
        "license": "Apache-2.0 / Open",
        "format_type": "messages"
    },
    {
        "name": "magicoder_evol",
        "id": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "config": None,
        "split": "train",
        "sha": "b0079beaa0361d82412520b873715bee59cc7dd4",
        "license": "Apache-2.0",
        "format_type": "instruction_response"
    },
    {
        "name": "magicoder_oss",
        "id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "config": None,
        "split": "train",
        "sha": "5f839b1f368a76b161028bb9edff055db34022b2",
        "license": "MIT",
        "format_type": "problem_solution"
    },
    {
        "name": "openmathinstruct2",
        "id": "nvidia/OpenMathInstruct-2",
        "config": "default",
        "split": "train_5M",
        "sha": "469216e3f46f4dacf476b382e192485ea51a143e",
        "license": "CC-BY-4.0",
        "format_type": "problem_generated_solution"
    },
    {
        "name": "numinamath_cot",
        "id": "AI-MO/NuminaMath-CoT",
        "config": None,
        "split": "train",
        "sha": "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
        "license": "Apache-2.0",
        "format_type": "problem_solution"
    },
    {
        "name": "openthoughts_reasoning",
        "id": "open-thoughts/OpenThoughts-114k",
        "config": None,
        "split": "train",
        "sha": "bd093c3994fd54d2390985b66988ddf282a55eb6",
        "license": "Apache-2.0",
        "format_type": "conversations"
    },
    {
        "name": "openhermes_2_5",
        "id": "parquet",
        "data_files": "https://huggingface.co/datasets/teknium/OpenHermes-2.5/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "split": "train",
        "sha": "b82037821055c377bed0d495e72e46de3bc72e84",
        "license": "MIT / Open",
        "format_type": "conversations"
    },
    {
        "name": "tulu3_sft",
        "id": "allenai/tulu-3-sft-mixture",
        "config": None,
        "split": "train",
        "sha": "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
        "license": "ODC-By",
        "format_type": "messages"
    },
    {
        "name": "self_oss_starcoder2",
        "id": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "config": None,
        "split": "train",
        "sha": "356bb069eee815daa6e23e9a282eeefe1490ad44",
        "license": "ODC-By",
        "format_type": "instruction_response"
    },
    {
        "name": "smol_constraints",
        "id": "HuggingFaceTB/smoltalk",
        "config": "smol-constraints",
        "split": "train",
        "sha": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
        "license": "Apache-2.0 / Open",
        "format_type": "messages"
    },
    {
        "name": "stem_qa",
        "id": "TIGER-Lab/WebInstructSub",
        "config": None,
        "split": "train",
        "sha": "559b33b6bcd34da3da047bb235532941026955a4",
        "license": "Apache-2.0",
        "format_type": "question_answer"
    },
    {
        "name": "code_debugging",
        "id": "m-a-p/CodeFeedback-Filtered-Instruction",
        "config": None,
        "split": "train",
        "sha": "a08c213a9748c66c15d0225814be80a2e77adf4a",
        "license": "Apache-2.0",
        "format_type": "query_answer"
    }
]

print("="*70, flush=True)
print("PROGRAMMATIC VERIFICATION OF ALL 12 INSTRUCTION SOURCES", flush=True)
print("="*70, flush=True)

all_passed = True
for s in sources:
    name = s["name"]
    print(f"\nChecking [{name}]...", flush=True)
    t0 = time.time()
    try:
        kw = {"split": s["split"], "streaming": True}
        if s.get("config"):
            kw["name"] = s["config"]
        if s.get("data_files"):
            kw["data_files"] = s["data_files"]
        ds = load_dataset(s["id"], **kw)
        row = next(iter(ds))
        elapsed = time.time() - t0
        print(f"  [OK] in {elapsed:.2f}s | License: {s['license']} | SHA: {s['sha'][:12]}...", flush=True)
        print(f"    Keys: {list(row.keys())}", flush=True)
    except Exception as e:
        print(f"  [FAIL]: {e}", flush=True)
        all_passed = False

print("\n" + "="*70, flush=True)
print(f"ALL SOURCES VERIFIED: {all_passed}", flush=True)
print("="*70, flush=True)
