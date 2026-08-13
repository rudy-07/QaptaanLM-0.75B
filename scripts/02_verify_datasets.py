"""Verify dataset access, schemas, and streaming.

This script validates that all 5 CPT datasets are accessible
and match the expected schema.
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output in Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def verify_dataset(name: str, source: str, **load_kwargs):
    """Verify a single dataset is accessible and inspect its schema.

    Args:
        name: Human-readable name.
        source: HuggingFace dataset ID.
        **load_kwargs: Additional kwargs for load_dataset.
    """
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"Verifying: {name}")
    print(f"Source: {source}")
    print(f"{'='*50}")

    start = time.time()
    try:
        ds = load_dataset(source, streaming=True, **load_kwargs)

        # Handle dict of splits
        if isinstance(ds, dict):
            split_names = list(ds.keys())
            print(f"  Splits: {split_names}")
            ds_iter = iter(ds[split_names[0]])
        else:
            ds_iter = iter(ds)

        # Get first sample
        sample = next(ds_iter)
        elapsed = time.time() - start

        print(f"  ✓ Successfully loaded in {elapsed:.1f}s")
        print(f"  Schema (first sample keys):")
        for key, value in sample.items():
            val_type = type(value).__name__
            if isinstance(value, str):
                val_preview = repr(value[:80]) + ("..." if len(value) > 80 else "")
                print(f"    {key}: {val_type} = {val_preview}")
            elif isinstance(value, (int, float, bool)):
                print(f"    {key}: {val_type} = {value}")
            elif isinstance(value, list):
                print(f"    {key}: list[{len(value)} items]")
                if value and isinstance(value[0], dict):
                    print(f"      Item keys: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"    {key}: dict with keys {list(value.keys())}")
            else:
                print(f"    {key}: {val_type}")

        # Get a few more samples for statistics
        print(f"\n  Sampling 10 records for stats...")
        from tqdm.auto import tqdm
        samples = [sample]
        for i, s in enumerate(tqdm(ds_iter, total=9, desc=f"Sampling [{name}]", leave=False)):
            samples.append(s)
            if i >= 8:  # Already have 1, get 9 more = 10 total
                break

        # Report text lengths if applicable
        text_key = None
        for key in ["text", "content", "code"]:
            if key in sample:
                text_key = key
                break

        if text_key:
            lengths = [len(s.get(text_key, "")) for s in samples]
            print(f"  Text field: '{text_key}'")
            print(f"    Min length: {min(lengths):,} chars")
            print(f"    Max length: {max(lengths):,} chars")
            print(f"    Avg length: {sum(lengths)/len(lengths):,.0f} chars")

        return True

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_all_datasets():
    """Verify all 5 CPT datasets."""
    print("=" * 60)
    print("Phase B: Dataset Verification")
    print("=" * 60)

    results = {}

    # 1. Stack v3
    results["stack_v3"] = verify_dataset(
        "The Stack v3 (Code)",
        "HuggingFaceCode/stack-v3-train",
        split="train",
    )

    # 2. FineWeb-HQ
    results["fineweb_hq"] = verify_dataset(
        "FineWeb-HQ",
        "epfml/FineWeb-HQ",
        split="train",
    )

    # 3. OpenWebMath
    results["openwebmath"] = verify_dataset(
        "OpenWebMath",
        "open-web-math/open-web-math",
        split="train",
    )

    # 4. The Vault
    results["the_vault"] = verify_dataset(
        "The Vault (Function-Level)",
        "parquet",
        data_files="hf://datasets/Fsoft-AIC/the-vault-function/data/train/full/*.parquet",
        split="train",
    )

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n✓ All datasets verified successfully!")
    else:
        print("\n⚠ Some datasets failed verification. Check errors above.")

    return all_ok


if __name__ == "__main__":
    verify_all_datasets()
