"""Verify the base model and tokenizer.

This script performs Phase A audit tasks:
1. Verify model files exist and are valid
2. Load the model (text-only, stripping vision)
3. Test a forward pass
4. Verify tokenizer special tokens
5. Report model architecture details
"""

import sys
import os
import json
from pathlib import Path

# Ensure UTF-8 output in Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def verify_model(model_path: str = None):
    """Run all model verification checks."""
    from src.utils.config import load_config, detect_environment

    print("=" * 60)
    print("Phase A: Model Verification")
    print("=" * 60)

    # Load config
    config = load_config()
    env = detect_environment()
    print(f"\nEnvironment: {env}")

    if model_path is None:
        model_path = config["model"]["name_or_path"]
    print(f"Model: {model_path}")

    # Step 1: Check if model files exist locally
    print("\n--- Step 1: Checking model files ---")
    local_path = Path(model_path)
    if local_path.exists():
        files = list(local_path.iterdir())
        print(f"Local path exists with {len(files)} files:")
        for f in sorted(files):
            size = f.stat().st_size if f.is_file() else 0
            print(f"  {f.name}: {size / (1024*1024):.1f} MB")

        # Check for safetensors
        safetensors = [f for f in files if f.suffix == ".safetensors"]
        if safetensors:
            print(f"✓ Found {len(safetensors)} safetensor file(s)")
        else:
            print("⚠ No safetensors files found locally")
    else:
        print(f"Model not found locally at {local_path}")
        print("Will load from HuggingFace Hub")

    # Step 2: Load tokenizer
    print("\n--- Step 2: Loading tokenizer ---")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False
    )
    print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model max length: {tokenizer.model_max_length}")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  BOS token: {tokenizer.bos_token}")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # Check special tokens
    special_tokens = {
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
        "fim_prefix": "<|fim_prefix|>",
        "fim_middle": "<|fim_middle|>",
        "fim_suffix": "<|fim_suffix|>",
        "endoftext": "<|endoftext|>",
    }
    print("\n  Special tokens:")
    for name, token in special_tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    {name}: {token} -> ID {token_id}")

    # Test encode/decode roundtrip
    test_text = "def hello():\n    print('Hello, World!')\n"
    encoded = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    print(f"\n  Encode/decode test:")
    print(f"    Input:   {repr(test_text[:50])}")
    print(f"    Tokens:  {len(encoded)}")
    print(f"    Decoded: {repr(decoded[:50])}")
    roundtrip_ok = decoded.strip() == test_text.strip()
    print(f"    Roundtrip: {'✓ OK' if roundtrip_ok else '✗ MISMATCH'}")

    # Step 3: Load model (text-only)
    print("\n--- Step 3: Loading model (text-only CausalLM) ---")
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    # First check config
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    print(f"  Config type: {type(model_config).__name__}")
    print(f"  Model type: {model_config.model_type}")

    # Check for text config
    text_config = getattr(model_config, "text_config", None)
    if text_config:
        print(f"  Text hidden size: {text_config.hidden_size}")
        print(f"  Text layers: {text_config.num_hidden_layers}")
        print(f"  Text attention heads: {text_config.num_attention_heads}")
        print(f"  Text KV heads: {text_config.num_key_value_heads}")
        print(f"  Text vocab size: {text_config.vocab_size}")
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types:
            linear_count = sum(1 for lt in layer_types if lt == "linear_attention")
            full_count = sum(1 for lt in layer_types if lt == "full_attention")
            print(f"  Layer types: {linear_count} linear + {full_count} full attention")

    # Check vision config
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config:
        print(f"\n  Vision encoder detected:")
        print(f"    Hidden size: {vision_config.hidden_size}")
        print(f"    Depth: {vision_config.depth}")
        print(f"    Heads: {vision_config.num_heads}")

    # Try loading as CausalLM (text-only, strips vision)
    print("\n  Loading Qwen3_5ForCausalLM (text-only)...")
    try:
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load on CPU for verification
            trust_remote_code=False,
        )
        print(f"  ✓ Model loaded as Qwen3_5ForCausalLM")
        has_vision = hasattr(model, 'visual') or hasattr(model, 'model') and hasattr(getattr(model, 'model', None), 'visual')
        print(f"  Vision components present: {has_vision}")
    except Exception as e:
        print(f"  ⚠ Qwen3_5ForCausalLM failed: {e}")
        print("  Falling back to AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=False,
        )
        print(f"  ✓ Model loaded as {type(model).__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Step 4: Forward pass test
    print("\n--- Step 4: Forward pass test ---")
    test_input = "The quick brown fox"
    input_ids = tokenizer.encode(test_input, return_tensors="pt")
    print(f"  Input: {repr(test_input)}")
    print(f"  Input IDs shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")

    # Check if output makes sense
    next_token_id = logits[0, -1, :].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"  Predicted next token: {repr(next_token)} (ID: {next_token_id})")

    # Step 5: Generation test
    print("\n--- Step 5: Generation test ---")
    gen_input = tokenizer.encode("def fibonacci(n):\n", return_tensors="pt")
    with torch.no_grad():
        gen_output = model.generate(
            gen_input,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
        )
    generated = tokenizer.decode(gen_output[0], skip_special_tokens=True)
    print(f"  Generated:\n{generated}")

    print("\n" + "=" * 60)
    print("✓ Model verification complete!")
    print("=" * 60)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify Qwen3.5-0.8B-Base model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (default: from config)",
    )
    args = parser.parse_args()

    verify_model(args.model_path)
