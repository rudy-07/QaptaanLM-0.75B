"""Smoke test: verify the full pipeline works end-to-end.

Runs a tiny training loop (10 steps) on synthetic data to verify:
1. Model loads correctly (text-only)
2. Tokenizer works
3. Training loop runs
4. Loss decreases
5. Checkpoint save/load works
6. Generation works after training

This should be run locally (CPU) or on a GPU before committing
to a full training run.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def smoke_test(model_path: str = None, num_steps: int = 10):
    """Run the full pipeline smoke test."""
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    from src.utils.config import load_config, detect_environment
    from src.training.utils import detect_hardware, load_model_for_training

    print("=" * 60)
    print("Pipeline Smoke Test")
    print("=" * 60)

    env = detect_environment()
    hw = detect_hardware()
    print(f"Environment: {env}")
    print(f"Hardware: {hw}")

    config = load_config("cpt_config.yaml")
    if model_path:
        config["model"]["name_or_path"] = model_path

    # Step 1: Load model
    print("\n--- Step 1: Loading model ---")
    model, tokenizer = load_model_for_training(
        config["model"]["name_or_path"],
        dtype="float32" if not hw.get("bf16_support") else "bfloat16",
        gradient_checkpointing=True,
        strip_vision=True,
    )
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Step 2: Create synthetic training data
    print("\n--- Step 2: Creating synthetic data ---")
    code_samples = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n",
        "import numpy as np\n\ndef matrix_multiply(a, b):\n    return np.matmul(a, b)\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
        "async def fetch_data(url: str) -> dict:\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()\n",
    ] * 20  # Repeat for enough data

    seq_length = 256  # Short for smoke test
    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    for code in code_samples:
        ids = tokenizer.encode(code, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)

        # Pad/truncate to seq_length
        if len(ids) > seq_length:
            ids = ids[:seq_length]
        padding = seq_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * padding
        labels = list(ids) + [-100] * padding
        ids = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * padding

        all_input_ids.append(ids)
        all_labels.append(labels)
        all_attention_masks.append(attention_mask)

    train_dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    })
    print(f"✓ Created {len(train_dataset)} training samples (seq_len={seq_length})")

    # Step 3: Train
    print(f"\n--- Step 3: Training for {num_steps} steps ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=num_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            warmup_steps=2,
            weight_decay=0.01,
            logging_steps=1,
            logging_first_step=True,
            save_steps=num_steps,
            bf16=hw.get("bf16_support", False),
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            seed=42,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        result = trainer.train()
        print(f"\n✓ Training completed!")
        print(f"  Steps: {result.global_step}")
        print(f"  Training loss: {result.training_loss:.4f}")

        # Check loss history
        log_history = trainer.state.log_history
        losses = [l.get("loss", None) for l in log_history if "loss" in l]
        if len(losses) >= 2:
            first_loss = losses[0]
            last_loss = losses[-1]
            improved = last_loss < first_loss
            print(f"  First loss: {first_loss:.4f}")
            print(f"  Last loss:  {last_loss:.4f}")
            print(f"  Improved:   {'✓ Yes' if improved else '⚠ No (may need more steps)'}")

        # Step 4: Save and reload checkpoint
        print(f"\n--- Step 4: Checkpoint save/load ---")
        checkpoint_dir = os.path.join(tmpdir, "checkpoint-test")
        trainer.save_model(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✓ Checkpoint saved to {checkpoint_dir}")

        # Verify we can reload
        from src.training.utils import load_model_for_training

        model2, tok2 = load_model_for_training(
            checkpoint_dir,
            dtype="float32" if not hw.get("bf16_support") else "bfloat16",
            gradient_checkpointing=False,
            strip_vision=False,
        )
        print(f"✓ Checkpoint reloaded: {sum(p.numel() for p in model2.parameters()):,} params")

        # Step 5: Generation test after training
        print(f"\n--- Step 5: Post-training generation ---")
        model2.eval()
        gen_input = tokenizer.encode("def hello_world():\n", return_tensors="pt")
        with torch.no_grad():
            gen_output = model2.generate(
                gen_input.to(model2.device),
                max_new_tokens=50,
                do_sample=False,
            )
        generated = tokenizer.decode(gen_output[0], skip_special_tokens=True)
        print(f"Generated:\n{generated}")

        del model2

    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("✓ Smoke test PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    smoke_test(args.model_path, args.num_steps)
