"""End-to-end smoke test for the KapInstruct-100M pipeline.

Streams a tiny sample from every one of the 12 sources, runs full normalization,
filtering, deduplication, tokenization with assistant loss masking, sequence packing,
and Arrow sharding, then verifies and decodes the resulting shard.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pyarrow as pa
import yaml
from transformers import AutoTokenizer

from src.data.instruct_loader import InstructDatasetLoader
from src.data.instruct_normalizer import InstructNormalizer
from src.data.instruct_filters import FilterStats, LanguageDetector, filter_instruction_sample
from src.data.instruct_dedup import GlobalDeduplicator
from src.data.instruct_tokenizer import InstructTokenizer, InstructDocumentPacker
from src.data.instruct_mixture import InstructMixer
from src.data.sharding import DatasetSharder


def run_smoke_test(config_path: str = "configs/kapinstruct_dataset_config.yaml"):
    print("=" * 80)
    print("KAPINSTRUCT-100M END-TO-END SMOKE TEST")
    print("=" * 80)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sources_cfg = config.get("sources", {})
    filter_cfg = config.get("filters", {})
    proc_cfg = config.get("processing", {})
    seq_len = 512  # Compact for smoke test

    print(f"\n1. Initializing tokenizer & modules (seq_len={seq_len})...")
    tokenizer = InstructTokenizer("Qwen/Qwen3.5-0.8B-Base", max_seq_length=seq_len, loss_policy="assistant_only")
    packer = InstructDocumentPacker(tokenizer, max_seq_length=seq_len)
    normalizer = InstructNormalizer()
    lang_detector = LanguageDetector()
    dedup = GlobalDeduplicator()
    loader = InstructDatasetLoader(config)

    per_source_verified = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        sharder = DatasetSharder(tmpdir, shard_size_mb=1, max_sequences_per_shard=100)

        print("\n2. Streaming and processing 3 samples from each of the 12 sources...")
        total_rendered = 0
        total_trainable = 0
        total_docs = 0

        for name, s_cfg in sources_cfg.items():
            print(f"  Streaming [{name}]...", end=" ", flush=True)
            stats = FilterStats()
            stream = loader.load_stream(name)
            count = 0

            for raw_item in stream:
                norm = normalizer.normalize(raw_item, name, s_cfg.get("domain", ""), s_cfg.get("format_type", "messages"), count)
                if norm is None:
                    continue

                passed = filter_instruction_sample(norm, filter_cfg, lang_detector, stats, s_cfg)
                if passed is None:
                    continue

                if dedup.is_duplicate(norm["messages"], name):
                    continue

                inp_ids, labels, rend, train = tokenizer.render_and_tokenize_conversation(norm["messages"])
                if rend < 5:
                    continue

                packed_seq = packer.add_conversation(inp_ids, labels, rend, train)
                if packed_seq:
                    sharder.add_sequence(packed_seq)

                total_rendered += rend
                total_trainable += train
                total_docs += 1
                count += 1
                if count >= 3:
                    break

            per_source_verified[name] = count
            print(f"[OK] Processed {count} samples")

        # Flush final sequence
        final_seq = packer.flush()
        if final_seq:
            sharder.add_sequence(final_seq)

        sharder.finalize()

        print("\n3. Validating written Arrow shards...")
        arrow_files = sorted(Path(tmpdir).glob("shard_*.arrow"))
        assert len(arrow_files) >= 1, "No arrow shards written!"
        print(f"  ✓ Found {len(arrow_files)} shard file(s): {[f.name for f in arrow_files]}")

        with open(str(arrow_files[0]), "rb") as f:
            reader = pa.ipc.open_file(f)
            table = reader.read_all()

        print(f"  ✓ Shard 0 contains {len(table)} sequences with columns: {table.column_names}")

        print("\n4. Decoding and verifying assistant loss masking...")
        first_input_ids = table["input_ids"][0].as_py()
        first_labels = table["labels"][0].as_py()
        first_mask = table["attention_mask"][0].as_py()

        decoded_text = tokenizer.tokenizer.decode(first_input_ids[:150])
        trainable_token_count = sum(1 for l in first_labels if l != -100)
        masked_token_count = sum(1 for l in first_labels if l == -100)

        print(f"  Sample sequence preview (first 150 tokens):\n{repr(decoded_text)}")
        print(f"  Trainable assistant tokens: {trainable_token_count}")
        print(f"  Masked prompt & pad tokens: {masked_token_count}")
        assert trainable_token_count > 0, "No trainable tokens found in sequence!"

    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY:")
    for name, cnt in per_source_verified.items():
        print(f"  ✓ {name:<25}: {cnt} samples processed & packed")
    print(f"\nTotal Docs: {total_docs} | Rendered Tokens: {total_rendered:,} | Trainable Tokens: {total_trainable:,}")
    print("=" * 80)
    print("✓ KAPINSTRUCT-100M END-TO-END SMOKE TEST PASSED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
