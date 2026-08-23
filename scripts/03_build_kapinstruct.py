"""End-to-end dataset processing and building pipeline for KapInstruct-100M.

Orchestrates:
1. Streaming multi-source loader for 12 instruction datasets
2. Schema normalization into canonical {id, messages, source, domain, language, code_language, metadata}
3. Domain-specific and quality filtering (English, code languages, secrets, LaTeX, prompt injection)
4. Cross-source global SHA-256 deduplication
5. ChatML chat rendering with token-level assistant-only loss masking
6. Deficit-based exact token quota mixture tracking
7. Sequence packing (4096 tokens) and Arrow IPC sharding
8. Manifest and report generation (manifest.json, mixture_report.json, filter_report.json, licenses.json, DATASET_CARD.md)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Prevent torchvision binary mismatch crashes in Colab/Kaggle
try:
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    _iu.is_torchvision_available = lambda: False
except Exception:
    pass

import yaml
from src.data.instruct_loader import InstructDatasetLoader
from src.data.instruct_normalizer import InstructNormalizer
from src.data.instruct_filters import FilterStats, LanguageDetector, filter_instruction_sample
from src.data.instruct_dedup import GlobalDeduplicator
from src.data.instruct_tokenizer import InstructTokenizer, InstructDocumentPacker
from src.data.instruct_mixture import InstructMixer
from src.data.sharding import DatasetSharder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_kapinstruct")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset_card(
    output_dir: Path,
    manifest: Dict[str, Any],
    mixture_report: Dict[str, Any],
    filter_report: Dict[str, Any],
    licenses: Dict[str, Any],
):
    """Generate Markdown DATASET_CARD.md for KapInstruct-100M."""
    card_content = f"""# KapInstruct-100M Dataset Card

**KapInstruct-100M** is a high-fidelity, multi-source instruction tuning dataset containing **{manifest.get('total_tokens', 0):,} usable content tokens** formatted with Qwen ChatML and tokenized using `Qwen/Qwen3.5-0.8B-Base`.

## Dataset Summary
- **Total Rendered Tokens**: {manifest.get('achieved_rendered_tokens', 0):,}
- **Trainable Assistant Tokens**: {manifest.get('achieved_trainable_tokens', 0):,}
- **Total Documents / Conversations**: {manifest.get('total_documents', 0):,}
- **Sequence Length**: {manifest.get('sequence_length', 4096)}
- **Number of Shards**: {manifest.get('num_shards', 0)}
- **Loss Masking Policy**: `{manifest.get('loss_policy', 'assistant_only')}` (Loss computed strictly on assistant turns; prompt tokens masked to `-100`)

## Mixture Composition & Source-Specific Licenses

| Source | Domain | Share Target | Rendered Tokens | Trainable Tokens | License | Pinned Commit SHA |
|--------|--------|--------------|-----------------|------------------|---------|-------------------|
"""
    for name, s_data in mixture_report.get("sources", {}).items():
        lic = licenses.get(name, {}).get("license", "Unknown")
        sha = licenses.get(name, {}).get("commit_sha", "Unknown")
        card_content += f"| `{name}` | {s_data.get('domain', name)} | {s_data.get('target_share', 0):.0%} | {s_data.get('achieved_rendered_tokens', 0):,} | {s_data.get('achieved_trainable_tokens', 0):,} | {lic} | `{sha[:10]}...` |\n"

    card_content += """
## Licensing and Provenance Notice
Each individual subset in this mixture retains its own upstream license as listed above. Users and researchers must adhere to the individual terms of each constituent source (e.g. CC-BY-4.0 attribution for OpenMathInstruct-2, ODC-By for Tulu-3 / Self-OSS, Apache-2.0, MIT). No single overarching permissive license is claimed over the composite corpus.

## Loading & Inspection Example
```python
import pyarrow as pa
import glob

# Load Arrow shards directly
shard_files = sorted(glob.glob("data/kapinstruct/*.arrow"))
for shard_path in shard_files[:1]:
    reader = pa.ipc.open_file(shard_path)
    table = reader.read_all()
    print(f"Loaded {len(table)} sequences from {shard_path}")
    print("Columns:", table.column_names)
```
"""
    with open(output_dir / "DATASET_CARD.md", "w", encoding="utf-8") as f:
        f.write(card_content)
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(card_content)


def build_kapinstruct(
    config_path: str = "configs/kapinstruct_dataset_config.yaml",
    output_dir: str = "data/kapinstruct",
    target_tokens: Optional[int] = None,
    max_samples: Optional[int] = None,
    loss_policy: Optional[str] = None,
    resume: bool = True,
    allow_redistribution: bool = False,
):
    """Build the complete KapInstruct dataset."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    g_cfg = config.get("global", {})
    sources_cfg = config.get("sources", {})
    filter_cfg = config.get("filters", {})
    dedup_cfg = config.get("deduplication", {})
    proc_cfg = config.get("processing", {})

    total_target = target_tokens or g_cfg.get("target_usable_tokens", 100_000_000)
    loss_pol = loss_policy or g_cfg.get("loss_policy", "assistant_only")
    tokenizer_name = g_cfg.get("tokenizer", "Qwen/Qwen3.5-0.8B-Base")
    max_seq_len = proc_cfg.get("max_seq_length", 4096)
    shard_size = proc_cfg.get("shard_size_mb", 50)
    max_seq_per_shard = proc_cfg.get("max_sequences_per_shard", 2000)

    logger.info("=" * 75)
    logger.info(f"Building KapInstruct Dataset: Target={total_target:,} tokens | Loss Policy={loss_pol}")
    logger.info(f"Output Directory: {out_dir.resolve()}")
    logger.info("=" * 75)

    # Initialize modules
    tokenizer = InstructTokenizer(tokenizer_name, max_seq_length=max_seq_len, loss_policy=loss_pol)
    packer = InstructDocumentPacker(tokenizer, max_seq_length=max_seq_len)
    normalizer = InstructNormalizer()
    lang_detector = LanguageDetector()
    deduplicator = GlobalDeduplicator(normalize_whitespace=dedup_cfg.get("global_exact", {}).get("normalize_whitespace", True))
    mixer = InstructMixer(sources_cfg, target_total_tokens=total_target, seed=g_cfg.get("seed", 42), allow_redistribution=allow_redistribution)
    loader = InstructDatasetLoader(config)
    sharder = DatasetSharder(str(out_dir), shard_size_mb=shard_size, max_sequences_per_shard=max_seq_per_shard, resume=resume)

    # Track filter statistics per source
    filter_stats: Dict[str, FilterStats] = {n: FilterStats() for n in sources_cfg}

    # Initialize streaming iterators
    active_sources = set(sources_cfg.keys())
    stream_iters = {name: iter(loader.load_stream(name)) for name in active_sources}

    sample_counter = 0
    start_time = time.time()

    while active_sources and not mixer.is_total_fulfilled():
        # Select next source by deficit
        source_name = mixer._select_next_source(active_sources)
        if source_name is None:
            break

        s_cfg = sources_cfg[source_name]
        try:
            raw_item = next(stream_iters[source_name])
        except StopIteration:
            logger.info(f"Source [{source_name}] stream exhausted.")
            active_sources.remove(source_name)
            if not allow_redistribution and not mixer.is_source_fulfilled(source_name):
                logger.warning(f"Source [{source_name}] exhausted before meeting target token quota!")
            continue
        except Exception as e:
            logger.warning(f"Stream read error for [{source_name}]: {e}")
            continue

        sample_counter += 1
        if max_samples and sample_counter > max_samples:
            logger.info(f"Reached max_samples limit ({max_samples}). Stopping.")
            break

        # 1. Normalize
        norm_record = normalizer.normalize(
            raw_item,
            source_name=source_name,
            domain=s_cfg.get("domain", ""),
            format_type=s_cfg.get("format_type", "messages"),
            sample_idx=sample_counter,
        )
        if norm_record is None:
            filter_stats[source_name].record_reject("malformed_schema")
            continue

        # 2. Filter
        passed_filter = filter_instruction_sample(
            norm_record,
            config=filter_cfg,
            lang_detector=lang_detector,
            stats=filter_stats[source_name],
            source_cfg=s_cfg,
        )
        if passed_filter is None:
            continue

        # 3. Global Cross-Source Deduplication
        if deduplicator.is_duplicate(norm_record["messages"], source_name):
            filter_stats[source_name].record_reject("global_duplicate")
            continue

        # 4. Tokenize & Mask Loss
        input_ids, labels, rend_tokens, train_tokens = tokenizer.render_and_tokenize_conversation(
            norm_record["messages"]
        )

        if rend_tokens < 5:
            filter_stats[source_name].record_reject("too_few_tokens")
            continue

        # 5. Record Token Quota in Mixer
        mixer.record_sample(source_name, rendered_tokens=rend_tokens, trainable_tokens=train_tokens)

        # Check if source fulfilled
        if mixer.is_source_fulfilled(source_name):
            logger.info(f"Source [{source_name}] reached its quota ({mixer.actual_rendered_tokens[source_name]:,} tokens).")
            active_sources.discard(source_name)

        # 6. Pack Document & Shard
        packed_seq = packer.add_conversation(input_ids, labels, rend_tokens, train_tokens)
        if packed_seq:
            sharder.add_sequence(packed_seq)

        # Progress logging
        if sample_counter % 2000 == 0 or mixer.is_total_fulfilled():
            tot_rend = sum(mixer.actual_rendered_tokens.values())
            pct = tot_rend / max(total_target, 1)
            logger.info(
                f"Progress: {tot_rend:,} / {total_target:,} tokens ({pct:.1%}) | "
                f"Docs: {sum(mixer.actual_docs.values()):,} | "
                f"Shards: {sharder._shard_index} | "
                f"Rate: {tot_rend / max(time.time() - start_time, 1):.0f} tok/s"
            )

    # Flush remaining packed sequence
    final_seq = packer.flush()
    if final_seq:
        sharder.add_sequence(final_seq)

    # Finalize shards
    shard_stats = sharder.finalize()

    # Generate Checksums for all shards
    shard_files = sorted(out_dir.glob("shard_*.arrow")) + sorted(out_dir.glob("shard_*.parquet"))
    checksums = {f.name: compute_file_sha256(f) for f in shard_files}

    # Generate Reports
    mixture_rep = mixer.get_mixture_report()
    filter_rep = {n: filter_stats[n].summary() for n in sources_cfg}
    dedup_rep = deduplicator.stats()

    licenses = {
        name: {
            "license": s_cfg.get("license", "Unknown"),
            "source_id": s_cfg.get("source_id"),
            "commit_sha": s_cfg.get("commit_sha", "unpinned"),
        }
        for name, s_cfg in sources_cfg.items()
    }

    manifest = {
        "dataset_name": g_cfg.get("name", "KapInstruct-100M"),
        "version": g_cfg.get("version", "1.0.0"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": tokenizer_name,
        "sequence_length": max_seq_len,
        "loss_policy": loss_pol,
        "target_tokens": total_target,
        "achieved_rendered_tokens": mixture_rep["achieved_rendered_tokens"],
        "achieved_trainable_tokens": mixture_rep["achieved_trainable_tokens"],
        "total_documents": mixture_rep["total_documents"],
        "num_shards": shard_stats.get("num_shards", 0),
        "total_sequences": shard_stats.get("total_sequences", 0),
        "packing_stats": packer.stats(),
        "shard_checksums": checksums,
        "source_licenses": licenses,
    }

    # Save JSON reports
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "mixture_report.json", "w", encoding="utf-8") as f:
        json.dump(mixture_rep, f, indent=2)
    with open(out_dir / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump(filter_rep, f, indent=2)
    with open(out_dir / "licenses.json", "w", encoding="utf-8") as f:
        json.dump(licenses, f, indent=2)
    with open(out_dir / "dedup_report.json", "w", encoding="utf-8") as f:
        json.dump(dedup_rep, f, indent=2)

    # Generate Markdown documentation
    generate_dataset_card(out_dir, manifest, mixture_rep, filter_rep, licenses)

    logger.info("=" * 75)
    logger.info(f"KapInstruct-100M build complete!")
    logger.info(f"Rendered tokens:  {mixture_rep['achieved_rendered_tokens']:,}")
    logger.info(f"Trainable tokens: {mixture_rep['achieved_trainable_tokens']:,}")
    logger.info(f"Shards written:   {shard_stats.get('num_shards', 0)} to {out_dir}")
    logger.info("=" * 75)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KapInstruct-100M instruction tuning dataset")
    parser.add_argument("--config", type=str, default="configs/kapinstruct_dataset_config.yaml")
    parser.add_argument("--output-dir", type=str, default="data/kapinstruct")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--loss-policy", type=str, default=None, choices=["assistant_only", "all_tokens"])
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--allow-redistribution", action="store_true", default=False)
    args = parser.parse_args()

    build_kapinstruct(
        config_path=args.config,
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        max_samples=args.max_samples,
        loss_policy=args.loss_policy,
        resume=args.resume,
        allow_redistribution=args.allow_redistribution,
    )
