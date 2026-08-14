"""End-to-end data processing pipeline for CPT.

Orchestrates the full flow:
1. Load dataset streams
2. Apply per-dataset filters
3. Deduplicate
4. Tokenize and pack
5. Mix according to target proportions
6. Write output shards

Usage:
    python scripts/03_process_data.py                    # Full processing
    python scripts/03_process_data.py --max-samples 1000 # Small test
    python scripts/03_process_data.py --datasets stack_v3_code  # Single dataset
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Prevent broken torchvision binary mismatch crashes in Colab/Kaggle
try:
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    _iu.is_torchvision_available = lambda: False
except Exception:
    pass

from src.utils.config import load_config, load_dataset_config, detect_environment
from src.utils.logging_utils import setup_logging, save_json_log
from src.data.loader import DatasetLoader
from src.data.filters import (
    FilterStats,
    LanguageDetector,
    filter_code_sample,
    filter_doc_sample,
    filter_web_sample,
    filter_math_sample,
    filter_vault_sample,
)
from src.data.dedup import DeduplicationPipeline
from src.data.tokenize_and_pack import Tokenizer, create_packed_dataset
from src.data.mixture import DatasetMixer
from src.data.sharding import DatasetSharder

logger = logging.getLogger(__name__)

def build_filtered_stream(
    loader: DatasetLoader,
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
    dedup: DeduplicationPipeline,
    lang_detector: Optional[LanguageDetector] = None,
    max_samples: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Build a filtered, deduplicated stream for a single dataset.

    Args:
        loader: DatasetLoader instance.
        dataset_name: Name of the dataset.
        dataset_cfg: Configuration for this dataset.
        dedup: Deduplication pipeline.
        lang_detector: Optional language detector.
        max_samples: Maximum samples to yield (for testing).

    Yields:
        Filtered, deduplicated samples.
    """
    stats = FilterStats()
    yielded = 0
    consecutive_rejections = 0

    # Select the right loader and filter
    if dataset_name == "stack_v3_code":
        raw_stream = loader.load_stack_v3_code()
        filter_fn = lambda s: filter_code_sample(s, dataset_cfg, stats)
    elif dataset_name == "stack_v3_docs":
        raw_stream = loader.load_stack_v3_docs()
        filter_fn = lambda s: filter_doc_sample(s, dataset_cfg, lang_detector, stats)
    elif dataset_name == "fineweb_hq":
        raw_stream = loader.load_fineweb_hq()
        filter_fn = lambda s: filter_web_sample(s, dataset_cfg, lang_detector, stats)
    elif dataset_name == "openwebmath":
        raw_stream = loader.load_openwebmath()
        filter_fn = lambda s: filter_math_sample(s, dataset_cfg, lang_detector, stats)
    elif dataset_name == "the_vault":
        raw_stream = loader.load_the_vault()
        filter_fn = lambda s: filter_vault_sample(s, dataset_cfg, stats)
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return

    logger.info(f"[{dataset_name}] Stream connection initialized. Reading samples...")

    for sample in raw_stream:
        # Apply filter
        filtered = filter_fn(sample)
        if filtered is None:
            consecutive_rejections += 1
            if consecutive_rejections == 5000:
                logger.warning(
                    f"[{dataset_name}] High rejection alert: 5,000 consecutive items rejected! "
                    f"Current top rejection reasons: {stats.summary().get('rejection_reasons', {})}"
                )
            continue

        # Deduplicate
        text = filtered.get("text", "")
        doc_id = f"{dataset_name}:{yielded}"
        if dedup.is_duplicate(text, doc_id):
            consecutive_rejections += 1
            continue

        consecutive_rejections = 0
        yielded += 1

        if yielded == 1:
            logger.info(f"[{dataset_name}] First document passed filters and deduplication.")

        yield filtered

        if max_samples and yielded >= max_samples:
            break

        # Log filter progress periodically
        if yielded % 10_000 == 0:
            logger.info(
                f"[{dataset_name}] Streamed & yielded {yielded:,} documents | "
                f"Pass rate: {stats.pass_rate:.1%}"
            )

    # Final stats
    logger.info(
        f"[{dataset_name}] Stream finished: {yielded:,} documents yielded. "
        f"Filter stats: {stats.summary()}"
    )
    logger.info(f"[{dataset_name}] Dedup stats: {dedup.stats()}")


def process_data(
    max_samples: Optional[int] = None,
    datasets_filter: Optional[list] = None,
    output_dir: Optional[str] = None,
    target_tokens_override: Optional[int] = None,
    disable_minhash: bool = False,
):
    """Run the full data processing pipeline."""
    start_time = time.time()

    # Load configs
    train_config = load_config("cpt_config.yaml")
    dataset_config = load_dataset_config("dataset_config.yaml")
    env = detect_environment()

    # Setup logging
    log_dir = str(project_root / "logs")
    setup_logging(log_dir=log_dir, log_name="data_processing")

    # Target tokens
    global_cfg = dataset_config.get("global", {})
    target_tokens = target_tokens_override or global_cfg.get("target_total_tokens", 1_000_000_000)
    if max_samples:
        target_tokens = min(target_tokens, max_samples * 500 * (len(datasets_filter) if datasets_filter else 5))

    logger.info("=" * 60)
    logger.info("CPT Data Processing & Token Sharding Pipeline")
    logger.info("=" * 60)
    logger.info(f"Environment: {env}")
    logger.info(f"Target Total Tokens: {target_tokens:,}")
    logger.info(f"Max samples per dataset: {max_samples or 'unlimited'}")

    # Initialize components
    logger.info("\n--- Initializing components ---")

    lang_detector = LanguageDetector()
    dedup_config = dataset_config.get("deduplication", {})
    if disable_minhash and "minhash" in dedup_config:
        dedup_config["minhash"]["enabled"] = False

    dedup = DeduplicationPipeline(dedup_config)

    tokenizer_name = global_cfg.get("tokenizer", "Qwen/Qwen3.5-0.8B-Base")
    processing_cfg = dataset_config.get("processing", {})
    seq_length = processing_cfg.get("packing", {}).get("max_seq_length", 4096)
    tokenizer = Tokenizer(tokenizer_name, seq_length)

    loader = DatasetLoader(dataset_config)

    datasets_cfg = dataset_config.get("datasets", {})
    if datasets_filter:
        datasets_to_process = {
            k: v for k, v in datasets_cfg.items() if k in datasets_filter
        }
    else:
        datasets_to_process = datasets_cfg

    # Build filtered streams
    logger.info("\n--- Building filtered streams ---")
    filtered_streams = {}
    for name, cfg in datasets_to_process.items():
        logger.info(f"Setting up stream: {name}")
        filtered_streams[name] = build_filtered_stream(
            loader=loader,
            dataset_name=name,
            dataset_cfg=cfg,
            dedup=dedup,
            lang_detector=lang_detector,
            max_samples=max_samples,
        )

    # Setup mixer
    proportions = {
        name: cfg.get("target_proportion", 0)
        for name, cfg in datasets_to_process.items()
    }
    total_prop = sum(proportions.values())
    if total_prop > 0 and abs(total_prop - 1.0) > 0.01:
        proportions = {k: v / total_prop for k, v in proportions.items()}

    target_sequences = max(1, target_tokens // seq_length)

    mixer = DatasetMixer(
        target_proportions=proportions,
        target_total_tokens=target_tokens,
        seed=global_cfg.get("seed", 42),
    )

    logger.info("\n--- Mixing, Tokenizing & Sharding ---")
    logger.info(f"Target Total Tokens:    {target_tokens:,}")
    logger.info(f"Target Total Sequences: {target_sequences:,} (seq_len={seq_length})")

    def text_stream_from_mixed(mixed_stream):
        for doc in mixed_stream:
            yield doc.get("text", "")

    fim_cfg = datasets_cfg.get("stack_v3_code", {}).get("fim", {})
    fim_rate = fim_cfg.get("rate", 0.5) if fim_cfg.get("enabled") else 0.0

    # Use fast character estimation in mixer to avoid double-tokenizing every doc
    mixed = mixer.mix(
        streams=filtered_streams,
        token_count_fn=None,
    )

    packed = create_packed_dataset(
        documents=text_stream_from_mixed(mixed),
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        fim_rate=fim_rate,
        source_type="mixed",
    )

    if output_dir is None:
        output_dir = str(project_root / "data" / "processed")

    sharder = DatasetSharder(
        output_dir=output_dir,
        shard_size_mb=processing_cfg.get("shard_size_mb", 50),
        output_format=processing_cfg.get("output_format", "arrow"),
        max_sequences_per_shard=2000,
    )

    logger.info(f"Writing shards to: {output_dir}")

    # Track progress - log frequently for responsive feedback
    log_interval = 25  # Log every 25 sequences (~102,400 tokens)

    for i, sequence in enumerate(packed):
        sharder.add_sequence(sequence)

        current_seq = i + 1
        current_tokens = current_seq * seq_length
        progress_pct = min(100.0, (current_tokens / target_tokens) * 100)

        if current_seq % log_interval == 0 or current_tokens >= target_tokens or current_seq == 1:
            elapsed = time.time() - start_time
            seq_per_sec = current_seq / max(elapsed, 1)
            tok_per_sec = current_tokens / max(elapsed, 1)

            remaining_tokens = max(0, target_tokens - current_tokens)
            eta_sec = remaining_tokens / max(tok_per_sec, 1)
            eta_hrs = eta_sec / 3600

            logger.info(
                f"[Sharding Progress {progress_pct:.1f}%] "
                f"Sequences: {current_seq:,} / {target_sequences:,} | "
                f"Tokens: {current_tokens:,} / {target_tokens:,} | "
                f"Shards: {sharder._shard_index} | "
                f"Speed: {seq_per_sec:.1f} seq/s ({tok_per_sec/1000:.0f}K tok/s) | "
                f"ETA: {eta_hrs:.2f}h"
            )

        if current_tokens >= target_tokens:
            logger.info(f"Reached target tokens ({target_tokens:,}). Finishing sharding.")
            break

    # Finalize
    final_stats = sharder.finalize()
    mixture_report = mixer.get_mixture_report()

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Processing Complete!")
    logger.info("=" * 60)
    logger.info(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    logger.info(f"Shards: {final_stats}")
    logger.info(f"Mixture: {json.dumps(mixture_report, indent=2)}")

    # Save reports
    save_json_log(
        {
            "final_stats": final_stats,
            "mixture_report": mixture_report,
            "dedup_stats": dedup.stats(),
            "elapsed_seconds": elapsed,
        },
        str(Path(output_dir) / "processing_report.json"),
    )

    return final_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CPT datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (for testing)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Target total tokens to shard (e.g. 50000000 for 50M tokens)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Only process these datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--disable-minhash",
        action="store_true",
        help="Force disable MinHash near-deduplication for maximum speed",
    )
    args = parser.parse_args()

    process_data(
        max_samples=args.max_samples,
        datasets_filter=args.datasets,
        output_dir=args.output_dir,
        target_tokens_override=args.target_tokens,
        disable_minhash=args.disable_minhash,
    )
