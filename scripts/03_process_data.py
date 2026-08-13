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

from tqdm.auto import tqdm

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

    pbar = tqdm(desc=f"Streaming & Filtering [{dataset_name}]", unit="doc", leave=False)

    for sample in raw_stream:
        pbar.update(1)
        # Apply filter
        filtered = filter_fn(sample)
        if filtered is None:
            continue

        # Deduplicate
        text = filtered.get("text", "")
        doc_id = f"{dataset_name}:{yielded}"
        if dedup.is_duplicate(text, doc_id):
            continue

        yielded += 1
        pbar.set_postfix({"yielded": f"{yielded:,}", "pass_rate": f"{stats.pass_rate:.1%}"})
        yield filtered

        if max_samples and yielded >= max_samples:
            break

    pbar.close()
    # Final stats
    logger.info(
        f"[{dataset_name}] Final: {yielded:,} samples yielded. "
        f"Filter stats: {stats.summary()}"
    )
    logger.info(f"[{dataset_name}] Dedup stats: {dedup.stats()}")


def process_data(
    max_samples: Optional[int] = None,
    datasets_filter: Optional[list] = None,
    output_dir: Optional[str] = None,
):
    """Run the full data processing pipeline.

    Args:
        max_samples: Max samples per dataset (for testing).
        datasets_filter: Only process these datasets (for testing).
        output_dir: Override output directory.
    """
    start_time = time.time()

    # Load configs
    train_config = load_config("cpt_config.yaml")
    dataset_config = load_dataset_config("dataset_config.yaml")
    env = detect_environment()

    # Setup logging
    log_dir = str(project_root / "logs")
    setup_logging(log_dir=log_dir, log_name="data_processing")

    logger.info("=" * 60)
    logger.info("CPT Data Processing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Environment: {env}")
    logger.info(f"Max samples per dataset: {max_samples or 'unlimited'}")

    # Initialize components
    logger.info("\n--- Initializing components ---")

    # Language detector
    lang_detector = LanguageDetector()

    # Deduplication
    dedup_config = dataset_config.get("deduplication", {})
    dedup = DeduplicationPipeline(dedup_config)

    # Tokenizer
    global_cfg = dataset_config.get("global", {})
    tokenizer_name = global_cfg.get("tokenizer", "Qwen/Qwen3.5-0.8B-Base")
    processing_cfg = dataset_config.get("processing", {})
    seq_length = processing_cfg.get("packing", {}).get("max_seq_length", 4096)
    tokenizer = Tokenizer(tokenizer_name, seq_length)

    # Dataset loader
    loader = DatasetLoader(dataset_config)

    # Determine which datasets to process
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
    # Renormalize if processing subset
    total_prop = sum(proportions.values())
    if total_prop > 0 and abs(total_prop - 1.0) > 0.01:
        proportions = {k: v / total_prop for k, v in proportions.items()}

    target_tokens = global_cfg.get("target_total_tokens", 1_000_000_000)
    if max_samples:
        # Scale down target tokens for testing
        target_tokens = min(target_tokens, max_samples * 500 * len(datasets_to_process))

    mixer = DatasetMixer(
        target_proportions=proportions,
        target_total_tokens=target_tokens,
        seed=global_cfg.get("seed", 42),
    )

    # Mix streams
    logger.info("\n--- Mixing and tokenizing ---")

    def text_stream_from_mixed(mixed_stream):
        """Extract text from mixed stream for packing."""
        for doc in mixed_stream:
            yield doc.get("text", "")

    # Get FIM config
    fim_cfg = datasets_cfg.get("stack_v3_code", {}).get("fim", {})
    fim_rate = fim_cfg.get("rate", 0.5) if fim_cfg.get("enabled") else 0.0

    # Create mixed stream
    mixed = mixer.mix(
        streams=filtered_streams,
        token_count_fn=tokenizer.count_tokens,
    )

    # Tokenize and pack
    packed = create_packed_dataset(
        documents=text_stream_from_mixed(mixed),
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        fim_rate=fim_rate,
        source_type="mixed",
    )

    # Output
    if output_dir is None:
        output_dir = str(project_root / "data" / "processed")

    sharder = DatasetSharder(
        output_dir=output_dir,
        shard_size_mb=processing_cfg.get("shard_size_mb", 500),
        output_format=processing_cfg.get("output_format", "arrow"),
    )

    logger.info(f"Writing shards to {output_dir}")

    pack_pbar = tqdm(desc="Packing & Sharding Sequences", unit="seq")

    for i, sequence in enumerate(packed):
        sharder.add_sequence(sequence)
        pack_pbar.update(1)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            pack_pbar.set_postfix({
                "tokens": f"{sharder._total_tokens:,}",
                "shards": sharder._shard_index,
            })

    pack_pbar.close()

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
    args = parser.parse_args()

    process_data(
        max_samples=args.max_samples,
        datasets_filter=args.datasets,
        output_dir=args.output_dir,
    )
