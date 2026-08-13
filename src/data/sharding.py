"""Dataset sharding — write processed data to efficient file shards.

Creates Arrow or Parquet shard files with metadata manifests
for reproducible, resumable training.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class DatasetSharder:
    """Write processed, packed sequences to shard files.

    Creates numbered shard files with a manifest tracking
    contents and statistics.
    """

    def __init__(
        self,
        output_dir: str,
        shard_size_mb: int = 500,
        output_format: str = "arrow",
        resume: bool = True,
    ):
        """Initialize the sharder.

        Args:
            output_dir: Directory to write shards to.
            shard_size_mb: Target size per shard in MB.
            output_format: "arrow" or "parquet".
            resume: Whether to resume from existing shards.
        """
        self.output_dir = Path(output_dir)
        self.shard_size_mb = shard_size_mb
        self.output_format = output_format
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._current_shard: List[Dict[str, Any]] = []
        self._current_shard_bytes = 0
        self._shard_index = 0
        self._total_sequences = 0
        self._total_tokens = 0
        self._shard_metadata: List[Dict[str, Any]] = []

        # Resume from existing shards
        if resume:
            self._resume_from_manifest()

    def _resume_from_manifest(self):
        """Resume from existing manifest if available."""
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            self._shard_index = manifest.get("num_shards", 0)
            self._total_sequences = manifest.get("total_sequences", 0)
            self._total_tokens = manifest.get("total_tokens", 0)
            self._shard_metadata = manifest.get("shards", [])
            logger.info(
                f"Resuming from shard {self._shard_index} "
                f"({self._total_sequences:,} sequences, "
                f"{self._total_tokens:,} tokens)"
            )

    def add_sequence(self, sequence: Dict[str, Any]) -> Optional[str]:
        """Add a packed sequence to the current shard.

        Args:
            sequence: Dict with input_ids, labels, attention_mask.

        Returns:
            Path to the written shard file if a shard was flushed,
            None otherwise.
        """
        self._current_shard.append(sequence)
        self._total_sequences += 1

        # Estimate size (rough: 4 bytes per int32 token × 3 arrays)
        seq_len = len(sequence.get("input_ids", []))
        self._total_tokens += seq_len
        estimated_bytes = seq_len * 4 * 3  # input_ids + labels + attention_mask
        self._current_shard_bytes += estimated_bytes

        # Check if shard is full
        if self._current_shard_bytes >= self.shard_size_mb * 1024 * 1024:
            return self._flush_shard()

        return None

    def _flush_shard(self) -> str:
        """Write the current shard to disk."""
        if not self._current_shard:
            return ""

        ext = "arrow" if self.output_format == "arrow" else "parquet"
        shard_name = f"shard_{self._shard_index:05d}.{ext}"
        shard_path = self.output_dir / shard_name

        try:
            import pyarrow as pa

            # Convert to columnar format
            columns = {}
            for key in self._current_shard[0].keys():
                columns[key] = [seq[key] for seq in self._current_shard]

            table = pa.table(columns)

            if self.output_format == "parquet":
                import pyarrow.parquet as pq

                pq.write_table(table, str(shard_path))
            else:
                # Arrow IPC format
                writer = pa.ipc.new_file(str(shard_path), table.schema)
                writer.write_table(table)
                writer.close()

        except ImportError:
            # Fallback: save as JSON lines
            shard_name = f"shard_{self._shard_index:05d}.jsonl"
            shard_path = self.output_dir / shard_name
            with open(shard_path, "w", encoding="utf-8") as f:
                for seq in self._current_shard:
                    f.write(json.dumps(seq) + "\n")
            logger.warning(
                "PyArrow not available, falling back to JSONL format. "
                "Install with: pip install pyarrow"
            )

        # Record shard metadata
        shard_meta = {
            "shard_name": shard_name,
            "shard_index": self._shard_index,
            "num_sequences": len(self._current_shard),
            "size_bytes": os.path.getsize(shard_path),
        }
        self._shard_metadata.append(shard_meta)

        logger.info(
            f"Wrote shard {shard_name}: "
            f"{len(self._current_shard)} sequences, "
            f"{os.path.getsize(shard_path) / (1024*1024):.1f} MB"
        )

        # Reset
        self._shard_index += 1
        self._current_shard = []
        self._current_shard_bytes = 0

        # Update manifest
        self._write_manifest()

        return str(shard_path)

    def flush(self) -> Optional[str]:
        """Flush any remaining data as a final shard."""
        if self._current_shard:
            path = self._flush_shard()
            return path
        return None

    def _write_manifest(self):
        """Write the manifest file with all shard metadata."""
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_format": self.output_format,
            "num_shards": self._shard_index,
            "total_sequences": self._total_sequences,
            "total_tokens": self._total_tokens,
            "shards": self._shard_metadata,
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def finalize(self) -> Dict[str, Any]:
        """Finalize the dataset: flush remaining data and return stats."""
        self.flush()
        self._write_manifest()

        stats = {
            "output_dir": str(self.output_dir),
            "num_shards": self._shard_index,
            "total_sequences": self._total_sequences,
            "total_tokens": self._total_tokens,
            "format": self.output_format,
        }

        logger.info(f"Dataset finalized: {stats}")
        return stats
