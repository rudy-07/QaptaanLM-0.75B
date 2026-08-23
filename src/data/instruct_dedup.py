"""Global cross-source deduplication for KapInstruct-100M.

Implements exact SHA-256 deduplication on normalized conversation text across all
sources, tracking duplicate collision pairs, source priorities, and hash persistence.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def compute_normalized_dialogue_hash(messages: List[Dict[str, str]], normalize_whitespace: bool = True) -> str:
    """Compute SHA-256 hash of normalized conversation turns."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user").strip().lower()
        content = msg.get("content", "").strip()
        if normalize_whitespace:
            content = re.sub(r"[ \t]+", " ", content)
            content = re.sub(r"\n{3,}", "\n\n", content)
        parts.append(f"{role}:{content}")
    full_str = "\n---\n".join(parts)
    return hashlib.sha256(full_str.encode("utf-8", errors="replace")).hexdigest()


class GlobalDeduplicator:
    """Cross-source exact SHA-256 deduplicator."""

    def __init__(self, normalize_whitespace: bool = True):
        self.normalize_whitespace = normalize_whitespace
        self.seen_hashes: Dict[str, str] = {}  # hash -> first_seen_source
        self.duplicates_by_source: Dict[str, int] = {}
        self.cross_source_collisions: Dict[str, int] = {}
        self.total_seen = 0
        self.total_duplicates = 0

    def is_duplicate(self, messages: List[Dict[str, str]], source_name: str) -> bool:
        """Check if a conversation is a duplicate across any source.

        Args:
            messages: List of normalized role/content dicts.
            source_name: Name of current dataset source.

        Returns:
            True if duplicate detected, False if unique and registered.
        """
        self.total_seen += 1
        h = compute_normalized_dialogue_hash(messages, self.normalize_whitespace)

        if h in self.seen_hashes:
            first_src = self.seen_hashes[h]
            self.total_duplicates += 1
            self.duplicates_by_source[source_name] = self.duplicates_by_source.get(source_name, 0) + 1
            
            pair_key = f"{first_src} -> {source_name}"
            self.cross_source_collisions[pair_key] = self.cross_source_collisions.get(pair_key, 0) + 1
            return True

        self.seen_hashes[h] = source_name
        return False

    def stats(self) -> Dict[str, Any]:
        """Return comprehensive deduplication metrics."""
        return {
            "total_seen": self.total_seen,
            "unique_records": len(self.seen_hashes),
            "total_duplicates": self.total_duplicates,
            "dedup_rate": f"{self.total_duplicates / max(self.total_seen, 1):.2%}",
            "duplicates_by_source": self.duplicates_by_source,
            "top_cross_source_collisions": dict(
                sorted(self.cross_source_collisions.items(), key=lambda x: -x[1])[:10]
            ),
        }

    def save_hashes(self, file_path: str):
        """Save hashes to file for resuming processing."""
        import json
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.seen_hashes, f)
        logger.info(f"Saved {len(self.seen_hashes)} deduplication hashes to {file_path}")

    def load_hashes(self, file_path: str):
        """Load hashes from file."""
        import json
        from pathlib import Path
        if Path(file_path).exists():
            with open(file_path, "r", encoding="utf-8") as f:
                self.seen_hashes = json.load(f)
            logger.info(f"Loaded {len(self.seen_hashes)} deduplication hashes from {file_path}")
