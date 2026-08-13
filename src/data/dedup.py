"""Deduplication utilities for CPT dataset processing.

Implements:
- Exact deduplication via SHA-256 content hashing
- Normalized deduplication (strip whitespace, normalize)
- Near-deduplication via MinHash/LSH (optional)
"""

import hashlib
import logging
import re
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class ExactDeduplicator:
    """Exact deduplication using content hashing.

    Maintains a set of seen hashes and rejects duplicates.
    Memory usage: ~64 bytes per unique document (SHA-256 hex).
    For 10M docs: ~640MB RAM.
    """

    def __init__(self, normalize: bool = True):
        """Initialize the deduplicator.

        Args:
            normalize: Whether to normalize text before hashing
                      (strip whitespace, lowercase).
        """
        self.normalize = normalize
        self.seen_hashes: Set[str] = set()
        self.total_seen = 0
        self.duplicates_found = 0

    def _compute_hash(self, text: str) -> str:
        """Compute hash of text, optionally after normalization."""
        if self.normalize:
            # Normalize: strip, collapse whitespace, lowercase
            text = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate and register it.

        Args:
            text: Input text to check.

        Returns:
            True if the text is a duplicate, False if it's new.
        """
        self.total_seen += 1
        h = self._compute_hash(text)

        if h in self.seen_hashes:
            self.duplicates_found += 1
            return True

        self.seen_hashes.add(h)
        return False

    def stats(self) -> Dict[str, Any]:
        """Return deduplication statistics."""
        return {
            "total_seen": self.total_seen,
            "unique": len(self.seen_hashes),
            "duplicates_found": self.duplicates_found,
            "dedup_rate": (
                f"{self.duplicates_found / max(self.total_seen, 1):.2%}"
            ),
        }

    def save_hashes(self, path: str) -> None:
        """Save seen hashes to a file for resumability."""
        with open(path, "w", encoding="utf-8") as f:
            for h in sorted(self.seen_hashes):
                f.write(h + "\n")
        logger.info(f"Saved {len(self.seen_hashes)} hashes to {path}")

    def load_hashes(self, path: str) -> None:
        """Load previously seen hashes from a file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    h = line.strip()
                    if h:
                        self.seen_hashes.add(h)
            logger.info(f"Loaded {len(self.seen_hashes)} hashes from {path}")
        except FileNotFoundError:
            logger.info(f"No existing hash file at {path}, starting fresh")


class MinHashDeduplicator:
    """Near-deduplication using MinHash and LSH.

    Uses the datasketch library for efficient approximate deduplication.
    Detects documents that are highly similar but not exactly identical.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.7,
        ngram_size: int = 5,
    ):
        """Initialize MinHash deduplicator.

        Args:
            num_perm: Number of permutations for MinHash.
            threshold: Jaccard similarity threshold for considering
                      documents as near-duplicates.
            ngram_size: Size of character n-grams for shingling.
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.total_seen = 0
        self.duplicates_found = 0
        self._lsh = None
        self._minhashes = {}

        try:
            from datasketch import MinHash, MinHashLSH

            self._lsh = MinHashLSH(
                threshold=threshold,
                num_perm=num_perm,
            )
            self._MinHash = MinHash
            self._available = True
        except ImportError:
            logger.warning(
                "datasketch not installed. MinHash deduplication disabled. "
                "Install with: pip install datasketch"
            )
            self._available = False

    def _create_minhash(self, text: str):
        """Create a MinHash from text using character n-grams."""
        m = self._MinHash(num_perm=self.num_perm)
        # Create shingles (character n-grams)
        text = re.sub(r"\s+", " ", text.strip().lower())
        for i in range(len(text) - self.ngram_size + 1):
            shingle = text[i : i + self.ngram_size]
            m.update(shingle.encode("utf-8"))
        return m

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """Check if text is a near-duplicate.

        Args:
            text: Input text.
            doc_id: Unique identifier for the document.

        Returns:
            True if a near-duplicate exists, False otherwise.
        """
        if not self._available:
            return False

        self.total_seen += 1
        minhash = self._create_minhash(text)

        # Query for similar documents
        result = self._lsh.query(minhash)
        if result:
            self.duplicates_found += 1
            return True

        # Insert into LSH index
        try:
            self._lsh.insert(doc_id, minhash)
        except ValueError:
            # doc_id already exists (shouldn't happen, but be safe)
            pass

        return False

    def stats(self) -> Dict[str, Any]:
        """Return deduplication statistics."""
        return {
            "method": "minhash_lsh",
            "available": self._available,
            "num_perm": self.num_perm,
            "threshold": self.threshold,
            "total_seen": self.total_seen,
            "duplicates_found": self.duplicates_found,
            "dedup_rate": (
                f"{self.duplicates_found / max(self.total_seen, 1):.2%}"
            ),
        }


class DeduplicationPipeline:
    """Combined deduplication pipeline.

    Runs exact dedup first, then optionally MinHash near-dedup.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize from deduplication config.

        Args:
            config: Deduplication section of dataset_config.yaml.
        """
        exact_cfg = config.get("exact", {})
        minhash_cfg = config.get("minhash", {})

        self.exact = None
        self.minhash = None

        if exact_cfg.get("enabled", True):
            self.exact = ExactDeduplicator(
                normalize=exact_cfg.get("normalize", True)
            )

        if minhash_cfg.get("enabled", False):
            self.minhash = MinHashDeduplicator(
                num_perm=minhash_cfg.get("num_perm", 128),
                threshold=minhash_cfg.get("threshold", 0.7),
                ngram_size=minhash_cfg.get("ngram_size", 5),
            )

    def is_duplicate(self, text: str, doc_id: str = "") -> bool:
        """Check if text is a duplicate using all enabled methods.

        Args:
            text: Input text.
            doc_id: Optional document ID for MinHash.

        Returns:
            True if duplicate detected by any method.
        """
        # Exact check first (faster)
        if self.exact and self.exact.is_duplicate(text):
            return True

        # MinHash check (slower, catches near-dupes)
        if self.minhash and doc_id:
            if self.minhash.is_duplicate(text, doc_id):
                return True

        return False

    def stats(self) -> Dict[str, Any]:
        """Return combined deduplication statistics."""
        result = {}
        if self.exact:
            result["exact"] = self.exact.stats()
        if self.minhash:
            result["minhash"] = self.minhash.stats()
        return result
