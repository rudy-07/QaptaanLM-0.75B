"""Dataset loading and streaming for all CPT data sources.

Provides a unified interface for loading and iterating over:
- HuggingFaceCode/stack-v3-train (source code)
- HuggingFaceCode/stack-v3-train (documentation)
- epfml/FineWeb-HQ (web text)
- open-web-math/open-web-math (math)
- Fsoft-AIC/the-vault-function (function-level code)
"""

import logging
from typing import Any, Dict, Generator, List, Optional, Set

from datasets import load_dataset, IterableDataset

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Unified loader for all CPT datasets.

    Handles the different schemas and access patterns of each dataset,
    providing a consistent stream of (text, metadata) tuples.
    """

    def __init__(self, dataset_config: Dict[str, Any]):
        """Initialize with dataset configuration.

        Args:
            dataset_config: Full dataset config dict from dataset_config.yaml.
        """
        self.config = dataset_config
        self.datasets_cfg = dataset_config.get("datasets", {})

    def load_stack_v3_code(self) -> Generator[Dict[str, Any], None, None]:
        """Stream source code files from Stack v3.

        Stack v3 stores data at the repository level. Each row contains
        a `files` array. We unpack this to individual file records and
        filter by target programming languages.

        Yields:
            Dict with keys: text, language, file_path, source, metadata
        """
        cfg = self.datasets_cfg.get("stack_v3_code", {})
        if not cfg:
            logger.warning("stack_v3_code not configured, skipping")
            return

        target_langs = set(cfg.get("target_languages", []))
        repo_filters = cfg.get("repo_filters", {})
        exclude_forks = repo_filters.get("exclude_forks", True)
        exclude_vendor = repo_filters.get("exclude_vendor", True)

        logger.info(
            f"Loading Stack v3 code — target languages: {sorted(target_langs)}"
        )

        ds = load_dataset(
            cfg["source"],
            split=cfg.get("split", "train"),
            streaming=cfg.get("streaming", True),
        )

        for repo in ds:
            # Repository-level filtering
            gh_meta = repo.get("github_metadata", {}) or {}
            if exclude_forks and gh_meta.get("is_fork", False):
                continue

            files = repo.get("files", [])
            if not files:
                continue

            repo_path = repo.get("repo_path", "")

            for file_info in files:
                lang = file_info.get("language", "")
                if lang not in target_langs:
                    continue

                if exclude_vendor and file_info.get("is_vendor", False):
                    continue

                content = file_info.get("content")
                if not content:
                    continue

                file_path = file_info.get("file_path", "")
                size_bytes = file_info.get("size_bytes", 0)

                yield {
                    "text": content,
                    "language": lang,
                    "file_path": file_path,
                    "repo_path": repo_path,
                    "size_bytes": size_bytes,
                    "stars": gh_meta.get("stars", 0),
                    "source": "stack_v3_code",
                    "license_type": file_info.get("license_type", ""),
                }

    def load_stack_v3_docs(self) -> Generator[Dict[str, Any], None, None]:
        """Stream documentation files from Stack v3.

        Extracts .md, .rst, README, and other doc files from repos.

        Yields:
            Dict with keys: text, file_path, source, metadata
        """
        cfg = self.datasets_cfg.get("stack_v3_docs", {})
        if not cfg:
            logger.warning("stack_v3_docs not configured, skipping")
            return

        doc_extensions = set(cfg.get("doc_extensions", [".md", ".rst", ".txt"]))
        doc_patterns = cfg.get("doc_filename_patterns", [])
        doc_pattern_lower = [p.lower().rstrip("*") for p in doc_patterns]
        exclude_patterns = cfg.get("doc_filters", {}).get("exclude_patterns", [])
        exclude_lower = [p.lower().rstrip("*") for p in exclude_patterns]

        logger.info(f"Loading Stack v3 docs — extensions: {doc_extensions}")

        ds = load_dataset(
            cfg["source"],
            split=cfg.get("split", "train"),
            streaming=cfg.get("streaming", True),
        )

        for repo in ds:
            files = repo.get("files", [])
            if not files:
                continue

            repo_path = repo.get("repo_path", "")

            for file_info in files:
                file_path = file_info.get("file_path", "")
                file_path_lower = file_path.lower()

                # Check if it's a documentation file
                is_doc = False

                # Check extension
                for ext in doc_extensions:
                    if file_path_lower.endswith(ext):
                        is_doc = True
                        break

                # Check filename patterns
                if not is_doc:
                    basename = file_path_lower.rsplit("/", 1)[-1] if "/" in file_path_lower else file_path_lower
                    for pattern in doc_pattern_lower:
                        if basename.startswith(pattern) or pattern in file_path_lower:
                            is_doc = True
                            break

                if not is_doc:
                    continue

                # Exclude license/legal files
                basename = file_path_lower.rsplit("/", 1)[-1] if "/" in file_path_lower else file_path_lower
                skip = False
                for excl in exclude_lower:
                    if basename.startswith(excl):
                        skip = True
                        break
                if skip:
                    continue

                content = file_info.get("content")
                if not content:
                    continue

                yield {
                    "text": content,
                    "file_path": file_path,
                    "repo_path": repo_path,
                    "source": "stack_v3_docs",
                    "size_bytes": file_info.get("size_bytes", 0),
                }

    def load_fineweb_hq(self) -> Generator[Dict[str, Any], None, None]:
        """Stream documents from FineWeb-HQ.

        Yields:
            Dict with keys: text, url, quality_score, source
        """
        cfg = self.datasets_cfg.get("fineweb_hq", {})
        if not cfg:
            logger.warning("fineweb_hq not configured, skipping")
            return

        logger.info("Loading FineWeb-HQ")

        ds = load_dataset(
            cfg["source"],
            split=cfg.get("split", "train"),
            streaming=cfg.get("streaming", True),
        )

        for sample in ds:
            text = sample.get("text", "")
            if not text:
                continue

            yield {
                "text": text,
                "url": sample.get("url", ""),
                "quality_score": sample.get("quality_score", None),
                "source": "fineweb_hq",
            }

    def load_openwebmath(self) -> Generator[Dict[str, Any], None, None]:
        """Stream documents from OpenWebMath.

        Yields:
            Dict with keys: text, url, source, metadata
        """
        cfg = self.datasets_cfg.get("openwebmath", {})
        if not cfg:
            logger.warning("openwebmath not configured, skipping")
            return

        logger.info("Loading OpenWebMath")

        ds = load_dataset(
            cfg["source"],
            split=cfg.get("split", "train"),
            streaming=cfg.get("streaming", True),
        )

        for sample in ds:
            text = sample.get("text", "")
            if not text:
                continue

            yield {
                "text": text,
                "url": sample.get("url", ""),
                "source": "openwebmath",
                "metadata": sample.get("metadata", {}),
            }

    def load_the_vault(self) -> Generator[Dict[str, Any], None, None]:
        """Stream function-level code from The Vault.

        The Vault provides function-level code with identifiers and docstrings.

        Yields:
            Dict with keys: text, language, identifier, source
        """
        cfg = self.datasets_cfg.get("the_vault", {})
        if not cfg:
            logger.warning("the_vault not configured, skipping")
            return

        target_langs = set(cfg.get("target_languages", []))
        include_docstring = cfg.get("include_docstring", True)
        include_identifier = cfg.get("include_identifier", True)
        split_set = cfg.get("split_set", ["train"])

        logger.info(
            f"Loading The Vault — languages: {sorted(target_langs)}"
        )

        try:
            # Direct parquet data_files streaming (compatible with datasets >= 3.0.0)
            ds = load_dataset(
                "parquet",
                data_files="hf://datasets/Fsoft-AIC/the-vault-function/data/train/full/*.parquet",
                split="train",
                streaming=cfg.get("streaming", True),
            )
        except Exception as e:
            logger.warning(
                f"Direct parquet load failed ({e}), attempting default load_dataset"
            )
            ds = load_dataset(
                cfg["source"],
                split="train",
                streaming=cfg.get("streaming", True),
            )

        yield from self._iter_vault_split(
            ds, target_langs, include_docstring, include_identifier
        )

    def _iter_vault_split(
        self,
        ds,
        target_langs: Set[str],
        include_docstring: bool,
        include_identifier: bool,
    ) -> Generator[Dict[str, Any], None, None]:
        """Iterate over a single Vault split."""
        for sample in ds:
            lang = sample.get("language", "")
            if target_langs and lang not in target_langs:
                continue

            code = sample.get("code", "")
            if not code:
                continue

            # Build the text representation
            parts = []
            if include_identifier and sample.get("identifier"):
                parts.append(f"# Function: {sample['identifier']}")
            if include_docstring and sample.get("docstring"):
                parts.append(f'"""{sample["docstring"]}"""')
            parts.append(code)

            text = "\n".join(parts)

            yield {
                "text": text,
                "language": lang,
                "identifier": sample.get("identifier", ""),
                "path": sample.get("path", ""),
                "source": "the_vault",
            }

    def stream_all(self) -> Dict[str, Generator[Dict[str, Any], None, None]]:
        """Get generators for all configured datasets.

        Returns:
            Dict mapping dataset name to its generator.
        """
        streams = {}

        if "stack_v3_code" in self.datasets_cfg:
            streams["stack_v3_code"] = self.load_stack_v3_code()
        if "stack_v3_docs" in self.datasets_cfg:
            streams["stack_v3_docs"] = self.load_stack_v3_docs()
        if "fineweb_hq" in self.datasets_cfg:
            streams["fineweb_hq"] = self.load_fineweb_hq()
        if "openwebmath" in self.datasets_cfg:
            streams["openwebmath"] = self.load_openwebmath()
        if "the_vault" in self.datasets_cfg:
            streams["the_vault"] = self.load_the_vault()

        return streams
