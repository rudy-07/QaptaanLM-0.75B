"""Tokenization and document packing for CPT training.

Handles:
- Tokenization with Qwen3.5 tokenizer
- Document packing (multiple docs per sequence with EOS separators)
- Attention mask creation for packed sequences
- FIM (Fill-in-the-Middle) formatting for code samples
- Token counting and statistics
"""

import logging
import random
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Tokenizer:
    """Wrapper around HuggingFace tokenizer with CPT-specific utilities."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base",
        max_seq_length: int = 4096,
    ):
        """Initialize the tokenizer.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            max_seq_length: Maximum sequence length for packing.
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
        )
        self.max_seq_length = max_seq_length

        # Cache special token IDs
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

        # FIM token IDs
        self.fim_prefix_id = self.tokenizer.convert_tokens_to_ids("<|fim_prefix|>")
        self.fim_middle_id = self.tokenizer.convert_tokens_to_ids("<|fim_middle|>")
        self.fim_suffix_id = self.tokenizer.convert_tokens_to_ids("<|fim_suffix|>")

        logger.info(
            f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}, "
            f"eos_id={self.eos_token_id}, max_seq_length={max_seq_length}"
        )

    def tokenize(self, text: str, add_eos: bool = True) -> List[int]:
        """Tokenize text to token IDs.

        Args:
            text: Input text.
            add_eos: Whether to append EOS token.

        Returns:
            List of token IDs.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def count_tokens(self, text: str) -> int:
        """Count tokens in text without storing them."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def format_fim(self, code: str, rng: Optional[random.Random] = None) -> str:
        """Format code as Fill-in-the-Middle (FIM).

        Splits code at a random point and rearranges as:
        <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>middle

        This teaches the model to do code completion/infilling.

        Args:
            code: Source code text.
            rng: Random number generator for reproducibility.

        Returns:
            FIM-formatted text string.
        """
        if rng is None:
            rng = random.Random()

        lines = code.split("\n")
        if len(lines) < 3:
            return code  # Too short for FIM

        # Choose a random split point (not at the very start or end)
        split_line = rng.randint(1, max(1, len(lines) - 2))
        prefix_lines = lines[:split_line]
        # Choose how many lines to "mask" (1-5 lines)
        mask_len = rng.randint(1, min(5, len(lines) - split_line - 1))
        middle_lines = lines[split_line : split_line + mask_len]
        suffix_lines = lines[split_line + mask_len :]

        prefix = "\n".join(prefix_lines)
        middle = "\n".join(middle_lines)
        suffix = "\n".join(suffix_lines)

        return (
            f"<|fim_prefix|>{prefix}"
            f"<|fim_suffix|>{suffix}"
            f"<|fim_middle|>{middle}"
        )


class DocumentPacker:
    """Pack multiple documents into fixed-length sequences.

    This is essential for efficient training — instead of padding
    short documents to max_seq_length, we concatenate multiple docs
    with EOS separators and create attention masks that prevent
    cross-document attention.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_length: int = 4096,
        create_attention_mask: bool = True,
    ):
        """Initialize the packer.

        Args:
            tokenizer: Tokenizer instance.
            max_seq_length: Target sequence length.
            create_attention_mask: Whether to create document-boundary
                                 attention masks.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.create_attention_mask = create_attention_mask

        # Current buffer
        self._buffer: List[int] = []
        self._doc_boundaries: List[int] = []  # Start indices of each doc
        self._current_doc_start = 0

        # Statistics
        self.total_tokens_packed = 0
        self.total_sequences_created = 0
        self.total_docs_packed = 0
        self.total_padding_tokens = 0

    def add_document(self, token_ids: List[int]) -> Optional[Dict[str, Any]]:
        """Add a tokenized document to the packing buffer.

        If adding this document fills or exceeds max_seq_length,
        returns a complete packed sequence. Otherwise returns None.

        Args:
            token_ids: Token IDs for one document (including EOS).

        Returns:
            A packed sequence dict if buffer is full, None otherwise.
        """
        # If the document alone exceeds max_seq_length, truncate it
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[: self.max_seq_length - 1] + [
                self.tokenizer.eos_token_id
            ]

        # Check if adding this doc would exceed the limit
        if len(self._buffer) + len(token_ids) > self.max_seq_length:
            # Emit current buffer as a sequence
            result = self._emit_sequence()

            # Start new buffer with this document
            self._buffer = list(token_ids)
            self._doc_boundaries = [0]
            self._current_doc_start = 0
            self.total_docs_packed += 1

            return result

        # Add to buffer
        self._doc_boundaries.append(len(self._buffer))
        self._buffer.extend(token_ids)
        self.total_docs_packed += 1

        # Check if buffer is now full
        if len(self._buffer) >= self.max_seq_length:
            return self._emit_sequence()

        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        """Flush any remaining documents in the buffer.

        Returns:
            A packed sequence dict if buffer is non-empty, None otherwise.
        """
        if self._buffer:
            return self._emit_sequence()
        return None

    def _emit_sequence(self) -> Dict[str, Any]:
        """Create a packed sequence from the current buffer."""
        input_ids = self._buffer[: self.max_seq_length]

        # Pad if necessary
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            self.total_padding_tokens += padding_length

        # Create labels (same as input_ids for CLM, but mask padding)
        labels = list(input_ids)
        if padding_length > 0:
            for i in range(len(labels) - padding_length, len(labels)):
                labels[i] = -100  # Ignore padding in loss

        result = {
            "input_ids": input_ids,
            "labels": labels,
        }

        # Create attention mask
        if self.create_attention_mask:
            # Simple attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1] * (self.max_seq_length - padding_length) + [
                0
            ] * padding_length
            result["attention_mask"] = attention_mask

        self.total_tokens_packed += self.max_seq_length - padding_length
        self.total_sequences_created += 1

        # Reset buffer
        self._buffer = []
        self._doc_boundaries = []

        return result

    def stats(self) -> Dict[str, Any]:
        """Return packing statistics."""
        total_tokens = self.total_tokens_packed + self.total_padding_tokens
        return {
            "total_sequences": self.total_sequences_created,
            "total_docs_packed": self.total_docs_packed,
            "total_tokens_packed": self.total_tokens_packed,
            "total_padding_tokens": self.total_padding_tokens,
            "padding_ratio": (
                f"{self.total_padding_tokens / max(total_tokens, 1):.2%}"
            ),
            "avg_docs_per_seq": (
                f"{self.total_docs_packed / max(self.total_sequences_created, 1):.1f}"
            ),
        }


def create_packed_dataset(
    documents: Generator[str, None, None],
    tokenizer: Tokenizer,
    max_seq_length: int = 4096,
    fim_rate: float = 0.0,
    fim_seed: int = 42,
    source_type: str = "code",
) -> Generator[Dict[str, Any], None, None]:
    """Create a packed dataset from a stream of documents.

    Args:
        documents: Generator yielding document text strings.
        tokenizer: Tokenizer instance.
        max_seq_length: Target sequence length.
        fim_rate: Rate at which to apply FIM formatting (0.0-1.0).
                 Only applied to code source_type.
        fim_seed: Seed for FIM randomization.
        source_type: Type of source ("code", "docs", "web", "math").

    Yields:
        Packed sequence dicts with input_ids, labels, attention_mask.
    """
    packer = DocumentPacker(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    fim_rng = random.Random(fim_seed)
    apply_fim = fim_rate > 0.0 and source_type == "code"

    for text in documents:
        # Optionally apply FIM formatting
        if apply_fim and fim_rng.random() < fim_rate:
            text = tokenizer.format_fim(text, rng=fim_rng)

        # Tokenize
        token_ids = tokenizer.tokenize(text, add_eos=True)

        # Skip empty or too-short documents
        if len(token_ids) < 5:
            continue

        # Add to packer
        result = packer.add_document(token_ids)
        if result is not None:
            yield result

    # Flush remaining
    final = packer.flush()
    if final is not None:
        yield final

    logger.info(f"Packing stats: {packer.stats()}")
