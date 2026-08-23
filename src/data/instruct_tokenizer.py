"""Tokenization, ChatML rendering, and assistant-only loss masking for KapInstruct-100M.

Handles:
- Applying Qwen chat template or deterministic ChatML formatting
- Precise token-level assistant loss masking (prompt tokens = -100, assistant tokens = token_id)
- Multi-turn conversation packing into fixed-length sequences (4096 tokens)
- Separate metric tracking: rendered tokens vs trainable assistant tokens vs padding tokens
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InstructTokenizer:
    """Wrapper around Qwen AutoTokenizer with instruction formatting and loss masking."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base",
        max_seq_length: int = 4096,
        loss_policy: str = "assistant_only",
    ):
        """Initialize the tokenizer and special token IDs.

        Args:
            model_name_or_path: Hugging Face model identifier or local path.
            max_seq_length: Maximum packed sequence length (default: 4096).
            loss_policy: 'assistant_only' (recommended for SFT) or 'all_tokens'.
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
        )
        self.max_seq_length = max_seq_length
        self.loss_policy = loss_policy

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token_id

        # Cache ChatML header token IDs
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        logger.info(
            f"InstructTokenizer loaded: vocab={self.tokenizer.vocab_size}, "
            f"eos_id={self.eos_token_id}, loss_policy={self.loss_policy}"
        )

    def render_and_tokenize_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[List[int], List[int], int, int]:
        """Render a conversation to token IDs and corresponding labels.

        Args:
            messages: List of {'role': 'system'|'user'|'assistant', 'content': '...'}

        Returns:
            Tuple of:
            - input_ids: List[int]
            - labels: List[int] (-100 for non-trainable, token_id for trainable)
            - rendered_tokens_count: Total rendered tokens
            - trainable_assistant_tokens_count: Number of assistant loss tokens
        """
        input_ids: List[int] = []
        labels: List[int] = []
        trainable_count = 0

        # Construct turns with deterministic ChatML formatting
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            turn_header = f"<|im_start|>{role}\n"
            turn_body = f"{content}<|im_end|>\n"

            header_ids = self.tokenizer.encode(turn_header, add_special_tokens=False)
            body_ids = self.tokenizer.encode(turn_body, add_special_tokens=False)

            turn_ids = header_ids + body_ids

            if self.loss_policy == "assistant_only":
                if role == "assistant":
                    # Header is not loss-trained
                    turn_labels = [-100] * len(header_ids)
                    # Assistant body tokens are loss-trained (including <|im_end|>)
                    turn_labels += list(body_ids)
                    trainable_count += len(body_ids)
                else:
                    # System and user turns are completely masked
                    turn_labels = [-100] * len(turn_ids)
            else:  # all_tokens
                turn_labels = list(turn_ids)
                trainable_count += len(turn_ids)

            input_ids.extend(turn_ids)
            labels.extend(turn_labels)

        rendered_count = len(input_ids)
        return input_ids, labels, rendered_count, trainable_count


class InstructDocumentPacker:
    """Pack multiple tokenized conversations into fixed-length 4096-token sequences."""

    def __init__(
        self,
        tokenizer: InstructTokenizer,
        max_seq_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self._buffer_input_ids: List[int] = []
        self._buffer_labels: List[int] = []

        # Metrics
        self.total_sequences = 0
        self.total_rendered_tokens = 0
        self.total_trainable_tokens = 0
        self.total_padding_tokens = 0
        self.total_conversations_packed = 0

    def add_conversation(
        self,
        input_ids: List[int],
        labels: List[int],
        rendered_count: int,
        trainable_count: int,
    ) -> Optional[Dict[str, Any]]:
        """Add a tokenized conversation to the sequence packing buffer."""
        # Truncate single oversized conversations
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

        emitted = None
        # Check if buffer would overflow
        if len(self._buffer_input_ids) + len(input_ids) > self.max_seq_length:
            emitted = self._emit_sequence()
            self._buffer_input_ids = list(input_ids)
            self._buffer_labels = list(labels)
            self.total_conversations_packed += 1
        else:
            self._buffer_input_ids.extend(input_ids)
            self._buffer_labels.extend(labels)
            self.total_conversations_packed += 1

            if len(self._buffer_input_ids) >= self.max_seq_length:
                emitted = self._emit_sequence()

        self.total_rendered_tokens += rendered_count
        self.total_trainable_tokens += trainable_count
        return emitted

    def flush(self) -> Optional[Dict[str, Any]]:
        """Flush any remaining tokens in buffer as a padded sequence."""
        if self._buffer_input_ids:
            return self._emit_sequence()
        return None

    def _emit_sequence(self) -> Dict[str, Any]:
        """Create a complete padded sequence."""
        seq_len = len(self._buffer_input_ids)
        padding_len = self.max_seq_length - seq_len

        input_ids = self._buffer_input_ids[:self.max_seq_length]
        labels = self._buffer_labels[:self.max_seq_length]

        if padding_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len
            attention_mask = [1] * seq_len + [0] * padding_len
            self.total_padding_tokens += padding_len
        else:
            attention_mask = [1] * self.max_seq_length

        self.total_sequences += 1
        self._buffer_input_ids = []
        self._buffer_labels = []

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def stats(self) -> Dict[str, Any]:
        """Return packing statistics."""
        total_capacity = self.total_sequences * self.max_seq_length
        return {
            "total_sequences": self.total_sequences,
            "total_conversations": self.total_conversations_packed,
            "rendered_tokens": self.total_rendered_tokens,
            "trainable_assistant_tokens": self.total_trainable_tokens,
            "padding_tokens": self.total_padding_tokens,
            "sequence_capacity": total_capacity,
            "padding_ratio": f"{self.total_padding_tokens / max(total_capacity, 1):.2%}",
        }
