"""Views over packed token datasets used by the training loop.

The processed CPT corpus is packed at a fixed length (currently 4096 tokens),
while accelerator-specific training may use a shorter sequence length.  A plain
slice in the collator would silently discard the tail of every packed record.
This module exposes non-overlapping windows without materialising a second
Arrow dataset on disk.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from torch.utils.data import Dataset as TorchDataset


class PackedWindowDataset(TorchDataset):
    """Non-overlapping fixed-size windows over a packed token dataset.

    ``source_dataset`` is expected to implement ``__len__`` and indexed access
    (Hugging Face ``Dataset`` does).  The view is deliberately lazy: only the
    selected source row is read for each item, so expanding 244k packed rows to
    976k 1024-token windows does not duplicate the Arrow data in RAM or on
    disk.
    """

    def __init__(
        self,
        source_dataset: Any,
        window_length: int,
        packed_length: int | None = None,
    ) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        if len(source_dataset) == 0:
            raise ValueError("Cannot window an empty dataset")

        first = source_dataset[0]
        first_ids = first.get("input_ids")
        if first_ids is None:
            raise ValueError("Dataset must contain an input_ids column")

        inferred_length = len(first_ids)
        configured_length = int(packed_length) if packed_length else inferred_length
        # Be tolerant of a preprocessed shard produced with a different
        # packing setting.  The record itself is authoritative; failing here
        # would make a valid 1024-token shard unusable with the default 4096
        # training config.
        self.packed_length = inferred_length if configured_length != inferred_length else configured_length
        if self.packed_length <= 0:
            raise ValueError("packed_length must be positive")

        self.source_dataset = source_dataset
        self.window_length = int(window_length)
        self.windows_per_record = math.ceil(self.packed_length / self.window_length)

    def __len__(self) -> int:
        return len(self.source_dataset) * self.windows_per_record

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        source_index, window_index = divmod(index, self.windows_per_record)
        start = window_index * self.window_length
        end = min(start + self.window_length, self.packed_length)
        record = self.source_dataset[source_index]

        item: Dict[str, Any] = {
            "input_ids": record["input_ids"][start:end],
        }
        for key in ("labels", "attention_mask"):
            if key in record:
                item[key] = record[key][start:end]
        return item
