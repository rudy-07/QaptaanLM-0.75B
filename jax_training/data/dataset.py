"""Zero-copy memory-mapped dataset reader, lazy windowing, and splitting for JAX."""

import glob
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class ArrowShardDataset:
    """Zero-copy memory-mapped reader for Apache Arrow and Parquet shards."""

    def __init__(self, file_paths: List[str]):
        if not file_paths:
            raise ValueError("No shard files provided to ArrowShardDataset")
        self.file_paths = sorted(file_paths)
        self._dataset = None
        self._load_dataset()

    def _load_dataset(self):
        try:
            from datasets import load_dataset
            first = self.file_paths[0].lower()
            fmt = "parquet" if first.endswith(".parquet") else "arrow"
            self._hf_dataset = load_dataset(fmt, data_files=self.file_paths, split="train", keep_in_memory=False).with_format("numpy")
            self._total_rows = len(self._hf_dataset)
        except Exception as e:
            import pyarrow.dataset as ds
            first = self.file_paths[0].lower()
            fmt = "parquet" if first.endswith(".parquet") else "arrow"
            self._dataset = ds.dataset(self.file_paths, format=fmt)
            self._total_rows = self._dataset.count_rows()

    def __len__(self) -> int:
        return self._total_rows

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0:
            index += self._total_rows
        if index < 0 or index >= self._total_rows:
            raise IndexError(f"Index {index} out of range for dataset of size {self._total_rows}")

        if hasattr(self, "_hf_dataset") and self._hf_dataset is not None:
            item = self._hf_dataset[index]
            res = {
                "input_ids": item["input_ids"].astype(np.int32, copy=False),
                "labels": item["labels"].astype(np.int32, copy=False) if "labels" in item else item["input_ids"].astype(np.int32, copy=False),
            }
            if "attention_mask" in item:
                res["attention_mask"] = item["attention_mask"].astype(np.int8, copy=False)
            return res

        # PyArrow take single record (zero-copy, no table materialization)
        cols = ["input_ids"]
        if "labels" in self._dataset.schema.names:
            cols.append("labels")
        if "attention_mask" in self._dataset.schema.names:
            cols.append("attention_mask")
        batch = self._dataset.take([index], columns=cols)
        input_ids = np.array(batch["input_ids"][0].as_py(), dtype=np.int32)
        if "labels" in batch.column_names:
            labels = np.array(batch["labels"][0].as_py(), dtype=np.int32)
        else:
            labels = input_ids.copy()
        
        res = {"input_ids": input_ids, "labels": labels}
        if "attention_mask" in batch.column_names:
            res["attention_mask"] = np.array(batch["attention_mask"][0].as_py(), dtype=np.int8)
        return res


class SubsetDataset:
    """Indexed subset wrapper over any indexable dataset."""

    def __init__(self, source_dataset: Any, indices: Sequence[int]):
        self.source_dataset = source_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0:
            index += len(self.indices)
        if index < 0 or index >= len(self.indices):
            raise IndexError(f"Index {index} out of range for subset of size {len(self.indices)}")
        actual_idx = self.indices[index]
        return self.source_dataset[actual_idx]


class WindowedDataset:
    """Lazy, non-overlapping sequence windowing over packed token records.

    Exposes packed records (e.g. 4096 tokens) as multiple non-overlapping windows of
    length `window_length` without duplicating memory on disk or in RAM.
    """

    def __init__(self, source_dataset: Any, window_length: int = 1024, packed_length: int = 4096):
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}")
        self.source_dataset = source_dataset
        self.window_length = int(window_length)
        self.packed_length = int(packed_length)
        self.windows_per_record = math.ceil(self.packed_length / self.window_length)
        self._total_windows = len(source_dataset) * self.windows_per_record

    def __len__(self) -> int:
        return self._total_windows

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0:
            index += self._total_windows
        if index < 0 or index >= self._total_windows:
            raise IndexError(f"Window index {index} out of bounds")

        record_idx, window_idx = divmod(index, self.windows_per_record)
        record = self.source_dataset[record_idx]

        start = window_idx * self.window_length
        end = start + self.window_length

        raw_ids = record["input_ids"]
        raw_labels = record.get("labels", raw_ids)

        input_ids = np.array(raw_ids[start:end], dtype=np.int32)
        labels = np.array(raw_labels[start:end], dtype=np.int32)

        # If window is shorter than window_length (tail of record), pad to static shape
        if len(input_ids) < self.window_length:
            pad_len = self.window_length - len(input_ids)
            input_ids = np.pad(input_ids, (0, pad_len), mode="constant", constant_values=0)
            labels = np.pad(labels, (0, pad_len), mode="constant", constant_values=-100)

        res = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if "attention_mask" in record:
            raw_mask = record["attention_mask"]
            mask = np.array(raw_mask[start:end], dtype=np.int8)
            if len(mask) < self.window_length:
                pad_len = self.window_length - len(mask)
                mask = np.pad(mask, (0, pad_len), mode="constant", constant_values=0)
            res["attention_mask"] = mask

        return res


def split_dataset(
    dataset: Any,
    eval_ratio: float = 0.015,
    seed: int = 42,
) -> Tuple[Any, Optional[Any]]:
    """Split a dataset into train and validation splits deterministically."""
    total_len = len(dataset)
    if eval_ratio <= 0.0 or total_len <= 1:
        return dataset, None

    eval_size = max(1, int(total_len * eval_ratio))
    train_size = total_len - eval_size

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(total_len).tolist()

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)

    return train_dataset, val_dataset


def load_sharded_dataset(data_dir: Optional[str] = None) -> Any:
    """Auto-discover and load Arrow/Parquet shards from data_dir or /kaggle/input."""
    files_to_load = []
    file_type = None

    search_dirs = []
    if data_dir:
        search_dirs.append(Path(data_dir))
    if os.path.exists("/kaggle/input"):
        search_dirs.append(Path("/kaggle/input"))
    if os.path.exists("data/processed"):
        search_dirs.append(Path("data/processed"))
    if os.path.exists("data/kapinstruct"):
        search_dirs.append(Path("data/kapinstruct"))

    for d in search_dirs:
        if not d.exists():
            continue
        arrow_files = sorted([str(f) for f in d.rglob("*.arrow") if not f.name.startswith(".")])
        parquet_files = sorted([str(f) for f in d.rglob("*.parquet") if not f.name.startswith(".")])
        if arrow_files:
            files_to_load = arrow_files
            file_type = "arrow"
            break
        elif parquet_files:
            files_to_load = parquet_files
            file_type = "parquet"
            break

    if files_to_load:
        from datasets import load_dataset
        return load_dataset(file_type, data_files=files_to_load, split="train", keep_in_memory=False)

    return None
