"""Asynchronous background prefetch loader for device feeding."""

import queue
import threading
from typing import Any, Dict, Iterator, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


class PrefetchLoader:
    """Multi-threaded asynchronous data loader with device prefetching."""

    def __init__(
        self,
        dataset: Any,
        global_batch_size: int,
        seq_length: int,
        sharding: Optional[NamedSharding] = None,
        shuffle: bool = True,
        seed: int = 42,
        buffer_size: int = 4,
    ):
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.seq_length = seq_length
        self.sharding = sharding
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size

        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _worker(self):
        rng = np.random.default_rng(self.seed)
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)

        while not self._stop_event.is_set():
            if self.shuffle:
                rng.shuffle(indices)

            for i in range(0, num_samples - self.global_batch_size + 1, self.global_batch_size):
                if self._stop_event.is_set():
                    break

                batch_indices = indices[i : i + self.global_batch_size]
                input_ids_list = []
                labels_list = []

                for idx in batch_indices:
                    item = self.dataset[int(idx)]
                    input_ids_list.append(item["input_ids"][: self.seq_length])
                    labels_list.append(item["labels"][: self.seq_length])

                batch_input_ids = np.stack(input_ids_list, axis=0).astype(np.int32)
                batch_labels = np.stack(labels_list, axis=0).astype(np.int32)
                batch = {
                    "input_ids": batch_input_ids,
                    "labels": batch_labels,
                }

                while not self._stop_event.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        self.start()
        while True:
            try:
                batch = self._queue.get(timeout=60.0)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

    def __del__(self):
        self.stop()
