"""Native JAX/Flax Distributed Trainer for Qwen3.5 CPT on TPU v5e-8."""

import functools
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax
from flax.training import train_state

from jax_training.models.config import Qwen3_5Config
from jax_training.models.qwen3_5 import Qwen3_5ForCausalLM
from jax_training.models.loss import chunked_linear_cross_entropy
from jax_training.models.convert import load_hf_weights_into_flax
from jax_training.training.checkpoint import CheckpointManager
from jax_training.data.dataset import WindowedDataset
from jax_training.data.prefetch import PrefetchLoader

logger = logging.getLogger(__name__)

# TPU BF16 peak TFLOPs per core by device generation
TPU_PEAK_TFLOPS_MAP = {
    "v3": 123.0,
    "v4": 275.0,
    "v5e": 197.0,
    "v5p": 459.0,
}


@dataclass
class TrainConfig:
    """Training configuration parameters."""

    # Model & Tokenizer
    model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base"
    output_dir: str = "checkpoints/jax_cpt"
    
    # Dataset & Sequences
    max_seq_length: int = 1024
    dataset_packed_seq_length: int = 4096
    target_tokens: int = 1_000_000_000
    
    # Batch & Hardware (per_device_batch_size=2 and loss_chunk_size=256 guaranteed safe on 16GB HBM)
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    loss_chunk_size: int = 256
    
    # Optimizer & Schedule
    learning_rate: float = 2e-5
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.02
    warmup_steps: Optional[int] = None
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Precision & Execution
    dtype: str = "bfloat16"
    seed: int = 42
    
    # Steps & Checkpointing (save every 2500 steps, max 2 checkpoints)
    max_steps: int = -1
    logging_steps: int = 50
    save_steps: int = 2500
    save_total_limit: int = 2
    resume_from_checkpoint: Optional[str] = None
    smoke_test: bool = False


class JAXTrainer:
    """Distributed JAX Trainer for Full-Parameter Continued Pre-Training."""

    def __init__(self, config: TrainConfig, model_config: Optional[Qwen3_5Config] = None):
        self.config = config
        self.model_config = model_config or Qwen3_5Config(dtype=config.dtype)
        self.is_primary = jax.process_index() == 0

        # Setup Mesh and Sharding
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        self.mesh = Mesh(self.devices, ("data",))
        self.data_sharding = NamedSharding(self.mesh, P("data", None))
        self.replicated_sharding = NamedSharding(self.mesh, P())

        # Effective batch sizes
        self.global_batch_size = (
            self.config.per_device_batch_size
            * self.num_devices
            * self.config.gradient_accumulation_steps
        )
        self.tokens_per_step = self.global_batch_size * self.config.max_seq_length

        # Compute total steps
        if self.config.max_steps <= 0:
            self.total_steps = math.ceil(self.config.target_tokens / self.tokens_per_step)
        else:
            self.total_steps = self.config.max_steps

        # Compute warmup steps
        if self.config.warmup_steps is None:
            self.warmup_steps = max(1, int(self.total_steps * self.config.warmup_ratio))
        else:
            self.warmup_steps = self.config.warmup_steps

        if self.is_primary:
            logger.info("=" * 60)
            logger.info("JAX Distributed Trainer Initialized")
            logger.info(f"Devices ({self.num_devices}): {[str(d) for d in self.devices]}")
            logger.info(f"Micro-batch per device: {self.config.per_device_batch_size}")
            logger.info(f"Global batch size: {self.global_batch_size} sequences")
            logger.info(f"Sequence length: {self.config.max_seq_length}")
            logger.info(f"Tokens per step: {self.tokens_per_step:,}")
            logger.info(f"Total target tokens: {self.config.target_tokens:,}")
            logger.info(f"Total training steps: {self.total_steps:,} (warmup: {self.warmup_steps:,})")
            logger.info("=" * 60)

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.config.output_dir,
            max_to_keep=self.config.save_total_limit,
            save_interval_steps=self.config.save_steps,
        )

        # Model instance
        self.model = Qwen3_5ForCausalLM(self.model_config)

    def _build_optimizer(self) -> optax.GradientTransformation:
        """Create AdamW optimizer with cosine warmup schedule."""
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=self.config.learning_rate,
            transition_steps=self.warmup_steps,
        )
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        decay_fn = optax.cosine_decay_schedule(
            init_value=self.config.learning_rate,
            decay_steps=decay_steps,
            alpha=self.config.min_lr_ratio,
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[self.warmup_steps],
        )

        def weight_decay_mask(params):
            """Mask out bias, 1D normalization weights, and decay parameters from weight decay."""
            return jax.tree_util.tree_map(lambda v: v.ndim >= 2, params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adamw(
                learning_rate=schedule_fn,
                b1=self.config.adam_beta1,
                b2=self.config.adam_beta2,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
                mask=weight_decay_mask,
            ),
        )
        return optimizer

    def init_train_state(self, dummy_input: Optional[jax.Array] = None) -> train_state.TrainState:
        """Initialize parameters and optimizer state."""
        rng = jax.random.PRNGKey(self.config.seed)
        if dummy_input is None:
            dummy_input = jnp.ones((1, self.config.max_seq_length), dtype=jnp.int32)

        if self.config.model_name_or_path and (
            Path(self.config.model_name_or_path).exists() or "/" in self.config.model_name_or_path
        ):
            try:
                if self.is_primary:
                    logger.info(f"Loading pretrained weights from {self.config.model_name_or_path}...")
                params = load_hf_weights_into_flax(
                    self.config.model_name_or_path,
                    self.model_config,
                    target_dtype=self.config.dtype,
                )
            except Exception as e:
                if self.is_primary:
                    logger.warning(f"Could not load pretrained weights: {e}. Initializing randomly.")
                variables = self.model.init(rng, dummy_input)
                params = variables["params"]
        else:
            variables = self.model.init(rng, dummy_input)
            params = variables["params"]

        optimizer = self._build_optimizer()
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
        )
        return state

    def _calculate_flops_per_token(self) -> float:
        """Calculate approximate FLOPs per token for Qwen3.5-0.8B."""
        # 6N for standard attention/linear attention causal forward + backward
        # Parameters ~752M
        num_params = 752_000_000
        return 6.0 * num_params

    def train(self, train_dataset: Any):
        """Execute full distributed Continued Pre-Training."""
        # Setup lazy windowing if dataset is packed
        if not isinstance(train_dataset, WindowedDataset):
            train_dataset = WindowedDataset(
                source_dataset=train_dataset,
                window_length=self.config.max_seq_length,
                packed_length=self.config.dataset_packed_seq_length,
            )
            if self.is_primary:
                logger.info(
                    f"Expanded packed dataset: {len(train_dataset.source_dataset):,} records -> "
                    f"{len(train_dataset):,} sequences of {self.config.max_seq_length} tokens"
                )

        # Setup asynchronous prefetch data loader
        loader = PrefetchLoader(
            dataset=train_dataset,
            global_batch_size=self.global_batch_size,
            seq_length=self.config.max_seq_length,
            sharding=self.data_sharding,
            shuffle=True,
            seed=self.config.seed,
        )

        # Initialize or restore TrainState
        state = self.init_train_state()
        start_step = 0
        tokens_trained = 0

        # Auto-resume from latest checkpoint if requested
        resume_ckpt = self.config.resume_from_checkpoint
        if resume_ckpt is None:
            latest = self.checkpoint_manager.find_latest_checkpoint()
            if latest:
                resume_ckpt = str(latest)

        if resume_ckpt:
            if self.is_primary:
                logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
            state = self.checkpoint_manager.restore(resume_ckpt, state)
            start_step = int(state.step)
            tokens_trained = start_step * self.tokens_per_step
            if self.is_primary:
                logger.info(f"Resumed at step {start_step:,} ({tokens_trained:,} tokens)")

        # Setup SPMD Sharding specs and buffer placement
        state_sharding = jax.tree_util.tree_map(lambda _: self.replicated_sharding, state)
        data_sharding = self.data_sharding
        state = jax.device_put(state, state_sharding)
        loss_chunk_size = self.config.loss_chunk_size

        def loss_fn(params, batch):
            hidden_states = self.model.apply({"params": params}, batch["input_ids"])
            embedding_weights = params["model"]["embed_tokens"]["embedding"]
            loss, valid_tokens = chunked_linear_cross_entropy(
                hidden_states=hidden_states,
                labels=batch["labels"],
                embedding_weights=embedding_weights,
                chunk_size=loss_chunk_size,
            )
            return loss, (loss, valid_tokens)

        @functools.partial(
            jax.jit,
            in_shardings=(state_sharding, data_sharding),
            out_shardings=(state_sharding, self.replicated_sharding),
            donate_argnums=(0,),
        )
        def train_step(train_state_obj, batch):
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, (raw_loss, valid_tokens)), grads = grad_fn(train_state_obj.params, batch)
            # Apply optimizer update
            new_train_state = train_state_obj.apply_gradients(grads=grads)
            return new_train_state, {"loss": raw_loss, "valid_tokens": valid_tokens}

        if self.is_primary:
            logger.info("Starting training loop...")

        # Benchmark compilation explicitly on Step 1
        data_iter = iter(loader)
        flops_per_token = self._calculate_flops_per_token()
        
        # Detect TPU device peak TFLOPs
        dev_kind = str(self.devices[0].device_kind).lower() if self.devices else "v5e"
        peak_per_core = TPU_PEAK_TFLOPS_MAP.get("v5e", 197.0)
        for k, v in TPU_PEAK_TFLOPS_MAP.items():
            if k in dev_kind:
                peak_per_core = v
                break
        total_peak_tflops = peak_per_core * max(1, self.num_devices)

        step_times = []
        last_log_time = time.time()
        start_train_time = time.time()

        for step in range(start_step + 1, self.total_steps + 1):
            batch = next(data_iter)

            if step == start_step + 1:
                if self.is_primary:
                    logger.info("Compiling JIT computation graph on TPU...")
                compile_start = time.time()
                state, metrics = train_step(state, batch)
                # Block until ready
                metrics["loss"].block_until_ready()
                compile_duration = time.time() - compile_start
                if self.is_primary:
                    logger.info(f"✓ Compilation completed in {compile_duration:.2f}s")
                last_log_time = time.time()
                continue

            t0 = time.time()
            state, metrics = train_step(state, batch)
            metrics["loss"].block_until_ready()
            step_dt = time.time() - t0
            step_times.append(step_dt)

            tokens_trained += self.tokens_per_step

            # Logging
            should_log = (
                step % self.config.logging_steps == 0
                or step == self.total_steps
                or self.config.smoke_test
            )
            if should_log:
                now = time.time()
                elapsed = now - last_log_time
                avg_step_time = np.mean(step_times[-self.config.logging_steps :])
                tok_per_sec = self.tokens_per_step / avg_step_time
                steps_per_sec = 1.0 / avg_step_time
                
                # MFU calculation
                achieved_tflops = (tok_per_sec * flops_per_token) / 1e12
                mfu_percent = (achieved_tflops / total_peak_tflops) * 100.0

                # ETA
                remaining_steps = self.total_steps - step
                eta_seconds = remaining_steps * avg_step_time
                eta_hours = eta_seconds / 3600.0

                current_loss = float(metrics["loss"])
                lr = float(self.config.learning_rate)

                if self.is_primary:
                    pct = (step / self.total_steps) * 100.0
                    logger.info(
                        f"Step {step:,}/{self.total_steps:,} ({pct:.2f}%) | "
                        f"Loss: {current_loss:.4f} | "
                        f"Speed: {tok_per_sec:,.0f} tok/s ({steps_per_sec:.2f} steps/s) | "
                        f"MFU: {mfu_percent:.1f}% | "
                        f"Trained: {tokens_trained / 1e6:.2f}M tokens | "
                        f"ETA: {eta_hours:.2f}h"
                    )
                last_log_time = now

            # Checkpointing (asynchronous non-blocking during training, blocking only at final step)
            is_final_step = (step == self.total_steps) or (self.config.smoke_test and step >= start_step + 5)
            if step % self.config.save_steps == 0 or is_final_step:
                self.checkpoint_manager.save(
                    step=step,
                    state=state,
                    metadata={
                        "loss": float(metrics["loss"]),
                        "tokens_trained": tokens_trained,
                        "step": step,
                    },
                    blocking=is_final_step,
                )

            # Smoke test early exit
            if self.config.smoke_test and step >= start_step + 5:
                if self.is_primary:
                    logger.info(f"✓ Smoke test completed successfully after {step} steps!")
                break

        loader.stop()
        self.checkpoint_manager.wait_until_finished()
        total_time = time.time() - start_train_time
        if self.is_primary:
            logger.info("=" * 60)
            logger.info(f"✓ Training finished! Total time: {total_time / 3600.0:.2f} hours")
            logger.info("=" * 60)
