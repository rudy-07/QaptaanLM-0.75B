"""Unified Model Loader and Inference Engine for Benchmarks.

Supports Hugging Face Hub IDs, local checkpoint paths, custom modeling scripts,
and both generative (greedy/sampled) and log-likelihood scoring modes.
"""

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class BenchmarkModelWrapper:
    """Wrapper around HuggingFace CausalLM for standardized benchmark inference."""

    def __init__(
        self,
        model_path_or_id: str,
        device: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_path_or_id = model_path_or_id
        self.trust_remote_code = trust_remote_code

        # Determine target device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine torch dtype
        if dtype is None:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif self.device == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        elif isinstance(dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "auto": "auto",
            }
            self.dtype = dtype_map.get(dtype.lower(), torch.float32)
        else:
            self.dtype = dtype

        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """Loads tokenizer and causal language model with proper error handling."""
        logger.info(f"Loading tokenizer from: {self.model_path_or_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path_or_id,
            trust_remote_code=self.trust_remote_code,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

        logger.info(
            f"Loading model from: {self.model_path_or_id} (dtype={self.dtype}, device={self.device})"
        )

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }

        if self.dtype != "auto":
            load_kwargs["torch_dtype"] = self.dtype

        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path_or_id,
            **load_kwargs,
        )

        if not (self.load_in_8bit or self.load_in_4bit or hasattr(self.model, "hf_device_map")):
            self.model = self.model.to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        do_sample: Optional[bool] = None,
        batch_size: int = 1,
    ) -> List[str]:
        """Generates completions for one or more prompts.

        Args:
            prompts: Single string prompt or list of string prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. If <= 0, uses greedy decoding.
            top_p: Nucleus sampling probability threshold.
            stop_sequences: List of string tokens/sequences to truncate generation at.
            do_sample: Explicitly override sampling behavior.
            batch_size: Batch size for batched generation.

        Returns:
            List of generated text completions (prompt removed).
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if do_sample is None:
            do_sample = temperature > 0.0

        completions: List[str] = []

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            # Move inputs to target model device
            target_device = self.model.device if hasattr(self.model, "device") else self.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            outputs = self.model.generate(**inputs, **gen_kwargs)

            for idx, output in enumerate(outputs):
                gen_toks = output[inputs["input_ids"].shape[1] :]
                text = self.tokenizer.decode(gen_toks, skip_special_tokens=True)

                if stop_sequences:
                    for stop in stop_sequences:
                        if stop in text:
                            text = text.split(stop)[0]

                completions.append(text)

        return completions

    @torch.inference_mode()
    def compute_loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> Tuple[float, bool]:
        """Computes log-likelihood of a continuation given a context.

        Args:
            context: Context prefix string.
            continuation: Target continuation string.

        Returns:
            Tuple of (log_likelihood: float, is_greedy: bool).
        """
        target_device = self.model.device if hasattr(self.model, "device") else self.device

        context_enc = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
        full_enc = self.tokenizer(
            context + continuation, return_tensors="pt", add_special_tokens=False
        )

        context_len = context_enc["input_ids"].shape[1]
        full_ids = full_enc["input_ids"].to(target_device)

        if full_ids.shape[1] <= context_len:
            return 0.0, True

        logits = self.model(full_ids).logits
        shift_logits = logits[0, context_len - 1 : -1, :]
        shift_labels = full_ids[0, context_len:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        cont_log_probs = torch.gather(log_probs, 1, shift_labels.unsqueeze(1)).squeeze(1)

        total_log_likelihood = cont_log_probs.sum().item()
        greedy_preds = torch.argmax(shift_logits, dim=-1)
        is_greedy = bool(torch.equal(greedy_preds, shift_labels))

        return total_log_likelihood, is_greedy

    @torch.inference_mode()
    def evaluate_multiple_choice_loglikelihood(
        self,
        context: str,
        choices: List[str],
    ) -> int:
        """Evaluates multiple choice question via log-likelihood scoring.

        Args:
            context: The question / context prompt.
            choices: List of choice strings (e.g. [" Option A", " Option B"]).

        Returns:
            Index of the choice with highest log-likelihood.
        """
        scores = []
        for choice in choices:
            prefix = " " if not choice.startswith(" ") else ""
            cand = prefix + choice
            score, _ = self.compute_loglikelihood(context, cand)
            scores.append(score)

        return int(torch.tensor(scores).argmax().item())

    def unload(self) -> None:
        """Cleans up model from GPU memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
