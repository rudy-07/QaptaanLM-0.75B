"""Exact token-budget mixture scheduler for KapInstruct-100M.

Maintains per-source token targets and interleaves streams via deficit-weighted
scheduling, accurately measuring tokens using the Qwen tokenizer after normalization,
filtering, and deduplication.
"""

import logging
import random
from typing import Any, Dict, Generator, List, Optional, Set

logger = logging.getLogger(__name__)


class InstructMixer:
    """Exact deficit-based token mixer for multi-source instruction datasets."""

    def __init__(
        self,
        sources_cfg: Dict[str, Any],
        target_total_tokens: int = 100_000_000,
        seed: int = 42,
        allow_redistribution: bool = False,
    ):
        self.sources_cfg = sources_cfg
        self.target_total_tokens = target_total_tokens
        self.allow_redistribution = allow_redistribution
        self.rng = random.Random(seed)

        # Compute per-source token quotas
        self.target_tokens: Dict[str, int] = {}
        self.target_shares: Dict[str, float] = {}
        for name, cfg in sources_cfg.items():
            share = cfg.get("target_share", 0.0)
            self.target_shares[name] = share
            self.target_tokens[name] = cfg.get("target_tokens", int(share * target_total_tokens))

        # Token & document accounting
        self.actual_rendered_tokens: Dict[str, int] = {n: 0 for n in sources_cfg}
        self.actual_trainable_tokens: Dict[str, int] = {n: 0 for n in sources_cfg}
        self.actual_docs: Dict[str, int] = {n: 0 for n in sources_cfg}

        logger.info(f"InstructMixer initialized with target: {target_total_tokens:,} tokens across {len(sources_cfg)} sources.")

    def _select_next_source(self, active_sources: Set[str]) -> Optional[str]:
        """Select next source based on relative deficit."""
        total_tokens_so_far = sum(self.actual_rendered_tokens.values())
        if total_tokens_so_far >= self.target_total_tokens:
            return None

        # Filter to active sources with remaining deficit
        deficits = {}
        for name in active_sources:
            target = self.target_tokens[name]
            actual = self.actual_rendered_tokens[name]
            if actual < target:
                deficits[name] = (target - actual) / max(target, 1)

        if not deficits:
            if self.allow_redistribution and active_sources:
                return self.rng.choice(list(active_sources))
            return None

        # Weighted deficit selection
        names = list(deficits.keys())
        weights = [deficits[n] for n in names]
        total_weight = sum(weights)
        if total_weight == 0:
            return self.rng.choice(names)

        r = self.rng.random() * total_weight
        cumulative = 0.0
        for name, weight in zip(names, weights):
            cumulative += weight
            if r <= cumulative:
                return name
        return names[-1]

    def record_sample(
        self,
        source_name: str,
        rendered_tokens: int,
        trainable_tokens: int,
    ):
        """Record token counts for an accepted sample."""
        self.actual_rendered_tokens[source_name] += rendered_tokens
        self.actual_trainable_tokens[source_name] += trainable_tokens
        self.actual_docs[source_name] += 1

    def is_source_fulfilled(self, source_name: str) -> bool:
        """Check whether a source has reached its target token allocation."""
        return self.actual_rendered_tokens[source_name] >= self.target_tokens[source_name]

    def is_total_fulfilled(self) -> bool:
        """Check whether the entire dataset target has been reached."""
        return sum(self.actual_rendered_tokens.values()) >= self.target_total_tokens

    def get_mixture_report(self) -> Dict[str, Any]:
        """Generate structured mixture report with exact token statistics."""
        total_rendered = sum(self.actual_rendered_tokens.values())
        total_trainable = sum(self.actual_trainable_tokens.values())
        total_docs = sum(self.actual_docs.values())

        report = {
            "target_total_tokens": self.target_total_tokens,
            "achieved_rendered_tokens": total_rendered,
            "achieved_trainable_tokens": total_trainable,
            "total_documents": total_docs,
            "completion_rate": f"{total_rendered / max(self.target_total_tokens, 1):.2%}",
            "sources": {},
        }

        for name in sorted(self.sources_cfg.keys()):
            actual_rend = self.actual_rendered_tokens[name]
            actual_train = self.actual_trainable_tokens[name]
            target_tok = self.target_tokens[name]
            target_share = self.target_shares[name]
            actual_share = actual_rend / max(total_rendered, 1)

            report["sources"][name] = {
                "target_share": target_share,
                "achieved_share": actual_share,
                "target_tokens": target_tok,
                "achieved_rendered_tokens": actual_rend,
                "achieved_trainable_tokens": actual_train,
                "achieved_docs": self.actual_docs[name],
                "shortfall": max(0, target_tok - actual_rend),
                "fulfilled": actual_rend >= target_tok,
            }

        return report
