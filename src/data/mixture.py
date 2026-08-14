"""Dataset mixture and sampling for balanced CPT training.

Implements weighted interleaving of multiple dataset streams
to achieve the target token distribution.
"""

import logging
import random
from typing import Any, Dict, Generator, List, Optional, Set

logger = logging.getLogger(__name__)


class DatasetMixer:
    """Mix multiple dataset streams according to target proportions.

    Uses weighted sampling to interleave documents from different
    sources, tracking actual token counts to enforce mixture ratios.
    """

    def __init__(
        self,
        target_proportions: Dict[str, float],
        target_total_tokens: int = 1_000_000_000,
        seed: int = 42,
    ):
        """Initialize the mixer.

        Args:
            target_proportions: Dict mapping dataset name to target proportion
                               (should sum to 1.0).
            target_total_tokens: Total tokens to produce.
            seed: Random seed for reproducibility.
        """
        self.target_proportions = target_proportions
        self.target_total_tokens = target_total_tokens
        self.rng = random.Random(seed)

        # Compute per-dataset token targets
        self.target_tokens = {
            name: int(prop * target_total_tokens)
            for name, prop in target_proportions.items()
        }

        # Track actual tokens consumed per source
        self.actual_tokens: Dict[str, int] = {name: 0 for name in target_proportions}
        self.actual_docs: Dict[str, int] = {name: 0 for name in target_proportions}

        logger.info("Dataset mixer initialized:")
        for name, target in self.target_tokens.items():
            logger.info(
                f"  {name}: target={target:,} tokens "
                f"({target_proportions[name]:.0%})"
            )

    def _select_source(self, exhausted: Optional[Set[str]] = None) -> Optional[str]:
        """Select the next source to sample from.

        Uses the deficit-based approach: preferentially sample from
        sources that are furthest behind their target proportion,
        excluding exhausted sources.
        """
        total_actual = sum(self.actual_tokens.values())
        if total_actual >= self.target_total_tokens:
            return None

        exhausted_set = exhausted or set()

        # Compute deficit for each unexhausted source
        deficits = {}
        for name, target in self.target_tokens.items():
            if name in exhausted_set:
                continue
            actual = self.actual_tokens[name]
            if actual >= target:
                continue  # This source has met its quota
            deficit = (target - actual) / max(target, 1)
            deficits[name] = deficit

        if not deficits:
            # If all non-exhausted sources met quota but total target not reached,
            # sample from any remaining available non-exhausted source
            available = [name for name in self.target_tokens.keys() if name not in exhausted_set]
            if not available:
                return None
            return self.rng.choice(available)

        # Weight selection by deficit (sources further behind get priority)
        names = list(deficits.keys())
        weights = [deficits[n] for n in names]
        total_weight = sum(weights)
        if total_weight == 0:
            return self.rng.choice(names)

        # Weighted random selection
        r = self.rng.random() * total_weight
        cumulative = 0.0
        for name, weight in zip(names, weights):
            cumulative += weight
            if r <= cumulative:
                return name

        return names[-1]  # Fallback

    def mix(
        self,
        streams: Dict[str, Generator[Dict[str, Any], None, None]],
        token_count_fn: Optional[callable] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Mix multiple streams according to target proportions.

        Args:
            streams: Dict mapping dataset name to document generator.
                    Each document should have a 'text' field.
            token_count_fn: Optional function to count tokens in text.
                          If None, estimates at 4 chars per token.

        Yields:
            Documents from mixed sources with 'source' field set.
        """
        # Create iterators
        iterators = {name: iter(stream) for name, stream in streams.items()}
        exhausted = set()

        while True:
            source = self._select_source(exhausted=exhausted)
            if source is None:
                break

            try:
                doc = next(iterators[source])
            except StopIteration:
                exhausted.add(source)
                logger.info(
                    f"Source '{source}' exhausted at "
                    f"{self.actual_tokens[source]:,} tokens"
                )
                continue

            # Count tokens
            text = doc.get("text", "")
            if token_count_fn:
                n_tokens = token_count_fn(text)
            else:
                # Rough estimate: ~4 chars per token for English/code
                n_tokens = max(1, len(text) // 4)

            self.actual_tokens[source] += n_tokens
            self.actual_docs[source] += 1
            doc["source"] = source

            yield doc

            # Check if we've hit total target
            total = sum(self.actual_tokens.values())
            if total >= self.target_total_tokens:
                break

        logger.info("Mixing complete. Final distribution:")
        self._log_mixture_report()

    def _log_mixture_report(self):
        """Log the final mixture distribution."""
        total_actual = sum(self.actual_tokens.values())

        header = f"{'Dataset':<30} {'Target':>10} {'Actual':>10} {'Tokens':>15} {'Docs':>10}"
        logger.info(header)
        logger.info("-" * len(header))

        for name in sorted(self.target_proportions.keys()):
            target_pct = self.target_proportions[name]
            actual_tokens = self.actual_tokens[name]
            actual_pct = actual_tokens / max(total_actual, 1)
            docs = self.actual_docs[name]

            logger.info(
                f"{name:<30} {target_pct:>9.1%} {actual_pct:>9.1%} "
                f"{actual_tokens:>14,} {docs:>9,}"
            )

        logger.info("-" * len(header))
        total_docs = sum(self.actual_docs.values())
        logger.info(
            f"{'TOTAL':<30} {'100.0%':>10} {'100.0%':>10} "
            f"{total_actual:>14,} {total_docs:>9,}"
        )

    def get_mixture_report(self) -> Dict[str, Any]:
        """Get mixture distribution as a structured dict."""
        total_actual = sum(self.actual_tokens.values())

        report = {
            "total_tokens": total_actual,
            "total_docs": sum(self.actual_docs.values()),
            "target_total": self.target_total_tokens,
            "completion": f"{total_actual / max(self.target_total_tokens, 1):.1%}",
            "datasets": {},
        }

        for name in sorted(self.target_proportions.keys()):
            actual = self.actual_tokens[name]
            report["datasets"][name] = {
                "target_proportion": self.target_proportions[name],
                "actual_proportion": actual / max(total_actual, 1),
                "target_tokens": self.target_tokens[name],
                "actual_tokens": actual,
                "docs": self.actual_docs[name],
            }

        return report
