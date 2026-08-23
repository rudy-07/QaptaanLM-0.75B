"""Streaming dataset loader for all 12 KapInstruct-100M instruction sources.

Provides unified streaming generators for:
1. smol_magpie_ultra (HuggingFaceTB/smoltalk:smol-magpie-ultra)
2. magicoder_evol (ise-uiuc/Magicoder-Evol-Instruct-110K)
3. magicoder_oss (ise-uiuc/Magicoder-OSS-Instruct-75K)
4. openmathinstruct2 (nvidia/OpenMathInstruct-2)
5. numinamath_cot (AI-MO/NuminaMath-CoT)
6. openthoughts_reasoning (open-thoughts/OpenThoughts-114k)
7. openhermes_2_5 (teknium/OpenHermes-2.5)
8. tulu3_sft (allenai/tulu-3-sft-mixture)
9. self_oss_starcoder2 (bigcode/self-oss-instruct-sc2-exec-filter-50k)
10. smol_constraints (HuggingFaceTB/smoltalk:smol-constraints)
11. stem_qa (TIGER-Lab/WebInstructSub)
12. code_debugging (m-a-p/CodeFeedback-Filtered-Instruction)
"""

import logging
import time
from typing import Any, Dict, Generator, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


class InstructDatasetLoader:
    """Unified loader providing streaming iterators for all instruction sources."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources_cfg = config.get("sources", {})

    def load_stream(self, source_name: str) -> Generator[Dict[str, Any], None, None]:
        """Load streaming generator for a given source name with automatic retries."""
        cfg = self.sources_cfg.get(source_name)
        if not cfg:
            raise ValueError(f"Source '{source_name}' not found in configuration.")

        source_id = cfg["source_id"]
        split = cfg.get("split", "train")
        config_name = cfg.get("config")
        data_files = cfg.get("data_files")
        streaming = cfg.get("streaming", True)

        kwargs: Dict[str, Any] = {"streaming": streaming}
        if config_name:
            kwargs["name"] = config_name
        if data_files:
            kwargs["data_files"] = data_files
        if split:
            kwargs["split"] = split

        logger.info(f"Initializing stream for [{source_name}] ({source_id})...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                ds = load_dataset(source_id, **kwargs)
                for item in ds:
                    yield item
                return
            except Exception as e:
                logger.warning(
                    f"[{source_name}] Stream connection interrupted (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"[{source_name}] Stream failed permanently after {max_retries} attempts.")
                    raise e

    def get_all_streams(self) -> Dict[str, Generator[Dict[str, Any], None, None]]:
        """Get dict of streaming generators for all configured sources."""
        return {
            source_name: self.load_stream(source_name)
            for source_name in self.sources_cfg.keys()
        }
