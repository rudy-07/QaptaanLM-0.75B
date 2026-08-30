"""Orchestrator for LiveCodeBench (LCB).

LiveCodeBench evaluates uncontaminated programming capabilities across
LeetCode, AtCoder, and Codeforces contest problems.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_livecodebench_command(
    model_path_or_id: str,
    scenario: str = "codegeneration",
    release_version: str = "release_latest",
    dtype: str = "bfloat16",
    output_path: Optional[str] = None,
) -> List[str]:
    """Builds LiveCodeBench runner CLI command."""
    cmd = [
        "python", "-m", "livecodebench.evaluate",
        "--model", model_path_or_id,
        "--scenario", scenario,
        "--release_version", release_version,
        "--dtype", dtype,
    ]
    if output_path:
        cmd.extend(["--output_path", output_path])
    return cmd
