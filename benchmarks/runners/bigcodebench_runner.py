"""Orchestrator for BigCodeBench.

BigCodeBench evaluates LLMs on 1,140 complex software engineering tasks
requiring multi-library utilization (Numpy, Pandas, Matplotlib, Pytorch, Scikit-learn, etc.).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_bigcodebench_generate_command(
    model_path_or_id: str,
    subset: str = "instruct",
    split: str = "complete",
    backend: str = "hf",
    dtype: str = "bfloat16",
    output_dir: Optional[str] = None,
) -> List[str]:
    """Builds BigCodeBench generation command."""
    cmd = [
        "bigcodebench.generate",
        "--model", model_path_or_id,
        "--subset", subset,
        "--split", split,
        "--backend", backend,
        "--dtype", dtype,
    ]
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    return cmd


def build_bigcodebench_evaluate_command(
    samples_path: str,
    split: str = "complete",
    execution_timeout: float = 10.0,
) -> List[str]:
    """Builds BigCodeBench sandboxed execution evaluation command."""
    return [
        "bigcodebench.evaluate",
        "--samples", samples_path,
        "--split", split,
        "--execution_timeout", str(execution_timeout),
    ]
