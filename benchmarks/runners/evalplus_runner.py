"""Orchestrator for EvalPlus Framework (HumanEval+ and MBPP+).

EvalPlus provides 80x amplified test suites with rigorous contracts to detect
false positives in standard code generation benchmarks.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_evalplus_command(
    model_path_or_id: str,
    dataset: str = "humaneval",
    backend: str = "hf",
    greedy: bool = True,
    output_dir: Optional[str] = None,
    dtype: str = "bfloat16",
    i_just_wanna_run: bool = True,
) -> List[str]:
    """Builds the official evalplus CLI evaluation command."""
    cmd = [
        "evalplus.evaluate",
        "--model", model_path_or_id,
        "--dataset", dataset,
        "--backend", backend,
        "--dtype", dtype,
    ]

    if greedy:
        cmd.append("--greedy")

    if output_dir:
        cmd.extend(["--output_dir", output_dir])

    if i_just_wanna_run:
        cmd.append("--i-just-wanna-run")

    return cmd


def parse_evalplus_output(result_json_path: Path) -> Dict[str, Any]:
    """Parses EvalPlus output JSON file into normalized benchmark metrics."""
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # EvalPlus reports base pass@1 and plus pass@1
    base_pass = data.get("pass@1", {}).get("base")
    plus_pass = data.get("pass@1", {}).get("plus")

    return {
        "dataset": data.get("dataset"),
        "base_pass@1": round(base_pass * 100.0, 2) if base_pass is not None else None,
        "plus_pass@1": round(plus_pass * 100.0, 2) if plus_pass is not None else None,
        "raw": data,
    }
