"""Model Comparison and Leaderboard Generator.

Loads one or more evaluated benchmark JSON result files, compares metrics side-by-side,
computes percentage improvements/regressions, and generates Markdown and HTML reports.

Usage:
    python -m benchmarks.scripts.compare_models --results metrics/results/base.json metrics/results/cpt.json --baseline Qwen2.5-Coder-0.5B
    python -m benchmarks.scripts.compare_models --dir metrics/results --output-dir reports/comparison
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.core.report_generator import (
    generate_html_report,
    generate_markdown_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("compare_models")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Evaluated Model Benchmarks")
    parser.add_argument(
        "--results",
        nargs="+",
        default=[],
        help="List of benchmark JSON result files to compare",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing benchmark JSON files to automatically load",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Model name to treat as baseline for calculating delta improvements",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/comparison",
        help="Directory to save generated comparison reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files_to_load = [Path(f) for f in args.results]

    if args.dir:
        dir_p = Path(args.dir)
        if dir_p.exists():
            files_to_load.extend(list(dir_p.glob("*.json")))

    if not files_to_load:
        logger.error("No benchmark JSON files specified or found! Use --results or --dir.")
        sys.exit(1)

    eval_results: Dict[str, Dict[str, Any]] = {}
    for f in files_to_load:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                model_name = data.get("model", f.stem)
                eval_results[model_name] = data
                logger.info(f"Loaded benchmark result for: {model_name} from {f.name}")
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    if not eval_results:
        logger.error("No valid benchmark results parsed.")
        sys.exit(1)

    # Load reference baselines
    ref_baselines_path = Path(__file__).resolve().parent.parent / "reference_baselines" / "published_scores.json"
    ref_baselines = {}
    if ref_baselines_path.exists():
        with open(ref_baselines_path, "r", encoding="utf-8") as f:
            ref_baselines = json.load(f).get("models", {})

    baseline_model = args.baseline
    if not baseline_model and len(eval_results) > 1:
        # Default baseline to first model or model containing 'base' in name
        for m in eval_results:
            if "base" in m.lower() or "0.5b" in m.lower():
                baseline_model = m
                break
        if not baseline_model:
            baseline_model = list(eval_results.keys())[0]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_report = generate_markdown_report(
        eval_results=eval_results,
        baseline_model=baseline_model,
        reference_baselines=ref_baselines,
        title="Head-to-Head Benchmark Comparison Leaderboard",
    )

    md_file = out_dir / "benchmark_comparison.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info(f"Saved Markdown comparison: {md_file}")

    html_file = out_dir / "benchmark_comparison.html"
    generate_html_report(
        eval_results=eval_results,
        output_path=html_file,
        baseline_model=baseline_model,
        reference_baselines=ref_baselines,
    )
    logger.info(f"Saved HTML comparison leaderboard: {html_file}")

    print("\n" + md_report)


if __name__ == "__main__":
    main()
