"""Generates comparison reports (Markdown & HTML) from benchmarking runs."""

import json
from pathlib import Path
from typing import Any, Dict


def generate_html_report(coding_data: Dict[str, Any], reasoning_data: Dict[str, Any], output_html: str):
    base_code = coding_data.get("base_model", {})
    cpt_code = coding_data.get("cpt_model", {})
    base_reas = reasoning_data.get("base_model", {})
    cpt_reas = reasoning_data.get("cpt_model", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QaptaanLM-0.75B Benchmark Evaluation Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 24px; text-align: center; }}
        h1 {{ margin: 0 0 8px 0; color: #38bdf8; }}
        .subtitle {{ color: #94a3b8; margin: 0; font-size: 15px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; text-align: center; }}
        .card-title {{ color: #94a3b8; font-size: 13px; text-transform: uppercase; margin-bottom: 8px; }}
        .card-value {{ font-size: 28px; font-weight: bold; color: #f8fafc; }}
        .card-sub {{ font-size: 12px; color: #4ade80; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #0b1120; color: #94a3b8; text-transform: uppercase; font-size: 12px; }}
        tr:last-child td {{ border-bottom: none; }}
        .badge-base {{ background: #1e3a8a; color: #93c5fd; padding: 3px 8px; border-radius: 6px; font-size: 12px; }}
        .badge-cpt {{ background: #14532d; color: #86efac; padding: 3px 8px; border-radius: 6px; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🚀 QaptaanLM-0.75B vs Qwen3.5-0.8B-Base Evaluation</h1>
        <p class="subtitle">Comprehensive Benchmark Results across Coding, Mathematical Reasoning & Science QA</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Coding Pass@1 (CPT)</div>
            <div class="card-value">{cpt_code.get('pass_rate', 0):.1f}%</div>
            <div class="card-sub">Base: {base_code.get('pass_rate', 0):.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Reasoning Accuracy (CPT)</div>
            <div class="card-value">{cpt_reas.get('overall_accuracy', 0):.1f}%</div>
            <div class="card-sub">Base: {base_reas.get('overall_accuracy', 0):.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Inference Speed (CPT)</div>
            <div class="card-value">{cpt_code.get('throughput_tok_s', 0):.1f} tok/s</div>
            <div class="card-sub">O(1) Recurrent Caching Enabled</div>
        </div>
    </div>

    <h2>📊 Detailed Domain Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Benchmark Domain</th>
                <th>Base Model (Qwen3.5-0.8B)</th>
                <th>CPT Model (QaptaanLM-0.75B)</th>
                <th>Delta</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Python Coding (HumanEval + MBPP)</strong></td>
                <td><span class="badge-base">{base_code.get('pass_rate', 0):.1f}%</span> ({base_code.get('passed', 0)}/{base_code.get('total', 0)})</td>
                <td><span class="badge-cpt">{cpt_code.get('pass_rate', 0):.1f}%</span> ({cpt_code.get('passed', 0)}/{cpt_code.get('total', 0)})</td>
                <td style="color: {'#4ade80' if cpt_code.get('pass_rate', 0) >= base_code.get('pass_rate', 0) else '#f87171'};">
                    {cpt_code.get('pass_rate', 0) - base_code.get('pass_rate', 0):+.1f}%
                </td>
            </tr>
"""

    for domain_name in base_reas.get("domains", {}):
        b_dom = base_reas["domains"].get(domain_name, {})
        c_dom = cpt_reas["domains"].get(domain_name, {})
        b_acc = b_dom.get("accuracy", 0)
        c_acc = c_dom.get("accuracy", 0)
        diff = c_acc - b_acc
        color = "#4ade80" if diff >= 0 else "#f87171"
        html += f"""
            <tr>
                <td><strong>{domain_name}</strong></td>
                <td><span class="badge-base">{b_acc:.1f}%</span> ({b_dom.get('correct', 0)}/{b_dom.get('total', 0)})</td>
                <td><span class="badge-cpt">{c_acc:.1f}%</span> ({c_dom.get('correct', 0)}/{c_dom.get('total', 0)})</td>
                <td style="color: {color};">{diff:+.1f}%</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
</div>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ Generated HTML dashboard: {output_html}")


if __name__ == "__main__":
    coding_path = Path("coding_benchmark_results.json")
    reasoning_path = Path("reasoning_benchmark_results.json")

    coding_data = json.loads(coding_path.read_text(encoding="utf-8")) if coding_path.exists() else {}
    reasoning_data = json.loads(reasoning_path.read_text(encoding="utf-8")) if reasoning_path.exists() else {}

    generate_html_report(coding_data, reasoning_data, "benchmark_report.html")
