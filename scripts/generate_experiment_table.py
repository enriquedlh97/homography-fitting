#!/usr/bin/env python3
"""Scan experiment directories and generate a markdown comparison table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def extract_name(dirname: str) -> str:
    """Strip the timestamp prefix (YYYY-MM-DD_HH-MM-SS_) from the directory name."""
    m = re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.*)", dirname)
    return m.group(1) if m else dirname


def logo_name(config: dict) -> str:
    """Extract logo filename without extension from config."""
    logo_path = (config.get("input") or {}).get("logo", "")
    return Path(logo_path).stem if logo_path else ""


def safe_get(d: dict, *keys, default=None):  # type: ignore[type-arg]
    """Nested dict access with fallback."""
    current: object = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default
    return current


def collect_row(exp_dir: Path) -> dict | None:
    """Read all JSON/YAML files from a single experiment and return a row dict."""
    metrics_path = exp_dir / "metrics.json"
    quality_path = exp_dir / "quality_metrics.json"

    if not metrics_path.exists() or not quality_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(quality_path) as f:
        quality = json.load(f)

    config: dict = {}
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    compositor_params = safe_get(config, "pipeline", "compositor", "params", default={})

    # dE: prefer inpaint_dark_delta_E, fall back to inpaint_color_delta_E_mean
    de = quality.get("inpaint_dark_delta_E")
    if de is None:
        de = quality.get("inpaint_color_delta_E_mean")

    return {
        "name": extract_name(exp_dir.name),
        "gpu": metrics.get("gpu", ""),
        "fps": metrics.get("output_fps"),
        "fitter": metrics.get("fitter_type", ""),
        "logo": logo_name(config),
        "method": compositor_params.get("inpaint_method", ""),
        "dilate": compositor_params.get("mask_dilate_px"),
        "radius": compositor_params.get("inpaint_radius"),
        "de": de,
        "jitter": quality.get("jitter_ratio"),
        "ssim": quality.get("temporal_ssim_mean"),
        "passed": quality.get("metrics_passed"),
        "total": quality.get("metrics_total"),
    }


def short_gpu(name: str) -> str:
    """Shorten GPU name: 'NVIDIA H100 80GB HBM3' -> 'H100'."""
    for tag in ("B200", "H100", "A100", "T4", "L4", "A10G"):
        if tag in name:
            return tag
    return name


def fmt(val, spec: str = "") -> str:
    """Format a value for the table, returning '-' for None."""
    if val is None:
        return "-"
    if spec:
        return format(val, spec)
    return str(val)


def build_table(rows: list[dict]) -> str:
    """Build a markdown table from row dicts, sorted by dE ascending."""
    rows.sort(key=lambda r: (r["de"] is None, r["de"] if r["de"] is not None else 0))

    header = "| Name | GPU | FPS | Fitter | Logo | Method | d | r | dE | Jitter | SSIM | Pass |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]

    for r in rows:
        passed = f"{r['passed']}/{r['total']}" if r["passed"] is not None else "-"
        line = (
            f"| {r['name']}"
            f" | {short_gpu(r['gpu'])}"
            f" | {fmt(r['fps'], '.1f')}"
            f" | {r['fitter']}"
            f" | {r['logo']}"
            f" | {r['method']}"
            f" | {fmt(r['dilate'])}"
            f" | {fmt(r['radius'])}"
            f" | {fmt(r['de'], '.3f')}"
            f" | {fmt(r['jitter'], '.3f')}"
            f" | {fmt(r['ssim'], '.4f')}"
            f" | {passed} |"
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment comparison table")
    parser.add_argument(
        "--experiments-dir",
        default="experiments/",
        help="Root directory containing experiment folders (default: experiments/)",
    )
    parser.add_argument(
        "--output",
        default="analysis/experiment_comparison.md",
        help="Output markdown file (default: analysis/experiment_comparison.md)",
    )
    args = parser.parse_args()

    exp_root = Path(args.experiments_dir)
    if not exp_root.is_dir():
        raise SystemExit(f"experiments directory not found: {exp_root}")

    rows: list[dict] = []
    skipped = 0
    for child in sorted(exp_root.iterdir()):
        if not child.is_dir():
            continue
        row = collect_row(child)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if not rows:
        raise SystemExit("No experiments with both metrics.json and quality_metrics.json found.")

    table = build_table(rows)
    print(table)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {len(rows)} experiments to {out_path}  (skipped {skipped})")


if __name__ == "__main__":
    main()
