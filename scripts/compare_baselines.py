#!/usr/bin/env python3
"""Compare two sets of experiment metrics and print a delta table.

Usage
-----
    # Compare a single before/after pair
    uv run python scripts/compare_baselines.py \
        --before experiments/perf/00_baseline/b200_11p.json \
        --after experiments/2026-04-08_17-30-00_11prompts_B200/metrics.json

    # Compare two directories (matches files by basename)
    uv run python scripts/compare_baselines.py \
        --before experiments/perf/00_baseline \
        --after experiments/perf/02_crop_roi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _mean_of(metric):
    """Extract a scalar from a metric (which may be a dict with 'mean' or a number)."""
    if isinstance(metric, dict):
        return metric.get("mean", 0.0)
    return metric or 0.0


def _load(path: Path) -> dict:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    # Directory: load the first metrics.json found inside
    for candidate in [path / "metrics.json", *sorted(path.glob("*.json"))]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(f"No metrics.json found in {path}")


def _format_pct(before: float, after: float) -> str:
    if before == 0:
        return "—"
    delta = (after - before) / before * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _format_speedup(before: float, after: float) -> str:
    if after == 0:
        return "—"
    return f"{before / after:.2f}×" if after < before else f"1/{after / before:.2f}×"


def diff_pair(before: dict, after: dict, label: str) -> None:
    print(f"\n## {label}")
    print()
    print("| Metric | Before | After | Δ% | Speedup |")
    print("|---|---:|---:|---:|---:|")

    keys = [
        ("output_fps", "FPS (higher better)"),
        ("total_s", "total time (s)"),
        ("segment_total_s", "segment (s)"),
        ("fit_mean_ms", "fit mean (ms/frame)"),
        ("composite_mean_ms", "composite mean (ms/frame)"),
        ("write_video_s", "write video (s)"),
    ]

    for key, label_pretty in keys:
        b = _mean_of(before.get(key, 0))
        a = _mean_of(after.get(key, 0))
        if b == 0 and a == 0:
            continue
        # For FPS, "higher is better" so we invert the speedup interpretation
        if key == "output_fps":
            speedup = f"{a / b:.2f}×" if b > 0 else "—"
        else:
            speedup = _format_speedup(b, a)
        print(f"| {label_pretty} | {b:.3f} | {a:.3f} | {_format_pct(b, a)} | {speedup} |")

    # Composite breakdown if both runs have it
    bb = before.get("composite_breakdown_ms", {})
    ab = after.get("composite_breakdown_ms", {})
    if bb or ab:
        print()
        print("**Composite stage breakdown (ms/frame):**")
        print()
        print("| Step | Before | After | Δ% |")
        print("|---|---:|---:|---:|")
        for step in sorted(set(bb) | set(ab)):
            b = bb.get(step, 0)
            a = ab.get(step, 0)
            print(f"| `{step}` | {b:.2f} | {a:.2f} | {_format_pct(b, a)} |")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment metrics")
    parser.add_argument("--before", required=True, help="Baseline metrics file or dir")
    parser.add_argument("--after", required=True, help="New metrics file or dir")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)

    if before_path.is_dir() and after_path.is_dir():
        # Match files by basename within both dirs
        before_files = {p.name: p for p in before_path.glob("*.json")}
        after_files = {p.name: p for p in after_path.glob("*.json")}
        common = sorted(set(before_files) & set(after_files))
        if not common:
            # Fall back: pair the single metrics.json each contains
            try:
                b = _load(before_path)
                a = _load(after_path)
                diff_pair(b, a, f"{before_path.name} vs {after_path.name}")
            except FileNotFoundError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
            return
        for name in common:
            with open(before_files[name]) as f:
                b = json.load(f)
            with open(after_files[name]) as f:
                a = json.load(f)
            diff_pair(b, a, name)
    else:
        b = _load(before_path)
        a = _load(after_path)
        diff_pair(b, a, f"{before_path.name} vs {after_path.name}")


if __name__ == "__main__":
    main()
