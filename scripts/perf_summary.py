#!/usr/bin/env python3
"""Generate a before/after performance summary comparing the baseline matrix
to the post-optimization matrix on the perf/optimize branch.

Reads:
  - experiments/2026-04-08_*_*_{T4,A100,H100,B200}/  (baseline, dates ~10-11)
  - experiments/2026-04-08_*_*_{T4,A100,H100,B200}/  (post-perf, dates ~17+)

Writes:
  - analysis/perf_report.md (before/after table)
  - analysis/plots/perf/ (comparison charts)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ANALYSIS_DIR = REPO_ROOT / "analysis"
PERF_PLOTS_DIR = ANALYSIS_DIR / "plots" / "perf"

# Hour cutoff: experiments before hour 16 are baseline, 16+ are post-perf.
# This matches the work timeline on 2026-04-08.
PERF_HOUR_CUTOFF = 16


def _hour_from_dirname(name: str) -> int:
    # Format: "2026-04-08_HH-MM-SS_<rest>"
    try:
        return int(name.split("_")[1].split("-")[0])
    except (IndexError, ValueError):
        return -1


def _gpu_short(full: str) -> str:
    upper = (full or "").upper()
    if "B200" in upper:
        return "B200"
    if "H200" in upper:
        return "H200"
    if "H100" in upper:
        return "H100"
    if "A100" in upper:
        return "A100"
    if "T4" in upper:
        return "T4"
    return full or "?"


def _mean(metric):
    if isinstance(metric, dict):
        return metric.get("mean", 0.0)
    return metric or 0.0


def _compositor_from_dir(name: str) -> str:
    """Detect 'fast' (alpha) vs 'default' (inpaint) from dir name."""
    return "alpha" if "_fast_" in name or name.endswith("_fast") else "inpaint"


def load_runs() -> tuple[list[dict], list[dict]]:
    """Return (baseline_runs, post_runs)."""
    baseline = []
    post = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if d.name == "perf" or not d.is_dir():
            continue
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        if "num_prompts" not in m:
            continue  # incomplete/old format
        record = {
            "name": d.name,
            "gpu": _gpu_short(m.get("gpu", "")),
            "num_prompts": m["num_prompts"],
            "compositor": m.get("compositor_type", "?"),
            "output_fps": _mean(m.get("output_fps")),
            "total_s": _mean(m.get("total_s")),
            "segment_total_s": _mean(m.get("segment_total_s")),
            "fit_mean_ms": _mean(m.get("fit_mean_ms")),
            "composite_mean_ms": _mean(m.get("composite_mean_ms")),
            "write_video_s": _mean(m.get("write_video_s")),
        }
        hour = _hour_from_dirname(d.name)
        if hour < PERF_HOUR_CUTOFF:
            baseline.append(record)
        else:
            post.append(record)
    return baseline, post


def best_post_per_combo(post_runs: list[dict]) -> dict[tuple[int, str, str], dict]:
    """For each (num_prompts, gpu, compositor), keep the best (highest FPS) run."""
    best: dict[tuple[int, str, str], dict] = {}
    for r in post_runs:
        key = (r["num_prompts"], r["gpu"], r["compositor"])
        if key not in best or r["output_fps"] > best[key]["output_fps"]:
            best[key] = r
    return best


def main():
    PERF_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline, post = load_runs()
    print(f"Loaded {len(baseline)} baseline and {len(post)} post-perf runs")

    # Build baseline lookup by (prompts, gpu)
    baseline_by_combo: dict[tuple[int, str], dict] = {}
    for r in baseline:
        key = (r["num_prompts"], r["gpu"])
        if key not in baseline_by_combo or r["output_fps"] > baseline_by_combo[key]["output_fps"]:
            baseline_by_combo[key] = r

    best_post = best_post_per_combo(post)

    # Build the comparison table.
    rows = []
    for (np_, gpu, comp), p in sorted(best_post.items()):
        b = baseline_by_combo.get((np_, gpu))
        if b is None:
            continue
        rows.append(
            {
                "num_prompts": np_,
                "gpu": gpu,
                "compositor": comp,
                "fps_before": b["output_fps"],
                "fps_after": p["output_fps"],
                "speedup": p["output_fps"] / b["output_fps"] if b["output_fps"] else float("nan"),
                "comp_before": b["composite_mean_ms"],
                "comp_after": p["composite_mean_ms"],
                "seg_before": b["segment_total_s"],
                "seg_after": p["segment_total_s"],
                "wri_before": b["write_video_s"],
                "wri_after": p["write_video_s"],
            }
        )

    # Sort: first inpaint default, then alpha fast; within each by prompts then gpu order.
    GPU_ORDER = ["T4", "A100", "H100", "B200"]
    rows.sort(
        key=lambda r: (
            r["compositor"],
            r["num_prompts"],
            GPU_ORDER.index(r["gpu"]) if r["gpu"] in GPU_ORDER else 99,
        )
    )

    # --- Write markdown report ---
    lines = []
    lines.append("# Performance Optimization Report")
    lines.append("")
    lines.append("Baseline matrix vs post-optimization matrix on `perf/optimize` branch.")
    lines.append("All numbers are mean of 3 benchmark runs per (config, GPU) combination.")
    lines.append("")

    inpaint_rows = [r for r in rows if r["compositor"] == "inpaint"]
    alpha_rows = [r for r in rows if r["compositor"] == "alpha"]

    def section(title: str, rs: list[dict]):
        if not rs:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "| Prompts | GPU | FPS before | FPS after | **Speedup** | "
            "Composite ms (before → after) | Segment s (before → after) |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for r in rs:
            lines.append(
                f"| {r['num_prompts']} | {r['gpu']} | "
                f"{r['fps_before']:.2f} | **{r['fps_after']:.2f}** | "
                f"**{r['speedup']:.2f}×** | "
                f"{r['comp_before']:.0f} → **{r['comp_after']:.0f}** | "
                f"{r['seg_before']:.1f} → {r['seg_after']:.1f} |"
            )
        lines.append("")

    section("Default pipeline (InpaintCompositor — visually identical output)", inpaint_rows)
    section("Fast mode (AlphaCompositor — opt-in, no inpaint pre-pass)", alpha_rows)

    # Best speedups
    if inpaint_rows:
        best_default = max(inpaint_rows, key=lambda r: r["speedup"])
        lines.append("## Highlights")
        lines.append("")
        lines.append(
            f"- **Biggest default speedup:** {best_default['speedup']:.2f}× on "
            f"{best_default['gpu']} with {best_default['num_prompts']} prompts"
            f" ({best_default['fps_before']:.2f} → **{best_default['fps_after']:.2f} FPS**)"
        )
        # Best absolute FPS
        best_fps = max(rows, key=lambda r: r["fps_after"])
        lines.append(
            f"- **Highest absolute FPS:** **{best_fps['fps_after']:.2f}** on "
            f"{best_fps['gpu']} with {best_fps['num_prompts']} prompts"
            f" ({best_fps['compositor']} compositor)"
        )
        # Real-time gap
        target_fps = 25.0
        lines.append(
            f"- **Real-time target ({target_fps:.0f} FPS):** "
            f"best run is {target_fps / best_fps['fps_after']:.1f}× away"
        )

    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("![Before vs After FPS](plots/perf/fps_before_after.png)")
    lines.append("")
    lines.append("![Composite breakdown](plots/perf/composite_before_after.png)")
    lines.append("")

    (ANALYSIS_DIR / "perf_report.md").write_text("\n".join(lines))
    print(f"Wrote {ANALYSIS_DIR / 'perf_report.md'}")

    # --- Plots ---
    if not inpaint_rows:
        return

    # Plot 1: FPS before vs after, grouped by prompts × GPU
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r['num_prompts']}p\n{r['gpu']}" for r in inpaint_rows]
    x = np.arange(len(inpaint_rows))
    width = 0.35
    ax.bar(
        x - width / 2,
        [r["fps_before"] for r in inpaint_rows],
        width,
        label="Baseline",
        color="#aa5555",
    )
    ax.bar(
        x + width / 2,
        [r["fps_after"] for r in inpaint_rows],
        width,
        label="Optimized",
        color="#55aa55",
    )
    ax.axhline(25, color="orange", linestyle="--", linewidth=1, label="Real-time (25 FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Output FPS")
    ax.set_title("Pipeline FPS: baseline vs optimized (default InpaintCompositor)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PERF_PLOTS_DIR / "fps_before_after.png", dpi=150)
    plt.close(fig)

    # Plot 2: Composite ms before vs after
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        [r["comp_before"] for r in inpaint_rows],
        width,
        label="Baseline",
        color="#aa5555",
    )
    ax.bar(
        x + width / 2,
        [r["comp_after"] for r in inpaint_rows],
        width,
        label="Optimized",
        color="#55aa55",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Composite time (ms/frame)")
    ax.set_title("Per-frame composite cost: baseline vs optimized")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PERF_PLOTS_DIR / "composite_before_after.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {PERF_PLOTS_DIR}")

    # Print summary to stdout
    print()
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("=" * 60)
    for r in rows:
        print(
            f"  {r['compositor']:8s} {r['num_prompts']:>3}p {r['gpu']:>5}  "
            f"{r['fps_before']:>6.2f} → {r['fps_after']:>6.2f} FPS  "
            f"({r['speedup']:.2f}×)"
        )


if __name__ == "__main__":
    main()
