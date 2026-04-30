#!/usr/bin/env python3
"""Run the banner pipeline on Modal using the SAM3 text-prompted segmenter.

The pipeline is identical to ``modal_run.py`` but builds a SAM3 image
(facebookresearch/sam3) instead of SAM2 and accepts a natural-language prompt
("logo" by default). Detection is fully automatic — no click coordinates.

Usage
-----
    # First time:
    uv run modal setup
    # Make sure a `hf-token` secret exists in your Modal workspace.

    uv run modal run scripts/modal_run_sam3.py \
        --config configs/experiments/sam3_auto.yaml \
        --gpu A100-80GB --mode video
"""

from __future__ import annotations

import sys

import modal

# ---------------------------------------------------------------------------
# Parse --gpu before decorators run.
# ---------------------------------------------------------------------------

_GPU = "A100-80GB"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _GPU = sys.argv[i + 1]
        break

# ---------------------------------------------------------------------------
# Modal image: Linux + CUDA torch + SAM3 + our pipeline code
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git", "build-essential")
    .pip_install(
        "torch>=2.0",
        "torchvision>=0.15",
        "opencv-python-headless>=4.8",
        "matplotlib>=3.7",
        "Pillow>=10.0",
        "scipy>=1.11",
        "pyyaml>=6.0",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /root/sam3",
        "cd /root/sam3 && pip install -e '.[notebooks]'",
        "pip install 'numpy>=1.26,<2' supervision==0.27.0.post2",
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("banner-pipeline-sam3", image=image)


# ---------------------------------------------------------------------------
# Remote GPU function
# ---------------------------------------------------------------------------


@app.function(
    gpu=_GPU,
    timeout=86400,
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_on_gpu(
    config_dict: dict,
    video_bytes: bytes,
    logo_bytes: bytes | None,
    benchmark_runs: int = 1,
) -> dict:
    """Run the SAM3 pipeline on a GPU. Returns metrics + output bytes."""
    import os
    import sys
    import tempfile
    import time

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, "/root/src")

    from banner_pipeline.pipeline import run

    # --- Write input files to temp dir ---
    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "input.mp4")
    with open(video_path, "wb") as f:
        f.write(video_bytes)
    config_dict["input"]["video"] = video_path

    if logo_bytes:
        logo_path = os.path.join(tmpdir, "logo.png")
        with open(logo_path, "wb") as f:
            f.write(logo_bytes)
        config_dict["input"]["logo"] = logo_path

    # --- Report GPU info ---
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        raise RuntimeError("SAM3 requires a CUDA GPU.")

    mode = config_dict.get("pipeline", {}).get("mode", "video")
    output_ext = ".mp4" if mode == "video" else ".png"

    # --- Run pipeline (with optional benchmark) ---
    all_metrics: list[dict] = []
    output_bytes: bytes | None = None

    for i in range(benchmark_runs):
        if benchmark_runs > 1:
            print(f"\n=== Run {i + 1}/{benchmark_runs} ===")

        t_start = time.perf_counter()
        output_video_path = os.path.join(tmpdir, "output.mp4")
        results = run(config_dict, output_path=output_video_path)
        t_total = time.perf_counter() - t_start

        m = results.get("metrics", {})
        m["run_total_s"] = t_total
        m["gpu"] = gpu_name
        m["gpu_memory_gb"] = round(gpu_mem, 1)
        m["mode"] = mode
        all_metrics.append(m)

        if mode == "video" and results.get("output_path"):
            with open(results["output_path"], "rb") as f:
                output_bytes = f.read()
        elif results.get("composited") is not None:
            _, buf = cv2.imencode(".png", results["composited"])
            output_bytes = buf.tobytes()

    # --- Aggregate ---
    report: dict = {
        "runs": benchmark_runs,
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "mode": mode,
    }

    metadata_keys = [
        "num_prompts",
        "num_prompt_points",
        "num_detected_objects",
        "num_segmented_objects",
        "num_substituted_objects",
        "filter_min_area_frac",
        "filter_min_confidence",
        "filter_min_frame_count",
        "filter_rejected_by_area",
        "filter_rejected_by_confidence",
        "filter_rejected_by_persistence",
        "filter_rejected_total",
        "tracking_enabled",
        "num_frames",
        "input_fps",
        "duration_s",
        "frame_width",
        "frame_height",
        "video_path",
        "fitter_type",
        "compositor_type",
        "checkpoint",
    ]
    for key in metadata_keys:
        if all_metrics and key in all_metrics[0]:
            report[key] = all_metrics[0][key]

    timing_keys = [
        "load_frame_s",
        "segment_s",
        "segment_total_s",
        "fit_s",
        "fit_mean_ms",
        "composite_s",
        "composite_mean_ms",
        "write_video_s",
        "total_s",
        "run_total_s",
        "output_fps",
    ]
    if benchmark_runs > 1:
        for key in timing_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                report[key] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "min": round(float(np.min(values)), 4),
                    "max": round(float(np.max(values)), 4),
                }
    else:
        for key in timing_keys:
            if all_metrics and key in all_metrics[0]:
                report[key] = all_metrics[0][key]

    return {
        "metrics": report,
        "output_bytes": output_bytes,
        "output_ext": output_ext,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    config: str = "configs/experiments/sam3_auto.yaml",
    gpu: str = "A100-80GB",
    mode: str = "video",
    benchmark: int = 1,
    name: str = "",
    prompt_text: str = "",
):
    import json
    import os
    from datetime import datetime

    import yaml

    with open(config) as f:
        config_dict = yaml.safe_load(f)

    if mode:
        config_dict.setdefault("pipeline", {})["mode"] = mode

    # Optional CLI override: replace every prompt's text.
    if prompt_text:
        for p in config_dict.get("input", {}).get("prompts", []):
            p["text"] = prompt_text

    # Read input files as bytes.
    video_path = config_dict["input"]["video"]
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    print(f"Video: {video_path} ({len(video_bytes) / 1024:.0f} KB)")

    logo_bytes = None
    logo_path = config_dict["input"].get("logo")
    if logo_path and os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_bytes = f.read()
        print(f"Logo: {logo_path} ({len(logo_bytes) / 1024:.0f} KB)")

    print(f"GPU: {gpu}")
    print(f"Benchmark runs: {benchmark}")

    result = run_on_gpu.remote(
        config_dict=config_dict,
        video_bytes=video_bytes,
        logo_bytes=logo_bytes,
        benchmark_runs=benchmark,
    )

    # --- Save results locally ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fitter_type = config_dict["pipeline"]["fitter"]["type"]
    exp_name = name or f"sam3_{fitter_type}_{gpu}"
    exp_dir = os.path.join(config_dict["output"]["dir"], f"{timestamp}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    config_out = os.path.join(exp_dir, "config.yaml")
    with open(config_out, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    metrics = result["metrics"]
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if result.get("output_bytes"):
        out_dir = os.path.join(exp_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        ext = result.get("output_ext", ".mp4")
        out_path = os.path.join(out_dir, "composited" + ext)
        with open(out_path, "wb") as f:
            f.write(result["output_bytes"])
        print(f"Saved: {out_path}")

    print(f"\n{'=' * 50}")
    print(f"RESULTS — SAM3 / {gpu}")
    print(f"{'=' * 50}")
    print(json.dumps(metrics, indent=2))
    print(f"\nExperiment saved: {exp_dir}")
