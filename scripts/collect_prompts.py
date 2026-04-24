#!/usr/bin/env python3
"""Collect click prompts interactively and save them to a config YAML.

Opens a selected frame of the video, lets you click prompt points,
and writes the coordinates into the config. Saved prompts include
``points``, ``labels``, and ``frame_idx``. No SAM2, no GPU needed.

Usage
-----
    # Click on regions, coordinates saved into the config
    uv run python scripts/collect_prompts.py --config configs/default.yaml

    # Collect prompts on a later frame
    uv run python scripts/collect_prompts.py --config configs/sam3_default.yaml --frame 20

    # Use a different video
    uv run python scripts/collect_prompts.py --config configs/default.yaml --video path/to/video.mp4

    # For SAM3, preview the saved prompts before running video mode
    uv run modal run scripts/modal_run.py --config configs/sam3_default.yaml --gpu A100 --mode image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from banner_pipeline.io import load_frame
from banner_pipeline.ui import collect_clicks


def _segmenter_type(config: dict) -> str:
    return config.get("pipeline", {}).get("segmenter", {}).get("type", "sam2_image")


def _recommended_gpu_for_segmenter(segmenter_type: str) -> str:
    return "A100" if segmenter_type == "sam3_video" else "T4"


def _next_modal_command(config_path: str, segmenter_type: str, mode: str = "video") -> str:
    gpu = _recommended_gpu_for_segmenter(segmenter_type)
    return f"uv run modal run scripts/modal_run.py --config {config_path} --gpu {gpu} --mode {mode}"


def main():
    parser = argparse.ArgumentParser(description="Collect click prompts and save to config")
    parser.add_argument("--config", required=True, help="Path to YAML config (will be updated)")
    parser.add_argument("--video", default=None, help="Override video path (default: from config)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to display (default: 0)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    segmenter_type = _segmenter_type(config)

    video_path = args.video or config["input"]["video"]
    print(f"Loading frame {args.frame} from: {video_path}")
    frame = load_frame(video_path, frame_idx=args.frame)
    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")

    prompts = collect_clicks(frame, frame_idx=args.frame)
    if not prompts:
        print("No clicks — config unchanged.")
        return

    config["input"]["prompts"] = []
    for prompt in prompts:
        entry = {
            "obj_id": prompt.obj_id,
            "points": prompt.points.astype(int).tolist(),
            "labels": prompt.labels.astype(int).tolist(),
        }
        if prompt.frame_idx != 0:
            entry["frame_idx"] = int(prompt.frame_idx)
        if getattr(prompt, "surface_type", "banner") != "banner":
            entry["surface_type"] = str(prompt.surface_type)
        if getattr(prompt, "geometry_model", None) is not None:
            entry["geometry_model"] = str(prompt.geometry_model)
        config["input"]["prompts"].append(entry)

    with open(args.config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved {len(prompts)} prompt(s) to: {args.config}")
    for prompt in prompts:
        pts = ", ".join(
            f"({int(x)},{int(y)},{'+' if int(label) == 1 else '-'})"
            for (x, y), label in zip(prompt.points, prompt.labels, strict=True)
        )
        suffix = f" @ frame {prompt.frame_idx}" if prompt.frame_idx != 0 else ""
        print(f"  obj {prompt.obj_id}{suffix}: {pts}")
    print("\nNow run:")
    if segmenter_type == "sam3_video":
        print(f"  Preview: {_next_modal_command(args.config, segmenter_type, mode='image')}")
        print(f"  Video:   {_next_modal_command(args.config, segmenter_type, mode='video')}")
        print("  Prompting: start with 1-2 positive clicks inside each banner,")
        print("             then add negative clicks on nearby background if masks bleed.")
        print("  SAM3 supports L4/A10G/L40S/A100/A100-80GB/H100/H200/B200; T4 is rejected.")
    else:
        print(f"  {_next_modal_command(args.config, segmenter_type)}")


if __name__ == "__main__":
    main()
