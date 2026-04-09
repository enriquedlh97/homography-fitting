#!/usr/bin/env python3
"""Collect click prompts interactively and save them to a config YAML.

Opens the first frame of the video, lets you click on banner regions,
and writes the coordinates into the config. No SAM2, no GPU needed.

Usage
-----
    # Click on regions, coordinates saved into the config
    uv run python scripts/collect_prompts.py --config configs/default.yaml

    # Use a different video
    uv run python scripts/collect_prompts.py --config configs/default.yaml --video path/to/video.mp4
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


def _next_modal_command(config_path: str, segmenter_type: str) -> str:
    gpu = _recommended_gpu_for_segmenter(segmenter_type)
    return f"uv run modal run scripts/modal_run.py --config {config_path} --gpu {gpu}"


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

    click_groups = collect_clicks(frame)
    if not click_groups:
        print("No clicks — config unchanged.")
        return

    # Convert clicks to prompt format.
    prompts = []
    for idx, group in enumerate(click_groups):
        prompts.append(
            {
                "obj_id": idx + 1,
                "points": [[x, y] for x, y in group],
            }
        )

    config["input"]["prompts"] = prompts

    with open(args.config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved {len(prompts)} prompt(s) to: {args.config}")
    for p in prompts:
        pts = ", ".join(f"({x},{y})" for x, y in p["points"])
        print(f"  obj {p['obj_id']}: {pts}")
    print("\nNow run:")
    print(f"  {_next_modal_command(args.config, segmenter_type)}")
    if segmenter_type == "sam3_video":
        print("  SAM3 supports L4/A10G/L40S/A100/A100-80GB/H100/H200/B200; T4 is rejected.")


if __name__ == "__main__":
    main()
