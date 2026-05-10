#!/usr/bin/env python3
"""Full evaluation: quality metrics + visual crop extraction.

Usage:
    python scripts/full_eval.py \
        --experiment experiments/<dir> \
        --original data/<video>

Runs quality_eval.py metrics, then extracts crops for visual inspection.
Prints a pass/fail summary and the visual QA checklist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def extract_all_crops(experiment_dir: str, original_video: str) -> None:
    """Extract comprehensive crops for visual inspection."""
    composited = os.path.join(experiment_dir, "outputs", "composited.mp4")
    crops_dir = os.path.join(experiment_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    cap = cv2.VideoCapture(composited)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 6 evenly spaced frames
    frame_indices = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue

        # Full frame
        cv2.imwrite(os.path.join(crops_dir, f"full_{fi:04d}.png"), frame)

        # Banner strip (top of frame, full width including ANZ)
        banner_y0, banner_y1 = 5, min(90, h)
        banner = frame[banner_y0:banner_y1, 50 : w - 50]
        banner_up = cv2.resize(
            banner,
            (banner.shape[1] * 2, banner.shape[0] * 3),
            interpolation=cv2.INTER_LANCZOS4,
        )
        cv2.imwrite(os.path.join(crops_dir, f"banner_{fi:04d}.png"), banner_up)

        # Court floor — bottom portion (MELBOURNE area)
        melb_y0 = max(0, h - 200)
        melb_region = frame[melb_y0:h, w // 4 : 3 * w // 4]
        if melb_region.size > 0:
            melb_up = cv2.resize(
                melb_region,
                (melb_region.shape[1] * 2, melb_region.shape[0] * 2),
                interpolation=cv2.INTER_LANCZOS4,
            )
            cv2.imwrite(os.path.join(crops_dir, f"court_bottom_{fi:04d}.png"), melb_up)

        # Court floor — left area (blua/YoPRO)
        left_court = frame[h // 3 : 2 * h // 3, 50 : w // 3]
        if left_court.size > 0:
            left_up = cv2.resize(
                left_court,
                (left_court.shape[1] * 2, left_court.shape[0] * 2),
                interpolation=cv2.INTER_LANCZOS4,
            )
            cv2.imwrite(os.path.join(crops_dir, f"court_left_{fi:04d}.png"), left_up)

        # Side panels (KIA area, left side)
        side = frame[h // 6 : h // 3, 0 : w // 5]
        if side.size > 0:
            side_up = cv2.resize(
                side,
                (side.shape[1] * 3, side.shape[0] * 3),
                interpolation=cv2.INTER_LANCZOS4,
            )
            cv2.imwrite(os.path.join(crops_dir, f"side_panels_{fi:04d}.png"), side_up)

    cap.release()
    print(f"Extracted crops for {len(frame_indices)} frames -> {crops_dir}")


def run_quality_metrics(experiment_dir: str, original_video: str) -> dict:
    """Run quality_eval.py and return results."""
    # Import the quality eval module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from scripts.quality_eval import run_evaluation

    results = run_evaluation(experiment_dir, original_video)

    # Save results (convert numpy types for JSON)
    metrics_path = os.path.join(experiment_dir, "quality_metrics.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, list):
            continue
        if isinstance(v, np.bool_ | np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, bool):
            v = int(v)
        serializable[k] = v
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return results


def print_summary(results: dict, experiment_dir: str) -> bool:
    """Print evaluation summary. Returns True if all metrics pass."""
    print("\n" + "=" * 60)
    print("FULL EVALUATION SUMMARY")
    print("=" * 60)

    checks = {
        "jitter_ratio": ("Jitter ratio", lambda v: v <= 1.05, "≤ 1.05"),
        "corner_max_jump_px": ("Corner max jump", lambda v: v < 2.0, "< 2.0 px"),
        "logo_area_cv": ("Logo area CV", lambda v: v < 0.05, "< 0.05"),
        "overlay_accel_p95_px": ("Overlay accel p95", lambda v: v < 1.0, "< 1.0 px"),
        "inpaint_color_de": ("Inpaint color dE", lambda v: v < 5.0, "< 5.0"),
        "temporal_ssim_mean": ("Temporal SSIM", lambda v: v > 0.95, "> 0.95"),
    }

    all_pass = True
    for key, (name, check_fn, threshold) in checks.items():
        value = results.get(key)
        if value is None:
            status = "  [?]"
            all_pass = False
        elif check_fn(value):
            status = "  [+]"
        else:
            status = "  [x]"
            all_pass = False
        val_str = f"{value:.4f}" if value is not None else "N/A"
        print(f"{status} {name:<30s} {val_str:>10s}  ({threshold})")

    print()
    if all_pass:
        print("  RESULT: ALL METRICS PASS ✓")
    else:
        print("  RESULT: SOME METRICS FAIL ✗")

    print()
    print("VISUAL QA CHECKLIST (check crops manually):")
    print(f"  Crops directory: {experiment_dir}/crops/")
    print("  [ ] Original text fully erased?")
    print("  [ ] Logo correctly positioned?")
    print("  [ ] Logo size appropriate?")
    print("  [ ] Color matches surface?")
    print("  [ ] No visible patch boundary?")
    print("  [ ] No artifacts?")
    print("  [ ] Player occlusion correct?")
    print("  [ ] Perspective plausible?")
    print("  [ ] Would a viewer notice this is fake?")
    print("=" * 60)

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Full experiment evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment directory")
    parser.add_argument("--original", required=True, help="Original video path")
    args = parser.parse_args()

    if not os.path.isdir(args.experiment):
        print(f"Error: {args.experiment} not found")
        sys.exit(1)

    # Step 1: Extract crops
    print("=" * 60)
    print("STEP 1: Extracting visual inspection crops")
    print("=" * 60)
    extract_all_crops(args.experiment, args.original)

    # Step 2: Run quality metrics
    print()
    print("=" * 60)
    print("STEP 2: Running quality metrics")
    print("=" * 60)
    results = run_quality_metrics(args.experiment, args.original)

    # Step 3: Print summary
    all_pass = print_summary(results, args.experiment)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
