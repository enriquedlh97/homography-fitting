"""Calibrate court_rect for an obj_3 placement_quad on frame 0.

Usage:
    uv run python scripts/calibrate_court_rect.py \
        --config configs/experiments/eval_walkover_p2_c005_a1_v68static_tol99999.yaml

Reads the input video's frame 0, runs CourtGeometryEstimator, computes
H_inv on the obj_3 placement_quad to recover court-plane points, and
prints a calibrated [u0, v0, u1, v1] court_rect (axis-aligned bbox in
court-plane coordinates).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from banner_pipeline import court_geometry as cg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--obj-id", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=8,
                        help="Run estimator on N frames first to let smoothing settle.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    video_path = cfg["input"]["video"]
    pipeline_cfg = cfg.get("pipeline", {})
    geom_cfg = cg.GeometryConfig.from_dict(pipeline_cfg.get("geometry"))

    prompts = cfg.get("objects") or cfg["input"]["prompts"]
    obj = next(o for o in prompts if int(o["obj_id"]) == args.obj_id)
    quad_img = np.asarray(obj["placement_quad"], dtype=np.float64).reshape(-1, 1, 2)
    print(f"obj_{args.obj_id} placement_quad (image): {quad_img.reshape(-1, 2).tolist()}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}", file=sys.stderr)
        return 1

    estimator = cg.CourtGeometryEstimator(geom_cfg)
    H = None
    for i in range(args.warmup_frames):
        ok, frame = cap.read()
        if not ok:
            break
        est = estimator.estimate(frame)
        if est.court_homography is not None:
            H = est.court_homography
    cap.release()

    if H is None:
        print("ERROR: CourtGeometryEstimator never produced a homography.", file=sys.stderr)
        return 2

    print(f"H (court→image), used after {args.warmup_frames} warmup frames:\n{H}")
    H_inv = np.linalg.inv(H.astype(np.float64))
    quad_court = cv2.perspectiveTransform(quad_img.astype(np.float32),
                                          H_inv.astype(np.float32)).reshape(-1, 2)
    print(f"placement_quad in court plane:\n{quad_court}")

    u0 = float(np.min(quad_court[:, 0]))
    u1 = float(np.max(quad_court[:, 0]))
    v0 = float(np.min(quad_court[:, 1]))
    v1 = float(np.max(quad_court[:, 1]))

    print()
    print("Calibrated court_rect (axis-aligned bbox of placement_quad in court plane):")
    print(f"  [u0, v0, u1, v1] = [{u0:.4f}, {v0:.4f}, {u1:.4f}, {v1:.4f}]")
    print()
    print("YAML form (rect — convenient but lossy due to foreshortening):")
    print("    court_plane_placement:")
    print("      target: corners")
    print(f"      court_rect: [{u0:.4f}, {v0:.4f}, {u1:.4f}, {v1:.4f}]")
    print()
    print("YAML form (quad — preserves the exact image-space placement_quad):")
    print("    court_plane_placement:")
    print("      target: corners")
    print("      court_quad:")
    for u, v in quad_court:
        print(f"      - [{u:.4f}, {v:.4f}]")

    # Sanity check: project back through H, compare to original.
    rect = np.array([[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]], dtype=np.float32)
    back = cv2.perspectiveTransform(rect, H.astype(np.float32)).reshape(-1, 2)
    print()
    print(f"Sanity: projected calibrated rect back to image: {back.tolist()}")
    print(f"        original placement_quad:                  {quad_img.reshape(-1, 2).tolist()}")
    diffs = back - quad_img.reshape(-1, 2)
    print(f"        max per-corner diff: {np.max(np.abs(diffs)):.2f}px")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
