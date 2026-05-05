"""Recalibrate court_quad against the MEDIAN per-frame H.

P2-C008 + scripts/dump_estimator_displacement.py revealed that the
line-based CourtGeometryEstimator has BIASED-but-stable H estimates:
median absolute displacement vs v68 placement_quad seed is 23 px, but
frame-to-frame |Δdisp| is only 4 px. So a single-frame calibration
(scripts/calibrate_court_rect.py uses warmup-8 frame) gives a court_quad
that is wrong by ~23 px on most frames.

This script computes per-frame "ideal court_quad" (the court_quad that
would project exactly to v68 placement_quad on that frame) and takes
the median across all frames. If the bias is consistent, median ideal
court_quad should produce much smaller per-frame displacement when
applied uniformly.
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
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    video_path = cfg["input"]["video"]
    pipeline_cfg = cfg.get("pipeline", {})
    geom_cfg = cg.GeometryConfig.from_dict(pipeline_cfg.get("geometry"))

    prompts = cfg.get("objects") or cfg["input"]["prompts"]
    obj = next(o for o in prompts if int(o["obj_id"]) == args.obj_id)
    seed = np.asarray(obj["placement_quad"], dtype=np.float64).reshape(1, 4, 2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 1

    estimator = cg.CourtGeometryEstimator(geom_cfg)
    per_frame_court_quads = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        est = estimator.estimate(frame)
        if est.court_homography is None:
            continue
        H_inv = np.linalg.inv(est.court_homography.astype(np.float64))
        cq = cv2.perspectiveTransform(seed.astype(np.float32),
                                      H_inv.astype(np.float32)).reshape(-1, 2)
        per_frame_court_quads.append(cq)
    cap.release()

    arr = np.stack(per_frame_court_quads)  # (N_frames, 4, 2)
    print(f"frames with valid H: {len(arr)}")

    median_cq = np.median(arr, axis=0)
    mean_cq = np.mean(arr, axis=0)

    print(f"median court_quad:\n{median_cq}")
    print(f"mean court_quad:\n{mean_cq}")
    print(f"per-corner stddev across frames:\n{np.std(arr, axis=0)}")

    # Predict: with median court_quad applied to every frame's H, what's
    # the per-frame max displacement vs seed?
    print("\nPredicted displacement distribution if we use MEDIAN court_quad:")
    seed_xy = seed.reshape(-1, 2)

    cap = cv2.VideoCapture(video_path)
    estimator = cg.CourtGeometryEstimator(geom_cfg)
    disps = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        est = estimator.estimate(frame)
        if est.court_homography is None:
            continue
        proj = cv2.perspectiveTransform(median_cq.reshape(1, 4, 2).astype(np.float32),
                                        est.court_homography.astype(np.float32)).reshape(-1, 2)
        disps.append(np.max(np.linalg.norm(proj - seed_xy, axis=1)))
    cap.release()

    disps = np.array(disps)
    print(f"  max_disp_px: mean={disps.mean():.2f} median={np.median(disps):.2f} p95={np.percentile(disps,95):.2f} max={disps.max():.2f}")
    for thr in (4, 8, 15, 30):
        n = int((disps > thr).sum())
        print(f"  frames with max_disp > {thr}px: {n}/{len(disps)} ({n/len(disps)*100:.0f}%)")

    print()
    print("YAML form (median-calibrated court_quad):")
    print("    court_plane_placement:")
    print("      target: corners")
    print("      court_quad:")
    for u, v in median_cq:
        print(f"      - [{u:.4f}, {v:.4f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
