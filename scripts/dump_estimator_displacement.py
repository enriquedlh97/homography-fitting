"""Dump per-frame court estimator displacement vs seed for one config.

Runs `CourtGeometryEstimator` over every frame of the input video and
computes for each frame: max-corner displacement between the seed
`placement_quad` and the projected court_quad (or court_rect) through
that frame's H. Writes a CSV for plotting / diagnostics.

Usage:
    uv run python scripts/dump_estimator_displacement.py \
        --config configs/experiments/eval_walkover_p2_c006_a1_court_quad_tol4.yaml
"""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    video_path = cfg["input"]["video"]
    pipeline_cfg = cfg.get("pipeline", {})
    geom_cfg = cg.GeometryConfig.from_dict(pipeline_cfg.get("geometry"))

    prompts = cfg.get("objects") or cfg["input"]["prompts"]
    obj = next(o for o in prompts if int(o["obj_id"]) == args.obj_id)
    seed = np.asarray(obj["placement_quad"], dtype=np.float64)
    cpp = obj.get("court_plane_placement") or {}
    court_quad = cpp.get("court_quad")
    court_rect = cpp.get("court_rect")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}", file=sys.stderr)
        return 1

    estimator = cg.CourtGeometryEstimator(geom_cfg)
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        est = estimator.estimate(frame)
        H = est.court_homography
        if H is None:
            rows.append((frame_idx, None, None, None, None))
            frame_idx += 1
            continue

        if court_quad is not None:
            quad_arr = np.asarray(court_quad, dtype=np.float32).reshape(1, 4, 2)
            projected = cv2.perspectiveTransform(quad_arr, H.astype(np.float32)).reshape(-1, 2)
        elif court_rect is not None:
            u0, v0, u1, v1 = court_rect
            quad_arr = np.array(
                [[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]], dtype=np.float32
            )
            projected = cv2.perspectiveTransform(quad_arr, H.astype(np.float32)).reshape(-1, 2)
        else:
            print("ERROR: no court_quad or court_rect", file=sys.stderr)
            return 2

        diffs = projected - seed
        per_corner = np.linalg.norm(diffs, axis=1)
        max_disp = float(np.max(per_corner))
        mean_disp = float(np.mean(per_corner))
        rows.append((frame_idx, max_disp, mean_disp,
                     float(per_corner[0]), float(per_corner[2])))
        frame_idx += 1
    cap.release()

    out_csv = args.out_csv or f"/tmp/estimator_displacement_{Path(args.config).stem}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "max_disp_px", "mean_disp_px", "TL_disp", "BR_disp"])
        for r in rows:
            w.writerow(r)

    valid = [r for r in rows if r[1] is not None]
    if valid:
        max_disps = np.array([r[1] for r in valid])
        print(f"Wrote {out_csv}")
        print(f"Frames: {len(rows)} (valid H: {len(valid)})")
        print(f"max_disp_px: mean={max_disps.mean():.2f} median={np.median(max_disps):.2f}")
        print(f"            p5={np.percentile(max_disps,5):.2f} p25={np.percentile(max_disps,25):.2f}")
        print(f"            p50={np.percentile(max_disps,50):.2f} p75={np.percentile(max_disps,75):.2f}")
        print(f"            p95={np.percentile(max_disps,95):.2f} max={max_disps.max():.2f}")
        for thr in (4, 8, 15, 30):
            pct = float((max_disps > thr).mean()) * 100
            print(f"  frames with max_disp > {thr}px: {(max_disps > thr).sum()}/{len(valid)} ({pct:.0f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
