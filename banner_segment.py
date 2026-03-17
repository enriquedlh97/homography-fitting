# -*- coding: utf-8 -*-
"""
banner_segment.py
-----------------
Interactive banner segmentation + parallelogram fitting.

1. Show frame 0 of a video
2. Click on banner regions (left-click = add point, N = next object, Enter/Space = done)
3. SAM2 segments each clicked object
4. Fit a parallelogram to each mask
5. Visualize results

Usage
-----
    python banner_segment.py video.mp4
"""

from __future__ import annotations
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Click UI
# ---------------------------------------------------------------------------

OBJ_COLORS_UI = [
    (0, 255, 0), (0, 100, 255), (0, 0, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
]


def collect_clicks(frame: np.ndarray) -> list[list[tuple[int, int]]]:
    """Show frame in an OpenCV window; collect grouped clicks.

    - Left-click : add point to current object
    - N          : finish current object, start a new one
    - Enter/Space: finish all
    - Escape     : cancel

    Returns a list of groups, e.g. [[(x1,y1),(x2,y2)], [(x3,y3)], ...]
    """
    groups: list[list[tuple[int, int]]] = [[]]
    display = frame.copy()
    win = "Click banners (N=next object, Enter=done, Esc=cancel)"

    def current_color():
        return OBJ_COLORS_UI[(len(groups) - 1) % len(OBJ_COLORS_UI)]

    def redraw_status():
        obj_idx = len(groups)
        n_pts = len(groups[-1])
        label = f"Object {obj_idx}  ({n_pts} pts)  |  N=next  Enter=done"
        cv2.rectangle(display, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(display, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, display)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            groups[-1].append((x, y))
            col = current_color()
            pt_idx = len(groups[-1])
            cv2.drawMarker(display, (x, y), col, cv2.MARKER_STAR, 20, 2)
            cv2.putText(display, f"{len(groups)}.{pt_idx}", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
            redraw_status()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, on_mouse)
    redraw_status()

    print("[UI] Left-click to add points. N = next object. Enter/Space = done. Esc = cancel.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # Enter or Space
            break
        if key == 27:  # Escape
            groups.clear()
            break
        if key in (ord('n'), ord('N')):
            if groups[-1]:  # only advance if current group has points
                print(f"  Object {len(groups)} done ({len(groups[-1])} pts). Starting object {len(groups) + 1}…")
                groups.append([])
                redraw_status()

    cv2.destroyAllWindows()
    # Drop trailing empty group (if user pressed N then immediately Enter)
    return [g for g in groups if g]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame0(video_path: str) -> np.ndarray:
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vframes", "1", "-q:v", "2",
         tmp, "-y", "-loglevel", "error"],
        check=True,
    )
    frame = cv2.imread(tmp)
    os.unlink(tmp)
    if frame is None:
        raise RuntimeError(f"Could not read frame from {video_path}")
    return frame


# ---------------------------------------------------------------------------
# SAM2 (image predictor — fast, single-frame)
# ---------------------------------------------------------------------------

def run_sam2(frame_bgr: np.ndarray, click_groups: list[list[tuple[int, int]]],
             checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
             model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"):
    """Run SAM2 image predictor: one group of positive clicks per object.
    Returns dict[obj_id] -> binary mask (H, W) for the given frame."""
    import sys
    # The sam2 repo dir shadows the installed package — point Python inside it
    _repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  SAM2 device: {device}", flush=True)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print("  Loading model …", flush=True)
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print("  Setting image …", flush=True)
    predictor.set_image(frame_rgb)

    masks_out: dict[int, np.ndarray] = {}
    for idx, group in enumerate(click_groups):
        obj_id = idx + 1
        point_coords = np.array(group, dtype=np.float32)
        point_labels = np.ones(len(group), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best = masks[np.argmax(scores)]
        masks_out[obj_id] = best
        pts_str = ", ".join(f"({x},{y})" for x, y in group)
        print(f"  Obj {obj_id}: {len(group)} pts [{pts_str}] → score={scores.max():.3f}", flush=True)

    print(f"  Got {len(masks_out)} masks", flush=True)
    return masks_out


# ---------------------------------------------------------------------------
# Perspective-aware quadrilateral fitting
# ---------------------------------------------------------------------------

def _fit_line_pts(pts: np.ndarray):
    """Fit a line to 2D points. Returns (point_on_line, unit_direction)."""
    vx, vy, cx, cy = cv2.fitLine(pts.reshape(-1, 1, 2).astype(np.float32),
                                  cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([cx, cy], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def fit_quadrilateral(mask: np.ndarray) -> np.ndarray | None:
    """Fit a perspective-aware quadrilateral to a binary mask.

    Splits contour into top/bottom edge points and fits an independent line to
    each, capturing natural perspective convergence. Returns [TL, TR, BR, BL]
    or None on failure.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2).astype(np.float64)

    rect = cv2.minAreaRect(largest)
    center = np.array(rect[0])
    angle_deg = rect[2]
    w, h = rect[1]
    if w < h:
        angle_deg += 90

    angle_rad = np.deg2rad(angle_deg)
    short_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    rel = pts - center
    proj_short = rel @ short_dir

    top_pts = pts[proj_short >= 0]
    bot_pts = pts[proj_short < 0]
    print(f"  Top edge: {len(top_pts)} pts, Bottom edge: {len(bot_pts)} pts")

    if len(top_pts) < 5 or len(bot_pts) < 5:
        print("  [warn] Not enough points, falling back to minAreaRect")
        return cv2.boxPoints(rect).astype(np.float32)

    pt_top, dir_top = _fit_line_pts(top_pts)
    pt_bot, dir_bot = _fit_line_pts(bot_pts)

    if dir_top @ dir_bot < 0:
        dir_bot = -dir_bot

    d_avg = (dir_top + dir_bot) / 2.0
    d_avg /= np.linalg.norm(d_avg)

    proj_long = pts @ d_avg
    p_min, p_max = proj_long.min(), proj_long.max()

    def point_on_line(pt_on, d, target):
        denom = d @ d_avg
        if abs(denom) < 1e-9:
            return pt_on
        t = (target - pt_on @ d_avg) / denom
        return pt_on + t * d

    tl = point_on_line(pt_top, dir_top, p_min)
    tr = point_on_line(pt_top, dir_top, p_max)
    br = point_on_line(pt_bot, dir_bot, p_max)
    bl = point_on_line(pt_bot, dir_bot, p_min)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    angle_top = np.degrees(np.arctan2(dir_top[1], dir_top[0]))
    angle_bot = np.degrees(np.arctan2(dir_bot[1], dir_bot[0]))
    print(f"  Top line angle: {angle_top:.2f}°, Bottom line angle: {angle_bot:.2f}°")
    print(f"  Convergence: {abs(angle_top - angle_bot):.2f}°")

    # Sort into TL/TR/BR/BL
    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]
    return np.array([
        corners[np.argmin(s)],
        corners[np.argmax(d)],
        corners[np.argmax(s)],
        corners[np.argmin(d)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

OBJ_COLORS = [
    (0, 200, 0), (200, 0, 0), (0, 0, 200),
    (200, 200, 0), (200, 0, 200), (0, 200, 200),
]


def visualize(frame: np.ndarray,
              masks: dict[int, np.ndarray],
              corners_map: dict[int, np.ndarray],
              save_path: str = "banner_result.png"):
    vis = frame.copy()

    # Overlay masks
    for obj_id, mask in masks.items():
        col = np.array(OBJ_COLORS[obj_id % len(OBJ_COLORS)], dtype=np.uint8)
        m = mask.astype(bool)
        vis[m] = (vis[m].astype(np.float32) * 0.55 + col.astype(np.float32) * 0.45).astype(np.uint8)

    # Draw parallelogram quads
    corner_labels = ["TL", "TR", "BR", "BL"]
    for obj_id, corners in corners_map.items():
        col = OBJ_COLORS[obj_id % len(OBJ_COLORS)]
        quad = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [quad], isClosed=True, color=(255, 255, 255), thickness=2)
        for pt, clbl in zip(corners, corner_labels):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), 6, col, -1)
            cv2.putText(vis, f"{obj_id}-{clbl}", (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, vis)
    print(f"  Saved: {save_path}")

    # Warped top-down views
    warpeds = []
    for obj_id in sorted(corners_map):
        c = corners_map[obj_id]
        dst_w, dst_h = 300, 450
        dst = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
        H, _ = cv2.findHomography(c, dst)
        if H is not None:
            warpeds.append((obj_id, cv2.warpPerspective(frame, H, (dst_w, dst_h))))

    ncols = 1 + len(warpeds)
    ratios = [3] + [1] * len(warpeds)
    fig, axes = plt.subplots(1, ncols, figsize=(8 + 4 * len(warpeds), 6),
                             gridspec_kw={"width_ratios": ratios})
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Masks + parallelogram fit ({len(corners_map)} objects)")
    axes[0].axis("off")

    for i, (oid, warped) in enumerate(warpeds):
        axes[1 + i].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1 + i].set_title(f"Obj {oid} top-down")
        axes[1 + i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path.rsplit(".", 1)[0] + "_full.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive banner segmentation + parallelogram fit")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--save", default="banner_result.png", help="Output image path")
    parser.add_argument("--checkpoint", default="sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--mask-dir", default="masks", help="Directory to save masks + frame (default: masks/)")
    args = parser.parse_args()

    print("[1/4] Extracting frame 0 …", flush=True)
    frame = extract_frame0(args.video)

    print("[2/4] Collecting clicks …", flush=True)
    clicks = collect_clicks(frame)
    if not clicks:
        print("No clicks — exiting.")
        return
    print(f"  {len(clicks)} object(s): {clicks}", flush=True)

    print("[3/4] Running SAM2 (image mode) …", flush=True)
    masks = run_sam2(frame, clicks,
                     checkpoint=args.checkpoint, model_cfg=args.model_cfg)

    # Save masks + original frame for offline experimentation
    mask_dir = args.mask_dir
    os.makedirs(mask_dir, exist_ok=True)
    frame_path = os.path.join(mask_dir, "frame0.png")
    cv2.imwrite(frame_path, frame)
    print(f"  Saved frame: {frame_path}", flush=True)
    for obj_id, mask in masks.items():
        mask_path = os.path.join(mask_dir, f"mask_obj{obj_id}.png")
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
        print(f"  Saved mask: {mask_path}", flush=True)

    print("[4/4] Fitting parallelograms …", flush=True)
    corners_map = {}
    for obj_id, mask in masks.items():
        print(f"  Object {obj_id}:")
        corners = fit_quadrilateral(mask)
        if corners is not None:
            corners_map[obj_id] = corners
            for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
                print(f"    {lbl}: ({int(pt[0])}, {int(pt[1])})")

    print(f"  Visualizing ({len(corners_map)} parallelograms) …", flush=True)
    visualize(frame, masks, corners_map, save_path=args.save)
    print("Done.")


if __name__ == "__main__":
    main()
