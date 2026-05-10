#!/usr/bin/env python3
"""Build a clean video whose court quad is replaced by a temporal median plate.

This is a lightweight post-process for clean videos produced by inpainting
models. It targets moving residue, such as player-shaped smears, while keeping
the rest of the clean video unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_quad(quad_text: str) -> np.ndarray:
    """Parse a JSON quad like [[x,y], ...] into a float32 array."""
    quad = np.array(json.loads(quad_text), dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError("quad must be a JSON list of four [x, y] points")
    return quad


def read_video_size(video_path: Path) -> tuple[int, int]:
    """Return video size as (width, height)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return width, height


def scale_quad_to_video(
    quad: np.ndarray,
    source_video_path: Path,
    reference_video_path: Path | None,
) -> np.ndarray:
    """Scale a quad from reference video coordinates into source video coordinates."""
    if reference_video_path is None:
        return quad

    source_width, source_height = read_video_size(source_video_path)
    reference_width, reference_height = read_video_size(reference_video_path)
    scale = np.array(
        [source_width / reference_width, source_height / reference_height],
        dtype=np.float32,
    )
    return quad * scale


def build_quad_mask(frame_shape: tuple[int, int, int], quad: np.ndarray) -> np.ndarray:
    """Build a uint8 mask for the quad in frame coordinates."""
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    return mask


def collect_quad_roi_frames(
    clean_video_path: Path,
    quad_mask: np.ndarray,
) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    """Read all frame ROIs needed to compute the temporal median."""
    nonzero_y, nonzero_x = np.where(quad_mask > 0)
    if len(nonzero_x) == 0 or len(nonzero_y) == 0:
        raise RuntimeError("quad mask is empty")

    x_min = int(nonzero_x.min())
    x_max = int(nonzero_x.max()) + 1
    y_min = int(nonzero_y.min())
    y_max = int(nonzero_y.max()) + 1
    roi_bounds = (x_min, y_min, x_max, y_max)

    capture = cv2.VideoCapture(str(clean_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {clean_video_path}")

    roi_frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        roi_frames.append(frame[y_min:y_max, x_min:x_max].copy())
    capture.release()

    if not roi_frames:
        raise RuntimeError("no frames were read")
    return roi_frames, roi_bounds


def write_temporal_median_video(
    clean_video_path: Path,
    output_video_path: Path,
    quad_mask: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
    median_roi: np.ndarray,
) -> None:
    """Write a copy of the clean video with median pixels inside the quad."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(clean_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {clean_video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 59.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output writer: {output_video_path}")

    x_min, y_min, x_max, y_max = roi_bounds
    roi_mask = quad_mask[y_min:y_max, x_min:x_max] > 0
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        roi = frame[y_min:y_max, x_min:x_max].copy()
        roi[roi_mask] = median_roi[roi_mask]
        frame[y_min:y_max, x_min:x_max] = roi
        writer.write(frame)
        frame_count += 1

    capture.release()
    writer.release()
    print(f"Wrote {frame_count} frames -> {output_video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace a clean video's quad region with its temporal median plate.",
    )
    parser.add_argument("--clean-video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quad", required=True, help="JSON quad in reference/video coordinates")
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=None,
        help="If set, scale --quad from this video's size to the clean-video size.",
    )
    args = parser.parse_args()

    quad = parse_quad(args.quad)
    scaled_quad = scale_quad_to_video(quad, args.clean_video, args.reference_video)

    capture = cv2.VideoCapture(str(args.clean_video))
    ok, first_frame = capture.read()
    capture.release()
    if not ok or first_frame is None:
        raise RuntimeError(f"Could not read first frame: {args.clean_video}")

    quad_mask = build_quad_mask(first_frame.shape, scaled_quad)
    roi_frames, roi_bounds = collect_quad_roi_frames(args.clean_video, quad_mask)
    median_roi = np.median(np.stack(roi_frames, axis=0), axis=0).astype(np.uint8)
    write_temporal_median_video(
        args.clean_video,
        args.output,
        quad_mask,
        roi_bounds,
        median_roi,
    )


if __name__ == "__main__":
    main()
