"""Shared mask/quad quality helpers for preview and video validation."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

MIN_PRIMARY_FIT_AREA_RATIO = 5e-4
MIN_MASK_AREA_PX = 16
MIN_SMALL_BBOX_EDGE_PX = 8
MIN_SMALL_MASK_COMPACTNESS = 0.12
MAX_QUAD_AREA_RATIO = 0.55
MAX_QUAD_ASPECT_RATIO = 80.0
MIN_QUAD_EDGE_PX = 4.0
MIN_QUAD_AREA_PX = 16.0
MIN_QUAD_MASK_IOU = 0.03
MIN_MASK_COVERAGE = 0.08
MIN_QUAD_COVERAGE = 0.08


def mask_area_and_bbox(mask: np.ndarray | None) -> tuple[int, list[int] | None]:
    if mask is None:
        return 0, None
    mask_2d = np.asarray(mask).squeeze()
    if mask_2d.ndim != 2 or not mask_2d.any():
        return 0, None
    ys, xs = np.nonzero(mask_2d)
    return int(len(xs)), [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_dims(mask_bbox: list[int] | None) -> tuple[int, int]:
    if mask_bbox is None:
        return 0, 0
    x0, y0, x1, y1 = mask_bbox
    return x1 - x0 + 1, y1 - y0 + 1


def polygon_area(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def quad_edge_lengths(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def quad_mask_overlap(mask: np.ndarray, corners: np.ndarray) -> dict[str, float]:
    mask_2d = np.asarray(mask).squeeze().astype(bool)
    quad_mask = np.zeros(mask_2d.shape, dtype=np.uint8)
    polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(quad_mask, polygon, 255)
    quad_bool = quad_mask.astype(bool)
    intersection = int(np.logical_and(mask_2d, quad_bool).sum())
    union = int(np.logical_or(mask_2d, quad_bool).sum())
    quad_area = int(quad_bool.sum())
    mask_area = int(mask_2d.sum())
    return {
        "iou": 0.0 if union == 0 else intersection / union,
        "mask_coverage": 0.0 if mask_area == 0 else intersection / mask_area,
        "quad_coverage": 0.0 if quad_area == 0 else intersection / quad_area,
    }


def geometry_flags(
    mask: np.ndarray,
    corners: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[list[str], dict[str, float]]:
    height, width = frame_shape[:2]
    frame_area = float(height * width)
    edges = quad_edge_lengths(corners)
    quad_area = polygon_area(corners)
    aspect_ratio = float(edges.max() / max(edges.min(), 1e-6))
    overlap = quad_mask_overlap(mask, corners)

    stats = {
        "quad_area_px": round(quad_area, 2),
        "quad_aspect_ratio": round(aspect_ratio, 2),
        "quad_mask_iou": round(overlap["iou"], 4),
        "mask_coverage": round(overlap["mask_coverage"], 4),
        "quad_coverage": round(overlap["quad_coverage"], 4),
    }
    flags: list[str] = []
    if quad_area < MIN_QUAD_AREA_PX:
        flags.append("quad_area_too_small")
    if quad_area > frame_area * MAX_QUAD_AREA_RATIO:
        flags.append("quad_area_too_large")
    if float(edges.min()) < MIN_QUAD_EDGE_PX:
        flags.append("quad_edge_too_short")
    if aspect_ratio > MAX_QUAD_ASPECT_RATIO:
        flags.append("quad_aspect_ratio_too_high")
    if overlap["iou"] < MIN_QUAD_MASK_IOU:
        flags.append("quad_mask_iou_low")
    if overlap["mask_coverage"] < MIN_MASK_COVERAGE:
        flags.append("mask_coverage_low")
    if overlap["quad_coverage"] < MIN_QUAD_COVERAGE:
        flags.append("quad_coverage_low")

    corners_np = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if (
        (corners_np[:, 0] < -0.05 * width).any()
        or (corners_np[:, 0] > 1.05 * width).any()
        or (corners_np[:, 1] < -0.05 * height).any()
        or (corners_np[:, 1] > 1.05 * height).any()
    ):
        flags.append("quad_outside_frame")
    return flags, stats


def fit_min_area_rect_quad(mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    if mask_u8.ndim != 2 or not mask_u8.any():
        return None
    nonzero = cv2.findNonZero(mask_u8)
    if nonzero is None or len(nonzero) < 4:
        return None
    rect = cv2.minAreaRect(nonzero)
    width, height = rect[1]
    if min(width, height) < 2.0:
        return None
    box = cv2.boxPoints(rect).astype(np.float32)
    sums = box.sum(axis=1)
    diffs = (box[:, 0] - box[:, 1]).reshape(-1)
    return np.array(
        [
            box[np.argmin(sums)],
            box[np.argmax(diffs)],
            box[np.argmax(sums)],
            box[np.argmin(diffs)],
        ],
        dtype=np.float32,
    )


def fit_corners_with_fallback(
    *,
    mask: np.ndarray,
    mask_area_px: int,
    mask_bbox: list[int] | None,
    frame_shape: tuple[int, int],
    fit_primary: Callable[[np.ndarray], np.ndarray | None],
) -> tuple[np.ndarray | None, str, str | None]:
    bbox_w, bbox_h = bbox_dims(mask_bbox)
    bbox_area = max(bbox_w * bbox_h, 1)
    compactness = mask_area_px / bbox_area
    primary_area_px = max(
        64,
        int(frame_shape[0] * frame_shape[1] * MIN_PRIMARY_FIT_AREA_RATIO),
    )

    if mask_area_px < MIN_MASK_AREA_PX or min(bbox_w, bbox_h) < MIN_QUAD_EDGE_PX:
        return None, "not_run", "mask_area_too_small"

    if mask_area_px < primary_area_px:
        if min(bbox_w, bbox_h) < MIN_SMALL_BBOX_EDGE_PX or compactness < MIN_SMALL_MASK_COMPACTNESS:
            return None, "not_run", "small_mask_not_compact"
        fallback = fit_min_area_rect_quad(mask)
        if fallback is None:
            return None, "min_area_rect_fallback", "fit_failed"
        return fallback, "min_area_rect_fallback", None

    primary = fit_primary(mask)
    if primary is not None:
        return primary, "primary", None

    fallback = fit_min_area_rect_quad(mask)
    if fallback is None:
        return None, "primary", "fit_failed"
    return fallback, "min_area_rect_fallback", None


def validate_tracking_mask(
    mask: np.ndarray | None,
    frame_shape: tuple[int, int],
) -> tuple[bool, np.ndarray | None, list[str], dict[str, float]]:
    mask_area_px, _mask_bbox = mask_area_and_bbox(mask)
    if mask_area_px < MIN_MASK_AREA_PX:
        return False, None, ["mask_area_too_small"], {}
    if mask is None:
        return False, None, ["empty_mask"], {}
    quad = fit_min_area_rect_quad(mask)
    if quad is None:
        return False, None, ["fit_failed"], {}
    flags, stats = geometry_flags(mask, quad, frame_shape)
    return len(flags) == 0, quad, flags, stats
