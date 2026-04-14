"""Temporal mask stabilization for video banner tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from banner_pipeline import quality as quality_mod


@dataclass(frozen=True)
class StabilizationConfig:
    """Configuration for hybrid temporal mask stabilization."""

    enabled: bool = False
    mode: str = "hybrid"
    static_motion_threshold_px: float = 0.75
    hold_corner_rms_px: float = 1.25
    mask_iou_gate: float = 0.55
    max_hold_frames: int = 6
    predicted_mask_weight: float = 0.65
    morph_kernel_px: int = 3

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> StabilizationConfig:
        if config is None:
            return cls()
        return cls(
            enabled=bool(config.get("enabled", False)),
            mode=str(config.get("mode", "hybrid")),
            static_motion_threshold_px=float(config.get("static_motion_threshold_px", 0.75)),
            hold_corner_rms_px=float(config.get("hold_corner_rms_px", 1.25)),
            mask_iou_gate=float(config.get("mask_iou_gate", 0.55)),
            max_hold_frames=int(config.get("max_hold_frames", 6)),
            predicted_mask_weight=float(config.get("predicted_mask_weight", 0.65)),
            morph_kernel_px=int(config.get("morph_kernel_px", 3)),
        )


@dataclass(frozen=True)
class MotionEstimate:
    """Per-step inter-frame motion estimate."""

    homography: np.ndarray
    median_corner_disp_px: float
    near_static: bool
    estimation_ok: bool


@dataclass
class _ObjectState:
    stabilized_mask: np.ndarray | None = None
    empty_hold_streak: int = 0


def stabilize_video_segments(
    *,
    frame_dir: str,
    frame_names: list[str],
    video_segments: dict[int, dict[int, np.ndarray]],
    tracked_obj_ids: list[int],
    config: dict[str, Any] | None,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[str, Any]]:
    """Stabilize tracked masks across frames using sparse optical flow."""
    stab_config = StabilizationConfig.from_dict(config)
    if not stab_config.enabled:
        return video_segments, {}
    if stab_config.mode != "hybrid":
        raise ValueError(f"Unsupported stabilization mode: {stab_config.mode}")

    t0 = time.perf_counter()
    motion_steps, frame_shape = _estimate_consecutive_frame_motion(
        frame_dir=frame_dir,
        frame_names=frame_names,
        config=stab_config,
    )
    stabilized_segments, object_stats = _stabilize_masks_with_motion(
        video_segments=video_segments,
        tracked_obj_ids=tracked_obj_ids,
        motion_steps=motion_steps,
        frame_shape=frame_shape,
        config=stab_config,
    )
    total_s = round(time.perf_counter() - t0, 4)

    num_motion_steps = max(len(motion_steps) - 1, 1)
    static_steps = sum(1 for motion_step in motion_steps[1:] if motion_step.near_static)
    aggregate_held = sum(int(stats["frames_held"]) for stats in object_stats.values())
    aggregate_blended = sum(int(stats["frames_blended"]) for stats in object_stats.values())
    aggregate_raw = sum(int(stats["frames_raw_accepted"]) for stats in object_stats.values())

    metrics: dict[str, Any] = {
        "stabilization_total_s": total_s,
        "stabilization_static_frame_ratio": round(static_steps / num_motion_steps, 4),
        "stabilization_frames_held": aggregate_held,
        "stabilization_frames_blended": aggregate_blended,
        "stabilization_frames_raw_accepted": aggregate_raw,
        "stabilization_object_stats": object_stats,
    }
    return stabilized_segments, metrics


def _estimate_consecutive_frame_motion(
    *,
    frame_dir: str,
    frame_names: list[str],
    config: StabilizationConfig,
) -> tuple[list[MotionEstimate], tuple[int, int]]:
    first_frame = cv2.imread(str(Path(frame_dir) / frame_names[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame for stabilization: {frame_names[0]}")

    frame_shape = first_frame.shape[:2]
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    motion_steps = [
        MotionEstimate(
            homography=np.eye(3, dtype=np.float32),
            median_corner_disp_px=0.0,
            near_static=True,
            estimation_ok=True,
        )
    ]

    for frame_name in frame_names[1:]:
        frame_bgr = cv2.imread(str(Path(frame_dir) / frame_name))
        if frame_bgr is None:
            raise RuntimeError(f"Could not read frame for stabilization: {frame_name}")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        homography, disp_px, estimation_ok = _estimate_pair_homography(prev_gray, gray, frame_shape)
        motion_steps.append(
            MotionEstimate(
                homography=homography,
                median_corner_disp_px=disp_px,
                near_static=estimation_ok and disp_px < config.static_motion_threshold_px,
                estimation_ok=estimation_ok,
            )
        )
        prev_gray = gray

    return motion_steps, frame_shape


def _estimate_pair_homography(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, float, bool]:
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if prev_pts is None or len(prev_pts) < 4:
        return np.eye(3, dtype=np.float32), 0.0, False

    curr_pts, status, _errs = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if curr_pts is None or status is None:
        return np.eye(3, dtype=np.float32), 0.0, False

    good = status.reshape(-1) == 1
    if int(good.sum()) < 4:
        return np.eye(3, dtype=np.float32), 0.0, False

    src = prev_pts[good].reshape(-1, 1, 2)
    dst = curr_pts[good].reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None or not np.isfinite(homography).all():
        return np.eye(3, dtype=np.float32), 0.0, False
    if inlier_mask is None or int(inlier_mask.sum()) < 4:
        return np.eye(3, dtype=np.float32), 0.0, False

    disp_px = _median_corner_displacement(homography, frame_shape)
    return homography.astype(np.float32), disp_px, True


def _median_corner_displacement(homography: np.ndarray, frame_shape: tuple[int, int]) -> float:
    height, width = frame_shape
    corners = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    disp = np.linalg.norm(warped - corners.reshape(-1, 2), axis=1)
    return float(np.median(disp))


def _stabilize_masks_with_motion(
    *,
    video_segments: dict[int, dict[int, np.ndarray]],
    tracked_obj_ids: list[int],
    motion_steps: list[MotionEstimate],
    frame_shape: tuple[int, int],
    config: StabilizationConfig,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[str, dict[str, int]]]:
    num_frames = len(motion_steps)
    all_obj_ids = sorted(
        set(int(obj_id) for obj_id in tracked_obj_ids)
        | {int(obj_id) for masks_by_obj in video_segments.values() for obj_id in masks_by_obj}
    )
    stabilized_segments: dict[int, dict[int, np.ndarray]] = {
        frame_idx: {} for frame_idx in range(num_frames)
    }
    states = {obj_id: _ObjectState() for obj_id in all_obj_ids}
    object_stats = {
        str(obj_id): {
            "frames_held": 0,
            "frames_empty_reused": 0,
            "frames_blended": 0,
            "frames_raw_accepted": 0,
            "frames_dropped": 0,
            "max_empty_hold_streak": 0,
        }
        for obj_id in all_obj_ids
    }

    for frame_idx in range(num_frames):
        masks_by_obj = video_segments.get(frame_idx, {})
        motion_step = motion_steps[frame_idx]
        for obj_id in all_obj_ids:
            state = states[obj_id]
            stats = object_stats[str(obj_id)]
            raw_mask = _coerce_mask_u8(masks_by_obj.get(obj_id))
            pred_mask_prob = None
            pred_mask_u8 = None
            if frame_idx > 0 and state.stabilized_mask is not None:
                pred_mask_prob = _warp_mask_prob(
                    state.stabilized_mask,
                    motion_step.homography,
                    frame_shape,
                )
                pred_mask_u8 = _threshold_mask(pred_mask_prob)

            if _mask_is_empty(raw_mask):
                if pred_mask_u8 is not None and state.empty_hold_streak < config.max_hold_frames:
                    stabilized_segments[frame_idx][obj_id] = pred_mask_u8
                    state.stabilized_mask = pred_mask_u8
                    state.empty_hold_streak += 1
                    stats["frames_held"] += 1
                    stats["frames_empty_reused"] += 1
                    stats["max_empty_hold_streak"] = max(
                        int(stats["max_empty_hold_streak"]),
                        state.empty_hold_streak,
                    )
                else:
                    state.stabilized_mask = None
                    state.empty_hold_streak = 0
                    stats["frames_dropped"] += 1
                continue

            if pred_mask_prob is None or pred_mask_u8 is None:
                stabilized_segments[frame_idx][obj_id] = raw_mask
                state.stabilized_mask = raw_mask
                state.empty_hold_streak = 0
                stats["frames_raw_accepted"] += 1
                continue

            iou = _mask_iou(raw_mask, pred_mask_u8)
            if iou >= config.mask_iou_gate:
                if motion_step.near_static and _should_hard_hold(
                    raw_mask=raw_mask,
                    pred_mask=pred_mask_u8,
                    hold_corner_rms_px=config.hold_corner_rms_px,
                ):
                    stabilized_segments[frame_idx][obj_id] = pred_mask_u8
                    state.stabilized_mask = pred_mask_u8
                    state.empty_hold_streak = 0
                    stats["frames_held"] += 1
                    continue

                fused_mask = _fuse_masks(
                    pred_mask_prob=pred_mask_prob,
                    raw_mask=raw_mask,
                    predicted_mask_weight=config.predicted_mask_weight,
                    morph_kernel_px=config.morph_kernel_px,
                )
                stabilized_segments[frame_idx][obj_id] = fused_mask
                state.stabilized_mask = fused_mask
                state.empty_hold_streak = 0
                stats["frames_blended"] += 1
                continue

            stabilized_segments[frame_idx][obj_id] = raw_mask
            state.stabilized_mask = raw_mask
            state.empty_hold_streak = 0
            stats["frames_raw_accepted"] += 1

    return stabilized_segments, object_stats


def _coerce_mask_u8(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    mask_2d = np.asarray(mask).squeeze()
    if mask_2d.ndim != 2 or mask_2d.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if mask_2d.dtype == np.uint8 and set(np.unique(mask_2d)).issubset({0, 255}):
        return mask_2d.copy()
    return ((mask_2d > 0).astype(np.uint8) * 255).copy()


def _mask_is_empty(mask: np.ndarray) -> bool:
    return mask.size == 0 or not bool(mask.any())


def _warp_mask_prob(
    mask: np.ndarray,
    homography: np.ndarray,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    height, width = frame_shape
    if mask.dtype == np.uint8:
        mask_float = mask.astype(np.float32) / 255.0
    else:
        mask_float = mask.astype(np.float32)
    return cv2.warpPerspective(
        mask_float,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _threshold_mask(mask_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mask_prob >= threshold).astype(np.uint8) * 255


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(a, b).sum())
    return intersection / union


def _should_hard_hold(
    *,
    raw_mask: np.ndarray,
    pred_mask: np.ndarray,
    hold_corner_rms_px: float,
) -> bool:
    raw_quad = quality_mod.fit_min_area_rect_quad(raw_mask)
    pred_quad = quality_mod.fit_min_area_rect_quad(pred_mask)
    if raw_quad is None or pred_quad is None:
        return False

    corner_rms_px = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (raw_quad.astype(np.float32) - pred_quad.astype(np.float32)) ** 2,
                    axis=1,
                )
            )
        )
    )
    raw_area = quality_mod.polygon_area(raw_quad)
    pred_area = quality_mod.polygon_area(pred_quad)
    area_delta_ratio = abs(raw_area - pred_area) / max(max(raw_area, pred_area), 1.0)
    return corner_rms_px < hold_corner_rms_px and area_delta_ratio < 0.12


def _fuse_masks(
    *,
    pred_mask_prob: np.ndarray,
    raw_mask: np.ndarray,
    predicted_mask_weight: float,
    morph_kernel_px: int,
) -> np.ndarray:
    raw_float = raw_mask.astype(np.float32) / 255.0
    fused_prob = np.clip(
        predicted_mask_weight * pred_mask_prob + (1.0 - predicted_mask_weight) * raw_float,
        0.0,
        1.0,
    )
    fused_mask = _threshold_mask(fused_prob)
    return _cleanup_mask(fused_mask, morph_kernel_px)


def _cleanup_mask(mask: np.ndarray, morph_kernel_px: int) -> np.ndarray:
    if morph_kernel_px <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_px, morph_kernel_px))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened
