"""Vanishing-point-constrained banner fitting."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.geometry import intersect_parametric, sort_corners_tlbr


@dataclass(frozen=True)
class VPConstrainedFitResult:
    """Result bundle for a VP-constrained banner fit."""

    corners: np.ndarray | None
    support_offsets: tuple[float, float] | None
    ray_angles: tuple[float, float] | None


def _normalize_vec(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector.astype(np.float64) / norm


def _largest_component_contour(mask: np.ndarray) -> np.ndarray | None:
    mask_u8: np.ndarray = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    if mask_u8.ndim != 2 or not mask_u8.any():
        return None

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    if n_labels > 2:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask_u8 = ((labels == largest_label) * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float64)


def _circular_blend(prev_angle: float, current_angle: float, alpha: float) -> float:
    prev_vec = np.array([np.cos(prev_angle), np.sin(prev_angle)], dtype=np.float64)
    curr_vec = np.array([np.cos(current_angle), np.sin(current_angle)], dtype=np.float64)
    blended = alpha * prev_vec + (1.0 - alpha) * curr_vec
    if np.linalg.norm(blended) < 1e-6:
        return current_angle
    return float(np.arctan2(blended[1], blended[0]))


class VPConstrainedBannerFitter(QuadFitter):
    """Fit a banner quad from mask support lines plus a shared vanishing point."""

    @property
    def name(self) -> str:
        return "vp_constrained"

    def fit(self, mask: np.ndarray, **kwargs) -> np.ndarray | None:
        result = self.fit_with_params(
            mask,
            vp=kwargs.get("vp"),
            parallel_dir=kwargs.get("parallel_dir"),
            tangent_margin_px=float(kwargs.get("tangent_margin_px", 0.0)),
        )
        return result.corners

    def fit_with_params(
        self,
        mask: np.ndarray,
        *,
        vp: np.ndarray | None,
        parallel_dir: np.ndarray | None,
        tangent_margin_px: float = 0.0,
    ) -> VPConstrainedFitResult:
        contour = _largest_component_contour(mask)
        if contour is None or vp is None or parallel_dir is None:
            return VPConstrainedFitResult(None, None, None)

        parallel_unit = _normalize_vec(np.asarray(parallel_dir, dtype=np.float64))
        if parallel_unit is None:
            return VPConstrainedFitResult(None, None, None)
        vp_xy = np.asarray(vp, dtype=np.float64).reshape(2)

        normal = np.array([-parallel_unit[1], parallel_unit[0]], dtype=np.float64)
        projections = contour @ normal
        top_offset = float(projections.min() - tangent_margin_px)
        bottom_offset = float(projections.max() + tangent_margin_px)
        if bottom_offset - top_offset < 2.0:
            return VPConstrainedFitResult(None, None, None)

        vp_rays = contour - vp_xy
        ray_norms = np.linalg.norm(vp_rays, axis=1)
        valid = ray_norms > 1e-3
        if int(valid.sum()) < 2:
            return VPConstrainedFitResult(None, None, None)
        vp_rays = vp_rays[valid]
        ray_points = contour[valid]

        ray_angles = np.arctan2(vp_rays[:, 1], vp_rays[:, 0])
        left_idx = int(np.argmin(ray_angles))
        right_idx = int(np.argmax(ray_angles))
        left_dir = _normalize_vec(ray_points[left_idx] - vp_xy)
        right_dir = _normalize_vec(ray_points[right_idx] - vp_xy)
        if left_dir is None or right_dir is None:
            return VPConstrainedFitResult(None, None, None)

        corners = self.reconstruct_from_params(
            vp=vp_xy,
            parallel_dir=parallel_unit,
            support_offsets=(top_offset, bottom_offset),
            ray_angles=(float(ray_angles[left_idx]), float(ray_angles[right_idx])),
        )
        if corners is None:
            return VPConstrainedFitResult(None, None, None)
        return VPConstrainedFitResult(
            corners=corners,
            support_offsets=(top_offset, bottom_offset),
            ray_angles=(float(ray_angles[left_idx]), float(ray_angles[right_idx])),
        )

    @staticmethod
    def blend_params(
        *,
        prev_support_offsets: tuple[float, float],
        prev_ray_angles: tuple[float, float],
        support_offsets: tuple[float, float],
        ray_angles: tuple[float, float],
        alpha: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        blended_offsets = (
            float(alpha * prev_support_offsets[0] + (1.0 - alpha) * support_offsets[0]),
            float(alpha * prev_support_offsets[1] + (1.0 - alpha) * support_offsets[1]),
        )
        blended_angles = (
            _circular_blend(prev_ray_angles[0], ray_angles[0], alpha),
            _circular_blend(prev_ray_angles[1], ray_angles[1], alpha),
        )
        return blended_offsets, blended_angles

    @staticmethod
    def reconstruct_from_params(
        *,
        vp: np.ndarray,
        parallel_dir: np.ndarray,
        support_offsets: tuple[float, float],
        ray_angles: tuple[float, float],
    ) -> np.ndarray | None:
        vp_xy = np.asarray(vp, dtype=np.float64).reshape(2)
        parallel_unit = _normalize_vec(np.asarray(parallel_dir, dtype=np.float64))
        if parallel_unit is None:
            return None

        top_offset, bottom_offset = support_offsets
        left_angle, right_angle = ray_angles
        normal = np.array([-parallel_unit[1], parallel_unit[0]], dtype=np.float64)
        top_point = normal * top_offset
        bottom_point = normal * bottom_offset
        left_dir = np.array([np.cos(left_angle), np.sin(left_angle)], dtype=np.float64)
        right_dir = np.array([np.cos(right_angle), np.sin(right_angle)], dtype=np.float64)

        tl = intersect_parametric(top_point, parallel_unit, vp_xy, left_dir)
        tr = intersect_parametric(top_point, parallel_unit, vp_xy, right_dir)
        br = intersect_parametric(bottom_point, parallel_unit, vp_xy, right_dir)
        bl = intersect_parametric(bottom_point, parallel_unit, vp_xy, left_dir)
        if any(point is None for point in (tl, tr, br, bl)):
            return None
        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        return sort_corners_tlbr(corners)
