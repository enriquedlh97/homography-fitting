"""Fronto-parallel wall banner fitting."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.geometry import intersect_parametric, sort_corners_tlbr


@dataclass(frozen=True)
class FrontoParallelFitResult:
    """Result bundle for a fronto-parallel wall banner fit."""

    corners: np.ndarray | None
    support_offsets: tuple[float, float] | None
    lateral_offsets: tuple[float, float] | None


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


class FrontoParallelBannerFitter(QuadFitter):
    """Fit a wall banner as an oriented rectangle without a vanishing point."""

    @property
    def name(self) -> str:
        return "fronto_parallel"

    def fit(self, mask: np.ndarray, **kwargs) -> np.ndarray | None:
        result = self.fit_with_params(
            mask,
            parallel_dir=kwargs.get("parallel_dir"),
            margin_px=float(kwargs.get("margin_px", 0.0)),
        )
        return result.corners

    def fit_with_params(
        self,
        mask: np.ndarray,
        *,
        parallel_dir: np.ndarray | None,
        margin_px: float = 0.0,
    ) -> FrontoParallelFitResult:
        contour = _largest_component_contour(mask)
        if contour is None:
            return FrontoParallelFitResult(None, None, None)

        parallel_unit = _normalize_vec(np.asarray(parallel_dir, dtype=np.float64))
        if parallel_unit is None:
            parallel_unit = np.array([1.0, 0.0], dtype=np.float64)
        normal = np.array([-parallel_unit[1], parallel_unit[0]], dtype=np.float64)

        parallel_proj = contour @ parallel_unit
        normal_proj = contour @ normal
        left_offset = float(parallel_proj.min() - margin_px)
        right_offset = float(parallel_proj.max() + margin_px)
        top_offset = float(normal_proj.min() - margin_px)
        bottom_offset = float(normal_proj.max() + margin_px)
        if right_offset - left_offset < 2.0 or bottom_offset - top_offset < 2.0:
            return FrontoParallelFitResult(None, None, None)

        corners = self.reconstruct_from_params(
            parallel_dir=parallel_unit,
            support_offsets=(top_offset, bottom_offset),
            lateral_offsets=(left_offset, right_offset),
        )
        if corners is None:
            return FrontoParallelFitResult(None, None, None)
        return FrontoParallelFitResult(
            corners=corners,
            support_offsets=(top_offset, bottom_offset),
            lateral_offsets=(left_offset, right_offset),
        )

    @staticmethod
    def blend_params(
        *,
        prev_support_offsets: tuple[float, float],
        prev_lateral_offsets: tuple[float, float],
        support_offsets: tuple[float, float],
        lateral_offsets: tuple[float, float],
        alpha: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        blended_support = (
            float(alpha * prev_support_offsets[0] + (1.0 - alpha) * support_offsets[0]),
            float(alpha * prev_support_offsets[1] + (1.0 - alpha) * support_offsets[1]),
        )
        blended_lateral = (
            float(alpha * prev_lateral_offsets[0] + (1.0 - alpha) * lateral_offsets[0]),
            float(alpha * prev_lateral_offsets[1] + (1.0 - alpha) * lateral_offsets[1]),
        )
        return blended_support, blended_lateral

    @staticmethod
    def reconstruct_from_params(
        *,
        parallel_dir: np.ndarray,
        support_offsets: tuple[float, float],
        lateral_offsets: tuple[float, float],
    ) -> np.ndarray | None:
        parallel_unit = _normalize_vec(np.asarray(parallel_dir, dtype=np.float64))
        if parallel_unit is None:
            return None

        normal = np.array([-parallel_unit[1], parallel_unit[0]], dtype=np.float64)
        top_offset, bottom_offset = support_offsets
        left_offset, right_offset = lateral_offsets
        top_point = normal * top_offset
        bottom_point = normal * bottom_offset
        left_point = parallel_unit * left_offset
        right_point = parallel_unit * right_offset

        tl = intersect_parametric(top_point, parallel_unit, left_point, normal)
        tr = intersect_parametric(top_point, parallel_unit, right_point, normal)
        br = intersect_parametric(bottom_point, parallel_unit, right_point, normal)
        bl = intersect_parametric(bottom_point, parallel_unit, left_point, normal)
        if any(point is None for point in (tl, tr, br, bl)):
            return None
        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        return sort_corners_tlbr(corners)
