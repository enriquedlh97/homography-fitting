"""Court-geometry estimation and VP-constrained banner fitting."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from banner_pipeline import quality as quality_mod
from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.fitting.vp_constrained import VPConstrainedBannerFitter
from banner_pipeline.geometry import intersect_parametric, line_from_points, sort_corners_tlbr
from banner_pipeline.segment.base import ObjectPrompt

SUPPORTED_GEOMETRY_MODELS = {
    "mask_free_quad",
    "vp_constrained_horizontal_banner",
    "vp_constrained_vertical_banner",
    "court_plane",
}
SURFACE_TO_GEOMETRY_MODEL = {
    "back_wall_banner": "vp_constrained_horizontal_banner",
    "side_wall_banner": "vp_constrained_vertical_banner",
    "court_marking": "court_plane",
}
SUPPORTED_GEOMETRY_SURFACE_TYPES = set(SURFACE_TO_GEOMETRY_MODEL)


@dataclass(frozen=True)
class GeometryConfig:
    """Configuration for court-geometry constrained banner fitting."""

    enabled: bool = False
    court_backend: str = "classical_lines_v1"
    vp_smoothing_alpha: float = 0.7
    line_smoothing_alpha: float = 0.65
    hold_frames: int = 8
    fallback_after_frames: int = 3
    vp_confidence_min: float = 0.35
    tangent_margin_px: float = 2.0

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> GeometryConfig:
        if config is None:
            return cls()
        return cls(
            enabled=bool(config.get("enabled", False)),
            court_backend=str(config.get("court_backend", "classical_lines_v1")),
            vp_smoothing_alpha=float(config.get("vp_smoothing_alpha", 0.7)),
            line_smoothing_alpha=float(config.get("line_smoothing_alpha", 0.65)),
            hold_frames=int(config.get("hold_frames", 8)),
            fallback_after_frames=int(config.get("fallback_after_frames", 3)),
            vp_confidence_min=float(config.get("vp_confidence_min", 0.35)),
            tangent_margin_px=float(config.get("tangent_margin_px", 2.0)),
        )


@dataclass(frozen=True)
class CourtGeometryEstimate:
    """Smoothed per-frame estimate of court vanishing geometry."""

    vp_width: np.ndarray | None
    vp_depth: np.ndarray | None
    dir_width: np.ndarray | None
    dir_depth: np.ndarray | None
    court_homography: np.ndarray | None
    geometry_confidence: float
    vp_width_confidence: float
    vp_depth_confidence: float
    cut_reset: bool = False


@dataclass
class FitDetail:
    """Per-object fitting detail for the latest frame."""

    geometry_model: str
    fit_method: str
    held: bool = False
    used_fallback: bool = False


@dataclass
class _ObjectState:
    support_offsets: tuple[float, float] | None = None
    ray_angles: tuple[float, float] | None = None
    last_quad: np.ndarray | None = None
    last_corners_local: np.ndarray | None = None
    failure_streak: int = 0
    hold_streak: int = 0


def is_enabled(config: dict[str, Any] | None) -> bool:
    return GeometryConfig.from_dict(config).enabled


def normalize_geometry_model(geometry_model: object | None) -> str | None:
    if geometry_model is None:
        return None
    text = str(geometry_model).strip().lower()
    if not text:
        return None
    if text not in SUPPORTED_GEOMETRY_MODELS:
        raise ValueError(f"Unsupported geometry_model: {geometry_model}")
    return text


def resolve_geometry_model(prompt: ObjectPrompt) -> str:
    if prompt.geometry_model is not None:
        geometry_model = normalize_geometry_model(prompt.geometry_model)
        if geometry_model is not None:
            return geometry_model
    return SURFACE_TO_GEOMETRY_MODEL.get(str(prompt.surface_type).strip().lower(), "mask_free_quad")


def supports_surface_type(surface_type: str, geometry_enabled: bool) -> bool:
    normalized = str(surface_type).strip().lower() or "banner"
    if normalized == "banner":
        return True
    return geometry_enabled and normalized in SUPPORTED_GEOMETRY_SURFACE_TYPES


def supported_surface_types(geometry_enabled: bool) -> set[str]:
    base = {"banner"}
    if geometry_enabled:
        base |= SUPPORTED_GEOMETRY_SURFACE_TYPES
    return base


def _normalize_vec(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector.astype(np.float64) / norm


def _blend_unit_vectors(
    prev_vec: np.ndarray | None,
    current_vec: np.ndarray | None,
    alpha: float,
) -> np.ndarray | None:
    prev_unit = _normalize_vec(prev_vec) if prev_vec is not None else None
    curr_unit = _normalize_vec(current_vec) if current_vec is not None else None
    if prev_unit is None:
        return curr_unit
    if curr_unit is None:
        return prev_unit
    if np.dot(prev_unit, curr_unit) < 0:
        curr_unit = -curr_unit
    blended = alpha * prev_unit + (1.0 - alpha) * curr_unit
    return _normalize_vec(blended)


def _segment_direction(segment: np.ndarray) -> np.ndarray | None:
    start = np.asarray(segment[0], dtype=np.float64)
    end = np.asarray(segment[1], dtype=np.float64)
    return _normalize_vec(end - start)


def _segment_midpoint(segment: np.ndarray) -> np.ndarray:
    return (
        np.asarray(segment[0], dtype=np.float64) + np.asarray(segment[1], dtype=np.float64)
    ) / 2.0


def _detect_line_segments(frame_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(bright, 50, 150, apertureSize=3)
    min_len = max(int(min(frame_bgr.shape[:2]) * 0.08), 24)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=45,
        minLineLength=min_len,
        maxLineGap=18,
    )
    if lines is None:
        return []
    out: list[np.ndarray] = []
    for line in lines.reshape(-1, 4):
        p0 = np.array([float(line[0]), float(line[1])], dtype=np.float64)
        p1 = np.array([float(line[2]), float(line[3])], dtype=np.float64)
        if np.linalg.norm(p1 - p0) < min_len:
            continue
        out.append(np.stack([p0, p1], axis=0))
    return out


def _split_line_families(segments: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    width_family: list[np.ndarray] = []
    depth_family: list[np.ndarray] = []
    for segment in segments:
        direction = _segment_direction(segment)
        if direction is None:
            continue
        if abs(direction[0]) >= abs(direction[1]):
            width_family.append(segment)
        else:
            depth_family.append(segment)
    return width_family, depth_family


def _estimate_family_direction(
    family: list[np.ndarray],
    *,
    reference: np.ndarray,
) -> np.ndarray | None:
    if not family:
        return None
    accumulator = np.zeros(2, dtype=np.float64)
    for segment in family:
        direction = _segment_direction(segment)
        if direction is None:
            continue
        if np.dot(direction, reference) < 0:
            direction = -direction
        accumulator += direction
    return _normalize_vec(accumulator)


def _estimate_family_vp(
    family: list[np.ndarray],
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray | None, float]:
    if len(family) < 2:
        return None, 0.0

    lines = [line_from_points(segment[0], segment[1]) for segment in family]
    intersections: list[np.ndarray] = []
    for idx, line_a in enumerate(lines):
        for line_b in lines[idx + 1 :]:
            point = _intersect_implicit(line_a, line_b)
            if point is None:
                continue
            if not np.isfinite(point).all():
                continue
            intersections.append(point.astype(np.float64))

    if not intersections:
        return None, 0.0

    points = np.vstack(intersections)
    vp = np.median(points, axis=0)
    spread = float(np.median(np.linalg.norm(points - vp, axis=1)))
    frame_scale = float(max(frame_shape))
    confidence = max(0.0, 1.0 - spread / max(frame_scale * 4.0, 1.0))
    confidence *= min(1.0, len(intersections) / 8.0)
    return vp.astype(np.float32), float(np.clip(confidence, 0.0, 1.0))


def _intersect_implicit(
    line_a: tuple[float, float, float],
    line_b: tuple[float, float, float],
) -> np.ndarray | None:
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)


def _family_support_offsets(
    family: list[np.ndarray], family_dir: np.ndarray
) -> tuple[float, float] | None:
    if not family:
        return None
    normal = np.array([-family_dir[1], family_dir[0]], dtype=np.float64)
    offsets = [float(_segment_midpoint(segment) @ normal) for segment in family]
    return min(offsets), max(offsets)


def _estimate_court_homography(
    *,
    width_family: list[np.ndarray],
    depth_family: list[np.ndarray],
    dir_width: np.ndarray | None,
    dir_depth: np.ndarray | None,
) -> np.ndarray | None:
    if dir_width is None or dir_depth is None:
        return None
    width_offsets = _family_support_offsets(width_family, dir_width)
    depth_offsets = _family_support_offsets(depth_family, dir_depth)
    if width_offsets is None or depth_offsets is None:
        return None

    width_normal = np.array([-dir_width[1], dir_width[0]], dtype=np.float64)
    depth_normal = np.array([-dir_depth[1], dir_depth[0]], dtype=np.float64)
    top_point = width_normal * width_offsets[0]
    bottom_point = width_normal * width_offsets[1]
    left_point = depth_normal * depth_offsets[0]
    right_point = depth_normal * depth_offsets[1]

    tl = intersect_parametric(top_point, dir_width, left_point, dir_depth)
    tr = intersect_parametric(top_point, dir_width, right_point, dir_depth)
    br = intersect_parametric(bottom_point, dir_width, right_point, dir_depth)
    bl = intersect_parametric(bottom_point, dir_width, left_point, dir_depth)
    if any(point is None for point in (tl, tr, br, bl)):
        return None

    corners = sort_corners_tlbr(np.array([tl, tr, br, bl], dtype=np.float32))
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    homography, _ = cv2.findHomography(src, corners)
    if homography is None or not np.isfinite(homography).all():
        return None
    return homography.astype(np.float32)


class CourtGeometryEstimator:
    """Estimate stable vanishing geometry from court lines."""

    def __init__(self, config: GeometryConfig) -> None:
        self.config = config
        self._prev_gray: np.ndarray | None = None
        self._vp_width: np.ndarray | None = None
        self._vp_depth: np.ndarray | None = None
        self._dir_width: np.ndarray | None = None
        self._dir_depth: np.ndarray | None = None
        self._court_homography: np.ndarray | None = None

    def estimate(self, frame_bgr: np.ndarray) -> CourtGeometryEstimate:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cut_reset = False
        if self._prev_gray is not None:
            diff = float(
                np.mean(np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32)))
            )
            if diff > 18.0:
                self._vp_width = None
                self._vp_depth = None
                self._dir_width = None
                self._dir_depth = None
                self._court_homography = None
                cut_reset = True
        self._prev_gray = gray

        segments = _detect_line_segments(frame_bgr)
        width_family, depth_family = _split_line_families(segments)
        raw_dir_width = _estimate_family_direction(width_family, reference=np.array([1.0, 0.0]))
        raw_dir_depth = _estimate_family_direction(depth_family, reference=np.array([0.0, 1.0]))
        raw_vp_width, vp_width_conf = _estimate_family_vp(width_family, frame_bgr.shape[:2])
        raw_vp_depth, vp_depth_conf = _estimate_family_vp(depth_family, frame_bgr.shape[:2])
        raw_h = _estimate_court_homography(
            width_family=width_family,
            depth_family=depth_family,
            dir_width=raw_dir_width,
            dir_depth=raw_dir_depth,
        )

        alpha = self.config.vp_smoothing_alpha
        self._dir_width = _blend_unit_vectors(self._dir_width, raw_dir_width, alpha)
        self._dir_depth = _blend_unit_vectors(self._dir_depth, raw_dir_depth, alpha)
        self._vp_width = _blend_points(self._vp_width, raw_vp_width, alpha)
        self._vp_depth = _blend_points(self._vp_depth, raw_vp_depth, alpha)
        self._court_homography = _blend_homographies(self._court_homography, raw_h, alpha)

        geometry_confidence = max(vp_width_conf, vp_depth_conf)
        return CourtGeometryEstimate(
            vp_width=self._vp_width,
            vp_depth=self._vp_depth,
            dir_width=self._dir_width.astype(np.float32) if self._dir_width is not None else None,
            dir_depth=self._dir_depth.astype(np.float32) if self._dir_depth is not None else None,
            court_homography=self._court_homography,
            geometry_confidence=float(np.clip(geometry_confidence, 0.0, 1.0)),
            vp_width_confidence=vp_width_conf,
            vp_depth_confidence=vp_depth_conf,
            cut_reset=cut_reset,
        )


def _blend_points(
    prev_point: np.ndarray | None,
    current_point: np.ndarray | None,
    alpha: float,
) -> np.ndarray | None:
    if current_point is None:
        return prev_point
    if prev_point is None:
        return current_point.astype(np.float32)
    blended = alpha * prev_point.astype(np.float64) + (1.0 - alpha) * current_point.astype(
        np.float64,
    )
    return blended.astype(np.float32)


def _blend_homographies(
    prev_h: np.ndarray | None,
    current_h: np.ndarray | None,
    alpha: float,
) -> np.ndarray | None:
    if current_h is None:
        return prev_h
    if prev_h is None:
        return current_h.astype(np.float32)
    blended = alpha * prev_h.astype(np.float64) + (1.0 - alpha) * current_h.astype(np.float64)
    scale = float(blended[2, 2]) if abs(float(blended[2, 2])) > 1e-6 else 1.0
    blended /= scale
    return blended.astype(np.float32)


class GeometryFittingEngine:
    """Stateful geometry fitting engine used by the preview and video pipelines."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None,
        prompts: list[ObjectPrompt],
        fallback_fitter: QuadFitter,
        fitter_params: dict[str, Any] | None,
    ) -> None:
        self.config = GeometryConfig.from_dict(config)
        self.prompts = prompts
        self.fallback_fitter = fallback_fitter
        self.fitter_params = fitter_params or {}
        self.estimator = CourtGeometryEstimator(self.config)
        self.vp_fitter = VPConstrainedBannerFitter()
        self.states = {int(prompt.obj_id): _ObjectState() for prompt in prompts}
        self.details: dict[int, FitDetail] = {}
        self.frames_processed = 0
        self.frames_held = 0
        self.frames_fallback = 0
        self.vp_width_valid_frames = 0
        self.vp_depth_valid_frames = 0
        self.object_geometry_model = {
            str(int(prompt.obj_id)): resolve_geometry_model(prompt) for prompt in prompts
        }

    def fit_frame(
        self,
        *,
        frame_idx: int,
        frame_bgr: np.ndarray,
        masks_by_obj: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
        self.details = {}
        estimate = self.estimator.estimate(frame_bgr)
        self.frames_processed += 1
        if estimate.vp_width_confidence >= self.config.vp_confidence_min:
            self.vp_width_valid_frames += 1
        if estimate.vp_depth_confidence >= self.config.vp_confidence_min:
            self.vp_depth_valid_frames += 1

        corners_map: dict[int, np.ndarray] = {}
        rejection_reasons: dict[int, list[str]] = {}
        for prompt in self.prompts:
            obj_id = int(prompt.obj_id)
            surface_type = str(prompt.surface_type).strip().lower() or "banner"
            geometry_model = resolve_geometry_model(prompt)
            self.details[obj_id] = FitDetail(
                geometry_model=geometry_model,
                fit_method="not_run",
            )

            if not supports_surface_type(surface_type, geometry_enabled=self.config.enabled):
                rejection_reasons[obj_id] = [f"unsupported_surface_type:{surface_type}"]
                continue

            mask = masks_by_obj.get(obj_id)
            mask_area_px, mask_bbox = quality_mod.mask_area_and_bbox(mask)
            if mask is None or mask_area_px == 0:
                rejection_reasons[obj_id] = ["empty_mask"]
                self.states[obj_id].failure_streak = 0
                self.states[obj_id].hold_streak = 0
                continue

            corners = self._fit_for_prompt(
                prompt=prompt,
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
                estimate=estimate,
            )
            if corners is None:
                reasons = rejection_reasons.setdefault(obj_id, [])
                if not reasons:
                    reasons.append("fit_failed")
                continue

            flags, _stats = quality_mod.geometry_flags(mask, corners, frame_shape)
            if flags:
                rejection_reasons[obj_id] = flags
                continue
            corners_map[obj_id] = corners

        return corners_map, rejection_reasons

    def finalize_metrics(self) -> dict[str, Any]:
        total_frames = max(self.frames_processed, 1)
        return {
            "geometry_total_s": None,
            "geometry_frames_held": self.frames_held,
            "geometry_fallback_frames": self.frames_fallback,
            "vp_width_valid_ratio": round(self.vp_width_valid_frames / total_frames, 4),
            "vp_depth_valid_ratio": round(self.vp_depth_valid_frames / total_frames, 4),
            "object_geometry_model": self.object_geometry_model,
        }

    def _fit_for_prompt(
        self,
        *,
        prompt: ObjectPrompt,
        mask: np.ndarray,
        mask_area_px: int,
        mask_bbox: list[int] | None,
        frame_shape: tuple[int, int],
        estimate: CourtGeometryEstimate,
    ) -> np.ndarray | None:
        obj_id = int(prompt.obj_id)
        geometry_model = resolve_geometry_model(prompt)
        detail = self.details[obj_id]
        state = self.states[obj_id]

        if geometry_model == "mask_free_quad" or not self.config.enabled:
            detail.fit_method = "mask_free_quad"
            corners = self._fit_free_quad(
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
            )
            if corners is not None:
                state.last_quad = corners
            return corners

        if geometry_model == "court_plane":
            corners = self._fit_court_plane(
                state=state,
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
                estimate=estimate,
            )
            detail.fit_method = "court_plane" if corners is not None else "court_plane_failed"
            return corners

        if geometry_model == "vp_constrained_horizontal_banner":
            parallel_dir = estimate.dir_width
            vp = estimate.vp_depth
            vp_conf = estimate.vp_depth_confidence
        else:
            parallel_dir = estimate.dir_depth
            vp = estimate.vp_width
            vp_conf = estimate.vp_width_confidence

        if (
            vp is None
            or parallel_dir is None
            or vp_conf < self.config.vp_confidence_min
            or estimate.geometry_confidence < self.config.vp_confidence_min
        ):
            return self._hold_or_fallback(
                state=state,
                detail=detail,
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
            )

        raw_fit = self.vp_fitter.fit_with_params(
            mask,
            vp=vp,
            parallel_dir=parallel_dir,
            tangent_margin_px=self.config.tangent_margin_px,
        )
        if raw_fit.corners is None or raw_fit.support_offsets is None or raw_fit.ray_angles is None:
            return self._hold_or_fallback(
                state=state,
                detail=detail,
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
            )

        support_offsets = raw_fit.support_offsets
        ray_angles = raw_fit.ray_angles
        if state.support_offsets is not None and state.ray_angles is not None:
            support_offsets, ray_angles = self.vp_fitter.blend_params(
                prev_support_offsets=state.support_offsets,
                prev_ray_angles=state.ray_angles,
                support_offsets=support_offsets,
                ray_angles=ray_angles,
                alpha=self.config.line_smoothing_alpha,
            )
        corners = self.vp_fitter.reconstruct_from_params(
            vp=vp,
            parallel_dir=parallel_dir,
            support_offsets=support_offsets,
            ray_angles=ray_angles,
        )
        if corners is None:
            return self._hold_or_fallback(
                state=state,
                detail=detail,
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
            )

        state.support_offsets = support_offsets
        state.ray_angles = ray_angles
        state.last_quad = corners
        state.failure_streak = 0
        state.hold_streak = 0
        detail.fit_method = geometry_model
        return corners

    def _fit_court_plane(
        self,
        *,
        state: _ObjectState,
        mask: np.ndarray,
        mask_area_px: int,
        mask_bbox: list[int] | None,
        frame_shape: tuple[int, int],
        estimate: CourtGeometryEstimate,
    ) -> np.ndarray | None:
        if estimate.court_homography is None:
            return self._fit_free_quad(
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
            )

        if state.last_corners_local is None:
            fallback_corners = self._fit_free_quad(
                mask=mask,
                mask_area_px=mask_area_px,
                mask_bbox=mask_bbox,
                frame_shape=frame_shape,
            )
            if fallback_corners is None:
                return None
            inv_h = np.linalg.inv(estimate.court_homography.astype(np.float64))
            local = cv2.perspectiveTransform(
                fallback_corners.reshape(1, -1, 2).astype(np.float32),
                inv_h.astype(np.float32),
            ).reshape(-1, 2)
            state.last_corners_local = local.astype(np.float32)
            state.last_quad = fallback_corners
            return fallback_corners

        projected = cv2.perspectiveTransform(
            state.last_corners_local.reshape(1, -1, 2).astype(np.float32),
            estimate.court_homography.astype(np.float32),
        ).reshape(-1, 2)
        corners = sort_corners_tlbr(projected.astype(np.float32))
        state.last_quad = corners
        state.failure_streak = 0
        state.hold_streak = 0
        return corners

    def _hold_or_fallback(
        self,
        *,
        state: _ObjectState,
        detail: FitDetail,
        mask: np.ndarray,
        mask_area_px: int,
        mask_bbox: list[int] | None,
        frame_shape: tuple[int, int],
    ) -> np.ndarray | None:
        state.failure_streak += 1
        if (
            state.last_quad is not None
            and state.failure_streak <= self.config.fallback_after_frames
            and state.hold_streak < self.config.hold_frames
        ):
            state.hold_streak += 1
            detail.held = True
            detail.fit_method = "hold_last_good"
            self.frames_held += 1
            return state.last_quad

        fallback = self._fit_free_quad(
            mask=mask,
            mask_area_px=mask_area_px,
            mask_bbox=mask_bbox,
            frame_shape=frame_shape,
        )
        if fallback is not None:
            detail.used_fallback = True
            detail.fit_method = "mask_free_quad_fallback"
            self.frames_fallback += 1
            state.last_quad = fallback
            state.failure_streak = 0
            state.hold_streak = 0
            return fallback

        if state.last_quad is not None and state.hold_streak < self.config.hold_frames:
            state.hold_streak += 1
            detail.held = True
            detail.fit_method = "hold_last_good"
            self.frames_held += 1
            return state.last_quad
        return None

    def _fit_free_quad(
        self,
        *,
        mask: np.ndarray,
        mask_area_px: int,
        mask_bbox: list[int] | None,
        frame_shape: tuple[int, int],
    ) -> np.ndarray | None:
        corners, _fit_method, _flag = quality_mod.fit_corners_with_fallback(
            mask=mask,
            mask_area_px=mask_area_px,
            mask_bbox=mask_bbox,
            frame_shape=frame_shape,
            fit_primary=lambda current_mask: self.fallback_fitter.fit(
                current_mask,
                **self.fitter_params,
            ),
        )
        return corners


def fit_video_segments_with_geometry(
    *,
    config: dict[str, Any] | None,
    prompts: list[ObjectPrompt],
    frame_names: list[str],
    frame_reader: Callable[[int], np.ndarray],
    masks_getter: Callable[[int], dict[int, np.ndarray]],
    fallback_fitter: QuadFitter,
    fitter_params: dict[str, Any] | None,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[str, Any]]:
    """Offline helper mainly used by tests."""
    engine = GeometryFittingEngine(
        config=config,
        prompts=prompts,
        fallback_fitter=fallback_fitter,
        fitter_params=fitter_params,
    )
    t0 = time.perf_counter()
    corners_by_frame: dict[int, dict[int, np.ndarray]] = {}
    rejections_by_frame: dict[int, dict[int, list[str]]] = {}
    for frame_idx, _name in enumerate(frame_names):
        frame = frame_reader(frame_idx)
        masks = masks_getter(frame_idx)
        corners, rejections = engine.fit_frame(
            frame_idx=frame_idx,
            frame_bgr=frame,
            masks_by_obj=masks,
            frame_shape=frame.shape[:2],
        )
        corners_by_frame[frame_idx] = corners
        rejections_by_frame[frame_idx] = rejections

    metrics = engine.finalize_metrics()
    metrics["geometry_total_s"] = round(time.perf_counter() - t0, 4)
    metrics["geometry_rejections_by_frame"] = rejections_by_frame
    return corners_by_frame, metrics
