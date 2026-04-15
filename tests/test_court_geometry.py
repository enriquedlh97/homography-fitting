from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline import court_geometry as court_geometry_mod
from banner_pipeline.fitting.vp_constrained import VPConstrainedBannerFitter


def _horizontal_banner_mask(*, jagged: bool = False) -> np.ndarray:
    mask = np.zeros((120, 220), dtype=np.uint8)
    mask[40:72, 24:196] = 255
    if jagged:
        mask[40:46, 60:68] = 0
        mask[64:72, 108:116] = 0
        mask[48:56, 150:158] = 0
    return mask


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = b - a
    ca = c - a
    return float(abs(ba[0] * ca[1] - ba[1] * ca[0]))


def _line_point_at_x(vp: np.ndarray, anchor: np.ndarray, x: float) -> tuple[int, int]:
    denom = float(anchor[0] - vp[0])
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6
    t = (x - float(vp[0])) / denom
    y = float(vp[1]) + t * float(anchor[1] - vp[1])
    return int(round(x)), int(round(y))


def _line_point_at_y(vp: np.ndarray, anchor: np.ndarray, y: float) -> tuple[int, int]:
    denom = float(anchor[1] - vp[1])
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6
    t = (y - float(vp[1])) / denom
    x = float(vp[0]) + t * float(anchor[0] - vp[0])
    return int(round(x)), int(round(y))


def _synthetic_court_frame(*, width_shift_px: int = 0) -> np.ndarray:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    vp_width = np.array([470.0 + width_shift_px, 52.0], dtype=np.float32)
    vp_depth = np.array([156.0, -175.0], dtype=np.float32)

    for y in (52.0, 88.0, 126.0):
        anchor = np.array([24.0, y], dtype=np.float32)
        p0 = _line_point_at_x(vp_width, anchor, 16.0)
        p1 = _line_point_at_x(vp_width, anchor, 304.0)
        cv2.line(frame, p0, p1, (255, 255, 255), 3, cv2.LINE_AA)

    for x in (74.0, 156.0, 240.0):
        anchor = np.array([x, 168.0], dtype=np.float32)
        p0 = _line_point_at_y(vp_depth, anchor, 20.0)
        p1 = _line_point_at_y(vp_depth, anchor, 172.0)
        cv2.line(frame, p0, p1, (255, 255, 255), 3, cv2.LINE_AA)

    return frame


def test_vp_constrained_banner_fitter_preserves_perspective_family_on_jagged_masks() -> None:
    fitter = VPConstrainedBannerFitter()
    vp = np.array([112.0, -140.0], dtype=np.float32)
    parallel_dir = np.array([1.0, 0.0], dtype=np.float32)

    base_fit = fitter.fit_with_params(_horizontal_banner_mask(), vp=vp, parallel_dir=parallel_dir)
    jagged_fit = fitter.fit_with_params(
        _horizontal_banner_mask(jagged=True),
        vp=vp,
        parallel_dir=parallel_dir,
    )

    assert base_fit.corners is not None
    assert jagged_fit.corners is not None

    for corners in (base_fit.corners, jagged_fit.corners):
        tl, tr, br, bl = corners.astype(np.float64)
        top = tr - tl
        bottom = br - bl
        assert abs(top[0] * bottom[1] - top[1] * bottom[0]) < 1e-3
        assert _triangle_area(vp.astype(np.float64), tl, bl) < 1e-2
        assert _triangle_area(vp.astype(np.float64), tr, br) < 1e-2

    assert abs(float(base_fit.corners[0, 1] - jagged_fit.corners[0, 1])) < 8.0
    assert abs(float(base_fit.corners[3, 1] - jagged_fit.corners[3, 1])) < 8.0


def test_vp_constrained_banner_fitter_rejects_degenerate_inputs() -> None:
    fitter = VPConstrainedBannerFitter()
    result = fitter.fit_with_params(
        _horizontal_banner_mask(),
        vp=None,
        parallel_dir=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert result.corners is None
    assert result.support_offsets is None
    assert result.ray_angles is None


def test_court_geometry_estimator_recovers_both_vanishing_families_and_smooths_jitter() -> None:
    config = court_geometry_mod.GeometryConfig(enabled=True, vp_smoothing_alpha=0.85)
    estimator = court_geometry_mod.CourtGeometryEstimator(config)

    frame_a = _synthetic_court_frame(width_shift_px=0)
    frame_b = _synthetic_court_frame(width_shift_px=18)

    estimate_a = estimator.estimate(frame_a)
    estimate_b = estimator.estimate(frame_b)

    assert estimate_a.vp_width is not None
    assert estimate_a.vp_depth is not None
    assert estimate_b.vp_width is not None
    assert estimate_b.vp_depth is not None
    assert estimate_a.dir_width is not None
    assert estimate_a.dir_depth is not None

    segments = court_geometry_mod._detect_line_segments(frame_b)
    width_family, _depth_family = court_geometry_mod._split_line_families(segments)
    raw_vp_width, _conf = court_geometry_mod._estimate_family_vp(width_family, frame_b.shape[:2])
    assert raw_vp_width is not None

    smoothed_shift = np.linalg.norm(estimate_b.vp_width - estimate_a.vp_width)
    raw_shift = np.linalg.norm(raw_vp_width - estimate_a.vp_width)
    assert smoothed_shift < raw_shift


def test_court_geometry_estimator_resets_state_on_large_cut() -> None:
    config = court_geometry_mod.GeometryConfig(enabled=True)
    estimator = court_geometry_mod.CourtGeometryEstimator(config)

    estimator.estimate(_synthetic_court_frame())
    cut_estimate = estimator.estimate(np.zeros((180, 320, 3), dtype=np.uint8))

    assert cut_estimate.cut_reset is True
    assert cut_estimate.geometry_confidence == 0.0
