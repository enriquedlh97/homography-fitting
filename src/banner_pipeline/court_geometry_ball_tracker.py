"""BallTrackerNet learned-keypoint court-geometry estimator (P3-A1 port).

This module provides a learned 14-keypoint detector + RANSAC-homography
court-geometry estimator that plugs into ``GeometryFittingEngine`` as a
drop-in replacement for the line-based ``CourtGeometryEstimator``.

Backend identifier
------------------
Selected via ``geometry.court_backend: ball_tracker_net_v1`` in the YAML
configs.  The default ``classical_lines_v1`` continues to use the existing
line-detection estimator (no behaviour change for existing configs).

What it does
------------
For each frame:
  1. Resize to 640x360, normalise, and run the BallTrackerNet CNN to
     produce a 14-channel heatmap.
  2. Extract one keypoint per heatmap channel via Hough-circle peak
     detection (matches the original sibling implementation).
  3. Compute a court-reference->image homography using RANSAC over all
     detected keypoints (>=5) or best-of-12 fallback.
  4. Compose with a fixed ``image -> court_quad-image`` transform so that
     the returned ``court_homography`` maps unit-square ``court_quad``
     coordinates to image coordinates — matching the contract of the
     classical estimator.  Downstream ``_fit_court_plane`` is unchanged.

Vendored upstream code
----------------------
Ported from sibling repo
``/Users/enriquediazdeleonhicks/repositories/capstone-data-candidates/tennis-virtual-ads``
calibrator (``tennis_court_detector.py`` + ``_tcd_adapted/``) and the
upstream ``TennisCourtDetector`` repository (``tracknet.py``).  The
network architecture, court reference layout, and homography solver are
inlined here to keep the port self-contained.

Weights
-------
The ``BallTrackerNet`` weights file (``tennis_court_detector.pt``,
~80 MB) is **not** redistributed in this repo.  At runtime, the
estimator searches the following locations in order and uses the first
one found:

    weights/tennis_court_detector.pt                 (in this repo)
    ../tennis-virtual-ads/weights/tennis_court_detector.pt
    <env: BANNER_PIPELINE_BTN_WEIGHTS>

If none is found, the estimator raises ``FileNotFoundError`` with a
download URL and target paths.  Download from:

    https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG

If the user has the sibling ``tennis-virtual-ads`` checkout the second
fallback path resolves automatically — no copy required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from banner_pipeline.court_geometry import CourtGeometryEstimate, GeometryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 360
NUM_KEYPOINTS = 14
MODEL_OUTPUT_CHANNELS = 15  # 14 keypoints + 1 unused centre channel
RANSAC_REPROJ_THRESHOLD_PX = 5.0
MIN_KEYPOINTS_FOR_RANSAC = 5

WEIGHTS_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
)


def _candidate_weights_paths() -> list[Path]:
    """Locations searched for the BallTrackerNet weights file.

    Walks upward from this file looking for ``weights/`` and the sibling
    ``tennis-virtual-ads`` and ``TennisCourtDetector`` repos.  This works
    regardless of whether the pipeline runs from the repo root, a
    sibling worktree directory, or a Modal-deployed image where the
    repo lives at an arbitrary path.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # Search a generous number of ancestor directories.
    for ancestor in here.parents:
        candidates.extend(
            [
                ancestor / "weights" / "tennis_court_detector.pt",
                ancestor / "tennis-virtual-ads" / "weights" / "tennis_court_detector.pt",
                ancestor / "TennisCourtDetector" / "model_tennis_court_det.pth",
            ]
        )
    env_path = os.environ.get("BANNER_PIPELINE_BTN_WEIGHTS")
    if env_path:
        candidates.insert(0, Path(env_path))
    return candidates


def _resolve_weights_path() -> Path:
    """Resolve the weights file path or raise FileNotFoundError."""
    candidates = _candidate_weights_paths()
    for path in candidates:
        if path.exists():
            return path
    formatted = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        "BallTrackerNet weights not found. Tried:\n"
        f"{formatted}\n"
        f"Download from {WEIGHTS_DOWNLOAD_URL} and place at one of the\n"
        "above locations, or set BANNER_PIPELINE_BTN_WEIGHTS to a custom path."
    )


# ---------------------------------------------------------------------------
# Vendored court reference (from TennisCourtDetector, MIT-licensed)
# ---------------------------------------------------------------------------


class _CourtReference:
    """14-keypoint court reference, copied from upstream verbatim."""

    def __init__(self) -> None:
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))

        self.key_points = [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
            *self.middle_line,
        ]

        self.court_conf = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [
                self.left_inner_line[0], self.right_inner_line[0],
                self.left_inner_line[1], self.right_inner_line[1],
            ],
            3: [
                self.left_inner_line[0], self.right_court_line[0],
                self.left_inner_line[1], self.right_court_line[1],
            ],
            4: [
                self.left_court_line[0], self.right_inner_line[0],
                self.left_court_line[1], self.right_inner_line[1],
            ],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [
                *self.top_inner_line,
                self.left_inner_line[1], self.right_inner_line[1],
            ],
            7: [
                self.left_inner_line[0], self.right_inner_line[0],
                *self.bottom_inner_line,
            ],
            8: [
                self.right_inner_line[0], self.right_court_line[0],
                self.right_inner_line[1], self.right_court_line[1],
            ],
            9: [
                self.left_court_line[0], self.left_inner_line[0],
                self.left_court_line[1], self.left_inner_line[1],
            ],
            10: [
                self.top_inner_line[0], self.middle_line[0],
                self.bottom_inner_line[0], self.middle_line[1],
            ],
            11: [
                self.middle_line[0], self.top_inner_line[1],
                self.middle_line[1], self.bottom_inner_line[1],
            ],
            12: [
                *self.bottom_inner_line,
                self.left_inner_line[1], self.right_inner_line[1],
            ],
        }
        # Court extents (unit-quad mapping uses the inner playing surface).
        self.court_xmin = float(self.left_court_line[0][0])
        self.court_xmax = float(self.right_court_line[0][0])
        self.court_ymin = float(self.baseline_top[0][1])
        self.court_ymax = float(self.baseline_bottom[0][1])


_COURT_REF_SINGLETON: _CourtReference | None = None


def _court_reference() -> _CourtReference:
    global _COURT_REF_SINGLETON
    if _COURT_REF_SINGLETON is None:
        _COURT_REF_SINGLETON = _CourtReference()
    return _COURT_REF_SINGLETON


def _refer_kps_array() -> np.ndarray:
    """All 14 reference keypoints as ``(14, 1, 2) float32`` for cv2."""
    ref = _court_reference()
    return np.array(ref.key_points, dtype=np.float32).reshape((-1, 1, 2))


def _court_conf_indices() -> dict[int, list[int]]:
    """Map config-id -> list of 4 indices into ``key_points``."""
    ref = _court_reference()
    out: dict[int, list[int]] = {}
    for cfg_id, conf_pts in ref.court_conf.items():
        out[cfg_id] = [ref.key_points.index(point) for point in conf_pts[:4]]
    return out


# ---------------------------------------------------------------------------
# Vendored homography solver
# ---------------------------------------------------------------------------


def _solve_court_homography(
    points: list[tuple[float | None, float | None]],
    *,
    ransac_threshold_px: float = RANSAC_REPROJ_THRESHOLD_PX,
) -> tuple[np.ndarray | None, str, int]:
    """Compute court-ref -> image homography from raw keypoints.

    Returns ``(matrix_or_None, method, inlier_count)``.  Tries RANSAC
    over all detected keypoints when >=5 are available, falls back to
    best-of-12 configuration selection otherwise.
    """
    ref = _court_reference()
    detected_indices: list[int] = []
    src_points: list[tuple[float, float]] = []
    dst_points: list[tuple[float, float]] = []
    for idx, (x_value, y_value) in enumerate(points):
        if x_value is not None and y_value is not None:
            src_points.append(ref.key_points[idx])
            dst_points.append((x_value, y_value))
            detected_indices.append(idx)

    if len(detected_indices) >= MIN_KEYPOINTS_FOR_RANSAC:
        src_array = np.array(src_points, dtype=np.float32)
        dst_array = np.array(dst_points, dtype=np.float32)
        matrix, mask = cv2.findHomography(
            src_array, dst_array, cv2.RANSAC, ransac_threshold_px,
        )
        if matrix is not None:
            inliers = int(mask.sum()) if mask is not None else 0
            return matrix.astype(np.float64), "ransac", inliers

    # Fallback: best-of-12 configuration search.
    refer_kps = _refer_kps_array()
    conf_indices = _court_conf_indices()

    best_matrix: np.ndarray | None = None
    best_error: float = float("inf")
    for cfg_id, indices in conf_indices.items():
        config_pts = ref.court_conf[cfg_id]
        correspondences = [points[indices[i]] for i in range(4)]
        if any(corr[0] is None for corr in correspondences):
            continue
        source = np.array(config_pts, dtype=np.float32)
        destination = np.array(correspondences, dtype=np.float32)
        matrix, _ = cv2.findHomography(source, destination, method=0)
        if matrix is None:
            continue

        projected = cv2.perspectiveTransform(refer_kps, matrix)
        distances: list[float] = []
        for keypoint_index in range(NUM_KEYPOINTS):
            if keypoint_index in indices:
                continue
            if points[keypoint_index][0] is None:
                continue
            detected = np.array(points[keypoint_index], dtype=np.float64)
            projected_point = projected[keypoint_index].flatten()
            distances.append(float(np.linalg.norm(detected - projected_point)))
        mean_error = float(np.mean(distances)) if distances else float("inf")
        if mean_error < best_error:
            best_error = mean_error
            best_matrix = matrix.astype(np.float64)

    if best_matrix is not None:
        return best_matrix, "best_of_12", len(detected_indices)
    return None, "none", len(detected_indices)


# ---------------------------------------------------------------------------
# Heatmap postprocessing (inlined from upstream postprocess.py without sympy)
# ---------------------------------------------------------------------------


def _peak_from_heatmap(
    heatmap: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    low_threshold: int = 170,
    min_radius: int = 10,
    max_radius: int = 25,
) -> tuple[float | None, float | None]:
    """Return ``(x, y)`` of the strongest peak in a single heatmap channel."""
    _, thresholded = cv2.threshold(heatmap, low_threshold, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        thresholded,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        x_pred = float(circles[0][0][0] * scale_x)
        y_pred = float(circles[0][0][1] * scale_y)
        return x_pred, y_pred
    return None, None


# ---------------------------------------------------------------------------
# BallTrackerNet architecture (inlined from upstream tracknet.py)
# ---------------------------------------------------------------------------


def _build_ball_tracker_net(out_channels: int = MODEL_OUTPUT_CHANNELS):
    """Construct the BallTrackerNet PyTorch module.

    Vendored from upstream ``TennisCourtDetector/tracknet.py``.  Imported
    lazily so the rest of the pipeline does not pay the torch import cost
    when this backend is unused.
    """
    import torch
    from torch import nn

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.block(x)

    class _BallTrackerNet(nn.Module):
        def __init__(self, out_ch: int = out_channels) -> None:
            super().__init__()
            self.out_channels = out_ch
            self.conv1 = _ConvBlock(3, 64)
            self.conv2 = _ConvBlock(64, 64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = _ConvBlock(64, 128)
            self.conv4 = _ConvBlock(128, 128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv5 = _ConvBlock(128, 256)
            self.conv6 = _ConvBlock(256, 256)
            self.conv7 = _ConvBlock(256, 256)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.conv8 = _ConvBlock(256, 512)
            self.conv9 = _ConvBlock(512, 512)
            self.conv10 = _ConvBlock(512, 512)
            self.ups1 = nn.Upsample(scale_factor=2)
            self.conv11 = _ConvBlock(512, 256)
            self.conv12 = _ConvBlock(256, 256)
            self.conv13 = _ConvBlock(256, 256)
            self.ups2 = nn.Upsample(scale_factor=2)
            self.conv14 = _ConvBlock(256, 128)
            self.conv15 = _ConvBlock(128, 128)
            self.ups3 = nn.Upsample(scale_factor=2)
            self.conv16 = _ConvBlock(128, 64)
            self.conv17 = _ConvBlock(64, 64)
            self.conv18 = _ConvBlock(64, self.out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.conv1(x); x = self.conv2(x); x = self.pool1(x)
            x = self.conv3(x); x = self.conv4(x); x = self.pool2(x)
            x = self.conv5(x); x = self.conv6(x); x = self.conv7(x); x = self.pool3(x)
            x = self.conv8(x); x = self.conv9(x); x = self.conv10(x)
            x = self.ups1(x); x = self.conv11(x); x = self.conv12(x); x = self.conv13(x)
            x = self.ups2(x); x = self.conv14(x); x = self.conv15(x)
            x = self.ups3(x); x = self.conv16(x); x = self.conv17(x); x = self.conv18(x)
            return x

    return _BallTrackerNet(out_ch=out_channels)


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------


def _court_quad_image_unit_homography() -> np.ndarray:
    """Map unit-square coordinates (0..1) to court-reference image coordinates.

    The classical line-based estimator builds its ``court_homography``
    from the outermost detected width / depth lines — typically the
    ``baseline_top`` / ``baseline_bottom`` and the ``left_court_line`` /
    ``right_court_line`` (i.e. the doubles outer rectangle).  The
    fractional ``court_quad`` coordinates in YAML configs (e.g.
    ``[0.3833, 0.9923]`` for the Red Bull floor logo) are calibrated
    against THIS rectangle.

    We therefore map our unit square to the same outer doubles
    rectangle in court-reference coordinates, so that the BallTrackerNet
    backend's ``court_homography`` is interchangeable with the classical
    backend's at the same fractional ``court_quad``.
    """
    ref = _court_reference()
    src = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    # Outer doubles rectangle: left_court_line.x .. right_court_line.x
    # by baseline_top.y .. baseline_bottom.y
    dst = np.array(
        [
            [float(ref.left_court_line[0][0]), float(ref.baseline_top[0][1])],
            [float(ref.right_court_line[0][0]), float(ref.baseline_top[0][1])],
            [float(ref.right_court_line[0][0]), float(ref.baseline_bottom[0][1])],
            [float(ref.left_court_line[0][0]), float(ref.baseline_bottom[0][1])],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    return homography.astype(np.float64)


class BallTrackerNetCourtEstimator:
    """Drop-in replacement for ``CourtGeometryEstimator``.

    Exposes the same ``estimate(frame_bgr) -> CourtGeometryEstimate``
    contract.  Only the ``court_homography`` field is populated meaningfully;
    line / vp_* fields stay ``None`` so back/side wall fitters that depend on
    them gracefully fall back to their hold/free-quad paths.

    The estimator is robust to torch / weights being missing: it raises
    ``ImportError`` / ``FileNotFoundError`` with actionable messages on
    construction so the manager-thread can decide to fall back to the
    classical backend.
    """

    def __init__(
        self,
        config: GeometryConfig,
        *,
        device: str | None = None,
        weights_path: str | os.PathLike[str] | None = None,
        ransac_threshold_px: float = RANSAC_REPROJ_THRESHOLD_PX,
        bridge_to_classical: bool = True,
    ) -> None:
        # Lazy torch import so unused backends do not pay the cost.
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "ball_tracker_net_v1 backend requires torch>=2.0 (already a "
                "project dep). Install with `uv sync`."
            ) from exc

        self._torch = torch
        self.config = config
        self._ransac_threshold_px = float(ransac_threshold_px)
        self._unit_to_court_ref = _court_quad_image_unit_homography()

        # Resolve weights and device.
        if weights_path is None:
            self._weights_path = _resolve_weights_path()
        else:
            self._weights_path = Path(weights_path)
            if not self._weights_path.exists():
                raise FileNotFoundError(
                    f"BallTrackerNet weights not found at {self._weights_path}. "
                    f"Download from {WEIGHTS_DOWNLOAD_URL}."
                )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Build & load the model lazily on first use to keep init cheap when
        # the estimator is constructed but never used (e.g., dry-run config
        # validation).
        self._model: Any = None

        # Smoothing state (matches classical estimator's EMA).
        self._smoothed_court_homography: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None

        # First-frame bridge to keep YAML configs that were calibrated
        # against the classical backend's frame-dependent unit-rectangle
        # interchangeable with the BTN backend.  When enabled, on the
        # first successful frame the estimator runs the classical
        # backend in parallel and computes
        # ``bridge = H_classical(0) @ H_btn(0)^-1``.  Subsequent frames
        # return ``bridge @ H_btn(t)`` so the absolute reference
        # rectangle matches the YAML's calibrated ``court_quad``.
        self._bridge_to_classical = bool(bridge_to_classical)
        self._bridge: np.ndarray | None = None
        self._bridge_classical: Any = None  # lazily-created CourtGeometryEstimator

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        torch = self._torch
        model = _build_ball_tracker_net(out_channels=MODEL_OUTPUT_CHANNELS).to(self._device)
        state_dict = torch.load(
            str(self._weights_path), map_location=self._device, weights_only=False
        )
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        logger.info(
            "Loaded BallTrackerNet from %s on %s",
            self._weights_path, self._device,
        )
        return model

    # ------------------------------------------------------------------
    # Main entry point — matches CourtGeometryEstimator.estimate()
    # ------------------------------------------------------------------

    def estimate(self, frame_bgr: np.ndarray) -> CourtGeometryEstimate:
        torch = self._torch
        model = self._load_model()
        original_height, original_width = frame_bgr.shape[:2]

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cut_reset = False
        if self._prev_gray is not None:
            diff = float(
                np.mean(np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32)))
            )
            if diff > 18.0:
                self._smoothed_court_homography = None
                cut_reset = True
        self._prev_gray = gray

        scale_x = original_width / MODEL_INPUT_WIDTH
        scale_y = original_height / MODEL_INPUT_HEIGHT

        resized = cv2.resize(frame_bgr, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        normalised = resized.astype(np.float32) / 255.0
        chw = np.rollaxis(normalised, 2, 0)
        input_tensor = torch.tensor(chw).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = model(input_tensor.float())[0]
            heatmap_stack = torch.sigmoid(output).detach().cpu().numpy()

        raw_keypoints: list[tuple[float | None, float | None]] = []
        for keypoint_index in range(NUM_KEYPOINTS):
            heatmap = (heatmap_stack[keypoint_index] * 255).astype(np.uint8)
            x_pred, y_pred = _peak_from_heatmap(
                heatmap, scale_x=scale_x, scale_y=scale_y,
            )
            raw_keypoints.append((x_pred, y_pred))

        detected_count = sum(1 for kp in raw_keypoints if kp[0] is not None)

        court_ref_to_image: np.ndarray | None = None
        method = "none"
        inlier_count = 0
        if detected_count >= 4:
            court_ref_to_image, method, inlier_count = _solve_court_homography(
                raw_keypoints,
                ransac_threshold_px=self._ransac_threshold_px,
            )

        # Compose unit-square -> court-ref -> image so the returned
        # ``court_homography`` matches the classical estimator's contract
        # (unit-square court_quad coordinates -> image pixels).
        unit_to_image: np.ndarray | None = None
        if court_ref_to_image is not None:
            unit_to_image = (court_ref_to_image @ self._unit_to_court_ref).astype(np.float64)

        # Compute a one-shot bridge at frame 0 that maps our default
        # unit-rectangle (outer doubles court) to whatever rectangle the
        # classical estimator picked at frame 0.  Lets the BTN backend
        # be a drop-in replacement for configs whose ``court_quad``
        # values were calibrated against the classical estimator.
        if (
            self._bridge_to_classical
            and self._bridge is None
            and unit_to_image is not None
        ):
            try:
                from banner_pipeline.court_geometry import CourtGeometryEstimator
                if self._bridge_classical is None:
                    self._bridge_classical = CourtGeometryEstimator(self.config)
                classical_estimate = self._bridge_classical.estimate(frame_bgr)
                if classical_estimate.court_homography is not None:
                    # H_classical(0) = bridge @ H_btn(0)
                    h_btn_inv = np.linalg.inv(unit_to_image)
                    h_cls = classical_estimate.court_homography.astype(np.float64)
                    self._bridge = (h_cls @ h_btn_inv).astype(np.float64)
                    logger.info(
                        "BallTrackerNet bridge to classical court_homography "
                        "established on frame 0 (det_kp=%d, inliers=%d).",
                        detected_count, inlier_count,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "BallTrackerNet bridge bootstrap failed: %s. "
                    "Falling back to raw BTN homography (configs may need "
                    "court_quad recalibration).",
                    exc,
                )
                # Disable further bridge attempts.
                self._bridge_to_classical = False

        # Apply the bridge if we have one.
        if (
            self._bridge_to_classical
            and self._bridge is not None
            and unit_to_image is not None
        ):
            unit_to_image = (self._bridge @ unit_to_image).astype(np.float64)
            scale = float(unit_to_image[2, 2])
            if abs(scale) > 1e-6:
                unit_to_image = unit_to_image / scale

        # Apply EMA smoothing on the homography across frames.
        alpha = float(self.config.vp_smoothing_alpha)
        smoothed = _blend_homographies(
            self._smoothed_court_homography, unit_to_image, alpha,
        )
        self._smoothed_court_homography = smoothed

        confidence = detected_count / NUM_KEYPOINTS
        return CourtGeometryEstimate(
            vp_width=None,
            vp_depth=None,
            dir_width=None,
            dir_depth=None,
            court_homography=(
                smoothed.astype(np.float32) if smoothed is not None else None
            ),
            geometry_confidence=float(np.clip(confidence, 0.0, 1.0)),
            width_family_confidence=0.0,
            depth_family_confidence=0.0,
            vp_width_confidence=0.0,
            vp_depth_confidence=0.0,
            width_candidate_count=detected_count,
            depth_candidate_count=inlier_count,
            top_width_line=None,
            bottom_width_line=None,
            left_depth_line=None,
            right_depth_line=None,
            cut_reset=cut_reset,
        )


def _blend_homographies(
    prev_h: np.ndarray | None,
    current_h: np.ndarray | None,
    alpha: float,
) -> np.ndarray | None:
    """EMA-blend two 3x3 homographies, normalising to ``H[2,2]==1``.

    Mirrors ``_blend_homographies`` in ``court_geometry.py`` so the two
    backends produce comparable smoothing behaviour at equivalent alpha.
    """
    if current_h is None:
        return prev_h
    if prev_h is None:
        return current_h.astype(np.float64)
    blended = alpha * prev_h.astype(np.float64) + (1.0 - alpha) * current_h.astype(np.float64)
    scale = float(blended[2, 2]) if abs(float(blended[2, 2])) > 1e-6 else 1.0
    blended /= scale
    return blended


# ---------------------------------------------------------------------------
# Factory helper used by GeometryFittingEngine
# ---------------------------------------------------------------------------


def build_estimator(config: GeometryConfig) -> Any:
    """Return the estimator selected by ``config.court_backend``.

    Recognised backends:
      - ``classical_lines_v1`` (default): the line-based
        :class:`banner_pipeline.court_geometry.CourtGeometryEstimator`.
      - ``ball_tracker_net_v1``: this module's
        :class:`BallTrackerNetCourtEstimator`.

    Unknown backend strings raise ``ValueError``.
    """
    backend = (config.court_backend or "classical_lines_v1").strip().lower()
    if backend == "classical_lines_v1":
        # Local import avoids a cycle at module import time.
        from banner_pipeline.court_geometry import CourtGeometryEstimator
        return CourtGeometryEstimator(config)
    if backend == "ball_tracker_net_v1":
        return BallTrackerNetCourtEstimator(config)
    raise ValueError(
        f"Unknown geometry.court_backend: {config.court_backend!r}. "
        "Supported: 'classical_lines_v1', 'ball_tracker_net_v1'."
    )
