"""Stateful rectified-plane compositor for temporally stable banner replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from banner_pipeline.composite.base import Compositor
from banner_pipeline.composite.inpaint import InpaintCompositor

_RESET_CORNER_RMS_PX = 40.0
_MAX_FRAME_GAP_BEFORE_RESET = 8
_MIN_COURT_RATIO = 0.6
_MAX_COURT_RATIO = 1.4


@dataclass
class _ObjectState:
    model: str
    plate_size: tuple[int, int] | None = None
    clean_plate: np.ndarray | None = None
    shading_field: np.ndarray | None = None
    last_quad: np.ndarray | None = None
    last_frame_idx: int | None = None
    last_rectified_frame: np.ndarray | None = None
    last_rectified_composite: np.ndarray | None = None
    plate_init_frame: int | None = None
    plate_reused_frames: int = 0
    plate_reset_count: int = 0
    delegated_inpaint_frames: int = 0
    court_shading_updates: int = 0


def _kernel_size(size: int) -> int:
    size = max(int(size), 1)
    return size if size % 2 == 1 else size + 1


def _quad_corner_rms(prev_quad: np.ndarray, quad: np.ndarray) -> float:
    prev = np.asarray(prev_quad, dtype=np.float32).reshape(4, 2)
    curr = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return float(np.sqrt(np.mean(np.sum((curr - prev) ** 2, axis=1))))


def _quad_metrics(corners: np.ndarray) -> tuple[float, float, float, float]:
    quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    w_top = float(np.linalg.norm(quad[1] - quad[0]))
    w_bottom = float(np.linalg.norm(quad[2] - quad[3]))
    h_right = float(np.linalg.norm(quad[2] - quad[1]))
    h_left = float(np.linalg.norm(quad[3] - quad[0]))
    return w_top, w_bottom, h_right, h_left


def _rectified_size(corners: np.ndarray, *, rectified_min_size_px: int) -> tuple[int, int]:
    w_top, w_bottom, h_right, h_left = _quad_metrics(corners)
    avg_w = max((w_top + w_bottom) / 2.0, 1.0)
    avg_h = max((h_left + h_right) / 2.0, 1.0)
    scale_up = max(1.0, float(rectified_min_size_px) / max(avg_w, avg_h))
    return max(int(round(avg_w * scale_up)), 1), max(int(round(avg_h * scale_up)), 1)


def _rectified_homographies(
    corners: np.ndarray,
    plate_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dst_w, dst_h = plate_size
    dst_rect = np.array(
        [
            [0.0, 0.0],
            [float(dst_w - 1), 0.0],
            [float(dst_w - 1), float(dst_h - 1)],
            [0.0, float(dst_h - 1)],
        ],
        dtype=np.float32,
    )
    corners_f32 = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    h_to_rect, _ = cv2.findHomography(corners_f32, dst_rect)
    h_to_image, _ = cv2.findHomography(dst_rect, corners_f32)
    if h_to_rect is None or h_to_image is None:
        raise RuntimeError("Could not compute rectified homography for temporal compositor.")
    return h_to_rect, h_to_image, dst_rect


def _warp_to_rectified(
    frame: np.ndarray,
    mask: np.ndarray | None,
    corners: np.ndarray,
    plate_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    h_to_rect, _, _dst_rect = _rectified_homographies(corners, plate_size)
    dst_w, dst_h = plate_size
    rectified_frame = cv2.warpPerspective(frame, h_to_rect, (dst_w, dst_h))
    rectified_mask = None
    if mask is not None:
        mask_u8 = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
        rectified_mask = cv2.warpPerspective(
            mask_u8,
            h_to_rect,
            (dst_w, dst_h),
            flags=cv2.INTER_NEAREST,
        )
    return rectified_frame, rectified_mask


def _inpaint_rectified_plate(
    rectified_frame: np.ndarray,
    rectified_mask: np.ndarray | None,
    *,
    erase_mask_dilate_px: int,
) -> np.ndarray:
    if rectified_mask is None or not np.asarray(rectified_mask).any():
        return rectified_frame.copy()
    mask_u8 = rectified_mask.astype(np.uint8)
    dilate_px = _kernel_size(erase_mask_dilate_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    mask_u8 = cv2.dilate(mask_u8, kernel).astype(np.uint8)
    return cv2.inpaint(rectified_frame, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def _build_overlay_plate(
    overlay: np.ndarray,
    plate_size: tuple[int, int],
    *,
    padding: float,
    shading_field: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dst_w, dst_h = plate_size
    overlay_bgra = overlay
    if overlay.ndim == 3 and overlay.shape[2] == 3:
        overlay_bgra = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    logo_h, logo_w = overlay_bgra.shape[:2]
    pad_w = int(round(dst_w * padding))
    pad_h = int(round(dst_h * padding))
    avail_w = max(dst_w - 2 * pad_w, 1)
    avail_h = max(dst_h - 2 * pad_h, 1)
    scale = min(avail_w / max(logo_w, 1), avail_h / max(logo_h, 1))
    new_w = max(int(round(logo_w * scale)), 1)
    new_h = max(int(round(logo_h * scale)), 1)
    logo_resized = cv2.resize(overlay_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rgb_canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((dst_h, dst_w), dtype=np.uint8)
    x0 = (dst_w - new_w) // 2
    y0 = (dst_h - new_h) // 2
    rgb_canvas[y0 : y0 + new_h, x0 : x0 + new_w] = logo_resized[:, :, :3]
    alpha_canvas[y0 : y0 + new_h, x0 : x0 + new_w] = logo_resized[:, :, 3]

    if shading_field is not None:
        shaded = np.clip(
            rgb_canvas.astype(np.float32) * shading_field.astype(np.float32),
            0.0,
            255.0,
        )
        rgb_canvas = shaded.astype(np.uint8)

    return rgb_canvas, alpha_canvas


def _compose_rectified_plate(
    clean_plate: np.ndarray,
    overlay: np.ndarray,
    *,
    padding: float,
    shading_field: np.ndarray | None = None,
) -> np.ndarray:
    plate_size = (int(clean_plate.shape[1]), int(clean_plate.shape[0]))
    overlay_rgb, overlay_alpha = _build_overlay_plate(
        overlay,
        plate_size,
        padding=padding,
        shading_field=shading_field,
    )
    composite = clean_plate.astype(np.float32)
    alpha = cv2.GaussianBlur(overlay_alpha, (5, 5), 1.0).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    composite = overlay_rgb.astype(np.float32) * alpha + composite * (1.0 - alpha)
    return composite.astype(np.uint8)


def _replace_quad_region(
    frame: np.ndarray,
    corners: np.ndarray,
    rectified_composite: np.ndarray,
) -> np.ndarray:
    frame_out = frame.copy()
    _h_to_rect, h_to_image, _dst_rect = _rectified_homographies(
        corners,
        (int(rectified_composite.shape[1]), int(rectified_composite.shape[0])),
    )
    warped = cv2.warpPerspective(rectified_composite, h_to_image, (frame.shape[1], frame.shape[0]))
    quad_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, np.asarray(corners, dtype=np.int32), 255)
    frame_out[quad_mask > 0] = warped[quad_mask > 0]
    return frame_out


def _triptych(
    current_rectified: np.ndarray | None,
    clean_plate: np.ndarray | None,
    composite_rectified: np.ndarray | None,
) -> np.ndarray | None:
    if current_rectified is None or clean_plate is None or composite_rectified is None:
        return None
    panels = []
    for label, image in [
        ("Current", current_rectified),
        ("Cached Plate", clean_plate),
        ("Composite", composite_rectified),
    ]:
        panel = image.copy()
        cv2.putText(
            panel,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.concatenate(panels, axis=1)


class TemporalRectifiedCompositor(Compositor):
    """Temporally stable rectified-plane compositor."""

    def __init__(
        self,
        *,
        padding: float = 0.05,
        rectified_min_size_px: int = 500,
        erase_mask_dilate_px: int = 7,
        wall_freeze_after_init: bool = True,
        court_shading_enabled: bool = True,
        court_shading_blur_px: int = 41,
        court_shading_alpha: float = 0.9,
    ) -> None:
        self.padding = float(padding)
        self.rectified_min_size_px = int(rectified_min_size_px)
        self.erase_mask_dilate_px = int(erase_mask_dilate_px)
        self.wall_freeze_after_init = bool(wall_freeze_after_init)
        self.court_shading_enabled = bool(court_shading_enabled)
        self.court_shading_blur_px = _kernel_size(court_shading_blur_px)
        self.court_shading_alpha = float(court_shading_alpha)
        self._delegate = InpaintCompositor()
        self._states: dict[int, _ObjectState] = {}
        self._runtime_used = False
        self._object_models: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "temporal_rectified"

    def composite(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        obj_id = int(kwargs.get("obj_id", -1))
        surface_type = str(kwargs.get("surface_type", "banner")).strip().lower() or "banner"
        frame_idx = int(kwargs.get("frame_idx", 0))
        geometry_fit_method = str(kwargs.get("geometry_fit_method", "") or "")
        geometry_held = bool(kwargs.get("geometry_held", False))

        if surface_type == "back_wall_banner":
            model = "temporal_wall_plate"
        elif surface_type == "court_marking":
            model = "temporal_court_plate"
        else:
            model = "delegated_inpaint"

        self._runtime_used = True
        self._object_models[str(obj_id)] = model
        state = self._states.setdefault(obj_id, _ObjectState(model=model))
        state.model = model

        if model == "delegated_inpaint":
            state.delegated_inpaint_frames += 1
            return self._delegate.composite(frame, corners, overlay, mask=mask, **kwargs)

        if obj_id < 0:
            raise RuntimeError(
                "TemporalRectifiedCompositor requires obj_id for stateful compositing."
            )

        plate_size = state.plate_size or _rectified_size(
            corners,
            rectified_min_size_px=self.rectified_min_size_px,
        )

        if self._should_reset_state(state, corners, frame_idx):
            state.clean_plate = None
            state.shading_field = None
            state.plate_size = None
            state.plate_reset_count += 1
            plate_size = _rectified_size(
                corners,
                rectified_min_size_px=self.rectified_min_size_px,
            )

        if state.plate_size is None:
            state.plate_size = plate_size
        else:
            plate_size = state.plate_size

        rectified_frame, rectified_mask = _warp_to_rectified(frame, mask, corners, plate_size)
        state.last_rectified_frame = rectified_frame.copy()

        if state.clean_plate is None:
            state.clean_plate = _inpaint_rectified_plate(
                rectified_frame,
                rectified_mask,
                erase_mask_dilate_px=self.erase_mask_dilate_px,
            )
            state.plate_init_frame = frame_idx
        elif model == "temporal_wall_plate":
            should_freeze = (
                self.wall_freeze_after_init
                or geometry_held
                or geometry_fit_method == "hold_last_good"
            )
            if not should_freeze:
                state.clean_plate = _inpaint_rectified_plate(
                    rectified_frame,
                    rectified_mask,
                    erase_mask_dilate_px=self.erase_mask_dilate_px,
                )
            else:
                state.plate_reused_frames += 1
        else:
            state.plate_reused_frames += 1

        if model == "temporal_court_plate" and state.clean_plate is not None:
            self._update_court_shading(
                state,
                rectified_frame,
                rectified_mask,
                geometry_fit_method=geometry_fit_method,
                geometry_held=geometry_held,
            )

        shading_field = state.shading_field if model == "temporal_court_plate" else None
        assert state.clean_plate is not None
        rectified_composite = _compose_rectified_plate(
            state.clean_plate,
            overlay,
            padding=self.padding,
            shading_field=shading_field,
        )
        state.last_rectified_composite = rectified_composite.copy()
        state.last_quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        state.last_frame_idx = frame_idx
        return _replace_quad_region(frame, corners, rectified_composite)

    def _should_reset_state(self, state: _ObjectState, corners: np.ndarray, frame_idx: int) -> bool:
        if state.clean_plate is None or state.last_quad is None:
            return False
        if (
            state.last_frame_idx is not None
            and frame_idx - state.last_frame_idx > _MAX_FRAME_GAP_BEFORE_RESET
        ):
            return True
        return _quad_corner_rms(state.last_quad, corners) > _RESET_CORNER_RMS_PX

    def _update_court_shading(
        self,
        state: _ObjectState,
        rectified_frame: np.ndarray,
        rectified_mask: np.ndarray | None,
        *,
        geometry_fit_method: str,
        geometry_held: bool,
    ) -> None:
        if not self.court_shading_enabled or state.clean_plate is None:
            return
        if geometry_held or geometry_fit_method == "hold_last_good":
            return
        clean_plate = np.maximum(state.clean_plate.astype(np.float32), 1.0)
        observed = np.maximum(rectified_frame.astype(np.float32), 1.0)
        ratio = np.ones_like(observed, dtype=np.float32)
        if rectified_mask is not None and np.asarray(rectified_mask).any():
            valid = rectified_mask <= 0
            if np.any(valid):
                ratio[valid] = observed[valid] / clean_plate[valid]
        else:
            ratio = (observed / clean_plate).astype(np.float32)
        ratio = cv2.GaussianBlur(
            ratio,
            (self.court_shading_blur_px, self.court_shading_blur_px),
            0,
        ).astype(np.float32)
        ratio = np.clip(ratio, _MIN_COURT_RATIO, _MAX_COURT_RATIO).astype(np.float32)
        if state.shading_field is None:
            state.shading_field = ratio
        else:
            state.shading_field = (
                self.court_shading_alpha * state.shading_field
                + (1.0 - self.court_shading_alpha) * ratio
            )
        state.court_shading_updates += 1

    def finalize_metrics(self) -> dict[str, Any]:
        stats: dict[str, dict[str, int | str | None]] = {}
        for obj_id, state in self._states.items():
            stats[str(obj_id)] = {
                "plate_init_frame": state.plate_init_frame,
                "plate_reused_frames": int(state.plate_reused_frames),
                "plate_reset_count": int(state.plate_reset_count),
                "delegated_inpaint_frames": int(state.delegated_inpaint_frames),
                "court_shading_updates": int(state.court_shading_updates),
            }
        return {
            "compositor_runtime_enabled": self._runtime_used,
            "compositor_object_model": self._object_models,
            "compositor_object_stats": stats,
        }

    def render_debug_artifacts(self) -> dict[str, np.ndarray]:
        artifacts: dict[str, np.ndarray] = {}
        for obj_id, state in self._states.items():
            triptych = _triptych(
                state.last_rectified_frame,
                state.clean_plate,
                state.last_rectified_composite,
            )
            if triptych is not None:
                artifacts[f"compositor_rectified_obj_{obj_id}"] = triptych
        return artifacts
