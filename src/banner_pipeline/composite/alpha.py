"""Oriented-homography + alpha-blend compositor (from region_overlay.py).

Strategy: uses the camera-aware oriented homography to warp the overlay
with correct physical aspect ratio, then alpha-composites into the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.composite.base import Compositor


def estimate_rectified_border_fill(
    rectified_bgr: np.ndarray,
    *,
    band_fraction: float = 0.12,
) -> dict[str, object]:
    """Estimate a stable banner background color from a rectified border band."""
    height, width = rectified_bgr.shape[:2]
    band_px = max(2, int(round(min(height, width) * band_fraction)))
    band_px = min(band_px, max(2, min(height, width) // 4))

    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:band_px, :] = True
    border_mask[-band_px:, :] = True
    border_mask[:, :band_px] = True
    border_mask[:, -band_px:] = True

    border_pixels = rectified_bgr[border_mask].reshape(-1, 3).astype(np.float32)
    if border_pixels.size == 0:
        border_pixels = rectified_bgr.reshape(-1, 3).astype(np.float32)

    median = np.median(border_pixels, axis=0)
    spread = np.median(np.abs(border_pixels - median), axis=0)
    fill_color_bgr = tuple(int(round(float(channel))) for channel in median)
    fill_spread_bgr = tuple(round(float(channel), 2) for channel in spread)
    fill_unstable = max(fill_spread_bgr) > 28.0

    return {
        "fill_color_bgr": fill_color_bgr,
        "fill_spread_bgr": fill_spread_bgr,
        "fill_band_px": band_px,
        "fill_unstable": fill_unstable,
        "fill_warning_reason": "background_fill_unstable" if fill_unstable else None,
    }


class AlphaCompositor(Compositor):
    """Composites using oriented homography for aspect-ratio-correct warping."""

    def __init__(self) -> None:
        # Per-obj_id EMA cache of background fill colour (BGR float).
        # Stabilises the canvas behind the new logo across frames so the
        # surroundings of the substituted logo don't jitter.
        self._bg_color_ema: dict[int, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "alpha"

    def composite(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Requires ``homo`` (oriented-homography dict) in *kwargs*.

        Optional kwargs
        ---------------
        obj_id : int
            Object identifier — keys the bg_color EMA cache so each
            tracked banner gets its own stable background colour.
        bg_color_ema_alpha : float, default 0.1
            EMA weight on the freshly-sampled bg_color. Lower = smoother
            (slower to react), higher = closer to the per-frame value.
            0.0 freezes the bg_color after the first frame.
        """
        homo: dict = kwargs["homo"]
        padding: float = kwargs.get("padding", 0.05)
        debug_info: dict[str, object] | None = kwargs.get("debug_info")
        obj_id: int | None = kwargs.get("obj_id")
        bg_alpha: float = float(kwargs.get("bg_color_ema_alpha", 0.1))

        dst_w, dst_h = homo["dst_w"], homo["dst_h"]
        H_final = homo["H"]

        frame_h, frame_w = frame.shape[:2]

        # ROI bbox around the quad: every cv2 op below operates here, not
        # on the full frame.
        xs, ys = corners[:, 0], corners[:, 1]
        roi_pad = 4  # alpha has no inpaint dilation, small pad is enough
        x0 = max(0, int(xs.min()) - roi_pad)
        y0 = max(0, int(ys.min()) - roi_pad)
        x1 = min(frame_w, int(xs.max()) + roi_pad)
        y1 = min(frame_h, int(ys.max()) + roi_pad)
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            return frame
        corners_roi = corners - np.array([x0, y0], dtype=corners.dtype)
        frame_roi = frame[y0:y1, x0:x1]

        avail_w = int(dst_w * (1 - 2 * padding))
        avail_h = int(dst_h * (1 - 2 * padding))

        ov_h, ov_w = overlay.shape[:2]
        scale = min(avail_w / ov_w, avail_h / ov_h)
        new_w = max(1, int(round(ov_w * scale)))
        new_h = max(1, int(round(ov_h * scale)))
        ov_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Sample background colour from the frame's banner region (rectified).
        # H_to_rect maps full-frame corners → dst_rect, but we already have
        # frame_roi and corners_roi, so build a ROI version.
        H_to_rect_roi, _ = cv2.findHomography(corners_roi, homo["dst_rect"])
        warped_orig = cv2.warpPerspective(frame_roi, H_to_rect_roi, (dst_w, dst_h))
        fill_info = estimate_rectified_border_fill(warped_orig)
        sampled_bgr = np.asarray(fill_info["fill_color_bgr"], dtype=np.float32)

        # Temporal EMA on bg_color, keyed by obj_id when provided. Falls
        # back to the raw per-frame sample when obj_id is missing or
        # bg_alpha == 1 (= no smoothing).
        if obj_id is not None and 0.0 <= bg_alpha < 1.0:
            prev = self._bg_color_ema.get(int(obj_id))
            if prev is None:
                smoothed = sampled_bgr
            else:
                smoothed = bg_alpha * sampled_bgr + (1.0 - bg_alpha) * prev
            self._bg_color_ema[int(obj_id)] = smoothed
            bg_color = tuple(int(round(float(c))) for c in smoothed)
        else:
            bg_color = tuple(int(round(float(c))) for c in sampled_bgr)

        if debug_info is not None:
            debug_info.update(fill_info)
            debug_info["bg_color_ema"] = bg_color

        canvas = np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
        ox = (dst_w - new_w) // 2
        oy = (dst_h - new_h) // 2

        if ov_resized.ndim == 3 and ov_resized.shape[2] == 4:
            rgb = ov_resized[:, :, :3].astype(np.float32)
            alpha = ov_resized[:, :, 3:].astype(np.float32) / 255.0
            patch = canvas[oy : oy + new_h, ox : ox + new_w].astype(np.float32)
            canvas[oy : oy + new_h, ox : ox + new_w] = (rgb * alpha + patch * (1 - alpha)).astype(
                np.uint8
            )
        else:
            canvas[oy : oy + new_h, ox : ox + new_w] = ov_resized[:, :, :3]

        # H_final maps dst_rect → full-frame corners. Translate to ROI by
        # composing with a translation: corners_roi = corners - [x0, y0].
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=H_final.dtype)
        H_roi = T @ H_final
        warped_canvas_roi = cv2.warpPerspective(canvas, H_roi, (roi_w, roi_h))

        region_mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(region_mask_roi, corners_roi.astype(np.int32), 255)

        # In-place write back into the source frame slice.
        frame_roi[region_mask_roi > 0] = warped_canvas_roi[region_mask_roi > 0]
        return frame
