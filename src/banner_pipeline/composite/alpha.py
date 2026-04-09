"""Oriented-homography + alpha-blend compositor (from region_overlay.py).

Strategy: uses the camera-aware oriented homography to warp the overlay
with correct physical aspect ratio, then alpha-composites into the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.composite.base import Compositor


class AlphaCompositor(Compositor):
    """Composites using oriented homography for aspect-ratio-correct warping."""

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
        """Requires ``homo`` (oriented-homography dict) in *kwargs*."""
        homo: dict = kwargs["homo"]
        padding: float = kwargs.get("padding", 0.05)

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
        bg_color = tuple(int(c) for c in cv2.mean(warped_orig)[:3])

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
