"""Inpaint + luminosity-matching compositor (from banner_segment.py).

Strategy: inpaint old region → build logo canvas with aspect-aware
sizing → match luminosity in LAB space → alpha-blend with soft edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline._perf import Timer
from banner_pipeline.composite.base import Compositor


class InpaintCompositor(Compositor):
    """Inpaints the old logo away, then warps the new one with luminosity matching."""

    def __init__(self) -> None:
        # Cached BGRA copies of overlays we've seen, keyed by id().
        self._logo_bgra_cache: dict[int, np.ndarray] = {}
        # Cached resized logo canvases, keyed by (id(overlay), new_w, new_h).
        self._logo_resize_cache: dict[tuple[int, int, int], np.ndarray] = {}
        # Reusable structuring element for the inpaint dilation step.
        self._dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @property
    def name(self) -> str:
        return "inpaint"

    def composite(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        padding: float = kwargs.get("padding", 0.05)

        # Cache BGRA overlay (constant across the entire video run).
        with Timer("inpaint.bgr2bgra"):
            oid = id(overlay)
            cached_bgra = self._logo_bgra_cache.get(oid)
            if cached_bgra is None:
                cached_bgra = (
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA) if overlay.shape[2] == 3 else overlay
                )
                self._logo_bgra_cache[oid] = cached_bgra
            overlay = cached_bgra

        frame_h, frame_w = frame.shape[:2]

        # --- Compute ROI bbox from quad corners (with padding for inpaint
        #     dilation + alpha feather). Everything below operates on the
        #     small ROI, not the full frame. ---
        with Timer("inpaint.roi_setup"):
            xs, ys = corners[:, 0], corners[:, 1]
            roi_pad = 32
            x0 = max(0, int(xs.min()) - roi_pad)
            y0 = max(0, int(ys.min()) - roi_pad)
            x1 = min(frame_w, int(xs.max()) + roi_pad)
            y1 = min(frame_h, int(ys.max()) + roi_pad)
            roi_w, roi_h = x1 - x0, y1 - y0
            if roi_w <= 0 or roi_h <= 0:
                return frame  # degenerate quad outside the frame
            corners_roi = corners - np.array([x0, y0], dtype=corners.dtype)
            frame_roi = frame[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1] if mask is not None else None

        # --- Step 1: inpaint (ROI only) ---
        with Timer("inpaint.inpaint"):
            if mask_roi is not None:
                mask_u8_roi = (mask_roi > 0).astype(np.uint8) * 255
                mask_u8_roi = cv2.dilate(mask_u8_roi, self._dilate_kern)
                inpainted_roi = cv2.inpaint(
                    frame_roi, mask_u8_roi, inpaintRadius=5, flags=cv2.INPAINT_TELEA
                )
            else:
                inpainted_roi = frame_roi.copy()

        # --- Step 2: build logo + alpha canvases (size depends on quad,
        #     not ROI; this stays as-is) ---
        with Timer("inpaint.build_canvas"):
            w_top = np.linalg.norm(corners[1] - corners[0])
            w_bot = np.linalg.norm(corners[2] - corners[3])
            h_left = np.linalg.norm(corners[3] - corners[0])
            h_right = np.linalg.norm(corners[2] - corners[1])
            avg_w = (w_top + w_bot) / 2
            avg_h = (h_left + h_right) / 2
            scale_up = max(1.0, 500.0 / max(float(avg_w), float(avg_h)))
            canvas_w = max(int(avg_w * scale_up), 1)
            canvas_h = max(int(avg_h * scale_up), 1)

            rgb_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

            logo_h, logo_w = overlay.shape[:2]
            pad_w = int(canvas_w * padding)
            pad_h = int(canvas_h * padding)
            scale = min((canvas_w - 2 * pad_w) / logo_w, (canvas_h - 2 * pad_h) / logo_h)
            new_w, new_h = int(logo_w * scale), int(logo_h * scale)
            # Cache the resized logo: same overlay + same target size repeats
            # across frames (and across same-sized banners within a frame).
            resize_key = (oid, new_w, new_h)
            logo_resized = self._logo_resize_cache.get(resize_key)
            if logo_resized is None:
                logo_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Bound the cache so it doesn't grow unbounded across runs.
                if len(self._logo_resize_cache) > 64:
                    self._logo_resize_cache.clear()
                self._logo_resize_cache[resize_key] = logo_resized

            cx0 = (canvas_w - new_w) // 2
            cy0 = (canvas_h - new_h) // 2
            rgb_canvas[cy0 : cy0 + new_h, cx0 : cx0 + new_w] = logo_resized[:, :, :3]
            alpha_canvas[cy0 : cy0 + new_h, cx0 : cx0 + new_w] = logo_resized[:, :, 3]

        # --- Step 3: warp into ROI space (small dest, not full frame) ---
        with Timer("inpaint.warp"):
            src = np.array(
                [[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]],
                dtype=np.float32,
            )
            H_roi, _ = cv2.findHomography(src, corners_roi.astype(np.float32))
            warped_rgb = cv2.warpPerspective(rgb_canvas, H_roi, (roi_w, roi_h))
            warped_alpha = cv2.warpPerspective(alpha_canvas, H_roi, (roi_w, roi_h))

        # --- Step 4: luminosity matching (LAB) on ROI ---
        with Timer("inpaint.lab_match"):
            logo_pixels = warped_alpha > 0
            if logo_pixels.any() and mask_roi is not None:
                orig_lab = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
                orig_mask_l = orig_lab[mask_roi > 0, 0]
                if orig_mask_l.size > 0:
                    orig_l_lo, orig_l_hi = np.percentile(orig_mask_l, [10, 90])

                    new_lab = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
                    new_l = new_lab[logo_pixels, 0]
                    new_l_lo, new_l_hi = np.percentile(new_l, [10, 90])

                    s = (
                        (orig_l_hi - orig_l_lo) / (new_l_hi - new_l_lo)
                        if new_l_hi - new_l_lo > 1
                        else 1.0
                    )
                    new_lab[logo_pixels, 0] = np.clip(
                        (new_l - new_l_lo) * s + orig_l_lo,
                        0,
                        255,
                    )
                    warped_rgb = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # --- Step 5: soft-edge alpha blend (ROI only) + in-place write-back ---
        with Timer("inpaint.blend"):
            warped_alpha = cv2.GaussianBlur(warped_alpha, (5, 5), 1.0)
            a = (warped_alpha.astype(np.float32) / 255.0)[..., None]
            result_roi = (
                warped_rgb.astype(np.float32) * a + inpainted_roi.astype(np.float32) * (1.0 - a)
            ).astype(np.uint8)
            # Write back into the source frame in-place. Safe because the
            # video pipeline always passes a freshly-loaded frame.
            frame[y0:y1, x0:x1] = result_roi

        return frame
