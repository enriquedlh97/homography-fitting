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
    """Inpaints the old logo away, then warps the new one with luminosity matching.

    Supports:
    - ``lum_strength`` (0-1): controls how aggressively the luminosity is
      remapped.  1.0 = full remap (our original behavior); lower values
      preserve more of the logo's native appearance.
    - Auto-captured ``ref_lum``: the luminosity stats from the first call
      are reused for all subsequent calls, preventing per-frame flicker.
    - ``occlusion_mask``: pixels where this mask is True are excluded
      from the logo overlay (e.g. a player in front of the banner).
    """

    def __init__(self) -> None:
        # Cached BGRA copies of overlays we've seen, keyed by id().
        self._logo_bgra_cache: dict[int, np.ndarray] = {}
        # Cached resized logo canvases, keyed by (id(overlay), new_w, new_h).
        self._logo_resize_cache: dict[tuple[int, int, int], np.ndarray] = {}
        # Reusable structuring element for the inpaint dilation step.
        self._dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Reference luminosity stats auto-captured on the first composite call
        # per object. Reused for temporal consistency across frames.
        self._ref_lum_captured: bool = False
        self._ref_lum: tuple[float, float] | None = None

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
        lum_strength: float = kwargs.get("lum_strength", 1.0)
        do_inpaint: bool = kwargs.get("inpaint", True)
        inpaint_method: str = kwargs.get("inpaint_method", "telea")
        inpaint_radius: int = kwargs.get("inpaint_radius", 3)
        mask_dilate_px: int = kwargs.get("mask_dilate_px", 3)
        alpha_feather_px: int = kwargs.get("alpha_feather_px", 5)
        occlusion_mask: np.ndarray | None = kwargs.get("occlusion_mask")

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
        # When inpaint=False, skip Telea inpainting and just alpha-blend
        # the new logo directly on top. This avoids the color mismatch
        # artifact where the inpainted black differs from the banner tone.
        with Timer("inpaint.inpaint"):
            if do_inpaint and mask_roi is not None:
                mask_u8_roi = (mask_roi > 0).astype(np.uint8) * 255
                dilate_kern = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (mask_dilate_px, mask_dilate_px),
                )
                mask_u8_roi = cv2.dilate(mask_u8_roi, dilate_kern)

                if inpaint_method == "median_fill":
                    # Simple median-color fill: sample the border pixels
                    # around the mask and fill the interior with their
                    # median color. Avoids the Telea "paint brush" artifact.
                    border = (
                        cv2.dilate(
                            mask_u8_roi,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                        )
                        & ~mask_u8_roi
                    )
                    inpainted_roi = frame_roi.copy()
                    if np.any(border > 0):
                        fill_color = np.median(
                            frame_roi[border > 0],
                            axis=0,
                        ).astype(np.uint8)
                        inpainted_roi[mask_u8_roi > 0] = fill_color
                elif inpaint_method == "ns":
                    # Navier-Stokes inpainting (better at preserving gradients).
                    inpainted_roi = cv2.inpaint(
                        frame_roi,
                        mask_u8_roi,
                        inpaintRadius=inpaint_radius,
                        flags=cv2.INPAINT_NS,
                    )
                else:
                    # Telea inpainting (default, iterative algorithm).
                    inpainted_roi = cv2.inpaint(
                        frame_roi,
                        mask_u8_roi,
                        inpaintRadius=inpaint_radius,
                        flags=cv2.INPAINT_TELEA,
                    )
            else:
                inpainted_roi = frame_roi.copy()

        # --- Step 2: build logo + alpha canvases ---
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
            resize_key = (oid, new_w, new_h)
            logo_resized = self._logo_resize_cache.get(resize_key)
            if logo_resized is None:
                logo_resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
        # Uses ref_lum from the first call for temporal consistency.
        # lum_strength blends between the original L and the remapped L
        # so the logo doesn't get over-corrected.
        with Timer("inpaint.lab_match"):
            logo_pixels = warped_alpha > 0
            has_mask = mask_roi is not None
            has_ref = self._ref_lum_captured and self._ref_lum is not None
            if logo_pixels.any() and (has_mask or has_ref):
                # Use captured ref_lum for temporal consistency, or compute
                # fresh stats from this frame (and capture for next time).
                if self._ref_lum_captured and self._ref_lum is not None:
                    orig_l_lo, orig_l_hi = self._ref_lum
                elif mask_roi is not None:
                    orig_lab = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
                    orig_mask_l = orig_lab[mask_roi > 0, 0]
                    if orig_mask_l.size > 0:
                        orig_l_lo, orig_l_hi = (
                            float(np.percentile(orig_mask_l, 10)),
                            float(np.percentile(orig_mask_l, 90)),
                        )
                        if not self._ref_lum_captured:
                            self._ref_lum = (orig_l_lo, orig_l_hi)
                            self._ref_lum_captured = True
                    else:
                        orig_l_lo, orig_l_hi = 0.0, 255.0
                else:
                    # No mask and no ref_lum yet: skip LAB matching.
                    orig_l_lo, orig_l_hi = 0.0, 255.0

                new_lab = cv2.cvtColor(warped_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
                new_l = new_lab[logo_pixels, 0]
                new_l_lo, new_l_hi = (
                    float(np.percentile(new_l, 10)),
                    float(np.percentile(new_l, 90)),
                )

                s = (
                    (orig_l_hi - orig_l_lo) / (new_l_hi - new_l_lo)
                    if new_l_hi - new_l_lo > 1
                    else 1.0
                )
                remapped_l = np.clip((new_l - new_l_lo) * s + orig_l_lo, 0, 255)
                # Blend: lum_strength=1.0 = full remap; <1 preserves logo's
                # native luminosity, preventing over-correction and tinting.
                new_lab[logo_pixels, 0] = new_l * (1 - lum_strength) + remapped_l * lum_strength
                warped_rgb = cv2.cvtColor(new_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # --- Step 5: shadow-preserving blend + soft-edge alpha ---
        # Ported from tennis-virtual-ads painted_blend.py: the logo inherits
        # the frame's illumination field so it looks painted on the surface.
        shade_blend: bool = kwargs.get("shade_blend", False)

        with Timer("inpaint.blend"):
            ksize = alpha_feather_px if alpha_feather_px % 2 == 1 else alpha_feather_px + 1
            warped_alpha = cv2.GaussianBlur(warped_alpha, (ksize, ksize), 0)

            # Subtract occlusion mask so occluding objects (players) appear
            # in front of the composited logo.
            if occlusion_mask is not None:
                occ_roi = occlusion_mask[y0:y1, x0:x1]
                occ_u8 = (np.squeeze(occ_roi) > 0).astype(np.uint8) * 255
                occ_dilated = cv2.dilate(
                    occ_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                )
                occ_soft = cv2.GaussianBlur(occ_dilated, (11, 11), 3.0)
                warped_alpha = np.clip(
                    warped_alpha.astype(np.float32) - occ_soft.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)

            a = (warped_alpha.astype(np.float32) / 255.0)[..., None]

            if shade_blend:
                # Shadow-preserving: extract illumination from the original
                # frame ROI, normalize, and apply to the warped logo.
                gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
                illumination = cv2.GaussianBlur(gray_roi, (41, 41), 0)
                # Normalize by the median brightness in the mask area.
                if mask_roi is not None and np.any(mask_roi > 0):
                    median_bright = float(np.median(illumination[mask_roi > 0]))
                else:
                    median_bright = float(np.median(illumination))
                median_bright = max(median_bright, 1.0)
                shade_map = np.clip(illumination / median_bright, 0.6, 1.4)
                # Apply shading to warped logo colors only (not background,
                # which already has the correct illumination from inpainting).
                warped_float = warped_rgb.astype(np.float32) * shade_map[:, :, np.newaxis]
                warped_float = np.clip(warped_float, 0, 255)
                result_roi = (
                    warped_float * a + inpainted_roi.astype(np.float32) * (1.0 - a)
                ).astype(np.uint8)
            else:
                result_roi = (
                    warped_rgb.astype(np.float32) * a + inpainted_roi.astype(np.float32) * (1.0 - a)
                ).astype(np.uint8)

            frame[y0:y1, x0:x1] = result_roi

        return frame
