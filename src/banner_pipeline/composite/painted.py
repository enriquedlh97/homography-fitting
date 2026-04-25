"""Court-floor "painted" compositing.

Adapted from tennis-virtual-ads painted_blend.py. Warps the logo into
full-frame space, applies shade map from court illumination, and uses
occlusion mask so players appear in front.

This is used for court_floor surfaces where the ROI-based inpaint
compositor produces artifacts (visible patches, wrong perspective).
"""

from __future__ import annotations

import cv2
import numpy as np


def painted_court_composite(
    frame: np.ndarray,
    corners: np.ndarray,
    overlay_rgba: np.ndarray,
    occlusion_mask: np.ndarray | None = None,
    shade_blur_ksize: int = 41,
    shade_strength: float = 1.0,
    alpha_feather_px: int = 3,
    alpha_scale: float = 0.75,
    shade_clip_min: float = 0.6,
    shade_clip_max: float = 1.4,
) -> np.ndarray:
    """Composite a logo onto the court surface with painted-on look.

    1. Warps the overlay into full-frame space using the quad corners.
    2. Extracts illumination field from the original frame.
    3. Applies shade map so the logo darkens in shadows.
    4. Applies occlusion mask so players appear in front.
    5. Alpha-blends onto the frame.

    Parameters
    ----------
    frame : (H, W, 3) uint8 BGR frame (mutated in-place).
    corners : (4, 2) float32 quad corners [TL, TR, BR, BL].
    overlay_rgba : (h, w, 4) uint8 BGRA logo image.
    occlusion_mask : (H, W) float32 person mask, 1.0 = person.
    """
    fh, fw = frame.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    # --- 1. Warp logo into full-frame space ---
    src_corners = np.array(
        [[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]],
        dtype=np.float32,
    )
    dst_corners = corners.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped_rgba = cv2.warpPerspective(
        overlay_rgba,
        M,
        (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    warped_bgr = warped_rgba[:, :, :3]
    warped_mask = warped_rgba[:, :, 3].astype(np.float32) / 255.0

    # --- 2. Alpha feather ---
    if alpha_feather_px > 0:
        ksize = 2 * alpha_feather_px + 1
        warped_mask = cv2.GaussianBlur(warped_mask, (ksize, ksize), 0)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)

    # --- 3. Scale alpha for painted-on transparency ---
    if alpha_scale < 1.0:
        warped_mask = warped_mask * alpha_scale

    # --- 4. Apply occlusion mask (players in front) ---
    if occlusion_mask is not None:
        effective_alpha = warped_mask * (1.0 - occlusion_mask)
    else:
        effective_alpha = warped_mask

    # --- 5. Shade map from court illumination ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_ksize = shade_blur_ksize if shade_blur_ksize % 2 == 1 else shade_blur_ksize + 1
    illumination = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Normalize by median brightness in the logo footprint.
    ad_pixels = illumination[effective_alpha > 0.01]
    median_illum = max(float(np.median(ad_pixels)), 1.0) if len(ad_pixels) > 0 else 128.0
    shade_map = np.clip(illumination / median_illum, shade_clip_min, shade_clip_max)

    if shade_strength != 1.0:
        shade_map = np.power(shade_map, shade_strength)

    # --- 6. Apply shading to logo colors ---
    shaded_bgr = warped_bgr.astype(np.float32) / 255.0
    shaded_bgr = shaded_bgr * shade_map[:, :, np.newaxis]
    shaded_bgr = np.clip(shaded_bgr * 255.0, 0, 255)

    # --- 7. Composite ---
    alpha_3ch = effective_alpha[:, :, np.newaxis]
    blended = frame.astype(np.float32) * (1.0 - alpha_3ch) + shaded_bgr * alpha_3ch
    frame[:] = blended.astype(np.uint8)

    return frame
