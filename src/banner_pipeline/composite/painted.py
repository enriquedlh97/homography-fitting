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
    sam_mask: np.ndarray | None = None,
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
    4. Uses SAM mask for natural occlusion — the SAM video propagation
       mask naturally excludes players walking over the text, so no
       separate person masker is needed.
    5. Alpha-blends onto the frame.

    Parameters
    ----------
    frame : (H, W, 3) uint8 BGR frame (mutated in-place).
    corners : (4, 2) float32 quad corners [TL, TR, BR, BL].
    overlay_rgba : (h, w, 4) uint8 BGRA logo image.
    sam_mask : (H, W) binary mask from SAM video propagation.
              Naturally excludes occluders (players). When provided,
              the logo is only shown where SAM says the surface is visible.
    occlusion_mask : (H, W) float32 person mask (fallback if no SAM mask).
    """
    fh, fw = frame.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    # --- 1. Build canvas preserving logo aspect ratio ---
    # Same approach as the ROI-based compositor: create a canvas
    # at the quad's dimensions, then center the logo with padding.
    w_top = float(np.linalg.norm(corners[1] - corners[0]))
    w_bot = float(np.linalg.norm(corners[2] - corners[3]))
    h_left = float(np.linalg.norm(corners[3] - corners[0]))
    h_right = float(np.linalg.norm(corners[2] - corners[1]))
    avg_w = (w_top + w_bot) / 2
    avg_h = (h_left + h_right) / 2
    canvas_w = max(int(avg_w), 1)
    canvas_h = max(int(avg_h), 1)

    # Scale logo to fit canvas, preserving aspect ratio.
    pad_w = int(canvas_w * alpha_scale * 0.0)  # no extra padding
    pad_h = int(canvas_h * 0.0)
    scale = min((canvas_w - 2 * pad_w) / ow, (canvas_h - 2 * pad_h) / oh)
    new_w, new_h = int(ow * scale), int(oh * scale)
    if new_w < 1 or new_h < 1:
        return frame
    logo_resized = cv2.resize(overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on canvas.
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = logo_resized

    # --- 2. Warp canvas into full-frame space ---
    src_corners = np.array(
        [[0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]],
        dtype=np.float32,
    )
    dst_corners = corners.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped_rgba = cv2.warpPerspective(
        canvas,
        M,
        (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    warped_bgr = warped_rgba[:, :, :3]
    warped_mask = warped_rgba[:, :, 3].astype(np.float32) / 255.0

    # --- 3. Alpha feather ---
    if alpha_feather_px > 0:
        ksize = 2 * alpha_feather_px + 1
        warped_mask = cv2.GaussianBlur(warped_mask, (ksize, ksize), 0)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)

    # --- 4. Scale alpha for painted-on transparency ---
    if alpha_scale < 1.0:
        warped_mask = warped_mask * alpha_scale

    # --- 5. Apply occlusion masks ---
    # Combine SAM mask (text region) AND person mask (player exclusion)
    # for robust occlusion. SAM handles most cases but can leak at
    # player feet/shadow edges — the person mask catches those.
    effective_alpha = warped_mask
    if sam_mask is not None:
        sam_float = sam_mask.astype(np.float32)
        if sam_float.max() > 1:
            sam_float = sam_float / 255.0
        # Dilate SAM mask so logo extends beyond strict text pixels.
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        sam_dilated = cv2.dilate(sam_float, kern, iterations=1)
        sam_dilated = cv2.GaussianBlur(sam_dilated, (11, 11), 0)
        sam_dilated = np.clip(sam_dilated, 0.0, 1.0)
        effective_alpha = warped_mask * sam_dilated
    # Always subtract person mask if available (belt + suspenders).
    if occlusion_mask is not None:
        effective_alpha = effective_alpha * (1.0 - occlusion_mask)

    # --- 6. Shade map from court illumination ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_ksize = shade_blur_ksize if shade_blur_ksize % 2 == 1 else shade_blur_ksize + 1
    illumination = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Normalize by median brightness in the logo footprint.
    ad_pixels = illumination[effective_alpha > 0.01]
    median_illum = max(float(np.median(ad_pixels)), 1.0) if len(ad_pixels) > 0 else 128.0
    shade_map = np.clip(illumination / median_illum, shade_clip_min, shade_clip_max)

    if shade_strength != 1.0:
        shade_map = np.power(shade_map, shade_strength)

    # --- 7. Apply shading to logo colors ---
    shaded_bgr = warped_bgr.astype(np.float32) / 255.0
    shaded_bgr = shaded_bgr * shade_map[:, :, np.newaxis]
    shaded_bgr = np.clip(shaded_bgr * 255.0, 0, 255)

    # --- 8. Composite ---
    alpha_3ch = effective_alpha[:, :, np.newaxis]
    blended = frame.astype(np.float32) * (1.0 - alpha_3ch) + shaded_bgr * alpha_3ch
    frame[:] = blended.astype(np.uint8)

    return frame
