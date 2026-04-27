"""Court-floor "painted" compositing.

Adapted from tennis-virtual-ads painted_blend.py. Two-phase approach:
1. Erase original text via NS inpainting (polygon mask from corners).
2. Warp the logo into full-frame space with shade map + occlusion.

Phase 1 ensures the original text (e.g. MELBOURNE) is fully removed.
Phase 2 overlays the new logo with perspective-correct shading.
When a player walks over the logo, the occlusion mask reveals the
*inpainted* court (not the original text).
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
    # --- Shade map ---
    shade_blur_ksize: int = 41,
    shade_strength: float = 1.0,
    shade_clip_min: float = 0.6,
    shade_clip_max: float = 1.4,
    # --- Logo alpha ---
    alpha_feather_px: int = 3,
    alpha_scale: float = 0.95,
    # --- Text erasure (Phase 1) ---
    erase_text: bool = True,
    erase_dilate_px: int = 12,
    erase_feather_px: int = 20,
    erase_inpaint_radius: int = 7,
    # --- Occlusion (Phase 2) ---
    occlusion_dilate_px: int = 10,
    occlusion_feather_ksize: int = 21,
    occlusion_feather_sigma: float = 4.0,
    # --- Quad expansion ---
    quad_expand_px: int = 0,
    # --- Erase-only mode ---
    erase_only: bool = False,
) -> np.ndarray:
    """Composite a logo onto the court surface with painted-on look.

    Parameters
    ----------
    frame : (H, W, 3) uint8 BGR frame (mutated in-place).
    corners : (4, 2) float32 quad corners [TL, TR, BR, BL].
    overlay_rgba : (h, w, 4) uint8 BGRA logo image.
    sam_mask : (H, W) binary mask from SAM (text region).
    occlusion_mask : (H, W) float32 person mask from Mask R-CNN.
                     Should be the RAW mask (pre-dilation) — this
                     function applies its own dilation + feathering.
    erase_text : If True, inpaint original text before logo overlay.
    erase_dilate_px : Dilation for text erasure mask (pixels).
    erase_feather_px : Feathering for text erasure boundary.
    erase_inpaint_radius : cv2.inpaint radius parameter.
    occlusion_dilate_px : Dilation for person mask (pixels).
    occlusion_feather_ksize : Gaussian blur kernel size for person mask.
    occlusion_feather_sigma : Gaussian blur sigma for person mask.
    erase_only : If True, only erase text (Phase 1) — skip logo overlay.
    """
    fh, fw = frame.shape[:2]

    # ================================================================
    # PHASE 1: Erase original text via color fill
    # ================================================================
    if erase_text or erase_only:
        _erase_original_text(
            frame,
            corners,
            sam_mask,
            dilate_px=erase_dilate_px,
            feather_px=erase_feather_px,
            inpaint_radius=erase_inpaint_radius,
            occlusion_mask=occlusion_mask,
        )

    # In erase-only mode, skip logo overlay entirely.
    if erase_only:
        return frame

    oh, ow = overlay_rgba.shape[:2]

    # ================================================================
    # PHASE 2: Overlay new logo
    # ================================================================

    # --- 2a. Expand quad if requested (to cover text that leaks beyond SAM bbox) ---
    if quad_expand_px > 0:
        center = np.mean(corners, axis=0)
        expanded = corners.copy()
        for i in range(4):
            direction = corners[i] - center
            direction_len = np.linalg.norm(direction)
            if direction_len > 0:
                expanded[i] = corners[i] + (direction / direction_len) * quad_expand_px
        corners = expanded

    # --- 2b. Build canvas preserving logo aspect ratio ---
    w_top = float(np.linalg.norm(corners[1] - corners[0]))
    w_bot = float(np.linalg.norm(corners[2] - corners[3]))
    h_left = float(np.linalg.norm(corners[3] - corners[0]))
    h_right = float(np.linalg.norm(corners[2] - corners[1]))
    avg_w = (w_top + w_bot) / 2
    avg_h = (h_left + h_right) / 2
    canvas_w = max(int(avg_w), 1)
    canvas_h = max(int(avg_h), 1)

    scale = min(canvas_w / ow, canvas_h / oh)
    new_w, new_h = int(ow * scale), int(oh * scale)
    if new_w < 1 or new_h < 1:
        return frame
    logo_resized = cv2.resize(overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # If not erasing text, fill canvas with sampled court color so the
    # opaque background covers original text (e.g. MELBOURNE).
    # If erasing text, transparent canvas is fine (text already erased).
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    if not erase_text:
        # Sample court color from regions LEFT and RIGHT of the ORIGINAL
        # text bbox (before quad expansion). These positions are on the
        # same court surface as the text, giving the best color match.
        # Use the prompt bbox corners (pre-expansion) for reference.
        x_min = int(np.min(corners[:, 0]))
        x_max = int(np.max(corners[:, 0]))
        y_min = int(np.min(corners[:, 1]))
        y_max = int(np.max(corners[:, 1]))
        # Ensure we sample from the vertical CENTER of the quad
        # (not at edges which might be off the court surface).
        y_mid = (y_min + y_max) // 2
        y_half = max((y_max - y_min) // 2, 10)
        pad = 20  # distance outside quad to sample
        sample_w = 40  # width of sampling region
        samples = []
        # Left of quad
        left = frame[
            max(0, y_mid - y_half) : min(fh, y_mid + y_half),
            max(0, x_min - pad - sample_w) : max(1, x_min - pad),
        ]
        if left.size > 0:
            samples.append(left.reshape(-1, 3))
        # Right of quad
        right = frame[
            max(0, y_mid - y_half) : min(fh, y_mid + y_half),
            min(fw - 1, x_max + pad) : min(fw, x_max + pad + sample_w),
        ]
        if right.size > 0:
            samples.append(right.reshape(-1, 3))
        # Above quad (small strip)
        above = frame[
            max(0, y_min - pad - 15) : max(1, y_min - pad),
            max(0, x_min) : min(fw, x_max),
        ]
        if above.size > 0:
            samples.append(above.reshape(-1, 3))
        if samples:
            all_samples = np.concatenate(samples, axis=0)
            court_bgr = np.median(all_samples, axis=0).astype(np.uint8)
        else:
            court_bgr = np.array([158, 102, 66], dtype=np.uint8)
        # Fill canvas with court color (fully opaque).
        canvas[:, :, 0] = court_bgr[0]
        canvas[:, :, 1] = court_bgr[1]
        canvas[:, :, 2] = court_bgr[2]
        canvas[:, :, 3] = 255
    x_off = (canvas_w - new_w) // 2
    y_off = (canvas_h - new_h) // 2
    # Alpha-composite logo onto canvas (don't overwrite — that would
    # replace the opaque court-colored background with transparent logo
    # pixels, letting original text bleed through).
    region = canvas[y_off : y_off + new_h, x_off : x_off + new_w]
    logo_a = logo_resized[:, :, 3:4].astype(np.float32) / 255.0
    inv_a = 1.0 - logo_a
    region[:, :, :3] = (
        logo_resized[:, :, :3].astype(np.float32) * logo_a
        + region[:, :, :3].astype(np.float32) * inv_a
    ).astype(np.uint8)
    region[:, :, 3] = np.maximum(region[:, :, 3], logo_resized[:, :, 3])

    # --- 2b. Warp canvas into full-frame space ---
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

    # --- 2c. Alpha feather ---
    if alpha_feather_px > 0:
        ksize = 2 * alpha_feather_px + 1
        warped_mask = cv2.GaussianBlur(warped_mask, (ksize, ksize), 0)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)

    # --- 2d. Scale alpha for painted-on transparency ---
    if alpha_scale < 1.0:
        warped_mask = warped_mask * alpha_scale

    # --- 2e. Apply occlusion ---
    # Two modes:
    # (a) SAM2 mask mode: sam_mask defines where the logo should appear.
    #     Players naturally create "holes" in the SAM2 mask, so no
    #     separate person masker is needed. This is the preferred mode.
    # (b) Person mask mode: separate occlusion_mask from Mask R-CNN/RVM.
    #     When erase_text=False (overlay mode), ALWAYS use person mask
    #     mode — the warped quad already covers the text area, so we
    #     don't need SAM to restrict the overlay. Using SAM would leave
    #     gaps where SAM missed text pixels.
    effective_alpha = warped_mask
    if not erase_text and occlusion_mask is not None:
        # Overlay mode (tennis-virtual-ads approach): use BINARY dilated
        # player mask with NO feathering. This gives a clean hard cut
        # at the player boundary — feet are never semi-transparent.
        # Feathering the occlusion mask was the root cause of soft feet.
        occ = occlusion_mask.astype(np.float32)
        if occ.max() > 1:
            occ = occ / 255.0
        occ_binary = (occ > 0.5).astype(np.uint8)
        if np.any(occ_binary) and occlusion_dilate_px > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * occlusion_dilate_px + 1, 2 * occlusion_dilate_px + 1),
            )
            occ_binary = cv2.dilate(occ_binary, kern, iterations=1)
        effective_alpha = warped_mask * (1.0 - occ_binary.astype(np.float32))
    elif sam_mask is not None and np.any(sam_mask > 0):
        # SAM2 mask mode: logo only appears where SAM2 says "text".
        # SAM2 naturally excludes players → clean occlusion.
        sam_float = sam_mask.astype(np.float32)
        if sam_float.max() > 1:
            sam_float = sam_float / 255.0
        # Slight Gaussian blur for anti-aliased boundaries.
        sam_float = cv2.GaussianBlur(sam_float, (5, 5), 1.0)
        sam_float = np.clip(sam_float, 0.0, 1.0)
        effective_alpha = warped_mask * sam_float
    elif occlusion_mask is not None:
        # Fallback: separate person mask.
        occ = occlusion_mask.astype(np.float32)
        if occ.max() > 1:
            occ = occ / 255.0
        is_binary = np.all((occ == 0) | (occ == 1))
        if is_binary and np.any(occ > 0):
            occ = _process_occlusion_mask(
                occ,
                dilate_px=occlusion_dilate_px,
                feather_ksize=occlusion_feather_ksize,
                feather_sigma=occlusion_feather_sigma,
            )
        effective_alpha = warped_mask * (1.0 - np.clip(occ, 0.0, 1.0))

    # --- 2f. Shade map from court illumination ---
    # Computed from the (now inpainted) frame — no text artifacts.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_ksize = shade_blur_ksize if shade_blur_ksize % 2 == 1 else shade_blur_ksize + 1
    illumination = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    ad_pixels = illumination[effective_alpha > 0.01]
    median_illum = max(float(np.median(ad_pixels)), 1.0) if len(ad_pixels) > 0 else 128.0
    shade_map = np.clip(illumination / median_illum, shade_clip_min, shade_clip_max)
    if shade_strength != 1.0:
        shade_map = np.power(shade_map, shade_strength)

    # --- 2g. Apply shading to logo colors ---
    shaded_bgr = warped_bgr.astype(np.float32) / 255.0
    shaded_bgr = shaded_bgr * shade_map[:, :, np.newaxis]
    shaded_bgr = np.clip(shaded_bgr * 255.0, 0, 255)

    # --- 2h. Composite ---
    alpha_3ch = effective_alpha[:, :, np.newaxis]
    blended = frame.astype(np.float32) * (1.0 - alpha_3ch) + shaded_bgr * alpha_3ch
    frame[:] = blended.astype(np.uint8)

    return frame


def _erase_original_text(
    frame: np.ndarray,
    corners: np.ndarray,
    sam_mask: np.ndarray | None,
    dilate_px: int = 12,
    feather_px: int = 20,
    inpaint_radius: int = 7,
    occlusion_mask: np.ndarray | None = None,
) -> None:
    """Erase original court text via color fill.

    Creates a polygon mask from the quad corners (more reliable than SAM
    for low-contrast text on court surfaces). If a SAM mask is provided,
    it's merged in for better text boundary coverage.

    When *occlusion_mask* is provided (player mask), the text erasure is
    suppressed under players — the original frame is preserved there
    because the player's body covers the text anyway.

    The fill region is feathered into the surrounding court for a
    seamless transition — no visible rectangular patches.
    """
    fh, fw = frame.shape[:2]

    # Save original frame — used to composite the player ON TOP of the
    # erased court, guaranteeing feet are never affected by erasure.
    original_frame = frame.copy()

    # Build polygon mask from quad corners.
    poly = corners.astype(np.int32).reshape((-1, 1, 2))
    text_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(text_mask, [poly], 255)

    # Merge SAM mask if available (better text boundary).
    if sam_mask is not None:
        sam_binary = np.zeros((fh, fw), dtype=np.uint8)
        if sam_mask.dtype == np.float32 or sam_mask.dtype == np.float64:
            sam_binary[sam_mask > 0.5] = 255
        else:
            sam_binary[sam_mask > 0] = 255
        text_mask = np.maximum(text_mask, sam_binary)

    # Dilate to ensure full text coverage.
    if dilate_px > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_px + 1, 2 * dilate_px + 1))
        text_mask = cv2.dilate(text_mask, kern, iterations=1)

    # NOTE: We erase text EVERYWHERE (including under the player).
    # Then we composite the original player on top afterwards. This
    # avoids any boundary artifacts at the shoe-court contact point.

    # Color fill — replace text with court color preserving illumination.
    # Pre-fill text with median court color before blurring to prevent
    # white text from contaminating the blur.

    # Sample court color from surrounding (non-text) region.
    surround_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * 30 + 1, 2 * 30 + 1))
    surround_mask = cv2.dilate(text_mask, surround_kern, iterations=1)
    surround_mask = surround_mask & (~text_mask.astype(bool)).astype(np.uint8) * 255

    if np.any(surround_mask > 0):
        surround_pixels = frame[surround_mask > 0]
        court_bgr = np.median(surround_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        court_bgr = np.array([158, 102, 66], dtype=np.uint8)  # fallback blue

    # Pre-fill text area with court color, then blur.
    pre_filled = frame.copy()
    pre_filled[text_mask > 0] = court_bgr
    blur_fill = cv2.GaussianBlur(pre_filled, (81, 81), 0)

    # Sample texture noise from surrounding court.
    if np.any(surround_mask > 0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        texture_residual = gray - gray_smooth
        surround_gray = texture_residual[surround_mask > 0]
        noise_std = float(np.std(surround_gray)) if len(surround_gray) > 0 else 1.0
    else:
        noise_std = 1.0

    # Add matched texture noise to the blurred fill.
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_std, size=frame.shape).astype(np.float32)
    fill = np.clip(blur_fill.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Feathered blend: smooth transition at the mask boundary.
    if feather_px > 0:
        fk = 2 * feather_px + 1
        blend_alpha = cv2.GaussianBlur(text_mask.astype(np.float32) / 255.0, (fk, fk), 0)
        blend_alpha = np.clip(blend_alpha, 0.0, 1.0)
        blend_3ch = blend_alpha[:, :, np.newaxis]
        frame[:] = (
            frame.astype(np.float32) * (1.0 - blend_3ch) + fill.astype(np.float32) * blend_3ch
        ).astype(np.uint8)
    else:
        frame[text_mask > 0] = fill[text_mask > 0]

    # Composite original player ON TOP of erased court.
    # The player's body (including shoe soles) always comes from
    # the untouched original frame, so feet can never disappear.
    if occlusion_mask is not None:
        occ = occlusion_mask.astype(np.float32)
        if occ.max() > 1:
            occ = occ / 255.0
        occ_binary = (occ > 0.5).astype(np.uint8)
        if np.any(occ_binary):
            # Moderate dilation — enough to cover mask imprecision at
            # feet but not so large that text around the player is
            # restored from the original frame.
            occ_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            occ_dilated = cv2.dilate(occ_binary, occ_kern, iterations=1)
            # Minimal feather to avoid jagged edges.
            pm_float = occ_dilated.astype(np.float32)
            pm_float = cv2.GaussianBlur(pm_float, (5, 5), 0.8)
            pm_3 = np.clip(pm_float, 0.0, 1.0)[:, :, np.newaxis]
            # Blend: erased court * (1 - player) + original * player
            frame[:] = (
                frame.astype(np.float32) * (1.0 - pm_3) + original_frame.astype(np.float32) * pm_3
            ).astype(np.uint8)


def _process_occlusion_mask(
    mask: np.ndarray,
    dilate_px: int = 10,
    feather_ksize: int = 21,
    feather_sigma: float = 4.0,
) -> np.ndarray:
    """Process person mask for smooth occlusion.

    Applies dilation (to cover imprecise Mask R-CNN foot boundaries)
    followed by Gaussian feathering for a natural transition.

    Returns float32 mask in [0, 1].
    """
    occ = mask.astype(np.float32)
    if occ.max() > 1:
        occ = occ / 255.0

    if not np.any(occ > 0.1):
        return occ

    # Binarize before morphological ops.
    occ_binary = (occ > 0.5).astype(np.uint8)

    # Dilate to cover Mask R-CNN's imprecise foot edges.
    if dilate_px > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        occ_binary = cv2.dilate(occ_binary, kern, iterations=1)

    # Gaussian feather for smooth transition at the boundary.
    occ_float = occ_binary.astype(np.float32)
    if feather_ksize > 0:
        fk = feather_ksize if feather_ksize % 2 == 1 else feather_ksize + 1
        occ_float = cv2.GaussianBlur(occ_float, (fk, fk), feather_sigma)

    return np.clip(occ_float, 0.0, 1.0)
