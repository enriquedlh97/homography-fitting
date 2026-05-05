"""Smoke tests for clean-video soft player-band lettering cleanup."""

import numpy as np

from banner_pipeline.pipeline import _apply_soft_player_band_text_cleanup


def test_soft_player_band_cleanup_replaces_matching_pixels() -> None:
    shape = (30, 40, 3)
    court_bgr = (
        np.array([222, 55, 32], dtype=np.uint8)[None, None, :]
        .repeat(shape[0], axis=0)
        .repeat(shape[1], axis=1)
    )
    lettering_bgr = np.array([255, 255, 255], dtype=np.uint8)

    original_frame_bgr = court_bgr.copy()
    original_frame_bgr[10:22, 12:26] = lettering_bgr

    clean_frame_bgr = court_bgr.copy()

    composite_bgr = clean_frame_bgr.copy()
    mid_row, mid_col = 15, 18
    alpha = np.zeros((shape[0], shape[1]), dtype=np.float32)
    alpha[mid_row - 6 : mid_row + 7, mid_col - 6 : mid_col + 7] = 0.52
    for channel in range(3):
        composite_bgr[:, :, channel] = (
            (1.0 - alpha) * clean_frame_bgr[:, :, channel].astype(np.float32)
            + alpha * original_frame_bgr[:, :, channel].astype(np.float32)
        ).astype(np.uint8)

    clean_quad_mask = np.ones((shape[0], shape[1]), dtype=np.uint8) * 255
    input_cfg: dict[str, object] = {
        "clean_video_soft_band_cleanup": True,
        "clean_video_soft_band_box": [0, 0, shape[1] - 1, shape[0] - 1],
        "clean_video_soft_band_ref_width": float(shape[1]),
        "clean_video_soft_band_ref_height": float(shape[0]),
        "clean_video_soft_band_dilate_px": 0,
        "clean_video_soft_band_delta_threshold": 10.0,
        "clean_video_soft_band_survival_threshold": 0.2,
        "clean_video_soft_band_gray_min": 50,
        "clean_video_soft_band_original_gray_min": 50,
        "clean_video_soft_band_close_px": 0,
        "clean_video_soft_band_replace_alpha": 1.0,
    }

    result = _apply_soft_player_band_text_cleanup(
        frame_bgr=composite_bgr.copy(),
        original_frame_bgr=original_frame_bgr,
        clean_frame_bgr=clean_frame_bgr,
        clean_quad_mask=clean_quad_mask,
        person_mask_raw=alpha,
        input_cfg=input_cfg,
    )
    blended_cell = tuple(result[mid_row, mid_col].tolist())
    assert blended_cell != tuple(composite_bgr[mid_row, mid_col].tolist())
    delta_to_clean_after = np.abs(
        result.astype(np.float32) - clean_frame_bgr.astype(np.float32),
    ).mean()
    delta_to_clean_before = np.abs(
        composite_bgr.astype(np.float32) - clean_frame_bgr.astype(np.float32),
    ).mean()
    assert delta_to_clean_after < delta_to_clean_before


def test_soft_player_band_cleanup_skips_when_disabled() -> None:
    shape = (20, 20, 3)
    composite = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    composite_copy = composite.copy()
    cfg: dict[str, object] = {"clean_video_soft_band_cleanup": False}
    out = _apply_soft_player_band_text_cleanup(
        frame_bgr=composite,
        original_frame_bgr=np.zeros_like(composite),
        clean_frame_bgr=np.zeros_like(composite),
        clean_quad_mask=np.ones((shape[0], shape[1]), dtype=np.uint8) * 255,
        person_mask_raw=np.ones((shape[0], shape[1]), dtype=np.float32) * 0.5,
        input_cfg=cfg,
    )
    assert np.all(out == composite_copy)
