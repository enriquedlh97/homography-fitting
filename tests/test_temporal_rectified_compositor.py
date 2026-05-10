from __future__ import annotations

import numpy as np

from banner_pipeline.composite.temporal_rectified import TemporalRectifiedCompositor


def _overlay_rgba() -> np.ndarray:
    overlay = np.zeros((12, 24, 4), dtype=np.uint8)
    overlay[:, :, 2] = 255
    overlay[:, :, 3] = 255
    return overlay


def test_temporal_wall_plate_freezes_cached_plate_after_init() -> None:
    compositor = TemporalRectifiedCompositor(
        padding=0.0,
        rectified_min_size_px=32,
        wall_freeze_after_init=True,
    )
    overlay = _overlay_rgba()
    corners = np.array([[10, 8], [69, 8], [69, 31], [10, 31]], dtype=np.float32)

    frame_a = np.zeros((40, 80, 3), dtype=np.uint8)
    frame_a[8:32, 10:70] = [60, 100, 140]
    mask_a = np.zeros((40, 80), dtype=np.uint8)
    mask_a[8:32, 10:70] = 255
    output_a = compositor.composite(
        frame_a.copy(),
        corners,
        overlay,
        mask=mask_a,
        obj_id=2,
        surface_type="back_wall_banner",
        frame_idx=0,
        geometry_fit_method="fronto_parallel_wall_banner",
        geometry_held=False,
    )

    frame_b = np.zeros((40, 80, 3), dtype=np.uint8)
    frame_b[8:32, 10:70] = [180, 180, 30]
    mask_b = np.zeros((40, 80), dtype=np.uint8)
    mask_b[9:33, 11:71] = 255
    output_b = compositor.composite(
        frame_b.copy(),
        corners,
        overlay,
        mask=mask_b,
        obj_id=2,
        surface_type="back_wall_banner",
        frame_idx=1,
        geometry_fit_method="hold_last_good",
        geometry_held=True,
    )

    np.testing.assert_array_equal(output_a[8:32, 10:70], output_b[8:32, 10:70])
    metrics = compositor.finalize_metrics()
    assert metrics["compositor_runtime_enabled"] is True
    assert metrics["compositor_object_model"] == {"2": "temporal_wall_plate"}
    assert metrics["compositor_object_stats"]["2"]["plate_reused_frames"] == 1


def test_temporal_wall_plate_resets_after_large_quad_jump() -> None:
    compositor = TemporalRectifiedCompositor(
        padding=0.0,
        rectified_min_size_px=32,
        wall_freeze_after_init=True,
    )
    overlay = _overlay_rgba()
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    frame[:, :] = [45, 60, 75]
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[12:36, 10:70] = 255

    compositor.composite(
        frame.copy(),
        np.array([[10, 12], [69, 12], [69, 35], [10, 35]], dtype=np.float32),
        overlay,
        mask=mask,
        obj_id=4,
        surface_type="back_wall_banner",
        frame_idx=0,
        geometry_fit_method="fronto_parallel_wall_banner",
        geometry_held=False,
    )
    compositor.composite(
        frame.copy(),
        np.array([[80, 20], [139, 20], [139, 43], [80, 43]], dtype=np.float32),
        overlay,
        mask=mask,
        obj_id=4,
        surface_type="back_wall_banner",
        frame_idx=1,
        geometry_fit_method="fronto_parallel_wall_banner",
        geometry_held=False,
    )

    metrics = compositor.finalize_metrics()
    assert metrics["compositor_object_stats"]["4"]["plate_reset_count"] == 1


def test_temporal_court_plate_updates_shading_field_smoothly() -> None:
    compositor = TemporalRectifiedCompositor(
        padding=0.0,
        rectified_min_size_px=32,
        court_shading_enabled=True,
        court_shading_blur_px=9,
        court_shading_alpha=0.8,
    )
    overlay = _overlay_rgba()
    corners = np.array([[12, 12], [51, 12], [51, 35], [12, 35]], dtype=np.float32)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[16:32, 18:46] = 255

    frame_a = np.zeros((48, 64, 3), dtype=np.uint8)
    frame_a[12:36, 12:52] = [35, 100, 35]
    out_a = compositor.composite(
        frame_a.copy(),
        corners,
        overlay,
        mask=mask,
        obj_id=5,
        surface_type="court_marking",
        frame_idx=0,
        geometry_fit_method="court_plane",
        geometry_held=False,
    )

    frame_b = np.zeros((48, 64, 3), dtype=np.uint8)
    frame_b[12:36, 12:52] = [55, 130, 55]
    out_b = compositor.composite(
        frame_b.copy(),
        corners,
        overlay,
        mask=mask,
        obj_id=5,
        surface_type="court_marking",
        frame_idx=1,
        geometry_fit_method="court_plane",
        geometry_held=False,
    )

    assert not np.array_equal(out_a[12:36, 12:52], out_b[12:36, 12:52])
    metrics = compositor.finalize_metrics()
    assert metrics["compositor_object_model"] == {"5": "temporal_court_plate"}
    assert metrics["compositor_object_stats"]["5"]["court_shading_updates"] == 2
    assert "compositor_rectified_obj_5" in compositor.render_debug_artifacts()
