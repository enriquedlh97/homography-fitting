from __future__ import annotations

import numpy as np

from banner_pipeline.composite.alpha import AlphaCompositor, estimate_rectified_border_fill


def test_estimate_rectified_border_fill_ignores_bright_center() -> None:
    rectified = np.zeros((40, 100, 3), dtype=np.uint8)
    rectified[10:30, 30:70] = [0, 180, 255]

    fill_info = estimate_rectified_border_fill(rectified)

    assert fill_info["fill_color_bgr"] == (0, 0, 0)
    assert fill_info["fill_unstable"] is False


def test_alpha_compositor_uses_border_fill_for_banner_background() -> None:
    frame = np.zeros((40, 100, 3), dtype=np.uint8)
    frame[10:30, 30:70] = [0, 180, 255]
    overlay = np.zeros((12, 24, 4), dtype=np.uint8)
    overlay[:, :, 1] = 255
    overlay[:, :, 3] = 255
    corners = np.array([[0, 0], [99, 0], [99, 39], [0, 39]], dtype=np.float32)
    homo = {
        "dst_w": 100,
        "dst_h": 40,
        "dst_rect": np.array([[0, 0], [99, 0], [99, 39], [0, 39]], dtype=np.float32),
        "H": np.eye(3, dtype=np.float32),
    }
    debug_info: dict[str, object] = {}

    composited = AlphaCompositor().composite(
        frame.copy(),
        corners,
        overlay,
        homo=homo,
        debug_info=debug_info,
    )

    assert composited[0, 0].max() < 16
    assert debug_info["fill_color_bgr"] == (0, 0, 0)
