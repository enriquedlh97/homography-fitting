"""Unit tests for banner_pipeline.eval.walkover detection.

Uses a synthetic clean-vs-original signal to verify the threshold/longest-run
detection without requiring the real Melbourne clip on the test machine.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from banner_pipeline.eval import walkover as walkover_mod


def _write_synthetic_pair(tmp_path: Path, n_frames: int = 60, walk_start: int = 25, walk_end: int = 35) -> tuple[Path, Path]:
    """Write a 320x240 grayscale-ish original and clean video.

    Frames inside [walk_start, walk_end] inject a bright moving square inside
    the floor-quad region of the original; the clean video stays uniform.
    """
    original_path = tmp_path / "orig.mp4"
    clean_path = tmp_path / "clean.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    size = (320, 240)
    cap_o = cv2.VideoWriter(str(original_path), fourcc, fps, size)
    cap_c = cv2.VideoWriter(str(clean_path), fourcc, fps, size)
    base = np.full((240, 320, 3), 80, dtype=np.uint8)  # uniform "court"
    for i in range(n_frames):
        clean_frame = base.copy()
        orig_frame = base.copy()
        if walk_start <= i <= walk_end:
            x = 100 + (i - walk_start) * 4
            y = 100 + (i - walk_start) * 2
            cv2.rectangle(orig_frame, (x, y), (x + 40, y + 40), (240, 240, 240), -1)
        cap_o.write(orig_frame)
        cap_c.write(clean_frame)
    cap_o.release()
    cap_c.release()
    return original_path, clean_path


def test_detect_walkover_window_finds_synthetic_event(tmp_path: Path) -> None:
    original, clean = _write_synthetic_pair(tmp_path, n_frames=60, walk_start=25, walk_end=35)
    floor_quad = np.asarray(
        [[80, 80], [220, 80], [220, 180], [80, 180]], dtype=np.float32
    )
    window = walkover_mod.detect_walkover_window(
        original_video=original,
        clean_video=clean,
        floor_quad=floor_quad,
        smoothing=3,
        sigma_k=1.0,
        pad_frames=2,
    )
    assert window is not None
    assert window.method == "delta_threshold"
    # Window should cover the inserted run (with smoothing/padding slack).
    assert window.start <= 25
    assert window.end >= 35


def test_detect_walkover_window_returns_none_on_uniform_video(tmp_path: Path) -> None:
    # No walk event injected: original == clean for all frames.
    original, clean = _write_synthetic_pair(
        tmp_path, n_frames=30, walk_start=999, walk_end=999
    )
    floor_quad = np.asarray(
        [[80, 80], [220, 80], [220, 180], [80, 180]], dtype=np.float32
    )
    window = walkover_mod.detect_walkover_window(
        original_video=original, clean_video=clean, floor_quad=floor_quad
    )
    assert window is None


def test_detect_walkover_window_temporal_median_fallback(tmp_path: Path) -> None:
    """When clean_video is None, the temporal-median fallback should still detect events."""
    original, _ = _write_synthetic_pair(
        tmp_path, n_frames=60, walk_start=25, walk_end=35
    )
    floor_quad = np.asarray(
        [[80, 80], [220, 80], [220, 180], [80, 180]], dtype=np.float32
    )
    window = walkover_mod.detect_walkover_window(
        original_video=original,
        clean_video=None,
        floor_quad=floor_quad,
        smoothing=3,
        sigma_k=1.0,
        pad_frames=2,
    )
    assert window is not None
    assert window.method == "no_clean_video"
