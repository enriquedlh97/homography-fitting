from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from banner_pipeline import pipeline as pipeline_mod


class _FakeVideoSegmenter:
    def __init__(
        self,
        video_segments: dict[int, dict[int, np.ndarray]],
        frame_dir: str,
        frame_names: list[str],
    ) -> None:
        self.video_segments = video_segments
        self.frame_dir = frame_dir
        self.frame_names = frame_names

    def segment_video(
        self,
        _video_path: str,
        _prompts: list[object],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        return self.video_segments, self.frame_dir, self.frame_names


class _FakeWriter:
    instances: list[_FakeWriter] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.writes = 0
        self.closed = False
        type(self).instances.append(self)

    def write(self, _frame: np.ndarray) -> None:
        self.writes += 1

    def close(self) -> None:
        self.closed = True


class _NullFitter:
    def fit(self, _mask: np.ndarray, **_kwargs) -> None:
        return None


class _FakeCompositor:
    name = "alpha"

    def composite(
        self,
        frame: np.ndarray,
        _corners: np.ndarray,
        _overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **_kwargs,
    ) -> np.ndarray:
        return frame


def _video_config() -> dict:
    return {
        "pipeline": {
            "mode": "video",
            "segmenter": {"type": "sam3_video"},
            "fitter": {"type": "pca", "params": {}},
            "compositor": {"type": "alpha", "params": {}},
            "camera": {"focal_length": None},
        },
        "input": {
            "video": "/tmp/input.mp4",
            "logo": "/tmp/logo.png",
            "prompts": [
                {
                    "obj_id": 1,
                    "points": [[10, 10]],
                }
            ],
        },
    }


def _install_common_video_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    segmenter: _FakeVideoSegmenter,
) -> None:
    monkeypatch.setattr(pipeline_mod, "build_video_segmenter", lambda _cfg: segmenter)
    monkeypatch.setattr(pipeline_mod, "get_video_fps", lambda _path: 25.0)
    monkeypatch.setattr(
        pipeline_mod,
        "load_frame",
        lambda _path, frame_idx=0: np.zeros((24, 32, 3), dtype=np.uint8),
    )

    def _fake_imread(_path: str, flags: int | None = None) -> np.ndarray:
        if flags == pipeline_mod.cv2.IMREAD_UNCHANGED:
            return np.zeros((8, 8, 4), dtype=np.uint8)
        return np.zeros((24, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline_mod.cv2, "imread", _fake_imread)


def test_run_pipeline_video_fails_when_tracking_produces_no_masks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    segmenter = _FakeVideoSegmenter(
        video_segments={0: {}, 1: {}},
        frame_dir=str(frame_dir),
        frame_names=["00000.jpg", "00001.jpg"],
    )
    _FakeWriter.instances = []
    _install_common_video_mocks(monkeypatch, segmenter=segmenter)
    monkeypatch.setattr(
        pipeline_mod,
        "build_fitter",
        lambda _cfg: pytest.fail("build_fitter should not run when no masks were parsed"),
    )
    monkeypatch.setattr(pipeline_mod, "StreamingVideoWriter", _FakeWriter)

    with pytest.raises(RuntimeError, match="frames_with_masks=0"):
        pipeline_mod.run_pipeline_video(
            _video_config(),
            output_path=str(tmp_path / "output.mp4"),
        )

    assert _FakeWriter.instances == []


def test_run_pipeline_video_fails_when_logo_is_configured_but_nothing_is_composited(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    segmenter = _FakeVideoSegmenter(
        video_segments={
            0: {1: np.ones((4, 4), dtype=np.uint8) * 255},
            1: {1: np.ones((4, 4), dtype=np.uint8) * 255},
        },
        frame_dir=str(frame_dir),
        frame_names=["00000.jpg", "00001.jpg"],
    )
    _FakeWriter.instances = []
    _install_common_video_mocks(monkeypatch, segmenter=segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _NullFitter())
    monkeypatch.setattr(pipeline_mod, "build_compositor", lambda _cfg: _FakeCompositor())
    monkeypatch.setattr(pipeline_mod, "StreamingVideoWriter", _FakeWriter)

    with pytest.raises(RuntimeError, match="frames_composited=0"):
        pipeline_mod.run_pipeline_video(
            _video_config(),
            output_path=str(tmp_path / "output.mp4"),
        )

    assert len(_FakeWriter.instances) == 1
    assert _FakeWriter.instances[0].writes == 2
    assert _FakeWriter.instances[0].closed is True
