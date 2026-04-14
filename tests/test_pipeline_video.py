from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from banner_pipeline import pipeline as pipeline_mod


def _banner_mask() -> np.ndarray:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    return mask


class _FakeVideoSegmenter:
    def __init__(
        self,
        video_segments: dict[int, dict[int, np.ndarray]],
        frame_dir: str,
        frame_names: list[str],
        *,
        last_tracking_stats: dict | None = None,
    ) -> None:
        self.video_segments = video_segments
        self.frame_dir = frame_dir
        self.frame_names = frame_names
        self.last_tracking_stats = last_tracking_stats or {}

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
    name = "inpaint"

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
            "compositor": {"type": "inpaint", "params": {}},
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


class _QuadFitter:
    def fit(self, _mask: np.ndarray, **_kwargs) -> np.ndarray:
        return np.array([[4, 4], [11, 4], [11, 11], [4, 11]], dtype=np.float32)


class _BadQuadFitter:
    def fit(self, _mask: np.ndarray, **_kwargs) -> np.ndarray:
        return np.array([[0, 0], [80, 0], [80, 8], [0, 8]], dtype=np.float32)


class _RecordingFitter:
    def __init__(self) -> None:
        self.masks: list[np.ndarray] = []

    def fit(self, mask: np.ndarray, **_kwargs) -> np.ndarray:
        self.masks.append(mask.copy())
        return np.array([[4, 4], [11, 4], [11, 11], [4, 11]], dtype=np.float32)


def test_run_pipeline_video_records_coverage_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    segmenter = _FakeVideoSegmenter(
        video_segments={
            0: {1: _banner_mask()},
            1: {},
            2: {1: _banner_mask()},
        },
        frame_dir=str(frame_dir),
        frame_names=["00000.jpg", "00001.jpg", "00002.jpg"],
        last_tracking_stats={
            "sam3_reanchor_events": [{"obj_id": 1, "frame_idx": 2, "refresh_count": 1}]
        },
    )
    _FakeWriter.instances = []
    _install_common_video_mocks(monkeypatch, segmenter=segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _QuadFitter())
    monkeypatch.setattr(pipeline_mod, "build_compositor", lambda _cfg: _FakeCompositor())
    monkeypatch.setattr(pipeline_mod, "StreamingVideoWriter", _FakeWriter)

    results = pipeline_mod.run_pipeline_video(
        _video_config(),
        output_path=str(tmp_path / "output.mp4"),
    )

    metrics = results["metrics"]
    assert metrics["frames_with_masks"] == 2
    assert metrics["frames_with_valid_objects"] == 2
    assert metrics["object_masks_total"] == 2
    assert metrics["first_frame_with_mask"] == 0
    assert metrics["last_frame_with_mask"] == 2
    assert metrics["max_consecutive_mask_gap"] == 1
    assert metrics["object_frame_coverage"] == {
        "1": {"frames_with_masks": 2, "coverage_ratio": 0.6667}
    }
    assert metrics["object_valid_frame_coverage"] == {
        "1": {"frames_valid": 2, "coverage_ratio": 0.6667}
    }
    assert metrics["object_rejection_counts"] == {"1": 1}
    assert metrics["object_rejection_reasons"] == {"1": {"empty_mask": 1}}
    assert metrics["sam3_reanchor_events"] == [{"obj_id": 1, "frame_idx": 2, "refresh_count": 1}]


def test_run_pipeline_video_rejects_persistent_bad_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    segmenter = _FakeVideoSegmenter(
        video_segments={
            0: {1: _banner_mask()},
            1: {1: _banner_mask()},
        },
        frame_dir=str(frame_dir),
        frame_names=["00000.jpg", "00001.jpg"],
    )
    _FakeWriter.instances = []
    _install_common_video_mocks(monkeypatch, segmenter=segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _BadQuadFitter())
    monkeypatch.setattr(pipeline_mod, "build_compositor", lambda _cfg: _FakeCompositor())
    monkeypatch.setattr(pipeline_mod, "StreamingVideoWriter", _FakeWriter)

    with pytest.raises(RuntimeError, match="frames_with_valid_objects=0"):
        pipeline_mod.run_pipeline_video(
            _video_config(),
            output_path=str(tmp_path / "output.mp4"),
        )


def test_run_pipeline_video_skips_court_marking_prompts_but_keeps_banner_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    segmenter = _FakeVideoSegmenter(
        video_segments={
            0: {1: np.ones((24, 32), dtype=np.uint8) * 255},
            1: {1: np.ones((24, 32), dtype=np.uint8) * 255},
        },
        frame_dir=str(frame_dir),
        frame_names=["00000.jpg", "00001.jpg"],
    )
    _FakeWriter.instances = []
    _install_common_video_mocks(monkeypatch, segmenter=segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _QuadFitter())
    monkeypatch.setattr(pipeline_mod, "build_compositor", lambda _cfg: _FakeCompositor())
    monkeypatch.setattr(pipeline_mod, "StreamingVideoWriter", _FakeWriter)

    config = _video_config()
    config["input"]["prompts"] = [
        {"obj_id": 1, "points": [[10, 10]], "labels": [1]},
        {
            "obj_id": 2,
            "points": [[20, 20], [24, 20], [20, 16]],
            "labels": [1, 1, 0],
            "surface_type": "court_marking",
        },
    ]

    results = pipeline_mod.run_pipeline_video(
        config,
        output_path=str(tmp_path / "output.mp4"),
    )

    metrics = results["metrics"]
    assert metrics["frames_with_valid_objects"] == 2
    assert metrics["object_frame_coverage"]["2"] == {"frames_with_masks": 0, "coverage_ratio": 0.0}
    assert metrics["object_valid_frame_coverage"]["1"] == {
        "frames_valid": 2,
        "coverage_ratio": 1.0,
    }
    assert metrics["object_valid_frame_coverage"]["2"] == {
        "frames_valid": 0,
        "coverage_ratio": 0.0,
    }
    assert metrics["object_rejection_counts"]["2"] == 2
    assert metrics["object_rejection_reasons"]["2"] == {"unsupported_surface_type:court_marking": 2}


def test_run_pipeline_video_uses_stabilized_masks_before_fitting_and_records_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    raw_mask = _banner_mask()
    jittered_mask = np.zeros_like(raw_mask)
    jittered_mask[4:12, 5:13] = 255
    stabilized_mask = raw_mask.copy()
    segmenter = _FakeVideoSegmenter(
        video_segments={
            0: {1: raw_mask},
            1: {1: jittered_mask},
        },
        frame_dir=str(frame_dir),
        frame_names=["00000.jpg", "00001.jpg"],
    )
    _FakeWriter.instances = []
    _install_common_video_mocks(monkeypatch, segmenter=segmenter)
    recording_fitter = _RecordingFitter()
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: recording_fitter)
    monkeypatch.setattr(pipeline_mod, "build_compositor", lambda _cfg: _FakeCompositor())
    monkeypatch.setattr(pipeline_mod, "StreamingVideoWriter", _FakeWriter)

    stabilize_calls: list[dict[str, object]] = []

    def _fake_stabilize_video_segments(**kwargs):
        stabilize_calls.append(kwargs)
        return (
            {
                0: {1: stabilized_mask},
                1: {1: stabilized_mask},
            },
            {
                "stabilization_total_s": 0.1234,
                "stabilization_static_frame_ratio": 1.0,
                "stabilization_frames_held": 1,
                "stabilization_frames_blended": 0,
                "stabilization_frames_raw_accepted": 1,
                "stabilization_object_stats": {
                    "1": {
                        "frames_held": 1,
                        "frames_empty_reused": 0,
                        "frames_blended": 0,
                        "frames_raw_accepted": 1,
                        "frames_dropped": 0,
                        "max_empty_hold_streak": 0,
                    }
                },
            },
        )

    monkeypatch.setattr(
        pipeline_mod.stabilization_mod,
        "stabilize_video_segments",
        _fake_stabilize_video_segments,
    )

    config = _video_config()
    config["pipeline"]["stabilization"] = {
        "enabled": True,
        "mode": "hybrid",
    }

    results = pipeline_mod.run_pipeline_video(
        config,
        output_path=str(tmp_path / "output.mp4"),
    )

    assert len(stabilize_calls) == 1
    assert stabilize_calls[0]["tracked_obj_ids"] == [1]
    assert len(recording_fitter.masks) == 2
    assert np.array_equal(recording_fitter.masks[0], stabilized_mask)
    assert np.array_equal(recording_fitter.masks[1], stabilized_mask)

    metrics = results["metrics"]
    assert metrics["stabilization_total_s"] == 0.1234
    assert metrics["stabilization_static_frame_ratio"] == 1.0
    assert metrics["stabilization_frames_held"] == 1
    assert metrics["stabilization_frames_blended"] == 0
    assert metrics["stabilization_frames_raw_accepted"] == 1
    assert metrics["stabilization_object_stats"]["1"]["frames_held"] == 1
