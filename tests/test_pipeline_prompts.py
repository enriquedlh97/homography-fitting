from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from banner_pipeline import pipeline as pipeline_mod
from banner_pipeline.segment.base import ObjectPrompt


class _FakePreviewSegmenter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ObjectPrompt]]] = []

    def preview_frame(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[np.ndarray, dict[int, np.ndarray], int, dict[int, dict[str, object]]]:
        self.calls.append((video_path, prompts))
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        masks = {1: np.ones((24, 32), dtype=np.uint8) * 255}
        diagnostics = {
            1: {
                "obj_id": 1,
                "frame_idx": 5,
                "usable_outputs": True,
                "parsed_nonempty_masks": 1,
                "mask_area_px": int(masks[1].astype(bool).sum()),
                "mask_bbox": [0, 0, 31, 23],
            }
        }
        return frame, masks, 5, diagnostics


class _PreviewFitter:
    def fit(self, _mask: np.ndarray, **_kwargs) -> np.ndarray:
        return np.array([[1, 1], [14, 1], [14, 10], [1, 10]], dtype=np.float32)


class _PreviewCompositor:
    name = "inpaint"

    def composite(
        self,
        frame: np.ndarray,
        _corners: np.ndarray,
        _overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **_kwargs,
    ) -> np.ndarray:
        preview = frame.copy()
        preview[0, 0] = [255, 255, 255]
        return preview


class _WarnAlphaPreviewCompositor:
    name = "alpha"

    def composite(
        self,
        frame: np.ndarray,
        _corners: np.ndarray,
        _overlay: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        debug_info = kwargs["debug_info"]
        debug_info.update(
            {
                "fill_color_bgr": (8, 8, 8),
                "fill_spread_bgr": (32.0, 4.0, 4.0),
                "fill_band_px": 4,
                "fill_unstable": True,
                "fill_warning_reason": "background_fill_unstable",
            }
        )
        return frame


class _TinyMaskPreviewSegmenter(_FakePreviewSegmenter):
    def preview_frame(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[np.ndarray, dict[int, np.ndarray], int, dict[int, dict[str, object]]]:
        self.calls.append((video_path, prompts))
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[0, 0] = 255
        return (
            frame,
            {1: mask},
            5,
            {
                1: {
                    "obj_id": 1,
                    "frame_idx": 5,
                    "usable_outputs": True,
                    "parsed_nonempty_masks": 1,
                    "mask_area_px": 1,
                    "mask_bbox": [0, 0, 0, 0],
                }
            },
        )


class _CompactSmallMaskPreviewSegmenter(_FakePreviewSegmenter):
    def preview_frame(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[np.ndarray, dict[int, np.ndarray], int, dict[int, dict[str, object]]]:
        self.calls.append((video_path, prompts))
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[4:12, 4:12] = np.array(
            [
                [255, 255, 255, 255, 255, 255, 255, 255],
                [255, 0, 0, 0, 0, 0, 0, 255],
                [255, 0, 255, 255, 255, 255, 0, 255],
                [255, 0, 255, 0, 0, 255, 0, 255],
                [255, 0, 255, 0, 0, 255, 0, 255],
                [255, 0, 255, 255, 255, 255, 0, 255],
                [255, 0, 0, 0, 0, 0, 0, 255],
                [255, 255, 255, 255, 255, 255, 255, 255],
            ],
            dtype=np.uint8,
        )
        return (
            frame,
            {1: mask},
            5,
            {
                1: {
                    "obj_id": 1,
                    "frame_idx": 5,
                    "usable_outputs": True,
                    "parsed_nonempty_masks": 1,
                    "mask_area_px": int(mask.astype(bool).sum()),
                    "mask_bbox": [4, 4, 11, 11],
                }
            },
        )


class _ExhaustedRetryPreviewSegmenter(_FakePreviewSegmenter):
    def preview_frame(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[np.ndarray, dict[int, np.ndarray], int, dict[int, dict[str, object]]]:
        self.calls.append((video_path, prompts))
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        return (
            frame,
            {},
            5,
            {
                1: {
                    "obj_id": 1,
                    "frame_idx": 5,
                    "usable_outputs": False,
                    "parsed_nonempty_masks": 0,
                    "mask_area_px": 0,
                    "mask_bbox": None,
                    "seed_retry_attempted": True,
                    "seed_retry_succeeded": False,
                    "seed_retry_exhausted": True,
                }
            },
        )


def test_prompt_config_round_trip_preserves_labels_and_frame_idx(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = {"input": {"video": "data/tennis-clip.mp4"}}
    prompts = [
        ObjectPrompt(
            obj_id=3,
            points=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            labels=np.array([1, 0], dtype=np.int32),
            frame_idx=7,
        )
    ]

    pipeline_mod._save_prompts_to_config(config, prompts, str(config_path))

    saved = yaml.safe_load(config_path.read_text())
    assert saved["input"]["prompts"] == [
        {
            "obj_id": 3,
            "points": [[10.0, 20.0], [30.0, 40.0]],
            "labels": [1, 0],
            "frame_idx": 7,
        }
    ]

    round_trip = pipeline_mod._prompts_from_config(saved["input"]["prompts"])
    assert len(round_trip) == 1
    np.testing.assert_allclose(round_trip[0].points, prompts[0].points)
    np.testing.assert_array_equal(round_trip[0].labels, prompts[0].labels)
    assert round_trip[0].frame_idx == 7


def test_load_prompts_warns_for_legacy_sam3_outline_configs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "input": {
            "prompts": [
                {
                    "obj_id": 1,
                    "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                }
            ]
        }
    }

    prompts = pipeline_mod._load_or_collect_prompts(
        config=config,
        config_path=None,
        video_path="/tmp/input.mp4",
        segmenter_type="sam3_video",
        log_prefix="[test]",
    )

    captured = capsys.readouterr()
    assert len(prompts) == 1
    assert "legacy SAM2 outline prompt set" in captured.out


def test_run_pipeline_uses_sam3_preview_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "pipeline": {
            "mode": "image",
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
                    "points": [[10, 10], [14, 10], [10, 4]],
                    "labels": [1, 1, 0],
                    "frame_idx": 5,
                }
            ],
        },
    }
    preview_segmenter = _FakePreviewSegmenter()
    monkeypatch.setattr(pipeline_mod, "build_video_segmenter", lambda _cfg: preview_segmenter)
    monkeypatch.setattr(
        pipeline_mod,
        "build_segmenter",
        lambda _cfg: pytest.fail("run_pipeline should not use the single-frame segmenter for SAM3"),
    )
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _PreviewFitter())
    monkeypatch.setattr(pipeline_mod, "build_compositor", lambda _cfg: _PreviewCompositor())
    monkeypatch.setattr(
        pipeline_mod.cv2,
        "imread",
        lambda _path, flags=None: (
            np.zeros((8, 8, 4), dtype=np.uint8)
            if flags == pipeline_mod.cv2.IMREAD_UNCHANGED
            else np.zeros((24, 32, 3), dtype=np.uint8)
        ),
    )

    results = pipeline_mod.run_pipeline(config)

    assert len(preview_segmenter.calls) == 1
    assert preview_segmenter.calls[0][0] == "/tmp/input.mp4"
    assert len(preview_segmenter.calls[0][1]) == 1
    assert results["frame"].shape == (24, 32, 3)
    assert set(results["masks"]) == {1}
    assert set(results["corners_map"]) == {1}
    assert results["composited"] is not None
    assert set(results["preview_artifacts"]) == {"preview_prompts", "preview_masks", "composited"}
    assert results["metrics"]["preview_frame_idx"] == 5
    assert results["metrics"]["preview_objects_with_masks"] == 1
    assert results["metrics"]["preview_objects_with_quads"] == 1
    assert results["metrics"]["preview_ok"] is True


def test_run_pipeline_marks_degenerate_sam3_preview_masks_as_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "pipeline": {
            "mode": "image",
            "segmenter": {"type": "sam3_video"},
            "fitter": {"type": "pca", "params": {}},
            "compositor": {"type": "inpaint", "params": {}},
            "camera": {"focal_length": None},
        },
        "input": {
            "video": "/tmp/input.mp4",
            "prompts": [
                {
                    "obj_id": 1,
                    "points": [[10, 10], [14, 10], [10, 4]],
                    "labels": [1, 1, 0],
                    "frame_idx": 5,
                }
            ],
        },
    }
    preview_segmenter = _TinyMaskPreviewSegmenter()
    monkeypatch.setattr(pipeline_mod, "build_video_segmenter", lambda _cfg: preview_segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _PreviewFitter())

    results = pipeline_mod.run_pipeline(config)

    assert results["metrics"]["preview_ok"] is False
    assert results["metrics"]["preview_objects_with_quads"] == 0
    assert "fit failure (mask_area_too_small)" in results["metrics"]["preview_failure_reasons"][0]
    assert results["metrics"]["preview_object_diagnostics"]["1"]["fit_method"] == "not_run"


def test_run_pipeline_uses_min_area_rect_fallback_for_small_compact_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "pipeline": {
            "mode": "image",
            "segmenter": {"type": "sam3_video"},
            "fitter": {"type": "pca", "params": {}},
            "compositor": {"type": "inpaint", "params": {}},
            "camera": {"focal_length": None},
        },
        "input": {
            "video": "/tmp/input.mp4",
            "prompts": [
                {
                    "obj_id": 1,
                    "points": [[10, 10], [14, 10], [10, 4]],
                    "labels": [1, 1, 0],
                    "frame_idx": 5,
                }
            ],
        },
    }
    preview_segmenter = _CompactSmallMaskPreviewSegmenter()
    monkeypatch.setattr(pipeline_mod, "build_video_segmenter", lambda _cfg: preview_segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _PreviewFitter())

    results = pipeline_mod.run_pipeline(config)

    assert results["metrics"]["preview_ok"] is True
    assert results["metrics"]["preview_objects_with_quads"] == 1
    assert (
        results["metrics"]["preview_object_diagnostics"]["1"]["fit_method"]
        == "min_area_rect_fallback"
    )


def test_run_pipeline_reports_compositor_stage_failures_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "pipeline": {
            "mode": "image",
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
                    "points": [[10, 10], [14, 10], [10, 4]],
                    "labels": [1, 1, 0],
                    "frame_idx": 5,
                }
            ],
        },
    }
    preview_segmenter = _FakePreviewSegmenter()
    monkeypatch.setattr(pipeline_mod, "build_video_segmenter", lambda _cfg: preview_segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _PreviewFitter())
    monkeypatch.setattr(
        pipeline_mod,
        "build_compositor",
        lambda _cfg: _WarnAlphaPreviewCompositor(),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "compute_oriented_homography",
        lambda corners, _K: {
            "dst_w": 16,
            "dst_h": 8,
            "dst_rect": np.array([[0, 0], [16, 0], [16, 8], [0, 8]], dtype=np.float32),
            "H": np.eye(3, dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        pipeline_mod.cv2,
        "imread",
        lambda _path, flags=None: (
            np.zeros((8, 8, 4), dtype=np.uint8)
            if flags == pipeline_mod.cv2.IMREAD_UNCHANGED
            else np.zeros((24, 32, 3), dtype=np.uint8)
        ),
    )

    results = pipeline_mod.run_pipeline(config)

    diag = results["metrics"]["preview_object_diagnostics"]["1"]
    assert results["metrics"]["preview_ok"] is False
    assert (
        "composite failure (background_fill_unstable)"
        in results["metrics"]["preview_failure_reasons"][0]
    )
    assert diag["fit_status"] == "ok"
    assert diag["composite_status"] == "warning"
    assert diag["background_fill_color_bgr"] == (8, 8, 8)


def test_run_pipeline_reports_seed_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "pipeline": {
            "mode": "image",
            "segmenter": {"type": "sam3_video"},
            "fitter": {"type": "pca", "params": {}},
            "compositor": {"type": "inpaint", "params": {}},
            "camera": {"focal_length": None},
        },
        "input": {
            "video": "/tmp/input.mp4",
            "prompts": [
                {
                    "obj_id": 1,
                    "points": [[10, 10], [14, 10], [10, 4]],
                    "labels": [1, 1, 0],
                    "frame_idx": 5,
                }
            ],
        },
    }
    preview_segmenter = _ExhaustedRetryPreviewSegmenter()
    monkeypatch.setattr(pipeline_mod, "build_video_segmenter", lambda _cfg: preview_segmenter)
    monkeypatch.setattr(pipeline_mod, "build_fitter", lambda _cfg: _PreviewFitter())

    results = pipeline_mod.run_pipeline(config)

    assert results["metrics"]["preview_ok"] is False
    assert "mask failure (seed_retry_exhausted)" in results["metrics"]["preview_failure_reasons"][0]
