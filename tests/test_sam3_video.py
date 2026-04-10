from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from banner_pipeline.segment import sam3_video as sam3_video_mod
from banner_pipeline.segment.base import ObjectPrompt


class _FakePredictor:
    def __init__(
        self,
        stream_responses: list[dict] | list[list[dict]],
        *,
        add_prompt_response: dict | None = None,
        add_prompt_responses: list[dict] | None = None,
        start_error: Exception | None = None,
        propagate_error: Exception | None = None,
    ) -> None:
        self.requests: list[dict] = []
        self.stream_requests: list[dict] = []
        self.events: list[str] = []
        if stream_responses and isinstance(stream_responses[0], list):
            self._stream_response_batches = list(stream_responses)
        else:
            self._stream_response_batches = [cast(list[dict], stream_responses)]
        self._add_prompt_response = add_prompt_response
        self._add_prompt_responses = list(add_prompt_responses or [])
        self._start_error = start_error
        self._propagate_error = propagate_error
        self._started = False
        self._prompt_count = 0
        self._stream_call_count = 0

    def handle_request(self, request: dict) -> dict:
        self.requests.append(request)
        self.events.append(request["type"])
        if request["type"] == "start_session":
            if self._start_error is not None:
                raise self._start_error
            self._started = True
            return {"session_id": "session-1"}
        if request["type"] == "add_prompt":
            assert self._started, "SAM3 add_prompt must happen after start_session"
            self._prompt_count += 1
            if self._add_prompt_responses:
                return self._add_prompt_responses.pop(0)
            if self._add_prompt_response is not None:
                return self._add_prompt_response
            return {
                "outputs": {
                    "obj_ids": torch.tensor([request["obj_id"]]),
                    "video_res_masks": torch.tensor([[[1.0]]]),
                }
            }
        if request["type"] == "close_session":
            return {"outputs": {}}
        raise AssertionError(f"Unexpected SAM3 handle_request payload: {request}")

    def handle_stream_request(self, request: dict):
        self.stream_requests.append(request)
        self.events.append(request["type"])
        assert self._started, "SAM3 propagation must happen after start_session"
        assert self._prompt_count > 0, "SAM3 propagation requires at least one prompt"
        if self._propagate_error is not None:
            raise self._propagate_error
        if self._stream_call_count >= len(self._stream_response_batches):
            return
        batch = self._stream_response_batches[self._stream_call_count]
        self._stream_call_count += 1
        yield from batch


def _build_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    predictor: _FakePredictor,
    *,
    frame_names: list[str] | None = None,
) -> sam3_video_mod.SAM3VideoSegmenter:
    monkeypatch.setattr(sam3_video_mod, "detect_device", lambda _device="auto": "cpu")
    monkeypatch.setattr(
        sam3_video_mod,
        "load_sam3_video_predictor",
        lambda checkpoint, device: predictor,
    )
    monkeypatch.setattr(
        sam3_video_mod,
        "extract_all_frames",
        lambda _video_path, _frame_dir: frame_names or ["00000.jpg", "00001.jpg"],
    )
    monkeypatch.setattr(
        sam3_video_mod.cv2,
        "imread",
        lambda _path: np.zeros((100, 200, 3), dtype=np.uint8),
    )
    return sam3_video_mod.SAM3VideoSegmenter(checkpoint="/tmp/sam3.1_multiplex.pt")


def test_sam3_video_segmenter_uses_point_labels_and_streamed_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [
            {
                "frame_index": 0,
                "outputs": {
                    "obj_ids": torch.tensor([1]),
                    "video_res_masks": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                },
            },
            {
                "frame_index": 1,
                "outputs": {
                    "masks": {
                        "1": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                    }
                },
            },
        ]
    )
    segmenter = _build_segmenter(monkeypatch, predictor)

    prompt = ObjectPrompt(
        obj_id=1,
        points=np.array([[20.0, 25.0], [100.0, 50.0]], dtype=np.float32),
        labels=np.array([1, 0], dtype=np.int32),
    )
    video_segments, _, frame_names = segmenter.segment_video("/tmp/video.mp4", [prompt])

    assert frame_names == ["00000.jpg", "00001.jpg"]
    assert predictor.events == [
        "start_session",
        "add_prompt",
        "propagate_in_video",
        "close_session",
    ]
    add_prompt_request = predictor.requests[1]
    assert add_prompt_request["type"] == "add_prompt"
    assert "labels" not in add_prompt_request
    assert isinstance(add_prompt_request["points"], torch.Tensor)
    assert isinstance(add_prompt_request["point_labels"], torch.Tensor)
    assert add_prompt_request["points"].dtype == torch.float32
    assert add_prompt_request["point_labels"].dtype == torch.int32
    assert torch.allclose(
        add_prompt_request["points"],
        torch.tensor([[0.1, 0.25], [0.5, 0.5]], dtype=torch.float32),
    )
    assert torch.equal(
        add_prompt_request["point_labels"],
        torch.tensor([1, 0], dtype=torch.int32),
    )

    assert predictor.stream_requests == [
        {
            "type": "propagate_in_video",
            "session_id": "session-1",
            "start_frame_index": 0,
        }
    ]
    assert predictor.requests[-1] == {
        "type": "close_session",
        "session_id": "session-1",
    }

    assert sorted(video_segments) == [0, 1]
    assert set(video_segments[0]) == {1}
    assert set(video_segments[1]) == {1}
    assert video_segments[0][1].shape == (100, 200)
    assert video_segments[1][1].shape == (100, 200)
    assert set(np.unique(video_segments[0][1])).issubset({0, 255})
    assert set(np.unique(video_segments[1][1])).issubset({0, 255})


def test_sam3_video_segmenter_accepts_official_streamed_output_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictor = _FakePredictor(
        [
            {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": torch.tensor([7]),
                    "out_binary_masks": torch.tensor([[[True, False], [False, True]]]),
                },
            }
        ]
    )
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=7,
        points=np.array([[20.0, 25.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    video_segments, _, _ = segmenter.segment_video("/tmp/video.mp4", [prompt])

    assert set(video_segments[0]) == {7}
    assert video_segments[0][7].shape == (100, 200)
    assert set(np.unique(video_segments[0][7])).issubset({0, 255})

    captured = capsys.readouterr()
    assert "First propagated response" in captured.out
    assert "parsed_nonempty_masks=1" in captured.out


def test_sam3_video_segmenter_previews_prompt_stage_masks_without_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [],
        add_prompt_responses=[
            {
                "outputs": {
                    "out_obj_ids": torch.tensor([4]),
                    "out_binary_masks": torch.tensor([[[True, False], [False, True]]]),
                }
            }
        ],
    )
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=4,
        points=np.array([[20.0, 25.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        frame_idx=1,
    )

    frame, masks, frame_idx, prompt_diagnostics = segmenter.preview_frame(
        "/tmp/video.mp4",
        [prompt],
    )

    assert frame.shape == (100, 200, 3)
    assert frame_idx == 1
    assert predictor.events == ["start_session", "add_prompt", "close_session"]
    assert predictor.stream_requests == []
    assert set(masks) == {4}
    assert masks[4].shape == (100, 200)
    mask_area_px = prompt_diagnostics[4]["mask_area_px"]
    assert isinstance(mask_area_px, int)
    assert mask_area_px > 0


def test_sam3_video_segmenter_retries_empty_preview_seed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [],
        add_prompt_responses=[
            {"outputs": {}},
            {
                "outputs": {
                    "out_obj_ids": torch.tensor([4]),
                    "out_binary_masks": torch.tensor([[[True, False], [False, True]]]),
                }
            },
        ],
    )
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=4,
        points=np.array([[20.0, 25.0], [24.0, 25.0], [22.0, 30.0]], dtype=np.float32),
        labels=np.array([1, 1, 0], dtype=np.int32),
        frame_idx=1,
    )

    _frame, masks, _frame_idx, prompt_diagnostics = segmenter.preview_frame(
        "/tmp/video.mp4",
        [prompt],
    )

    assert predictor.events == ["start_session", "add_prompt", "add_prompt", "close_session"]
    assert set(masks) == {4}
    assert prompt_diagnostics[4]["seed_retry_attempted"] is True
    assert prompt_diagnostics[4]["seed_retry_succeeded"] is True
    assert prompt_diagnostics[4]["seed_retry_exhausted"] is False


def test_sam3_video_segmenter_marks_seed_retry_exhaustion_in_preview_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [],
        add_prompt_responses=[
            {"outputs": {}},
            {"outputs": {}},
        ],
    )
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=4,
        points=np.array([[20.0, 25.0], [24.0, 25.0], [22.0, 30.0]], dtype=np.float32),
        labels=np.array([1, 1, 0], dtype=np.int32),
        frame_idx=1,
    )

    _frame, masks, _frame_idx, prompt_diagnostics = segmenter.preview_frame(
        "/tmp/video.mp4",
        [prompt],
    )

    assert masks == {}
    assert prompt_diagnostics[4]["seed_retry_attempted"] is True
    assert prompt_diagnostics[4]["seed_retry_succeeded"] is False
    assert prompt_diagnostics[4]["seed_retry_exhausted"] is True


def test_sam3_video_segmenter_preview_rejects_mixed_prompt_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor([])
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompts = [
        ObjectPrompt(
            obj_id=1,
            points=np.array([[20.0, 25.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            frame_idx=0,
        ),
        ObjectPrompt(
            obj_id=2,
            points=np.array([[30.0, 35.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            frame_idx=3,
        ),
    ]

    with pytest.raises(RuntimeError, match="single frame"):
        segmenter.preview_frame("/tmp/video.mp4", prompts)

    assert predictor.events == []


def test_sam3_video_segmenter_wraps_start_session_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor([], start_error=RuntimeError("start failed"))
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=1,
        points=np.array([[20.0, 25.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="session initialization"):
        segmenter.segment_video("/tmp/video.mp4", [prompt])


def test_sam3_video_segmenter_fails_early_when_all_prompt_outputs_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [],
        add_prompt_response={"outputs": {}},
    )
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=1,
        points=np.array([[20.0, 25.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="produced no usable outputs"):
        segmenter.segment_video("/tmp/video.mp4", [prompt])


def test_sam3_video_segmenter_wraps_known_propagation_prompt_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [],
        propagate_error=RuntimeError(
            "No prompts are received on any frames. "
            "Please add prompt on at least one frame before propagation."
        ),
    )
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=1,
        points=np.array([[20.0, 25.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="did not treat those point prompts as propagation"):
        segmenter.segment_video("/tmp/video.mp4", [prompt])


def test_sam3_video_segmenter_forwards_earliest_prompt_frame_to_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [
            {
                "frame_index": 2,
                "outputs": {
                    "obj_ids": torch.tensor([1]),
                    "video_res_masks": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                },
            }
        ]
    )
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompts = [
        ObjectPrompt(
            obj_id=1,
            points=np.array([[20.0, 25.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            frame_idx=5,
        ),
        ObjectPrompt(
            obj_id=2,
            points=np.array([[30.0, 35.0]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            frame_idx=2,
        ),
    ]

    segmenter.segment_video("/tmp/video.mp4", prompts)

    assert predictor.stream_requests == [
        {
            "type": "propagate_in_video",
            "session_id": "session-1",
            "start_frame_index": 2,
        }
    ]


def test_sam3_video_segmenter_rejects_box_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor([])
    segmenter = _build_segmenter(monkeypatch, predictor)
    prompt = ObjectPrompt(
        obj_id=1,
        points=np.array([[10.0, 10.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        box=np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="point prompts only"):
        segmenter.segment_video("/tmp/video.mp4", [prompt])


def test_sam3_video_segmenter_reanchors_after_tracking_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _FakePredictor(
        [
            [
                {
                    "frame_index": 0,
                    "outputs": {
                        "obj_ids": torch.tensor([1]),
                        "video_res_masks": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                    },
                },
                {
                    "frame_index": 1,
                    "outputs": {
                        "obj_ids": torch.tensor([1]),
                        "video_res_masks": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                    },
                },
                {
                    "frame_index": 2,
                    "outputs": {
                        "obj_ids": torch.tensor([]),
                        "video_res_masks": torch.empty((0, 2, 2)),
                    },
                },
                {
                    "frame_index": 3,
                    "outputs": {
                        "obj_ids": torch.tensor([]),
                        "video_res_masks": torch.empty((0, 2, 2)),
                    },
                },
            ],
            [
                {
                    "frame_index": 3,
                    "outputs": {
                        "obj_ids": torch.tensor([1]),
                        "video_res_masks": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                    },
                }
            ],
        ]
    )
    segmenter = _build_segmenter(
        monkeypatch,
        predictor,
        frame_names=["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"],
    )
    prompt = ObjectPrompt(
        obj_id=1,
        points=np.array([[20.0, 25.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    video_segments, _, _ = segmenter.segment_video("/tmp/video.mp4", [prompt])

    assert predictor.events == [
        "start_session",
        "add_prompt",
        "propagate_in_video",
        "add_prompt",
        "propagate_in_video",
        "close_session",
    ]
    refresh_request = predictor.requests[2]
    assert refresh_request["type"] == "add_prompt"
    assert torch.equal(
        refresh_request["point_labels"],
        torch.tensor([1, 1, 0, 0], dtype=torch.int32),
    )
    assert video_segments[3][1].shape == (100, 200)
    assert segmenter.last_tracking_stats["sam3_reanchor_events"] == [
        {"obj_id": 1, "frame_idx": 3, "refresh_count": 1}
    ]


def test_parse_frame_outputs_accepts_dict_of_masks() -> None:
    outputs = {
        "video_res_masks": {
            "3": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        }
    }

    parsed = sam3_video_mod._parse_frame_outputs(outputs, frame_shape=(2, 2))

    assert set(parsed) == {3}
    assert parsed[3].dtype == np.uint8
    assert parsed[3].tolist() == [[255, 0], [0, 255]]


def test_parse_frame_outputs_accepts_official_sam31_output_format() -> None:
    outputs = {
        "out_obj_ids": torch.tensor([5]),
        "out_binary_masks": torch.tensor([[[True, False], [False, True]]]),
    }

    parsed = sam3_video_mod._parse_frame_outputs(outputs, frame_shape=(2, 2))

    assert set(parsed) == {5}
    assert parsed[5].dtype == np.uint8
    assert parsed[5].tolist() == [[255, 0], [0, 255]]
