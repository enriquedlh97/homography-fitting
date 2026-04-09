from __future__ import annotations

import numpy as np
import pytest
import torch

from banner_pipeline.segment import sam3_video as sam3_video_mod
from banner_pipeline.segment.base import ObjectPrompt


class _FakePredictor:
    def __init__(
        self,
        stream_responses: list[dict],
        *,
        add_prompt_response: dict | None = None,
        start_error: Exception | None = None,
        propagate_error: Exception | None = None,
    ) -> None:
        self.requests: list[dict] = []
        self.stream_requests: list[dict] = []
        self.events: list[str] = []
        self._stream_responses = stream_responses
        self._add_prompt_response = add_prompt_response
        self._start_error = start_error
        self._propagate_error = propagate_error
        self._started = False
        self._prompt_count = 0

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
        yield from self._stream_responses


def _build_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    predictor: _FakePredictor,
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
        lambda _video_path, _frame_dir: ["00000.jpg", "00001.jpg"],
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
