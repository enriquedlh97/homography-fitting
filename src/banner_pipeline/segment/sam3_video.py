"""SAM3 video predictor — multi-frame object tracking via SAM 3.1.

Drop-in replacement for SAM2VideoSegmenter. The public ``segment_video``
method returns the same ``(video_segments, frame_dir, frame_names)`` tuple
so ``pipeline.py`` needs no changes.

User-facing prompts stay in the repo's absolute-pixel ``ObjectPrompt``
format. The adapter translates them into the current upstream SAM3.1
video contract: relative point coordinates, ``point_labels``, and
streamed ``propagate_in_video`` responses. Model loading remains
compatibility-driven and FlashAttention backend selection stays in
``device.py``. T4 is intentionally unsupported for SAM3.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import cv2
import numpy as np
import torch

from banner_pipeline.device import detect_device, load_sam3_video_predictor
from banner_pipeline.io import extract_all_frames
from banner_pipeline.segment.base import ObjectPrompt

# Default SAM 3.1 multiplex checkpoint. The loader in ``device.py`` adapts
# this file path to the builder API exposed by the installed ``sam3`` package
# and configures the GPU-family FlashAttention backend before model creation.
DEFAULT_CHECKPOINT = "sam3/checkpoints/sam3.1_multiplex.pt"

OBJECT_ID_KEYS = ("obj_ids", "object_ids", "out_obj_ids", "instance_ids")
MASK_KEYS = ("video_res_masks", "masks", "pred_masks", "out_mask_logits", "mask_logits")

Sam3Response = dict[str, object]


class _Sam3Predictor(Protocol):
    """Minimal request/response surface used by the SAM3 adapter."""

    def handle_request(self, request: Sam3Response) -> Sam3Response: ...


@runtime_checkable
class _Sam3StreamingPredictor(_Sam3Predictor, Protocol):
    """Predictors that expose streamed propagation responses."""

    def handle_stream_request(self, request: Sam3Response) -> Iterable[Sam3Response]: ...


class SAM3VideoSegmenter:
    """Wraps SAM 3's video predictor for full-video object tracking.

    Uses SAM 3.1's Object Multiplex for joint multi-object tracking,
    which is significantly faster than tracking objects sequentially.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "auto",
    ) -> None:
        self._device = detect_device(device)
        print(f"[SAM3Video] Loading model on {self._device} …", flush=True)
        self._predictor = cast(_Sam3Predictor, load_sam3_video_predictor(checkpoint, self._device))

    @property
    def name(self) -> str:
        return "sam3_video"

    # ------------------------------------------------------------------
    # Core propagation
    # ------------------------------------------------------------------

    def _propagate(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        """Start a SAM 3 session, add prompts, propagate masks.

        Returns
        -------
        video_segments : dict[frame_idx, dict[obj_id, np.ndarray]]
            Per-frame binary masks for each tracked object.
        frame_dir : str
            Temporary directory with extracted JPEG frames.
            **Caller is responsible for cleanup** (``shutil.rmtree``).
        frame_names : list[str]
            Sorted frame filenames within *frame_dir*.
        """
        video_path = str(Path(video_path).expanduser().resolve())

        # Extract frames (SAM3's video predictor also accepts a JPEG folder).
        frame_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        print("[SAM3Video] Extracting frames …", flush=True)
        frame_names = extract_all_frames(video_path, frame_dir)
        print(f"[SAM3Video] {len(frame_names)} frames → {frame_dir}")
        if not frame_names:
            raise RuntimeError("SAM3 frame extraction produced no frames.")

        first_frame = cv2.imread(str(Path(frame_dir) / frame_names[0]))
        if first_frame is None:
            raise RuntimeError(f"Could not read first extracted SAM3 frame: {frame_names[0]}")
        frame_height, frame_width = first_frame.shape[:2]

        # Start a fresh session using the official SAM3 flow.
        print("[SAM3Video] Starting session …", flush=True)
        session_id = _start_session(self._predictor, frame_dir)
        seed_frame_idx = min(int(prompt.frame_idx) for prompt in prompts)

        try:
            usable_prompt_count = _seed_session_prompts(
                self._predictor,
                prompts=prompts,
                session_id=session_id,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            # Propagate across all frames.
            print("[SAM3Video] Propagating masks …", flush=True)
            try:
                video_segments = _propagate_streamed_outputs(
                    self._predictor,
                    session_id=session_id,
                    seed_frame_idx=seed_frame_idx,
                    num_frames=len(frame_names),
                    frame_shape=(frame_height, frame_width),
                )
            except RuntimeError as exc:
                if "No prompts are received on any frames" not in str(exc):
                    raise
                raise RuntimeError(
                    "SAM3 propagate_in_video reported that no prompts were registered "
                    f"from seed frame {seed_frame_idx}, even though {usable_prompt_count} "
                    "add_prompt call(s) were accepted and produced prompt-stage outputs. "
                    "This suggests the predictor did not treat those point prompts as "
                    "propagation start state without an explicit start_frame_index."
                ) from exc
        finally:
            with suppress(Exception):
                self._predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )

        return video_segments, frame_dir, frame_names

    # ------------------------------------------------------------------
    # Public API — identical signature to SAM2VideoSegmenter
    # ------------------------------------------------------------------

    def segment_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        """Track objects across all frames and return per-frame masks.

        Returns
        -------
        video_segments : dict[frame_idx, dict[obj_id, np.ndarray]]
            Per-frame binary masks.
        frame_dir : str
            Temporary directory with extracted JPEG frames.
            **Caller is responsible for cleanup** (``shutil.rmtree``).
        frame_names : list[str]
            Sorted frame filenames within *frame_dir*.
        """
        return self._propagate(video_path, prompts)


# ---------------------------------------------------------------------------
# Request / response adaptation
# ---------------------------------------------------------------------------


def _start_session(predictor: _Sam3Predictor, frame_dir: str) -> str:
    """Create a fresh SAM3 session for prompt registration."""
    try:
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frame_dir,
            )
        )
        session_id = response.get("session_id")
        if not isinstance(session_id, str):
            raise RuntimeError(f"SAM3 start_session returned no session_id: {response!r}")
        return session_id
    except Exception as exc:
        raise RuntimeError(
            "SAM3 session initialization failed before prompt registration. "
            "Ensure the predictor session can be started successfully."
        ) from exc


def _seed_session_prompts(
    predictor: _Sam3Predictor,
    prompts: list[ObjectPrompt],
    session_id: str,
    frame_width: int,
    frame_height: int,
) -> int:
    """Register all prompts for a SAM3 session with a clear failure mode."""
    usable_prompt_count = 0
    for prompt in prompts:
        req = _build_add_prompt_request(
            prompt,
            session_id=session_id,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        try:
            response = predictor.handle_request(request=req)
        except Exception as exc:
            raise RuntimeError(
                "SAM3 prompt registration failed during session initialization. "
                "The predictor rejected an add_prompt request before propagation."
            ) from exc
        diagnostics = _summarize_prompt_response(response)
        if diagnostics["usable_outputs"]:
            usable_prompt_count += 1
        print(
            "[SAM3Video] Prompt "
            f"obj_id={prompt.obj_id} frame={prompt.frame_idx} "
            f"response_keys={diagnostics['response_keys']} "
            f"outputs_present={diagnostics['outputs_present']} "
            f"known_output_keys={diagnostics['known_output_keys']} "
            f"usable_outputs={diagnostics['usable_outputs']}",
            flush=True,
        )

    if usable_prompt_count == 0:
        raise RuntimeError(
            "SAM3 add_prompt completed for all prompts but produced no usable outputs. "
            "Prompt registration may not have been accepted by the predictor."
        )
    return usable_prompt_count


def _summarize_prompt_response(response: object) -> dict[str, object]:
    """Extract compact diagnostics from a SAM3 add_prompt response."""
    response_keys: list[str] = []
    outputs_present = False
    known_output_keys: list[str] = []
    usable_outputs = False

    if isinstance(response, dict):
        response_keys = sorted(str(key) for key in response)
        outputs_present = "outputs" in response and response["outputs"] is not None
        outputs = response.get("outputs")
        if isinstance(outputs, dict):
            known_output_keys = [key for key in (*OBJECT_ID_KEYS, *MASK_KEYS) if key in outputs]
            usable_outputs = any(
                _response_value_is_nonempty(outputs[key]) for key in known_output_keys
            )

    return {
        "response_keys": response_keys,
        "outputs_present": outputs_present,
        "known_output_keys": known_output_keys,
        "usable_outputs": usable_outputs,
    }


def _response_value_is_nonempty(value: object) -> bool:
    """Return whether a response payload contains any actual output entries."""
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_response_value_is_nonempty(item) for item in value.values())
    if isinstance(value, list | tuple | set | str | bytes):
        return len(value) > 0
    if hasattr(value, "numel"):
        return bool(value.numel())
    try:
        array = np.asarray(value)
    except Exception:
        return True
    return bool(array.size)


def _build_add_prompt_request(
    prompt: ObjectPrompt,
    session_id: str,
    frame_width: int,
    frame_height: int,
) -> dict:
    """Convert a repo ``ObjectPrompt`` into the current SAM3 video request."""
    if prompt.box is not None:
        raise RuntimeError(
            "SAM3VideoSegmenter currently supports point prompts only. "
            "Box prompts are not wired until the upstream SAM3 video box contract "
            "is validated for this repo."
        )
    if prompt.points is None or len(prompt.points) == 0:
        raise RuntimeError("SAM3VideoSegmenter requires at least one point per object.")

    points = np.asarray(prompt.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid SAM3 point prompt shape: expected (N, 2), got {points.shape}.")

    labels = np.asarray(prompt.labels, dtype=np.int32)
    if labels.ndim != 1 or len(labels) != len(points):
        raise ValueError(
            "SAM3 point labels must be a 1D array with the same length as the points array."
        )

    rel_points = points.copy()
    rel_points[:, 0] /= float(frame_width)
    rel_points[:, 1] /= float(frame_height)

    return dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=int(prompt.frame_idx),
        obj_id=int(prompt.obj_id),
        points=torch.tensor(rel_points, dtype=torch.float32),
        point_labels=torch.tensor(labels, dtype=torch.int32),
    )


def _propagate_streamed_outputs(
    predictor: _Sam3Predictor,
    session_id: str,
    seed_frame_idx: int,
    num_frames: int,
    frame_shape: tuple[int, int],
) -> dict[int, dict[int, np.ndarray]]:
    """Convert SAM3 propagation outputs into the SAM2-compatible format."""
    if isinstance(predictor, _Sam3StreamingPredictor):
        responses = predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=seed_frame_idx,
            )
        )
        return _parse_stream_responses(responses, num_frames=num_frames, frame_shape=frame_shape)

    response = predictor.handle_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=seed_frame_idx,
        )
    )
    if "frame_index" in response and "outputs" in response:
        return _parse_stream_responses([response], num_frames=num_frames, frame_shape=frame_shape)
    return _parse_legacy_propagate_response(
        response,
        num_frames=num_frames,
        frame_shape=frame_shape,
    )


def _parse_stream_responses(
    responses: Iterable[Sam3Response],
    num_frames: int,
    frame_shape: tuple[int, int],
) -> dict[int, dict[int, np.ndarray]]:
    """Parse streamed SAM3 ``propagate_in_video`` responses."""
    video_segments: dict[int, dict[int, np.ndarray]] = {}
    for response in responses:
        frame_idx = _coerce_frame_index(response["frame_index"])
        outputs = response.get("outputs")
        video_segments[frame_idx] = _parse_frame_outputs(outputs, frame_shape)

    for i in range(num_frames):
        video_segments.setdefault(i, {})
    return video_segments


def _parse_frame_outputs(
    outputs: object,
    frame_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Parse one frame of SAM3 outputs into ``obj_id -> uint8 mask``."""
    h, w = frame_shape
    masks_by_obj: dict[int, np.ndarray] = {}
    if outputs is None or not isinstance(outputs, dict):
        return masks_by_obj

    obj_ids = None
    for key in OBJECT_ID_KEYS:
        if key in outputs:
            obj_ids = _robust_np(outputs[key]).reshape(-1).tolist()
            break

    masks_blob = None
    for key in MASK_KEYS:
        if key in outputs:
            masks_blob = outputs[key]
            break

    if isinstance(masks_blob, dict):
        for obj_id, mask in masks_blob.items():
            mask_u8 = _mask_to_uint8(mask, h, w)
            if mask_u8.any():
                masks_by_obj[_coerce_obj_id(obj_id)] = mask_u8
        return masks_by_obj

    if masks_blob is not None:
        masks_np = _robust_np(masks_blob)
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        elif masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 4 and masks_np.shape[0] == 1:
            masks_np = masks_np[0]

        if masks_np.ndim == 3:
            n_masks = masks_np.shape[0]
            if obj_ids is None:
                obj_ids = list(range(n_masks))
            usable = min(n_masks, len(obj_ids))
            for i in range(usable):
                mask_u8 = _mask_to_uint8(masks_np[i], h, w)
                if mask_u8.any():
                    masks_by_obj[_coerce_obj_id(obj_ids[i])] = mask_u8
            return masks_by_obj

    for value in outputs.values():
        if not isinstance(value, dict):
            continue
        maybe_masks = True
        for nested_value in value.values():
            arr = _robust_np(nested_value)
            if arr is None or arr.ndim not in (2, 3):
                maybe_masks = False
                break
        if not maybe_masks:
            continue
        for obj_id, mask in value.items():
            mask_u8 = _mask_to_uint8(mask, h, w)
            if mask_u8.any():
                masks_by_obj[_coerce_obj_id(obj_id)] = mask_u8
        return masks_by_obj

    return masks_by_obj


def _parse_legacy_propagate_response(
    response: Sam3Response,
    num_frames: int,
    frame_shape: tuple[int, int],
) -> dict[int, dict[int, np.ndarray]]:
    """Fallback for older non-streamed SAM3 response shapes."""
    h, w = frame_shape
    video_segments: dict[int, dict[int, np.ndarray]] = {}
    outputs = response.get("outputs")
    if not isinstance(outputs, list | tuple):
        outputs = []

    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        frame_idx = _coerce_frame_index(entry["frame_index"])
        obj_id = _coerce_obj_id(entry["obj_id"])
        mask_u8 = _mask_to_uint8(entry["mask"], h, w)
        if mask_u8.any():
            video_segments.setdefault(frame_idx, {})[obj_id] = mask_u8

    for i in range(num_frames):
        video_segments.setdefault(i, {})
    return video_segments


def _robust_np(value: object) -> np.ndarray:
    """Detach/CPU numpy conversion for tensors and tensor-like values."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _mask_to_uint8(mask: object, height: int, width: int) -> np.ndarray:
    """Normalize one SAM3 mask candidate to ``(H, W)`` uint8 ``0/255``."""
    mask_np = _robust_np(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        raise ValueError(f"Unsupported SAM3 mask shape: expected 2D mask, got {mask_np.shape}.")
    if mask_np.shape != (height, width):
        mask_np = cv2.resize(
            mask_np.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    return (mask_np > 0.0).astype(np.uint8) * 255


def _coerce_obj_id(obj_id: object) -> int:
    """Normalize SAM3 object identifiers to ints when possible."""
    if isinstance(obj_id, int | np.integer):
        return int(obj_id)
    text = str(obj_id)
    if text.isdigit():
        return int(text)
    raise ValueError(f"Unsupported SAM3 object id: {obj_id!r}")


def _coerce_frame_index(frame_idx: object) -> int:
    """Normalize SAM3 frame indices to ints when possible."""
    if isinstance(frame_idx, int | np.integer):
        return int(frame_idx)
    text = str(frame_idx)
    if text.isdigit():
        return int(text)
    raise ValueError(f"Unsupported SAM3 frame index: {frame_idx!r}")
