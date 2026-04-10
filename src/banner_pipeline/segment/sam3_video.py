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

import shutil
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import cv2
import numpy as np
import torch

from banner_pipeline import quality as quality_mod
from banner_pipeline.device import detect_device, load_sam3_video_predictor
from banner_pipeline.io import extract_all_frames
from banner_pipeline.segment.base import ObjectPrompt

# Default SAM 3.1 multiplex checkpoint. The loader in ``device.py`` adapts
# this file path to the builder API exposed by the installed ``sam3`` package
# and configures the GPU-family FlashAttention backend before model creation.
DEFAULT_CHECKPOINT = "sam3/checkpoints/sam3.1_multiplex.pt"

MAX_REANCHOR_EVENTS_PER_OBJECT = 3
REANCHOR_COOLDOWN_FRAMES = 8
REANCHOR_BAD_STREAK_THRESHOLD = 4

OBJECT_ID_KEYS = ("out_obj_ids", "obj_ids", "object_ids", "instance_ids")
MASK_KEYS = (
    "out_binary_masks",
    "video_res_masks",
    "masks",
    "pred_masks",
    "out_mask_logits",
    "mask_logits",
)

Sam3Response = dict[str, object]


class _Sam3Predictor(Protocol):
    """Minimal request/response surface used by the SAM3 adapter."""

    def handle_request(self, request: Sam3Response) -> Sam3Response: ...


@runtime_checkable
class _Sam3StreamingPredictor(_Sam3Predictor, Protocol):
    """Predictors that expose streamed propagation responses."""

    def handle_stream_request(self, request: Sam3Response) -> Iterable[Sam3Response]: ...


@dataclass
class _ReanchorState:
    last_good_quad: np.ndarray | None = None
    bad_streak: int = 0
    last_refresh_frame: int = -REANCHOR_COOLDOWN_FRAMES
    refresh_count: int = 0


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
        self.last_tracking_stats: dict[str, object] = {}
        print(f"[SAM3Video] Loading model on {self._device} …", flush=True)
        self._predictor = cast(_Sam3Predictor, load_sam3_video_predictor(checkpoint, self._device))

    @property
    def name(self) -> str:
        return "sam3_video"

    # ------------------------------------------------------------------
    # Core propagation
    # ------------------------------------------------------------------

    def preview_frame(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[np.ndarray, dict[int, np.ndarray], int, dict[int, dict[str, object]]]:
        """Return prompt-stage SAM3 masks for a single preview frame."""
        preview_frame_idx = _require_single_prompt_frame_idx(prompts)
        video_path = str(Path(video_path).expanduser().resolve())

        frame_dir = tempfile.mkdtemp(prefix="sam3_preview_frames_")
        print("[SAM3Video] Extracting frames for preview …", flush=True)
        frame_names = extract_all_frames(video_path, frame_dir)
        print(f"[SAM3Video] {len(frame_names)} frames → {frame_dir}")
        if not frame_names:
            raise RuntimeError("SAM3 frame extraction produced no frames.")
        if preview_frame_idx < 0 or preview_frame_idx >= len(frame_names):
            raise RuntimeError(
                f"SAM3 preview frame {preview_frame_idx} is outside the extracted frame range "
                f"(0..{len(frame_names) - 1})."
            )

        preview_frame = cv2.imread(str(Path(frame_dir) / frame_names[preview_frame_idx]))
        if preview_frame is None:
            raise RuntimeError(
                "Could not read preview frame "
                f"{preview_frame_idx}: {frame_names[preview_frame_idx]}"
            )
        frame_height, frame_width = preview_frame.shape[:2]

        print("[SAM3Video] Starting preview session …", flush=True)
        session_id = _start_session(self._predictor, frame_dir)
        try:
            _usable_prompt_count, prompt_stage_segments, prompt_diagnostics = _seed_session_prompts(
                self._predictor,
                prompts=prompts,
                session_id=session_id,
                frame_width=frame_width,
                frame_height=frame_height,
                require_usable_outputs=False,
            )
            masks = prompt_stage_segments.get(preview_frame_idx, {})
            return preview_frame, masks, preview_frame_idx, prompt_diagnostics
        finally:
            with suppress(Exception):
                self._predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
            shutil.rmtree(frame_dir, ignore_errors=True)

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
        tracked_obj_ids = sorted({int(prompt.obj_id) for prompt in prompts})

        try:
            usable_prompt_count, _prompt_stage_segments, _prompt_diagnostics = (
                _seed_session_prompts(
                    self._predictor,
                    prompts=prompts,
                    session_id=session_id,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    require_usable_outputs=True,
                )
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
            video_segments, reanchor_events = _recover_tracking_with_reanchors(
                self._predictor,
                session_id=session_id,
                video_segments=video_segments,
                prompts=prompts,
                frame_shape=(frame_height, frame_width),
                frame_width=frame_width,
                frame_height=frame_height,
                num_frames=len(frame_names),
            )
        finally:
            with suppress(Exception):
                self._predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )

        self.last_tracking_stats = _summarize_tracking_coverage(
            video_segments,
            num_frames=len(frame_names),
            tracked_obj_ids=tracked_obj_ids,
            reanchor_events=reanchor_events,
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
    *,
    require_usable_outputs: bool,
) -> tuple[int, dict[int, dict[int, np.ndarray]], dict[int, dict[str, object]]]:
    """Register all prompts for a SAM3 session with a clear failure mode."""
    usable_prompt_count = 0
    prompt_stage_segments: dict[int, dict[int, np.ndarray]] = {}
    prompt_diagnostics: dict[int, dict[str, object]] = {}
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
        frame_idx = _coerce_prompt_frame_index(response, fallback=prompt.frame_idx)
        parsed_masks = _parse_frame_outputs(response.get("outputs"), (frame_height, frame_width))
        prompt_has_usable_outputs = bool(diagnostics["usable_outputs"])
        if parsed_masks:
            prompt_stage_segments.setdefault(frame_idx, {}).update(parsed_masks)
        prompt_mask = parsed_masks.get(int(prompt.obj_id))
        retry_attempted = False
        retry_succeeded = False
        retry_exhausted = False
        retry_diagnostics: dict[str, object] | None = None
        retry_points: int | None = None

        if not _mask_is_nonempty(prompt_mask):
            retry_prompt = _build_seed_retry_prompt(
                prompt,
                frame_shape=(frame_height, frame_width),
            )
            if retry_prompt is not None:
                retry_attempted = True
                retry_points = len(retry_prompt.points)
                retry_response = predictor.handle_request(
                    request=_build_add_prompt_request(
                        retry_prompt,
                        session_id=session_id,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )
                retry_diagnostics = _summarize_prompt_response(retry_response)
                prompt_has_usable_outputs = prompt_has_usable_outputs or bool(
                    retry_diagnostics["usable_outputs"]
                )
                retry_frame_idx = _coerce_prompt_frame_index(
                    retry_response,
                    fallback=prompt.frame_idx,
                )
                retry_masks = _parse_frame_outputs(
                    retry_response.get("outputs"),
                    (frame_height, frame_width),
                )
                if retry_masks:
                    prompt_stage_segments.setdefault(retry_frame_idx, {}).update(retry_masks)
                    parsed_masks.update(retry_masks)
                frame_idx = retry_frame_idx
                prompt_mask = parsed_masks.get(int(prompt.obj_id))
                retry_succeeded = _mask_is_nonempty(prompt_mask)
                retry_exhausted = not retry_succeeded

        if prompt_has_usable_outputs:
            usable_prompt_count += 1
        mask_area_px, mask_bbox = _mask_area_and_bbox(prompt_mask)
        prompt_diagnostics[int(prompt.obj_id)] = {
            "obj_id": int(prompt.obj_id),
            "frame_idx": int(frame_idx),
            "response_keys": diagnostics["response_keys"],
            "outputs_present": diagnostics["outputs_present"],
            "known_output_keys": diagnostics["known_output_keys"],
            "usable_outputs": diagnostics["usable_outputs"],
            "parsed_nonempty_masks": len(parsed_masks),
            "mask_area_px": mask_area_px,
            "mask_bbox": mask_bbox,
            "seed_retry_attempted": retry_attempted,
            "seed_retry_succeeded": retry_succeeded,
            "seed_retry_exhausted": retry_exhausted,
            "seed_retry_num_points": retry_points,
            "seed_retry_response_keys": (
                [] if retry_diagnostics is None else retry_diagnostics["response_keys"]
            ),
            "seed_retry_known_output_keys": (
                [] if retry_diagnostics is None else retry_diagnostics["known_output_keys"]
            ),
            "seed_retry_usable_outputs": (
                False if retry_diagnostics is None else retry_diagnostics["usable_outputs"]
            ),
        }
        print(
            "[SAM3Video] Prompt "
            f"obj_id={prompt.obj_id} frame={prompt.frame_idx} "
            f"response_keys={diagnostics['response_keys']} "
            f"outputs_present={diagnostics['outputs_present']} "
            f"known_output_keys={diagnostics['known_output_keys']} "
            f"usable_outputs={diagnostics['usable_outputs']} "
            f"parsed_nonempty_masks={len(parsed_masks)} "
            f"seed_retry_attempted={retry_attempted} "
            f"seed_retry_succeeded={retry_succeeded}",
            flush=True,
        )

    if require_usable_outputs and usable_prompt_count == 0:
        raise RuntimeError(
            "SAM3 add_prompt completed for all prompts but produced no usable outputs. "
            "Prompt registration may not have been accepted by the predictor."
        )
    return usable_prompt_count, prompt_stage_segments, prompt_diagnostics


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


def _mask_is_nonempty(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    mask_2d = np.asarray(mask).squeeze()
    return bool(mask_2d.ndim == 2 and mask_2d.size and mask_2d.any())


def _build_seed_retry_prompt(
    prompt: ObjectPrompt,
    *,
    frame_shape: tuple[int, int],
) -> ObjectPrompt | None:
    points = np.asarray(prompt.points, dtype=np.float32)
    labels = np.asarray(prompt.labels, dtype=np.int32)
    positives = points[labels == 1]
    negatives = points[labels == 0]
    if len(positives) == 0:
        return None

    retry_points: list[np.ndarray] = []
    retry_labels: list[int] = []
    pos_center = positives.mean(axis=0)
    span = float(np.max(np.ptp(points, axis=0))) if len(points) > 1 else 0.0
    expand_px = max(6.0, min(18.0, span * 0.35 if span > 0.0 else 8.0))

    def _append(point: np.ndarray, label: int) -> None:
        for existing_point, existing_label in zip(retry_points, retry_labels, strict=True):
            if existing_label == label and np.linalg.norm(existing_point - point) <= 1.5:
                return
        retry_points.append(point.astype(np.float32))
        retry_labels.append(int(label))

    for point in positives:
        _append(point, 1)
    for point in negatives:
        _append(point, 0)

    if len(positives) == 1:
        positive_point = positives[0]
        if len(negatives) > 0:
            base_dir = _normalize_vec(positive_point - negatives.mean(axis=0))
        else:
            base_dir = np.array([1.0, 0.0], dtype=np.float32)
        ortho_dir = np.array([-base_dir[1], base_dir[0]], dtype=np.float32)
        _append(positive_point + base_dir * expand_px * 0.6, 1)
        _append(positive_point - base_dir * expand_px * 0.6, 1)
        _append(positive_point + ortho_dir * expand_px * 0.45, 1)
    else:
        _append(pos_center, 1)
        for point in positives:
            _append(pos_center + (point - pos_center) * 0.45, 1)

    for point in negatives:
        outward_dir = _normalize_vec(point - pos_center)
        _append(point + outward_dir * expand_px, 0)

    retry_points_np = _clip_points_to_frame(np.vstack(retry_points), frame_shape)
    retry_labels_np = np.asarray(retry_labels, dtype=np.int32)
    return ObjectPrompt(
        obj_id=int(prompt.obj_id),
        points=retry_points_np.astype(np.float32),
        labels=retry_labels_np,
        frame_idx=int(prompt.frame_idx),
        surface_type=prompt.surface_type,
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


def _recover_tracking_with_reanchors(
    predictor: _Sam3Predictor,
    session_id: str,
    video_segments: dict[int, dict[int, np.ndarray]],
    prompts: list[ObjectPrompt],
    frame_shape: tuple[int, int],
    frame_width: int,
    frame_height: int,
    num_frames: int,
) -> tuple[dict[int, dict[int, np.ndarray]], list[dict[str, int]]]:
    """Re-issue prompt points when tracking collapses for an object."""
    if not prompts:
        return video_segments, []

    states = {int(prompt.obj_id): _ReanchorState() for prompt in prompts}
    reanchor_events: list[dict[str, int]] = []
    frame_idx = min(int(prompt.frame_idx) for prompt in prompts)
    while frame_idx < num_frames:
        refreshed = False
        for prompt in prompts:
            obj_id = int(prompt.obj_id)
            state = states[obj_id]
            mask = video_segments.get(frame_idx, {}).get(obj_id)
            good_mask, quad = _evaluate_tracking_mask(mask, frame_shape)
            if good_mask:
                state.last_good_quad = quad
                state.bad_streak = 0
                continue

            state.bad_streak += 1
            if not _should_reanchor_object(state, frame_idx):
                continue

            last_good_quad = state.last_good_quad
            if last_good_quad is None:
                continue

            refresh_prompt = _build_refresh_prompt_from_quad(
                obj_id=obj_id,
                frame_idx=frame_idx,
                quad=np.asarray(last_good_quad, dtype=np.float32),
                frame_shape=frame_shape,
                surface_type=prompt.surface_type,
            )
            refresh_request = _build_add_prompt_request(
                refresh_prompt,
                session_id=session_id,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            response = predictor.handle_request(request=refresh_request)
            diagnostics = _summarize_prompt_response(response)
            parsed_masks = _parse_frame_outputs(response.get("outputs"), frame_shape)
            if parsed_masks:
                video_segments.setdefault(frame_idx, {}).update(parsed_masks)
            refreshed_mask = parsed_masks.get(obj_id)
            refreshed_ok, refreshed_quad = _evaluate_tracking_mask(refreshed_mask, frame_shape)
            if not diagnostics["usable_outputs"] or not refreshed_ok or refreshed_quad is None:
                continue

            print(
                "[SAM3Video] Re-anchoring "
                f"obj_id={obj_id} at frame={frame_idx} "
                f"refresh_count={state.refresh_count + 1}",
                flush=True,
            )
            refreshed_segments = _propagate_streamed_outputs(
                predictor,
                session_id=session_id,
                seed_frame_idx=frame_idx,
                num_frames=num_frames,
                frame_shape=frame_shape,
            )
            _merge_video_segments(video_segments, refreshed_segments, start_frame_idx=frame_idx)
            state.last_good_quad = refreshed_quad
            state.last_refresh_frame = frame_idx
            state.refresh_count += 1
            state.bad_streak = 0
            reanchor_events.append(
                {
                    "obj_id": obj_id,
                    "frame_idx": frame_idx,
                    "refresh_count": state.refresh_count,
                }
            )
            refreshed = True
            break

        if refreshed:
            continue
        frame_idx += 1

    return video_segments, reanchor_events


def _should_reanchor_object(state: _ReanchorState, frame_idx: int) -> bool:
    if state.refresh_count >= MAX_REANCHOR_EVENTS_PER_OBJECT:
        return False
    if state.bad_streak < REANCHOR_BAD_STREAK_THRESHOLD:
        return False
    return frame_idx - state.last_refresh_frame >= REANCHOR_COOLDOWN_FRAMES


def _build_refresh_prompt_from_quad(
    *,
    obj_id: int,
    frame_idx: int,
    quad: np.ndarray,
    frame_shape: tuple[int, int],
    surface_type: str = "banner",
) -> ObjectPrompt:
    """Create banner-specific refresh clicks from the last good quad."""
    ordered_quad = _order_quad_corners(quad)
    center = ordered_quad.mean(axis=0)
    left_mid = (ordered_quad[0] + ordered_quad[3]) / 2.0
    right_mid = (ordered_quad[1] + ordered_quad[2]) / 2.0
    top_mid = (ordered_quad[0] + ordered_quad[1]) / 2.0
    bottom_mid = (ordered_quad[2] + ordered_quad[3]) / 2.0

    axis_lr = right_mid - left_mid
    axis_tb = bottom_mid - top_mid
    if np.linalg.norm(axis_lr) >= np.linalg.norm(axis_tb):
        long_axis = axis_lr
        short_axis = axis_tb
    else:
        long_axis = axis_tb
        short_axis = axis_lr

    long_dir = _normalize_vec(long_axis)
    short_dir = _normalize_vec(short_axis)
    long_extent = max(float(np.linalg.norm(long_axis)), 1.0) * 0.22
    short_extent = max(float(np.linalg.norm(short_axis)), 6.0) * 0.7

    points = np.stack(
        [
            center - long_dir * long_extent,
            center + long_dir * long_extent,
            center - short_dir * short_extent,
            center + short_dir * short_extent,
        ],
        axis=0,
    )
    points = _clip_points_to_frame(points, frame_shape)
    labels = np.array([1, 1, 0, 0], dtype=np.int32)
    return ObjectPrompt(
        obj_id=obj_id,
        points=points.astype(np.float32),
        labels=labels,
        frame_idx=frame_idx,
        surface_type=surface_type,
    )


def _evaluate_tracking_mask(
    mask: np.ndarray | None,
    frame_shape: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    is_valid, quad, _flags, _stats = quality_mod.validate_tracking_mask(mask, frame_shape)
    return is_valid, quad


def _fit_reanchor_quad_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    mask_2d = np.asarray(mask).squeeze()
    if mask_2d.ndim != 2 or not mask_2d.any():
        return None
    quad = quality_mod.fit_min_area_rect_quad(mask_2d)
    if quad is None:
        return None
    return _order_quad_corners(quad)


def _order_quad_corners(quad: np.ndarray) -> np.ndarray:
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vec / norm


def _clip_points_to_frame(points: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    clipped = np.asarray(points, dtype=np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height - 1))
    return clipped


def _merge_video_segments(
    base_segments: dict[int, dict[int, np.ndarray]],
    refreshed_segments: dict[int, dict[int, np.ndarray]],
    *,
    start_frame_idx: int,
) -> None:
    for frame_idx, masks_by_obj in refreshed_segments.items():
        if frame_idx < start_frame_idx:
            continue
        base_segments[frame_idx] = {obj_id: mask.copy() for obj_id, mask in masks_by_obj.items()}


def _summarize_tracking_coverage(
    video_segments: dict[int, dict[int, np.ndarray]],
    *,
    num_frames: int,
    tracked_obj_ids: list[int],
    reanchor_events: list[dict[str, int]],
) -> dict[str, object]:
    frames_with_masks = 0
    first_frame_with_mask: int | None = None
    last_frame_with_mask: int | None = None
    current_gap = 0
    max_gap = 0
    object_frame_coverage = {
        str(obj_id): {"frames_with_masks": 0, "coverage_ratio": 0.0} for obj_id in tracked_obj_ids
    }

    for frame_idx in range(num_frames):
        masks_by_obj = video_segments.get(frame_idx, {})
        nonempty_obj_ids = []
        for obj_id in tracked_obj_ids:
            mask = masks_by_obj.get(obj_id)
            mask_2d = np.asarray(mask).squeeze() if mask is not None else np.array([])
            if mask_2d.size and mask_2d.any():
                nonempty_obj_ids.append(obj_id)
                object_frame_coverage[str(obj_id)]["frames_with_masks"] = (
                    int(object_frame_coverage[str(obj_id)]["frames_with_masks"]) + 1
                )

        if nonempty_obj_ids:
            frames_with_masks += 1
            if first_frame_with_mask is None:
                first_frame_with_mask = frame_idx
            last_frame_with_mask = frame_idx
            current_gap = 0
        else:
            current_gap += 1
            max_gap = max(max_gap, current_gap)

    for obj_id in tracked_obj_ids:
        frames = int(object_frame_coverage[str(obj_id)]["frames_with_masks"])
        object_frame_coverage[str(obj_id)]["coverage_ratio"] = round(frames / max(num_frames, 1), 4)

    return {
        "frames_with_masks": frames_with_masks,
        "first_frame_with_mask": first_frame_with_mask,
        "last_frame_with_mask": last_frame_with_mask,
        "max_consecutive_mask_gap": max_gap,
        "object_frame_coverage": object_frame_coverage,
        "sam3_reanchor_events": reanchor_events,
    }


def _parse_stream_responses(
    responses: Iterable[Sam3Response],
    num_frames: int,
    frame_shape: tuple[int, int],
) -> dict[int, dict[int, np.ndarray]]:
    """Parse streamed SAM3 ``propagate_in_video`` responses."""
    video_segments: dict[int, dict[int, np.ndarray]] = {}
    logged_first_response = False
    for response in responses:
        frame_idx = _coerce_frame_index(response["frame_index"])
        outputs = response.get("outputs")
        masks_by_obj = _parse_frame_outputs(outputs, frame_shape)
        if not logged_first_response:
            output_keys = sorted(str(key) for key in outputs) if isinstance(outputs, dict) else []
            print(
                "[SAM3Video] First propagated response "
                f"frame={frame_idx} output_keys={output_keys} "
                f"parsed_nonempty_masks={len(masks_by_obj)}",
                flush=True,
            )
            logged_first_response = True
        video_segments[frame_idx] = masks_by_obj

    if not logged_first_response:
        print("[SAM3Video] Propagation returned no streamed frame outputs.", flush=True)

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
    if mask_np.ndim == 0:
        mask_np = mask_np.reshape(1, 1)
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


def _coerce_prompt_frame_index(response: Sam3Response, fallback: int) -> int:
    """Use a response frame index when present, else fall back to the prompt frame."""
    frame_idx = response.get("frame_index")
    if frame_idx is None:
        return int(fallback)
    return _coerce_frame_index(frame_idx)


def _require_single_prompt_frame_idx(prompts: list[ObjectPrompt]) -> int:
    """Ensure preview prompts all target the same frame."""
    if not prompts:
        raise RuntimeError("SAM3 preview requires at least one prompt.")
    frame_indices = sorted({int(prompt.frame_idx) for prompt in prompts})
    if len(frame_indices) != 1:
        raise RuntimeError(
            "SAM3 preview requires all prompts to target a single frame. "
            f"Found frame_idx values: {frame_indices}."
        )
    return frame_indices[0]


def _mask_area_and_bbox(mask: np.ndarray | None) -> tuple[int, list[int] | None]:
    if mask is None:
        return 0, None
    mask_2d = np.asarray(mask).squeeze()
    if mask_2d.ndim != 2 or not mask_2d.any():
        return 0, None
    ys, xs = np.nonzero(mask_2d)
    return int(len(xs)), [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
