"""SAM3 video predictor — text-prompted automatic detection + tracking.

Mirrors the contract of :class:`banner_pipeline.segment.sam2_video.SAM2VideoSegmenter`
so it plugs into the existing video pipeline unchanged. The only difference is
that prompts carry a natural-language ``text`` field instead of click points;
SAM3 performs detection automatically and propagates masks across the video.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from banner_pipeline.io import extract_all_frames
from banner_pipeline.segment.base import ObjectPrompt

DEFAULT_BPE_PATH = "/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


class SAM3VideoSegmenter:
    """Wraps SAM3's video predictor for text-prompted full-video tracking."""

    def __init__(
        self,
        bpe_path: str = DEFAULT_BPE_PATH,
        device: str = "auto",  # noqa: ARG002 — SAM3 picks GPU via gpus_to_use
    ) -> None:
        import torch

        from sam3.model_builder import build_sam3_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 requires a CUDA GPU.")

        gpus = [torch.cuda.current_device()]
        print(f"[SAM3Video] Loading model on cuda:{gpus[0]} …", flush=True)
        self._predictor = build_sam3_video_predictor(
            bpe_path=bpe_path,
            gpus_to_use=gpus,
        )
        # Per-object confidence trace populated during _propagate.
        # Maps obj_id -> list of per-frame confidence values (from SAM3
        # `out_probs`). Used by pipeline.py for confidence-based filtering.
        self.confidence_by_obj: dict[int, list[float]] = {}

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
        """Run SAM3 detection + tracking and return per-frame masks.

        Returns
        -------
        video_segments : dict[frame_idx, dict[obj_id, np.ndarray (H, W) bool]]
        frame_dir      : path to extracted JPEG frames (caller cleans up)
        frame_names    : sorted frame filenames
        """
        text_prompts = [p for p in prompts if p.text]
        if not text_prompts:
            raise ValueError(
                "SAM3VideoSegmenter requires at least one prompt with a `text` field "
                "(e.g. text='logo'). Got prompts without text."
            )

        video_path_abs = str(Path(video_path).expanduser().resolve())

        # Extract frames so the downstream pipeline (fit + composite) can read
        # them frame-by-frame, exactly like the SAM2 video path.
        frame_dir = tempfile.mkdtemp(prefix="sam3_frames_")
        print("[SAM3Video] Extracting frames …", flush=True)
        frame_names = extract_all_frames(video_path_abs, frame_dir)
        print(f"[SAM3Video] {len(frame_names)} frames → {frame_dir}")

        # SAM3 reads the video file directly (it does not need pre-extracted
        # frames for inference; the extraction above is purely so the rest of
        # the pipeline can read frames as BGR images).
        response = self._predictor.handle_request(
            request={"type": "start_session", "resource_path": video_path_abs}
        )
        session_id = response["session_id"]

        self._predictor.handle_request(
            request={"type": "reset_session", "session_id": session_id}
        )

        for prompt in text_prompts:
            self._predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt.frame_idx,
                    "text": prompt.text,
                }
            )
            print(
                f"[SAM3Video] Prompt text='{prompt.text}' frame={prompt.frame_idx}",
                flush=True,
            )

        print("[SAM3Video] Propagating masks …", flush=True)
        stream = self._predictor.handle_stream_request(
            request={"type": "propagate_in_video", "session_id": session_id}
        )

        # Reset confidence trace for this run (segmenter may be reused).
        self.confidence_by_obj = {}

        video_segments: dict[int, dict[int, np.ndarray]] = {}
        for response in stream:
            frame_index = int(response["frame_index"])
            outputs = response["outputs"]
            masks = outputs["out_binary_masks"]  # (N, H, W) bool
            obj_ids = outputs["out_obj_ids"]  # list[int]
            probs = outputs.get("out_probs")  # may be None / list / np.ndarray
            video_segments[frame_index] = {
                int(obj_id): np.asarray(masks[i], dtype=bool)
                for i, obj_id in enumerate(obj_ids)
            }
            if probs is not None:
                for i, obj_id in enumerate(obj_ids):
                    try:
                        conf = float(np.asarray(probs[i]).item())
                    except (TypeError, ValueError):
                        continue
                    self.confidence_by_obj.setdefault(int(obj_id), []).append(conf)

        return video_segments, frame_dir, frame_names

    # ------------------------------------------------------------------
    # Public API (mirrors SAM2VideoSegmenter)
    # ------------------------------------------------------------------

    def segment_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        """Detect + track objects across all frames given text prompts."""
        return self._propagate(video_path, prompts)

    def mask_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
        output_path: str,
        alpha: float = 0.45,  # noqa: ARG002 — kept for interface parity
    ) -> str:
        """Not implemented for SAM3 — the pipeline composites via the standard path."""
        _ = (video_path, prompts, output_path)
        raise NotImplementedError(
            "Use run_pipeline_video() for SAM3 — there is no overlay-only fallback."
        )

    @staticmethod
    def cleanup_frame_dir(frame_dir: str) -> None:
        shutil.rmtree(frame_dir, ignore_errors=True)
