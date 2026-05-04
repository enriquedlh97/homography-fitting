"""SAM3-light segmenter — HSV-similarity gated re-runs.

Idea (mirrors light_sam3.py from the SAM3_project):
    - frame 0 → run SAM3, save the resulting masks and the frame's
      HSV-histogram as the *target*.
    - for every subsequent frame, compute the HSV-histogram correlation
      with the target.  When the correlation drops below
      ``similarity_threshold`` we assume a camera change / zoom / new
      objects in scene → rerun SAM3 and adopt the current frame as the
      new target.  Otherwise we reuse the previous detections.

Online, frame-by-frame; SAM3 re-runs are bottlenecked on per-call session
setup, so the segmenter is dramatically cheaper on static videos.

Plugs into the same pipeline contract as :class:`SAM3VideoSegmenter`:
returns ``(video_segments, frame_dir, frame_names)`` so the rest of
the pipeline (detection_filter → stabilization → fit → tracker →
composite) is unchanged.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from banner_pipeline.io import extract_all_frames
from banner_pipeline.segment.base import ObjectPrompt

DEFAULT_BPE_PATH = "/root/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_SIMILARITY_THRESHOLD = 0.85


class SAM3LightVideoSegmenter:
    """Online SAM3 segmenter with HSV-similarity + motion-magnitude rerun gates.

    Two independent triggers cause a SAM3 re-run on the current frame:

    1. **HSV correlation** with the last target frame drops below
       ``similarity_threshold`` (catches camera cuts / colour-palette
       changes).
    2. **Cumulative motion** since the last rerun (median Lucas-Kanade
       displacement on a sparse grid) exceeds ``motion_threshold_px``
       (catches zoom / pan that HSV misses because the colour palette
       barely changes).

    Set ``motion_threshold_px = 0`` to disable the motion gate (HSV-only,
    legacy behaviour).
    """

    def __init__(
        self,
        bpe_path: str = DEFAULT_BPE_PATH,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        motion_threshold_px: float = 0.0,
        device: str = "auto",  # noqa: ARG002
    ) -> None:
        import torch

        from sam3.model_builder import build_sam3_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 requires a CUDA GPU.")

        gpus = [torch.cuda.current_device()]
        print(f"[SAM3Light] Loading model on cuda:{gpus[0]} …", flush=True)
        self._predictor = build_sam3_video_predictor(
            bpe_path=bpe_path,
            gpus_to_use=gpus,
        )
        self.similarity_threshold = float(similarity_threshold)
        self.motion_threshold_px = float(motion_threshold_px)
        # Per-detection confidence trace (matches SAM3VideoSegmenter API
        # so the detection_filter logic in pipeline.py works unchanged).
        self.confidence_by_obj: dict[int, list[float]] = {}
        # Frame indices on which SAM3 was actually re-executed.
        self.rerun_indices: list[int] = []
        # Mean HSV-correlation over the run (excluding the first frame).
        self.mean_similarity: float = 1.0
        # Motion-aware rerun bookkeeping (only populated when
        # motion_threshold_px > 0).
        self.mean_motion_px: float = 0.0
        self.num_motion_triggered_reruns: int = 0

    @property
    def name(self) -> str:
        return "sam3_light_video"

    # ------------------------------------------------------------------
    # HSV histogram + correlation
    # ------------------------------------------------------------------

    @staticmethod
    def _hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [32, 32, 32],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _similarity(h1: np.ndarray, h2: np.ndarray) -> float:
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

    @staticmethod
    def _motion_magnitude(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
        """Median per-point Lucas-Kanade displacement (pixels) between frames.

        Tracks a sparse 7×7 grid of points across the frame. Returns 0 when
        no points could be reliably tracked. Cheap (~3-5 ms on 1080p).
        """
        h, w = prev_gray.shape
        nx = ny = 7
        ys = np.linspace(int(h * 0.05), int(h * 0.95), ny)
        xs = np.linspace(int(w * 0.05), int(w * 0.95), nx)
        pts = np.array(
            [[float(x), float(y)] for y in ys for x in xs],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            cur_gray,
            pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if status is None:
            return 0.0
        valid = status.ravel() == 1
        if not valid.any():
            return 0.0
        disp = np.linalg.norm(
            new_pts[valid].reshape(-1, 2) - pts[valid].reshape(-1, 2),
            axis=1,
        )
        return float(np.median(disp))

    # ------------------------------------------------------------------
    # Single-frame SAM3 invocation (writes a 1-frame mp4 + new session)
    # ------------------------------------------------------------------

    def _run_sam3_on_frame(
        self,
        frame_bgr: np.ndarray,
        prompt_text: str,
        single_dir: Path,
    ) -> dict:
        tmp_img = single_dir / "frame.png"
        tmp_vid = single_dir / "frame.mp4"
        cv2.imwrite(str(tmp_img), frame_bgr)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-loop", "1", "-i", str(tmp_img),
                "-frames:v", "1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(tmp_vid),
            ],
            check=True,
        )

        resp = self._predictor.handle_request(
            request={"type": "start_session", "resource_path": tmp_vid.as_posix()}
        )
        sid = resp["session_id"]
        self._predictor.handle_request(
            request={"type": "reset_session", "session_id": sid}
        )
        self._predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": sid,
                "frame_index": 0,
                "text": prompt_text,
            }
        )
        stream = self._predictor.handle_stream_request(
            request={"type": "propagate_in_video", "session_id": sid}
        )
        output: dict | None = None
        for response in stream:
            output = response["outputs"]
            break

        if output is None:
            h, w = frame_bgr.shape[:2]
            output = {
                "out_binary_masks": np.zeros((0, h, w), dtype=bool),
                "out_probs": np.zeros((0,), dtype=np.float32),
                "out_obj_ids": np.zeros((0,), dtype=np.int32),
            }
        return output

    # ------------------------------------------------------------------
    # Main propagate (matches SAM3VideoSegmenter._propagate contract)
    # ------------------------------------------------------------------

    def _propagate(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        text_prompts = [p for p in prompts if p.text]
        if not text_prompts:
            raise ValueError(
                "SAM3LightVideoSegmenter requires at least one prompt with a "
                "`text` field (e.g. text='logo')."
            )
        prompt_text = text_prompts[0].text

        video_path_abs = str(Path(video_path).expanduser().resolve())

        # Extract frames so the rest of the pipeline can read them.
        frame_dir = tempfile.mkdtemp(prefix="sam3_light_frames_")
        print("[SAM3Light] Extracting frames …", flush=True)
        frame_names = extract_all_frames(video_path_abs, frame_dir)
        print(f"[SAM3Light] {len(frame_names)} frames → {frame_dir}")

        # Scratch dir for the per-rerun 1-frame mp4 trick.
        single_dir = Path(tempfile.mkdtemp(prefix="sam3_light_single_"))

        # Reset state for this run.
        self.confidence_by_obj = {}
        self.rerun_indices = []
        self.num_motion_triggered_reruns = 0
        similarity_scores: list[float] = []
        per_frame_motion: list[float] = []

        target_hist: np.ndarray | None = None
        # Active detections (re-keyed obj_ids → mask / prob), refreshed only
        # on a SAM3 rerun. obj_ids are offset by `rerun_count * 10000` to
        # avoid collisions across re-runs (each SAM3 session restarts from 1).
        current_masks_by_id: dict[int, np.ndarray] = {}
        current_probs_by_id: dict[int, float] = {}
        rerun_count = 0

        video_segments: dict[int, dict[int, np.ndarray]] = {}

        prev_gray: np.ndarray | None = None
        accumulated_motion = 0.0
        motion_gate_on = self.motion_threshold_px > 0.0

        try:
            print(
                f"[SAM3Light] Online run, sim_threshold={self.similarity_threshold}, "
                f"motion_threshold_px={self.motion_threshold_px}, "
                f"text='{prompt_text}'",
                flush=True,
            )
            for frame_idx, fname in enumerate(frame_names):
                frame_bgr = cv2.imread(os.path.join(frame_dir, fname))
                if frame_bgr is None:
                    raise RuntimeError(f"Could not read frame {frame_idx}: {fname}")

                cur_hist = self._hsv_hist(frame_bgr)
                cur_gray = (
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    if motion_gate_on
                    else None
                )

                # Per-frame motion since previous frame (only when gate is on).
                frame_motion = 0.0
                if cur_gray is not None and prev_gray is not None:
                    frame_motion = self._motion_magnitude(prev_gray, cur_gray)
                    per_frame_motion.append(frame_motion)
                    accumulated_motion += frame_motion

                rerun_reason: str | None = None
                if target_hist is None:
                    need_rerun = True
                    sim = 1.0
                    rerun_reason = "init"
                else:
                    sim = self._similarity(target_hist, cur_hist)
                    if sim < self.similarity_threshold:
                        need_rerun = True
                        rerun_reason = "similarity"
                    elif (
                        motion_gate_on
                        and accumulated_motion > self.motion_threshold_px
                    ):
                        need_rerun = True
                        rerun_reason = "motion"
                        self.num_motion_triggered_reruns += 1
                    else:
                        need_rerun = False
                similarity_scores.append(sim)

                if need_rerun:
                    output = self._run_sam3_on_frame(frame_bgr, prompt_text, single_dir)
                    obj_ids = output["out_obj_ids"]
                    masks = output["out_binary_masks"]
                    probs = output.get("out_probs")

                    offset = rerun_count * 10000
                    rerun_count += 1
                    self.rerun_indices.append(frame_idx)

                    current_masks_by_id = {
                        int(obj_ids[i]) + offset:
                            np.asarray(masks[i], dtype=bool)
                        for i in range(len(obj_ids))
                    }
                    if probs is not None:
                        current_probs_by_id = {
                            int(obj_ids[i]) + offset: float(np.asarray(probs[i]).item())
                            for i in range(len(obj_ids))
                        }
                    else:
                        current_probs_by_id = {}

                    target_hist = cur_hist
                    accumulated_motion = 0.0
                    print(
                        f"[SAM3Light] frame {frame_idx:5d}  sim={sim:.3f}  "
                        f"motion={accumulated_motion:5.1f}px  "
                        f"RERUN ({rerun_reason})  "
                        f"({len(current_masks_by_id)} objects)",
                        flush=True,
                    )

                # Snapshot the active detections for this frame. Shallow
                # copy is fine — masks themselves are read-only downstream.
                video_segments[frame_idx] = dict(current_masks_by_id)
                for oid, prob in current_probs_by_id.items():
                    self.confidence_by_obj.setdefault(oid, []).append(prob)

                prev_gray = cur_gray
        finally:
            shutil.rmtree(single_dir, ignore_errors=True)

        if per_frame_motion:
            self.mean_motion_px = float(np.mean(per_frame_motion))

        # Mean similarity excluding the seeding frame (which is by definition 1.0).
        if len(similarity_scores) > 1:
            self.mean_similarity = float(np.mean(similarity_scores[1:]))
        else:
            self.mean_similarity = 1.0

        print(
            f"[SAM3Light] Done — {len(self.rerun_indices)}/{len(frame_names)} "
            f"frames triggered a SAM3 rerun "
            f"({100.0 * len(self.rerun_indices) / max(1, len(frame_names)):.1f}%); "
            f"mean similarity={self.mean_similarity:.3f}",
            flush=True,
        )

        return video_segments, frame_dir, frame_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
    ) -> tuple[dict[int, dict[int, np.ndarray]], str, list[str]]:
        return self._propagate(video_path, prompts)

    def mask_video(
        self,
        video_path: str,
        prompts: list[ObjectPrompt],
        output_path: str,
        alpha: float = 0.45,  # noqa: ARG002
    ) -> str:
        _ = (video_path, prompts, output_path)
        raise NotImplementedError(
            "Use run_pipeline_video() for SAM3-light — overlay-only path is "
            "not implemented."
        )

    @staticmethod
    def cleanup_frame_dir(frame_dir: str) -> None:
        shutil.rmtree(frame_dir, ignore_errors=True)
