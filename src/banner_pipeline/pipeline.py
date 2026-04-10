"""Pipeline orchestration — config loading, factory functions, and run_pipeline()."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import yaml

from banner_pipeline import _perf
from banner_pipeline.composite.alpha import AlphaCompositor
from banner_pipeline.composite.base import Compositor
from banner_pipeline.composite.inpaint import InpaintCompositor
from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.fitting.hull_fit import HullFitter
from banner_pipeline.fitting.lp_fit import LPFitter
from banner_pipeline.fitting.pca_fit import PCAFitter
from banner_pipeline.homography.camera import compute_oriented_homography, estimate_camera_matrix
from banner_pipeline.io import StreamingVideoWriter, get_video_fps, load_frame
from banner_pipeline.segment.base import ObjectPrompt, SegmentationModel
from banner_pipeline.segment.sam2_image import SAM2ImageSegmenter
from banner_pipeline.segment.sam2_video import SAM2VideoSegmenter
from banner_pipeline.segment.sam3_video import SAM3VideoSegmenter
from banner_pipeline.ui import OBJ_COLORS_UI, collect_clicks

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

SEGMENTERS: dict[str, type] = {
    "sam2_image": SAM2ImageSegmenter,
}

FITTERS: dict[str, type[QuadFitter]] = {
    "pca": PCAFitter,
    "lp": LPFitter,
    "hull": HullFitter,
}

COMPOSITORS: dict[str, type[Compositor]] = {
    "inpaint": InpaintCompositor,
    "alpha": AlphaCompositor,
}

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_segmenter(cfg: dict) -> SegmentationModel:
    cls = SEGMENTERS[cfg["type"]]
    kwargs = {}
    if "checkpoint" in cfg:
        kwargs["checkpoint"] = cfg["checkpoint"]
    if "model_cfg" in cfg:
        kwargs["model_cfg"] = cfg["model_cfg"]
    if "device" in cfg:
        kwargs["device"] = cfg["device"]
    return cls(**kwargs)


def build_fitter(cfg: dict) -> QuadFitter:
    return FITTERS[cfg["type"]]()


def build_compositor(cfg: dict) -> Compositor:
    return COMPOSITORS[cfg["type"]]()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load a YAML config and return the dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _prompts_from_config(prompts_cfg: list[dict]) -> list[ObjectPrompt]:
    """Convert a list of prompt dicts from YAML to ObjectPrompt instances."""
    out = []
    for p in prompts_cfg:
        pts = np.array(p["points"], dtype=np.float32)
        labels = np.ones(len(pts), dtype=np.int32)
        if "labels" in p:
            labels = np.array(p["labels"], dtype=np.int32)
        if labels.shape != (len(pts),):
            raise ValueError(
                "Prompt labels must be a 1D array with the same length as the points list."
            )
        out.append(
            ObjectPrompt(
                obj_id=p["obj_id"],
                points=pts,
                labels=labels,
                frame_idx=p.get("frame_idx", 0),
            )
        )
    return out


def _prompt_to_config_entry(prompt: ObjectPrompt) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "obj_id": prompt.obj_id,
        "points": prompt.points.tolist(),
        "labels": prompt.labels.tolist(),
    }
    if prompt.frame_idx != 0:
        entry["frame_idx"] = int(prompt.frame_idx)
    return entry


def _save_prompts_to_config(
    config: dict,
    prompts: list[ObjectPrompt],
    config_path: str,
) -> None:
    """Write collected prompts back into the config YAML for replay."""
    config["input"]["prompts"] = [_prompt_to_config_entry(prompt) for prompt in prompts]
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Prompts saved to: {config_path}")


def _looks_like_legacy_sam2_outline_prompts(prompts: list[ObjectPrompt]) -> bool:
    """Heuristic warning for SAM2-style outline prompts reused with SAM3."""
    if not prompts:
        return False
    if not all(np.all(prompt.labels == 1) for prompt in prompts):
        return False
    total_points = sum(len(prompt.points) for prompt in prompts)
    return any(len(prompt.points) >= 4 for prompt in prompts) or total_points >= 3 * len(prompts)


def _warn_if_legacy_sam3_prompts(segmenter_type: str, prompts: list[ObjectPrompt]) -> None:
    if segmenter_type != "sam3_video":
        return
    if not _looks_like_legacy_sam2_outline_prompts(prompts):
        return
    print(
        "[SAM3Video] Warning: this config looks like a legacy SAM2 outline prompt set "
        "(all-positive multi-point contours). SAM3 quality is usually better with 1-2 "
        "positive clicks inside the banner plus negative clicks on nearby background.",
        flush=True,
    )


def _load_or_collect_prompts(
    *,
    config: dict,
    config_path: str | None,
    video_path: str,
    segmenter_type: str,
    log_prefix: str,
    frame_idx: int = 0,
) -> list[ObjectPrompt]:
    prompts_cfg = config["input"].get("prompts")
    if prompts_cfg:
        prompts = _prompts_from_config(prompts_cfg)
        print(f"{log_prefix} Loaded {len(prompts)} prompts from config")
    else:
        print(f"{log_prefix} Interactive mode — collecting clicks …")
        frame = load_frame(video_path, frame_idx=frame_idx)
        prompts = collect_clicks(frame, frame_idx=frame_idx)
        if not prompts:
            return []
        if config_path:
            _save_prompts_to_config(config, prompts, config_path)

    _warn_if_legacy_sam3_prompts(segmenter_type, prompts)
    return prompts


def _fit_corners(
    masks: dict[int, np.ndarray],
    fitter: QuadFitter,
    fitter_params: dict[str, Any],
) -> dict[int, np.ndarray]:
    corners_map: dict[int, np.ndarray] = {}
    for obj_id, mask in masks.items():
        corners = fitter.fit(mask, **fitter_params)
        if corners is not None:
            corners_map[obj_id] = corners
    return corners_map


def _load_overlay(logo_path: str | None) -> np.ndarray | None:
    if not logo_path:
        return None
    overlay = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise RuntimeError(f"Could not read logo: {logo_path}")
    return overlay


def _composite_preview_frame(
    frame: np.ndarray,
    corners_map: dict[int, np.ndarray],
    overlay: np.ndarray | None,
    compositor_cfg: dict,
    masks: dict[int, np.ndarray],
    focal_length: float | None,
) -> tuple[np.ndarray, float | None]:
    if overlay is None or not corners_map:
        return frame.copy(), None

    overlay_img = overlay
    compositor = build_compositor(compositor_cfg)
    compositor_params = compositor_cfg.get("params", {})
    preview = frame.copy()
    t0 = time.perf_counter()
    K = estimate_camera_matrix(frame.shape, focal_length=focal_length)
    for obj_id in sorted(corners_map):
        extra_kw = dict(compositor_params)
        if compositor.name == "alpha":
            extra_kw["homo"] = compute_oriented_homography(corners_map[obj_id], K)
        preview = compositor.composite(
            preview,
            corners_map[obj_id],
            overlay_img,
            mask=masks.get(obj_id),
            **extra_kw,
        )
    return preview, time.perf_counter() - t0


def _annotate_preview_frame(
    frame: np.ndarray,
    masks: dict[int, np.ndarray],
    corners_map: dict[int, np.ndarray],
) -> np.ndarray:
    preview = frame.copy()
    mask_overlay = frame.copy()
    alpha = 0.32

    for idx, obj_id in enumerate(sorted(masks)):
        color = OBJ_COLORS_UI[idx % len(OBJ_COLORS_UI)]
        mask = np.asarray(masks[obj_id]).squeeze()
        if mask.ndim != 2 or not mask.any():
            continue
        mask_bool = mask.astype(bool)
        mask_overlay[mask_bool] = color

    preview = cv2.addWeighted(mask_overlay, alpha, preview, 1.0 - alpha, 0.0)

    for idx, obj_id in enumerate(sorted(corners_map)):
        color = OBJ_COLORS_UI[idx % len(OBJ_COLORS_UI)]
        corners = np.asarray(corners_map[obj_id], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [corners], True, color, 2, cv2.LINE_AA)
        anchor = tuple(corners[0, 0])
        cv2.putText(
            preview,
            f"obj {obj_id}",
            (int(anchor[0]) + 8, int(anchor[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return preview


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    config: dict,
    config_path: str | None = None,
) -> dict:
    """Execute the full banner-replacement pipeline.

    Parameters
    ----------
    config : dict
        Parsed YAML config.
    config_path : str, optional
        Path to the config file (used for auto-saving prompts on interactive runs).

    Returns
    -------
    dict with keys: frame, masks, corners_map, composited, metrics
    """
    segmenter_type = config["pipeline"]["segmenter"]["type"]
    if segmenter_type == "sam3_video":
        return _run_sam3_image_preview(config, config_path=config_path)

    metrics: dict[str, Any] = {}
    pipeline_cfg = config["pipeline"]
    input_cfg = config["input"]

    # --- Load frame ---
    t0 = time.perf_counter()
    frame = load_frame(input_cfg["video"])
    metrics["load_frame_s"] = time.perf_counter() - t0
    print(f"[pipeline] Frame: {frame.shape[1]}x{frame.shape[0]}")

    # --- Get prompts (interactive or from config) ---
    prompts = _load_or_collect_prompts(
        config=config,
        config_path=config_path,
        video_path=input_cfg["video"],
        segmenter_type=segmenter_type,
        log_prefix="[pipeline]",
        frame_idx=0,
    )
    if not prompts:
        print("[pipeline] No clicks — exiting.")
        return {
            "frame": frame,
            "masks": {},
            "corners_map": {},
            "composited": None,
            "metrics": metrics,
        }

    metrics["num_prompts"] = len(prompts)
    metrics["num_prompt_points"] = sum(len(p.points) for p in prompts)
    metrics["video_path"] = input_cfg["video"]
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["checkpoint"] = pipeline_cfg["segmenter"].get("checkpoint", "")
    metrics["frame_height"], metrics["frame_width"] = frame.shape[:2]

    # --- Segment ---
    t0 = time.perf_counter()
    segmenter = build_segmenter(pipeline_cfg["segmenter"])
    masks = segmenter.segment(frame, prompts)
    metrics["segment_s"] = time.perf_counter() - t0
    print(f"[pipeline] Segmented {len(masks)} objects in {metrics['segment_s']:.2f}s")

    # --- Fit quads ---
    t0 = time.perf_counter()
    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})
    corners_map = _fit_corners(masks, fitter, fitter_params)
    metrics["fit_s"] = time.perf_counter() - t0
    print(f"[pipeline] Fitted {len(corners_map)} quads in {metrics['fit_s']:.2f}s")

    # --- Composite ---
    composited = None
    logo_path = input_cfg.get("logo")
    if logo_path and corners_map:
        overlay = _load_overlay(logo_path)
        assert overlay is not None
        overlay_img = overlay

        t0 = time.perf_counter()
        compositor = build_compositor(pipeline_cfg["compositor"])
        compositor_params = pipeline_cfg["compositor"].get("params", {})
        composited = frame.copy()

        # Camera matrix for alpha compositor.
        focal_length = pipeline_cfg.get("camera", {}).get("focal_length")
        K = estimate_camera_matrix(frame.shape, focal_length=focal_length)

        for obj_id in sorted(corners_map):
            extra_kw = dict(compositor_params)
            if compositor.name == "alpha":
                homo = compute_oriented_homography(corners_map[obj_id], K)
                extra_kw["homo"] = homo
            composited = compositor.composite(
                composited,
                corners_map[obj_id],
                overlay_img,
                mask=masks.get(obj_id),
                **extra_kw,
            )
        metrics["composite_s"] = time.perf_counter() - t0
        print(f"[pipeline] Composited in {metrics['composite_s']:.2f}s")

    metrics["total_s"] = sum(v for k, v in metrics.items() if k.endswith("_s"))
    return {
        "frame": frame,
        "masks": masks,
        "corners_map": corners_map,
        "composited": composited,
        "metrics": metrics,
    }


def _run_sam3_image_preview(
    config: dict,
    config_path: str | None = None,
) -> dict:
    """Preview SAM3 prompt-stage masks on a single frame."""
    metrics: dict[str, Any] = {}
    pipeline_cfg = config["pipeline"]
    input_cfg = config["input"]
    video_path = input_cfg["video"]

    prompts = _load_or_collect_prompts(
        config=config,
        config_path=config_path,
        video_path=video_path,
        segmenter_type="sam3_video",
        log_prefix="[pipeline]",
        frame_idx=0,
    )
    if not prompts:
        print("[pipeline] No clicks — exiting.")
        return {
            "frame": None,
            "masks": {},
            "corners_map": {},
            "composited": None,
            "metrics": metrics,
        }

    t0 = time.perf_counter()
    video_segmenter = build_video_segmenter(pipeline_cfg["segmenter"])
    if not hasattr(video_segmenter, "preview_frame"):
        raise RuntimeError("SAM3 image preview requires the sam3_video segmenter.")
    frame, masks, preview_frame_idx = video_segmenter.preview_frame(video_path, prompts)
    metrics["segment_s"] = time.perf_counter() - t0
    metrics["preview_frame_idx"] = preview_frame_idx
    metrics["preview_objects_with_masks"] = len(masks)
    metrics["num_prompts"] = len(prompts)
    metrics["num_prompt_points"] = sum(len(prompt.points) for prompt in prompts)
    metrics["video_path"] = video_path
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["checkpoint"] = pipeline_cfg["segmenter"].get("checkpoint", "")
    metrics["frame_height"], metrics["frame_width"] = frame.shape[:2]

    if prompts and not masks:
        raise RuntimeError(
            "SAM3 preview produced no prompt-stage masks. Adjust the clicks and rerun --mode image."
        )

    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})
    t0 = time.perf_counter()
    corners_map = _fit_corners(masks, fitter, fitter_params)
    metrics["fit_s"] = time.perf_counter() - t0
    metrics["preview_objects_with_quads"] = len(corners_map)

    overlay = _load_overlay(input_cfg.get("logo"))
    composited, composite_s = _composite_preview_frame(
        frame,
        corners_map,
        overlay,
        pipeline_cfg["compositor"],
        masks,
        focal_length=pipeline_cfg.get("camera", {}).get("focal_length"),
    )
    composited = _annotate_preview_frame(composited, masks, corners_map)
    if composite_s is not None:
        metrics["composite_s"] = composite_s

    metrics["total_s"] = sum(v for k, v in metrics.items() if k.endswith("_s"))
    print(
        f"[pipeline] Previewed frame {preview_frame_idx}: "
        f"{metrics['preview_objects_with_masks']} object mask(s), "
        f"{metrics['preview_objects_with_quads']} fitted quad(s)"
    )

    return {
        "frame": frame,
        "masks": masks,
        "corners_map": corners_map,
        "composited": composited,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Video pipeline
# ---------------------------------------------------------------------------


def build_video_segmenter(cfg: dict) -> SAM2VideoSegmenter | SAM3VideoSegmenter:
    segmenter_type = cfg.get("type", "sam2_video")
    if segmenter_type == "sam3_video":
        kwargs: dict = {}
        if "checkpoint" in cfg:
            kwargs["checkpoint"] = cfg["checkpoint"]
        if "device" in cfg:
            kwargs["device"] = cfg["device"]
        return SAM3VideoSegmenter(**kwargs)
    # Default: SAM2
    kwargs = {}
    if "checkpoint" in cfg:
        kwargs["checkpoint"] = cfg["checkpoint"]
    if "model_cfg" in cfg:
        kwargs["model_cfg"] = cfg["model_cfg"]
    if "device" in cfg:
        kwargs["device"] = cfg["device"]
    return SAM2VideoSegmenter(**kwargs)


def _count_nonempty_frame_masks(
    video_segments: dict[int, dict[int, np.ndarray]],
    num_frames: int,
) -> tuple[int, int]:
    """Return ``(frames_with_masks, object_masks_total)`` for a tracked video."""
    frames_with_masks = 0
    object_masks_total = 0

    for frame_idx in range(num_frames):
        nonempty_masks = 0
        for mask in video_segments.get(frame_idx, {}).values():
            mask_2d = np.asarray(mask).squeeze()
            if mask_2d.size and mask_2d.any():
                nonempty_masks += 1
        if nonempty_masks:
            frames_with_masks += 1
            object_masks_total += nonempty_masks

    return frames_with_masks, object_masks_total


def _raise_video_coverage_error(
    reason: str,
    *,
    num_prompts: int,
    frames_with_masks: int,
    frames_with_quads: int,
    frames_composited: int,
    object_masks_total: int,
) -> None:
    raise RuntimeError(
        f"{reason} Coverage: "
        f"num_prompts={num_prompts}, "
        f"frames_with_masks={frames_with_masks}, "
        f"frames_with_quads={frames_with_quads}, "
        f"frames_composited={frames_composited}, "
        f"object_masks_total={object_masks_total}."
    )


def run_pipeline_video(
    config: dict,
    output_path: str = "output.mp4",
    config_path: str | None = None,
) -> dict:
    """Execute the full video banner-replacement pipeline.

    Tracks objects across all frames, fits quads per frame, composites
    per frame, and writes an output video.

    Returns
    -------
    dict with keys: output_path, metrics
    """
    import os
    import shutil

    metrics: dict[str, Any] = {}
    pipeline_cfg = config["pipeline"]
    input_cfg = config["input"]
    video_path = input_cfg["video"]
    segmenter_type = pipeline_cfg["segmenter"]["type"]

    # --- Get prompts ---
    prompts = _load_or_collect_prompts(
        config=config,
        config_path=config_path,
        video_path=video_path,
        segmenter_type=segmenter_type,
        log_prefix="[video]",
        frame_idx=0,
    )
    if not prompts:
        print("[video] No clicks — exiting.")
        return {"output_path": None, "metrics": metrics}

    # --- Input video info ---
    input_fps = get_video_fps(video_path)
    metrics["input_fps"] = input_fps
    metrics["num_prompts"] = len(prompts)
    metrics["num_prompt_points"] = sum(len(p.points) for p in prompts)
    metrics["video_path"] = video_path
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["checkpoint"] = pipeline_cfg["segmenter"].get("checkpoint", "")

    # Read frame size from the first frame.
    first_frame = load_frame(video_path, frame_idx=0)
    metrics["frame_height"], metrics["frame_width"] = first_frame.shape[:2]

    # --- Segment + track across all frames ---
    t0 = time.perf_counter()
    video_segmenter = build_video_segmenter(pipeline_cfg["segmenter"])
    video_segments, frame_dir, frame_names = video_segmenter.segment_video(
        video_path,
        prompts,
    )
    metrics["segment_total_s"] = time.perf_counter() - t0
    metrics["num_frames"] = len(frame_names)
    metrics["duration_s"] = round(len(frame_names) / input_fps, 2)
    print(
        f"[video] Tracked {len(frame_names)} frames in {metrics['segment_total_s']:.2f}s",
    )

    frames_with_masks, object_masks_total = _count_nonempty_frame_masks(
        video_segments,
        len(frame_names),
    )
    frames_with_quads = 0
    frames_composited = 0
    metrics["frames_with_masks"] = frames_with_masks
    metrics["frames_with_quads"] = frames_with_quads
    metrics["frames_composited"] = frames_composited
    metrics["object_masks_total"] = object_masks_total

    if prompts and frames_with_masks == 0:
        _raise_video_coverage_error(
            "No usable propagated masks were produced for this video run.",
            num_prompts=len(prompts),
            frames_with_masks=frames_with_masks,
            frames_with_quads=frames_with_quads,
            frames_composited=frames_composited,
            object_masks_total=object_masks_total,
        )

    # --- Per-frame: fit + composite ---
    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})

    overlay = None
    logo_path = input_cfg.get("logo")
    if logo_path:
        overlay = _load_overlay(logo_path)

    compositor = build_compositor(pipeline_cfg["compositor"]) if overlay is not None else None
    compositor_params = pipeline_cfg["compositor"].get("params", {}) if overlay is not None else {}
    focal_length = pipeline_cfg.get("camera", {}).get("focal_length")

    fit_times: list[float] = []
    composite_times: list[float] = []
    write_video_s = 0.0  # accumulated time spent piping frames to ffmpeg
    num_written = 0

    # Reset perf counters before the per-frame loop. PERF_ENABLED is False
    # by default, so the Timer blocks in compositors are no-ops unless the
    # caller has set _perf.enable() (e.g. via --profile).
    _perf.reset()

    # Open the streaming video writer using the first frame's dimensions.
    # This avoids buffering all frames in RAM and replaces the legacy
    # mp4v→ffmpeg double-write with a single libx264 encode pass.
    first_bgr = cv2.imread(os.path.join(frame_dir, frame_names[0]))
    if first_bgr is None:
        raise RuntimeError(f"Could not read first frame: {frame_names[0]}")
    fh, fw = first_bgr.shape[:2]
    video_writer = StreamingVideoWriter(output_path, fw, fh, fps=input_fps)

    try:
        for frame_idx, fname in enumerate(frame_names):
            if frame_idx == 0:
                frame_bgr = first_bgr
            else:
                next_frame = cv2.imread(os.path.join(frame_dir, fname))
                if next_frame is None:
                    raise RuntimeError(f"Could not read frame {frame_idx}: {fname}")
                frame_bgr = next_frame

            masks_for_frame = video_segments.get(frame_idx, {})

            # Squeeze masks to 2D (SAM2 video outputs may have extra dims).
            masks_2d: dict[int, np.ndarray] = {
                obj_id: mask.squeeze() for obj_id, mask in masks_for_frame.items()
            }

            # Fit quads for this frame.
            t_fit = time.perf_counter()
            corners_map = _fit_corners(masks_2d, fitter, fitter_params)
            fit_times.append(time.perf_counter() - t_fit)
            if corners_map:
                frames_with_quads += 1

            # Composite for this frame.
            if overlay is not None and compositor is not None and corners_map:
                t_comp = time.perf_counter()
                K = estimate_camera_matrix(frame_bgr.shape, focal_length=focal_length)
                for obj_id in sorted(corners_map):
                    extra_kw = dict(compositor_params)
                    if compositor.name == "alpha":
                        homo = compute_oriented_homography(corners_map[obj_id], K)
                        extra_kw["homo"] = homo
                    frame_bgr = compositor.composite(
                        frame_bgr,
                        corners_map[obj_id],
                        overlay,
                        mask=masks_2d.get(obj_id),
                        **extra_kw,
                    )
                composite_times.append(time.perf_counter() - t_comp)
                frames_composited += 1

            # Stream this frame to ffmpeg immediately (no in-memory buffer).
            t_write = time.perf_counter()
            video_writer.write(frame_bgr)
            num_written += 1
            write_video_s += time.perf_counter() - t_write

            if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_names) - 1:
                print(f"[video] Processed frame {frame_idx + 1}/{len(frame_names)}")

    finally:
        video_writer.close()
        shutil.rmtree(frame_dir, ignore_errors=True)

    metrics["frames_with_quads"] = frames_with_quads
    metrics["frames_composited"] = frames_composited
    metrics["write_video_s"] = round(write_video_s, 4)
    print(f"[video] Wrote {num_written} frames → {output_path}")

    if overlay is not None and frames_composited == 0:
        _raise_video_coverage_error(
            "No frames were composited despite propagated masks and a configured logo.",
            num_prompts=len(prompts),
            frames_with_masks=frames_with_masks,
            frames_with_quads=frames_with_quads,
            frames_composited=frames_composited,
            object_masks_total=object_masks_total,
        )

    # --- Aggregate metrics ---
    fit_arr = np.array(fit_times) * 1000  # ms
    metrics["fit_mean_ms"] = round(float(fit_arr.mean()), 2)
    metrics["fit_std_ms"] = round(float(fit_arr.std()), 2)

    if composite_times:
        comp_arr = np.array(composite_times) * 1000
        metrics["composite_mean_ms"] = round(float(comp_arr.mean()), 2)
        metrics["composite_std_ms"] = round(float(comp_arr.std()), 2)

    metrics["total_s"] = round(
        metrics["segment_total_s"]
        + sum(fit_times)
        + sum(composite_times)
        + metrics["write_video_s"],
        4,
    )
    metrics["output_fps"] = round(len(frame_names) / metrics["total_s"], 2)

    # Per-stage breakdown from _perf timers (empty dict if profiling disabled).
    if _perf.PERF_ENABLED:
        metrics["composite_breakdown_ms"] = _perf.snapshot_ms(divisor=len(frame_names))

    return {
        "output_path": output_path,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run(
    config: dict,
    config_path: str | None = None,
    output_path: str = "output.mp4",
) -> dict:
    """Dispatch to single-frame or video pipeline based on config ``mode``."""
    mode = config.get("pipeline", {}).get("mode", "image")
    if mode == "video":
        return run_pipeline_video(config, output_path=output_path, config_path=config_path)
    return run_pipeline(config, config_path=config_path)
