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

MIN_PREVIEW_PRIMARY_FIT_AREA_RATIO = 5e-4
MIN_PREVIEW_MASK_AREA_PX = 16
MIN_PREVIEW_SMALL_BBOX_EDGE_PX = 8
MIN_PREVIEW_SMALL_MASK_COMPACTNESS = 0.12
MAX_PREVIEW_QUAD_AREA_RATIO = 0.55
MAX_PREVIEW_QUAD_ASPECT_RATIO = 80.0
MIN_PREVIEW_QUAD_EDGE_PX = 4.0
MIN_PREVIEW_QUAD_AREA_PX = 16.0
MIN_PREVIEW_QUAD_MASK_IOU = 0.03
MIN_PREVIEW_MASK_COVERAGE = 0.08
MIN_PREVIEW_QUAD_COVERAGE = 0.08

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


def _validate_prompt_points(segmenter_type: str, prompts: list[ObjectPrompt]) -> None:
    if segmenter_type != "sam3_video":
        return

    for prompt in prompts:
        points = np.asarray(prompt.points, dtype=np.float32)
        labels = np.asarray(prompt.labels, dtype=np.int32)
        ambiguous_foreground_background = False
        for idx in range(len(points)):
            for jdx in range(idx + 1, len(points)):
                dist_px = float(np.linalg.norm(points[idx] - points[jdx]))
                if dist_px <= 1.5:
                    raise RuntimeError(
                        "SAM3 prompt loading rejected duplicate clicks for "
                        f"obj_id={prompt.obj_id} at frame={prompt.frame_idx}. "
                        "Re-collect the prompt set without double-clicking the same location."
                    )
                if labels[idx] == labels[jdx] and dist_px < 4.0:
                    raise RuntimeError(
                        "SAM3 prompt loading rejected near-identical clicks for "
                        f"obj_id={prompt.obj_id} at frame={prompt.frame_idx}. "
                        "Spread repeated positive/negative clicks farther apart."
                    )
                if labels[idx] != labels[jdx] and dist_px <= 12.0:
                    ambiguous_foreground_background = True
        if ambiguous_foreground_background:
            print(
                "[SAM3Video] Warning: "
                f"obj_id={prompt.obj_id} mixes positive/negative clicks very close together. "
                "That usually means ambiguous foreground/background geometry and can hurt "
                "tracking on thin or painted banners.",
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
    _validate_prompt_points(segmenter_type, prompts)
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


def _annotate_prompt_markers(
    frame: np.ndarray,
    prompts: list[ObjectPrompt],
) -> np.ndarray:
    preview = frame.copy()
    for idx, prompt in enumerate(prompts):
        color = OBJ_COLORS_UI[idx % len(OBJ_COLORS_UI)]
        for point_idx, (point, label) in enumerate(
            zip(prompt.points, prompt.labels, strict=True),
            start=1,
        ):
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            marker = cv2.MARKER_STAR if int(label) == 1 else cv2.MARKER_TILTED_CROSS
            suffix = "+" if int(label) == 1 else "-"
            cv2.drawMarker(preview, (x, y), color, marker, 18, 2)
            cv2.putText(
                preview,
                f"{prompt.obj_id}.{point_idx}{suffix}",
                (x + 10, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return preview


def _mask_area_and_bbox(mask: np.ndarray | None) -> tuple[int, list[int] | None]:
    if mask is None:
        return 0, None
    mask_2d = np.asarray(mask).squeeze()
    if mask_2d.ndim != 2 or not mask_2d.any():
        return 0, None
    ys, xs = np.nonzero(mask_2d)
    return int(len(xs)), [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_dims(mask_bbox: list[int] | None) -> tuple[int, int]:
    if mask_bbox is None:
        return 0, 0
    x0, y0, x1, y1 = mask_bbox
    return x1 - x0 + 1, y1 - y0 + 1


def _polygon_area(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _quad_edge_lengths(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def _quad_mask_overlap(mask: np.ndarray, corners: np.ndarray) -> dict[str, float]:
    mask_2d = np.asarray(mask).squeeze().astype(bool)
    quad_mask = np.zeros(mask_2d.shape, dtype=np.uint8)
    polygon = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(quad_mask, polygon, 255)
    quad_bool = quad_mask.astype(bool)
    intersection = int(np.logical_and(mask_2d, quad_bool).sum())
    union = int(np.logical_or(mask_2d, quad_bool).sum())
    quad_area = int(quad_bool.sum())
    mask_area = int(mask_2d.sum())
    return {
        "iou": 0.0 if union == 0 else intersection / union,
        "mask_coverage": 0.0 if mask_area == 0 else intersection / mask_area,
        "quad_coverage": 0.0 if quad_area == 0 else intersection / quad_area,
    }


def _preview_geometry_flags(
    mask: np.ndarray,
    corners: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[list[str], dict[str, float]]:
    height, width = frame_shape[:2]
    frame_area = float(height * width)
    edges = _quad_edge_lengths(corners)
    quad_area = _polygon_area(corners)
    aspect_ratio = float(edges.max() / max(edges.min(), 1e-6))
    overlap = _quad_mask_overlap(mask, corners)

    stats = {
        "quad_area_px": round(quad_area, 2),
        "quad_aspect_ratio": round(aspect_ratio, 2),
        "quad_mask_iou": round(overlap["iou"], 4),
        "mask_coverage": round(overlap["mask_coverage"], 4),
        "quad_coverage": round(overlap["quad_coverage"], 4),
    }
    flags: list[str] = []
    if quad_area < MIN_PREVIEW_QUAD_AREA_PX:
        flags.append("quad_area_too_small")
    if quad_area > frame_area * MAX_PREVIEW_QUAD_AREA_RATIO:
        flags.append("quad_area_too_large")
    if float(edges.min()) < MIN_PREVIEW_QUAD_EDGE_PX:
        flags.append("quad_edge_too_short")
    if aspect_ratio > MAX_PREVIEW_QUAD_ASPECT_RATIO:
        flags.append("quad_aspect_ratio_too_high")
    if overlap["iou"] < MIN_PREVIEW_QUAD_MASK_IOU:
        flags.append("quad_mask_iou_low")
    if overlap["mask_coverage"] < MIN_PREVIEW_MASK_COVERAGE:
        flags.append("mask_coverage_low")
    if overlap["quad_coverage"] < MIN_PREVIEW_QUAD_COVERAGE:
        flags.append("quad_coverage_low")

    corners_np = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if (
        (corners_np[:, 0] < -0.05 * width).any()
        or (corners_np[:, 0] > 1.05 * width).any()
        or (corners_np[:, 1] < -0.05 * height).any()
        or (corners_np[:, 1] > 1.05 * height).any()
    ):
        flags.append("quad_outside_frame")
    return flags, stats


def _fit_min_area_rect_quad(mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    if mask_u8.ndim != 2 or not mask_u8.any():
        return None
    nonzero = cv2.findNonZero(mask_u8)
    if nonzero is None or len(nonzero) < 4:
        return None
    rect = cv2.minAreaRect(nonzero)
    width, height = rect[1]
    if min(width, height) < 2.0:
        return None
    box = cv2.boxPoints(rect).astype(np.float32)
    sums = box.sum(axis=1)
    diffs = (box[:, 0] - box[:, 1]).reshape(-1)
    return np.array(
        [
            box[np.argmin(sums)],
            box[np.argmax(diffs)],
            box[np.argmax(sums)],
            box[np.argmin(diffs)],
        ],
        dtype=np.float32,
    )


def _fit_preview_corners(
    *,
    mask: np.ndarray,
    mask_area_px: int,
    mask_bbox: list[int] | None,
    frame_shape: tuple[int, int],
    fitter: QuadFitter,
    fitter_params: dict[str, Any],
) -> tuple[np.ndarray | None, str, str | None]:
    bbox_w, bbox_h = _bbox_dims(mask_bbox)
    bbox_area = max(bbox_w * bbox_h, 1)
    compactness = mask_area_px / bbox_area
    primary_area_px = max(
        64,
        int(frame_shape[0] * frame_shape[1] * MIN_PREVIEW_PRIMARY_FIT_AREA_RATIO),
    )

    if mask_area_px < MIN_PREVIEW_MASK_AREA_PX or min(bbox_w, bbox_h) < MIN_PREVIEW_QUAD_EDGE_PX:
        return None, "not_run", "mask_area_too_small"

    if mask_area_px < primary_area_px:
        if (
            min(bbox_w, bbox_h) < MIN_PREVIEW_SMALL_BBOX_EDGE_PX
            or compactness < MIN_PREVIEW_SMALL_MASK_COMPACTNESS
        ):
            return None, "not_run", "small_mask_not_compact"
        fallback = _fit_min_area_rect_quad(mask)
        if fallback is None:
            return None, "min_area_rect_fallback", "fit_failed"
        return fallback, "min_area_rect_fallback", None

    primary = fitter.fit(mask, **fitter_params)
    if primary is not None:
        return primary, "primary", None

    fallback = _fit_min_area_rect_quad(mask)
    if fallback is None:
        return None, "primary", "fit_failed"
    return fallback, "min_area_rect_fallback", None


def _composite_preview_with_diagnostics(
    *,
    frame: np.ndarray,
    overlay: np.ndarray | None,
    compositor_cfg: dict,
    masks: dict[int, np.ndarray],
    corners_map: dict[int, np.ndarray],
    preview_object_diagnostics: dict[str, dict[str, Any]],
    focal_length: float | None,
) -> tuple[np.ndarray, float | None, list[str]]:
    preview = frame.copy()
    if overlay is None:
        for diag in preview_object_diagnostics.values():
            diag["composite_status"] = "skipped"
            diag["composite_failure_reason"] = "no_logo_configured"
        return preview, None, []

    compositor = build_compositor(compositor_cfg)
    compositor_params = compositor_cfg.get("params", {})
    K = estimate_camera_matrix(frame.shape, focal_length=focal_length)
    t0 = time.perf_counter()
    failures: list[str] = []

    for obj_id_text, diag in preview_object_diagnostics.items():
        obj_id = int(obj_id_text)
        diag.setdefault("background_fill_color_bgr", None)
        diag.setdefault("background_fill_spread_bgr", None)
        if obj_id not in corners_map:
            diag["composite_status"] = "skipped"
            diag["composite_failure_reason"] = "fit_unavailable"
            continue

        extra_kw = dict(compositor_params)
        debug_info: dict[str, object] = {}

        try:
            if compositor.name == "alpha":
                extra_kw["homo"] = compute_oriented_homography(corners_map[obj_id], K)
                extra_kw["debug_info"] = debug_info
            preview = compositor.composite(
                preview,
                corners_map[obj_id],
                overlay,
                mask=masks.get(obj_id),
                **extra_kw,
            )
        except Exception as exc:
            diag["composite_status"] = "failed"
            diag["composite_failure_reason"] = f"{type(exc).__name__}: {exc}"
            diag["failure_stage"] = "composite"
            failures.append(f"obj {obj_id}: composite failure ({diag['composite_failure_reason']})")
            continue

        diag["background_fill_color_bgr"] = debug_info.get("fill_color_bgr")
        diag["background_fill_spread_bgr"] = debug_info.get("fill_spread_bgr")
        if debug_info.get("fill_unstable"):
            diag["composite_status"] = "warning"
            diag["composite_failure_reason"] = str(debug_info.get("fill_warning_reason"))
            diag["failure_stage"] = "composite"
            failures.append(f"obj {obj_id}: composite failure ({diag['composite_failure_reason']})")
        else:
            diag["composite_status"] = "ok"
            diag["composite_failure_reason"] = None

    return preview, time.perf_counter() - t0, failures


def _build_preview_diagnostics(
    *,
    frame: np.ndarray,
    prompts: list[ObjectPrompt],
    masks: dict[int, np.ndarray],
    prompt_diagnostics: dict[int, dict[str, object]],
    fitter: QuadFitter,
    fitter_params: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, dict[str, Any]], list[str]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    valid_corners_map: dict[int, np.ndarray] = {}
    invalid_objects: list[str] = []
    frame_shape = frame.shape[:2]

    for prompt in prompts:
        obj_id = int(prompt.obj_id)
        mask = masks.get(obj_id)
        mask_area_px, mask_bbox = _mask_area_and_bbox(mask)
        base_diag = dict(prompt_diagnostics.get(obj_id, {}))
        base_diag.update(
            {
                "mask_area_px": mask_area_px,
                "mask_bbox": mask_bbox,
                "fit_status": "not_run",
                "fit_method": "not_run",
                "fit_geometry_flags": [],
                "seed_ok": False,
                "failure_stage": "mask",
                "composite_status": "not_run",
                "composite_failure_reason": None,
                "background_fill_color_bgr": None,
                "background_fill_spread_bgr": None,
            }
        )

        if mask_area_px == 0:
            retry_exhausted = bool(base_diag.get("seed_retry_exhausted"))
            base_diag["fit_geometry_flags"] = (
                ["seed_retry_exhausted"] if retry_exhausted else ["empty_mask"]
            )
            diagnostics[str(obj_id)] = base_diag
            invalid_objects.append(str(obj_id))
            continue

        assert mask is not None
        corners, fit_method, fit_failure_flag = _fit_preview_corners(
            mask=mask,
            mask_area_px=mask_area_px,
            mask_bbox=mask_bbox,
            frame_shape=frame_shape,
            fitter=fitter,
            fitter_params=fitter_params,
        )
        base_diag["fit_method"] = fit_method
        if corners is None:
            base_diag["fit_status"] = "failed"
            base_diag["failure_stage"] = "fit"
            base_diag["fit_geometry_flags"] = [fit_failure_flag or "fit_failed"]
            diagnostics[str(obj_id)] = base_diag
            invalid_objects.append(str(obj_id))
            continue

        flags, geometry_stats = _preview_geometry_flags(mask, corners, frame_shape)
        base_diag["fit_status"] = "ok" if not flags else "rejected"
        base_diag["failure_stage"] = None if not flags else "fit"
        base_diag["fit_geometry_flags"] = flags
        base_diag.update(geometry_stats)
        if flags:
            diagnostics[str(obj_id)] = base_diag
            invalid_objects.append(str(obj_id))
            continue

        base_diag["seed_ok"] = True
        diagnostics[str(obj_id)] = base_diag
        valid_corners_map[obj_id] = corners

    return valid_corners_map, diagnostics, invalid_objects


def _summarize_preview_failures(
    diagnostics: dict[str, dict[str, Any]],
    invalid_objects: list[str],
) -> list[str]:
    failures: list[str] = []
    for obj_id in invalid_objects:
        diag = diagnostics[obj_id]
        flags = diag.get("fit_geometry_flags", [])
        if diag.get("failure_stage") == "mask":
            failures.append(f"obj {obj_id}: mask failure ({', '.join(flags)})")
        elif diag.get("failure_stage") == "composite":
            reason = diag.get("composite_failure_reason") or "unknown_composite_failure"
            failures.append(f"obj {obj_id}: composite failure ({reason})")
        else:
            failures.append(f"obj {obj_id}: fit failure ({', '.join(flags)})")
    return failures


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
    frame, masks, preview_frame_idx, prompt_diagnostics = video_segmenter.preview_frame(
        video_path,
        prompts,
    )
    metrics["segment_s"] = time.perf_counter() - t0
    metrics["num_prompts"] = len(prompts)
    metrics["num_prompt_points"] = sum(len(prompt.points) for prompt in prompts)
    metrics["video_path"] = video_path
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["checkpoint"] = pipeline_cfg["segmenter"].get("checkpoint", "")
    metrics["frame_height"], metrics["frame_width"] = frame.shape[:2]
    metrics["preview_frame_idx"] = preview_frame_idx

    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})
    t0 = time.perf_counter()
    corners_map, preview_object_diagnostics, invalid_objects = _build_preview_diagnostics(
        frame=frame,
        prompts=prompts,
        masks=masks,
        prompt_diagnostics=prompt_diagnostics,
        fitter=fitter,
        fitter_params=fitter_params,
    )
    metrics["fit_s"] = time.perf_counter() - t0
    metrics["preview_objects_with_masks"] = sum(
        1 for diag in preview_object_diagnostics.values() if int(diag.get("mask_area_px", 0)) > 0
    )
    metrics["preview_objects_with_quads"] = len(corners_map)

    overlay = _load_overlay(input_cfg.get("logo"))
    composited, composite_s, composite_failures = _composite_preview_with_diagnostics(
        frame=frame,
        overlay=overlay,
        compositor_cfg=pipeline_cfg["compositor"],
        masks=masks,
        corners_map=corners_map,
        preview_object_diagnostics=preview_object_diagnostics,
        focal_length=pipeline_cfg.get("camera", {}).get("focal_length"),
    )
    composited = _annotate_preview_frame(composited, masks, corners_map)
    preview_failure_reasons = _summarize_preview_failures(
        preview_object_diagnostics,
        invalid_objects,
    )
    preview_failure_reasons.extend(composite_failures)
    metrics["preview_object_diagnostics"] = preview_object_diagnostics
    metrics["preview_ok"] = len(preview_failure_reasons) == 0 and len(corners_map) > 0
    metrics["preview_failure_reasons"] = preview_failure_reasons

    preview_artifacts = {
        "preview_prompts": _annotate_prompt_markers(frame, prompts),
        "preview_masks": _annotate_preview_frame(frame, masks, {}),
        "composited": composited,
    }
    if composite_s is not None:
        metrics["composite_s"] = composite_s

    metrics["total_s"] = sum(v for k, v in metrics.items() if k.endswith("_s"))
    print(
        f"[pipeline] Previewed frame {preview_frame_idx}: "
        f"{metrics['preview_objects_with_masks']} object mask(s), "
        f"{metrics['preview_objects_with_quads']} fitted quad(s)"
    )
    if metrics["preview_failure_reasons"]:
        print(
            "[pipeline] Preview failures: " + "; ".join(metrics["preview_failure_reasons"]),
            flush=True,
        )

    return {
        "frame": frame,
        "masks": masks,
        "corners_map": corners_map,
        "composited": composited,
        "preview_artifacts": preview_artifacts,
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


def _summarize_video_coverage(
    video_segments: dict[int, dict[int, np.ndarray]],
    num_frames: int,
    tracked_obj_ids: list[int],
) -> dict[str, Any]:
    frames_with_masks, object_masks_total = _count_nonempty_frame_masks(video_segments, num_frames)
    first_frame_with_mask: int | None = None
    last_frame_with_mask: int | None = None
    max_gap = 0
    current_gap = 0
    object_frame_coverage = {
        str(obj_id): {"frames_with_masks": 0, "coverage_ratio": 0.0} for obj_id in tracked_obj_ids
    }

    for frame_idx in range(num_frames):
        masks_by_obj = video_segments.get(frame_idx, {})
        frame_has_mask = False
        for obj_id in tracked_obj_ids:
            mask = masks_by_obj.get(obj_id)
            mask_2d = np.asarray(mask).squeeze() if mask is not None else np.array([])
            if mask_2d.size and mask_2d.any():
                frame_has_mask = True
                object_frame_coverage[str(obj_id)]["frames_with_masks"] = (
                    int(object_frame_coverage[str(obj_id)]["frames_with_masks"]) + 1
                )

        if frame_has_mask:
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
        "object_masks_total": object_masks_total,
        "first_frame_with_mask": first_frame_with_mask,
        "last_frame_with_mask": last_frame_with_mask,
        "max_consecutive_mask_gap": max_gap,
        "object_frame_coverage": object_frame_coverage,
    }


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

    coverage = _summarize_video_coverage(
        video_segments,
        len(frame_names),
        tracked_obj_ids=sorted({int(prompt.obj_id) for prompt in prompts}),
    )
    tracker_stats = getattr(video_segmenter, "last_tracking_stats", {})
    if isinstance(tracker_stats, dict):
        coverage.update({key: value for key, value in tracker_stats.items() if key not in coverage})

    frames_with_masks = int(coverage["frames_with_masks"])
    object_masks_total = int(coverage["object_masks_total"])
    frames_with_quads = 0
    frames_composited = 0
    metrics["frames_with_masks"] = frames_with_masks
    metrics["frames_with_quads"] = frames_with_quads
    metrics["frames_composited"] = frames_composited
    metrics["object_masks_total"] = object_masks_total
    metrics["first_frame_with_mask"] = coverage["first_frame_with_mask"]
    metrics["last_frame_with_mask"] = coverage["last_frame_with_mask"]
    metrics["max_consecutive_mask_gap"] = coverage["max_consecutive_mask_gap"]
    metrics["object_frame_coverage"] = coverage["object_frame_coverage"]
    if "sam3_reanchor_events" in coverage:
        metrics["sam3_reanchor_events"] = coverage["sam3_reanchor_events"]

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
