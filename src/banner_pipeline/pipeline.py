"""Pipeline orchestration — config loading, factory functions, and run_pipeline()."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import yaml

from banner_pipeline import _perf
from banner_pipeline import court_geometry as court_geometry_mod
from banner_pipeline import quality as quality_mod
from banner_pipeline import stabilization as stabilization_mod
from banner_pipeline.composite.alpha import AlphaCompositor
from banner_pipeline.composite.base import Compositor
from banner_pipeline.composite.inpaint import InpaintCompositor
from banner_pipeline.fitting.base import QuadFitter
from banner_pipeline.fitting.fronto_parallel import FrontoParallelBannerFitter
from banner_pipeline.fitting.hull_fit import HullFitter
from banner_pipeline.fitting.lp_fit import LPFitter
from banner_pipeline.fitting.pca_fit import PCAFitter
from banner_pipeline.fitting.vp_constrained import VPConstrainedBannerFitter
from banner_pipeline.homography.camera import compute_oriented_homography, estimate_camera_matrix
from banner_pipeline.io import StreamingVideoWriter, get_video_fps, load_frame
from banner_pipeline.segment.base import ObjectPrompt, SegmentationModel
from banner_pipeline.segment.sam2_image import SAM2ImageSegmenter
from banner_pipeline.segment.sam2_video import SAM2VideoSegmenter
from banner_pipeline.segment.sam3_video import SAM3VideoSegmenter
from banner_pipeline.tracking import CornerTracker
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
    "fronto_parallel": FrontoParallelBannerFitter,
    "vp_constrained": VPConstrainedBannerFitter,
}

COMPOSITORS: dict[str, type[Compositor]] = {
    "inpaint": InpaintCompositor,
    "alpha": AlphaCompositor,
}
SUPPORTED_BANNER_SURFACE_TYPES = {"banner"}
COURT_MARKING_SURFACE_TYPE = "court_marking"

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


def _normalize_surface_type(surface_type: object) -> str:
    if surface_type is None:
        return "banner"
    text = str(surface_type).strip().lower()
    return text or "banner"


def _surface_skip_reason(surface_type: str) -> str:
    return f"unsupported_surface_type:{surface_type}"


def _normalize_geometry_model(geometry_model: object | None) -> str | None:
    return court_geometry_mod.normalize_geometry_model(geometry_model)


def _geometry_enabled(pipeline_cfg: dict[str, Any]) -> bool:
    return court_geometry_mod.is_enabled(pipeline_cfg.get("geometry"))


def _stabilization_enabled(pipeline_cfg: dict[str, Any]) -> bool:
    return stabilization_mod.StabilizationConfig.from_dict(
        pipeline_cfg.get("stabilization")
    ).enabled


def _is_supported_banner_surface(
    prompt: ObjectPrompt,
    *,
    geometry_enabled: bool = False,
) -> bool:
    return court_geometry_mod.supports_surface_type(
        _normalize_surface_type(prompt.surface_type),
        geometry_enabled=geometry_enabled,
    )


def _preview_frame_idx_from_prompts(prompts: list[ObjectPrompt]) -> int:
    if not prompts:
        return 0
    frame_indices = sorted({int(prompt.frame_idx) for prompt in prompts})
    if len(frame_indices) != 1:
        raise RuntimeError(
            "SAM3 preview requires all prompts to target a single frame. "
            f"Found frame_idx values: {frame_indices}."
        )
    return frame_indices[0]


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
        box = None
        if "box" in p:
            box = np.array(p["box"], dtype=np.float32)
        placement_quad = None
        if "placement_quad" in p:
            placement_quad = np.array(p["placement_quad"], dtype=np.float32)
        out.append(
            ObjectPrompt(
                obj_id=p["obj_id"],
                points=pts,
                labels=labels,
                frame_idx=p.get("frame_idx", 0),
                surface_type=_normalize_surface_type(p.get("surface_type", "banner")),
                geometry_model=_normalize_geometry_model(p.get("geometry_model")),
                box=box,
                placement_quad=placement_quad,
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
    if _normalize_surface_type(prompt.surface_type) != "banner":
        entry["surface_type"] = _normalize_surface_type(prompt.surface_type)
    if _normalize_geometry_model(prompt.geometry_model) is not None:
        entry["geometry_model"] = _normalize_geometry_model(prompt.geometry_model)
    return entry


def _clicks_to_prompts(click_groups: list[list[tuple[int, int]]]) -> list[ObjectPrompt]:
    """Convert interactive click groups to ObjectPrompt list."""
    prompts = []
    for idx, group in enumerate(click_groups):
        obj_id = idx + 1
        pts = np.array(group, dtype=np.float32)
        labels = np.ones(len(group), dtype=np.int32)
        prompts.append(ObjectPrompt(obj_id=obj_id, points=pts, labels=labels))
    return prompts


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
    banner_prompts = [
        prompt for prompt in prompts if _is_supported_banner_surface(prompt, geometry_enabled=True)
    ]
    if not banner_prompts:
        return False
    if not all(np.all(prompt.labels == 1) for prompt in banner_prompts):
        return False
    total_points = sum(len(prompt.points) for prompt in banner_prompts)
    return any(len(prompt.points) >= 4 for prompt in banner_prompts) or total_points >= 3 * len(
        banner_prompts
    )


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
        if not _is_supported_banner_surface(prompt, geometry_enabled=True):
            continue
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


def _geometry_active_object_ids(
    prompts: list[ObjectPrompt],
    *,
    geometry_enabled: bool,
) -> list[int]:
    if not geometry_enabled:
        return []
    active_obj_ids: list[int] = []
    for prompt in prompts:
        if not _is_supported_banner_surface(prompt, geometry_enabled=True):
            continue
        if court_geometry_mod.resolve_geometry_model(prompt) == "mask_free_quad":
            continue
        active_obj_ids.append(int(prompt.obj_id))
    return sorted(set(active_obj_ids))


def _init_runtime_feature_metrics(
    metrics: dict[str, Any],
    *,
    pipeline_cfg: dict[str, Any],
    prompts: list[ObjectPrompt],
) -> None:
    geometry_enabled = _geometry_enabled(pipeline_cfg)
    metrics["geometry_config_enabled"] = geometry_enabled
    metrics["geometry_runtime_enabled"] = False
    metrics["geometry_active_objects"] = _geometry_active_object_ids(
        prompts,
        geometry_enabled=geometry_enabled,
    )
    metrics["stabilization_config_enabled"] = _stabilization_enabled(pipeline_cfg)
    metrics["stabilization_runtime_enabled"] = False


def _require_runtime_feature_metrics(
    *,
    feature_name: str,
    metrics: dict[str, Any],
    config_enabled: bool,
    runtime_enabled: bool,
    required_keys: list[str],
) -> None:
    if not config_enabled:
        return
    if not runtime_enabled:
        raise RuntimeError(
            f"{feature_name} was enabled in the config, but the runtime path did not execute."
        )
    missing_keys = [key for key in required_keys if key not in metrics]
    if missing_keys:
        raise RuntimeError(
            f"{feature_name} was enabled in the config, but runtime metrics were missing: "
            f"{missing_keys}."
        )


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
    return quality_mod.mask_area_and_bbox(mask)


def _bbox_dims(mask_bbox: list[int] | None) -> tuple[int, int]:
    return quality_mod.bbox_dims(mask_bbox)


def _polygon_area(corners: np.ndarray) -> float:
    return quality_mod.polygon_area(corners)


def _quad_edge_lengths(corners: np.ndarray) -> np.ndarray:
    return quality_mod.quad_edge_lengths(corners)


def _quad_mask_overlap(mask: np.ndarray, corners: np.ndarray) -> dict[str, float]:
    return quality_mod.quad_mask_overlap(mask, corners)


def _preview_geometry_flags(
    mask: np.ndarray,
    corners: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[list[str], dict[str, float]]:
    return quality_mod.geometry_flags(mask, corners, frame_shape)


def _fit_min_area_rect_quad(mask: np.ndarray) -> np.ndarray | None:
    return quality_mod.fit_min_area_rect_quad(mask)


def _fit_preview_corners(
    *,
    mask: np.ndarray,
    mask_area_px: int,
    mask_bbox: list[int] | None,
    frame_shape: tuple[int, int],
    fitter: QuadFitter,
    fitter_params: dict[str, Any],
) -> tuple[np.ndarray | None, str, str | None]:
    return quality_mod.fit_corners_with_fallback(
        mask=mask,
        mask_area_px=mask_area_px,
        mask_bbox=mask_bbox,
        frame_shape=frame_shape,
        fit_primary=lambda current_mask: fitter.fit(current_mask, **fitter_params),
    )


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
            if diag.get("failure_stage") == "surface":
                diag.setdefault("composite_status", "skipped")
                diag.setdefault("composite_failure_reason", diag.get("skip_reason"))
            else:
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
    geometry_engine: court_geometry_mod.GeometryFittingEngine | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, dict[str, Any]], list[str]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    valid_corners_map: dict[int, np.ndarray] = {}
    invalid_objects: list[str] = []
    frame_shape = frame.shape[:2]
    geometry_corners_map: dict[int, np.ndarray] = {}
    geometry_rejections: dict[int, list[str]] = {}
    if geometry_engine is not None:
        geometry_corners_map, geometry_rejections = geometry_engine.fit_frame(
            frame_idx=0,
            frame_bgr=frame,
            masks_by_obj={
                int(obj_id): np.asarray(mask).squeeze() for obj_id, mask in masks.items()
            },
            frame_shape=frame_shape,
        )

    for prompt in prompts:
        obj_id = int(prompt.obj_id)
        surface_type = _normalize_surface_type(prompt.surface_type)
        geometry_model = court_geometry_mod.resolve_geometry_model(prompt)
        mask = masks.get(obj_id)
        corners: np.ndarray | None = None
        fit_method = "not_run"
        fit_failure_flag: str | None = None
        mask_area_px, mask_bbox = _mask_area_and_bbox(mask)
        base_diag = dict(prompt_diagnostics.get(obj_id, {}))
        base_diag.update(
            {
                "mask_area_px": mask_area_px,
                "mask_bbox": mask_bbox,
                "surface_type": surface_type,
                "geometry_model": geometry_model,
                "fit_status": "not_run",
                "fit_method": "not_run",
                "fit_held": False,
                "fit_used_fallback": False,
                "fit_geometry_flags": [],
                "seed_ok": False,
                "failure_stage": "mask",
                "composite_status": "not_run",
                "composite_failure_reason": None,
                "background_fill_color_bgr": None,
                "background_fill_spread_bgr": None,
            }
        )
        if geometry_engine is not None and obj_id in geometry_engine.details:
            base_diag["fit_method"] = geometry_engine.details[obj_id].fit_method
            base_diag["fit_held"] = geometry_engine.details[obj_id].held
            base_diag["fit_used_fallback"] = geometry_engine.details[obj_id].used_fallback

        if geometry_engine is None and surface_type not in SUPPORTED_BANNER_SURFACE_TYPES:
            skip_reason = _surface_skip_reason(surface_type)
            base_diag["fit_status"] = "skipped"
            base_diag["failure_stage"] = "surface"
            base_diag["skip_reason"] = skip_reason
            base_diag["composite_status"] = "skipped"
            base_diag["composite_failure_reason"] = skip_reason
            diagnostics[str(obj_id)] = base_diag
            invalid_objects.append(str(obj_id))
            continue

        if mask_area_px == 0:
            retry_exhausted = bool(base_diag.get("seed_retry_exhausted"))
            base_diag["fit_geometry_flags"] = (
                ["seed_retry_exhausted"] if retry_exhausted else ["empty_mask"]
            )
            diagnostics[str(obj_id)] = base_diag
            invalid_objects.append(str(obj_id))
            continue

        assert mask is not None
        if geometry_engine is not None:
            corners = geometry_corners_map.get(obj_id)
            fit_failure_reasons = geometry_rejections.get(obj_id, [])
            fit_failure_flag = fit_failure_reasons[0] if fit_failure_reasons else "fit_failed"
        else:
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
            if fit_failure_flag is not None and fit_failure_flag.startswith(
                "unsupported_surface_type:"
            ):
                base_diag["fit_status"] = "skipped"
                base_diag["failure_stage"] = "surface"
                base_diag["skip_reason"] = fit_failure_flag
                base_diag["composite_status"] = "skipped"
                base_diag["composite_failure_reason"] = fit_failure_flag
                diagnostics[str(obj_id)] = base_diag
                invalid_objects.append(str(obj_id))
                continue
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
        if diag.get("failure_stage") == "surface":
            failures.append(f"obj {obj_id}: skipped ({diag.get('skip_reason')})")
        elif diag.get("failure_stage") == "mask":
            failures.append(f"obj {obj_id}: mask failure ({', '.join(flags)})")
        elif diag.get("failure_stage") == "composite":
            reason = diag.get("composite_failure_reason") or "unknown_composite_failure"
            failures.append(f"obj {obj_id}: composite failure ({reason})")
        else:
            failures.append(f"obj {obj_id}: fit failure ({', '.join(flags)})")
    return failures


def _fit_and_validate_video_objects(
    *,
    prompts: list[ObjectPrompt],
    masks_by_obj: dict[int, np.ndarray],
    frame_shape: tuple[int, int],
    fitter: QuadFitter,
    fitter_params: dict[str, Any],
    geometry_engine: court_geometry_mod.GeometryFittingEngine | None = None,
    frame_idx: int = 0,
    frame_bgr: np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    if geometry_engine is not None:
        if frame_bgr is None:
            raise ValueError("frame_bgr is required when geometry_engine is provided")
        return geometry_engine.fit_frame(
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
            masks_by_obj=masks_by_obj,
            frame_shape=frame_shape,
        )

    valid_corners_map: dict[int, np.ndarray] = {}
    rejection_reasons: dict[int, list[str]] = {}

    for prompt in prompts:
        obj_id = int(prompt.obj_id)
        surface_type = _normalize_surface_type(prompt.surface_type)
        if surface_type not in SUPPORTED_BANNER_SURFACE_TYPES:
            rejection_reasons[obj_id] = [_surface_skip_reason(surface_type)]
            continue

        mask = masks_by_obj.get(obj_id)
        mask_area_px, mask_bbox = _mask_area_and_bbox(mask)
        if mask_area_px == 0 or mask is None:
            rejection_reasons[obj_id] = ["empty_mask"]
            continue

        corners, _fit_method, fit_failure_flag = _fit_preview_corners(
            mask=mask,
            mask_area_px=mask_area_px,
            mask_bbox=mask_bbox,
            frame_shape=frame_shape,
            fitter=fitter,
            fitter_params=fitter_params,
        )
        if corners is None:
            rejection_reasons[obj_id] = [fit_failure_flag or "fit_failed"]
            continue

        flags, _geometry_stats = _preview_geometry_flags(mask, corners, frame_shape)
        if flags:
            rejection_reasons[obj_id] = flags
            continue

        valid_corners_map[obj_id] = corners

    return valid_corners_map, rejection_reasons


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
    _init_runtime_feature_metrics(metrics, pipeline_cfg=pipeline_cfg, prompts=prompts)

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
    geometry_engine = None
    geometry_cfg = pipeline_cfg.get("geometry")
    if _geometry_enabled(pipeline_cfg):
        geometry_engine = court_geometry_mod.GeometryFittingEngine(
            config=geometry_cfg,
            prompts=prompts,
            fallback_fitter=fitter,
            fitter_params=fitter_params,
        )
        corners_map, _rejection_reasons = geometry_engine.fit_frame(
            frame_idx=0,
            frame_bgr=frame,
            masks_by_obj={
                int(obj_id): np.asarray(mask).squeeze() for obj_id, mask in masks.items()
            },
            frame_shape=frame.shape[:2],
        )
    else:
        corners_map = _fit_corners(masks, fitter, fitter_params)

    # Enlarge tiny fitted quads to cover the prompt point bounding box.
    # Same logic as video_hybrid mode — the hull fitter often produces
    # quads much smaller than the panel because SAM masks are partial.
    prompt_bboxes: dict[int, np.ndarray] = {}
    for prompt in prompts:
        pts = prompt.points
        x0p, y0p = pts.min(axis=0)
        x1p, y1p = pts.max(axis=0)
        prompt_bboxes[prompt.obj_id] = np.array(
            [[x0p, y0p], [x1p, y0p], [x1p, y1p], [x0p, y1p]], dtype=np.float32
        )
    if prompt_bboxes:
        bbox_heights = {oid: float(b[2, 1] - b[0, 1]) for oid, b in prompt_bboxes.items()}
        ref_height = float(np.median(list(bbox_heights.values())))
        for oid in prompt_bboxes:
            bbox = prompt_bboxes[oid]
            h = bbox_heights[oid]
            if h > 0 and abs(h - ref_height) > 2:
                cy = (bbox[0, 1] + bbox[2, 1]) / 2
                bbox[0, 1] = bbox[1, 1] = cy - ref_height / 2
                bbox[2, 1] = bbox[3, 1] = cy + ref_height / 2
                prompt_bboxes[oid] = bbox
    for obj_id in list(corners_map.keys()):
        if obj_id in prompt_bboxes:
            fitted = corners_map[obj_id]
            bbox = prompt_bboxes[obj_id]
            fitted_area = cv2.contourArea(fitted.astype(np.float32))
            bbox_area = cv2.contourArea(bbox)
            if bbox_area > 0 and fitted_area < bbox_area * 0.5:
                corners_map[obj_id] = bbox
                print(f"[pipeline] obj {obj_id}: enlarged {fitted_area:.0f}→{bbox_area:.0f}px²")

    metrics["fit_s"] = time.perf_counter() - t0
    if geometry_engine is not None:
        geometry_metrics = geometry_engine.finalize_metrics()
        geometry_metrics["geometry_total_s"] = round(metrics["fit_s"], 4)
        metrics.update(geometry_metrics)
        metrics["geometry_runtime_enabled"] = bool(metrics.get("geometry_runtime_enabled"))
    _require_runtime_feature_metrics(
        feature_name="geometry",
        metrics=metrics,
        config_enabled=bool(metrics["geometry_config_enabled"]),
        runtime_enabled=bool(metrics["geometry_runtime_enabled"]),
        required_keys=[
            "geometry_total_s",
            "geometry_active_objects",
            "object_geometry_model",
        ],
    )
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
    geometry_enabled = _geometry_enabled(pipeline_cfg)
    active_prompts = [
        prompt
        for prompt in prompts
        if _is_supported_banner_surface(prompt, geometry_enabled=geometry_enabled)
    ]
    if active_prompts:
        video_segmenter = build_video_segmenter(pipeline_cfg["segmenter"])
        if not hasattr(video_segmenter, "preview_frame"):
            raise RuntimeError("SAM3 image preview requires the sam3_video segmenter.")
        frame, masks, preview_frame_idx, prompt_diagnostics = video_segmenter.preview_frame(
            video_path,
            active_prompts,
        )
    else:
        preview_frame_idx = _preview_frame_idx_from_prompts(prompts)
        frame = load_frame(video_path, frame_idx=preview_frame_idx)
        masks = {}
        prompt_diagnostics = {}
    metrics["segment_s"] = time.perf_counter() - t0
    metrics["num_prompts"] = len(prompts)
    metrics["num_prompt_points"] = sum(len(prompt.points) for prompt in prompts)
    metrics["video_path"] = video_path
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["checkpoint"] = pipeline_cfg["segmenter"].get("checkpoint", "")
    metrics["frame_height"], metrics["frame_width"] = frame.shape[:2]
    metrics["preview_frame_idx"] = preview_frame_idx
    _init_runtime_feature_metrics(metrics, pipeline_cfg=pipeline_cfg, prompts=prompts)

    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})
    geometry_engine = None
    geometry_cfg = pipeline_cfg.get("geometry")
    if geometry_enabled:
        geometry_engine = court_geometry_mod.GeometryFittingEngine(
            config=geometry_cfg,
            prompts=prompts,
            fallback_fitter=fitter,
            fitter_params=fitter_params,
        )
    t0 = time.perf_counter()
    corners_map, preview_object_diagnostics, invalid_objects = _build_preview_diagnostics(
        frame=frame,
        prompts=prompts,
        masks=masks,
        prompt_diagnostics=prompt_diagnostics,
        fitter=fitter,
        fitter_params=fitter_params,
        geometry_engine=geometry_engine,
    )
    metrics["fit_s"] = time.perf_counter() - t0
    if geometry_engine is not None:
        geometry_metrics = geometry_engine.finalize_metrics()
        geometry_metrics["geometry_total_s"] = round(metrics["fit_s"], 4)
        metrics.update(geometry_metrics)
        metrics["geometry_runtime_enabled"] = bool(metrics.get("geometry_runtime_enabled"))
    _require_runtime_feature_metrics(
        feature_name="geometry",
        metrics=metrics,
        config_enabled=bool(metrics["geometry_config_enabled"]),
        runtime_enabled=bool(metrics["geometry_runtime_enabled"]),
        required_keys=[
            "geometry_total_s",
            "geometry_active_objects",
            "object_geometry_model",
            "geometry_fit_method_counts",
        ],
    )
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
    if geometry_engine is not None and hasattr(geometry_engine, "render_debug_overlay"):
        preview_artifacts["preview_geometry"] = geometry_engine.render_debug_overlay(frame)
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


def _init_validity_metrics(
    prompts: list[ObjectPrompt],
) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    obj_ids = [str(int(prompt.obj_id)) for prompt in prompts]
    return (
        {obj_id: 0 for obj_id in obj_ids},
        {obj_id: 0 for obj_id in obj_ids},
        {obj_id: {} for obj_id in obj_ids},
    )


def _record_frame_rejections(
    rejection_reasons: dict[int, list[str]],
    object_rejection_counts: dict[str, int],
    object_rejection_reasons: dict[str, dict[str, int]],
) -> None:
    for obj_id, reasons in rejection_reasons.items():
        key = str(obj_id)
        object_rejection_counts[key] = object_rejection_counts.get(key, 0) + 1
        reason_counts = object_rejection_reasons.setdefault(key, {})
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1


def _finalize_valid_frame_coverage(
    valid_frame_counts: dict[str, int],
    object_rejection_counts: dict[str, int],
    object_rejection_reasons: dict[str, dict[str, int]],
    *,
    num_frames: int,
) -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, dict[str, int]]]:
    object_valid_frame_coverage = {
        obj_id: {
            "frames_valid": frames_valid,
            "coverage_ratio": round(frames_valid / max(num_frames, 1), 4),
        }
        for obj_id, frames_valid in valid_frame_counts.items()
    }
    return object_valid_frame_coverage, object_rejection_counts, object_rejection_reasons


def _raise_video_coverage_error(
    reason: str,
    *,
    num_prompts: int,
    frames_with_masks: int,
    frames_with_valid_objects: int,
    frames_with_quads: int,
    frames_composited: int,
    object_masks_total: int,
) -> None:
    raise RuntimeError(
        f"{reason} Coverage: "
        f"num_prompts={num_prompts}, "
        f"frames_with_masks={frames_with_masks}, "
        f"frames_with_valid_objects={frames_with_valid_objects}, "
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
    geometry_enabled = _geometry_enabled(pipeline_cfg)
    active_prompts = [
        prompt
        for prompt in prompts
        if segmenter_type != "sam3_video"
        or _is_supported_banner_surface(prompt, geometry_enabled=geometry_enabled)
    ]

    # --- Input video info ---
    input_fps = get_video_fps(video_path)
    metrics["input_fps"] = input_fps
    metrics["num_prompts"] = len(prompts)
    metrics["num_prompt_points"] = sum(len(p.points) for p in prompts)
    metrics["video_path"] = video_path
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["checkpoint"] = pipeline_cfg["segmenter"].get("checkpoint", "")
    _init_runtime_feature_metrics(metrics, pipeline_cfg=pipeline_cfg, prompts=prompts)

    # Read frame size from the first frame.
    first_frame = load_frame(video_path, frame_idx=0)
    metrics["frame_height"], metrics["frame_width"] = first_frame.shape[:2]
    if segmenter_type == "sam3_video" and not active_prompts:
        raise RuntimeError(
            "No supported banner prompts remain after filtering unsupported surface types. "
            f"Prompt obj_ids={[int(prompt.obj_id) for prompt in prompts]}."
        )

    # --- Segment + track across all frames ---
    t0 = time.perf_counter()
    video_segmenter = build_video_segmenter(pipeline_cfg["segmenter"])
    video_segments, frame_dir, frame_names = video_segmenter.segment_video(
        video_path,
        active_prompts,
    )
    stabilization_metrics: dict[str, Any] = {}
    stabilization_cfg = pipeline_cfg.get("stabilization")
    if stabilization_cfg:
        video_segments, stabilization_metrics = stabilization_mod.stabilize_video_segments(
            frame_dir=frame_dir,
            frame_names=frame_names,
            video_segments=video_segments,
            tracked_obj_ids=sorted({int(prompt.obj_id) for prompt in active_prompts}),
            config=stabilization_cfg,
        )
    metrics["segment_total_s"] = time.perf_counter() - t0
    metrics["num_frames"] = len(frame_names)
    metrics["duration_s"] = round(len(frame_names) / input_fps, 2)
    metrics.update(stabilization_metrics)
    if stabilization_metrics:
        metrics["stabilization_runtime_enabled"] = True
    _require_runtime_feature_metrics(
        feature_name="stabilization",
        metrics=metrics,
        config_enabled=bool(metrics["stabilization_config_enabled"]),
        runtime_enabled=bool(metrics["stabilization_runtime_enabled"]),
        required_keys=[
            "stabilization_total_s",
            "stabilization_static_frame_ratio",
            "stabilization_object_stats",
        ],
    )
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
    frames_with_valid_objects = 0
    frames_with_quads = 0
    frames_composited = 0
    metrics["frames_with_masks"] = frames_with_masks
    metrics["frames_with_valid_objects"] = frames_with_valid_objects
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
            frames_with_valid_objects=frames_with_valid_objects,
            frames_with_quads=frames_with_quads,
            frames_composited=frames_composited,
            object_masks_total=object_masks_total,
        )

    # --- Per-frame: fit + composite ---
    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})
    geometry_engine = None
    geometry_cfg = pipeline_cfg.get("geometry")
    if geometry_enabled:
        geometry_engine = court_geometry_mod.GeometryFittingEngine(
            config=geometry_cfg,
            prompts=prompts,
            fallback_fitter=fitter,
            fitter_params=fitter_params,
        )

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
    # EMA smoothing of fitted corners to eliminate per-frame jitter from
    # slightly-different SAM2 masks. Alpha=0.3 matches Raghav's CornerTracker.
    ema_alpha = pipeline_cfg.get("tracking", {}).get("ema_alpha", 0.3)
    smoothed_corners: dict[int, np.ndarray] = {}
    valid_frame_counts, object_rejection_counts, object_rejection_reasons = _init_validity_metrics(
        prompts
    )

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

            # Fit and validate quads for this frame.
            t_fit = time.perf_counter()
            corners_map, rejection_reasons = _fit_and_validate_video_objects(
                prompts=prompts,
                masks_by_obj=masks_2d,
                frame_shape=frame_bgr.shape[:2],
                fitter=fitter,
                fitter_params=fitter_params,
                geometry_engine=geometry_engine,
                frame_idx=frame_idx,
                frame_bgr=frame_bgr,
            )
            fit_times.append(time.perf_counter() - t_fit)
            _record_frame_rejections(
                rejection_reasons,
                object_rejection_counts,
                object_rejection_reasons,
            )
            if corners_map:
                frames_with_valid_objects += 1
                frames_with_quads += 1
                for obj_id in corners_map:
                    valid_frame_counts[str(obj_id)] = valid_frame_counts.get(str(obj_id), 0) + 1

            # EMA-smooth corners to eliminate per-frame jitter.
            for obj_id, corners in corners_map.items():
                if obj_id in smoothed_corners:
                    smoothed = ema_alpha * corners + (1 - ema_alpha) * smoothed_corners[obj_id]
                    corners_map[obj_id] = smoothed
                    smoothed_corners[obj_id] = smoothed
                else:
                    smoothed_corners[obj_id] = corners.copy()

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

    metrics["frames_with_valid_objects"] = frames_with_valid_objects
    metrics["frames_with_quads"] = frames_with_quads
    metrics["frames_composited"] = frames_composited
    if geometry_engine is not None:
        geometry_metrics = geometry_engine.finalize_metrics()
        geometry_metrics["geometry_total_s"] = round(sum(fit_times), 4)
        metrics.update(geometry_metrics)
        metrics["geometry_runtime_enabled"] = bool(metrics.get("geometry_runtime_enabled"))
    _require_runtime_feature_metrics(
        feature_name="geometry",
        metrics=metrics,
        config_enabled=bool(metrics["geometry_config_enabled"]),
        runtime_enabled=bool(metrics["geometry_runtime_enabled"]),
        required_keys=[
            "geometry_total_s",
            "geometry_active_objects",
            "object_geometry_model",
            "geometry_fit_method_counts",
        ],
    )
    (
        metrics["object_valid_frame_coverage"],
        metrics["object_rejection_counts"],
        metrics["object_rejection_reasons"],
    ) = _finalize_valid_frame_coverage(
        valid_frame_counts,
        object_rejection_counts,
        object_rejection_reasons,
        num_frames=len(frame_names),
    )
    metrics["write_video_s"] = round(write_video_s, 4)
    print(f"[video] Wrote {num_written} frames → {output_path}")

    if prompts and frames_with_valid_objects == 0:
        _raise_video_coverage_error(
            "No semantically valid tracked banner objects were produced for this video run.",
            num_prompts=len(prompts),
            frames_with_masks=frames_with_masks,
            frames_with_valid_objects=frames_with_valid_objects,
            frames_with_quads=frames_with_quads,
            frames_composited=frames_composited,
            object_masks_total=object_masks_total,
        )

    if overlay is not None and frames_composited == 0:
        _raise_video_coverage_error(
            "No frames were composited despite propagated masks and a configured logo.",
            num_prompts=len(prompts),
            frames_with_masks=frames_with_masks,
            frames_with_valid_objects=frames_with_valid_objects,
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
    metrics["output_fps"] = (
        round(len(frame_names) / metrics["total_s"], 2) if metrics["total_s"] > 0 else 0.0
    )

    # Per-stage breakdown from _perf timers (empty dict if profiling disabled).
    if _perf.PERF_ENABLED:
        metrics["composite_breakdown_ms"] = _perf.snapshot_ms(divisor=len(frame_names))

    return {
        "output_path": output_path,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Tracking-based video pipeline (SAM2 image on frame 0 + optical flow)
# ---------------------------------------------------------------------------


def run_pipeline_video_tracking(
    config: dict,
    output_path: str = "output.mp4",
    config_path: str | None = None,
) -> dict:
    """Video pipeline using corner tracking instead of per-frame segmentation.

    1. SAM2 image predictor segments frame 0 only.
    2. PCA fitter extracts initial quad corners from frame 0's masks.
    3. CornerTracker propagates corners via Lucas-Kanade optical flow + EMA.
    4. Compositor runs per frame with tracked corners.

    This is faster and produces more temporally stable output than
    ``run_pipeline_video`` (which re-segments and re-fits every frame).
    """
    import os

    metrics: dict[str, Any] = {}
    pipeline_cfg = config["pipeline"]
    input_cfg = config["input"]
    video_path = input_cfg["video"]

    # --- Get prompts ---
    prompts_cfg = input_cfg.get("prompts")
    if prompts_cfg:
        prompts = _prompts_from_config(prompts_cfg)
    else:
        frame0 = load_frame(video_path)
        prompts = collect_clicks(frame0)
        if not prompts:
            return {"output_path": None, "metrics": metrics}
        if config_path:
            _save_prompts_to_config(config, prompts, config_path)

    input_fps = get_video_fps(video_path)
    metrics["input_fps"] = input_fps
    metrics["num_prompts"] = len(prompts)
    metrics["num_prompt_points"] = sum(len(p.points) for p in prompts)
    metrics["video_path"] = video_path
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["checkpoint"] = pipeline_cfg["segmenter"].get("checkpoint", "")
    metrics["tracking_mode"] = True

    # --- Step 1: Segment frame 0 with SAM2 image predictor ---
    t0 = time.perf_counter()
    frame0 = load_frame(video_path, frame_idx=0)
    metrics["frame_height"], metrics["frame_width"] = frame0.shape[:2]
    segmenter = build_segmenter(pipeline_cfg["segmenter"])
    masks_frame0 = segmenter.segment(frame0, prompts)
    metrics["segment_total_s"] = time.perf_counter() - t0
    n_obj = len(masks_frame0)
    seg_t = metrics["segment_total_s"]
    print(f"[tracking] Segmented frame 0: {n_obj} objects in {seg_t:.2f}s")

    # --- Step 2: Fit quads on frame 0 ---
    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})
    corners_map: dict[int, np.ndarray] = {}
    for obj_id, mask in masks_frame0.items():
        mask_2d = mask.squeeze()
        corners = fitter.fit(mask_2d, **fitter_params)
        if corners is not None:
            corners_map[obj_id] = corners

    if not corners_map:
        print("[tracking] No quads fitted on frame 0")
        return {"output_path": None, "metrics": metrics}

    # --- Step 3: Initialize CornerTracker ---
    _tracking_cfg = pipeline_cfg.get("tracking", {})
    ema_alpha = _tracking_cfg.get("ema_alpha", 0.3)
    fb_threshold = _tracking_cfg.get("fb_threshold", 2.0)
    lk_win = _tracking_cfg.get("lk_win_size", 21)
    tracker = CornerTracker(ema_alpha=ema_alpha, fb_threshold=fb_threshold, lk_win_size=lk_win)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    for obj_id, corners in corners_map.items():
        tracker.init(obj_id, corners, gray0)

    # --- Step 4: Extract frames and process with tracking ---
    import shutil
    import tempfile

    from banner_pipeline.io import extract_all_frames

    frame_dir = tempfile.mkdtemp(prefix="tracking_frames_")
    frame_names = extract_all_frames(video_path, frame_dir)
    metrics["num_frames"] = len(frame_names)
    metrics["duration_s"] = round(len(frame_names) / input_fps, 2)
    print(f"[tracking] {len(frame_names)} frames extracted")

    overlay = None
    logo_path = input_cfg.get("logo")
    if logo_path:
        overlay = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise RuntimeError(f"Could not read logo: {logo_path}")

    compositor = build_compositor(pipeline_cfg["compositor"]) if overlay is not None else None
    compositor_params = pipeline_cfg["compositor"].get("params", {}) if overlay is not None else {}
    focal_length = pipeline_cfg.get("camera", {}).get("focal_length")

    composite_times: list[float] = []
    tracking_times: list[float] = []

    _perf.reset()

    fh, fw = frame0.shape[:2]
    video_writer = StreamingVideoWriter(output_path, fw, fh, fps=input_fps)
    write_video_s = 0.0
    num_written = 0

    try:
        for frame_idx, fname in enumerate(frame_names):
            frame_bgr = cv2.imread(os.path.join(frame_dir, fname))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read frame {frame_idx}: {fname}")

            # Track corners (frame 0 uses initial corners, 1+ uses flow).
            t_track = time.perf_counter()
            if frame_idx == 0:
                current_corners = corners_map
            else:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                current_corners = tracker.update(gray)
            tracking_times.append(time.perf_counter() - t_track)

            # Composite.
            if overlay is not None and compositor is not None and current_corners:
                t_comp = time.perf_counter()
                K = estimate_camera_matrix(frame_bgr.shape, focal_length=focal_length)
                for obj_id in sorted(current_corners):
                    extra_kw = dict(compositor_params)
                    if compositor.name == "alpha":
                        homo = compute_oriented_homography(current_corners[obj_id], K)
                        extra_kw["homo"] = homo
                    # Reuse frame 0's mask for all frames. The mask shape is
                    # stable across frames (banners don't move much), and the
                    # compositor needs it to inpaint the old logo away.
                    frame_mask = masks_frame0.get(obj_id)
                    if frame_mask is not None:
                        frame_mask = frame_mask.squeeze()
                    frame_bgr = compositor.composite(
                        frame_bgr,
                        current_corners[obj_id],
                        overlay,
                        mask=frame_mask,
                        **extra_kw,
                    )
                composite_times.append(time.perf_counter() - t_comp)

            t_write = time.perf_counter()
            video_writer.write(frame_bgr)
            num_written += 1
            write_video_s += time.perf_counter() - t_write

            if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_names) - 1:
                print(f"[tracking] Processed frame {frame_idx + 1}/{len(frame_names)}")

    finally:
        video_writer.close()
        shutil.rmtree(frame_dir, ignore_errors=True)

    metrics["write_video_s"] = round(write_video_s, 4)
    print(f"[tracking] Wrote {num_written} frames -> {output_path}")

    # Aggregate metrics.
    if tracking_times:
        track_arr = np.array(tracking_times) * 1000
        metrics["tracking_mean_ms"] = round(float(track_arr.mean()), 2)
    if composite_times:
        comp_arr = np.array(composite_times) * 1000
        metrics["composite_mean_ms"] = round(float(comp_arr.mean()), 2)

    metrics["total_s"] = round(
        metrics["segment_total_s"] + sum(tracking_times) + sum(composite_times) + write_video_s,
        4,
    )
    metrics["output_fps"] = round(len(frame_names) / metrics["total_s"], 2)

    if _perf.PERF_ENABLED:
        metrics["composite_breakdown_ms"] = _perf.snapshot_ms(divisor=len(frame_names))

    return {"output_path": output_path, "metrics": metrics}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run_pipeline_video_hybrid(
    config: dict,
    output_path: str = "output.mp4",
    config_path: str | None = None,
) -> dict:
    """Hybrid video pipeline: SAM masks for inpainting + CornerTracker for placement.

    Combines the best of both approaches:
    - SAM2/SAM3 video propagation for per-frame masks (correct inpainting, no drift)
    - CornerTracker optical flow for smooth logo placement (no jitter)

    Flow:
    1. SAM2/SAM3 video propagation produces per-frame masks
    2. PCA fitter runs on frame 0 to get initial corners
    3. CornerTracker propagates corners via optical flow + EMA
    4. Per frame: use SAM mask for inpainting, use tracked corners for logo warp
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
        log_prefix="[hybrid]",
        frame_idx=0,
    )
    if not prompts:
        return {"output_path": None, "metrics": metrics}

    geometry_enabled = _geometry_enabled(pipeline_cfg)
    active_prompts = [
        prompt
        for prompt in prompts
        if segmenter_type != "sam3_video"
        or _is_supported_banner_surface(prompt, geometry_enabled=geometry_enabled)
    ]

    input_fps = get_video_fps(video_path)
    metrics["input_fps"] = input_fps
    metrics["num_prompts"] = len(prompts)
    metrics["video_path"] = video_path
    metrics["fitter_type"] = pipeline_cfg["fitter"]["type"]
    metrics["compositor_type"] = pipeline_cfg["compositor"]["type"]
    metrics["mode"] = "video_hybrid"
    _init_runtime_feature_metrics(metrics, pipeline_cfg=pipeline_cfg, prompts=prompts)

    first_frame = load_frame(video_path, frame_idx=0)
    metrics["frame_height"], metrics["frame_width"] = first_frame.shape[:2]

    # --- Step 1: SAM video propagation for per-frame masks ---
    t0 = time.perf_counter()
    video_segmenter = build_video_segmenter(pipeline_cfg["segmenter"])
    video_segments, frame_dir, frame_names = video_segmenter.segment_video(
        video_path,
        active_prompts,
    )

    # Optional stabilization
    stabilization_metrics: dict[str, Any] = {}
    stabilization_cfg = pipeline_cfg.get("stabilization")
    if stabilization_cfg:
        video_segments, stabilization_metrics = stabilization_mod.stabilize_video_segments(
            frame_dir=frame_dir,
            frame_names=frame_names,
            video_segments=video_segments,
            tracked_obj_ids=sorted({int(p.obj_id) for p in active_prompts}),
            config=stabilization_cfg,
        )

    metrics["segment_total_s"] = time.perf_counter() - t0
    metrics["num_frames"] = len(frame_names)
    metrics["duration_s"] = round(len(frame_names) / input_fps, 2)
    metrics.update(stabilization_metrics)
    print(f"[hybrid] Tracked {len(frame_names)} frames in {metrics['segment_total_s']:.2f}s")

    # --- Step 2: Fit quads on frame 0 + init CornerTracker ---
    fitter = build_fitter(pipeline_cfg["fitter"])
    fitter_params = pipeline_cfg["fitter"].get("params", {})

    frame0_bgr = cv2.imread(os.path.join(frame_dir, frame_names[0]))
    if frame0_bgr is None:
        raise RuntimeError(f"Could not read frame 0: {frame_names[0]}")

    masks_frame0 = video_segments.get(0, {})
    masks_2d_frame0 = {oid: m.squeeze() for oid, m in masks_frame0.items()}
    corners_frame0: dict[int, np.ndarray] = {}
    for obj_id, mask_2d in masks_2d_frame0.items():
        corners = fitter.fit(mask_2d, **fitter_params)
        if corners is not None:
            corners_frame0[obj_id] = corners

    # Enlarge fitted corners to cover the prompt point bounding box.
    # The hull fitter often produces quads much smaller than the actual
    # banner panel because the SAM mask only covers part of the content.
    prompt_by_id: dict[int, ObjectPrompt] = {p.obj_id: p for p in prompts}
    prompt_bboxes: dict[int, np.ndarray] = {}
    for prompt in prompts:
        if prompt.placement_quad is not None:
            # Explicit placement quad.
            prompt_bboxes[prompt.obj_id] = prompt.placement_quad.copy()
        elif prompt.box is not None:
            # Derive from SAM box parameter [x0, y0, x1, y1].
            b = prompt.box
            prompt_bboxes[prompt.obj_id] = np.array(
                [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                dtype=np.float32,
            )
        else:
            # Derive from click points: axis-aligned bounding box.
            pts = prompt.points
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            prompt_bboxes[prompt.obj_id] = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
            )
    # Normalize bbox heights for banner surfaces only.
    banner_ids = {
        oid
        for oid, p in prompt_by_id.items()
        if _normalize_surface_type(p.surface_type) == "banner"
    }
    if banner_ids:
        banner_heights = {
            oid: float(prompt_bboxes[oid][2, 1] - prompt_bboxes[oid][0, 1])
            for oid in banner_ids
            if oid in prompt_bboxes
        }
        if banner_heights:
            ref_height = float(np.median(list(banner_heights.values())))
            for oid in banner_ids:
                if oid not in prompt_bboxes:
                    continue
                bbox = prompt_bboxes[oid]
                h = banner_heights.get(oid, 0.0)
                if h > 0 and abs(h - ref_height) > 2:
                    cy = (bbox[0, 1] + bbox[2, 1]) / 2
                    bbox[0, 1] = bbox[1, 1] = cy - ref_height / 2
                    bbox[2, 1] = bbox[3, 1] = cy + ref_height / 2
                    prompt_bboxes[oid] = bbox

    for obj_id in list(corners_frame0.keys()):
        if obj_id in prompt_bboxes:
            fitted = corners_frame0[obj_id]
            bbox = prompt_bboxes[obj_id]
            # Use the prompt bbox if the fitted quad is much smaller
            fitted_area = cv2.contourArea(fitted.astype(np.float32))
            bbox_area = cv2.contourArea(bbox)
            if bbox_area > 0 and fitted_area < bbox_area * 0.5:
                corners_frame0[obj_id] = bbox
                print(f"[hybrid] obj {obj_id}: enlarged {fitted_area:.0f}→{bbox_area:.0f}px²")

    _tracking_cfg = pipeline_cfg.get("tracking", {})
    ema_alpha = _tracking_cfg.get("ema_alpha", 0.3)
    corner_source = _tracking_cfg.get("corner_source", "bbox")

    if corner_source in ("bbox", "mask_offset", "homography_offset"):
        # Use prompt bboxes for sizing. "bbox" keeps them static;
        # "mask_offset" shifts via SAM mask centroids;
        # "homography_offset" warps via accumulated frame-to-frame homography.
        tracker = None
        static_corners = {oid: bbox.copy() for oid, bbox in prompt_bboxes.items()}
        for oid, c in corners_frame0.items():
            if oid not in static_corners:
                static_corners[oid] = c.astype(np.float32).copy()
        if corner_source == "homography_offset":
            _homo_prev_gray = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
            _homo_accum = np.eye(3, dtype=np.float64)
            _homo_ema: np.ndarray | None = None
            _homo_base_pts = {
                oid: c.reshape(-1, 1, 2).astype(np.float64) for oid, c in static_corners.items()
            }
            _cut_thresh = _tracking_cfg.get("cut_threshold", 30.0)
            n_obj = len(static_corners)
            print(f"[hybrid] homography_offset {n_obj} objects, ema={ema_alpha}")
        elif corner_source == "mask_offset":
            # Compute frame-0 mask bboxes as reference for tracking
            # both translation and scale (handles camera zoom).
            _mask_ref: dict[int, tuple[float, float, float, float]] = {}
            for oid, mask_2d in masks_2d_frame0.items():
                ys, xs = np.where(mask_2d > 0)
                if len(xs) > 10:
                    _mask_ref[oid] = (
                        float(xs.mean()),
                        float(ys.mean()),
                        float(xs.max() - xs.min()),
                        float(ys.max() - ys.min()),
                    )
            # Frame center for zoom scaling.
            _frame_cx = frame0_bgr.shape[1] / 2.0
            _frame_cy = frame0_bgr.shape[0] / 2.0
            n_obj = len(static_corners)
            print(f"[hybrid] mask_offset {n_obj} objects, ema={ema_alpha}")
        else:
            print(f"[hybrid] Using static bbox corners for {len(static_corners)} objects")
    elif corner_source == "mask_refit":
        # Re-fit from per-frame SAM masks + bbox fallback + EMA.
        tracker = None
        static_corners = None
        smoothed_corners: dict[int, np.ndarray] = {
            oid: c.astype(np.float32).copy() for oid, c in corners_frame0.items()
        }
        print(
            f"[hybrid] Using mask_refit corners with {len(corners_frame0)} objects, ema={ema_alpha}"
        )
    elif corner_source == "optical_flow":
        # Legacy optical flow tracking (can drift on textureless surfaces).
        static_corners = None
        fb_threshold = _tracking_cfg.get("fb_threshold", 2.0)
        lk_win = _tracking_cfg.get("lk_win_size", 21)
        tracker = CornerTracker(ema_alpha=ema_alpha, fb_threshold=fb_threshold, lk_win_size=lk_win)
        gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
        for obj_id, corners in corners_frame0.items():
            tracker.init(obj_id, corners, gray0)
        n_obj = len(corners_frame0)
        print(f"[hybrid] CornerTracker (optical_flow) {n_obj} objects, ema={ema_alpha}")
    else:
        raise ValueError(f"Unknown corner_source: {corner_source!r}")

    # --- Step 2b: Initialize person masker for occlusion ---
    _person_masker = None
    _occlusion_cfg = pipeline_cfg.get("occlusion_masker", {})
    if _occlusion_cfg.get("type") == "person":
        from banner_pipeline.masking import PersonMasker

        _person_masker = PersonMasker(
            confidence_threshold=_occlusion_cfg.get("confidence_threshold", 0.5),
            device=_occlusion_cfg.get("device"),
        )
        print("[hybrid] PersonMasker enabled for occlusion")

    # --- Step 3: Per-frame composite with SAM masks + tracked corners ---
    overlay = None
    logo_path = input_cfg.get("logo")
    compositor_params = pipeline_cfg["compositor"].get("params", {})
    _erase_only = compositor_params.get("erase_only", False)
    if logo_path:
        overlay = _load_overlay(logo_path)
    elif _erase_only:
        overlay = np.zeros((1, 1, 3), dtype=np.uint8)  # dummy for erase-only

    compositor = build_compositor(pipeline_cfg["compositor"]) if overlay is not None else None
    _surface_compositors: dict[str, Any] = {}  # per-surface instances
    if compositor is not None and not compositor_params:
        compositor_params = pipeline_cfg["compositor"].get("params", {})
    focal_length = pipeline_cfg.get("camera", {}).get("focal_length")

    composite_times: list[float] = []
    tracking_times: list[float] = []
    write_video_s = 0.0
    num_written = 0

    _perf.reset()

    fh, fw = frame0_bgr.shape[:2]
    video_writer = StreamingVideoWriter(output_path, fw, fh, fps=input_fps)

    try:
        for frame_idx, fname in enumerate(frame_names):
            if frame_idx == 0:
                frame_bgr = frame0_bgr
            else:
                frame_bgr = cv2.imread(os.path.join(frame_dir, fname))
                if frame_bgr is None:
                    raise RuntimeError(f"Could not read frame {frame_idx}: {fname}")

            # Get SAM masks for this frame (for inpainting).
            masks_for_frame = video_segments.get(frame_idx, {})
            masks_2d = {oid: m.squeeze() for oid, m in masks_for_frame.items()}

            # Get corners for this frame (for logo placement).
            t_track = time.perf_counter()
            if corner_source == "homography_offset":
                if frame_idx == 0:
                    current_corners = static_corners
                else:
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(_homo_prev_gray, gray)
                    if float(diff.mean()) > _cut_thresh:
                        _homo_accum = np.eye(3, dtype=np.float64)
                        _homo_ema = None
                    else:
                        H_pair, _, ok = stabilization_mod._estimate_pair_homography(
                            _homo_prev_gray, gray, (fh, fw)
                        )
                        if ok:
                            H = H_pair.astype(np.float64)
                            if abs(H[2, 2]) > 1e-9:
                                H /= H[2, 2]
                            _homo_accum = H @ _homo_accum
                        if _homo_ema is None:
                            _homo_ema = _homo_accum.copy()
                        else:
                            blended = ema_alpha * _homo_accum + (1 - ema_alpha) * _homo_ema
                            if abs(blended[2, 2]) > 1e-9:
                                blended = blended / blended[2, 2]
                            _homo_ema = blended
                    _homo_prev_gray = gray
                    H_use = _homo_ema if _homo_ema is not None else _homo_accum
                    current_corners = {}
                    for oid, base_pts in _homo_base_pts.items():
                        warped = cv2.perspectiveTransform(base_pts, H_use)
                        current_corners[oid] = warped.reshape(4, 2).astype(np.float32)
            elif corner_source == "mask_offset":
                # Shift + scale static bboxes using SAM mask bbox changes.
                assert static_corners is not None
                if frame_idx == 0:
                    current_corners = static_corners
                else:
                    dxs: list[float] = []
                    dys: list[float] = []
                    sxs: list[float] = []
                    for oid in static_corners:
                        if oid not in _mask_ref or oid not in masks_2d:
                            continue
                        m = masks_2d[oid]
                        ys, xs = np.where(m > 0)
                        if len(xs) < 10:
                            continue
                        ref_cx, ref_cy, ref_w, ref_h = _mask_ref[oid]
                        cur_cx = float(xs.mean())
                        cur_cy = float(ys.mean())
                        cur_w = float(xs.max() - xs.min())
                        dxs.append(cur_cx - ref_cx)
                        dys.append(cur_cy - ref_cy)
                        if ref_w > 5:
                            sxs.append(cur_w / ref_w)
                    if dxs:
                        med_dx = float(np.median(dxs))
                        med_dy = float(np.median(dys))
                        med_sx = float(np.median(sxs)) if sxs else 1.0
                        current_corners = {}
                        for oid, base in static_corners.items():
                            # Scale around frame center, then translate.
                            scaled = (base - [_frame_cx, _frame_cy]) * med_sx
                            shifted = scaled + [_frame_cx + med_dx, _frame_cy + med_dy]
                            current_corners[oid] = shifted.astype(np.float32)
                    else:
                        current_corners = static_corners
            elif static_corners is not None:
                # bbox mode: same corners every frame.
                current_corners = static_corners
            elif tracker is not None:
                # optical_flow mode.
                if frame_idx == 0:
                    current_corners = corners_frame0
                else:
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    current_corners = tracker.update(gray)
            else:
                # mask_refit mode.
                if frame_idx == 0:
                    current_corners = corners_frame0
                else:
                    raw_corners: dict[int, np.ndarray] = {}
                    for obj_id, mask_2d_frame in masks_2d.items():
                        refit = fitter.fit(mask_2d_frame, **fitter_params)
                        if refit is not None:
                            raw_corners[obj_id] = refit

                    for obj_id in list(raw_corners.keys()):
                        if obj_id in prompt_bboxes:
                            fitted = raw_corners[obj_id]
                            bbox = prompt_bboxes[obj_id]
                            fitted_area = cv2.contourArea(fitted.astype(np.float32))
                            bbox_area = cv2.contourArea(bbox)
                            if bbox_area > 0 and fitted_area < bbox_area * 0.5:
                                raw_corners[obj_id] = bbox.copy()

                    for obj_id in smoothed_corners:
                        if obj_id not in raw_corners and obj_id in prompt_bboxes:
                            raw_corners[obj_id] = prompt_bboxes[obj_id].copy()

                    for obj_id, rc in raw_corners.items():
                        rc_f = rc.astype(np.float32)
                        if obj_id in smoothed_corners:
                            smoothed_corners[obj_id] = (
                                ema_alpha * rc_f + (1 - ema_alpha) * smoothed_corners[obj_id]
                            )
                        else:
                            smoothed_corners[obj_id] = rc_f
                    current_corners = {oid: c.copy() for oid, c in smoothed_corners.items()}
            tracking_times.append(time.perf_counter() - t_track)

            # Detect person occlusion mask (once per frame).
            person_mask: np.ndarray | None = None
            if _person_masker is not None:
                person_mask = _person_masker.mask(frame_bgr)
                # Small dilation to cover racket/limb edges the model misses.
                occ_dilate = int(_occlusion_cfg.get("mask_dilate_px", 3))
                if occ_dilate > 0 and np.any(person_mask > 0):
                    kern = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * occ_dilate + 1, 2 * occ_dilate + 1),
                    )
                    person_mask = cv2.dilate(person_mask, kern, iterations=1)

            # Composite: SAM mask for inpaint, tracked corners for logo warp.
            if overlay is not None and compositor is not None and current_corners:
                t_comp = time.perf_counter()
                K = estimate_camera_matrix(frame_bgr.shape, focal_length=focal_length)
                surface_overrides = pipeline_cfg["compositor"].get("surface_overrides", {})
                for obj_id in sorted(current_corners):
                    prompt = prompt_by_id.get(obj_id)  # type: ignore[assignment]
                    st = (
                        _normalize_surface_type(prompt.surface_type)
                        if prompt is not None
                        else "banner"
                    )

                    # Court floor: use painted compositor (full-frame warp).
                    if st == "court_floor":
                        from banner_pipeline.composite.painted import (
                            painted_court_composite,
                        )

                        court_overrides = surface_overrides.get("court_floor", {})
                        # First inpaint (erase original text) using SAM mask.
                        sam_mask = masks_2d.get(obj_id)
                        if sam_mask is not None and np.any(sam_mask > 0):
                            mask_u8 = (sam_mask > 0).astype(np.uint8) * 255
                            dilate_px = int(court_overrides.get("mask_dilate_px", 8))
                            kern = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
                            )
                            mask_u8 = cv2.dilate(mask_u8, kern)
                            # Subtract person mask from inpainting mask so we
                            # don't inpaint under the player (causes artifacts).
                            if person_mask is not None:
                                person_u8 = (person_mask > 0.5).astype(np.uint8) * 255
                                person_dilated = cv2.dilate(
                                    person_u8,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                                )
                                mask_u8 = cv2.bitwise_and(mask_u8, cv2.bitwise_not(person_dilated))
                            if np.any(mask_u8 > 0):
                                frame_bgr = cv2.inpaint(frame_bgr, mask_u8, 3, cv2.INPAINT_NS)
                        # Then composite the logo with painted blend.
                        # Pass SAM mask for natural occlusion (SAM excludes
                        # players walking over the text automatically).
                        painted_court_composite(
                            frame_bgr,
                            current_corners[obj_id],
                            overlay,
                            sam_mask=sam_mask,
                            occlusion_mask=person_mask,
                            alpha_scale=float(court_overrides.get("alpha_scale", 0.75)),
                            alpha_feather_px=int(court_overrides.get("alpha_feather_px", 3)),
                        )
                        continue

                    extra_kw = dict(compositor_params)
                    # Apply per-surface-type compositor overrides.
                    if prompt is not None:
                        overrides = surface_overrides.get(st, {})
                        extra_kw.update(overrides)
                    # Pass person occlusion mask for side panel objects.
                    if person_mask is not None and st == "side_panel":
                        extra_kw["occlusion_mask"] = person_mask
                    # Use per-surface compositor to isolate caches.
                    comp = compositor
                    if st != "banner":
                        if st not in _surface_compositors:
                            _surface_compositors[st] = build_compositor(pipeline_cfg["compositor"])
                        comp = _surface_compositors[st]
                    if comp.name == "alpha":
                        homo = compute_oriented_homography(current_corners[obj_id], K)
                        extra_kw["homo"] = homo
                    frame_bgr = comp.composite(
                        frame_bgr,
                        current_corners[obj_id],
                        overlay,
                        mask=masks_2d.get(obj_id),
                        **extra_kw,
                    )
                composite_times.append(time.perf_counter() - t_comp)

            t_write = time.perf_counter()
            video_writer.write(frame_bgr)
            num_written += 1
            write_video_s += time.perf_counter() - t_write

            if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_names) - 1:
                print(f"[hybrid] Processed frame {frame_idx + 1}/{len(frame_names)}")

    finally:
        video_writer.close()
        shutil.rmtree(frame_dir, ignore_errors=True)

    metrics["write_video_s"] = round(write_video_s, 4)
    print(f"[hybrid] Wrote {num_written} frames -> {output_path}")

    if tracking_times:
        track_arr = np.array(tracking_times) * 1000
        metrics["tracking_mean_ms"] = round(float(track_arr.mean()), 2)
    if composite_times:
        comp_arr = np.array(composite_times) * 1000
        metrics["composite_mean_ms"] = round(float(comp_arr.mean()), 2)

    metrics["total_s"] = round(
        metrics["segment_total_s"] + sum(tracking_times) + sum(composite_times) + write_video_s,
        4,
    )
    metrics["output_fps"] = round(len(frame_names) / metrics["total_s"], 2)

    if _perf.PERF_ENABLED:
        metrics["composite_breakdown_ms"] = _perf.snapshot_ms(divisor=len(frame_names))

    return {"output_path": output_path, "metrics": metrics}


def run(
    config: dict,
    config_path: str | None = None,
    output_path: str = "output.mp4",
) -> dict:
    """Dispatch to single-frame, video, tracking, or hybrid pipeline."""
    mode = config.get("pipeline", {}).get("mode", "image")
    if mode == "video_hybrid":
        return run_pipeline_video_hybrid(
            config,
            output_path=output_path,
            config_path=config_path,
        )
    if mode == "video_tracking":
        return run_pipeline_video_tracking(
            config,
            output_path=output_path,
            config_path=config_path,
        )
    if mode == "video":
        return run_pipeline_video(config, output_path=output_path, config_path=config_path)
    return run_pipeline(config, config_path=config_path)
