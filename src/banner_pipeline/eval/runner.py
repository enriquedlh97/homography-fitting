"""Orchestrator: discovers regions, computes metrics, writes artifacts.

`run_eval(experiment_dir, ...)` is the single entrypoint imported by both
the CLI (`__main__.py`) and the public API (`__init__.py`).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from banner_pipeline.eval import (
    contact_sheets,
    metrics_color,
    metrics_geom,
    metrics_temporal,
    reference,
    regions as regions_mod,
    side_by_side,
    state as state_mod,
    walkover as walkover_mod,
)
from banner_pipeline.eval.report import (
    evaluate_pass,
    flatten_metrics,
    warning_flags,
    write_quality_json,
    write_report_md,
)


def run_eval(
    experiment_dir: str | Path,
    reference_arg: str | None = None,
    regions_subset: list[str] | None = None,
    walkover_window_override: tuple[int, int] | None = None,
    with_ai_review: bool = False,
    original_video: str | Path | None = None,
    clean_video: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run the full eval pipeline on `experiment_dir`.

    Returns (quality_metrics_dict, exit_code) where exit_code matches the CLI:
        0 = all pass + no regression
        2 = a per-region scorecard failed
        3 = all pass but regression vs reference
    """
    experiment_dir = Path(experiment_dir)
    composite_path = experiment_dir / "outputs" / "composited.mp4"
    if not composite_path.is_file():
        print(f"[eval] missing {composite_path}", file=sys.stderr)
        return {}, 1

    config = regions_mod.load_frozen_config(experiment_dir)
    discovered = regions_mod.discover_regions(config)
    if not discovered:
        print(f"[eval] no placeable regions in {experiment_dir}/config.yaml", file=sys.stderr)
        return {}, 1

    by_kind = regions_mod.regions_by_kind(discovered)

    # Resolve original/clean video paths from frozen config when not supplied.
    input_cfg = config.get("input") or {}
    original_video = _resolve_video(original_video, input_cfg.get("video"), experiment_dir)
    clean_video = _resolve_video(clean_video, input_cfg.get("clean_video"), experiment_dir)

    # Probe the composited video for frame count + dimensions.
    cap = cv2.VideoCapture(str(composite_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    state = state_mod.load_per_frame_state(experiment_dir, discovered, num_frames_hint=n_frames)

    # Reference resolution.
    gold = reference.resolve_gold(experiment_dir, reference_arg, config)
    gold_composite = (
        gold.gold_dir / "outputs" / "composited.mp4" if gold is not None else None
    )
    gold_state = (
        state_mod.load_per_frame_state(gold.gold_dir, discovered, num_frames_hint=n_frames)
        if gold is not None
        else None
    )

    eval_dir = experiment_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_regions(regions_subset)
    per_region: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, str] = {}

    # ---- Per-region scorecards ----
    for region_kind, region_list in (
        ("back", by_kind.get("back") or []),
        ("left", by_kind.get("left") or []),
        ("floor", by_kind.get("floor") or []),
    ):
        if region_kind not in selected or not region_list:
            continue
        m, art = _evaluate_region(
            region_kind=region_kind,
            region_list=region_list,
            state=state,
            gold_state=gold_state,
            composite_path=composite_path,
            gold_composite=gold_composite,
            original_video=original_video,
            frame_w=frame_w,
            frame_h=frame_h,
            n_frames=n_frames,
            eval_dir=eval_dir,
        )
        per_region[region_kind] = m
        artifacts.update(art)

    # ---- Full-frame rollup ----
    if "full" in selected:
        full_metrics, full_art = _evaluate_full(
            composite_path=composite_path,
            gold_composite=gold_composite,
            original_video=original_video,
            frame_w=frame_w,
            frame_h=frame_h,
            eval_dir=eval_dir,
        )
        per_region["full"] = full_metrics
        artifacts.update(full_art)

    # ---- Walkover detection + occlusion ----
    walkover_window: walkover_mod.WalkoverWindow | None = None
    walkover_metrics: dict[str, Any] = {}
    floor_regions = by_kind.get("floor") or []
    if "walkover" in selected and floor_regions and original_video:
        floor_quad = floor_regions[0].placement_quad
        if walkover_window_override is not None:
            walkover_window = walkover_mod.WalkoverWindow(
                start=walkover_window_override[0],
                end=walkover_window_override[1],
                method="manual_override",
            )
        else:
            walkover_window = walkover_mod.detect_walkover_window(
                original_video=original_video,
                clean_video=clean_video,
                floor_quad=floor_quad,
            )
        if walkover_window is not None:
            walkover_dir = eval_dir / "walkover"
            walkover_dir.mkdir(parents=True, exist_ok=True)
            (walkover_dir / "window.json").write_text(
                json.dumps(
                    {
                        "start": walkover_window.start,
                        "end": walkover_window.end,
                        "method": walkover_window.method,
                    },
                    indent=2,
                )
            )
            # Consecutive-frames strip for the window.
            roi = regions_mod.quad_to_roi(
                floor_quad, frame_w, frame_h, padding_x=30, padding_y=60
            )
            contact_sheets.consecutive_frames_strip(
                composite_path=composite_path,
                output_path=walkover_dir / "consecutive_frames.png",
                roi=roi,
                start_frame=walkover_window.start,
                n_frames=min(16, walkover_window.end - walkover_window.start + 1),
                upscale=1,
            )
            artifacts["walkover_consecutive_frames"] = str(
                walkover_dir / "consecutive_frames.png"
            )
            # Forensic sheet at the middle frame of the window.
            mid = (walkover_window.start + walkover_window.end) // 2
            if clean_video is not None:
                contact_sheets.build_forensic_sheet(
                    original_path=original_video,
                    clean_path=clean_video,
                    composite_path=composite_path,
                    output_path=walkover_dir / f"forensic_sheet_f{mid:04d}.png",
                    frame_index=mid,
                    roi=roi,
                )
                artifacts["walkover_forensic_sheet"] = str(
                    walkover_dir / f"forensic_sheet_f{mid:04d}.png"
                )
            # Occlusion metrics in window.
            walkover_metrics = walkover_mod.occlusion_metrics_in_window(
                composite_path=composite_path,
                original_path=original_video,
                clean_path=clean_video,
                reference_composite_path=gold_composite,
                floor_quad=floor_quad,
                window=walkover_window,
            )
            # Merge into floor region (so per_region["floor"] carries the score).
            if "floor" in per_region:
                per_region["floor"].update(walkover_metrics)

    # ---- Side-by-side video ----
    if gold is not None and gold.source != "self" and gold_composite is not None:
        side_by_side.write_side_by_side(
            current_path=composite_path,
            reference_path=gold_composite,
            output_path=eval_dir / "vs_reference_side_by_side.mp4",
            walkover_window=(walkover_window.start, walkover_window.end)
            if walkover_window is not None
            else None,
        )
        artifacts["vs_reference_video"] = str(eval_dir / "vs_reference_side_by_side.mp4")

    # ---- AI review (opt-in) ----
    if with_ai_review:
        from banner_pipeline.eval import ai_review as ai_mod

        ai_dir = eval_dir / "ai_review"
        ai_mod.write_rubric_version(ai_dir)
        for region_kind in ("back", "left", "floor", "full"):
            if region_kind not in per_region:
                continue
            images = _collect_region_images(eval_dir, region_kind)
            if walkover_window is not None and region_kind == "floor":
                images.extend(_collect_region_images(eval_dir, "walkover"))
            payload = ai_mod.review_region(region_kind, images, ai_dir)
            if payload is not None:
                per_region[region_kind][f"ai_review_min_score"] = payload.get("min_score")

    # ---- Reference comparison + regression flags ----
    vs_ref: dict[str, Any] = {}
    if gold is not None:
        gold_metrics_path = gold.gold_dir / "eval" / "quality_metrics.json"
        if gold_metrics_path.is_file():
            gold_payload = json.loads(gold_metrics_path.read_text())
            current_flat_for_compare = flatten_metrics(per_region)
            vs_ref = reference.detect_regressions(current_flat_for_compare, gold_payload)
        # Per-region SSIM + corner distance vs gold.
        if gold_composite is not None and gold_composite.is_file():
            for region_kind, region_list in by_kind.items():
                if region_kind == "back" and region_list:
                    quad = region_list[0].placement_quad
                else:
                    quad = region_list[0].placement_quad if region_list else None
                if quad is None:
                    continue
                roi = regions_mod.quad_to_roi(quad, frame_w, frame_h)
                vs_ref.update(
                    {
                        f"{region_kind}_{k}": v
                        for k, v in metrics_temporal.roi_ssim_vs_reference(
                            str(composite_path), str(gold_composite), roi
                        ).items()
                    }
                )
                # Per-object corner distance for the first object of this kind.
                obj_id = region_list[0].obj_id
                cur_traj = state.trajectory(obj_id)
                gold_traj = gold_state.trajectory(obj_id) if gold_state is not None else None
                vs_ref.update(
                    {
                        f"{region_kind}_{k}": v
                        for k, v in metrics_geom.corner_distance_vs_reference(
                            cur_traj, gold_traj
                        ).items()
                    }
                )

    # ---- Aggregate scorecard ----
    flat = flatten_metrics(per_region)
    payload: dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "reference_dir": str(gold.gold_dir) if gold is not None else None,
        "objects": {str(r.obj_id): r.region_kind for r in discovered},
        "geometric_source": state.source,
        **flat,
    }
    if walkover_window is not None:
        payload["walkover_window_start"] = walkover_window.start
        payload["walkover_window_end"] = walkover_window.end
        payload["walkover_window_method"] = walkover_window.method
    payload["vs_reference"] = vs_ref or None
    if vs_ref:
        payload["any_regression"] = vs_ref.get("any_regression", False)

    # Pass/fail per region.
    any_fail = False
    for region_kind in ("back", "left", "floor", "full"):
        if region_kind not in per_region:
            continue
        passed, failed = evaluate_pass(
            per_region[region_kind], region_kind, is_floor=(region_kind == "floor")
        )
        warns = warning_flags(per_region[region_kind])
        payload[f"{region_kind}_pass"] = passed
        payload[f"{region_kind}_failed_metrics"] = failed
        payload[f"{region_kind}_warnings"] = warns
        any_fail = any_fail or not passed

    # Persist per-region metrics.json.
    for region_kind, metrics in per_region.items():
        sub = eval_dir / _region_dirname(region_kind)
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))

    # Persist top-level outputs.
    write_quality_json(eval_dir / "quality_metrics.json", payload)
    write_report_md(eval_dir / "report.md", payload, artifacts)

    # Exit code.
    if any_fail:
        return payload, 2
    if payload.get("any_regression"):
        return payload, 3
    return payload, 0


def _select_regions(arg: list[str] | None) -> set[str]:
    return set(arg) if arg else {"back", "left", "floor", "full", "walkover"}


def _resolve_video(
    explicit: str | Path | None,
    config_value: str | None,
    experiment_dir: Path,
) -> str | None:
    """Pick the original/clean video path, preferring explicit, then config, then local data/."""
    if explicit and Path(explicit).is_file():
        return str(explicit)
    if config_value:
        # Try as-is, then under repo root, then under experiment_dir.
        candidates = [
            Path(config_value),
            Path.cwd() / config_value,
            experiment_dir / config_value,
        ]
        # Modal runs reference /tmp/<random>/input.mp4 — fall back to the
        # local data/ basename when the temp path doesn't exist.
        if not Path(config_value).is_absolute() or not Path(config_value).exists():
            candidates.append(Path("data") / Path(config_value).name)
        for c in candidates:
            if c.is_file():
                return str(c)
    return None


def _evaluate_region(
    *,
    region_kind: str,
    region_list,
    state,
    gold_state,
    composite_path: Path,
    gold_composite: Path | None,
    original_video: str | None,
    frame_w: int,
    frame_h: int,
    n_frames: int,
    eval_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Compute metrics + artifacts for one region kind. Picks the largest quad
    in the region as the canonical ROI for color/temporal metrics; geometric
    metrics aggregate across all objects of the kind via the worst case.
    """
    # Geometric metrics: max-of-objects (worst quad determines the gate).
    geom_acc: dict[str, list[float]] = {}
    for r in region_list:
        traj = state.trajectory(r.obj_id)
        for k, v in metrics_geom.corner_metrics(traj).items():
            if isinstance(v, int | float):
                geom_acc.setdefault(k, []).append(float(v))
    geom = {k: max(vs) if k != "frames_used" else int(min(vs)) for k, vs in geom_acc.items()}

    # Pick canonical ROI = largest-area quad among the region's objects.
    canonical = max(region_list, key=lambda r: _area(r.placement_quad))
    roi = regions_mod.quad_to_roi(canonical.placement_quad, frame_w, frame_h)
    neighbor_roi = regions_mod.neighbor_patch_roi(canonical.placement_quad, frame_w, frame_h)

    color: dict[str, Any] = {}
    if original_video is not None:
        color.update(metrics_color.roi_jitter_ratio(str(composite_path), original_video, roi))
    color.update(metrics_color.roi_delta_e_vs_neighbor(str(composite_path), roi, neighbor_roi))
    color.update(metrics_color.roi_noise_variance_ratio(str(composite_path), roi, neighbor_roi))
    color.update(metrics_color.roi_edge_sharpness_ratio(str(composite_path), roi))
    temporal = metrics_temporal.roi_temporal_ssim(str(composite_path), roi)

    region_dir = eval_dir / _region_dirname(region_kind)
    region_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets.crops_strip(
        composite_path=composite_path,
        output_path=region_dir / "crops_strip.png",
        roi=roi,
        n_frames=6,
        upscale=2 if region_kind != "floor" else 1,
    )
    # Three consecutive-frame strips at 15%/50%/85% of the clip.
    for label, anchor in (("early", 0.15), ("mid", 0.50), ("late", 0.85)):
        start = int(anchor * max(n_frames - 8, 0))
        contact_sheets.consecutive_frames_strip(
            composite_path=composite_path,
            output_path=region_dir / f"consecutive_frames_{label}.png",
            roi=roi,
            start_frame=start,
            n_frames=8,
            upscale=2 if region_kind != "floor" else 1,
        )

    artifacts = {
        f"{region_kind}_strip": str(region_dir / "crops_strip.png"),
        f"{region_kind}_consecutive_mid": str(region_dir / "consecutive_frames_mid.png"),
    }

    metrics = {**geom, **color, **temporal}
    return metrics, artifacts


def _evaluate_full(
    *,
    composite_path: Path,
    gold_composite: Path | None,
    original_video: str | None,
    frame_w: int,
    frame_h: int,
    eval_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Full-frame rollup: SSIM + jitter ratio across the whole video."""
    full_roi = (0, 0, frame_w, frame_h)
    metrics: dict[str, Any] = {}
    if original_video is not None:
        metrics.update(metrics_color.roi_jitter_ratio(str(composite_path), original_video, full_roi))
    metrics.update(metrics_temporal.roi_temporal_ssim(str(composite_path), full_roi))
    full_dir = eval_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets.crops_strip(
        composite_path=composite_path,
        output_path=full_dir / "crops_strip.png",
        roi=None,
        n_frames=6,
        upscale=1,
    )
    return metrics, {"full_strip": str(full_dir / "crops_strip.png")}


def _collect_region_images(eval_dir: Path, region_kind: str) -> list[str | Path]:
    sub = eval_dir / _region_dirname(region_kind)
    if not sub.is_dir():
        return []
    return sorted(p for p in sub.glob("*.png"))


def _region_dirname(region_kind: str) -> str:
    return {
        "back": "back_banners",
        "left": "left_logo",
        "floor": "floor_logo",
        "full": "full",
        "walkover": "walkover",
    }.get(region_kind, region_kind)


def _area(quad: np.ndarray) -> float:
    x = quad[:, 0]
    y = quad[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _json_default(o):
    try:
        if isinstance(o, np.bool_ | np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
