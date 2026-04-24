"""Helpers for building persisted experiment metric reports."""

from __future__ import annotations

from typing import Any

import numpy as np

_PASSTHROUGH_KEYS = [
    "num_prompts",
    "num_prompt_points",
    "num_frames",
    "input_fps",
    "duration_s",
    "frame_width",
    "frame_height",
    "video_path",
    "fitter_type",
    "compositor_type",
    "checkpoint",
    "preview_frame_idx",
    "preview_objects_with_masks",
    "preview_objects_with_quads",
    "preview_ok",
    "preview_failure_reasons",
    "preview_object_diagnostics",
    "object_frame_coverage",
    "object_valid_frame_coverage",
    "object_rejection_counts",
    "object_rejection_reasons",
    "sam3_reanchor_events",
    "geometry_config_enabled",
    "geometry_runtime_enabled",
    "geometry_active_objects",
    "object_geometry_model",
    "back_wall_runtime_model",
    "side_wall_runtime_model",
    "geometry_fit_method_counts",
    "stabilization_config_enabled",
    "stabilization_runtime_enabled",
    "stabilization_object_stats",
    "git_branch",
    "git_commit_sha",
    "requested_config_path",
    "frozen_config_path",
    "frozen_config_sha256",
]

_NUMERIC_KEYS = [
    "load_frame_s",
    "segment_s",
    "segment_total_s",
    "fit_s",
    "fit_mean_ms",
    "fit_std_ms",
    "composite_s",
    "composite_mean_ms",
    "composite_std_ms",
    "write_video_s",
    "total_s",
    "run_total_s",
    "output_fps",
    "frames_with_masks",
    "frames_with_valid_objects",
    "frames_with_quads",
    "frames_composited",
    "object_masks_total",
    "first_frame_with_mask",
    "last_frame_with_mask",
    "max_consecutive_mask_gap",
    "geometry_total_s",
    "geometry_frames_held",
    "geometry_fallback_frames",
    "vp_width_valid_ratio",
    "vp_depth_valid_ratio",
    "court_width_candidate_count",
    "court_depth_candidate_count",
    "stabilization_total_s",
    "stabilization_static_frame_ratio",
    "stabilization_frames_held",
    "stabilization_frames_blended",
    "stabilization_frames_raw_accepted",
]


def build_metrics_report(
    all_metrics: list[dict[str, Any]],
    *,
    benchmark_runs: int,
    gpu_name: str,
    gpu_mem_gb: float,
    mode: str,
) -> dict[str, Any]:
    """Build the persisted benchmark report for one or more pipeline runs."""
    report: dict[str, Any] = {
        "runs": benchmark_runs,
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem_gb, 1),
        "mode": mode,
    }
    if not all_metrics:
        return report

    first_run = all_metrics[0]
    for key in _PASSTHROUGH_KEYS:
        if key in first_run:
            report[key] = first_run[key]

    if benchmark_runs > 1:
        for key in _NUMERIC_KEYS:
            values = [metric[key] for metric in all_metrics if key in metric]
            if not values:
                continue
            report[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }
    else:
        for key in _NUMERIC_KEYS:
            if key in first_run:
                report[key] = first_run[key]

    if "composite_breakdown_ms" in first_run:
        breakdown_runs = [
            metric["composite_breakdown_ms"]
            for metric in all_metrics
            if "composite_breakdown_ms" in metric
        ]
        all_keys = sorted({key for breakdown in breakdown_runs for key in breakdown})
        report["composite_breakdown_ms"] = {
            key: round(float(np.mean([breakdown.get(key, 0.0) for breakdown in breakdown_runs])), 3)
            for key in all_keys
        }

    return report
