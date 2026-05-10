"""Write quality_metrics.json (flat) and report.md (human rollup)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


# Pass thresholds, mirrored from docs/EVALUATION.md.
# Notes on calibration (2026-05-04, v68 H200 gold baseline):
#   - roi_delta_E_lab compares the placed region's mean color to a neighbor
#     patch on the same surface; for a banner the placed region IS supposed
#     to differ visually, so the metric is warning-only until we replace it
#     with a "did we preserve the underlying surface color" measurement.
#   - walkover_logo_visible_pct gate is calibrated to the gold's 0.18 floor;
#     per docs/EVALUATION.md threshold should rise as we accumulate runs.
GATES: dict[str, tuple[str, float]] = {
    "corner_max_jump_px": ("<", 2.0),
    "quad_area_cv": ("<", 0.05),
    "corner_accel_p95_px": ("<", 1.0),
    "roi_jitter_ratio": ("<=", 1.05),
    "roi_temporal_ssim_mean": (">", 0.95),
}

WALKOVER_GATES: dict[str, tuple[str, float]] = {
    "walkover_logo_visible_pct": (">", 0.10),
    "walkover_occlusion_iou": (">", 0.80),
}

WARNINGS: dict[str, tuple[str, float]] = {
    "noise_variance_ratio": ("<", 0.30),
    "edge_sharpness_ratio": (">", 1.8),
    "roi_delta_E_lab": (">", 5.0),
}


def evaluate_pass(
    metrics: dict[str, Any],
    region_prefix: str,
    is_floor: bool = False,
) -> tuple[bool, list[str]]:
    """Return (pass, list_of_failed_metric_keys) for a region's metric set.

    `metrics` is the per-region metric dict (no prefix). `region_prefix` is
    used only for nicer failure naming.
    """
    failed: list[str] = []
    gates = dict(GATES)
    if is_floor:
        gates.update(WALKOVER_GATES)
    for key, (op, thresh) in gates.items():
        if key not in metrics:
            continue
        v = metrics[key]
        if v is None:
            continue
        if not _check(v, op, thresh):
            failed.append(f"{region_prefix}_{key}")
    return (len(failed) == 0, failed)


def _check(value: float, op: str, threshold: float) -> bool:
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    return False


def warning_flags(metrics: dict[str, Any]) -> list[str]:
    """Return list of warning-only metric keys that fired."""
    out: list[str] = []
    for key, (op, thresh) in WARNINGS.items():
        v = metrics.get(key)
        if v is None:
            continue
        if _check(v, op, thresh):
            out.append(key)
    return out


def flatten_metrics(per_region: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert {region_kind: {metric: value}} to {region_metric: value}."""
    out: dict[str, Any] = {}
    for region, m in per_region.items():
        for k, v in m.items():
            out[f"{region}_{k}"] = v
    return out


def write_quality_json(
    output_path: str | Path,
    payload: dict[str, Any],
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, **payload}, f, indent=2, default=_default)


def _default(o):
    try:
        import numpy as np

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


def write_report_md(
    output_path: str | Path,
    payload: dict[str, Any],
    artifacts: dict[str, str],
) -> None:
    """Write a scannable Markdown rollup with embedded artifact paths.

    `artifacts` maps logical names to paths (e.g. "back_strip" -> "eval/back_banners/crops_strip.png").
    """
    lines: list[str] = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append(f"- Experiment: `{payload.get('experiment_dir')}`")
    if payload.get("reference_dir"):
        lines.append(f"- Reference:  `{payload['reference_dir']}`")
    lines.append(f"- Geometric source: `{payload.get('geometric_source')}`")
    lines.append("")
    lines.append("## Per-region scorecards")
    lines.append("")
    lines.append("| Region | Pass | Failed metrics | Warnings |")
    lines.append("|---|---|---|---|")
    for region in ("back", "left", "floor", "full"):
        passed = payload.get(f"{region}_pass")
        failed = payload.get(f"{region}_failed_metrics") or []
        warn = payload.get(f"{region}_warnings") or []
        if passed is None:
            continue
        lines.append(
            f"| {region} | {'PASS' if passed else 'FAIL'} | {', '.join(failed) or '-'} | {', '.join(warn) or '-'} |"
        )
    lines.append("")
    if "any_regression" in payload:
        lines.append(f"**Any regression vs gold:** `{payload['any_regression']}`")
        lines.append("")
    if "walkover_window_start" in payload and payload.get("walkover_window_start") is not None:
        lines.append(
            f"**Walkover window:** frames "
            f"`{payload['walkover_window_start']}–{payload['walkover_window_end']}`"
        )
        lines.append("")
    lines.append("## Visual artifacts")
    for name, p in artifacts.items():
        lines.append(f"- {name}: `{p}`")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
