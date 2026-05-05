"""Reference (gold) resolution and regression flagging.

Reads `configs/eval/reference.yaml`, which maps an input video basename to
the gold experiment directory used as the comparison anchor for that clip.

Schema:

    melbourne-walking-over-logo.mov:
        gold_dir: experiments/2026-04-30_17-06-28_walkover_v68_..._H200/

A run's `--reference auto` resolves the gold by reading the frozen
`config.yaml`'s `input.video` basename and looking it up in the table.

Regression detection uses a 5% relative slop on continuous metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_YAML = REPO_ROOT / "configs" / "eval" / "reference.yaml"

REGRESSION_SLOP = 0.05  # 5%


@dataclass
class GoldRef:
    gold_dir: Path
    source: str  # "auto_resolved" | "explicit" | "self"


def resolve_gold(
    experiment_dir: str | Path,
    explicit: str | None,
    config: dict[str, Any],
    reference_yaml: str | Path | None = None,
) -> GoldRef | None:
    """Pick the gold experiment dir for `experiment_dir`.

    `explicit` is the user-supplied --reference argument:
        - None or "off"  -> no comparison
        - a path         -> use that path
        - "auto"         -> resolve via configs/eval/reference.yaml + input.video basename
    """
    if explicit in (None, "off"):
        return None
    if explicit and explicit != "auto":
        p = Path(explicit)
        if p.is_dir():
            return GoldRef(gold_dir=p, source="explicit")
        return None

    yaml_path = Path(reference_yaml or DEFAULT_REFERENCE_YAML)
    if not yaml_path.is_file():
        return None
    with yaml_path.open() as f:
        table = yaml.safe_load(f) or {}

    video_path = ((config.get("input") or {}).get("video")) or ""
    basename = os.path.basename(video_path)
    entry = table.get(basename)
    if entry is None:
        return None
    if isinstance(entry, str):
        gold_path = entry
    else:
        gold_path = entry.get("gold_dir")
    if not gold_path:
        return None
    p = Path(gold_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_dir():
        return None

    src = "self" if str(p.resolve()) == str(Path(experiment_dir).resolve()) else "auto_resolved"
    return GoldRef(gold_dir=p, source=src)


def detect_regressions(current: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Per-metric regression flags between current and gold quality_metrics.json.

    A metric regresses when:
      - its current value is "worse" than gold's (defined per-metric direction), AND
      - the relative change exceeds REGRESSION_SLOP (5%).

    Returns a dict of {flag_name: bool} including a top-level `any_regression`.
    """
    # Direction: True if higher is better, False if lower is better.
    higher_is_better = {
        "back_roi_temporal_ssim_mean",
        "left_roi_temporal_ssim_mean",
        "floor_roi_temporal_ssim_mean",
        "full_roi_temporal_ssim_mean",
        "floor_walkover_logo_visible_pct",
        "floor_walkover_occlusion_iou",
    }
    lower_is_better = {
        "back_corner_max_jump_px",
        "left_corner_max_jump_px",
        "floor_corner_max_jump_px",
        "back_corner_accel_p95_px",
        "left_corner_accel_p95_px",
        "floor_corner_accel_p95_px",
        "back_quad_area_cv",
        "left_quad_area_cv",
        "floor_quad_area_cv",
        "back_roi_jitter_ratio",
        "left_roi_jitter_ratio",
        "floor_roi_jitter_ratio",
        "full_roi_jitter_ratio",
        "back_roi_delta_E_lab",
        "left_roi_delta_E_lab",
        "floor_roi_delta_E_lab",
    }

    flags: dict[str, Any] = {}
    any_reg = False
    for key in higher_is_better | lower_is_better:
        if key not in current or key not in gold:
            continue
        cur = current[key]
        ref = gold[key]
        if cur is None or ref is None:
            continue
        if not isinstance(cur, int | float) or not isinstance(ref, int | float):
            continue
        if ref == 0:
            continue
        rel = (cur - ref) / abs(ref)
        worse = rel < -REGRESSION_SLOP if key in higher_is_better else rel > REGRESSION_SLOP
        flags[f"regression_{key}"] = bool(worse)
        any_reg = any_reg or bool(worse)

    flags["any_regression"] = any_reg
    return flags
