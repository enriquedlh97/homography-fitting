"""Geometric stability metrics from the per-frame quad trajectory.

All metrics are computed from `PerFrameState.trajectory(obj_id)`, which is a
(T, 4, 2) array. When the underlying state was loaded via the static fallback
(legacy experiments), the trajectory is constant by construction and these
metrics return 0 — that's the correct signal: "we cannot detect jitter".
"""

from __future__ import annotations

from typing import Any

import numpy as np


def corner_metrics(traj: np.ndarray) -> dict[str, Any]:
    """Compute corner-stability metrics from a (T, 4, 2) trajectory.

    Returns a dict suitable for merging into a region's metrics.json:

    - corner_max_jump_px      : max ||c_t - c_{t-1}||_2 across frames and corners
    - corner_mean_jump_px     : mean of the same
    - corner_accel_p95_px     : 95th percentile of |2nd-derivative| of mean centroid
    - quad_area_cv            : coefficient of variation of polygon area across frames
    """
    if traj is None or traj.ndim != 3 or traj.shape[1:] != (4, 2) or traj.shape[0] < 2:
        return {
            "corner_max_jump_px": 0.0,
            "corner_mean_jump_px": 0.0,
            "corner_accel_p95_px": 0.0,
            "quad_area_cv": 0.0,
            "frames_used": int(traj.shape[0]) if traj is not None else 0,
        }

    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=2)  # (T-1, 4)
    centroid = traj.mean(axis=1)  # (T, 2)
    if centroid.shape[0] >= 3:
        velocity = np.diff(centroid, axis=0)
        accel = np.diff(velocity, axis=0)
        accel_mag = np.linalg.norm(accel, axis=1)
        accel_p95 = float(np.percentile(accel_mag, 95)) if accel_mag.size else 0.0
    else:
        accel_p95 = 0.0

    areas = np.array([_polygon_area(q) for q in traj])
    area_mean = float(areas.mean()) if areas.size else 0.0
    area_std = float(areas.std()) if areas.size else 0.0
    area_cv = (area_std / area_mean) if area_mean > 0 else 0.0

    return {
        "corner_max_jump_px": round(float(diffs.max()), 4),
        "corner_mean_jump_px": round(float(diffs.mean()), 4),
        "corner_accel_p95_px": round(accel_p95, 4),
        "quad_area_cv": round(area_cv, 6),
        "frames_used": int(traj.shape[0]),
    }


def corner_distance_vs_reference(
    traj_a: np.ndarray | None,
    traj_b: np.ndarray | None,
) -> dict[str, Any]:
    """Per-frame corner Euclidean distance between two trajectories.

    Returns mean and 95th percentile across all (frame, corner) pairs that
    exist in both. NaN-safe when either trajectory is missing.
    """
    if traj_a is None or traj_b is None:
        return {"corner_distance_mean_px": None, "corner_distance_p95_px": None}
    n = min(traj_a.shape[0], traj_b.shape[0])
    if n == 0:
        return {"corner_distance_mean_px": None, "corner_distance_p95_px": None}
    a = traj_a[:n]
    b = traj_b[:n]
    d = np.linalg.norm(a - b, axis=2).reshape(-1)
    return {
        "corner_distance_mean_px": round(float(d.mean()), 4),
        "corner_distance_p95_px": round(float(np.percentile(d, 95)), 4),
    }


def _polygon_area(quad: np.ndarray) -> float:
    """Shoelace formula for a (4, 2) polygon."""
    x = quad[:, 0]
    y = quad[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
