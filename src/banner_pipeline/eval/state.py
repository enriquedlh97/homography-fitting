"""Load and query the pipeline's per_frame_state.json dump.

When the dump is present, geometric metrics (corner_max_jump_px, quad_area_cv,
corner_accel_p95_px) read from real per-frame trajectories.

When it's missing (legacy experiments run before the pipeline change), the
loader returns a static-fallback state where every frame's quad equals the
config's static `placement_quad`. Geometric jitter is then 0 by construction
and downstream code marks `geometric_source = "static_fallback"`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from banner_pipeline.eval.regions import RegionInfo


@dataclass
class PerFrameState:
    """Per-frame, per-object placement quads."""

    num_frames: int
    # frame_idx -> obj_id -> (4, 2) float32 corners
    frames: dict[int, dict[int, np.ndarray]]
    source: str  # "per_frame_state" | "static_fallback"

    def trajectory(self, obj_id: int) -> np.ndarray | None:
        """Return (T, 4, 2) array of corners for obj_id across frames where present."""
        rows = []
        for fid in sorted(self.frames):
            obj_frame = self.frames[fid]
            if obj_id in obj_frame:
                rows.append(obj_frame[obj_id])
        if not rows:
            return None
        return np.stack(rows, axis=0)


def load_per_frame_state(
    experiment_dir: str | Path,
    regions: list[RegionInfo],
    num_frames_hint: int | None = None,
) -> PerFrameState:
    """Read outputs/per_frame_state.json or fall back to static config quads.

    The fallback duplicates the config's placement_quad across all frames, so
    geometric jitter metrics read 0. The eval framework reports this via
    `geometric_source = "static_fallback"` so downstream consumers know the
    metric is uninformative.
    """
    path = Path(experiment_dir) / "outputs" / "per_frame_state.json"
    if path.is_file():
        with path.open() as f:
            payload = json.load(f)
        frames: dict[int, dict[int, np.ndarray]] = {}
        for fid_str, obj_dict in (payload.get("frames") or {}).items():
            fid = int(fid_str)
            frame_objs: dict[int, np.ndarray] = {}
            for oid_str, info in obj_dict.items():
                corners = info.get("corners") if isinstance(info, dict) else None
                arr = np.asarray(corners, dtype=np.float32) if corners is not None else None
                if arr is None or arr.shape != (4, 2):
                    continue
                frame_objs[int(oid_str)] = arr
            if frame_objs:
                frames[fid] = frame_objs
        return PerFrameState(
            num_frames=int(payload.get("num_frames") or len(frames)),
            frames=frames,
            source="per_frame_state",
        )

    # Static fallback: every frame uses the config's placement_quad.
    n = num_frames_hint or 1
    static_per_obj = {r.obj_id: r.placement_quad for r in regions}
    frames = {fid: {oid: q.copy() for oid, q in static_per_obj.items()} for fid in range(n)}
    return PerFrameState(num_frames=n, frames=frames, source="static_fallback")
