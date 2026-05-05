"""Region discovery from a frozen experiment config.

Reads the frozen `config.yaml` of an experiment dir, walks `input.prompts`,
and groups objects into region kinds:

    - "back"  : surface_type == "banner" with no logo_placement_quad nested in
                compositor_params (the back-wall ad slots)
    - "left"  : surface_type == "banner" WITH compositor_params.logo_placement_quad
                (the side banner whose visible logo lives inside a sub-quad)
    - "floor" : surface_type == "court_floor" (the walkover logo)

For each object we also compute the canonical placement quad in image
coordinates: `placement_quad` for "back" / "floor", and the nested
`compositor_params.logo_placement_quad` for "left".

This is the single point where Melbourne / per-clip pixel knowledge enters
the eval framework. Everything else parameterizes off `RegionInfo`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REGION_BACK = "back"
REGION_LEFT = "left"
REGION_FLOOR = "floor"


@dataclass
class RegionInfo:
    """One placed object's region metadata."""

    obj_id: int
    region_kind: str  # "back" | "left" | "floor"
    surface_type: str
    placement_quad: np.ndarray  # (4, 2) float32, image pixels
    asset: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_frozen_config(experiment_dir: str | Path) -> dict[str, Any]:
    """Load the experiment's frozen config.yaml."""
    cfg_path = Path(experiment_dir) / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config.yaml in {experiment_dir}")
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


def _coerce_quad(q: Any) -> np.ndarray | None:
    """Convert a nested-list quad to a (4, 2) float32 array."""
    if q is None:
        return None
    arr = np.asarray(q, dtype=np.float32)
    if arr.shape != (4, 2):
        return None
    return arr


def _classify_object(prompt: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    """Return (region_kind, placement_quad) for a prompt entry.

    region_kind is one of REGION_BACK/REGION_LEFT/REGION_FLOOR.
    placement_quad is the canonical 4x2 quad in image coordinates.
    Either may be None if the prompt does not define a placeable region.
    """
    surface = str(prompt.get("surface_type") or "banner").lower()
    placement_quad = _coerce_quad(prompt.get("placement_quad"))
    comp_params = prompt.get("compositor_params") or {}
    logo_quad = _coerce_quad(comp_params.get("logo_placement_quad"))

    if surface == "court_floor":
        return REGION_FLOOR, placement_quad
    if logo_quad is not None:
        return REGION_LEFT, logo_quad
    return REGION_BACK, placement_quad


def discover_regions(config: dict[str, Any]) -> list[RegionInfo]:
    """Walk config.input.prompts and produce a RegionInfo per placed object.

    Skips entries without a usable placement quad — they're segmentation-only
    prompts (no asset overlaid) and don't participate in eval.
    """
    prompts = (config.get("input") or {}).get("prompts") or []
    out: list[RegionInfo] = []
    for prompt in prompts:
        region_kind, quad = _classify_object(prompt)
        if quad is None:
            continue
        out.append(
            RegionInfo(
                obj_id=int(prompt["obj_id"]),
                region_kind=region_kind,
                surface_type=str(prompt.get("surface_type") or "banner"),
                placement_quad=quad,
                asset=prompt.get("asset"),
                extra={
                    "compositor_params": prompt.get("compositor_params") or {},
                    "court_plane_placement": prompt.get("court_plane_placement"),
                },
            )
        )
    return out


def regions_by_kind(regions: list[RegionInfo]) -> dict[str, list[RegionInfo]]:
    """Group regions by region_kind."""
    out: dict[str, list[RegionInfo]] = {REGION_BACK: [], REGION_LEFT: [], REGION_FLOOR: []}
    for r in regions:
        out.setdefault(r.region_kind, []).append(r)
    return out


def quad_to_roi(
    quad: np.ndarray,
    frame_w: int,
    frame_h: int,
    padding_x: int = 30,
    padding_y: int = 30,
) -> tuple[int, int, int, int]:
    """Axis-aligned bounding box around `quad`, clipped to frame, with padding.

    Returns (x0, y0, x1, y1) suitable for `frame[y0:y1, x0:x1]`.
    """
    if quad is None or quad.shape != (4, 2):
        raise ValueError("quad must be a (4, 2) array")
    x0 = max(0, int(np.floor(quad[:, 0].min())) - padding_x)
    y0 = max(0, int(np.floor(quad[:, 1].min())) - padding_y)
    x1 = min(frame_w, int(np.ceil(quad[:, 0].max())) + padding_x)
    y1 = min(frame_h, int(np.ceil(quad[:, 1].max())) + padding_y)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"degenerate roi from quad {quad.tolist()}")
    return x0, y0, x1, y1


def neighbor_patch_roi(
    quad: np.ndarray,
    frame_w: int,
    frame_h: int,
    direction: str = "auto",
) -> tuple[int, int, int, int]:
    """Pick a same-surface neighbor patch next to `quad`, clipped to frame.

    Used by color and noise-variance metrics to compare placed-region statistics
    against an adjacent patch of the underlying surface (court / wall) that the
    pipeline did not touch.

    `direction` is "left" / "right" / "above" / "below" / "auto"; "auto" picks
    the side with the most room.
    """
    x0, y0, x1, y1 = quad_to_roi(quad, frame_w, frame_h, padding_x=0, padding_y=0)
    w = x1 - x0
    h = y1 - y0

    if direction == "auto":
        room = {"left": x0, "right": frame_w - x1, "above": y0, "below": frame_h - y1}
        direction = max(room, key=lambda k: room[k])

    gap = 8
    if direction == "left":
        nx1 = max(0, x0 - gap)
        nx0 = max(0, nx1 - w)
        ny0, ny1 = y0, y1
    elif direction == "right":
        nx0 = min(frame_w, x1 + gap)
        nx1 = min(frame_w, nx0 + w)
        ny0, ny1 = y0, y1
    elif direction == "above":
        ny1 = max(0, y0 - gap)
        ny0 = max(0, ny1 - h)
        nx0, nx1 = x0, x1
    else:  # below
        ny0 = min(frame_h, y1 + gap)
        ny1 = min(frame_h, ny0 + h)
        nx0, nx1 = x0, x1

    if nx1 <= nx0 or ny1 <= ny0:
        # Fall back to a square next to the quad on whichever axis still fits.
        return max(0, x0 - max(20, w // 2)), y0, max(0, x0 - 1), y1
    return nx0, ny0, nx1, ny1
