"""Unit tests for banner_pipeline.eval.regions."""

from __future__ import annotations

import numpy as np
import pytest

from banner_pipeline.eval import regions as regions_mod


def _quad(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _config_with_prompts(prompts: list[dict]) -> dict:
    return {"input": {"prompts": prompts}}


def test_classify_court_floor_returns_floor_kind():
    cfg = _config_with_prompts([{
        "obj_id": 3,
        "surface_type": "court_floor",
        "placement_quad": _quad(800, 900, 1100, 960),
    }])
    regions = regions_mod.discover_regions(cfg)
    assert len(regions) == 1
    assert regions[0].region_kind == regions_mod.REGION_FLOOR
    assert regions[0].obj_id == 3


def test_classify_logo_placement_quad_returns_left_kind():
    cfg = _config_with_prompts([{
        "obj_id": 4,
        "surface_type": "banner",
        "placement_quad": _quad(50, 580, 440, 645),
        "compositor_params": {
            "logo_placement_quad": _quad(105, 589, 437, 640),
        },
    }])
    regions = regions_mod.discover_regions(cfg)
    assert len(regions) == 1
    assert regions[0].region_kind == regions_mod.REGION_LEFT
    # The canonical quad for a left logo is the *inner* logo_placement_quad.
    assert regions[0].placement_quad[0, 0] == 105
    assert regions[0].placement_quad[0, 1] == 589


def test_classify_back_banner_returns_back_kind():
    cfg = _config_with_prompts([{
        "obj_id": 1,
        "surface_type": "banner",
        "placement_quad": _quad(628, 64, 832, 122),
    }])
    regions = regions_mod.discover_regions(cfg)
    assert len(regions) == 1
    assert regions[0].region_kind == regions_mod.REGION_BACK


def test_discover_regions_skips_prompts_without_quads():
    cfg = _config_with_prompts([
        {"obj_id": 9, "surface_type": "banner"},
        {"obj_id": 1, "surface_type": "banner", "placement_quad": _quad(0, 0, 10, 10)},
    ])
    regions = regions_mod.discover_regions(cfg)
    assert [r.obj_id for r in regions] == [1]


def test_quad_to_roi_applies_padding_and_clips():
    quad = np.asarray(_quad(50, 50, 100, 100), dtype=np.float32)
    x0, y0, x1, y1 = regions_mod.quad_to_roi(
        quad, frame_w=200, frame_h=200, padding_x=20, padding_y=20
    )
    assert (x0, y0, x1, y1) == (30, 30, 120, 120)


def test_quad_to_roi_clips_to_frame_bounds():
    quad = np.asarray(_quad(0, 0, 50, 50), dtype=np.float32)
    x0, y0, x1, y1 = regions_mod.quad_to_roi(
        quad, frame_w=100, frame_h=100, padding_x=30, padding_y=30
    )
    assert (x0, y0) == (0, 0)
    assert x1 == 80 and y1 == 80


def test_quad_to_roi_rejects_degenerate_quad():
    bad = np.asarray([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float32)
    with pytest.raises(ValueError):
        regions_mod.quad_to_roi(bad, frame_w=100, frame_h=100, padding_x=0, padding_y=0)


def test_neighbor_patch_roi_picks_side_with_more_room():
    quad = np.asarray(_quad(900, 100, 1000, 200), dtype=np.float32)
    # Frame width 1920, so right side has ~920 px, left has 900 — should pick "right".
    x0, y0, x1, y1 = regions_mod.neighbor_patch_roi(
        quad, frame_w=1920, frame_h=400, direction="auto"
    )
    assert x0 >= 1000  # patch is to the right of the quad


def test_regions_by_kind_groups_correctly():
    cfg = _config_with_prompts([
        {"obj_id": 1, "surface_type": "banner", "placement_quad": _quad(0, 0, 10, 10)},
        {"obj_id": 2, "surface_type": "banner", "placement_quad": _quad(20, 0, 30, 10)},
        {"obj_id": 3, "surface_type": "court_floor", "placement_quad": _quad(0, 50, 30, 70)},
        {
            "obj_id": 4,
            "surface_type": "banner",
            "placement_quad": _quad(0, 80, 30, 100),
            "compositor_params": {"logo_placement_quad": _quad(2, 82, 28, 98)},
        },
    ])
    regions = regions_mod.discover_regions(cfg)
    grouped = regions_mod.regions_by_kind(regions)
    assert {r.obj_id for r in grouped[regions_mod.REGION_BACK]} == {1, 2}
    assert {r.obj_id for r in grouped[regions_mod.REGION_FLOOR]} == {3}
    assert {r.obj_id for r in grouped[regions_mod.REGION_LEFT]} == {4}
