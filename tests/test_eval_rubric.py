"""Unit tests for the v2 visual rubric — halo + edge_reflex calibration."""

from __future__ import annotations

from banner_pipeline.eval.ai_review import _format_manifest_md
from banner_pipeline.eval.rubric import (
    RUBRIC_VERSION,
    SURFACE_BEARING_REGIONS,
    schema_for,
    validate_rubric,
)


def test_rubric_version_bumped_to_v2() -> None:
    """v2 introduces halo_presence + edge_reflex on surface-bearing regions."""
    assert RUBRIC_VERSION >= 2


def test_floor_schema_has_halo_and_reflex() -> None:
    schema = schema_for("floor")
    assert "realism.halo_presence" in schema
    assert "realism.edge_reflex" in schema
    assert schema["realism.halo_presence"] == ("score", (1, 5))
    assert schema["realism.edge_reflex"] == ("score", (1, 5))


def test_back_left_walkover_schemas_have_halo_and_reflex() -> None:
    """All surface-bearing regions get the new dimensions, not just floor."""
    for region in ("back", "left", "walkover"):
        schema = schema_for(region)
        assert "realism.halo_presence" in schema, region
        assert "realism.edge_reflex" in schema, region


def test_full_schema_does_not_have_halo_or_reflex() -> None:
    """`full` is the global rollup; it should NOT double-count surface dims."""
    schema = schema_for("full")
    assert "realism.halo_presence" not in schema
    assert "realism.edge_reflex" not in schema


def test_existing_v1_dimensions_preserved() -> None:
    """v2 only ADDS fields. Existing rubric JSON consumers must not break."""
    schema = schema_for("floor")
    for legacy_field in (
        "realism.painted_on_vs_pasted_on",
        "realism.edge_seam_visibility",
        "realism.texture_match",
        "color.hue_match",
        "color.brightness_match",
        "color.saturation_match",
        "geometry.perspective_plausibility",
        "geometry.size_plausibility",
        "temporal.occlusion_realism",
        "temporal.jitter_visible",
        "temporal.player_contact_shadow",
    ):
        assert legacy_field in schema, legacy_field


def test_surface_bearing_regions_constant_matches_dim_assignment() -> None:
    assert SURFACE_BEARING_REGIONS == {"back", "left", "floor", "walkover"}


def test_validate_rubric_requires_new_fields_on_floor() -> None:
    """A v1-shaped payload is now incomplete on a surface-bearing region."""
    v1_floor_payload = {
        "realism": {
            "painted_on_vs_pasted_on": 5,
            "edge_seam_visibility": 5,
            "texture_match": 5,
        },
        "color": {"hue_match": 5, "brightness_match": 5, "saturation_match": 5},
        "geometry": {"perspective_plausibility": 5, "size_plausibility": 5},
        "temporal": {
            "occlusion_realism": 5,
            "jitter_visible": 5,
            "player_contact_shadow": 5,
        },
        "notes": "looks fine",
    }
    ok, errors = validate_rubric(v1_floor_payload, "floor")
    assert not ok
    assert any("halo_presence" in e for e in errors)
    assert any("edge_reflex" in e for e in errors)


def test_validate_rubric_passes_with_new_fields() -> None:
    payload = {
        "realism": {
            "painted_on_vs_pasted_on": 5,
            "edge_seam_visibility": 5,
            "texture_match": 5,
            "halo_presence": 2,
            "edge_reflex": 3,
        },
        "color": {"hue_match": 5, "brightness_match": 5, "saturation_match": 5},
        "geometry": {"perspective_plausibility": 5, "size_plausibility": 5},
        "temporal": {
            "occlusion_realism": 5,
            "jitter_visible": 5,
            "player_contact_shadow": 5,
        },
        "notes": "halo around logo on court paint, mild letter reflex",
    }
    ok, errors = validate_rubric(payload, "floor")
    assert ok, errors


def test_manifest_mentions_halo_and_reflex_callouts() -> None:
    """The MANIFEST.md must call out the user's two flagged artifact families."""
    md = _format_manifest_md(
        region_inputs={
            "back": ["eval/back_banners/crops_strip.png"],
            "left": ["eval/left_logo/crops_strip.png"],
            "floor": ["eval/floor_logo/crops_strip.png"],
            "walkover": ["eval/walkover/consecutive_frames.png"],
        },
        walkover_window=(685, 723),
    )
    # Both new rubric dimensions must appear by their dotted-path name.
    assert "realism.halo_presence" in md
    assert "realism.edge_reflex" in md
    # The user's specific concerns must be named so reviewers can't gloss over.
    assert "halo around the logo" in md.lower()
    assert "reflex" in md.lower() or "smearing" in md.lower()
    # Calibration callout block must exist with anti-collapse instruction.
    assert "DO NOT collapse these to 5" in md or "do not collapse" in md.lower()
