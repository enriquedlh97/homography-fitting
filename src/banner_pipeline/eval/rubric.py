"""Visual-inspection rubric definition + lightweight schema validation.

Used by both the eval framework (to format the MANIFEST.md a sub-agent reads)
and any code that needs to validate sub-agent-written rubric JSONs.

The rubric review path uses **sub-agent vision via the Read tool** — never an
SDK call. See `docs/EVALUATION.md` "Visual rubric review" section.
"""

from __future__ import annotations

RUBRIC_VERSION = 1

# Schema definition: {region_kind -> {field_path: (kind, range)}}
#   kind: "score" -> int 1..5; "text" -> free-form string
COMMON_FIELDS = {
    "realism.painted_on_vs_pasted_on": ("score", (1, 5)),
    "realism.edge_seam_visibility": ("score", (1, 5)),
    "realism.texture_match": ("score", (1, 5)),
    "color.hue_match": ("score", (1, 5)),
    "color.brightness_match": ("score", (1, 5)),
    "color.saturation_match": ("score", (1, 5)),
    "geometry.perspective_plausibility": ("score", (1, 5)),
    "geometry.size_plausibility": ("score", (1, 5)),
    "notes": ("text", None),
}

WALKOVER_EXTRA = {
    "temporal.occlusion_realism": ("score", (1, 5)),
    "temporal.jitter_visible": ("score", (1, 5)),
    "temporal.player_contact_shadow": ("score", (1, 5)),
}


def schema_for(region_kind: str) -> dict[str, tuple[str, tuple[int, int] | None]]:
    """Return the rubric schema for a region kind."""
    fields = dict(COMMON_FIELDS)
    if region_kind in {"floor", "walkover"}:
        fields.update(WALKOVER_EXTRA)
    return fields


def validate_rubric(payload: dict, region_kind: str) -> tuple[bool, list[str]]:
    """Validate a rubric payload against the expected schema.

    Returns (is_valid, errors). On invalid: errors lists path-level issues.
    """
    schema = schema_for(region_kind)
    errors: list[str] = []
    for path, (kind, rng) in schema.items():
        value = _dotget(payload, path)
        if kind == "score":
            if not isinstance(value, int):
                errors.append(f"{path}: expected int, got {type(value).__name__}")
                continue
            lo, hi = rng or (1, 5)
            if value < lo or value > hi:
                errors.append(f"{path}: {value} outside [{lo}, {hi}]")
        elif kind == "text":
            if value is not None and not isinstance(value, str):
                errors.append(f"{path}: expected string or None, got {type(value).__name__}")
    return (len(errors) == 0, errors)


def min_score(payload: dict, region_kind: str) -> int | None:
    """Lowest score across all rubric dimensions for this region."""
    schema = schema_for(region_kind)
    scores = []
    for path, (kind, _) in schema.items():
        if kind == "score":
            v = _dotget(payload, path)
            if isinstance(v, int):
                scores.append(v)
    return min(scores) if scores else None


def _dotget(payload: dict, path: str):
    """Walk a dotted path into a nested dict; returns None if missing."""
    cur = payload
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


# Note: the SDK-based prompts that previously lived here have been removed.
# The visual rubric is now driven by `ai_review.write_manifest`, which emits a
# Markdown manifest a sub-agent (vision-capable Claude) reads directly via the
# Read tool. See `src/banner_pipeline/eval/ai_review.py`.
