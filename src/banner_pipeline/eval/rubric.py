"""AI visual-inspection rubric definition + lightweight schema validation.

The rubric is identical for the manual (conversation-Claude) and automated
(Anthropic API) paths, so scores are comparable. Documented in
`docs/EVALUATION.md` under the "AI visual inspection" section.
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


# Prompt body for the automated (Anthropic API) path. The system prompt is
# intentionally static so that prompt-cache hits across calls.
SYSTEM_PROMPT = """You are a visual quality reviewer for virtual ad insertions on tennis broadcast footage.

You will receive a set of cropped images from one placement region of one experiment run.
Your job is to fill out a strict rubric (scores 1-5, integer; 1 = obviously broken, 5 = indistinguishable from a real painted-on ad).

Output ONLY valid JSON matching the schema you are given. Do not include prose outside the JSON.
"""


def prompt_for_region(region_kind: str) -> str:
    """The user-message prompt body. Region-aware so walkover-only fields are surfaced."""
    schema = schema_for(region_kind)
    fields = "\n".join(
        f'  - "{path}" : {"int 1-5" if kind == "score" else "string"}'
        for path, (kind, _) in schema.items()
    )
    return f"""Region kind: {region_kind}

Rate the following dimensions (return JSON):

{fields}

Return JSON shaped like:
{{
  "realism": {{"painted_on_vs_pasted_on": <1-5>, ...}},
  "color":   {{"hue_match": <1-5>, ...}},
  "geometry":{{"perspective_plausibility": <1-5>, ...}},
  {('"temporal":{"occlusion_realism": <1-5>, ...},' if region_kind in {"floor", "walkover"} else "")}
  "notes": "<free text caveats>"
}}
"""
