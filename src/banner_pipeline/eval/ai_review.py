"""Anthropic API path for the AI visual-inspection rubric.

Opt-in via `--with-ai-review`. Uses Claude Opus 4.7 with vision and
prompt caching (5-min TTL) so the system prompt + region rubric body
amortize across regions of the same run AND across runs in a batch.

Manual / interactive path (conversation Claude reads PNGs via Read tool)
does not need this module — it produces the same rubric shape by hand.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from banner_pipeline.eval.rubric import (
    RUBRIC_VERSION,
    SYSTEM_PROMPT,
    min_score,
    prompt_for_region,
    validate_rubric,
)

MODEL_ID = "claude-opus-4-7"


def review_region(
    region_kind: str,
    image_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any] | None:
    """Run the rubric against `image_paths` for one region, persist outputs.

    Writes:
      - <output_dir>/<region>.json  (rubric payload + min_score)
      - <output_dir>/<region>.md    (prose summary; same content as 'notes' field)

    Returns the parsed rubric dict (with `min_score` injected) or None on
    failure (missing dep, bad response, etc.).
    """
    try:
        import anthropic
    except ImportError:
        print(
            f"[ai_review] anthropic package not installed; skipping {region_kind}. "
            "Run `uv sync --extra ai` to enable."
        )
        return None

    client = anthropic.Anthropic()
    contents: list[dict[str, Any]] = []
    for p in image_paths:
        path = Path(p)
        if not path.is_file():
            continue
        with path.open("rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
        contents.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }
        )
    if not contents:
        return None
    contents.append({"type": "text", "text": prompt_for_region(region_kind)})

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": contents}],
    )

    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    payload = _extract_json(text)
    if payload is None:
        return None
    valid, errors = validate_rubric(payload, region_kind)
    if not valid:
        print(f"[ai_review] {region_kind} rubric validation failed: {errors}")

    payload["min_score"] = min_score(payload, region_kind)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{region_kind}.json").write_text(json.dumps(payload, indent=2))
    notes = str(payload.get("notes") or "").strip()
    (out_dir / f"{region_kind}.md").write_text(
        f"# AI Review — {region_kind}\n\nMin score: {payload['min_score']}\n\n{notes}\n"
    )
    return payload


def write_rubric_version(output_dir: str | Path) -> None:
    """Persist the rubric version so future calibration knows the schema."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "rubric_version.json").write_text(
        json.dumps({"rubric_version": RUBRIC_VERSION, "model_id": MODEL_ID}, indent=2)
    )


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json"):].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
