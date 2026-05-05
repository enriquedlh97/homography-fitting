"""Visual rubric review — manifest emitter (sub-agent vision path).

The eval framework emits a manifest listing the per-region crop PNGs and the
rubric schema. A sub-agent (general-purpose Claude with vision) is dispatched
SEPARATELY by the user/manager to read the PNGs via the Read tool and write
the rubric output files. There is no Anthropic SDK dependency — the
sub-agent's own vision IS the rubric.

This is the same pattern the C006/A1, C007/A1, C010/A1, C016/A1 visual
comparators used during the 2026-05-04 autonomous experimentation cycle.

Outputs WRITTEN BY THIS MODULE (always emitted):
  eval/ai_review/MANIFEST.md         — agent-readable prompt + PNG list + rubric
  eval/ai_review/rubric_version.json — schema version metadata

Outputs WRITTEN BY THE SUB-AGENT (not the framework):
  eval/ai_review/<region>.json       — rubric scores per region
  eval/ai_review/<region>.md         — short prose ('what would a viewer notice')
"""

from __future__ import annotations

import json
from pathlib import Path

from banner_pipeline.eval.rubric import RUBRIC_VERSION, schema_for


REGION_DIR_NAMES: dict[str, str] = {
    "back": "back_banners",
    "left": "left_logo",
    "floor": "floor_logo",
    "full": "full",
    "walkover": "walkover",
}


def write_manifest(
    eval_dir: str | Path,
    regions_present: list[str],
    walkover_window: tuple[int, int] | None = None,
) -> Path:
    """Write a manifest a sub-agent can consume to score the rubric on this run.

    The manifest enumerates the per-region crop PNGs already produced by the
    eval framework and pairs each region with its rubric schema. The
    sub-agent reads each listed PNG via the Read tool — that returns image
    content for vision-capable Claudes — scores the rubric, then writes
    `<region>.{json,md}` files into the same `ai_review/` directory.

    Returns the path to the MANIFEST.md so callers can print it.
    """
    eval_dir = Path(eval_dir)
    out_dir = eval_dir / "ai_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "rubric_version.json").write_text(
        json.dumps({"rubric_version": RUBRIC_VERSION}, indent=2) + "\n"
    )

    region_inputs: dict[str, list[str]] = {}
    # Paths are relative to the repo root (which is two levels above experiments/<run>/eval/).
    repo_root = eval_dir.parent.parent.parent
    for kind in regions_present:
        sub = REGION_DIR_NAMES.get(kind, kind)
        region_dir = eval_dir / sub
        if not region_dir.is_dir():
            continue
        pngs = sorted(p for p in region_dir.glob("*.png"))
        if not pngs:
            continue
        try:
            rels = [str(p.relative_to(repo_root)) for p in pngs]
        except ValueError:
            rels = [str(p) for p in pngs]
        region_inputs[kind] = rels

    manifest_path = out_dir / "MANIFEST.md"
    manifest_path.write_text(_format_manifest_md(region_inputs, walkover_window))
    return manifest_path


def load_existing_rubric(eval_dir: str | Path, region_kind: str) -> dict | None:
    """Read back a sub-agent-written rubric JSON for one region, if present."""
    path = Path(eval_dir) / "ai_review" / f"{region_kind}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _format_manifest_md(
    region_inputs: dict[str, list[str]],
    walkover_window: tuple[int, int] | None,
) -> str:
    """Format the manifest as a copy-paste-ready sub-agent prompt."""
    lines: list[str] = []
    lines.append("# Visual rubric review — manifest")
    lines.append("")
    lines.append(
        "**For:** a general-purpose sub-agent (Claude with vision) dispatched via the "
        "Agent tool. Pass this file's content as the agent's instructions, plus the "
        "region inputs below."
    )
    lines.append("")
    lines.append(
        "**How:** the agent reads each listed PNG via the **Read tool** — that "
        "returns the image content for the agent to actually see, exactly the same "
        "as if a user pasted the image into the chat. The agent scores the rubric "
        "from its own visual judgment, then writes the output files. **No SDK "
        "calls. No external API. No `--with-ai-review` flag needed.**"
    )
    lines.append("")
    lines.append("## Inputs (per region)")
    lines.append("")
    if not region_inputs:
        lines.append("_No region PNGs found. Re-run `python -m banner_pipeline.eval --experiment <dir>` first._")
        lines.append("")
    for region, paths in region_inputs.items():
        lines.append(f"### Region: `{region}`")
        for p in paths:
            lines.append(f"- `{p}`")
        lines.append("")

    if walkover_window is not None:
        lines.append(f"**Walkover window:** frames `{walkover_window[0]}`–`{walkover_window[1]}` (inclusive). Pay extra attention to the walkover region — that's where the player walks across the court-floor logo.")
        lines.append("")

    lines.append("## Rubric (per region)")
    lines.append("")
    lines.append("Score each dimension as an **integer 1–5** (1 = clearly broken / pasted-on; 5 = indistinguishable from a real painted-on ad). All `notes` fields are free-form prose, ≤150 words per region.")
    lines.append("")
    lines.append("### Calibration callouts (rubric v2 — DO NOT collapse these to 5)")
    lines.append("")
    lines.append("Two artifact families were under-scored by rubric v1 and are now explicit dimensions on every surface-bearing region (`back`, `left`, `floor`, `walkover`). Score these honestly — a v1 reviewer rated a run 5/5/5/5/5 on a clip the user immediately flagged for both:")
    lines.append("")
    lines.append("- **`realism.halo_presence`** — is there a **halo around the logo**? Bright glow / luminance halo at the logo perimeter that does NOT match the matte court paint or banner fabric. The user-flagged canonical case is the Red Bull *floor* logo on Melbourne with a visible halo around the wordmark and bulls. **A halo wider than ~1-2 px on the matte court is a 2 (visible glow), not a 5.**")
    lines.append("- **`realism.edge_reflex`** — is there a **reflex / smearing at letter edges**? A subtle ghost / drift / ringing along letter and icon edges (NOT a mirror reflection — the artifact is more like a faint duplicate or smeared echo). The user-flagged canonical case is the Red Bull *left side banner* where the rings around the bulls show paint drift and the letters show a reflex at top and bottom edges. **Visible smearing or letter-edge ghost on side banners is a 1-2, not a 5.**")
    lines.append("")
    lines.append("Read the original (top row) carefully: it is the baked-in advertiser ad and the quality bar. If you can spot the halo or reflex on the bottom row but the top row is clean, score the dimension 1-2.")
    lines.append("")
    for region in region_inputs:
        schema = schema_for(region if region != "walkover" else "walkover")
        lines.append(f"### `{region}`")
        for path, (kind, _) in schema.items():
            lines.append(f"- `{path}` — {'integer 1–5' if kind == 'score' else 'free-form text'}")
        lines.append("")

    lines.append("## Output the agent must write")
    lines.append("")
    lines.append("Per region in `region_inputs`, write two files into `eval/ai_review/` next to this MANIFEST.md:")
    lines.append("")
    lines.append("- `<region>.json` — JSON object matching the rubric schema, with an extra top-level `min_score` field equal to the lowest integer score across all dimensions for that region.")
    lines.append("- `<region>.md` — short prose ('what would a human viewer notice if they scrubbed this region?'), under 150 words.")
    lines.append("")
    lines.append("Plus one report-level file:")
    lines.append("")
    lines.append("- `eval/ai_review/CHECKLIST.md` — fill out a verbatim copy of the checklist below, marking `[x]` for every artifact you actually Read and `[ ]` for any you skipped (with a one-line reason).")
    lines.append("")
    lines.append("## Required visual-eval checklist (paste into CHECKLIST.md and into your final report)")
    lines.append("")
    lines.append("```")
    lines.append("[VISUAL EVAL CHECKLIST]")
    for region, paths in region_inputs.items():
        lines.append(f"## {region}")
        for p in paths:
            lines.append(f"- [ ] read `{p}`")
        lines.append(f"- [ ] wrote `eval/ai_review/{region}.json`")
        lines.append(f"- [ ] wrote `eval/ai_review/{region}.md`")
    lines.append("[/VISUAL EVAL CHECKLIST]")
    lines.append("```")
    lines.append("")
    lines.append("**The manager (parent agent) will grep this checklist for unchecked `[ ]` entries.** Any unchecked item without a one-line reason in the same line is treated as 'agent skipped this without saying so' — which is worse than reporting a problem honestly. If you cannot read a PNG (file missing, decode error), say so explicitly: `- [ ] read X — FILE MISSING` or similar.")
    lines.append("")
    lines.append("Example JSON shape:")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "realism": {"painted_on_vs_pasted_on": 5, "edge_seam_visibility": 4, "texture_match": 5,')
    lines.append('              "halo_presence": 2, "edge_reflex": 3},')
    lines.append('  "color":   {"hue_match": 5, "brightness_match": 5, "saturation_match": 5},')
    lines.append('  "geometry":{"perspective_plausibility": 5, "size_plausibility": 5},')
    lines.append('  "temporal":{"occlusion_realism": 4, "jitter_visible": 5, "player_contact_shadow": 3},')
    lines.append('  "notes": "Visible bright halo around the logo on the matte court paint; rings around the bulls show subtle smearing.",')
    lines.append('  "min_score": 2')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("(`temporal.*` only applies to `floor` / `walkover` regions. `realism.halo_presence` and `realism.edge_reflex` apply to `back`, `left`, `floor`, `walkover` — every region that paints a logo onto a real-world surface. They do NOT apply to `full`.)")
    lines.append("")
    lines.append("## Suggested sub-agent prompt template")
    lines.append("")
    lines.append("```")
    lines.append("You are a visual quality reviewer for virtual ad insertions on tennis broadcast")
    lines.append("footage. Read MANIFEST.md at `<run>/eval/ai_review/MANIFEST.md`, then Read each PNG")
    lines.append("listed under each Region. The Read tool returns the image content for you to see")
    lines.append("(your own vision IS the rubric — do not call any external API or SDK).")
    lines.append("")
    lines.append("For each region, fill out the rubric (integer scores 1–5 per dimension; 1 = clearly")
    lines.append("broken, 5 = indistinguishable from a real painted-on ad). Write the output files")
    lines.append("into the same `ai_review/` directory:")
    lines.append("  - <region>.json (strict JSON, with `min_score` injected)")
    lines.append("  - <region>.md (short prose; under 150 words)")
    lines.append("  - CHECKLIST.md (paste the [VISUAL EVAL CHECKLIST] verbatim, [x] each PNG you read)")
    lines.append("")
    lines.append("If a region has no PNGs listed, skip it. Pay extra attention to the walkover")
    lines.append("window if one is present — that's where the player walks on the court-floor logo.")
    lines.append("")
    lines.append("BEFORE submitting your final report:")
    lines.append("  1. Count `[x] read` entries in CHECKLIST.md. Verify it equals the number of PNGs")
    lines.append("     listed in this manifest. If different, recount — your summary `Total PNGs read`")
    lines.append("     line MUST match the manifest's listed count exactly.")
    lines.append("  2. Use the original (top-row) frames as your quality bar. The originals are real")
    lines.append("     baked-in ads (Kia, YoPRO, Melbourne wordmark, etc.). Score `realism.painted_on_")
    lines.append("     vs_pasted_on` by asking: 'does our composite read as natural as the original ad")
    lines.append("     does?' Without that comparison the rubric collapses to 'looks fine to me'.")
    lines.append("  3. EXPLICITLY check for the two rubric-v2 calibration artifacts on every")
    lines.append("     surface-bearing region (back, left, floor, walkover):")
    lines.append("       - `realism.halo_presence`: is there a halo around the logo? A bright glow at")
    lines.append("         the logo perimeter that does not match the matte court paint or banner")
    lines.append("         fabric. The Red Bull floor logo on Melbourne is the canonical halo case.")
    lines.append("       - `realism.edge_reflex`: is there a reflex / smearing at letter edges? A")
    lines.append("         subtle ghost or ringing along letter and icon edges (NOT a mirror")
    lines.append("         reflection). The Red Bull left-banner letters and bull rings are the")
    lines.append("         canonical reflex / smearing case.")
    lines.append("     Both were under-scored to 5 by the v1 rubric on runs the user immediately")
    lines.append("     flagged. If you can spot them on the bottom row but the top-row baked-in ad is")
    lines.append("     clean, that is a 1-2, not a 5. Do NOT round up.")
    lines.append("```")
    return "\n".join(lines) + "\n"
