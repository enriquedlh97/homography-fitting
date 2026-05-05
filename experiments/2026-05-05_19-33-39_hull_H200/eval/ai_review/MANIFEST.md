# Visual rubric review — manifest

**For:** a general-purpose sub-agent (Claude with vision) dispatched via the Agent tool. Pass this file's content as the agent's instructions, plus the region inputs below.

**How:** the agent reads each listed PNG via the **Read tool** — that returns the image content for the agent to actually see, exactly the same as if a user pasted the image into the chat. The agent scores the rubric from its own visual judgment, then writes the output files. **No SDK calls. No external API. No `--with-ai-review` flag needed.**

## Inputs (per region)

### Region: `back`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/crops_strip.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/motion_strip_early.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/motion_strip_late.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/motion_strip_mid.png`

### Region: `left`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/crops_strip.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/motion_strip_early.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/motion_strip_late.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/motion_strip_mid.png`

### Region: `floor`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/crops_strip.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/motion_strip_early.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/motion_strip_late.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/motion_strip_mid.png`

### Region: `full`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/full/crops_strip.png`

### Region: `walkover`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/consecutive_frames.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_contact_f0704.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_entry_f0685.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_exit_f0723.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_post_contact_f0713.png`
- `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_pre_contact_f0694.png`

**Walkover window:** frames `685`–`723` (inclusive). Pay extra attention to the walkover region — that's where the player walks across the court-floor logo.

## Rubric (per region)

Score each dimension as an **integer 1–5** (1 = clearly broken / pasted-on; 5 = indistinguishable from a real painted-on ad). All `notes` fields are free-form prose, ≤150 words per region.

### Calibration callouts (rubric v2 — DO NOT collapse these to 5)

Two artifact families were under-scored by rubric v1 and are now explicit dimensions on every surface-bearing region (`back`, `left`, `floor`, `walkover`). Score these honestly — a v1 reviewer rated a run 5/5/5/5/5 on a clip the user immediately flagged for both:

- **`realism.halo_presence`** — is there a **halo around the logo**? Bright glow / luminance halo at the logo perimeter that does NOT match the matte court paint or banner fabric. The user-flagged canonical case is the Red Bull *floor* logo on Melbourne with a visible halo around the wordmark and bulls. **A halo wider than ~1-2 px on the matte court is a 2 (visible glow), not a 5.**
- **`realism.edge_reflex`** — is there a **reflex / smearing at letter edges**? A subtle ghost / drift / ringing along letter and icon edges (NOT a mirror reflection — the artifact is more like a faint duplicate or smeared echo). The user-flagged canonical case is the Red Bull *left side banner* where the rings around the bulls show paint drift and the letters show a reflex at top and bottom edges. **Visible smearing or letter-edge ghost on side banners is a 1-2, not a 5.**

Read the original (top row) carefully: it is the baked-in advertiser ad and the quality bar. If you can spot the halo or reflex on the bottom row but the top row is clean, score the dimension 1-2.

### `back`
- `realism.painted_on_vs_pasted_on` — integer 1–5
- `realism.edge_seam_visibility` — integer 1–5
- `realism.texture_match` — integer 1–5
- `color.hue_match` — integer 1–5
- `color.brightness_match` — integer 1–5
- `color.saturation_match` — integer 1–5
- `geometry.perspective_plausibility` — integer 1–5
- `geometry.size_plausibility` — integer 1–5
- `notes` — free-form text
- `realism.halo_presence` — integer 1–5
- `realism.edge_reflex` — integer 1–5

### `left`
- `realism.painted_on_vs_pasted_on` — integer 1–5
- `realism.edge_seam_visibility` — integer 1–5
- `realism.texture_match` — integer 1–5
- `color.hue_match` — integer 1–5
- `color.brightness_match` — integer 1–5
- `color.saturation_match` — integer 1–5
- `geometry.perspective_plausibility` — integer 1–5
- `geometry.size_plausibility` — integer 1–5
- `notes` — free-form text
- `realism.halo_presence` — integer 1–5
- `realism.edge_reflex` — integer 1–5

### `floor`
- `realism.painted_on_vs_pasted_on` — integer 1–5
- `realism.edge_seam_visibility` — integer 1–5
- `realism.texture_match` — integer 1–5
- `color.hue_match` — integer 1–5
- `color.brightness_match` — integer 1–5
- `color.saturation_match` — integer 1–5
- `geometry.perspective_plausibility` — integer 1–5
- `geometry.size_plausibility` — integer 1–5
- `notes` — free-form text
- `realism.halo_presence` — integer 1–5
- `realism.edge_reflex` — integer 1–5
- `temporal.occlusion_realism` — integer 1–5
- `temporal.jitter_visible` — integer 1–5
- `temporal.player_contact_shadow` — integer 1–5

### `full`
- `realism.painted_on_vs_pasted_on` — integer 1–5
- `realism.edge_seam_visibility` — integer 1–5
- `realism.texture_match` — integer 1–5
- `color.hue_match` — integer 1–5
- `color.brightness_match` — integer 1–5
- `color.saturation_match` — integer 1–5
- `geometry.perspective_plausibility` — integer 1–5
- `geometry.size_plausibility` — integer 1–5
- `notes` — free-form text

### `walkover`
- `realism.painted_on_vs_pasted_on` — integer 1–5
- `realism.edge_seam_visibility` — integer 1–5
- `realism.texture_match` — integer 1–5
- `color.hue_match` — integer 1–5
- `color.brightness_match` — integer 1–5
- `color.saturation_match` — integer 1–5
- `geometry.perspective_plausibility` — integer 1–5
- `geometry.size_plausibility` — integer 1–5
- `notes` — free-form text
- `realism.halo_presence` — integer 1–5
- `realism.edge_reflex` — integer 1–5
- `temporal.occlusion_realism` — integer 1–5
- `temporal.jitter_visible` — integer 1–5
- `temporal.player_contact_shadow` — integer 1–5

## Output the agent must write

Per region in `region_inputs`, write two files into `eval/ai_review/` next to this MANIFEST.md:

- `<region>.json` — JSON object matching the rubric schema, with an extra top-level `min_score` field equal to the lowest integer score across all dimensions for that region.
- `<region>.md` — short prose ('what would a human viewer notice if they scrubbed this region?'), under 150 words.

Plus one report-level file:

- `eval/ai_review/CHECKLIST.md` — fill out a verbatim copy of the checklist below, marking `[x]` for every artifact you actually Read and `[ ]` for any you skipped (with a one-line reason).

## Required visual-eval checklist (paste into CHECKLIST.md and into your final report)

```
[VISUAL EVAL CHECKLIST]
## back
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/crops_strip.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/motion_strip_early.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/motion_strip_late.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/back_banners/motion_strip_mid.png`
- [ ] wrote `eval/ai_review/back.json`
- [ ] wrote `eval/ai_review/back.md`
## left
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/crops_strip.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/motion_strip_early.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/motion_strip_late.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/left_logo/motion_strip_mid.png`
- [ ] wrote `eval/ai_review/left.json`
- [ ] wrote `eval/ai_review/left.md`
## floor
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/crops_strip.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/motion_strip_early.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/motion_strip_late.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/floor_logo/motion_strip_mid.png`
- [ ] wrote `eval/ai_review/floor.json`
- [ ] wrote `eval/ai_review/floor.md`
## full
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/full/crops_strip.png`
- [ ] wrote `eval/ai_review/full.json`
- [ ] wrote `eval/ai_review/full.md`
## walkover
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/consecutive_frames.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_contact_f0704.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_entry_f0685.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_exit_f0723.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_post_contact_f0713.png`
- [ ] read `experiments/2026-05-05_19-33-39_hull_H200/eval/walkover/forensic_sheet_pre_contact_f0694.png`
- [ ] wrote `eval/ai_review/walkover.json`
- [ ] wrote `eval/ai_review/walkover.md`
[/VISUAL EVAL CHECKLIST]
```

**The manager (parent agent) will grep this checklist for unchecked `[ ]` entries.** Any unchecked item without a one-line reason in the same line is treated as 'agent skipped this without saying so' — which is worse than reporting a problem honestly. If you cannot read a PNG (file missing, decode error), say so explicitly: `- [ ] read X — FILE MISSING` or similar.

Example JSON shape:

```json
{
  "realism": {"painted_on_vs_pasted_on": 5, "edge_seam_visibility": 4, "texture_match": 5,
              "halo_presence": 2, "edge_reflex": 3},
  "color":   {"hue_match": 5, "brightness_match": 5, "saturation_match": 5},
  "geometry":{"perspective_plausibility": 5, "size_plausibility": 5},
  "temporal":{"occlusion_realism": 4, "jitter_visible": 5, "player_contact_shadow": 3},
  "notes": "Visible bright halo around the logo on the matte court paint; rings around the bulls show subtle smearing.",
  "min_score": 2
}
```

(`temporal.*` only applies to `floor` / `walkover` regions. `realism.halo_presence` and `realism.edge_reflex` apply to `back`, `left`, `floor`, `walkover` — every region that paints a logo onto a real-world surface. They do NOT apply to `full`.)

## Suggested sub-agent prompt template

```
You are a visual quality reviewer for virtual ad insertions on tennis broadcast
footage. Read MANIFEST.md at `<run>/eval/ai_review/MANIFEST.md`, then Read each PNG
listed under each Region. The Read tool returns the image content for you to see
(your own vision IS the rubric — do not call any external API or SDK).

For each region, fill out the rubric (integer scores 1–5 per dimension; 1 = clearly
broken, 5 = indistinguishable from a real painted-on ad). Write the output files
into the same `ai_review/` directory:
  - <region>.json (strict JSON, with `min_score` injected)
  - <region>.md (short prose; under 150 words)
  - CHECKLIST.md (paste the [VISUAL EVAL CHECKLIST] verbatim, [x] each PNG you read)

If a region has no PNGs listed, skip it. Pay extra attention to the walkover
window if one is present — that's where the player walks on the court-floor logo.

BEFORE submitting your final report:
  1. Count `[x] read` entries in CHECKLIST.md. Verify it equals the number of PNGs
     listed in this manifest. If different, recount — your summary `Total PNGs read`
     line MUST match the manifest's listed count exactly.
  2. Use the original (top-row) frames as your quality bar. The originals are real
     baked-in ads (Kia, YoPRO, Melbourne wordmark, etc.). Score `realism.painted_on_
     vs_pasted_on` by asking: 'does our composite read as natural as the original ad
     does?' Without that comparison the rubric collapses to 'looks fine to me'.
  3. EXPLICITLY check for the two rubric-v2 calibration artifacts on every
     surface-bearing region (back, left, floor, walkover):
       - `realism.halo_presence`: is there a halo around the logo? A bright glow at
         the logo perimeter that does not match the matte court paint or banner
         fabric. The Red Bull floor logo on Melbourne is the canonical halo case.
       - `realism.edge_reflex`: is there a reflex / smearing at letter edges? A
         subtle ghost or ringing along letter and icon edges (NOT a mirror
         reflection). The Red Bull left-banner letters and bull rings are the
         canonical reflex / smearing case.
     Both were under-scored to 5 by the v1 rubric on runs the user immediately
     flagged. If you can spot them on the bottom row but the top-row baked-in ad is
     clean, that is a 1-2, not a 5. Do NOT round up.
```
