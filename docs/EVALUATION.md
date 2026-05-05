# Evaluation Framework

Canonical reference for evaluating virtual banner placement runs. Sub-agents working in this repo MUST follow this document. Supersedes `docs/evaluation-protocol.md`.

## What this evaluates

The pipeline produces one composited video per run. That video contains up to **5 placed regions simultaneously**:

| obj_id | Region kind | Surface | Driving config field |
|--------|-------------|---------|----------------------|
| 1, 2, 5 | back-wall black banners | `banner` | `prompts[i].placement_quad` |
| 4 | left-side Red Bull side banner | `banner` | `prompts[i].compositor_params.logo_placement_quad` |
| 3 | court-floor Red Bull walkover logo | `court_floor` | `prompts[i].placement_quad` |

Each region is judged independently. Plus a **full-frame** rollup for global temporal consistency. For obj_3 (court floor), an additional **walkover-window** evaluation looks specifically at frames where the player overlaps the logo.

## Test clips

Evaluate on ALL clips before considering a change valid. Generalization is a hard requirement.

| Clip | Path | Purpose | Frames | FPS |
|------|------|---------|--------|-----|
| Original | `data/tennis-clip.mp4` | Back banner, court floor, side panels — baseline | 204 | 25 |
| Walking-over | `data/melbourne-walking-over-logo.mov` | Player occlusion of court-floor logo | 778 | 59 |
| Zoom | `data/zoom-clip-melbourne.mov` | Camera zoom stability | 327 | 56 |

A change that improves one clip but degrades another is a regression. Don't ship.

## How to run

### Single command, post-hoc on any experiment dir

```bash
uv run python -m banner_pipeline.eval \
    --experiment experiments/<run_dir>/ \
    [--reference auto]                                # auto-resolves via configs/eval/reference.yaml
    [--regions back,left,floor,full,walkover]        # subset
    [--walkover-window 690:745]                       # override auto-detect
    [--with-ai-review]                                # deprecated; visual rubric MANIFEST.md is now always emitted
```

The visual rubric `MANIFEST.md` is **always** written under `eval/ai_review/`; the rubric itself is scored by dispatching a sub-agent that reads the crop PNGs via the Read tool. No SDK, no API key. See the "Visual rubric review" section below.

Exit codes:
- **`0`** — every per-region scorecard passes AND no regression vs gold
- **`2`** — at least one per-region scorecard fails (independent of reference)
- **`3`** — all scorecards pass but a metric regressed vs gold reference
- **`1`** — framework error (bad path, unreadable config, etc.)

### Chained with a fresh run

```bash
scripts/run_and_eval.sh configs/experiments/<your_yaml>.yaml
```

This calls `scripts/run_experiment.py` (or `scripts/modal_run.py` if `MODAL=1`) and then `scripts/eval_run.py` on the resulting experiment dir. The launchers are not modified.

### Ad-hoc on an existing dir without a fresh run

The framework works on any experiment dir that has at minimum `outputs/composited.mp4` + `config.yaml`. Geometric jitter metrics are richer when `outputs/per_frame_state.json` is also present (emitted by the pipeline since v… of `pipeline.py`); without it, those metrics fall back to the static `placement_quad` from the frozen config (returning 0-jitter, marked with `geometric_source: "static_fallback"`). All other metrics work in both modes.

## What gets produced

```
experiments/<run>/
  outputs/
    composited.mp4
    per_frame_state.json                  # per-frame, per-object quad corners (when pipeline emits it)
  eval/
    quality_metrics.json                  # flat top-level dict, machine-readable
    report.md                             # human rollup with embedded image paths
    back_banners/
      metrics.json
      crops_strip.png                     # 6 evenly-spaced frames; TOP=original / BOTTOM=composite paired
      motion_strip_early.png              # 8 frames spanning ~0.5s at 15% of clip; paired
      motion_strip_mid.png                # ~0.5s at 50%; paired
      motion_strip_late.png               # ~0.5s at 85%; paired
      vs_reference.png                    # only with --reference
    left_logo/                            # same shape, obj_id 4
    floor_logo/                           # same shape, obj_id 3
    full/
      metrics.json
      crops_strip.png                     # 6 full frames paired
      vs_reference.png
    walkover/
      window.json                         # {"start": 690, "end": 745, "method": "..."}
      consecutive_frames.png              # ~16 sampled frames across window; TOP=original / BOTTOM=composite paired
      forensic_sheet_entry_f<N>.png       # 6-col forensic sheet at walkover window start
      forensic_sheet_pre_contact_f<N>.png # 25% across window
      forensic_sheet_contact_f<N>.png     # mid window (player ON the logo)
      forensic_sheet_post_contact_f<N>.png # 75% across window
      forensic_sheet_exit_f<N>.png        # window end
      # 6 columns each: orig | clean | composite | original-clean delta | survival | leak overlay
      occlusion_diagnostic.csv            # (when produced) per-frame leak_ratio, logo_visible_pct, occlusion_iou
    ai_review/
      MANIFEST.md                         # ALWAYS written: agent prompt + PNG paths + rubric schema + checklist
      rubric_version.json                 # ALWAYS written: schema version metadata
      back.{json,md}                      # written by sub-agent after dispatch (vision via Read)
      left.{json,md}
      floor.{json,md}                     # includes temporal.* (occlusion_realism, jitter_visible, player_contact_shadow)
      walkover.{json,md}                  # same temporal.* fields; the hardest/most-important region
      full.{json,md}
      CHECKLIST.md                        # written by sub-agent: every PNG marked [x] read or [ ] skipped + reason
    vs_reference_side_by_side.mp4         # only with --reference
```

**All per-region strips are paired** (top row = unmodified original broadcast, bottom row = our composite). The original IS the ground-truth quality bar — the real Kia/Melbourne/etc. ads that were in the broadcast. If our virtual ad reads as natural as the original did, we've succeeded. Without the pairing, agents have no anchor for the rubric and scores collapse to "everything looks fine."

## Quantitative metrics

Each per-region `metrics.json` is a subset; `eval/quality_metrics.json` is the flat union.

### Gated metrics (a per-region scorecard `pass` = all gated metrics pass)

| Metric | Back (1,2,5) | Left (4) | Floor (3) | Full | Threshold | Source |
|---|---|---|---|---|---|---|
| `corner_max_jump_px` | ✓ | ✓ | ✓ | — | < 2.0 | per_frame_state corner traj |
| `quad_area_cv` | ✓ | ✓ | ✓ | — | < 0.05 | per_frame_state corner traj |
| `corner_accel_p95_px` | ✓ | ✓ | ✓ | — | < 1.0 | per_frame_state corner traj |
| `roi_jitter_ratio` | ✓ | ✓ | ✓ | ✓ | ≤ 1.05 | mean abs frame-diff in ROI vs same ROI in original |
| `roi_temporal_ssim_mean` | ✓ | ✓ | ✓ | ✓ | > 0.95 | mean SSIM between consecutive ROI frames |
| `walkover_logo_visible_pct` | — | — | ✓ | — | > 0.10 (calibrated to v68 gold; raise as we accumulate runs) | fraction of placement_quad pixels with logo signal in non-player area |
| `walkover_occlusion_iou` | — | — | ✓ | — | > 0.80 (calibrate after more runs) | IoU between original-vs-clean delta and composite-vs-baked-logo delta inside placement_quad |

### Warning-only metrics (don't gate exit code, surfaced in report.md)

| Metric | Threshold | Meaning |
|---|---|---|
| `noise_variance_ratio` | < 0.30 | "Too clean" — placed region noise is much lower than neighbor patch noise; logo looks pasted-on |
| `edge_sharpness_ratio` | > 1.8 | Gradients on placement_quad boundary much higher than the rest of the frame; visible cutout edge |
| `roi_delta_E_lab` | > 5.0 | Lab ΔE between ROI mean and same-surface neighbor patch. **Warning-only**: a placed region is *supposed* to differ from its neighbor when carrying a logo; metric needs replacement with a "did we preserve the underlying surface color" measurement. Surfaces problems but does not gate. |
| `ai_review_*` | per rubric (1-5 scale, 1 = bad) | Vision-LLM rubric scores (warning-only at first; calibrate after ~10 graded runs) |

### Reference (gold) comparison — only when `--reference auto` resolves

| Metric | Regression flag |
|---|---|
| `corner_distance_p95_px` | regression if Δ > 5% over baseline |
| `roi_ssim_vs_reference` | regression if Δ < −5% |
| `roi_dE_vs_reference` | regression if Δ > 5% |
| `walkover_occlusion_iou` (delta vs gold) | regression if Δ < −5% |
| `any_regression` (top-level) | OR of all above |

A `regression: true` flag flips exit code to `3` even when all scorecards pass on their own.

## Walkover-specific evaluation

The court-floor logo (obj_3) is the most demanding — the player physically walks over it. The framework auto-detects the walkover window via clean-vs-original luminance delta inside the floor's `placement_quad`, smoothed (5-frame box), thresholded at `mu + 2*sigma`, longest contiguous run, padded ±10 frames. CLI override: `--walkover-window 690:745`.

For each frame in the window:
- `logo_visible_pct` — fraction of `placement_quad` pixels showing logo signal (composite-vs-clean delta above threshold) in player-absent regions
- `occlusion_iou` — IoU between `(|original − clean| > T)` and `(|composite − clean_with_logo_baked| > T)` inside `placement_quad`. The "baked logo" comes from the gold composited.mp4 at the same frame index (no warping required when both runs share the input video).
- `forensic_sheet_f<N>.png` — 6-column horizontal contact sheet: `original | clean | composite | original-clean delta heatmap | survival heatmap | suspected leak red overlay`

## Visual inspection — three layers

### Layer 1: PNG crops (always produced — automatic)

Every eval run produces ~15 crop PNGs across 5 region directories. **These are the inputs the visual rubric reviewer reads.** Generated by `scripts/eval_run.py` / `python -m banner_pipeline.eval` from each object's `placement_quad` in the frozen config:

```
eval/back_banners/
  crops_strip.png            # 6 evenly-spaced frames, 3× upscaled
  consecutive_frames_early.png  # 8 contiguous frames at 15% of clip
  consecutive_frames_mid.png    # 8 contiguous frames at 50%
  consecutive_frames_late.png   # 8 contiguous frames at 85%
eval/left_logo/              # same shape
eval/floor_logo/             # same shape
eval/full/
  crops_strip.png            # 6 evenly-spaced full frames (no upscale)
eval/walkover/
  consecutive_frames.png     # every frame in the auto-detected walkover window (~15-20 frames)
  forensic_sheet_f<N>.png    # 6-column orig|clean|composite|deltas|leak overlay
```

PNG strips are the floor of "scan in seconds" review. **Confirmed produced on every run** — see `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/eval/` for an example.

### Layer 2: side-by-side video (only with `--reference`)

`eval/vs_reference_side_by_side.mp4` stacks `current_composited | gold_composited | abs_diff_heatmap` horizontally at 0.5× width per panel. Lets you scrub motion against the gold without alt-tabbing between videos.

### Layer 3: Visual rubric review (sub-agent vision — no SDK)

**The rubric is scored by a sub-agent (general-purpose Claude) reading the crop PNGs via the Read tool — exactly the same as if a user pasted an image into the chat. There is no Anthropic SDK call, no API key, no `ANTHROPIC_API_KEY` to provision. The agent's own vision IS the rubric.**

This is the same pattern that ran the C006/A1, C007/A1, C010/A1, C016/A1 visual comparators during the 2026-05-04 autonomous experimentation cycle.

#### How it works

1. The eval framework writes `eval/ai_review/MANIFEST.md` on every run. The manifest enumerates the per-region crop PNGs and the rubric schema. (Always written; no flag needed.)
2. Whoever wants the rubric scored — the human reviewer or a manager-agent dispatching a sub-agent — uses the manifest's content as the sub-agent's prompt. The manifest doubles as a copy-paste-ready prompt template.
3. The sub-agent (general-purpose subagent_type, dispatched via the Agent tool) reads each listed PNG via Read, scores the rubric from its own visual judgment, and writes:
   - `eval/ai_review/<region>.json` — strict-shaped JSON with `min_score` injected
   - `eval/ai_review/<region>.md` — short prose ("what would a human viewer notice"), under 150 words.
4. The next time `python -m banner_pipeline.eval --experiment <dir>` runs, it picks up any sub-agent-written rubric files and surfaces `ai_review_min_score` keys per region in `eval/quality_metrics.json`.

#### Rubric (every region; some fields walkover-only)

```
realism:
  painted_on_vs_pasted_on  : 1-5   # 1 = obviously pasted, 5 = looks painted on the surface
  edge_seam_visibility     : 1-5   # 1 = visible cutout, 5 = invisible blend
  texture_match            : 1-5   # 1 = wrong material, 5 = matches surface texture
color:
  hue_match                : 1-5
  brightness_match         : 1-5
  saturation_match         : 1-5
geometry:
  perspective_plausibility : 1-5   # quad orientation looks right for the camera angle
  size_plausibility        : 1-5   # logo size is what a real ad would be
temporal (floor / walkover only):
  occlusion_realism        : 1-5   # 1 = obviously broken (logo bleeds onto player), 5 = player walks on logo
  jitter_visible           : 1-5   # 1 = jumpy, 5 = locked
  player_contact_shadow    : 1-5   # 1 = no shadow under foot, 5 = realistic contact
notes: <free-text, ≤150 words>
```

Mapped from the legacy 10-point visual QA checklist (in `docs/evaluation-protocol.md`) — preserved as rubric dimensions so different reviewers' scores are comparable.

Rubric scores are **warning-only** at first; never gate exit code. After ~10 graded runs we calibrate which rubric dimensions become gates.

#### Sub-agent prompt template (use verbatim or as a starting point)

```
You are a visual quality reviewer for virtual ad insertions on tennis broadcast footage.
Read MANIFEST.md at <run>/eval/ai_review/MANIFEST.md, then Read each PNG listed under each
Region. The Read tool returns image content for you to see (your own vision IS the rubric —
do not call any external API or SDK).

For each region, fill out the rubric (integer scores 1–5 per dimension; 1 = clearly broken,
5 = indistinguishable from a real painted-on ad). Write the output files into the same
ai_review/ directory:
  - <region>.json (strict JSON, with `min_score` injected)
  - <region>.md (short prose; under 150 words)

If a region has no PNGs listed, skip it. Pay extra attention to the walkover window if one
is present — that's where the player walks on the court-floor logo.
```

## Generalization to a new court

When porting to a new tournament clip:

1. Author a new `configs/experiments/<your_clip>.yaml` (manually clicked corners or geometry-driven, doesn't matter to the eval).
2. Add an entry to `configs/eval/reference.yaml`: `{<input_video_basename>: <gold_experiment_dir>}`. The gold for a new clip is the first run you certify by hand; subsequent runs compare against it.
3. The eval auto-discovers regions from the frozen config: `surface_type == "court_floor"` → floor region, `surface_type == "banner"` with `compositor_params.logo_placement_quad` → left-side region, `surface_type == "banner"` without → back banner. Object IDs and pixel coordinates do NOT need to match Melbourne.
4. Walkover detection auto-finds the window via clean-vs-original delta if `input.clean_video` exists in the config; falls back to a per-pixel temporal median of the original ROI if not.
5. Run `python -m banner_pipeline.eval --experiment <new_run> --reference auto`.

There is **no Melbourne-specific pixel coordinate** in the eval framework. All ROIs come from the frozen config.

## Determinism + repeatability

- Re-running the eval on the same experiment dir must produce a byte-identical `eval/quality_metrics.json`.
- Implementations must seed any random sampling (`np.random.seed(0)`).
- AI review (when on) is non-deterministic by nature; treat its scores as advisory.
- Schema versioning: `eval/quality_metrics.json` includes `"schema_version": <int>`. Bump on breaking changes; sub-agents should fail loudly on mismatch.

## When sub-agents run experiments

Sub-agents launched to iterate on the pipeline must:

1. Run the pipeline (Modal or local) and produce an experiment dir under `experiments/`.
2. Run `python -m banner_pipeline.eval --experiment <dir> --reference auto`. The `eval/ai_review/MANIFEST.md` is written automatically.
3. Read the resulting `eval/quality_metrics.json`. If `any_regression == true` or any per-region `pass == false`, the change is a regression — surface it in the agent's final report.
4. (Optional, recommended for checkpoint-grade runs) Dispatch a separate sub-agent (general-purpose) using the MANIFEST.md as its prompt to fill in the visual rubric. The sub-agent reads PNGs via Read tool (its own vision) and writes `eval/ai_review/<region>.{json,md}`. No SDK, no API key.
5. Cite specific failures from `eval/report.md` (file paths to the contact sheets) so a human can scrub the visual evidence.

A sub-agent SHOULD NOT claim "the change works" without producing a passing `quality_metrics.json` first.

## Schema (`eval/quality_metrics.json`)

```json
{
  "schema_version": 1,
  "experiment_dir": ".../<run>/",
  "reference_dir": ".../<gold>/" | null,
  "objects": {"1": "back", "2": "back", "5": "back", "4": "left_logo", "3": "floor_logo"},
  "geometric_source": "per_frame_state" | "static_fallback",
  "back_corner_max_jump_px": 1.4,
  "back_quad_area_cv": 0.012,
  "back_roi_jitter_ratio": 0.71,
  "back_roi_delta_E_lab": 1.8,
  "back_roi_temporal_ssim_mean": 0.998,
  "back_noise_variance_ratio": 0.42,
  "back_edge_sharpness_ratio": 1.2,
  "back_pass": true,
  "left_pass": true,
  "floor_walkover_logo_visible_pct": 0.91,
  "floor_walkover_occlusion_iou": 0.87,
  "floor_pass": true,
  "full_roi_temporal_ssim_mean": 0.992,
  "full_pass": true,
  "walkover_window_start": 690,
  "walkover_window_end": 745,
  "vs_reference": {
    "back_corner_distance_p95_px": 0.4,
    "back_roi_ssim_mean": 0.989,
    "floor_walkover_occlusion_iou_delta": -0.02,
    "any_regression": false
  },
  "ai_review_min_score": 4,
  "ai_review_walkover_occlusion_realism": 4
}
```

## Out of scope (deliberately)

- LPIPS / DreamSim / VGG-perceptual learned metrics. The sub-agent vision rubric covers perceptual realism qualitatively.
- **Anthropic SDK / API calls for the rubric.** Removed by design. The rubric is scored by sub-agent vision (Read on PNGs returns image content for the agent to see). No API key, no token cost, no SDK dependency.
- Full-clip GIF generation. PNG strips + side-by-side mp4 cover the visual need.
- Multi-run aggregation across many experiments (DB / dashboard). Separate later feature.
- Person-detector-based occlusion. We use the clean-vs-composite delta heuristic (no model dependency).
- Modal-side eval execution. Eval is local/CPU and runs after the pipeline finishes.
- Modifying `scripts/run_experiment.py` or `scripts/modal_run.py`. Use `scripts/run_and_eval.sh` to chain.

---

For internal design rationale and historical context, see [`docs/evaluation-protocol.md`](evaluation-protocol.md) (superseded — kept for reference).
