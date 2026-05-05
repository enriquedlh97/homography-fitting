# Experiment Ledger

Append-only log. Manager (Claude main thread) reads only the tail (last ~10 entries) per tick. Workers read this file + `docs/AGENT_BRIEFING.md` + `docs/EVALUATION.md`.

## Header

- **Started:** 2026-05-04 21:30 EDT
- **Deadline:** 2026-05-05 10:00 EDT (12.5h budget)
- **Branch:** `feat/quality-fixes-next` (off `feat/quality-fixes`)
- **Gold reference:** `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/` (manually clicked v68 static homography)
- **Working clip:** `data/melbourne-walking-over-logo.mov` (767 frames, 59 fps)
- **Starting axis:** **Floor logo gap** — improve `floor_walkover_logo_visible_pct` (gold = 0.18) and the visual realism of the player walking on the Red Bull court-floor logo. This is the user-facing wow moment.
- **AI review cadence:** off by default; manager flips on every ~10 cycles or for close-to-promotion candidates.
- **Modal authorization:** confirmed; agents may invoke `uv run modal run scripts/modal_run.py ...` without re-asking.

## Baseline (gold)

From `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/eval/quality_metrics.json`:

| Metric | Value | Status |
|---|---|---|
| back_pass | true | scorecard PASS |
| left_pass | true | scorecard PASS |
| floor_pass | true | scorecard PASS |
| full_pass | true | scorecard PASS |
| floor_walkover_logo_visible_pct | 0.1787 | the floor target metric — primary axis |
| floor_walkover_occlusion_iou | 1.0 | self-comparison; not informative until a different reference exists |
| floor_corner_max_jump_px | 0.0 | static fallback (no per_frame_state.json yet on gold) |
| walkover_window | 685:723 | auto-detected |
| floor_roi_temporal_ssim_mean | 0.997 | very stable |
| floor_roi_jitter_ratio | 0.494 | well below 1.05 gate |

A future improvement is gauged by: `floor_walkover_logo_visible_pct` strictly increases AND `any_regression == false` AND all per-region scorecards still pass.

## Plateau detector

Manager tracks `floor_walkover_logo_visible_pct` over a rolling window of the last 8 cycles. If best-in-window hasn't improved by ≥1% absolute over that window → declare plateau on this axis, pivot to a different dimension. Candidate next axes when this one plateaus: (a) reduce visible-edge / "pasted-on" appearance, measured by `floor_edge_sharpness_ratio` and AI-review `realism.painted_on_vs_pasted_on`, (b) improve back-banner stability if `feat/quality-fixes-next` accumulates regressions there, (c) re-derive `court_rect` for dynamic configs.

## Cycle plan template

Each cycle dispatches 1–3 agents (slots A1/A2/A3). Manager seeds each with a single config knob to perturb. Agents run pipeline+eval+commit and return a 250-word structured report.

---

## C001 — 2026-05-04 21:35 EDT — opening floor-knob sweep
Manager hypothesis: the floor walkover logo's low visibility (0.18) may stem from over-aggressive masker erosion or from logo edges being fuzzed too softly during compositing. Sweep three independent knobs in parallel.

- **A1 — `surface_overrides.court_floor.occlusion_dilate_px` 2 → 0**
  - Rationale: dilating the player mask before painting the logo erodes the logo around the player's feet. Setting it to 0 keeps more of the placement_quad showing logo signal.
  - Risk: player edges may bleed onto the logo (look ragged).
- **A2 — `surface_overrides.court_floor.alpha_feather_px` 25 → 10**
  - Rationale: the current 25-px feather softens the logo into the court so much that the logo signal is below the eval's delta threshold across most of the quad. Tightening should raise visible_pct.
  - Risk: edges look harder / less photoreal.
- **A3 — `surface_overrides.court_floor.quad_expand_px` 80 → 120**
  - Rationale: a slightly wider placement_quad should put more logo signal across the eval's measurement zone.
  - Risk: logo extends beyond visible court markings, looks oversized.

Targeting: `floor_walkover_logo_visible_pct` strictly > 0.18 with `any_regression == false`.

Status: dispatched in background.

---

<!-- Subsequent cycles append below this line. -->
