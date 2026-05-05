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

### C001 results — 2026-05-04 21:57 EDT

```
=== CYCLE C001 SLOT A1 REPORT (rerun) ===
Hypothesis: occlusion_dilate_px 2->0 on court_floor; expected to keep more logo visible around player feet.
Run dir: experiments/2026-05-04_21-55-06_hull_B200
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1786, delta=-0.06%
floor_walkover_occlusion_iou: 0.9838
Walkover window: 685:723
Failed metrics: none (warnings: back/left/floor roi_delta_E_lab)
Cost: Modal-B200 ~9min
Recommendation: dead end on this axis — dropping dilate from 2 to 0 produced essentially identical visibility (-0.06%, within noise).
=== END REPORT ===
```
Manager note: dead-end. visible_pct insensitive (-0.06%); no regression but no movement.

```
=== CYCLE C001 SLOT A2 REPORT (rerun) ===
Hypothesis: alpha_feather_px 25->10 on court_floor; sharper edges expected to raise visible_pct.
Run dir: experiments/2026-05-04_21-52-36_hull_B200
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: yes
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1785, delta=-0.11%
floor_edge_sharpness_ratio: 0.0465 (no warning, well below 1.8)
Walkover window: 685:723
Failed metrics: none failed; warnings: back/left/floor roi_delta_E_lab; regression flagged on floor_roi_delta_E_lab (7.672 vs gold)
Cost: Modal-B200 ~5.6min
Recommendation: dead end — feathering reduction from 25 to 10 produced essentially no change.
=== END REPORT ===
```
Manager note: dead-end. visible_pct unchanged (-0.11%); minor delta_E warning regression but warnings don't gate.

```
=== CYCLE C001 SLOT A3 REPORT ===
Hypothesis: quad_expand_px 80->120 on court_floor; wider quad expected to put more logo signal in eval measurement zone.
Run dir: experiments/2026-05-04_21-40-02_hull_B200
Exit code from eval: 2
Pass: back=P left=P floor=F full=P
Regression vs gold: yes
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1735, delta=-2.91%
floor_walkover_occlusion_iou: 0.4415 (FAIL gate; -55.85% vs gold)
floor_roi_ssim_vs_reference_mean: 0.5418 (large drop; quad geometry shifted)
Walkover window: 685:723
Failed metrics: floor_walkover_occlusion_iou
Cost: Modal-B200 ~14min
Recommendation: dead end. Wider quad regressed everything.
=== END REPORT ===
```
Manager note: dead-end with hard regression (floor_walkover_occlusion_iou 0.44 fails gate).

### C001 synthesis

All three C001 perturbations failed to move `floor_walkover_logo_visible_pct` materially (variance ≤0.5% across very different knob settings). Strongly suggests the metric is **asset-driven**, not tunable via floor-compositor params: the `redbull_white.png` logo asset only covers ~18% of the placement_quad with strong-signal pixels by design (it's mostly transparent except for the wordmark + bull). To move this metric, we need to perturb the asset or the underlying blend strategy, not edge/feather/erosion.

Best run from C001 by holistic criteria (no regression, all scorecards pass, occlusion_iou ~1.0): **A1 (occlusion_dilate=0)** — but it's not strictly better than the gold. The gold remains the best.

---

## C002 — 2026-05-04 22:00 EDT — pivot to asset + blend axes
Manager hypothesis: visible_pct is plateau'd against compositor edge knobs because it's asset-driven. Pivot to two changes that should genuinely move the rendered floor logo:

- **A1 — swap floor logo asset from `redbull_white.png` to `redbull_court_patch.png`**
  - Rationale: `redbull_court_patch.png` is a more-filled design (vs the wordmark-only `redbull_white.png`), so it should cover more of the placement_quad with strong signal, materially raising visible_pct. Direct test of the asset-driven hypothesis.
  - Risk: looks artificial / over-painted on the court. Watch occlusion_iou and edge_sharpness.
  - Implementation: change the `input.logo` field at the top of the YAML... BUT note: `input.logo` is a global asset; if changed, it affects ALL objects, not just floor. Workaround: many configs in this repo use `input.logo` as the global asset. To affect only obj_3 (court_floor), the agent must verify whether per-object asset overrides are supported in the config schema. If not, this hypothesis must be skipped or executed differently.
  - **Agent: first verify whether per-object logo override exists. If not, switch to: keep `redbull_white.png` global, but explicitly add `asset:` field on obj_3 prompt to point at `redbull_court_patch.png`. If neither works, skip the run and report the constraint.**

- **A2 — `clean_underlay_alpha` 0.0 → 0.3 on court_floor surface_overrides**
  - Rationale: the v68 config bakes the logo directly on top of the court text. Letting some clean plate (text-erased background) show through behind the logo could make it look more "painted on" the underlying clean court (vs floating on the original text). Should NOT change visible_pct meaningfully (since clean plate isn't logo signal) but might improve walkover_occlusion_iou and the AI rubric realism dimension.
  - Risk: too much clean underlay = ghost double-image of the court text. Watch occlusion_iou.

Targeting: A1 — strictly increase `floor_walkover_logo_visible_pct`; A2 — preserve visible_pct AND improve `floor_walkover_occlusion_iou` over A1's 0.984.

Status: dispatched in background.

---

<!-- Subsequent cycles append below this line. -->
