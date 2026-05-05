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

### C002 results — 2026-05-04 22:14 EDT

```
=== CYCLE C002 SLOT A1 REPORT ===
Hypothesis: floor logo asset redbull_white.png -> redbull_court_patch.png
Asset routing: global (affects all objects) — ObjectPrompt has no asset field; only input.logo is used by compositor pipeline.
Run dir: experiments/2026-05-04_22-06-09_hull_B200
Exit code from eval: 3
Pass: back=P left=P floor=P full=P
Regression vs gold: yes (any_regression=true)
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1882, delta=+5.32%
floor_walkover_occlusion_iou: 0.9261 (regression vs reference)
back_roi_ssim_vs_reference_mean: 0.8214 (global asset change DOES affect back banners)
Walkover window: 685:723
Failed metrics: none (all per-region scorecards pass)
Warnings fired: back/left/floor roi_delta_E_lab; vs_reference regressions on floor_roi_delta_E_lab, back_roi_delta_E_lab, floor_walkover_occlusion_iou
Cost: Modal-B200 ~5.8min
Recommendation: Pivot — patch asset improves floor visibility (+5.32%) but global swap regresses back-banner SSIM (0.82) and triggers any_regression. Need per-object asset support (code change) to isolate.
=== END REPORT ===
```
Manager note: real signal but cross-region contamination. Not a candidate (any_regression=true). Confirms visible_pct IS asset-driven; bottleneck is pipeline not honoring per-object assets.

```
=== CYCLE C002 SLOT A2 REPORT ===
Hypothesis: clean_underlay_alpha 0.0 -> 0.3 on court_floor
Run dir: experiments/2026-05-04_22-11-23_hull_B200
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1784, delta=-0.17%
floor_walkover_occlusion_iou: 0.9774
floor_roi_delta_E_lab: 6.934
Walkover window: 685:723
Failed metrics: none
Warnings fired: back/left/floor roi_delta_E_lab
Cost: Modal-B200 ~11min
Recommendation: continue same direction — visible_pct held but occlusion_iou (0.9774) below A1 baseline 0.984.
=== END REPORT ===
```
Manager note: no-change for visible_pct; small occlusion_iou cost (0.984 → 0.977). Not a candidate.

### C002 synthesis

A1 confirmed visible_pct is asset-driven (+5.32% gain from a single asset swap). But the current pipeline routes ALL objects through `input.logo` (global). Per-object asset routing is needed to apply the patch asset to obj_3 alone without breaking back banners. Pivot to a CODE change.

---

## C003 — 2026-05-04 22:14 EDT — code change: per-object asset support
Manager hypothesis: add a `asset: str | None` field to `ObjectPrompt` and have the three video pipeline paths honor it. Specifically: when `prompt.asset` is set, load that specific image as the overlay for that object's compositor calls instead of the global `overlay`. Backwards-compatible: empty / missing `asset:` falls back to the global behavior.

- **A1 — code-change agent (single agent this cycle, no Modal run)**
  - Changes:
    - `src/banner_pipeline/segment/base.py`: add `asset: str | None = None` field to `ObjectPrompt` dataclass.
    - `src/banner_pipeline/pipeline.py:_prompts_from_config`: parse `asset:` from each prompt entry.
    - In each of the three `run_pipeline_video*` functions: where `overlay` is used in the per-frame compositor loop, prefer `_per_obj_overlay.get(obj_id, overlay)` where `_per_obj_overlay` is a dict pre-loaded from each prompt's `asset` field at function start.
  - Verification (must all pass before commit):
    1. `uv run pytest tests/ -x -q` — all existing tests pass.
    2. Backwards-compat test: run eval framework on `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/` (which has NO `asset:` fields anywhere) — eval should still pass all four scorecards. Code change must not affect runs with no per-object asset.
  - Commit: `code: ObjectPrompt.asset for per-object overlay routing` + push.

After C003 lands, C004 will re-test the floor asset swap in isolation: copy v68 base, add `asset: data/logos/redbull_court_patch.png` only to obj_3, expect visible_pct +5.3% with back banners unchanged.

Status: dispatched in background.

### C003 results — 2026-05-04 22:19 EDT

```
=== CYCLE C003 SLOT A1 REPORT ===
Task: per-object asset routing on ObjectPrompt + 3 video paths.
Files modified: src/banner_pipeline/segment/base.py, src/banner_pipeline/pipeline.py
Lines added/changed: +49 -4
Backwards-compat verification:
  - syntax check: PASS
  - pytest tests/: 97 passed, 0 failed
  - eval on v68 gold: all 4 scorecards PASS (back, left, floor, full)
  - import smoke test: PASS (asset attr: None)
Per-object asset call sites updated:
  - run_pipeline_video: yes (line 2243)
  - run_pipeline_video_tracking: yes (line 2545)
  - run_pipeline_video_hybrid: yes (line 3556 painted_court_composite + 3620 comp.composite)
Loader used: _load_overlay (cv2.imread IMREAD_UNCHANGED) in video/hybrid; raw cv2 in tracking.
Commit SHA: 47b2665
Recommendation: ready for C004 (re-test floor asset swap with per-object isolation).
=== END REPORT ===
```
Manager note: SUCCESS. Code change clean, all verifications pass. Unlocks isolated per-object asset experiments.

---

## C004 — 2026-05-04 22:19 EDT — isolated floor asset swap (regression-safe)
Manager hypothesis: now that obj_3 can carry its own `asset:` field, swap ONLY obj_3's overlay to `redbull_court_patch.png` while back-wall objects (1, 2, 5) and left obj_4 keep using the global `redbull_white.png`. Expect visible_pct gain similar to C002/A1's +5.32% but WITHOUT back-banner cross-contamination — should yield `any_regression: false` and become the first true candidate.

- **A1 — single agent, single config knob**
  - Copy `eval_walkover_v68_clicked_homography_static_full.yaml` to `eval_walkover_c004_a1_floor_asset_patch_isolated.yaml`.
  - Add `asset: data/logos/redbull_court_patch.png` to the obj_3 prompt entry only (not the global `input.logo`, not the other prompts).
  - Run pipeline + eval + commit.
  - Target metric: `floor_walkover_logo_visible_pct` strictly > 0.1787, `any_regression: false`, all per-region scorecards pass.

Status: dispatched in background.

### C004 results — 2026-05-04 22:32 EDT

```
=== CYCLE C004 SLOT A1 REPORT ===
Hypothesis: ISOLATED floor asset swap obj_3 only to redbull_court_patch.png (uses new ObjectPrompt.asset code from 47b2665).
Run dir: experiments/2026-05-04_22-26-31_hull_B200
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1786, delta=-0.01%
floor_walkover_occlusion_iou: 0.9838
back_roi_ssim_vs_reference_mean: 0.9983 — ISOLATION CONFIRMED (vs 0.82 in C002/A1 global)
left_roi_ssim_vs_reference_mean: 0.9976
Walkover window: 685:723
Failed metrics: none
vs_reference any_regression: false
Cost: Modal-B200 ~5.7min
Recommendation: pivot — visible_pct didn't move; gain in C002/A1 was cross-region artifact.
=== END REPORT ===
```
Manager note: SUCCESS for code (per-object asset routing works), DEAD-END for hypothesis (visible_pct insensitive to obj_3 asset).

### C001-C004 axis exhaustion — pivot rationale

After 4 cycles + ~7 floor-targeted runs, `floor_walkover_logo_visible_pct` is essentially fixed at 0.178 ± 0.01 across very different perturbations (occlusion_dilate, alpha_feather, quad_expand, clean_underlay, asset content). The earlier +5.32% spike (C002/A1) was a CROSS-REGION measurement artifact, not a real signal. **Conclusion: visible_pct is not a useful optimization target for this clip with this placement_quad.** The metric is dominated by the placement_quad geometry and the eval's delta-threshold heuristic, both of which are essentially fixed.

Pivot strategy:
- Establish a NEW signal: **AI rubric scores on the gold**. Until we have that, we can't tell whether incremental config changes are improving or degrading visual quality.
- Broaden axes beyond floor-only: test global `mask_dilate_px` (affects back banners + left + floor inpaint).
- Move toward generalization tests: same config on different clip basenames (later cycles).

The framework's existing gates (corner_max_jump, jitter_ratio, SSIM, etc.) all remain green for the gold. Future winners need to either (a) hold all those gates while AI-rubric scores improve, or (b) materially improve a still-passing metric without regression.

---

## C005 — 2026-05-04 22:32 EDT — pivot to AI rubric baseline + broader knob
Manager hypotheses:

- **A1 — AI rubric on the gold** (no Modal). Establish per-region baseline scores (`realism.painted_on_vs_pasted_on`, `geometry.perspective_plausibility`, `temporal.occlusion_realism`, etc.). One-time cost ~$0.20-0.30. Output lands in `experiments/.../eval/ai_review/*.json`. Gives us a meaningful target for future cycles.
  - Agent task: `uv sync --extra ai` if needed; then `uv run python -m banner_pipeline.eval --experiment experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/ --reference auto --with-ai-review`. Read the produced ai_review/*.json files. Report rubric scores per region. Commit the new ai_review/ artifacts.

- **A2 — `mask_dilate_px` 20 → 10 globally** (config-only Modal run). Affects how much the inpaint compositor dilates SAM masks before painting. Lower value = tighter logo edges everywhere. Watch for back-banner regressions (this is a global knob).
  - New config: `eval_walkover_c005_a2_mask_dilate_10.yaml` (single field change in `pipeline.compositor.params`).
  - Target: `floor_walkover_logo_visible_pct` ≥ gold AND `back_roi_ssim_vs_reference_mean > 0.99` AND `any_regression: false`.

Status: dispatched in background.

---

<!-- Subsequent cycles append below this line. -->
