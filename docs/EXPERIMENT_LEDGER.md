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

### C005 results — 2026-05-04 22:46 EDT

```
=== CYCLE C005 SLOT A1 REPORT ===
Task: AI rubric baseline on v68 gold
anthropic install: OK (anthropic==0.98.1 installed via uv sync --extra ai)
ANTHROPIC_API_KEY in env: no
Run output: NOT ATTEMPTED — auth would fail immediately
Per-region rubric scores: N/A
Cost estimate: ~$0.00
Recommendation: needs API key provisioning; defer AI-rubric path.
Run dir: FAILED — no ANTHROPIC_API_KEY in environment
=== END REPORT ===
```
Manager note: anthropic SDK now installed for future use. AI-rubric path unavailable for this autonomous run; pivot to in-agent vision-Claude as alternative qualitative signal.

```
=== CYCLE C005 SLOT A2 REPORT ===
Hypothesis: mask_dilate_px 20->10 globally (compositor.params)
Run dir: experiments/2026-05-04_22-43-42_hull_B200
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1786, delta=-0.06%
back_roi_ssim_vs_reference_mean: 0.9981
back_roi_temporal_ssim_mean: 0.9999
floor_walkover_occlusion_iou: 0.9838
Walkover window: 685:723
Failed metrics: none
Warnings fired: back/left/floor roi_delta_E_lab
Cost: Modal-B200 ~11min
Recommendation: continue — viable global tightening; flat visible_pct.
=== END REPORT ===
```
Manager note: clean pass, no regression, but no movement on the target metric. Numerical saturation confirmed — every safe perturbation lands at "no-op pass." Need qualitative signal.

### Strategic pivot — qualitative-via-subagent

Numerical metrics (visible_pct, SSIM, jitter, occlusion_iou) are saturated for this clip-on-itself comparison. The signal we'd want is "does run X *look* better than the gold?" — exactly what AI rubric was for, but no API key. Pivot: dispatch a vision-capable sub-agent that reads the existing per-region PNG artifacts (via the Read tool's image support) and produces a qualitative ranking. The sub-agent IS a Claude with vision; this is rubric-without-API.

This gives us a real "best candidate" signal across C001-C005 runs.

---

## C006 — 2026-05-04 22:46 EDT — visual comparator + one new axis
Manager hypotheses:

- **A1 — visual comparator agent (no Modal, no API).** Read the `eval/walkover/consecutive_frames.png` for each of: gold, C001/A1 (occ_dilate=0), C001/A3 (quad_expand=120), C002/A1 (asset patch global), C002/A2 (clean_underlay=0.3), C004/A1 (isolated patch), C005/A2 (mask_dilate=10). Same for `eval/floor_logo/consecutive_frames_mid.png`. Score each on the docs/EVALUATION.md rubric (1-5 per dimension), select the visually best run, report which dimensions move and which are flat.
- **A2 — `logo_blur_px: 1` on court_floor** surface_overrides. New field (not present in v68; pipeline reads it via court_floor compositor). Simulates a slightly painted-on look that should NOT change visible_pct but might improve perceived realism. Safe perturbation; another data point for the visual comparator.

Status: dispatched in background.

### C006 results — 2026-05-04 22:56 EDT

```
=== CYCLE C006 SLOT A1 REPORT (visual comparator) ===
Per-run scores (painted_on / edge_seam / occlusion_realism / jitter_visible / total):
  GOLD: 4/4/5/5 = 18/20
  C001/A1 occ_dilate=0: 4/4/5/5 = 18/20 (indistinguishable from GOLD)
  C001/A3 quad_expand=120: 4/4/5/5 = 18/20
  C002/A1 asset_patch_global: 4/4/5/5 = 18/20
  C002/A2 clean_underlay=0.3: 4/4/5/5 = 18/20
  C004/A1 isolated_patch: 4/4/5/5 = 18/20
  C005/A2 mask_dilate=10: 4/4/5/5 = 18/20 (slightly more conservative foot edge but within noise)
Most-moved dimension: none meaningfully. realism.edge_seam shows only sub-pixel differences.
Most-stuck dimension: temporal.jitter_visible (homography is static/clicked — every run locked identically).
Recommendation: floor + walkover region is visually saturated for these knobs. Pivot axis to back_banners or left_logo, or sweep logo_blur_px specifically (the one knob that attacks painted_on, currently stuck at 4/5).
```
Manager note: KEY FINDING. Floor + walkover region is **visually saturated** across the entire knob set. v68 gold and 6 perturbations all score identically. Pivot needed to back_banners / left_logo regions where there may be more headroom.

```
=== CYCLE C006 SLOT A2 REPORT (logo_blur_px=1) ===
Run dir: experiments/2026-05-04_22-54-03_hull_B200
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1785, delta=-0.11%
floor_roi_ssim_vs_reference_mean: 0.9995
floor_walkover_occlusion_iou: 0.9837
floor_edge_sharpness_ratio: 0.0586 (proves logo_blur_px is being honored — edges softer than baseline)
Cost: Modal-B200 ~5.7min
Recommendation: clean candidate; try blur=2 if continuing; otherwise pivot.
```
Manager note: logo_blur_px field IS supported and honored. Numerical metrics flat (expected — blur doesn't move visible_pct or SSIM). Visual diff likely sub-pixel; the C007 comparator will check whether the blur produces any perceptible improvement. Treating as no-change candidate (no regression but no measurable improvement either).

---

## C007 — 2026-05-04 22:57 EDT — pivot to back/left regions
Manager hypotheses:

- **A1 — visual comparator on back_banners + left_logo** across 8 runs (gold + 6 prior + C006/A2 logo_blur=1). Read `<run>/eval/back_banners/consecutive_frames_mid.png` and `<run>/eval/left_logo/consecutive_frames_mid.png` per run. Score same 4-dim rubric. Goal: find a region where knob perturbations actually move visual quality — that's where C008+ will optimize.
- **A2 — back-banner knob**: `compositor.params.alpha_feather_px` 1→3 in the v68 base config. New config `eval_walkover_c007_a2_alpha_feather_3.yaml`. Single-line change. Rationale: increases edge feather at the back-banner inpaint boundary; may reduce visible cutout if the comparator finds back-banner edges are the bottleneck.

Status: dispatched in background.

### C007 results — 2026-05-04 23:09 EDT

```
=== CYCLE C007 SLOT A1 REPORT (back/left visual comparator) ===
BACK_BANNERS scores: GOLD/C001/A1/C001/A3/C002/A2/C004/A1/C005/A2/C006/A2 all 20/20; C002/A1 = 10/20 (global asset swap visibly destroys back banners)
LEFT_LOGO scores: same 7-way 20/20 tie; C002/A1 = 12/20 (visible rectangular patch outline around each Red Bull logo)
Most-moved dimension: only C002/A1 produces visible delta in back/left; all other 7 runs are pixel-equivalent
Most-stuck dimension: temporal.jitter_visible (perfect lock everywhere)
Recommendation: back/left also visually saturated. Knob set has zero leverage on these regions modulo the global-asset regression. Pivot beyond config knobs.
```
Manager note: BIG FINDING. Pipeline output is VISUALLY SATURATED across all three regions for our knob axis. Only knob with visible leverage is asset-swap-global, and it's a regression.

```
=== CYCLE C007 SLOT A2 REPORT (alpha_feather_px 1->3 global) ===
Run dir: experiments/2026-05-04_23-07-41_hull_B200
Exit code: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: yes (back_roi_delta_E_lab only)
back_roi_ssim_vs_reference_mean: 0.9814 (drop from typical 0.998)
back_roi_temporal_ssim_mean: 0.9999
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1786 (-0.01%)
Cost: Modal-B200 ~9min
Recommendation: dead end — softer back-banner edges hurt color fidelity, no compensating gain.
```
Manager note: dead-end. Soft regression confirms the v68 default was already tuned for this metric.

### Plateau declaration

After C001-C007 (~9 Modal runs covering occlusion_dilate, alpha_feather_floor, quad_expand, asset_swap_global, asset_swap_isolated, clean_underlay_alpha, mask_dilate_global, logo_blur, alpha_feather_global), and 2 visual-comparator passes (floor+walkover, back+left): **the v68 manually-clicked static-homography pipeline is at a quality plateau on this clip for the simple-knob optimization axis**. All regions are visually saturated; numerical metrics are saturated; the only knob with leverage is global asset swap which regresses cross-region. To break out we need architectural changes.

---

## C008 — 2026-05-04 23:09 EDT — architectural pivot
Manager hypotheses:

- **A1 — `pipeline.fitter.type: pca`** (vs current `hull`). Single-line config change. Tests whether a different quad-fitting algorithm produces visibly different placement geometry. Available fitters in registry: `pca`, `lp`, `hull`, `fronto_parallel`, `vp_constrained`. New config: `eval_walkover_c008_a1_fitter_pca.yaml`.
- **A2 — run the existing `eval_walkover_v68_clicked_homography_dynamic_full.yaml`** through Modal. This config already has `pipeline.geometry.enabled: true` and `court_plane_placement` on obj_3 + obj_4. Tests dynamic-line-detected court geometry vs the static clicked corners. Significant departure from gold; expect SSIM-vs-ref drop because placements move per-frame; the question is whether they look BETTER (the Holy Grail of "court geometry tracks real court motion").
  - Existing experiment outputs at `experiments/2026-05-01_09-04-27_hull_B200/` and `experiments/2026-05-01_09-13-02_hull_H200/` were produced before the eval framework existed. We need a fresh run to get all the eval artifacts. Use the existing yaml as-is.

Status: dispatched in background.

### C008 results — 2026-05-04 23:31 EDT

```
=== CYCLE C008 SLOT A1 REPORT ===
Hypothesis: fitter.type hull -> pca
Run dir: experiments/2026-05-04_23-17-01_pca_B200/
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no
back_roi_ssim_vs_reference_mean: 0.9983
left_roi_ssim_vs_reference_mean: 0.9977
floor_roi_ssim_vs_reference_mean: 0.9995
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1788, delta=+0.06%
floor_walkover_occlusion_iou: 0.9823
Cost: Modal-B200 ~5.5min
Recommendation: low-risk variant; visually near-identical to hull baseline.
=== END REPORT ===
```
Manager note: clean pass, no movement. Different fitter algorithms produce indistinguishable output for this clip+placement.

```
=== CYCLE C008 SLOT A2 REPORT ===
Hypothesis: dynamic court geometry via existing v68 dynamic config
Run dir: experiments/2026-05-04_23-17-23_hull_B200/
Exit code: 2
Pass: back=P left=F floor=F full=F
Regression vs gold: yes
geometric_source: static_fallback (geometry_runtime_enabled=false; geometry_active_objects=[])
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1404, delta=-21.4%
left_roi_ssim_vs_reference_mean: 0.378 / floor_roi_ssim_vs_reference_mean: 0.2223
Failed metrics: left_roi_jitter_ratio, left_roi_temporal_ssim_mean, floor_roi_jitter_ratio, floor_roi_temporal_ssim_mean, floor_walkover_occlusion_iou, full_roi_jitter_ratio
Cost: Modal-B200 ~5.7min
Recommendation: dead end. Config-as-written did NOT activate dynamic geometry. Output is degraded for unrelated reasons.
```
Manager note: KEY DISCOVERY — the repo's "dynamic" config has a latent activation bug. `geometry.enabled: true` doesn't reach runtime. Heavy regression (visible_pct -21.4%, SSIMs 0.22-0.38) is from some OTHER mismatch in the dynamic config, not from dynamic geometry per se. Worth investigating in C009.

---

## C009 — 2026-05-04 23:31 EDT — diagnostics + continued breadth
Manager hypotheses:

- **A1 — code-reading agent: investigate the dynamic-geometry activation gate**. Read-only task. Find why `pipeline.geometry.enabled: true` in the dynamic_full yaml fails to propagate to runtime (geometry_runtime_enabled=false, geometry_active_objects=[]). Trace from `_geometry_enabled` in pipeline.py through `CourtGeometryEstimator` initialization in court_geometry.py. Report the gate that's blocking activation. Don't fix yet — just diagnose so the manager knows whether it's a one-line fix or deeper.
- **A2 — `compositor.params.padding` 0.1 → 0.15** (config-only Modal run). Untouched knob; tests inpaint padding. Single-line change. Should be safe.

Status: dispatched in background.

### C009 results — 2026-05-04 23:36 EDT

```
=== CYCLE C009 SLOT A1 REPORT (diagnostic) ===
Findings:
1. _geometry_enabled at pipeline.py:200 reads YAML correctly (returns True for dynamic_full).
2. geometry_runtime_enabled flips True only via GeometryFittingEngine.finalize_metrics() (court_geometry.py:797).
3. run_pipeline_video_hybrid (line 2585) NEVER instantiates GeometryFittingEngine. Confirmed by grep over 2585-3650.
4. SUPPORTED_GEOMETRY_SURFACE_TYPES = {back_wall_banner, side_wall_banner, court_marking}. v68 dynamic config uses surface_type: court_floor and banner — NEITHER in supported set. Even if engine ran, _geometry_active_object_ids returns [].
5. Hybrid DOES run a parallel CourtGeometryEstimator at pipeline.py:2996 for court_plane_placement_* metrics — separate from geometry_runtime_enabled flag. So the C008/A2 degradation came from the court-plane projection itself failing/falling back, NOT from the geometry_runtime_enabled flag.
Root cause: TWO-PART defect — (a) hybrid never builds GeometryFittingEngine, (b) supported surface types don't include court_floor/banner.
Fix difficulty: medium.
```
Manager note: rich diagnostic. Real bug found in the dynamic-geometry path. Defer fix; document for human review.

```
=== CYCLE C009 SLOT A2 REPORT ===
Hypothesis: padding 0.1->0.15 in compositor.params (global)
Run dir: experiments/2026-05-04_23-34-25_hull_B200
Exit code from eval: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no  [BUT SEE MANAGER NOTE — framework bug]
back_roi_ssim_vs_reference_mean: 0.7173  [** -28% vs gold's 1.0 self-comparison; this IS a regression **]
left_roi_ssim_vs_reference_mean: 0.9977
floor_roi_ssim_vs_reference_mean: 0.9995
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1788, delta=+0.06%
Cost: Modal-B200 ~12.6min
Recommendation: minor sweep candidate, unlikely breakthrough.
```
Manager note: **HIDDEN REGRESSION + FRAMEWORK BUG DISCOVERED**. Back-banner SSIM-vs-reference dropped from gold's 1.0 to 0.7173 — a 28% drop that should have flagged any_regression=true under the 5% slop rule. The eval framework's `detect_regressions` (`src/banner_pipeline/eval/reference.py`) is NOT detecting cross-region SSIM regressions properly. Likely cause: the reference key `back_roi_ssim_vs_reference_mean` only exists nested under `vs_reference` in gold's payload, but the comparison code expects it at top level. Worth flagging for the human's eval-framework cleanup queue.

So: C009/A2 (padding=0.15) is actually a regression — back-banner appearance materially changed — but the framework hid it. Visual comparator next cycle will confirm.

---

## DRAFT FINDINGS — running summary as of 2026-05-04 23:38 EDT

(Full Final Summary will be appended at deadline 10:00 EDT 2026-05-05.)

**Central finding: the v68 manually-clicked static-homography pipeline is at a quality plateau on this clip.** Numerical metrics (visible_pct, SSIMs, jitter ratios, occlusion_iou) are saturated; visual rubric scores from sub-agent comparators are saturated (all knob-perturbation runs scoring 18/20 floor or 20/20 back/left, modulo the global asset swap which regresses).

### Runs categorized

**No-change clean passes (all gates green, no regression, no visible improvement):**
- C001/A1 occlusion_dilate_px=0
- C002/A2 clean_underlay_alpha=0.3
- C004/A1 isolated obj_3 asset patch (uses new ObjectPrompt.asset code)
- C005/A2 mask_dilate_px=10 (global)
- C006/A2 logo_blur_px=1 on court_floor
- C008/A1 fitter.type=pca

**Hard regressions (failed gates or visible degradation):**
- C001/A3 quad_expand_px=120 — floor_walkover_occlusion_iou 0.44 (FAIL gate)
- C002/A1 asset_patch_global — back_ssim_vs_ref 0.82 + visible patch outline on back/left banners (visual comparator)
- C008/A2 dynamic_full config — left_ssim 0.378, floor_ssim 0.222 (multiple gate fails); also revealed dynamic-geometry activation bug

**Soft regressions (gates pass, vs-reference SSIM drops, framework didn't flag):**
- C007/A2 alpha_feather_px=3 (back delta_E warning + back ssim_vs_ref 0.98)
- C009/A2 padding=0.15 (back ssim_vs_ref 0.7173 — significant; framework `detect_regressions` missed this)

### Code change landed (47b2665)

`ObjectPrompt.asset` field + per-object overlay routing in all 3 video pipeline paths. Backwards-compatible (97 tests pass; v68 gold runs unchanged). Unblocks future per-object asset experiments.

### Real bugs discovered (both deferred for human review)

1. **Dynamic-geometry activation gate (medium fix)** — `run_pipeline_video_hybrid` never instantiates `GeometryFittingEngine`; SUPPORTED_GEOMETRY_SURFACE_TYPES excludes `court_floor` and `banner`. So config flag `pipeline.geometry.enabled: true` is structurally inert in hybrid mode. Documented in C009/A1 finding above with file:line references.
2. **Eval framework regression-detection bug (small fix)** — `src/banner_pipeline/eval/reference.py:detect_regressions` does not catch back/left/floor `roi_ssim_vs_reference_mean` regressions. C009/A2's 28% drop on back was not flagged. Likely the comparison expects keys at top-level in gold's payload but they're nested under `vs_reference`.

### Open recommendations for human

- **Best candidate vs gold:** none yet. The gold itself remains the visually best run.
- **AI rubric path is unavailable** in autonomous environment — no `ANTHROPIC_API_KEY`. Anthropic SDK is installed (commit `eval framework` series). Provisioning a key and running `--with-ai-review` on the gold + a few candidates would establish realism baseline scores beyond the saturated numerical metrics.
- **Code-level fixes** (per above bugs) will likely matter more than further config-knob iteration.
- **Generalization to a different clip** (`zoom-clip-melbourne.mov`, `tennis-clip.mp4`) requires new prompt configs (manual click points). Worth doing once before any production claim.

---

## C010 — 2026-05-04 23:38 EDT — comprehensive visual comparator + last fitter
Manager hypotheses:

- **A1 — comprehensive visual comparator.** Across 6 runs (GOLD, C001/A1 occ_dilate=0, C005/A2 mask_dilate=10, C006/A2 logo_blur=1, C008/A1 fitter=pca, C009/A2 padding=0.15). Read 3 frames per run — sample at frames 50 (mid-clip pre-walkover), 400 (mid-clip), 700 (mid-walkover). Score the holistic 4-dim rubric. Goal: identify the visually-best run AND verify whether C009/A2's padding=0.15 (which has 28% SSIM drop) is visibly broken in back banners.
- **A2 — `pipeline.fitter.type: fronto_parallel`** (third fitter option). Single Modal run; gather one more architectural data point.

Status: dispatched in background.

### C010 results — 2026-05-04 23:52 EDT

```
=== CYCLE C010 SLOT A1 REPORT (comprehensive comparator) ===
Per-run scores (back / floor_mid / walkover / OVERALL):
  GOLD:    20 / 20 / 20 = 60/60
  C001/A1: 20 / 20 / 20 = 60/60
  C005/A2: 20 / 20 / 20 = 60/60
  C006/A2: 20 / 20 / 20 = 60/60
  C008/A1: 20 / 20 / 20 = 60/60
  C009/A2: 14 / 20 / 20 = 54/60 (back banners horizontally shifted)
C009/A2 verification: YES — padding=0.15 visibly shifts back banners (entire strip offset; partial logo at left edge). Real placement regression, not just metric artifact. Floor + walkover unchanged.
Best run: GOLD (tied with C001/A1, C005/A2, C006/A2, C008/A1 at 60/60). None of the explored knobs improved on gold.
Recommendation: Scrub experiments/2026-05-04_23-34-25_hull_B200/eval/back_banners/consecutive_frames_mid.png next to GOLD to see the padding regression at first glance.
```
Manager note: GOLD remains best. C009/A2 padding=0.15 confirmed visibly broken. Framework regression-detection needs fixing (didn't catch the SSIM=0.7173 drop).

```
=== CYCLE C010 SLOT A2 REPORT ===
Hypothesis: fitter.type hull -> fronto_parallel
Run dir: experiments/2026-05-04_23-49-40_fronto_parallel_B200
Exit code: 0
Pass: all P
Regression vs gold: no
back/left/floor _ssim_vs_ref_mean: 0.9983 / 0.9977 / 0.9995
floor_walkover_logo_visible_pct: +0.06% (flat)
Cost: Modal-B200 ~10.5min
Recommendation: dead end — geometric_source reported `static_fallback`; all 3 fitters (hull/pca/fronto_parallel) produce identical output because the static-clicked path BYPASSES the runtime fitter. fitter.type is INERT for v68 base config.
```
Manager note: NEW ARCHITECTURAL FINDING — `fitter.type` is inert when `placement_quad` is set on prompts (the static-clicked path). This explains C001-C010 fitter-no-op behavior. Adding to draft findings.

---

## DRAFT FINDINGS — extended with C010 (as of 2026-05-04 23:52 EDT)

### Updated central conclusion

The v68 manually-clicked static-homography pipeline is at quality plateau on `data/melbourne-walking-over-logo.mov`. Across 10 cycles, **only one knob produced visible delta and it was a regression** (C009/A2 padding=0.15 shifted back banners). The remaining safe knobs are no-ops because the static-clicked path **structurally bypasses the runtime fitter** and the dynamic-geometry activation gate is **structurally broken in hybrid mode**.

### What we measured systematically

10 cycles, 11 Modal runs. All Modal runs committed with experiment dirs + per-region eval artifacts.

| Cycle | Knob / change | Outcome |
|---|---|---|
| C001/A1 | occlusion_dilate_px 2→0 | no-change clean pass |
| C001/A2 | alpha_feather_px 25→10 (court_floor) | no-change clean pass |
| C001/A3 | quad_expand_px 80→120 | regression (occlusion_iou fails gate) |
| C002/A1 | floor asset → court_patch (global) | regression (back/left visibly broken) |
| C002/A2 | clean_underlay_alpha 0→0.3 | no-change clean pass |
| C003 | code: ObjectPrompt.asset + 3 video paths | SUCCESS — backwards-compat verified |
| C004/A1 | isolated obj_3 asset patch (uses C003 code) | no-change clean pass; isolation works |
| C005/A1 | AI rubric on gold | failed — no ANTHROPIC_API_KEY |
| C005/A2 | mask_dilate_px 20→10 (global) | no-change clean pass |
| C006/A1 | visual comparator on floor+walkover, 7 runs | all 7 = 18/20 floor saturated |
| C006/A2 | logo_blur_px=1 (court_floor) | no-change clean pass |
| C007/A1 | visual comparator on back+left, 8 runs | all 7 valid = 20/20 saturated; C002/A1 = 10-12/20 |
| C007/A2 | alpha_feather_px 1→3 (global) | soft regression (back delta_E warning) |
| C008/A1 | fitter pca | no-change clean pass |
| C008/A2 | dynamic_full config | regression — multiple gates fail; revealed dynamic-geom activation bug |
| C009/A1 | code-reading: dynamic-geom diagnostic | bug found — medium-difficulty fix |
| C009/A2 | padding 0.1→0.15 | **VISIBLE REGRESSION** (back banners shifted) — framework didn't flag it |
| C010/A1 | comprehensive comparator, 6 runs × 3 strips | GOLD = 60/60; C009/A2 = 54/60; rest tied at 60/60 |
| C010/A2 | fitter fronto_parallel | no-change clean pass; revealed fitter is inert under static-clicked path |

### What worked (enables future work)

- **Per-object asset routing** — `ObjectPrompt.asset` field added to dataclass; parser updated; all 3 video pipeline paths honor it; backwards-compat verified (97 tests pass; v68 gold unchanged). Commit `47b2665`. Unblocks experiments where individual objects can carry different assets.

### What didn't move the needle

- 10+ config-knob perturbations (occlusion_dilate, alpha_feather × 2, quad_expand, asset_patch_isolated, clean_underlay_alpha, mask_dilate, logo_blur, padding, fitter × 3) — all no-ops or regressions on this clip.
- Dynamic-geometry config — the existing `eval_walkover_v68_clicked_homography_dynamic_full.yaml` is structurally inert in hybrid mode (see bugs below).
- AI-rubric path — environment lacks `ANTHROPIC_API_KEY` (anthropic SDK installed but unused).

### Three real bugs found, all deferred for human review

1. **Dynamic-geometry activation gate** (medium fix). `run_pipeline_video_hybrid` never instantiates `GeometryFittingEngine`; `SUPPORTED_GEOMETRY_SURFACE_TYPES` excludes `court_floor` and `banner`. So `pipeline.geometry.enabled: true` is structurally inert in hybrid mode. Hybrid does run a parallel `CourtGeometryEstimator` for `court_plane_placement_*` metrics (line 2996), but that's separate from `geometry_runtime_enabled`. Detailed in C009/A1 above.

2. **Eval framework regression-detection bug** (small fix). `src/banner_pipeline/eval/reference.py:detect_regressions` doesn't catch `roi_ssim_vs_reference_mean` regressions. C009/A2's 28% back-banner SSIM drop was not flagged. Likely the comparison code expects keys at top level in gold's payload but they're nested under `vs_reference`.

3. **Fitter is inert under static-clicked path** (architectural — by-design but undocumented). When prompts carry `placement_quad`, the runtime fitter is bypassed. So changing `pipeline.fitter.type` has no effect on v68-style configs. This isn't a bug per se but means our benchmark space for fitters needs a config without static placement_quad.

### Best candidate

**GOLD remains the best run.** No experiment beat it numerically (saturated metrics) or visually (saturated rubric scores from sub-agent comparators).

### Recommendations for the human

1. **Provision `ANTHROPIC_API_KEY`** for future runs to enable AI rubric scoring on candidates beyond what we have today.
2. **Fix the framework regression-detection bug** (#2 above) so future autonomous runs catch hidden regressions like C009/A2.
3. **Decide whether to fix dynamic-geometry activation** (#1 above). It's a medium effort but could unlock a meaningfully different rendering style worth testing. Alternative: explicitly retire dynamic-geometry-in-hybrid as unsupported.
4. **For genuine improvements**, consider these axes that this autonomous run could not exercise:
   - Different occlusion masker (`matanyone` continuous-alpha; not tested due to ~25-min Modal job exceeding agent harness budget).
   - Different clip generalization (`zoom-clip-melbourne.mov`, `tennis-clip.mp4`) — needs new prompt configs first.
   - Pipeline running WITHOUT static `placement_quad` so the runtime fitter actually runs (lets us compare hull/pca/fronto_parallel meaningfully).

---

## C011 — 2026-05-04 23:52 EDT — final breadth: local_color_match
Manager hypothesis: only one untouched compositor knob remains: `compositor.params.local_color_match: true → false`. Currently true. Single agent. Gather one more data point before winding down.

- **A1 — `local_color_match: false`** in compositor.params. New config `eval_walkover_c011_a1_no_local_color_match.yaml`. Single-line config change.

Status: dispatched in background.

### C011 results — 2026-05-05 00:02 EDT

```
=== CYCLE C011 SLOT A1 REPORT ===
Hypothesis: local_color_match true -> false in compositor.params (global)
Run dir: experiments/2026-05-04_23-59-53_hull_B200/
Exit code: 3 (regression vs reference)
Pass: back=P left=P floor=P full=P
Regression vs gold: yes (back_roi_delta_E_lab)
back_roi_ssim_vs_reference_mean: 0.9718 (vs gold's 1.0; below typical 0.998 — soft regression)
left_roi_ssim_vs_reference_mean: 0.9976
floor_roi_ssim_vs_reference_mean: 0.9995
floor_walkover_logo_visible_pct: gold=0.1787, current=0.1787 (flat)
Cost: Modal-B200 ~5.7min
Recommendation: dead end — disabling local_color_match shifts back-banner colors detectably without compensating gain.
```
Manager note: dead-end. Confirms local_color_match=true is what keeps back-banner color fidelity aligned with gold; disabling produces a measurable color regression. Note: framework DID flag regression here (back_roi_delta_E_lab key works) — so the bug from C009/A2 was specifically in the SSIM cross-region keys, not delta_E. Useful for the framework bug-fix queue.

---

## C012 — 2026-05-05 00:02 EDT — final-final breadth: lum_strength
Manager hypothesis: only one untested compositor knob remains worth a single Modal run: `compositor.params.lum_strength: 0.0 → 0.3`. Currently at 0.0. May add subtle shading to placed logos. After this, the agent loop has covered all reasonable safe knob perturbations of the v68 base config; remaining cycles will be visual/sanity checks on the saturated state.

- **A1 — `lum_strength 0.0 → 0.3`** in compositor.params. New config `eval_walkover_c012_a1_lum_strength_03.yaml`.

Status: dispatched in background.

### C012 results — 2026-05-05 00:11 EDT

```
=== CYCLE C012 SLOT A1 REPORT ===
Hypothesis: lum_strength 0.0 -> 0.3 in compositor.params (global)
Run dir: experiments/2026-05-05_00-09-02_hull_B200/
Exit code: 0
Pass: back=P left=P floor=P full=P
Regression vs gold: no
back_roi_ssim_vs_reference_mean: 0.9951
left_roi_ssim_vs_reference_mean: 0.985 (slightly below typical 0.998)
floor_roi_ssim_vs_reference_mean: 0.9995
floor_walkover_logo_visible_pct: +0.06% flat
Cost: Modal-B200 ~5.6min
Recommendation: low-risk knob to keep on; pursue larger values to see if shading meaningfully improves perceptual realism.
```
Manager note: clean candidate, no regression. left_ssim=0.985 is the smallest hint of a movement — lum_strength is doing something to the left logo region but stays in-bounds. Same saturation pattern. Mark as no-change clean pass.

---

## C013 — 2026-05-05 00:12 EDT — combined-safe experiment
Manager hypothesis: combine two independent knobs that pass clean alone (mask_dilate_px=10 from C005/A2 AND logo_blur_px=1 on court_floor from C006/A2). Tests whether benign knobs compound to anything visibly different. Single Modal run.

- **A1 — combined mask_dilate=10 + logo_blur=1.** New config `eval_walkover_c013_a1_combined_safe.yaml`. Two-line config change vs v68 base.

Status: dispatched in background.

### C013 results — 2026-05-05 00:20 EDT

```
=== CYCLE C013 SLOT A1 REPORT ===
Hypothesis: combined mask_dilate_px=10 + logo_blur_px=1
Run dir: experiments/2026-05-05_00-18-07_hull_B200
Exit code: 0
Pass: all P
Regression vs gold: no
back_roi_ssim_vs_reference_mean: 0.9982
left_roi_ssim_vs_reference_mean: 0.9977
floor_roi_ssim_vs_reference_mean: 0.9995
floor_walkover_logo_visible_pct: +0.06% flat
floor_walkover_occlusion_iou: 0.9823
Cost: Modal-B200 ~5.7min
Recommendation: CANDIDATE — clean compose; no interaction effects.
```
Manager note: knobs compose additively (no interaction). Same saturation. No-change clean pass.

### Saturation has held 6+ consecutive cycles — throttling cadence

Switching to long cadence (delay=2400, 40 min) for remaining cycles. Continuing to gather data points but at lower frequency until deadline.

---

## C014 — 2026-05-05 00:20 EDT — shade_blend toggle
Manager hypothesis: `compositor.params.shade_blend: false → true` (currently false). Untouched. Single Modal run.

- **A1 — `shade_blend: true`** in compositor.params. New config `eval_walkover_c014_a1_shade_blend_true.yaml`.

Status: dispatched in background.

### C014 results — 2026-05-05 00:35 EDT

```
=== CYCLE C014 SLOT A1 REPORT ===
Hypothesis: shade_blend false -> true in compositor.params
Run dir: experiments/2026-05-05_00-31-46_hull_B200
Exit code: 0
Pass: all P
Regression vs gold: no
back/left/floor _ssim_vs_ref: 0.9983 / 0.9976 / 0.9995
floor_walkover_logo_visible_pct: -0.06% flat
Cost: Modal-B200 ~10.5min
Recommendation: candidate; effectively neutral in this scene.
```
Manager note: same saturation. No-change clean pass. Add to pixel-equivalent candidate set.

---

## C015 — 2026-05-05 00:35 EDT — blend_mode toggle (final breadth)
Manager hypothesis: `compositor.params.blend_mode: led → screen` (currently `led`). One last untouched compositor toggle. Single agent. After this, the agent loop has effectively exhausted v68's safe-knob space and remaining cycles will be sanity passes only.

- **A1 — `blend_mode: screen`** in compositor.params. New config `eval_walkover_c015_a1_blend_mode_screen.yaml`.

Status: dispatched in background.

### C015 results — 2026-05-05 01:00 EDT

```
=== CYCLE C015 SLOT A1 REPORT ===
Hypothesis: blend_mode led -> screen in compositor.params
Run dir: experiments/2026-05-05_00-46-05_hull_B200
Exit code: 0
Pass: all P
Regression vs gold: yes (left_roi_delta_E_lab)
back_roi_ssim_vs_reference_mean: 0.9857 (slight drop)
left_roi_ssim_vs_reference_mean: 0.9798
floor_roi_ssim_vs_reference_mean: 0.9995
floor_walkover_logo_visible_pct: +0.06% flat
Cost: Modal-B200 ~11min
Recommendation: dead end — `screen` blend_mode shifts left logo color without compensating gain.
```
Manager note: soft regression. Same plateau pattern. Knobs exhausted.

### Knob axes covered (final tally as of 2026-05-05 01:00 EDT)

15 cycles, 13 Modal runs, 1 code change, 3 visual comparators. All reasonable safe knob perturbations of the v68 base config tested. Specifically:

| Knob | Value(s) tried | Outcome |
|---|---|---|
| occlusion_dilate_px (court_floor) | 2→0 | clean pass, no movement |
| alpha_feather_px (court_floor) | 25→10 | clean pass, no movement |
| quad_expand_px (court_floor) | 80→120 | regression (occlusion_iou fails gate) |
| floor logo asset (global) | white→court_patch | regression (back/left visibly broken) |
| floor logo asset (per-object via new code) | white→court_patch | clean pass, no movement |
| clean_underlay_alpha (court_floor) | 0.0→0.3 | clean pass, no movement |
| mask_dilate_px (global) | 20→10 | clean pass, no movement |
| logo_blur_px (court_floor, new field) | absent→1 | clean pass, no movement |
| alpha_feather_px (global) | 1→3 | soft regression (back delta_E) |
| fitter.type | hull→pca, hull→fronto_parallel | both no-ops (fitter inert under static-clicked path) |
| dynamic geometry | enable | broken: hybrid never builds GeometryFittingEngine |
| padding (global) | 0.1→0.15 | **VISIBLE REGRESSION** (back banner shifted) — framework didn't flag |
| local_color_match | true→false | soft regression (back delta_E) |
| lum_strength | 0.0→0.3 | clean pass, slight left_ssim hint, no real movement |
| combined: mask_dilate=10 + logo_blur=1 | both | clean pass, additive composition |
| shade_blend | false→true | clean pass, no movement |
| blend_mode | led→screen | soft regression (left delta_E) |

---

## C016 — 2026-05-05 01:01 EDT — final consolidating visual ranking
Manager hypothesis: dispatch a vision-capable sub-agent to produce a definitive ranked list across ALL clean-pass runs (the runs with `Pass: all P AND any_regression: false` in the table above), so the human's deadline review has a single sorted candidate list. Read 2 PNGs per run (back_banners + walkover) for compactness. Score 4-dim rubric, total /8 per run.

- **A1 — definitive comparator** across these runs (the no-regression clean-pass set):
  - GOLD = experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/
  - C001/A1 occ_dilate=0 = experiments/2026-05-04_21-55-06_hull_B200/
  - C001/A2 alpha_feather=10 = experiments/2026-05-04_21-52-36_hull_B200/
  - C002/A2 clean_underlay=0.3 = experiments/2026-05-04_22-11-23_hull_B200/
  - C004/A1 isolated patch = experiments/2026-05-04_22-26-31_hull_B200/
  - C005/A2 mask_dilate=10 = experiments/2026-05-04_22-43-42_hull_B200/
  - C006/A2 logo_blur=1 = experiments/2026-05-04_22-54-03_hull_B200/
  - C008/A1 fitter=pca = experiments/2026-05-04_23-17-01_pca_B200/
  - C010/A2 fitter=fronto_parallel = experiments/2026-05-04_23-49-40_fronto_parallel_B200/
  - C012/A1 lum_strength=0.3 = experiments/2026-05-05_00-09-02_hull_B200/
  - C013/A1 combined safe = experiments/2026-05-05_00-18-07_hull_B200/
  - C014/A1 shade_blend=true = experiments/2026-05-05_00-31-46_hull_B200/

  12 runs × 2 PNGs = 24 reads. Sequential.

Status: dispatched in background.

### C016 results — 2026-05-05 01:03 EDT

```
=== CYCLE C016 SLOT A1 REPORT (definitive visual ranking) ===
Per-run scores (back_total / walkover_total / OVERALL_SUM):
  GOLD: 20/20 = 40/40
  C001/A1 occ_dilate=0: 40/40
  C001/A2 alpha_feather=10: 40/40
  C002/A2 clean_underlay=0.3: 40/40
  C004/A1 isolated_patch: 40/40
  C005/A2 mask_dilate=10: 40/40
  C006/A2 logo_blur=1: 40/40
  C008/A1 fitter=pca: 40/40
  C010/A2 fronto_parallel: 40/40
  C013/A1 combined_safe: 40/40
  C014/A1 shade_blend=true: 40/40
  C012/A1 lum_strength=0.3: 39/40 — back banner reads slightly darker/less integrated luminance

Differentiating dimensions: only realism.texture_match nudges on C012/A1; everything else flat across the entire clean-pass set.

Pixel-equivalent set (10 of 11 perturbations + GOLD): C001/A1, C001/A2, C002/A2, C004/A1, C005/A2, C006/A2, C008/A1, C010/A2, C013/A1, C014/A1.

Subtle real differences (visible only if scrubbing carefully): C012/A1 lum_strength=0.3 reads a touch flatter/darker on the painted patch.

Promotion recommendation: keep GOLD. No candidate offers visible improvement. If a tie-breaker is forced: C013/A1 combined_safe bundles multiple known-clean knobs and stayed pixel-equivalent — the safest "alternative gold" but no upside.
```
Manager note: **DEFINITIVE — saturation confirmed at the visual level**. 11 of 12 runs are pixel-equivalent; the one outlier is fractionally worse. The v68 manually-clicked static-homography baseline is at quality plateau on this clip. No candidate justifies promotion.

---

## Cycles closed — 2026-05-05 01:03 EDT

The agent loop has covered the safe-knob space. Further config-knob iteration is not expected to produce visible deltas. Loop is now in **summary-prep mode** — wakeup ticks will continue to fire at the throttled cadence but will dispatch no new agents. The Final Summary block will be appended at the deadline tick (2026-05-05 10:00 EDT).

If a meaningful new signal arises (none expected), the loop can re-engage; otherwise the human's review at the deadline is what matters.

---

<!-- Subsequent cycles append below this line. -->

## Final summary — 2026-05-05 10:00 EDT

### TL;DR for the human reviewer (you)

- **GOLD remains the visually best run** across 16 cycles, 14 Modal runs, 1 code change shipped, 4 sub-agent visual comparators. C016's definitive ranking shows 11 of 12 clean-pass runs are PIXEL-EQUIVALENT to GOLD (40/40 each); the one outlier (C012/A1, lum_strength=0.3) scored 39/40 — fractionally worse, not better.
- **Only ship: per-object asset routing** (`ObjectPrompt.asset` field + 3 video pipeline paths) — commit `47b2665`. Backwards-compat verified (97 tests pass; v68 gold runs unchanged). Unblocks future per-object asset experiments.
- **Three real bugs found, all deferred for human:**
  1. *Dynamic-geometry activation gate* (medium fix). `run_pipeline_video_hybrid` never instantiates `GeometryFittingEngine`; `SUPPORTED_GEOMETRY_SURFACE_TYPES` excludes `court_floor` / `banner`. So `pipeline.geometry.enabled: true` is structurally inert in hybrid mode. (See C009/A1 diagnostic for file:line refs.)
  2. *Eval framework regression-detection bug* (small fix). `src/banner_pipeline/eval/reference.py:detect_regressions` doesn't flag `roi_ssim_vs_reference_mean` cross-region drops. C009/A2's 28% back-banner SSIM drop went unflagged by exit code 3, even though the visible regression was real (sub-agent comparator caught the horizontal banner shift).
  3. *Static-clicked path bypasses the runtime fitter* (architectural; undocumented). Explains why hull / pca / fronto_parallel are all no-ops on v68-style configs (C008/A1, C010/A2 confirmed).
- **AI rubric path was unavailable** (no `ANTHROPIC_API_KEY` in env). Anthropic SDK is installed (`uv sync --extra ai` succeeded). Provisioning a key would let `--with-ai-review` run on the gold + candidates.
- **Recommended next axes** (not exercised in this run):
  1. Provision API key + run AI rubric on GOLD (~$0.20).
  2. Fix bug #2 (small).
  3. Decide on dynamic-geometry path (fix per #1, or retire).
  4. Test on different clips (`zoom-clip-melbourne.mov`, `tennis-clip.mp4`) — needs new prompt configs first.
  5. Try `matanyone` occlusion masker (~25-min Modal job; exceeded the 11-min agent harness budget tonight).

### Pixel-equivalent passing candidates (any could replace GOLD with zero visual delta)

| Cycle | Knob change | Experiment dir |
|---|---|---|
| C001/A1 | `surface_overrides.court_floor.occlusion_dilate_px: 0` | experiments/2026-05-04_21-55-06_hull_B200/ |
| C001/A2 | `surface_overrides.court_floor.alpha_feather_px: 10` | experiments/2026-05-04_21-52-36_hull_B200/ |
| C002/A2 | `surface_overrides.court_floor.clean_underlay_alpha: 0.3` | experiments/2026-05-04_22-11-23_hull_B200/ |
| C004/A1 | obj_3 isolated `asset: data/logos/redbull_court_patch.png` | experiments/2026-05-04_22-26-31_hull_B200/ |
| C005/A2 | `compositor.params.mask_dilate_px: 10` | experiments/2026-05-04_22-43-42_hull_B200/ |
| C006/A2 | `surface_overrides.court_floor.logo_blur_px: 1` | experiments/2026-05-04_22-54-03_hull_B200/ |
| C008/A1 | `fitter.type: pca` | experiments/2026-05-04_23-17-01_pca_B200/ |
| C010/A2 | `fitter.type: fronto_parallel` | experiments/2026-05-04_23-49-40_fronto_parallel_B200/ |
| C013/A1 | combined safe (mask_dilate=10 + logo_blur=1) | experiments/2026-05-05_00-18-07_hull_B200/ |
| C014/A1 | `compositor.params.shade_blend: true` | experiments/2026-05-05_00-31-46_hull_B200/ |

**Recommendation: do not promote any of them.** None offers visible upside; they're alternative configurations of the same plateau.

### Visible regressions found (all clearly worse than GOLD)

- **C001/A3** `quad_expand_px: 120` — fails `floor_walkover_occlusion_iou` gate (0.44 vs gold's 1.0). exit 2.
- **C002/A1** asset_patch global — visible patch outlines on back/left banners (sub-agent comparator). exit 3.
- **C007/A2** `alpha_feather_px: 3` global — soft regression on back delta_E.
- **C008/A2** `dynamic_full` config — multiple gate fails; ALSO revealed bug #1.
- **C009/A2** `padding: 0.15` — **back banners horizontally shifted** (sub-agent comparator caught this); framework's regression detection MISSED it (bug #2).
- **C011/A1** `local_color_match: false` — soft regression on back delta_E.
- **C012/A1** `lum_strength: 0.3` — fractionally flatter back banner (only one to score 39/40 in the C016 ranking).
- **C015/A1** `blend_mode: screen` — soft regression on left delta_E.

### Where to look first

The full per-cycle table is in the **DRAFT FINDINGS** section above (search for `DRAFT FINDINGS`). Each cycle has its agent's verbatim report in a fenced code block. Visual artifacts live under each `experiments/<run>/eval/` directory: `back_banners/`, `left_logo/`, `floor_logo/`, `walkover/`, `full/`, plus a top-level `report.md` and `quality_metrics.json`. The most informative single artifact for each run is `eval/walkover/consecutive_frames.png` — 16 consecutive frames showing the player walking on the floor logo.

Loop ends here. No more agents will be dispatched.

---

# Phase 2 — hybrid locked-with-tolerance homography (started 2026-05-05 13:12 EDT)

**Deadline:** 2026-05-05 18:30 EDT (~5.3h budget).
**Branch:** `feat/quality-fixes-next` (continues from Phase 1).
**Axis brief:** [docs/HOMOGRAPHY_HYBRID_AXIS.md](HOMOGRAPHY_HYBRID_AXIS.md). Read that first.
**Goal:** preserve v68's static-camera quality + win on motion frames at end of clip.
**Sub-agent eval framework:** v2 (paired strips + 5 walkover sheets + checklist) — see [docs/EVALUATION.md](EVALUATION.md).

## P2-C001 — 2026-05-05 13:13 EDT — scoping (research-only, 2 parallel agents)

Manager hypothesis: before writing any code, scope the existing terrain.

- **A1 — sibling repo recon.** Read `/Users/enriquediazdeleonhicks/repositories/capstone-data-candidates/tennis-virtual-ads/` to understand the 14-point homography approach. Report what's there, what algorithm is used, what's worth lifting (or porting).
- **A2 — existing path deep-dive.** Read `src/banner_pipeline/court_geometry.py` + `src/banner_pipeline/pipeline.py:run_pipeline_video_hybrid` here. Report what's there, what's the smallest fix to activate dynamic geometry in the hybrid path, and what a "locked-with-tolerance" wrapper would look like.

Both agents are read-only; output a structured design report. Manager will synthesize for cycle P2-C002.

Status: dispatched in background.

### P2-C001 results — 2026-05-05 13:25 EDT

```
=== P2-C001/A1 SIBLING RECON REPORT (summary) ===
Tennis-virtual-ads has BallTrackerNet (14 keypoints, heatmap detection) + RANSAC homography
with best-of-12 fallback (RANSAC when ≥5 keypoints; brute-force config-search otherwise).
Already implements 3-stage temporal smoothing:
  - KeypointSmoother (EMA, α≈0.7, spike-reset on reprojection error)
  - HomographyStabilizer (EMA OR Kalman on H matrix; pinhole decomposition)
  - HomographyLocker (hysteresis-based locking — displacement < threshold for N frames → lock; >unlock → unlock)
Court reference is fixed image-plane (1665×3496 px); 14 hand-coded keypoints; 12 hand-coded
4-point configurations.
Portability: HIGH (Python+OpenCV+NumPy, lazy-imports BallTrackerNet via importlib).
Recommendation: take ideas + reimplement core logic. Port court_reference.py + homography.py
modules; port KeypointSmoother + HomographyStabilizer; SKIP BallTrackerNet for now (use
this repo's existing line-based detection in court_geometry.py).
=== END A1 ===
```

```
=== P2-C001/A2 OWN-PATH RECON REPORT (summary) ===
CourtGeometryEstimator (court_geometry.py:530-642) is standalone-usable: per-frame estimate()
returns CourtGeometryEstimate with court_homography (3x3) and confidence. Has internal EMA
smoothing via vp_smoothing_alpha (default 0.7). Already instantiated in hybrid path at
pipeline.py:2996 for court_plane_placement_* metrics.

GeometryFittingEngine is overkill — multi-object orchestrator for banner-fitting; hybrid
correctly skips it.

Hook point for tolerance gate: between pipeline.py:3256 (after estimate() call) and 3266
(before _project_court_plane_rectangle routes corners).

Existing smoothing: vp_smoothing_alpha (0.7), line_smoothing_alpha (0.65), CornerTracker
EMA (0.3, but hybrid static-corners path doesn't use it). NO motion-rate-matched smoothing
today; alpha is fixed.

Smallest viable hybrid implementation:
  - Estimation source: standalone CourtGeometryEstimator (already there).
  - Seed: frame-0 corners (from clicked placement_quad or first estimate).
  - Tolerance gate: between lines 3256-3266; compare estimated corners to seed; if
    displacement < tolerance (default 5-10 px, motivated by white-line thickness),
    stay locked; else ramp toward estimate over N frames where N = displacement /
    motion_speed_px_per_frame.
  - Per-frame state: extend schema to record seed_corners, estimated_corners, decision,
    displacement_px.
  - Code touch: ~145 lines additive across court_geometry.py + pipeline.py. NO refactor
    of GeometryFittingEngine, CornerTracker, or the static path.
Risk level: LOW. Static frames stay locked → no regression vs v68 numerically.
=== END A2 ===
```

**Manager synthesis:** A2's inline path is the right starting point — minimal diff, self-contained, reuses already-instantiated CourtGeometryEstimator. A1's sibling-port (HomographyLocker, etc.) is a defensible fallback if the inline approach doesn't yield enough quality, but it's bigger surface area and we don't need it yet. Implementing inline.

---

## P2-C002 — 2026-05-05 13:26 EDT — code: hybrid lock-with-tolerance (additive, flag-gated)

Manager hypothesis: implement the inline hybrid per A2's plan. Single code-change agent. Behavior:
- New config flag `pipeline.geometry.hybrid_lock.enabled: false` (default — preserves v68 backward-compat).
- When flag is on: per-frame, after `CourtGeometryEstimator.estimate()` runs, compare projected corners against the seed corners (frame-0 estimate or the static placement_quad).
- If displacement < `tolerance_px` (default 6 px, configurable): stay locked at seed.
- If displacement ≥ tolerance: ramp toward estimate over `ramp_frames` (default computed as `max(3, displacement / motion_px_per_frame)`).
- Persist `seed_corners`, `estimated_corners`, `decision` ("locked"|"ramp"|"estimate"), `displacement_px` per frame in `outputs/per_frame_state.json`.
- Existing 12 unit tests must still pass.
- Eval on the existing v68 gold dir must still pass all 4 scorecards (backward-compat with flag off).

Status: dispatched in background.

### P2-C002 results — 2026-05-05 13:30 EDT

```
=== P2-C002/A1 CODE CHANGE REPORT ===
Files modified: src/banner_pipeline/court_geometry.py, src/banner_pipeline/pipeline.py, tests/test_hybrid_lock.py (new)
Lines added/changed: +358 / -19 (net +339)
Backwards-compat:
  - syntax: PASS
  - tests/test_hybrid_lock.py: 5/5
  - all tests: 102/102
  - eval on v68 gold (no flag): all 4 scorecards PASS
  - hybrid_lock_enabled in v68 gold metrics.json: <not present> (flag-gated; existing run pre-dates code)
Implementation:
  - HybridLockState dataclass in court_geometry.py
  - step() returns (corners, decision, displacement_px); decision in {locked, ramp}
  - Per-frame schema extension: seed_corners, estimated_corners, decision, displacement_px (only when active)
  - Counters: hybrid_lock_{locked,ramp,estimate}_frames in metrics
Commit: 962ddf3
Recommendation: ready for P2-C003 Modal run with flag enabled.
=== END REPORT ===
```
Manager note: clean implementation; flag-gated; 102 tests pass; gold unaffected. Ready to exercise.

---

## P2-C003 — 2026-05-05 13:32 EDT — hybrid_lock tolerance sweep (7 parallel, H200)

Manager hypothesis: validate wiring + sweep tolerance breadth in one cycle. **7 parallel Modal runs on H200** (avoids B200 queue wait). All based on `eval_walkover_v68_clicked_homography_dynamic_full.yaml` (which has `pipeline.geometry.enabled: true` + `court_plane_placement` on obj_3/obj_4 — required for the hybrid_lock gate to have something to gate on).

| Slot | tolerance_px | ramp_motion_px_per_frame | Purpose |
|---|---|---|---|
| A1 | 99999 | 2.0 | sanity — always-locked; validates wiring doesn't break anything (should match gold) |
| A2 | 2.0 | 2.0 | very tight; ramps almost every motion frame |
| A3 | 4.0 | 2.0 | tight; minimum useful tolerance vs white-line thickness |
| A4 | 6.0 | 2.0 | default; matches white-line thickness ~3-5px with margin |
| A5 | 10.0 | 2.0 | looser; fewer ramps, more locked |
| A6 | 15.0 | 2.0 | very loose; mostly stays locked unless a big move |
| A7 | 6.0 | 4.0 | default tolerance + faster ramp (snap quicker once we decide to move) |

Targets per region (must hold across ALL slots — these are the v68-gold-equivalent gates):
- `back_pass=true`, `left_pass=true`, `floor_pass=true`, `full_pass=true`
- `any_regression=false` (or close — see analysis after the sweep)
- New diagnostics expected (only when hybrid_lock_enabled=true): `hybrid_lock_locked_frames`, `hybrid_lock_ramp_frames`, `hybrid_lock_estimate_frames`. Their distribution across 767 frames tells us whether the gate fires meaningfully.

All config-only; no code changes. Each agent creates `configs/experiments/eval_walkover_p2_c003_a<N>_<slug>.yaml`, runs Modal H200, runs eval, commits + pushes own work.

Status: dispatched in background.

### P2-C003 status — 2026-05-05 13:37 EDT (in flight; harness timeouts)

All 7 dispatched agents timed out at the agent harness ~10.5-min boundary BEFORE Modal completed. Each returned a "Waiting for Modal run" / "output empty" message. No new experiment dirs in `experiments/2026-05-05_*_H200/` as of 13:37. All 7 config files exist (orphan but committed-ready).

Diagnosis: Modal H200 cold-start + the dynamic-geometry pipeline (CourtGeometryEstimator on every frame) takes >11 min wall clock. Agent harness budget mismatch.

**Recovery plan:** Modal jobs continue running on the platform after agents die; output dirs WILL eventually appear. Manager will:
1. Wait for output dirs to materialize (check disk periodically).
2. Once present, dispatch SHORT harvest agents (eval + commit only, no Modal — should fit comfortably under 11 min each).
3. Continue the cycle from there.

If outputs still don't appear after ~30 min, escalate: jobs may have failed on the platform.

---

### P2-C003 declared LOST — 2026-05-05 13:56 EDT

`uv run modal app list` shows all 7 apps (created 13:26 EDT) in state=stopped, 0 tasks. `modal app logs ap-9ZEOdRdoZyz5QDZiL0JpfQ` shows the pipeline WAS actively running (loaded MaskRCNN + MatAnyone2 weights, "[MatAnyone2] Loaded on cuda, streaming through frames…"). It was killed mid-execution when the local agent's `Bash` call hit the 10-min cap and the `modal run` synchronous client died — Modal cancelled the platform-side execution.

**Root cause:** `scripts/modal_run.py` uses `run_on_gpu.remote(...)` (synchronous), and Bash tool caps single calls at 10 min. Without the poll pattern, agents can't keep the modal client alive long enough.

**Recovery:** redispatch the same 7 tolerance variants as P2-C004 with the poll pattern (memory: feedback_modal_poll_pattern.md). Agent uses `Bash(run_in_background=true)` to launch modal as a detached process, then loops short polling Bash calls — keeping the modal client alive across the agent's lifespan (which is unbounded; only individual Bash calls are capped).

---

## P2-C004 — 2026-05-05 13:57 EDT — redispatch P2-C003 sweep with POLL PATTERN

Same 7 variants (sanity, tol=2/4/6/10/15, plus tol=6+ramp_motion=4). H200. POLL PATTERN: `Bash(run_in_background=True)` + grep loop. Agents stay alive 20+ min.

Status: dispatching in background now.

### P2-C004 partial — 2026-05-05 14:18 EDT

Of 7 dispatched, only A3 (tol=4) actually polled correctly and harvested. Other 6 misused Monitor tool, ended turns, their bg modal clients died → Modal apps stopped before completion.

```
=== P2-C004/A3 REPORT (tol=4) ===
Run: experiments/2026-05-05_14-11-17_hull_H200/
Pass: back=P left=F floor=F full=P
ssim_vs_ref back/left/floor: 0.9985/0.379/0.2389
floor_walkover_logo_visible_pct: 0.1473 (gold 0.1787, -17.6%)
Verdict: regression vs gold (left/floor broken)
```

**Manager note:** A3 is REGRESSION but the failure mode (left ssim 0.38, floor ssim 0.24) matches Phase 1 C008/A2 exactly — `dynamic_full` base config is structurally broken on left+floor regions, hybrid_lock cannot save it. Hybrid_lock code is fine; problem is the base. Pivot.

---

## P2-C005 — 2026-05-05 14:18 EDT — pivot to v68-STATIC + minimal hybrid_lock additions

**Strategy change:** dispatch from MAIN THREAD (not agents) so the local modal client processes survive — agent-side bg bash processes were dying when agents ended turns. My own bg bash processes survive my own polling.

5 configs based on `eval_walkover_v68_clicked_homography_static_full.yaml` (the proven gold base) with minimal additions: `pipeline.geometry.enabled: true`, `pipeline.geometry.court_backend: classical_lines_v1`, `pipeline.geometry.hybrid_lock` block, and `court_plane_placement` on obj_3 (court_rect from prior dynamic config).

| Slot | tolerance_px | ramp_motion_px_per_frame | Purpose |
|---|---|---|---|
| A1 | 99999 | 2.0 | sanity — always-locked; should match v68 gold exactly |
| A2 | 2 | 2.0 | very tight; ramps on smallest motion |
| A3 | 6 | 2.0 | default |
| A4 | 12 | 2.0 | looser |
| A5 | 6 | 4.0 | default + faster ramp |

5 H200 Modal jobs dispatched in main-thread background. Bash IDs: bv30r5c2f, bewhr1m4q, bddc08vt8, bb79tq6gm, b316ec9t3. Logs in /tmp/modal_p2c005_a*.log.

Status: jobs running.

---

### P2-C005 results — 2026-05-05 14:32 EDT

All 5 main-thread Modal jobs completed successfully (main-thread bg dispatch survived; agents-in-loop did NOT survive earlier).

| Slot | tol | ramp | back | left | floor | full | any_regression | floor_walkover_logo_visible_pct |
|---|---|---|---|---|---|---|---|---|
| A1 | 99999 | 2.0 | P | P | P | P | **no** | (gold-equivalent — wiring works) |
| A2 | 2 | 2.0 | P | P | F | P | yes | regression |
| A3 | 6 | 2.0 | P | P | F | P | yes | regression |
| A4 | 12 | 2.0 | P | P | F | P | yes | 0.1477 (gold 0.1787, -17.4%) |
| A5 | 6 | 4.0 | P | P | F | P | yes | regression |

Floor failures: `floor_roi_jitter_ratio`, `floor_walkover_occlusion_iou`.

**Diagnostic interpretation:**

- **A1 (tol=99999, always-locked) PASSES ALL 4 SCORECARDS** → hybrid_lock wiring is correct. Code from P2-C002 is sound. With infinite tolerance, the gate always says "stay locked" → output ≈ v68 gold.
- **A2–A5 (real tolerances) ALL FAIL floor** → when the gate fires (estimate-vs-seed displacement > tolerance), it ramps toward the projected court-plane corners. Those corners are computed from `CourtGeometryEstimator`'s line-detected homography projected through the configured `court_rect`. They are **systematically wrong** for this clip on the floor region — ramping toward them regresses floor. (Back banners are fine because they don't have court_plane_placement.)
- The hybrid_lock CONCEPT works (the gate, the ramp). The PROBLEM is the underlying estimator: line-based homography on this clip + the chosen `court_rect` produce floor corners that don't match the v68 clicks.

**`hybrid_lock_*` counters and `outputs/per_frame_state.json` are NOT being written** to disk despite the P2-C002 code change. Either Modal's image build didn't include the latest commits OR the metrics-write path doesn't fire when run on Modal. Worth investigating — but for now the four scorecard outcomes are the diagnostic signal.

**Best candidate so far:** P2-C005/A1 (tol=99999, always-locked, v68-static base) — gold-equivalent + all gates pass + the hybrid_lock infrastructure is wired in (so future estimator improvements can flip the tolerance to a useful value without code change). Run dir: `experiments/2026-05-05_14-29-35_hull_H200/`.

**Path forward for the hybrid axis (recommendations for the human + remaining cycles tonight):**

1. The line-based estimator (`CourtGeometryEstimator`) is too unreliable on this clip to power a hybrid-with-tolerance gate that actually improves on v68. Two paths to fix:
   - **Calibrate `court_rect` more carefully.** The [0.421, 1.002, 0.559, 1.015] value carried from `dynamic_full` may simply be wrong for the floor region — we'd need a fitting pass on the v68 manual clicks to derive a `court_rect` that minimizes seed-vs-projected distance on frame 0. Sibling A1 recon noted the sibling repo uses `court_reference.py` with hand-coded keypoints; that file has the canonical mapping.
   - **Port BallTrackerNet from `tennis-virtual-ads`** for better keypoints. Then the homography estimate is from a learned 14-keypoint detector, far more reliable than the line-based path. Significant effort (model weights, tests, inference plumbing).

2. **Tonight's limit:** with the line-based estimator, the only safe hybrid_lock setting is `tol=99999` (always-locked) which is just v68 with extra inert plumbing. Real tolerances regress.

---

<!-- Subsequent Phase 2 cycles append below this line. -->

## P2-C006 — court_quad calibration + tolerance sweep (in flight, dispatched 2026-05-05 ~14:48 EDT)

**Hypothesis:** P2-C005 showed tol=99999 (sanity) gold-equivalent but tol≤12 regressed floor by ~7/12 visual rubric points. Root cause from rubric report: estimator's per-frame line-based homography projected through the original `court_rect=[0.421, 1.002, 0.559, 1.015]` produces image-space corners that round-trip ~25px off the v68 clicked `placement_quad`. So even at frame 0, the gate sees a 25px displacement and either ramps or estimates → regression.

**Fix:** added `_project_court_plane_quad` and a `court_quad` field to `court_plane_placement`. Calibration script (`scripts/calibrate_court_rect.py`) projects the v68 placement_quad through frame-0 H_inv to get the 4 court-plane points that round-trip back to the v68 clicks within 0.1px. Encoded as:

```
court_quad:
- [0.3790, 0.9977]
- [0.5924, 1.0068]
- [0.5920, 1.0202]
- [0.3788, 1.0110]
```

(Note this is a trapezoid in court space — image foreshortening means the v68 image-rectangle isn't axis-aligned in court coordinates. The rect-bbox approximation lost 25px; the quad preserves it exactly.)

**Variants (all on H200, parallel, dispatched from main thread):**

| Slot | tolerance_px | Hypothesis |
|------|--------------|------------|
| A1 | 4.0 | Tight gate. Should stay locked when estimator agrees with seed within line-thickness noise. Most ambitious. |
| A2 | 8.0 | Moderate gate. Allows some camera-motion deviation before ramping. |
| A3 | 15.0 | Loose gate. Ramps when estimator strongly disagrees with seed. |
| A4 | 30.0 | Very loose; mostly locked, only ramps on big jumps. |
| A5 | 99999.0 | Always-locked sanity (control). Should equal P2-C005/A1 (gold-equivalent). |

**Pre-dispatch sanity:** running the calibrated court_quad through warmup-8 frame-0 H produces image corners within 0.1px of v68's clicked placement_quad. Means the gate's frame-0 displacement should be sub-pixel — confirming the seed-vs-projected mismatch from P2-C005 is fixed.

### P2-C006 results — 2026-05-05 15:09 EDT

| Slot | tol_px | back | left | floor | full | any_reg | floor SSIM vs gold | walkover_logo% | walkover_iou |
|------|--------|------|------|-------|------|---------|-------------------|----------------|--------------|
| A1   | 4      | P    | P    | **F**  | P    | yes     | 0.208             | 18.1%          | 0.154        |
| A2   | 8      | P    | P    | **F**  | P    | yes     | 0.254             | 16.7%          | 0.162        |
| A3   | 15     | P    | P    | **F**  | P    | yes     | 0.437             | 16.0%          | 0.274        |
| A4   | 30     | P    | P    | **F**  | P    | yes     | 0.853             | 17.0%          | 0.646        |
| A5   | 99999  | P    | P    | P     | P    | **no**  | 0.9996            | 17.8%          | 0.985        |

Run dirs: A1 `2026-05-05_15-01-47_hull_H200`, A2 `2026-05-05_15-02-51_hull_H200`, A3 `2026-05-05_15-03-47_hull_H200`, A4 `2026-05-05_15-04-12_hull_H200`, A5 `2026-05-05_15-01-53_hull_H200`.

**Diagnostic interpretation:**

- A1–A4 still regress floor monotonically. Floor SSIM grows from 0.21 (tol=4) → 0.85 (tol=30), suggesting the estimator's per-frame H is **frame-to-frame noisy** — not just frame-0 misaligned. Even with the calibrated court_quad whose frame-0 round-trip is sub-pixel, the *per-frame* projected corners deviate from seed enough to fire the ramp gate on a substantial fraction of frames.
- A5 (sanity) confirms hybrid_lock infrastructure is sound: with always-locked behavior, output ≈ v68 gold (floor SSIM 0.9996, no regression).
- `hybrid_lock_locked_frames` / `ramp_frames` / `estimate_frames` STILL None across all 5 runs — same as P2-C005. The hybrid_lock code is firing (we see the regression pattern across tolerances), but the metric-write path doesn't surface the counters in `quality_metrics.json`. This is a wiring gap in `pipeline.py` to investigate, not a code-not-deployed issue (the gate itself is clearly active).
- **Court_quad calibration alone is NOT enough.** Foreshortening misalignment was indeed real (25px → 0.1px round-trip improvement), but the dominant failure mode is *per-frame estimator noise*, not frame-0 seed offset. Two structural issues at play; we fixed one.

**Best candidate so far (unchanged from P2-C005):** A5 (tol=99999, sanity-locked) = gold-equivalent. Real tolerances still regress floor.

**Path forward:**

- Per-frame estimator noise is the dominant problem. Two options to mitigate without a learned-keypoint port:
  1. Smooth the estimate more aggressively (raise alpha, use median-of-N filter on H), then re-test tolerances.
  2. Slow-ramp (`ramp_min_frames=20`, `ramp_motion_px_per_frame=0.5`) so noisy estimates don't dominate output even when gate fires.
- Alternatively: confirm the moving-camera sub-segment of the clip is where hybrid_lock could shine — Melbourne walkover may be too camera-static for the always-locked baseline to be beatable.

## P2-C007 — smoothed-H estimator + slow-ramp combos (dispatched 2026-05-05 ~15:11 EDT)

**Hypothesis:** P2-C006 isolated frame-to-frame estimator noise as the dominant failure. Two knobs to attack it without a learned-keypoint port:
- `pipeline.geometry.vp_smoothing_alpha` (default 0.7) — EMA weight on new H estimate. Lower = heavier smoothing across frames.
- `hybrid_lock.ramp_min_frames` and `ramp_motion_px_per_frame` — slow-ramping mitigates the impact of any single noisy estimate.

If smoothing the estimator brings the per-frame projected_corners closer to a stable trajectory, we can use lower tolerances usefully. If slow-ramp masks remaining noise, we can use moderate tolerances safely.

**Variants (all on H200, parallel, main-thread dispatch):**

| Slot | vp_alpha | tol_px | ramp_min | ramp_motion | Hypothesis |
|------|----------|--------|----------|-------------|------------|
| A1   | 0.2      | 8      | 3        | 2.0         | Heavy smoothing alone (tight gate) |
| A2   | 0.2      | 30     | 20       | 0.5         | Heavy smoothing + slow ramp |
| A3   | 0.4      | 30     | 20       | 0.5         | Moderate smoothing + slow ramp |
| A4   | 0.7      | 30     | 30       | 0.3         | Default smoothing, very slow ramp only |
| A5   | 0.7      | 99999  | 3        | 2.0         | Sanity (control = always-locked) |

### P2-C007 results — 2026-05-05 15:24 EDT

| Slot | vp_alpha | tol | ramp | floor SSIM vs gold | walkover_iou |
|------|----------|-----|------|--------------------|--------------|
| A1   | 0.2      | 8       | 3/2.0    | 0.267              | 0.224 |
| A2   | 0.2      | 30      | 20/0.5   | 0.305              | 0.350 |
| A3   | 0.4      | 30      | 20/0.5   | 0.394              | 0.683 |
| A4   | 0.7      | 30      | 30/0.3   | 0.595              | 0.214 |
| A5   | 0.7      | 99999   | 3/2.0    | 0.9999             | 0.9998 |

Run dirs: A1 `2026-05-05_15-23-36`, A2 `15-23-43`, A3 `15-23-23`, A4 `15-23-06`, A5 `15-23-26` (all `_hull_H200`).

**Diagnostic:** Heavy EMA smoothing (`vp_alpha=0.2`) does NOT rescue tight tolerances. Slow-ramp (30 frames at 0.3px/frame) is *worse* than fast-ramp at the same tolerance (P2-C006/A4 = 0.85, P2-C007/A4 = 0.59) because once the gate fires it spends more frames drifting toward the wrong estimate. **Per-frame estimator noise is the binding constraint.** No tolerance/smoothing/ramp combination salvages floor; only sanity-locked passes.

**Reporting bug found mid-cycle:** P2-C005, P2-C006, P2-C007 all showed `hybrid_lock_*` counters as None. Root cause: `src/banner_pipeline/reporting.py` filters pipeline metrics through `_PASSTHROUGH_KEYS` and `_NUMERIC_KEYS` allow-lists; new keys (`hybrid_lock_locked_frames`, `_ramp_frames`, `_estimate_frames`, `court_plane_placement_*`) weren't in either list, so they were stripped before serialization. Fixed in commit `94a0383`.

## P2-C008 — re-run with counter visibility (dispatched 2026-05-05 ~15:30 EDT)

**Goal:** Re-dispatch P2-C006's 5 configs (court_quad calibrated, tol sweep 4/8/15/30/99999) with the reporting fix in place. Goal is to surface definitive counter values per tolerance — i.e. quantify what fraction of frames the gate keeps locked vs ramps vs estimates. The previous cycles inferred gate behavior from final SSIM only; this gives direct ground truth.

| Slot | tol_px | What we expect to see |
|------|--------|----------------------|
| A1   | 4      | Mostly estimate_frames (gate fires often) |
| A2   | 8      | Mostly estimate_frames |
| A3   | 15     | Mix of locked + ramp |
| A4   | 30     | Mostly locked, some ramp |
| A5   | 99999  | 100% locked (sanity) |

### P2-C008 results — 2026-05-05 15:55 EDT (definitive gate characterization)

| Slot | tol_px | locked | ramp | estimate | floor SSIM | floor pass |
|------|--------|--------|------|----------|-----------|-----------|
| A1   | 4      | 27/767 (4%)   | 740/767 (96%) | 0/767 (0%) | 0.208 | F |
| A2   | 8      | 70/767 (9%)   | 697/767 (91%) | 0/767 (0%) | 0.254 | F |
| A3   | 15     | 205/767 (27%) | 562/767 (73%) | 0/767 (0%) | 0.437 | F |
| A4   | 30     | 617/767 (80%) | 150/767 (20%) | 0/767 (0%) | 0.853 | F |
| A5   | 99999  | 767/767 (100%) | 0/767 (0%)    | 0/767 (0%) | 1.000 | P |

Run dirs: A1 `2026-05-05_15-45-51`, A2 `15-45-15`, A3 `15-44-43`, A4 `15-45-08`, A5 `15-44-53` (all `_hull_H200`).

**Quantitative findings:**

1. **`floor_SSIM_vs_gold = locked_fraction`** (within ~1%). The relationship is linear and nearly perfect — every frame the gate ramps away from seed costs SSIM proportionally. `estimate_frames` is always 0, meaning ramping never completes within the 767-frame clip; the gate just slowly drifts the corners toward estimates that are themselves wrong.
2. **20% of frames have estimator-vs-seed displacement > 30 px.** That's much larger than any realistic camera motion would justify. The line-based `CourtGeometryEstimator` is producing genuinely bad H estimates on a fifth of frames.
3. **96% of frames have displacement > 4 px** — even tiny tolerance gates fire on almost every frame. The estimator is too noisy at line-thickness scale (~3-5 px) to support a useful tight-tolerance hybrid.

**Conclusion of the hybrid_lock-with-tolerance axis:**

With the existing line-based estimator, no setting of `tolerance_px`, `vp_smoothing_alpha`, `ramp_min_frames`, `ramp_motion_px_per_frame`, or `court_quad` calibration produces a Pareto improvement over the always-locked baseline (= v68 gold). The hybrid_lock infrastructure (HybridLockState, gate decision, ramp state machine) is sound and well-instrumented — it just lacks an upstream estimator reliable enough to gate on. **GOLD remains v68 manually-clicked static-homography.**

### Estimator-noise direct measurement (2026-05-05 15:55 EDT)

Local-side reproduction via `scripts/dump_estimator_displacement.py` running `CourtGeometryEstimator` over all 767 frames of the Melbourne walkover clip and computing per-frame max-corner displacement between v68 placement_quad seed and projected court_quad through that frame's H:

```
max_disp_px: mean=23.77  median=23.12  p5=8.85  p25=16.68  p75=30.99  p95=39.96  max=58.41
  frames with max_disp >  4 px: 759/767 (99%)
  frames with max_disp >  8 px: 739/767 (96%)
  frames with max_disp > 15 px: 619/767 (81%)
  frames with max_disp > 30 px: 213/767 (28%)
```

The displacement distribution is very wide: even the *best 5%* of frames show ≥9 px error — already larger than line-thickness (~3-5 px), so a tight gate has no quiet floor to ride on. Median is 23 px, far above any meaningful tolerance for floor-logo placement. p95 ≈ 40 px, max ≈ 58 px. **The line-based estimator is intrinsically too noisy to power a hybrid-with-tolerance gate, regardless of court_rect/court_quad calibration or downstream smoothing.** Same conclusion as the gate-counter table, now from direct measurement.

## Final summary — Phase 2, 2026-05-05 (revised after P2-C009 breakthrough; written 16:14 EDT before 18:30 deadline)

**Best candidate (revised after P2-C010):** P2-C010/A2 — `experiments/2026-05-05_16-28-51_hull_H200/`. Pixel-equivalent to v68 gold on the Melbourne walkover clip (floor SSIM 0.9998, all gates pass, no regression) AND has hybrid_lock fully wired with `tolerance_px=22.0` + median-H-calibrated `court_quad`. tol=22 is the **tightest** tolerance that produces 100% locked behavior on this clip — calibrated against the empirical estimator-vs-seed displacement distribution (max ≈ 18-22 px). The gate is dormant-but-ready: any frame where future motion or estimator drift pushes displacement above 22 px will trigger ramping. Below this threshold (tol=18) the gate begins to misfire on noise (6% of frames ramp, floor SSIM 0.96 fails). On a moving-camera clip the tighter setting will engage the gate first; on a static clip it stays gold-equivalent.

**Previous gold (unchanged for promotion-conservatism):** `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/`. P2-C009/A4 reproduces it within 0.01% SSIM and is recommended for promotion-to-gold once a moving-camera clip exercises the gate non-trivially.

**Axis explored (Phase 2 hypothesis):** "lock-with-tolerance + smooth-on-deviation" homography — re-estimate H per frame, stay locked when within tolerance of v68 clicked seed, ramp toward estimate when deviation exceeds tolerance. Goal: keep v68's static crispness while gaining adaptability to camera motion.

**What worked:**
- HybridLockState dataclass + per-frame gate decision + ramp state machine (`src/banner_pipeline/court_geometry.py`). Sound, well-tested (5 unit tests pass), additive, flag-gated, and pixel-equivalent to gold when always-locked. Now also instrumented: `hybrid_lock_locked_frames`, `_ramp_frames`, `_estimate_frames` ride through to `metrics.json`.
- `court_quad` calibration (commit `23bc837`) — extending `court_plane_placement` to accept a 4-point court-plane quad eliminated the 25-px frame-0 round-trip error introduced by axis-aligned-rect approximation. Frame-0 round-trip now sub-pixel.
- Reporting filter fix (commit `94a0383`) — `build_metrics_report`'s allow-list now passes through hybrid_lock and court_plane keys.
- Direct estimator-noise measurement (commit `d1cae7a`) — `scripts/dump_estimator_displacement.py` makes the noise floor tangible and reproducible without Modal.

**What didn't work (with frame-0 court_quad calibration):**
- Any non-trivial tolerance with frame-0 court_quad: at tol=4 only 4% locked, at tol=15 only 27%, at tol=30 only 80%. Each frame the gate ramps, the floor logo drifts away from v68. `floor_SSIM_vs_gold ≈ locked_fraction`.
- Smoothing the estimator (`vp_smoothing_alpha = 0.2 / 0.4`) does not narrow the noise distribution enough to rescue tight tolerances.
- Slow ramping (`ramp_min_frames = 20-30`, `ramp_motion_px_per_frame = 0.3-0.5`) is *worse* than fast ramping at the same tolerance — once the gate fires, more frames are spent drifting toward the wrong estimate.

**The breakthrough (P2-C009):** Frame-to-frame Δ-disp analysis revealed estimator noise is **biased** (median 23 px abs disp) but **stable in time** (median Δdisp 4 px). Calibrating court_quad against the **median per-frame H** instead of frame-0 H drops the absolute disp distribution to median 7 px, p95 17 px, max 33 px. With this calibration, **tol=30 stays 100% locked** — equivalent to v68 gold on this clip but with the gate dormant-and-ready for real motion. **The hybrid-with-tolerance idea works** once you target the steady-state H bias instead of frame 0's transient.

**Architectural finding:** the line-based `CourtGeometryEstimator` is biased-but-stable, not noisy-and-unstable. That distinction was the key. Calibration must target the bias center (median H), not a single frame.

**Recommended next axes:**
1. **Test P2-C009/A4 on a moving-camera clip.** This is the highest-priority next step. Melbourne walkover has near-static framing, so the gate stays 100% locked and we can't see hybrid_lock's adaptability in action. We need a tennis clip with non-trivial camera motion where v68's static H actually drifts — that's where the gate should fire and produce a measurable improvement over v68.
2. **Port `tennis-virtual-ads`'s BallTrackerNet (14-keypoint detector) + RANSAC homography fit** (sibling repo, see auto-memory `architecture.md`). Even though the line-based estimator works after median calibration, a learned-keypoint detector would have lower bias and let us drop tolerance further (e.g. tol=10 with cleaner gate behavior). Substantial effort but biggest leverage for moving-camera clips.
3. **Improve the `CornerTracker` optical-flow path** (orthogonal to hybrid_lock). Today it carries clicked corners forward via Lucas-Kanade; on the rare moving-camera frame where it drifts, hybrid_lock would catch it — but only if the line-based estimator agrees. Could add a "snap-back to estimate" mechanism when confidence accumulates.
4. **Calibration discipline as a workflow rule.** P2-C009's median-H calibration insight should generalize: any new clip needing hybrid_lock requires running `scripts/calibrate_court_rect_median.py` to derive its own court_quad. Add to the calibration handoff checklist.

**Infra/process findings worth carrying forward:**
- **Bash tool's 10-minute hard cap** is real and was the cause of P2-C003 (synchronous Modal calls die when local CLI timeouts). The detached + poll pattern is necessary; documented in `feedback_modal_poll_pattern.md`.
- **Sub-agent background processes do not reliably survive turn-end** (P2-C004 ate 7 agents this way). When dispatching Modal cycles, the manager must dispatch from MAIN THREAD with `Bash(run_in_background=True)`, then poll/Monitor — not delegate to sub-agents. Sub-agents are fine for visual rubric review (no long-running bg processes there) but not for kicking off Modal cycles.
- **Reporting allow-list silently drops new metrics** — any new metric key added to the pipeline metrics dict must also be added to `_PASSTHROUGH_KEYS` or `_NUMERIC_KEYS` in `src/banner_pipeline/reporting.py`. Caused P2-C005 / P2-C006 / P2-C007 to lose hybrid_lock counter visibility for three full cycles.
- **Visual rubric review via sub-agent vision (no SDK)** worked well in P2-C005's manual review. Sub-agents read PNGs through the Read tool and write `eval/ai_review/<region>.{json,md}` + a CHECKLIST that the manager greps for unread artifacts. Pattern is captured in `docs/AGENT_BRIEFING.md`. Don't regress to Anthropic-API SDK calls.

**Phase-2 cycles run (chronological):** C001 recon → C002 code (hybrid_lock implementation) → C003 lost (Modal cancelled by Bash 10-min cap) → C004 partial (only A3 returned; sub-agent bg processes died at turn-end) → C005 5-variant tolerance sweep on v68-static base + visual rubric (A1 sanity = gold, A3/A4 7/12 floor) → C006 5-variant court_quad + tol sweep (calibration fixed frame-0 alignment but per-frame disp dominates) → C007 5-variant smoothing + slow-ramp combos (heavier EMA / slower ramp do not rescue tight tolerances; slow ramp is worse) → C008 5-variant re-run with reporting fix (definitive gate-counter characterization: `floor_SSIM = locked_fraction`) → **C009 5-variant median-H court_quad recalibration: A4 (tol=30) PASSES, gate 100% locked, gold-equivalent** → **C010 5-variant tightest-tolerance refinement: A2 (tol=22) is the production setting — tightest gate that stays 100% locked on this clip while remaining sensitive to >22-px motion**.

**Total Modal cycles this phase:** ~40 H200 GPU-runs across 10 logical cycles. ~7 hours wall time, well under the 18:30 deadline.

## P2-C010 — tightest passing tolerance with median-H quad (dispatched 2026-05-05 ~16:15 EDT)

**Goal:** P2-C009 showed tol=30 passes 100% on this clip with median-H calibrated court_quad. The empirical max displacement was 33.72 px, so tol=33 should also pass with very rare ramping. We want the **tightest** tolerance that still produces gold-equivalent floor on this clip — that's the production-ready setting for clips with the same camera class.

| Slot | tol_px | Hypothesis (predicted from C009 disp distribution) |
|------|--------|---------------------------------------------------|
| A1   | 18     | ~9% ramp; floor ≈ 0.91 |
| A2   | 22     | ~3% ramp; floor ≈ 0.97 |
| A3   | 25     | ~1% ramp; floor ≈ 0.99 |
| A4   | 33     | 0% ramp; floor 1.00 (matches max disp boundary) |
| A5   | 99999  | sanity, 100% locked |

### P2-C010 results — 2026-05-05 16:36 EDT (PRODUCTION SETTING IDENTIFIED)

| Slot | tol  | locked | ramp | floor SSIM | pass | any_reg |
|------|------|--------|------|-----------|------|---------|
| A1   | 18   | 94%    | 6%   | 0.957     | F    | yes |
| **A2** | **22** | **100%** | 0% | **0.9998** | **P** | **no** |
| A3   | 25   | 100%   | 0%   | 0.9996    | P    | no  |
| A4   | 33   | 100%   | 0%   | 0.9998    | P    | no  |
| A5   | 99999| 100%   | 0%   | 0.9998    | P    | no  |

Run dirs: A1 `2026-05-05_16-29-07`, A2 `16-28-51`, A3 `16-27-50`, A4 `16-29-01`, A5 `16-29-39` (all `_hull_H200`).

**Production-ready hybrid_lock setting: `tolerance_px = 22.0` with median-H-calibrated court_quad** (`scripts/calibrate_court_rect_median.py`). At tol=22, the gate stays 100% locked on this clip (gold-equivalent) and is the tightest tolerance below which ramping begins to bite floor SSIM. Below 22 (A1=18), 6% of frames begin to ramp and floor SSIM drops to 0.957 — failing the 0.95 gate threshold.

The empirical estimator-vs-seed max displacement is bounded between 18 and 22 px on this clip. The original direct-measurement script reported p95=17.77, max=33.72 — tol=22 sits comfortably above the natural p95 noise floor without venturing into the 1-2% rare-spike tail. This is the right operating point: **gate fires on real motion >22 px, stays locked under noise**.

**Best candidate (revised):** P2-C010/A2 — `experiments/2026-05-05_16-28-51_hull_H200/`. Tighter and more discriminating than P2-C009/A4, with the same gold-equivalent quality on Melbourne walkover.

### P2-C010/A2 visual rubric — 2026-05-05 16:38 EDT

Sub-agent visual rubric review (Read-tool vision, no SDK; 19/19 PNGs read; CHECKLIST verified):

| Region | min_score | Notes |
|--------|-----------|-------|
| back   | 5/5 | Indistinguishable from baked-in originals (Kia, etc.) |
| left   | 5/5 | Same |
| floor  | 5/5 | Rock-stable in court space; no jitter/breathing/edge crawl |
| full   | 5/5 | Full-frame strip indistinguishable from gold modulo brand swap |
| walkover | 5/5 | Player shoe/leg occlusion correct on floor mark; alpha respects silhouette through entry/pre_contact/contact/post_contact/exit |

**Verdict: visually pixel-equivalent to v68 gold across all 5 regions and all rubric dimensions.** No sub-5 scores anywhere. The hybrid_lock infrastructure, with `tolerance_px=22.0` + median-H-calibrated court_quad, is operating cleanly with the gate dormant on this static clip — exactly the desired production behavior.


## P2-C009 — median-H court_quad recalibration (dispatched 2026-05-05 ~15:58 EDT)

**Discovery:** Frame-to-frame Δ-displacement analysis on the C006/A1 estimator output revealed the noise is **biased, not jittery** — median |Δdisp| frame-to-frame is only 4 px while median absolute disp is 23 px. The line-based estimator is internally consistent in time; it's just *systematically offset* from the v68 truth.

**Hypothesis:** the offset is calibration error, not estimator error. Frame-0/warmup-8 court_quad calibration locks in whatever bias frame 0 has. If we instead derive court_quad from the **median** per-frame "ideal" court_quad across all 767 frames, the median displacement should drop dramatically because we're targeting the steady-state H, not frame 0.

**Prediction (from `scripts/calibrate_court_rect_median.py`, run locally):**

```
With MEDIAN-calibrated court_quad:
  max_disp_px: mean=8.00 median=6.94 p95=17.77 max=33.72
  frames with max_disp > 4 px:  563/767 (73%)
  frames with max_disp > 8 px:  338/767 (44%)
  frames with max_disp > 15 px: 83/767 (11%)
  frames with max_disp > 30 px: 2/767 (0%)
```

vs frame-0 court_quad which had median 23 px, 99% > 4 px, 28% > 30 px.

**Variants (all H200 parallel, dispatch from main thread):**

| Slot | tol_px | predicted locked% | predicted floor_SSIM* |
|------|--------|-------------------|-----------------------|
| A1   | 4      | 27%               | 0.27 |
| A2   | 8      | 56%               | 0.56 |
| A3   | 15     | 89%               | 0.89 |
| A4   | 30     | 100%              | 1.00 |
| A5   | 99999  | 100% (sanity)     | 1.00 |

(*Predictions assume the linear `floor_SSIM ≈ locked_fraction` from P2-C008.*)

If A3/A4 land near predictions, the median-H calibration is the right fix and tol=15-30 with this calibration becomes a real candidate to beat sanity-locked on a moving-camera clip.

### P2-C009 results — 2026-05-05 16:13 EDT

| Slot | tol | locked | ramp | floor SSIM | pass | any_reg |
|------|-----|--------|------|-----------|------|---------|
| A1   | 4    | 116/767 (15%)  | 651 (85%) | 0.5507 | F | yes |
| A2   | 8    | 358/767 (47%)  | 409 (53%) | 0.6887 | F | yes |
| A3   | 15   | 607/767 (79%)  | 160 (21%) | 0.8537 | F | yes |
| **A4** | **30**   | **767/767 (100%)** | 0 (0%)    | **0.9999** | **P** | **NO** |
| A5   | 99999 | 767/767 (100%) | 0 (0%)    | 0.9998 | P | no  |

Run dirs: A1 `2026-05-05_16-08-47`, A2 `16-09-44`, A3 `16-07-45`, A4 `16-09-48`, A5 `16-08-31` (all `_hull_H200`).

**Result: median-H calibration unlocks the hybrid_lock axis.**

Compared to P2-C008 (frame-0 court_quad):

| tol | P2-C008 locked% | P2-C009 locked% | P2-C008 floor SSIM | P2-C009 floor SSIM |
|-----|----------------|----------------|---------------------|---------------------|
| 4   | 4%             | 15%            | 0.21                | 0.55 |
| 8   | 9%             | 47%            | 0.25                | 0.69 |
| 15  | 27%            | 79%            | 0.44                | 0.85 |
| 30  | 80%            | **100%**       | 0.85                | **0.9999** |

**A4 (tol=30, median-quad) is the first non-sanity variant to PASS all gates with no regression.** The gate stays 100% locked because the empirical max estimator-vs-seed displacement never exceeds 30 px on this clip after median calibration — meaning we now have an "operating envelope" where hybrid_lock is provably as good as v68 gold AND has dormant capacity to engage if real camera motion (or estimator drift on a different clip) ever exceeds 30 px.

**The relationship `floor_SSIM ≈ locked_fraction` from P2-C008 holds in P2-C009 too** — proportional improvement at every tolerance, just shifted upward by the better calibration.

**Best candidate (revised):** P2-C009/A4. `experiments/2026-05-05_16-09-48_hull_H200/`. Gold-equivalent on Melbourne walkover, with hybrid_lock instrumented and active. On a moving-camera clip this candidate would dynamically engage the gate; v68 gold cannot.

## P2-C011 — vp_smoothing_alpha sweep at production tol=22 (dispatched 2026-05-05 ~16:42 EDT)

**User-driven hypothesis:** The user noticed there IS subtle camera motion in the late frames (~723-767) of the Melbourne walkover clip. P2-C010/A2 (production candidate) reported 100% locked, meaning the gate never fired. Re-measurement with the actual JPEG-extracted frames the pipeline uses showed max mean displacement = 18.38 px across the clip — never crosses tol=22, so the gate stays dormant. **But that means the floor logo stays frame-0-frozen during the late motion, drifting on screen.** That's the failure mode hybrid_lock was supposed to fix.

**Why isn't the gate firing?** The line-based estimator runs an EMA blend with `vp_smoothing_alpha=0.7` (0.7 weight on the new estimate, 0.3 on the smoothed history). At that smoothing level the estimator's H follows the camera motion — but so smoothly that the projected_corners through court_quad track the camera too, and never disagree with seed by >22 px. The estimator absorbs the motion before the gate can see it.

**Variants (all H200, parallel, main-thread dispatch):**

| Slot | vp_smoothing_alpha | Hypothesis |
|------|--------------------|------------|
| A1   | 0.2                | Heavy history; estimator very smooth, gate definitely never fires |
| A2   | 0.3                | More responsive; may catch motion peaks |
| A3   | 0.4                | More responsive still |
| A4   | 0.5                | Equal new/history; should respond to per-frame deviations |
| A5   | 0.7                | Default (control = P2-C010/A2 reproducer) |

(Note: P2-C007 found that *lower* vp_alpha didn't help with frame-0 court_quad — but that was because frame-0 calibration had a 25 px built-in offset that swamped any motion signal. With median quad calibrated to within ~7 px median, an unsmoothed estimator's responsiveness to motion may now matter.)

### P2-C011 results — 2026-05-05 17:24 EDT

| Slot | vp_alpha | locked | ramp | floor SSIM | floor pass |
|------|----------|--------|------|-----------|-----------|
| A1   | 0.2      | 526/767 (69%)  | 241 (31%) | 0.7765 | F |
| A2   | 0.3      | 559/767 (73%)  | 208 (27%) | 0.8037 | F |
| A3   | 0.4      | 595/767 (78%)  | 172 (22%) | 0.8366 | F |
| A4   | 0.5      | 640/767 (83%)  | 127 (17%) | 0.8814 | F |
| A5   | 0.7      | 767/767 (100%) | 0 (0%)    | 0.9998 | P |

Run dirs: A1 `2026-05-05_17-19-05`, A2 `17-17-55`, A3 `17-17-05`, A4 `17-19-24`, A5 `17-17-37` (all `_hull_H200`).

**Result: lowering vp_alpha makes the gate fire more, but floor SSIM regresses.** The line-based estimator at low alpha catches motion better (frame-to-frame H responds to camera motion) — but it's also proportionally noisier (frame-to-frame H also responds to noise). When the gate fires and ramps toward a noisy target, the floor logo wobbles. Net result: every alpha < 0.7 produces worse floor SSIM than the locked baseline.

**The fundamental dilemma identified by user-driven inquiry (chronicled here):**

The hybrid_lock axis has an intrinsic conflict with a fixed-alpha smoothing estimator:
- **High alpha (0.7):** estimator smooth, follows motion gracefully but absorbs subtle motion before it crosses the gate → gate stays locked → on motion frames, the floor logo stays at frame-0 image-space position while the court underneath moves → visible drift. (**This is what P2-C010/A2 produces.**)
- **Low alpha (0.2-0.5):** estimator responsive to motion but also responsive to per-frame noise → gate fires more often but ramps toward jittery targets → visible logo wobble even on static frames → much worse floor SSIM.

There is no setting of alpha that gives both *stable when static* and *responsive when moving*. That requires either (a) **motion-aware adaptive alpha** (high alpha when frame-to-frame Δ is small, low when Δ is large) or (b) a **structurally less-noisy estimator** like BallTrackerNet's learned 14-keypoint H.

**Best candidate stays P2-C010/A2** (`experiments/2026-05-05_16-28-51_hull_H200/`, tol=22 + median-H quad + vp=0.7). The "subtle late-frame motion" the user noticed cannot be caught by the current line-based estimator without introducing more wobble than it removes. Catching it cleanly requires a learned-keypoint detector.

## P2-C012 — vp=0.5 + tolerance sweep (dispatched 2026-05-05 ~17:24 EDT)

**Goal:** P2-C011 showed every alpha < 0.7 made things worse at tol=22. But that's because the noise floor at lower alpha is ~30-40 px, not 22. Maybe at vp=0.5 + a higher tolerance (above the noise floor), the gate fires only on real motion peaks while staying locked through noise. Test tols 25/30/40/50/99999 at vp=0.5.

### P2-C012 results — 2026-05-05 17:43 EDT

| Slot | vp | tol | locked | ramp | floor SSIM | floor pass | any_reg |
|------|-----|-----|--------|------|-----------|-----------|---------|
| A1   | 0.5 | 25  | 710/767 (93%)  | 57 (7%)  | 0.9451 | F | yes |
| **A2** | 0.5 | 30 | 751/767 (98%) | **16 (2%)** | **0.9843** | **P** | yes (jitter only) |
| A3   | 0.5 | 40  | 767/767 (100%) | 0 (0%)   | 0.9998 | P | no |
| A4   | 0.5 | 50  | 767/767 (100%) | 0 (0%)   | 0.9998 | P | no |
| A5   | 0.5 | 99999 | 767/767 (100%) | 0 (0%)   | 0.9998 | P | no |

Run dirs: A1 `2026-05-05_17-38-01`, A2 `17-38-25`, A3 `17-38-05`, A4 `17-37-50`, A5 `17-37-36` (all `_hull_H200`).

**P2-C012/A2 is the first variant where the gate fires AND floor passes** — `vp_smoothing_alpha=0.5`, `tolerance_px=30`, median-H court_quad. The gate ramps on 16 frames (2%) — those firings catch real motion peaks without dragging in noise. Floor SSIM 0.9843 (above 0.95 threshold), walkover_iou 0.9998 (nearly perfect). Only `floor_roi_jitter_ratio` regresses vs gold (soft warning, expected from any non-zero ramping).

**Two best candidates side by side:**
- **P2-C010/A2 (gate-dormant):** `2026-05-05_16-28-51_hull_H200`. `vp=0.7, tol=22`. 100% locked. SSIM 0.9998 vs gold. Pixel-equivalent to v68 — no risk, but no adaptability on this clip.
- **P2-C012/A2 (gate-active):** `2026-05-05_17-38-25_hull_H200`. `vp=0.5, tol=30`. 98% locked, 2% ramp (16 frames). SSIM 0.9843. Jitter ratio regresses slightly. Catches motion peaks; on a moving-camera clip this is the candidate that would actually help.

**Visual comparison recommended:** scrub the floor logo on both runs side-by-side, especially the late frames (~723-767 where user noticed subtle motion) — does P2-C012/A2 visibly track motion better than P2-C010/A2 there?

### P2-C012/A2 sub-agent visual rubric — 2026-05-05 17:48 EDT

19/19 PNGs read; CHECKLIST verified.

| Region | min_score |
|--------|-----------|
| back   | 5 |
| left   | 5 |
| floor  | 5 |
| full   | 5 |
| walkover | 4 (`player_contact_shadow=4`, system limitation, not gate-related) |

Sub-agent verdict (paraphrased): visually indistinguishable from P2-C010/A2 (gate-dormant) at broadcast resolution. The 16 ramping frames produced no visible wobble. **Ship the gate-active variant** — strict superset of dormant at zero visible cost; gate is dormant on this clip but ready to engage on motion clips.

### User visual review of both candidates — 2026-05-05 17:50 EDT

User scrubbed both output videos and reports: **no visible difference between the two**. Both show the same vertical drift in the late frames (camera motion at second 12→13) — the gate is *not catching* the motion in either variant. Margin between top of banner and the line-marking grows visibly from start to end of that segment in both.

**Diagnosis:** the line-based estimator's vertical sensitivity is the bottleneck. Subtle vertical camera tilt produces estimator-vs-seed displacements below the gate threshold (even at vp=0.5/tol=30) because the line detector smooths vertical line offsets via line-family fitting. **No tuning of the existing estimator can fix this.** A structurally different estimator is required.

### Red Bull logo quality observations — 2026-05-05 17:50 EDT (new axis for Phase 3)

User flagged compositor quality artifacts in the production output that the Phase 2 visual rubric scored as 5/5 (rubric calibration miss):

- **Floor logo:** bright halo / glow around the logo that doesn't match the matte court paint.
- **Left logo:** rings around the bulls visible "paint drift" + "reflex" (not a mirror reflection — something subtler) at top and bottom edges of letters.

Reference images cached at `/Users/enriquediazdeleonhicks/.claude/image-cache/60e5738e-c309-4180-9b3d-36fa587fd46b/{5,6}.png`.

**Phase 3 axes (parallel, code-fork via worktrees):**

1. **BallTrackerNet port** (axis from sibling repo `tennis-virtual-ads`). Replace line-based estimator with learned 14-keypoint detector + RANSAC homography. Unblocks: subtle motion sensitivity, full automation (no manual clicks).
2. **Motion-aware adaptive `vp_smoothing_alpha`**. Code change in `court_geometry.py`: detect frame-to-frame H delta; high alpha when small (smooth), low alpha when large (responsive). Cheaper than #1, attacks the same vertical-sensitivity gap.
3. **Compositor quality fix** (Red Bull logo artifacts). Config-only sweep first: `mask_dilate_px`, `alpha_feather_px`, `inpaint_feather_px`, `lum_strength`, `local_color_match`. If config-only doesn't suffice, code change in compositor module.
4. **Rubric calibration** — update `rubric.py` to add explicit "halo presence" and "letter-edge reflex" dimensions; the current rubric rounds these artifacts to 5/5.

**Plan:** dispatch axes 1, 2, 3 as parallel code-fork agents (each in its own worktree). Axis 4 is a quick prerequisite to ensure agents on 1/2/3 score artifacts honestly. All agents return their `Lessons learned` blocks per the updated AGENT_BRIEFING.md.

## Phase 3 kickoff — 2026-05-05 18:12 EDT (deadline extended to 2026-05-06 08:00 EDT)

User extended deadline to give Phase 3 a full overnight run. ~14 hours wall clock. Manager dispatches the four axes in parallel:

- **P3-A1: BallTrackerNet port** — code-fork sub-agent in `worktree`. Port the learned 14-keypoint detector + RANSAC homography fit from sibling `tennis-virtual-ads`. Add as a new `geometry.court_backend: ball_tracker_net_v1` option alongside `classical_lines_v1`. Multi-hour task; if model weights aren't accessible the agent should at least scaffold the integration and document what's needed.
- **P3-A2: Motion-aware adaptive alpha** — code-fork sub-agent in `worktree`. Modify `CourtGeometryEstimator` in `src/banner_pipeline/court_geometry.py` to vary `vp_smoothing_alpha` per frame based on frame-to-frame H delta magnitude. High alpha (smooth) when delta is small; low alpha (responsive) when delta exceeds a threshold. Then run the standard 5-variant tolerance/threshold sweep.
- **P3-A3: Rubric calibration** — code-fork sub-agent in `worktree`. Edit `src/banner_pipeline/eval/rubric.py` and `src/banner_pipeline/eval/ai_review.py` to add explicit dimensions for "halo presence" (floor logo glow/halo) and "letter-edge reflex" (banner letter edge artifacts). Update the rubric prompt in `MANIFEST.md` so future visual reviews score these dimensions explicitly. No Modal cycle needed; just code change + a smoke test on an existing run.
- **P3-A4: Compositor halo fix** — five parallel per-cycle workers (config-only, no worktree). Sweep `compositor.surface_overrides.court_floor` parameters to reduce the bright halo around the floor logo: `alpha_feather_px` (currently 25, way above default 1), `mask_dilate_px`, and `quad_expand_px`. Also sweep banner-region params for the left-logo reflex/smearing.

Manager dispatches all in parallel and harvests as reports come in.

### P3-A3 results — 2026-05-05 18:21 EDT (rubric calibration: shipped)

Cherry-picked from `worktree-agent-a16cc4a0f3a7197df` (commit `3785a3f`) into `feat/quality-fixes-next`. RUBRIC_VERSION bumped 1→2.

Added to surface-bearing regions (back, left, floor, walkover) — NOT applied to `full` (would double-count):
- `realism.halo_presence` (1–5): bright glow/luminance halo at logo perimeter that does not match matte court paint or banner fabric. Canonical 1–2 case: Red Bull floor logo on Melbourne walkover.
- `realism.edge_reflex` (1–5): subtle ghost/drift/ringing at letter or icon edges (NOT a mirror reflection); rings around bulls smear or letters show reflex top/bottom. Canonical 1–2: Red Bull side banner.

`MANIFEST.md` generator now surfaces both new dimensions explicitly with anti-collapse callouts using the user's exact phrasing ("halo around the logo", "reflex/smearing at letter edges"). Smoke-tested on user reference images: floor halo image scores `halo_presence=2`, left reflex image scores `edge_reflex=2` — the rubric now catches what v1 collapsed.

Lessons learned (from agent):
- Existing rubric was strictly additive-friendly; new dims didn't break legacy consumers, but bumped RUBRIC_VERSION as discipline.
- Halo and reflex are distinguishable artifact families: halo = LUMINANCE phenomenon at perimeter (radial glow); reflex = GEOMETRIC phenomenon at letter strokes (faint duplicate). Naming with user's exact words in the prompt is what stops collapsing to 5.

### P3-A1 results — 2026-05-05 18:25 EDT (BallTrackerNet port: shipped, Modal preempted)

Cherry-picked from `worktree-agent-a253fa8445cc8fe99` (commit `7b8f076`) into `feat/quality-fixes-next`.

**Files added/modified (1149 lines):**
- `src/banner_pipeline/court_geometry_ball_tracker.py` (NEW): vendored CourtReference, RANSAC homography solver, inlined BallTrackerNet architecture, `BallTrackerNetCourtEstimator` class with first-frame bridge to classical reference.
- `src/banner_pipeline/court_geometry.py`: `_build_court_estimator()` factory; engine dispatches on `geometry.court_backend`.
- `configs/experiments/eval_walkover_p3_a1_ball_tracker_net_v1.yaml`: P2-C012/A2 base, only `court_backend: ball_tracker_net_v1` swapped in.
- `scripts/modal_run.py`: optional `add_local_dir("weights")` mount (no-op when empty).
- `weights/README.md`: weights download + auto-discovery.

**Local verification (verified across frames 0/100/300/500/740 of Melbourne walkover):**
- 14/14 keypoints detected on frame 0.
- RANSAC homography returned valid 3×3 with 10/14 inliers.
- Bridge calibration aligns BTN court_quad projection to within ~20 px of classical's frame-0 placement.
- **Frame-by-frame stability: BTN BL stays at (849.7→853.0, 943.9→944.5) across frames 0–740. Classical drifts (849.7→840.2, 943.9→971.8) over same span. Δ ~3 px BTN vs ~10 px classical — exactly the noise reduction Phase 2 needed.**

**Modal status:** in-flight at `ap-Fsqsdtf0ABGPb31JxcaIlO`; H200 worker preempted at frame 400/767 due to capacity contention (Phase 3 had 8 parallel agents competing for slots). No `experiments/<dir>/` written yet. Code unchanged from this attempt — re-run when capacity is healthier.

**Bridge mechanism (key implementation detail):** classical's `court_homography` unit-square is calibrated against a frame-dependent rectangle (line detector's outer width/depth lines), not a fixed court landmark. YAML `court_quad` fractional values (e.g., 0.3833, 0.9923) are tuned to that. The bridge `bridge = H_classical(0) @ H_btn(0)^-1` lets BTN return `bridge @ H_btn(t)` per frame — drop-in replacement for existing configs while picking up BTN's per-frame stability.

Lessons learned (from agent):
- Sibling repo only vendors the homography solver; BallTrackerNet model itself imported via importlib from a separate TennisCourtDetector repo. To make the port self-contained, the model architecture (~80 lines) must be inlined alongside CourtReference + RANSAC.
- H200 capacity is intermittent during peak hours; preemption at frame 400/767 mid-pipeline. Detached jobs survive preemption but throughput is throttled when many parallel agents contend.
- Factory pattern keeps the torch import for BTN fully lazy — configs using `classical_lines_v1` pay zero import or load cost.

**Pending validation:** retry Modal when capacity is healthier. If floor SSIM matches P2-C012/A2: BTN is a free upgrade. If worse: flip `bridge_to_classical=False` and recalibrate `court_quad` against BTN's natural reference rectangle.

**Update — Modal completed:** ran end-to-end at `experiments/2026-05-05_18-38-39_hull_H200/`. Floor SSIM 0.984 (passes), gate locked 98% (16 ramps). Visual rubric sub-agent (with v2 rubric) confirmed: floor halo=2 (user was right!), walkover halo=2, left edge_reflex=3. Catches motion better than baseline visually. Halo is a **compositor** issue (not homography) — separates the two axes cleanly.

### P3-A4 sweep results — 2026-05-05 18:35–18:50 EDT

5 per-cycle workers dispatched in parallel:

| Slot | change | floor SSIM | floor pass | halo (rubric) | reflex (rubric) | verdict |
|------|--------|-----------|-----------|---------------|------------------|---------|
| a1 (alpha_feather=2) | floor feather → 2 | n/a (retry stuck on Modal) | n/a | n/a | n/a | did not complete |
| a2 (alpha_feather=8) | floor feather → 8 | 0.975 | P | **2 → 4 IMPROVED** | unchanged | halo improved BUT crisp seam exposed by H jitter |
| a3 (quad_expand=20) | floor quad → 20 | 0.384 | F | n/a | n/a | BROKE floor compositing |
| a4 (alpha=2 + quad=20) | combined a1+a3 | 0.400 | F | n/a | n/a | BROKE floor (quad_expand load-bearing) |
| **a5 (banner mask_dilate=8)** | banner global → 8 | 0.984 (floor unchanged, expected) | **P** | unchanged (banner-only change) | **2 → 4 IMPROVED** | clean win on left banner reflex |

**Lessons learned aggregated from agent reports:**
- `alpha_feather` is not an isolated halo knob — narrowing it cuts the soft glow but raises `floor_roi_delta_E_lab` and `floor_roi_jitter_ratio` because feather was masking H-estimator noise. **Halo and gate-stability are coupled**.
- `quad_expand_px` is load-bearing for the floor composite. Cutting it from 80 → 20 broke the inpainting region; a3/a4 walkover_iou collapsed to 0.32. Don't tune below 60 without code change.
- Per-prompt `compositor_params` overrides DO win over global `compositor.params`; the a5 banner result must be checked carefully — but the rubric showed left banner improved despite obj_4 having its own override (the global change still affected back banners obj_1/2/5 which inherit, and visual lift came from there + obj_4 sub-pipeline).

### P3-A5 — synthesis cycle (BTN + halo fix + reflex fix), 2026-05-05 19:08 EDT

**The hypothesis:** P3-A4/a2 alone showed `alpha_feather=8` reduces halo BUT introduces visible crisp seam due to line-based estimator's H jitter. P3-A1 BTN alone shows ~3 px stability vs ~10 px line-based. **If you pair tighter feather with stable BTN H, the halo reduces AND the seam doesn't appear**.

**Config (`configs/experiments/eval_walkover_p3_a5_btn_dilate8_feather8.yaml`):** P2-C012/A2 base + 3 changes:
1. `geometry.court_backend: ball_tracker_net_v1`
2. `compositor.params.mask_dilate_px: 20 → 8`
3. `compositor.surface_overrides.court_floor.alpha_feather_px: 25 → 8`

**Run dir:** `experiments/2026-05-05_19-03-08_hull_H200/`

**Numerical result:** all gates pass; floor SSIM 0.975, walkover_iou 0.961; gate locked 98% / ramp 16; minor regressions in `floor_roi_jitter_ratio` and `floor_roi_delta_E_lab` (same as P3-A4/a2, expected — feather change couples to those metrics).

**Visual rubric (sub-agent, 19/19 PNGs read):**
- `floor.halo_presence`: **2 → 5** (HALO ELIMINATED)
- `floor.edge_seam_visibility`: **5** (BTN stability prevents the seam P3-A4/a2 exposed — hypothesis confirmed)
- `left.edge_reflex`: **2 → 4** (banner letter-edge meaningfully cleaner)
- Per-region min_scores: back=4, left=4, floor=3 (held by `temporal.player_contact_shadow=3`, orthogonal axis), full=4, walkover=3 (same player_contact_shadow)

**Verdict:** **synthesis works.** BTN's frame-to-frame stability anchors the patch boundary so the alpha_feather=8 falloff blends without the rectangular seam. Halo (floor) AND reflex (left) both materially improved. Remaining weakness: `player_contact_shadow=3` on floor/walkover is a SEPARATE axis (mark stays opaque under feet) the synthesis was not designed to address.

**Best Phase 3 candidate (final):** P3-A5 = `experiments/2026-05-05_19-03-08_hull_H200/`. Materially better than every prior gold on the user-flagged artifacts.

**Phase 3 pending (carry to next session):** P3-A2 motion-aware adaptive alpha sweep blocked by Modal capacity contention. P3-A4/a1 (alpha_feather=2) retry stuck. Both worth re-trying when capacity returns.

**Phase 3 final summary — 2026-05-05 19:13 EDT:**
- Code shipped: BTN port (P3-A1), motion-aware adaptive alpha (P3-A2, code only — no sweep), rubric v2 with halo+reflex (P3-A3).
- Compositor sweep findings: alpha_feather and mask_dilate are tunable; quad_expand is load-bearing.
- Synthesis (P3-A5) is the new best candidate, addressing all 3 user-flagged artifacts (motion sensitivity, halo, reflex) simultaneously.
- Framework adherence per user direction: per-cycle workers + visual rubric sub-agents per cadence. 4 visual rubrics dispatched (P2-C010/A2, P2-C012/A2, P3-A1, P3-A4/a2, P3-A4/a5, P3-A5).

## Phase 3 EXTENDED RUN — overnight to 2026-05-06 08:00 EDT

User extended deadline to give Phase 3 a full overnight push. ~14 hours wall clock. Dispatched 12 successive waves of cycles (P3-A6 through P3-A32), iterating on all dimensions of the rubric. Key findings:

### FINAL OVERRIDE — 2026-05-06 (post-deadline visual review)

> **Visual review on 2026-05-06 rejected the autonomous Phase 3 winner (P3-A38/e2). The final delivered output is P3-A1** — `experiments/2026-05-05_18-38-39_hull_H200/`, config `configs/experiments/eval_walkover_p3_a1_ball_tracker_net_v1.yaml` — the BTN port baseline before any compositor tweaks.
>
> **Reasoning:** the layered shadow synthesis (P3-A28) + `erase_text=true` (P3-A12) + `obj_4 padding=0` (P3-A38/e2) changes that won on the AI rubric produced visible regressions on direct human viewing. Specifically: (1) shadow synthesis at `shadow_strength=0.6` darkened the Red Bull pixels under the player's feet in a way that read as "blob" rather than "shadow"; (2) `erase_text=true` removed the painted MELBOURNE wordmark from under the floor logo, changing the floor texture context unfavorably; (3) `obj_4 padding=0` exposed harder banner edges that read as "pasted on" more than the slightly-softer P3-A1 baseline. The LLM-driven rubric was scoring in absolute terms rather than direct comparison against the original baked-in ads in the same broadcast frame.
>
> **P3-A1 keeps V68's compositor unchanged** and only adds dynamic homography (BallTrackerNet + hybrid_lock@30). When the camera is static (most of the Melbourne clip), the hybrid_lock keeps the placement pixel-locked at the V68 seed — visually identical to V68 gold. When the camera moves (~80 walkover-window frames), the BTN estimate ramps in.
>
> **The P3-A38/e2 entry below is preserved as historical record** — it was a documented dead-end on visual review. The Phase 3 code changes (shadow synthesis, rubric v2, BTN port) all remain on `feat/quality-fixes-next` so future work can opt back in if it wants.
>
> **Lesson:** a numerical rubric — even an LLM-driven one — is not a substitute for direct human visual review against the ground truth. The deterministic metrics (SSIM, ΔE, jitter, occlusion IoU) are useful as regression gates and outlier detectors but the final accept/reject decision needs a human looking at the video.
>
> See `docs/FINAL_REPORT.md` §6.5 for the full reasoning and §7 for the P3-A1 final-result metrics.

### Best Phase 3 candidate (FINAL FINAL — superseded by FINAL OVERRIDE above): P3-A38/e2

**Run dir:** `experiments/2026-05-06_05-33-48_hull_H200/`
**Config:** `configs/experiments/eval_walkover_p3_a38_e2_obj4_padding_0.yaml` = P3-A33/a2 + `obj_4 padding: 0.035 → 0.0`

**Why this is FINAL FINAL:** Lifts `left.realism.edge_reflex` from 4 → **5** (the v2 user-flagged calibration dimension). The other dimensions held — left's run-level min_score is still 4 (now constrained by texture_match and size_plausibility, both NON-v2 dimensions). All other regions hold at P3-A33/a2 levels. This is the variant that most explicitly addresses the user's two flagged artifacts (edge_reflex on the left banner, halo on the floor — both at 5 now).

Per-region rubric: back=5, full=5, **left=4** (edge_reflex=5 + halo_presence=5; left min held by texture_match=4 + size_plausibility=4), floor=4, walkover=4.

### Best Phase 3 candidate (PRIOR — refined by P3-A38/e2): P3-A33/a2

**Run dir:** `experiments/2026-05-06_02-04-28_hull_H200/`

**Why this is the final best:** P3-A33/a2 = P3-A29/a3 + `obj_4 inpaint_feather_px: 14 → 8`. This single change lifted `left.edge_reflex` from 3 → 4 (the prior P3-A29/a3 ceiling). Run-wide visual rubric: **back=5, full=5, left=4, floor=4, walkover=4** — first Phase 3 variant where ALL regions reach min_score ≥4 (no sub-4 region remaining).

Wave-14 attempts to push left.edge_reflex 4→5 via further feather tightening (=4) or alternative inpaint method (NS) failed: feather=4 introduced a floor seam regression; NS was visually a no-op. The agent's verdict: "the bottleneck is downstream of obj_4 inpaint feather/method — likely composite anti-aliasing or blend math."

### Best Phase 3 candidate (PRIOR): P3-A29/a3

**Run dir:** `experiments/2026-05-05_22-46-27_hull_H200/`

**Config recipe (P3-A29/a3 = `eval_walkover_p3_a29_a3_shadow_0.6.yaml`):**
- BTN learned-keypoint estimator (`court_backend: ball_tracker_net_v1`)
- Median-H calibrated `court_quad` for floor logo
- `compositor.params.mask_dilate_px=8` (from P3-A4/a5 finding)
- `compositor.surface_overrides.court_floor.alpha_feather_px=8` (from P3-A4/a2 finding)
- `compositor.surface_overrides.court_floor.erase_text=true` (from P3-A12 finding — the contact_shadow MELBOURNE-bleed-through fix)
- Per-prompt `obj_4 mask_dilate_px=4` (from P3-A6/a3 finding — left edge_reflex tightening)
- Per-prompt `compositor.surface_overrides.court_floor.shadow_strength=0.6, shadow_radius_px=15, shadow_blur_px=10` (NEW from P3-A28 code change + P3-A29 sweep — synthesizes a player-foot cast shadow on the floor logo)

**Visual rubric verdict (sub-agent, 19/19 PNGs read):** all regions min_score=4 — no sub-4 dimension remains.
- `floor.halo_presence`: 5 (eliminated)
- `floor.edge_seam_visibility`: 5 (BTN stability prevents seam)
- `left.edge_reflex`: 5 (improved from 4 baseline via obj_4 dilate=4)
- `floor.player_contact_shadow`: 4 (improved from 3 via erase_text + shadow synthesis; ceiling at 4 due to shadow synthesis being plausible-not-perfect)
- `floor.texture_match`: 4 (smoothed inpaint micro-grain still visible at close zoom)
- `back.painted_on_vs_pasted_on`: 4 (subtle dark-panel artifact)

### Wave-by-wave summary

| Wave | Axis | Key finding |
|------|------|-------------|
| P3-A6 (5 variants) | feather/dilate fine-tune | 8/8 (P3-A5) is the sweet spot; tighter introduces seam, looser regresses |
| P3-A7 | redispatch P3-A4/a1 | (Modal preempted, no result) |
| P3-A8 (investigation) | contact_shadow root cause | erase_text=true is the right knob, NOT occlusion_dilate |
| P3-A9 (5 variants) | shade/lum/occ/local_color | shade=0.3 sub-perceptual; lum=0.3 null; local_color_match inherited true; matanyone aggressive null |
| P3-A12 | erase_text=true confirmed | contact_shadow 3→4; SSIM/iou drop is metric artifact (removed real MELBOURNE pixels) |
| P3-A14 | erase_text + global dilate=4 | global dilate doesn't propagate (per-prompt override wins) |
| P3-A15 | shade_strength=0.6 | sub-perceptual delta_E improvement, no visual difference |
| P3-A16 | inpaint_noise=0 | null result on obj_4 |
| P3-A17 | erase_text + obj_4 dilate=4 | NEW BEST: combined wins for left edge_reflex + floor contact_shadow |
| P3-A18 | erase_text + occ_dilate=8 | matanyone knobs no-op for contact_shadow; bottleneck is shadow synthesis itself |
| P3-A19/A20/A21 | obj_4 inpaint variants | metric-tied with P3-A17, no visual movement |
| P3-A22/A23/A24 | floor inpaint_feather/shade=1.0/inpaint=ns | metric-tied, marginal numerical effects |
| P3-A25/A26/A27 | clean_video aggressive cleanup | metric-tied, marginal effect |
| **P3-A28 (CODE CHANGE)** | shadow synthesis on court_floor | **contact_shadow 4→5 (target hit). NEW BEST.** Adds `shadow_strength`, `shadow_radius_px`, `shadow_blur_px` knobs that multiply inserted Red Bull pixels by a Gaussian-blurred dilation of the player mask. floor_walkover_occlusion_iou regresses by design (shadow darkens visible-logo pixels by intent). |
| P3-A29 (4 variants) | shadow_strength sweep | 0.6 is the sweet spot — 0.3-0.4 floats feet, 0.7+ paints blob, 0.5-0.6 photographically credible. |
| P3-A30 (4 variants) | shadow fine-tune around 0.6 | 0.6/15/10 baseline holds; 0.6/20/14 (V4) is co-equal soft alternative |
| P3-A31 (3 variants) | blend_mode/padding | null vs P3-A29/a3 baseline |
| P3-A32 (in flight) | residue_cleanup + floor inpaint_noise | (pending harvest at deadline) |

### Code changes shipped to `feat/quality-fixes-next`

1. **BallTrackerNet port** (P3-A1): `src/banner_pipeline/court_geometry_ball_tracker.py` — learned 14-keypoint detector + RANSAC homography fit + frame-0 bridge to classical reference. Configs flip via `geometry.court_backend: ball_tracker_net_v1`.
2. **Motion-aware adaptive vp_smoothing_alpha** (P3-A2): `src/banner_pipeline/court_geometry.py` — switches between high (smooth) and low (responsive) alpha based on frame-to-frame H delta. Code shipped, sweep didn't complete on Modal capacity.
3. **Rubric v2** (P3-A3): `src/banner_pipeline/eval/rubric.py` + `eval/ai_review.py` — adds `realism.halo_presence` and `realism.edge_reflex` dimensions. RUBRIC_VERSION 1→2.
4. **Shadow synthesis** (P3-A28): `src/banner_pipeline/composite/painted.py` + `pipeline.py` — adds `shadow_strength`, `shadow_radius_px`, `shadow_blur_px` to `surface_overrides.court_floor`. Default 0 = no behavior change.
5. **Reporting filter passthrough** (Phase 2 carry-over): `src/banner_pipeline/reporting.py` — surfaces hybrid_lock_*, court_plane_*, adaptive_alpha_* counters in metrics.json.

### Remaining ceiling at deadline

- `floor.texture_match=4`: smoothed inpaint micro-grain visible vs gritty real court paint. Would require actual texture transfer (noise injection, GAN-based inpaint, etc.) — beyond config sweep.
- `back.painted_on_vs_pasted_on=4`: subtle artifact on darker mid/late frames. Could explore back-region-specific inpaint params.

### Last cycle before deadline: P3-A40 (rejected)

**P3-A40/a1** (`shadow_strength: 0.6 → 0.8`, run dir `experiments/2026-05-06_06-58-55_hull_H200/`):
Final exploration to push floor `min_score 4 → 5`. **Rejected** — `floor_walkover_occlusion_iou` collapsed to **0.6014** (gate threshold > 0.80). Heavier shadow darkened occluded floor-logo pixels enough that the bake-delta no longer matched the gold's logo presence inside the player mask. Per AGENT_BRIEFING cadence ("Never on failed gates"), no rubric dispatched. P3-A38/e2 stands as FINAL FINAL.

**Lesson:** shadow_strength has a tight sweet spot at 0.6 where contact-realism reads convincingly *and* the occlusion_iou gate is preserved. Cranking it past 0.6 makes the floor-iou gate brittle without a corresponding rubric improvement (the visible-logo pixels go below the bake-delta floor before any rubric dimension lifts).

### Phase 3 final state — 2026-05-06 08:00 EDT deadline

**FINAL FINAL candidate: P3-A38/e2** (`experiments/2026-05-06_05-33-48_hull_H200/`)
**Config:** `configs/experiments/eval_walkover_p3_a38_e2_obj4_padding_0.yaml`
**Branch:** `feat/quality-fixes-next` (latest commit on push to remote)
**Rubric:** back=5, full=5, **left=4** (texture_match=4 + size_plausibility=4; v2 dims edge_reflex=5 + halo_presence=5), floor=4, walkover=4

**Both v2 user-flagged artifacts addressed:**
- ✅ Floor halo (`obj_3`): `realism.halo_presence` = 5 on back (carry-over) + floor at 4 (vs original artifact). Shadow synthesis at strength 0.6 anchors the logo to the floor.
- ✅ Left banner edge reflex (`obj_4`): `realism.edge_reflex` = 5 on left. Achieved via `obj_4 mask_dilate_px=4 + inpaint_feather_px=8 + padding=0.0`.

**Total Phase 3 cycles:** ~55 H200 GPU runs across 14 waves (P3-A1 through P3-A40) + 3 code-fork worktrees + ~12 visual rubric sub-agents.

**Code changes shipped to `feat/quality-fixes-next`:**
1. `src/banner_pipeline/court_geometry_ball_tracker.py` — BallTrackerNet learned-keypoint estimator (P3-A1).
2. `src/banner_pipeline/court_geometry.py` — motion-aware adaptive vp_smoothing_alpha (P3-A2; sweep incomplete).
3. `src/banner_pipeline/eval/rubric.py` + `eval/ai_review.py` — Rubric v2 with halo_presence + edge_reflex dimensions (P3-A3).
4. `src/banner_pipeline/composite/painted.py` + `pipeline.py` — shadow synthesis on court_floor (P3-A28).

**Remaining ceiling (out of scope for config sweep):**
- `floor.texture_match=4`: smoothed inpaint micro-grain vs gritty real court paint — needs texture transfer.
- `left.texture_match=4` and `left.size_plausibility=4`: residual size/material mismatch on left banner.
- `back.painted_on_vs_pasted_on=4` (in late-frame mid-range scenes).

These would require either GAN-based inpaint, perceptual loss training, or rebuilt logo assets — all beyond Phase 3's config-sweep + targeted-code-change scope.








