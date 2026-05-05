# Phase 2 axis — hybrid locked-with-tolerance homography

This is the active experimentation direction as of 2026-05-05. Sub-agents working on this axis MUST read this end-to-end before designing any change. Read also `docs/EVALUATION.md` (the eval contract) and `docs/AGENT_BRIEFING.md` (the worker contract).

## The problem

The current production gold is `eval_walkover_v68_clicked_homography_static_full.yaml` — manually clicked court corners, locked for the entire clip.

- **What it does well:** when the camera is static or near-static, v68 looks excellent. Sub-agent rubric review (2026-05-05): back banners 4/5, left side 5/5, floor logo 3/5, full frame 4/5 — within reach of the unmodified broadcast.
- **What it does poorly:** when the camera moves — even a little — the locked homography no longer matches the real court. The Red Bull placements drift and feel jittery against a smoothly moving background. The end of `data/melbourne-walking-over-logo.mov` shows this clearly.
- **Why this matters:** ESPN-quality is the goal. A static-only solution loses credibility the moment a camera operator pans.

## The hybrid we're going to try

Keep v68's clicks as a **seed**. Per-frame, run a **14-point homography estimation** (the approach used in the sibling `tennis-virtual-ads` repo / "10th virtual ad" — see references below). Decision rule per frame:

1. **If** the new estimate is within an **error tolerance** of the locked seed (motivated by the natural thickness of court white lines — there's intrinsic measurement uncertainty in line-based estimation), **stay locked**. No change to placements, no jitter introduced.
2. **If** the new estimate diverges beyond the tolerance, **re-estimate** — but **smooth the transition to match the camera-motion speed**. Don't snap; ramp. The output should look like a natural pan, not a jump.

The key insight (user, 2026-05-05): "The jitter doesn't come from re-estimation per se — it comes from instant updates against a smoothly moving camera. If the homography update rate matches the camera's motion rate, it should look smooth."

## What success looks like

- **Preserves v68's static-camera quality.** The static portion of the Melbourne walkover clip (frames roughly 0–500) should render visually indistinguishable from v68 today. Numerical: all per-region scorecards still pass, no regression vs gold across `back_roi_ssim_vs_reference_mean`, `left_roi_ssim_vs_reference_mean`, `floor_roi_ssim_vs_reference_mean`. Visual: rubric review on those frames matches v68's scores.
- **Wins on motion frames.** End of clip (frames roughly 600–767) — where v68 visibly drifts today — should track the camera's motion. Sub-agent rubric review on those frames should show improvement on `realism.painted_on_vs_pasted_on` and the `temporal.jitter_visible` dimension specifically. (Note: `temporal.player_contact_shadow` is a known weakness of the gold itself and is on a different axis — don't expect this hybrid to fix it.)
- **No regressions on the non-walkover regions.** Back banners and left side are at quality plateau; the hybrid must not break them.

## References / prior art

- **Sibling repo `tennis-virtual-ads`** at `/Users/enriquediazdeleonhicks/repositories/capstone-data-candidates/tennis-virtual-ads/`. Contains a 14-keypoint heatmap-based court detector (BallTrackerNet) and a homography fitter that lands corners from line-based estimation. Use as inspiration; **do not import or modify** that repo (read-only reference).
- **Existing dynamic geometry path here** in `src/banner_pipeline/court_geometry.py` — `CourtGeometryEstimator`. Note: as of 2026-05-04 there's a **known activation bug** in the hybrid pipeline path (see `docs/EXPERIMENT_LEDGER.md` C009/A1 finding): `run_pipeline_video_hybrid` never instantiates `GeometryFittingEngine`, AND `SUPPORTED_GEOMETRY_SURFACE_TYPES` excludes `court_floor` / `banner`. So the existing dynamic config is structurally inert in hybrid mode. Fixing this is part of the work for this axis OR an alternative implementation route.
- **Existing per-frame state dump** in `src/banner_pipeline/pipeline.py` already writes `outputs/per_frame_state.json` with per-frame, per-object quad corners (commit history in this branch). The hybrid should extend this to record (a) the locked-seed corners, (b) the estimated corners per frame, (c) which decision was taken (locked/re-estimated/smoothed). The eval framework's `metrics_geom.py` already consumes the dump.
- **Per-object asset routing** is now supported (commit `47b2665`): `ObjectPrompt.asset` field works in all 3 video paths. Use it if hybrid experiments need to swap the floor logo asset to compare placements.

## Failure modes to watch for

- **Over-snapping:** if the tolerance is too tight, the hybrid re-estimates too often → jitter against a static camera. Monitor `floor_corner_max_jump_px` and `back_corner_max_jump_px`.
- **Under-snapping:** if the tolerance is too loose, the hybrid stays locked through real camera motion → same drift as v68 has today. Visual rubric on motion frames will catch this.
- **Lag / smoothing too slow:** placements lag visibly behind the camera. Tune `vp_smoothing_alpha` / equivalent against motion frames in the eval rubric.
- **Re-estimation noise:** if the line detector occasionally misfires (a player's leg crosses a line at a bad angle), the hybrid should reject the noisy estimate. Outlier rejection is necessary.

## Constraints / hard rules for sub-agents

- **Regression-safety is the contract.** Any candidate must pass all per-region scorecards AND `any_regression: false` vs gold. The eval framework's `detect_regressions` is the gate. (Note: a known bug in `detect_regressions` for `roi_ssim_vs_reference_mean` is documented in EXPERIMENT_LEDGER.md and may need to be fixed as part of this axis if it masks regressions.)
- **Use the visual rubric as the qualitative signal.** Numerical metrics alone won't tell us whether the hybrid is winning the motion-frames test. Dispatch a sub-agent against the run's MANIFEST.md to get rubric scores; compare against v68 gold's rubric (already produced in `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/eval/ai_review/`).
- **You have full creativity within the axis.** If you see a cleaner path than the locked-with-tolerance formulation (e.g., a different estimation algorithm, a different smoothing model, a different gating signal than line-distance), propose it. The manager will read your reasoning. Don't silently pivot to a different problem.
- **Do not break the static-camera baseline** in pursuit of motion gains. If your candidate improves motion frames but regresses on static ones, that's not a win — it's a different tradeoff that needs the user's decision.

## Suggested first cycles

(Order is heuristic; manager may sequence differently.)

1. **Make the dynamic-geometry path actually activate.** Fix the two-part defect documented in EXPERIMENT_LEDGER.md C009/A1 — instantiate `GeometryFittingEngine` in `run_pipeline_video_hybrid`, OR extend `SUPPORTED_GEOMETRY_SURFACE_TYPES` to include `court_floor` / `banner`, OR find a third path. Run a baseline with dynamic geometry enabled (no smoothing, no locking — pure per-frame) to see what raw motion behavior looks like and what the failure modes are.
2. **Add the locking mechanism on top.** Take the v68 clicks as the seed. Implement the tolerance-based decision: stay locked if estimated corners are within X pixels of seed; re-estimate otherwise.
3. **Add motion-rate-matched smoothing.** When a re-estimation fires, ramp the placement transition over N frames where N is computed from observed camera-motion speed (e.g., from optical flow or from per-frame estimate-vs-seed delta).
4. **Calibrate the tolerance.** Tighten / loosen until the static portion stays locked AND the motion portion tracks. Visual rubric on each candidate is the arbiter.

## What this axis does NOT include

- Player-contact shadow rendering (separate axis; would target `temporal.player_contact_shadow`).
- Different occlusion masker (matanyone vs matanyone2).
- Different asset / logo content.
- Generalization to other clips (`zoom-clip-melbourne.mov`, `tennis-clip.mp4`). Reserved for after this axis converges on Melbourne.
