# Changelog

All notable milestones for the capstone project. Newest first.

## 2026-05-06 — Final hand-off

- **Final delivered output designated:** P3-A1 (config `configs/experiments/eval_walkover_p3_a1_ball_tracker_net_v1.yaml`, run `experiments/2026-05-05_18-38-39_hull_H200/`).
- **Recipe:** V68 manually-clicked court corners + BallTrackerNet learned-keypoint dynamic homography + hybrid_lock at 30-px tolerance + V68's compositor settings (median_fill inpaint, LED brightness re-baking, MatAnyone2 person-mask occlusion).
- **Visual review override:** rejected the autonomous Phase 3 winner (P3-A38/e2). Layered shadow synthesis + `erase_text=true` + tightened banner edges produced visible regressions on direct viewing despite scoring 5/5 on the LLM-driven rubric. P3-A1 keeps V68's compositor and only adds dynamic homography. See `docs/FINAL_REPORT.md` §6.5.
- **Documentation hand-off:** added `docs/FINAL_REPORT.md` (canonical narrative); restructured `README.md` as project front door; appended FINAL OVERRIDE section to `docs/EXPERIMENT_LEDGER.md`; removed AI-rubric mentions from public docs (`docs/EVALUATION.md`).

## 2026-05-05 → 2026-05-06 — Phase 3: BTN port + autonomous quality experimentation

- **BallTrackerNet port (P3-A1).** New module `src/banner_pipeline/court_geometry_ball_tracker.py` (~720 lines): learned 14-keypoint detector + RANSAC homography + frame-0 bridge to V68's manually-clicked corners. Drop-in replacement for `CourtGeometryEstimator`, selected via `geometry.court_backend: ball_tracker_net_v1`. Sufficient stability under hybrid_lock@30 for the production candidate.
- **Autonomous experimentation framework.** ~50 H200 GPU runs across 14 waves of self-experimenting agents (P3-A1 through P3-A40). Per-cycle worker contract + parallel manager + cross-agent lessons-learned sharing. Defined in `docs/AGENT_BRIEFING.md`.
- **Code shipped:**
  - Motion-aware adaptive `vp_smoothing_alpha` in `court_geometry.py` (P3-A2; sweep didn't conclude).
  - Shadow synthesis on `court_floor` surface override in `composite/painted.py` + `pipeline.py` (P3-A28; new knobs `shadow_strength`, `shadow_radius_px`, `shadow_blur_px`; default 0 = no behavior change).
- **P3-A38/e2 — autonomous winner (later rejected).** Recipe = P3-A1 + shadow_strength=0.6 + erase_text=true + obj_4 padding=0. Scored 5/5 on the rubric for the user-flagged artifacts. Visual review on 2026-05-06 rejected this in favor of the simpler P3-A1.

## 2026-05-04 — Phase 2: hybrid_lock + line-based dynamic estimator (failed axis)

- **Hypothesis.** Replace V68's static homography with `classical_lines_v1` per-frame estimation, gate via `HybridLockState` so noisy frames stay locked at the seed.
- **Sweep.** Two waves of parallel H200 runs over `tolerance_px ∈ {2, 4, 6, 10, 15, 30, 99999}` and `ramp_motion_px_per_frame ∈ {0.3, 1.0, 2.0}`. Configs `eval_walkover_p2_c003_*.yaml` and `eval_walkover_p2_c005_*.yaml`.
- **Conclusion.** With the line-based estimator, no setting of tolerance/smoothing/ramp produced a Pareto improvement over the always-locked V68 baseline. Per-frame estimator noise is the binding constraint. V68 static retained as eval-framework regression gold. The hybrid_lock infrastructure remained sound; it just needed a more stable upstream estimator (→ Phase 3 BTN port).
- **Bug fix.** `hybrid_lock_*` counters were filtered out of `quality_metrics.json` by an allow-list in `src/banner_pipeline/reporting.py`. Fixed in commit `94a0383`.
- **Eval framework built.** `src/banner_pipeline/eval/` module + `python -m banner_pipeline.eval` CLI + `configs/eval/reference.yaml` gold-mapping + per-region scorecards + walkover-window detection + crop strips + side-by-side regression video.

## 2026-04-30 — Phase 1: V68 baseline

- **End-to-end pipeline operational.** SAM2 image segmentation + hull quad fitting + manually-clicked court-corner static homography + inpaint compositor with LED brightness re-baking + MatAnyone2 person-mask occlusion.
- **V68 — the gold reference.** Config `eval_walkover_v68_clicked_homography_static_full.yaml` → run `experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200/`. All five virtual ad regions placed simultaneously (3 back banners + 1 left side banner + 1 court-floor walkover logo). Compositor hand-tuned to: `mask_dilate_px: 20`, `alpha_feather_px: 1`, `inpaint_method: median_fill`, `local_color_match: true`, `blend_mode: led`. These settings persisted all the way to the final.
- **Limitation.** Static homography fails when the camera moves — logos visibly drift off the court. This was the binding limitation that motivated Phase 2.
