# Evaluation report

- Experiment: `experiments/2026-05-05_15-23-06_hull_H200`
- Reference:  `/Users/enriquediazdeleonhicks/repositories/capstone-data-candidates/homography-fitting/experiments/2026-04-30_17-06-28_walkover_v68_clicked_homography_static_full_H200`
- Geometric source: `static_fallback`

## Per-region scorecards

| Region | Pass | Failed metrics | Warnings |
|---|---|---|---|
| back | PASS | - | roi_delta_E_lab |
| left | PASS | - | roi_delta_E_lab |
| floor | FAIL | floor_roi_jitter_ratio, floor_walkover_logo_visible_pct, floor_walkover_occlusion_iou | roi_delta_E_lab |
| full | PASS | - | - |

**Any regression vs gold:** `True`

**Walkover window:** frames `685–723`

## Visual artifacts
- back_strip: `experiments/2026-05-05_15-23-06_hull_H200/eval/back_banners/crops_strip.png`
- back_motion_early: `experiments/2026-05-05_15-23-06_hull_H200/eval/back_banners/motion_strip_early.png`
- back_motion_mid: `experiments/2026-05-05_15-23-06_hull_H200/eval/back_banners/motion_strip_mid.png`
- back_motion_late: `experiments/2026-05-05_15-23-06_hull_H200/eval/back_banners/motion_strip_late.png`
- left_strip: `experiments/2026-05-05_15-23-06_hull_H200/eval/left_logo/crops_strip.png`
- left_motion_early: `experiments/2026-05-05_15-23-06_hull_H200/eval/left_logo/motion_strip_early.png`
- left_motion_mid: `experiments/2026-05-05_15-23-06_hull_H200/eval/left_logo/motion_strip_mid.png`
- left_motion_late: `experiments/2026-05-05_15-23-06_hull_H200/eval/left_logo/motion_strip_late.png`
- floor_strip: `experiments/2026-05-05_15-23-06_hull_H200/eval/floor_logo/crops_strip.png`
- floor_motion_early: `experiments/2026-05-05_15-23-06_hull_H200/eval/floor_logo/motion_strip_early.png`
- floor_motion_mid: `experiments/2026-05-05_15-23-06_hull_H200/eval/floor_logo/motion_strip_mid.png`
- floor_motion_late: `experiments/2026-05-05_15-23-06_hull_H200/eval/floor_logo/motion_strip_late.png`
- full_strip: `experiments/2026-05-05_15-23-06_hull_H200/eval/full/crops_strip.png`
- walkover_consecutive_frames: `experiments/2026-05-05_15-23-06_hull_H200/eval/walkover/consecutive_frames.png`
- walkover_forensic_sheet_entry: `experiments/2026-05-05_15-23-06_hull_H200/eval/walkover/forensic_sheet_entry_f0685.png`
- walkover_forensic_sheet_pre_contact: `experiments/2026-05-05_15-23-06_hull_H200/eval/walkover/forensic_sheet_pre_contact_f0694.png`
- walkover_forensic_sheet_contact: `experiments/2026-05-05_15-23-06_hull_H200/eval/walkover/forensic_sheet_contact_f0704.png`
- walkover_forensic_sheet_post_contact: `experiments/2026-05-05_15-23-06_hull_H200/eval/walkover/forensic_sheet_post_contact_f0713.png`
- walkover_forensic_sheet_exit: `experiments/2026-05-05_15-23-06_hull_H200/eval/walkover/forensic_sheet_exit_f0723.png`
- vs_reference_video: `experiments/2026-05-05_15-23-06_hull_H200/eval/vs_reference_side_by_side.mp4`
