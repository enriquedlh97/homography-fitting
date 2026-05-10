# Branch: `feat/sam3-light-v1`

## Goal

Replace the per-frame SAM3 inference of `feat/sam3-v2` with a **light** variant
that runs SAM3 only when the scene actually changes, and reuses the previous
masks otherwise. This brings throughput from **~1 fps** (full SAM3) to
**~3–4 fps** on the same A100-80GB hardware, with comparable detection
quality on static / slowly-changing footage.

## Core idea

- **Frame 0** → full SAM3 run (detection + segmentation). The frame's HSV
  histogram is stored as the *target*.
- **Frame `t > 0`** (online):
  1. compute the HSV histogram of the current frame and the correlation
     `sim = correlation(target_hist, cur_hist)`;
  2. if `sim < similarity_threshold` (default `0.85`) we assume a camera
     change / zoom / new objects entered the scene → **rerun SAM3** and
     promote the current frame to the new *target*. 
  3. otherwise reuse the active masks from the last rerun (adapted by
     the existing tracking + Optical-Flow stage in the pipeline).

The rest of the pipeline (`detection_filter → stabilization → fit →
CornerTracker → inpaint median_fill`) is unchanged: the light segmenter
respects the same `(video_segments, frame_dir, frame_names)` contract as
`SAM3VideoSegmenter`.

Implementation: `src/banner_pipeline/segment/sam3_light_video.py`.
Config: `configs/experiments/sam3_light_auto.yaml`.

## How to run

All runs were executed on **A100-80GB** via Modal:

```bash
modal run scripts/modal_run_sam3.py \
  --config configs/experiments/sam3_light_auto.yaml \
  --mode video
```

Outputs land in `experiments/<timestamp>_sam3_pca_A100-80GB/outputs/composited.mp4`.

## New metrics in `metrics.json`

| Field | Meaning |
|---|---|
| `num_rerun_frames` | how many frames triggered a SAM3 rerun |
| `rerun_frame_indices` | the actual frame indices where a rerun fired |
| `mean_similarity_score` | mean HSV correlation across the run (excluding frame 0) |
| `similarity_threshold` | copy of the configured threshold value |

## Prompt tuning experiments (static tennis video)

The detection quality of SAM3 depends heavily on the text prompt. Summary
of the iterations on the static clip:

| Prompt | Run | Detected / Segmented | Output FPS | Notes |
|---|---|---|---|---|
| `logo` (baseline) | `2026-04-30_20-58-45_sam3_pca_A100-80GB` | — | 2.68 | very stable, but misses several logos |
| `advertising banner` | — | few | — | merges logos together |
| `sponsor logo on fixed courtside advertising board` | `2026-05-03_22-01-40_sam3_pca_A100-80GB` | 11 detected | 3.37 | better, still misses lateral banners |
| `sponsor logo on fixed advertising board at the bottom of the field on the court, one lateral` | — | 13 segmented | 2.70 | wordy, no real gain |
| `sponsor logo on fixed advertising board on tennis court perimeter` | — | 6 segmented | 4.11 | too restrictive |
| **`sponsor logo on fixed advertising board`** ✅ | `2026-05-03_22-39-21_sam3_pca_A100-80GB` | 12 detected | 3.11 | **chosen final prompt** |
| `sponsor logo on fixed advertising board` (rerun) | `2026-05-06_16-13-17_sam3_pca_A100-80GB` | 9 segmented | 3.95 | confirms stability |
| `KIA sponsor logo on fixed advertising board` | `2026-05-07_17-09-30_sam3_pca_A100-80GB` | 8 detected | **4.09** | brand-specific prompt, fastest run |

## Takeaway

- The HSV-similarity gate is what unlocks the speedup: SAM3 is only
  re-executed on a few selected keyframes, so cost scales with scene
  changes rather than frame count.
- Prompt phrasing matters as much as the algorithm — `sponsor logo on
  fixed advertising board` is the best generic prompt found so far for
  tennis footage; brand-specific prompts (e.g. `KIA …`) detect fewer
  logos but run faster.
- This branch is currently the preferred path forward over `feat/sam3-v2` (~1 fps) for static or slowly-changing footage.
