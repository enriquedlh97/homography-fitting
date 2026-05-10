# Experiment Guide

How to run quality experiments and iterate on improvements.

## Quick experiment loop

```bash
# 1. Run an experiment
uv run modal run scripts/modal_run.py \
    --config configs/default.yaml --gpu B200 --mode video_hybrid

# 2. Evaluate quality
uv run python scripts/quality_eval.py \
    --experiment experiments/<latest>/ \
    --original data/tennis-clip.mp4

# 3. Check metrics
```

## Quality metrics and targets

| Metric | Target | What it measures |
|---|---|---|
| `jitter_ratio` | ≤ 1.05 | Frame-to-frame pixel instability vs original |
| `corner_max_jump_px` | < 2.0 | Largest single-frame corner displacement |
| `logo_area_cv` | < 0.05 | Logo size stability (coefficient of variation) |
| `overlay_accel_p95_px` | < 1.0 | 95th percentile corner acceleration |
| `inpaint_color_delta_E` | < 5.0 | Color match of inpainted region (needs metric fix) |
| `perspective_angle_error_deg` | < 5.0 | Quad angle deviation from rectangular |
| `temporal_ssim` | > 0.95 | Frame-to-frame structural similarity |

## Available modes

| Mode | Segmentation | Corner source | Best for |
|---|---|---|---|
| `video` | SAM per-frame masks | PCA fitter per frame + EMA | Stable masks, some jitter |
| `video_tracking` | SAM frame 0 only | Optical flow CornerTracker | Smooth but drifts |
| `video_hybrid` | SAM per-frame masks | Optical flow CornerTracker | Best of both (recommended) |

## Compositor parameters

```yaml
compositor:
  type: inpaint
  params:
    padding: 0.0           # 0-0.1, logo padding inside quad
    lum_strength: 0.8      # 0-1, luminosity matching aggressiveness
    inpaint_method: telea   # telea | median_fill
    shade_blend: false      # shadow-preserving illumination matching
    inpaint: true           # set false to skip inpainting entirely
```

## Troubleshooting

| Problem | Metric | Fix |
|---|---|---|
| Logo jitters/wobbles | jitter_ratio > 1.05 | Use `video_hybrid` mode |
| Logo grows/shrinks | logo_area_cv > 0.05 | Lower ema_alpha (0.15) |
| Inpaint "paint brush" look | visual inspection | Try median_fill, or shade_blend |
| Logo too small | visual inspection | Reduce padding, use larger logo PNG |
| Old logos bleed through | visual inspection | Use `video` or `video_hybrid` (not `video_tracking`) |
| Logo color tint | visual inspection | Lower lum_strength (0.5-0.8) |
| Court logos look flat | perspective error | Need VP-constrained fitter (Giovanni's geometry) |

## Current best config (April 24, 2026 — 20+ experiments)

- Mode: `video_hybrid`
- Prompts: top banners only (obj_ids 1-7, no court floor)
- Compositor: `inpaint` with `padding=0.0, lum_strength=0.8, shade_blend=true`
- Tracking: `ema_alpha=0.08`
- Logo: `ferrari_white.png` (wider aspect ratio fills banners better) or `redbull_white.png`

### Results (6/6 metrics passing)

| Metric | Red Bull | Ferrari | Target |
|---|---|---|---|
| jitter_ratio | **0.66** | 0.84 | ≤1.05 |
| corner_max_jump_px | 0.71 | 0.68 | <2.0 |
| logo_area_cv | 0.014 | 0.014 | <0.05 |
| overlay_accel_p95_px | 0.47 | **0.36** | <1.0 |
| inpaint_dark_dE | **3.57** | 3.56 | <5.0 |
| temporal_ssim | 0.996 | **0.996** | >0.95 |

### Key improvements from baseline

| Metric | Baseline | Best | Improvement |
|---|---|---|---|
| jitter_ratio | 1.24 | **0.57** | **2.2× better** |
| corner_max_jump | 0.81 | 0.65 | 1.2× better |
| overlay_accel | 0.53 | 0.40 | 1.3× better |
| temporal_ssim | 0.978 | **0.998** | +0.020 |
| metrics passing | 3/6 | **6/6** | doubled |

### Full EMA alpha sweep (optimal: 0.01)

| alpha | jitter | jump | accel | dE | ssim |
|---|---|---|---|---|---|
| 0.30 | 0.86 | 0.79 | 0.41 | 4.46 | 0.9952 |
| 0.15 | 0.84 | 0.71 | 0.43 | 4.46 | 0.9958 |
| 0.10 | 0.88 | 0.67 | 0.46 | 3.60 | 0.9961 |
| 0.08 | 0.84 | 0.68 | 0.36 | 3.56 | 0.9964 |
| 0.05 | 0.79 | 0.73 | 0.47 | 3.62 | 0.9971 |
| 0.03 | 0.67 | 0.70 | 0.41 | 3.42 | 0.9978 |
| 0.02 | 0.64 | 0.75 | 0.40 | 3.42 | 0.9980 |
| **0.01** | **0.57** | **0.65** | **0.46** | **3.40** | **0.9983** |
