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

## Current best config

Based on experiments as of April 24, 2026:

- Mode: `video_hybrid`
- Compositor: `inpaint` with `padding=0.0, lum_strength=0.8, shade_blend=true, inpaint_method=median_fill`
- Results: jitter=0.62 (PASS), ssim=0.995 (PASS), 4/6 metrics passing
- Remaining issues: inpaint delta_E metric (needs fix), perspective metric (needs fix)
