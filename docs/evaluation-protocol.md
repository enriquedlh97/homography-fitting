# Evaluation Protocol

Every experiment MUST be evaluated before presenting results or iterating.
No exceptions. Run metrics + visual inspection on EVERY run.

## Test Clips

Evaluate on ALL clips, not just one. The system must generalize.

| Clip | Path | Purpose | Frames | FPS |
|------|------|---------|--------|-----|
| Original | `data/tennis-clip.mp4` | Baseline quality (back banner, court floor, side panels) | 204 | 25 |
| Walking-over | `data/melbourne-walking-over-logo.mov` | Player occlusion (walks over MELBOURNE) | 778 | 59 |
| Zoom | `data/zoom-clip-melbourne.mov` | Camera zoom stability | 327 | 56 |

A change is only valid if it works well on ALL clips. Overfitting to one
clip is not acceptable.

## Step 1: Run Quality Metrics

```bash
uv run python scripts/quality_eval.py \
  --experiment experiments/<experiment_dir> \
  --original data/<input_video>
```

### Metric Thresholds

| Metric | Threshold | What it measures |
|--------|-----------|------------------|
| Jitter ratio | ≤ 1.05 | Frame-to-frame pixel instability vs original |
| Corner max jump | < 2.0 px | Logo position stability |
| Logo area CV | < 0.05 | Logo sizing consistency |
| Overlay accel p95 | < 1.0 px | Smooth motion (no jerky jumps) |
| Inpaint color dE | < 5.0 | Color match between inpainted and original |
| Temporal SSIM | > 0.95 | No flickering or abrupt changes |

ALL metrics must pass. If any fail, diagnose and fix before proceeding.

## Step 2: Visual Inspection

Extract crops at 6 evenly-spaced frames. For each frame, extract:

### Crops to generate
1. **Full frame** — overall scene assessment
2. **Banner strip** (y=5:90, x=50:1870) — all 7 back banner slots + ANZ
3. **Court floor MELBOURNE** — close-up of the MELBOURNE replacement area
4. **Court floor left** (blua/YoPRO) — close-up of the left court logo
5. **KIA side panels** — close-up of the side barrier logos
6. **Player overlap** — frames where player walks near/over court logos

### Visual QA checklist

For EACH crop, answer these questions:

- [ ] **Original text fully erased?** No residual text visible under the logo
- [ ] **Logo correctly positioned?** At the right location, not offset
- [ ] **Logo size appropriate?** Not too small, not too large for the slot
- [ ] **Color matches surface?** Logo blends with the banner/court color
- [ ] **No visible patch boundary?** Inpainting transitions smoothly
- [ ] **No artifacts?** No ghosting, halos, color bleeding
- [ ] **Player occlusion correct?** Players appear IN FRONT of court logos
- [ ] **No foot distortion?** Player feet/legs look natural near logo edges
- [ ] **Perspective plausible?** Logo looks painted on the surface, not floating
- [ ] **Would a viewer notice?** The key test — would someone think this is fake?

If ANY answer is "no", the experiment FAILS visual QA. Document what
failed and iterate.

## Step 3: Cross-Clip Validation

After passing on one clip, run the SAME pipeline config on all other clips:

1. Adjust only the prompts (click points + box) for each clip
2. The pipeline code and compositor settings must be IDENTICAL
3. If quality degrades on a different clip, the approach is overfitting

### What should NOT change between clips:
- Pipeline code (compositor, fitter, masker)
- Compositor params and surface_overrides
- Blend modes, feathering, alpha_scale values

### What CAN change between clips:
- Prompt click points (where to click on text)
- Prompt bounding boxes (region around text)
- Video path

## Step 4: Document Results

After evaluation, record in the experiment directory:

```
experiments/<timestamp>/
  config.yaml          # frozen config
  metrics.json         # pipeline metrics
  quality_metrics.json # quality_eval.py output
  crops/               # visual inspection crops
  outputs/
    composited.mp4     # output video
```

## Feedback Loop

```
┌─────────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
│ Make change │ ──> │ Run on   │ ──> │ Evaluate     │ ──> │ Pass?    │
│ (code/cfg)  │     │ Modal    │     │ metrics +    │     │          │
│             │     │ (all     │     │ visual QA    │     │ YES: commit
│             │     │  clips)  │     │ (all clips)  │     │ NO: iterate
└─────────────┘     └──────────┘     └──────────────┘     └──────────┘
```

Never present results without running this full loop.
Never skip evaluation.
Never claim something works without checking the crops.
