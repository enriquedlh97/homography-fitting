# Banner Pipeline

Video banner and logo replacement pipeline for sports broadcast footage. Detects billboard and court-marking regions in video frames using SAM2 or SAM3.1 segmentation, tracks them across all frames, fits perspective-aware quadrilaterals using court-geometry constraints, stabilises the masks temporally, and composites replacement logos with correct aspect ratio, luminosity matching, and shading adaptation.

## Table of contents

- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
  - [Step 1: Collect prompts](#step-1-select-banner-regions-local-no-gpu-needed)
  - [Step 2: Run on GPU via Modal](#step-2-run-on-a-gpu-via-modal)
  - [SAM3 quick-check workflow](#sam3-quick-check)
  - [SAM3 prompting rules](#sam3-prompting-rules)
  - [Troubleshooting SAM3 previews](#if-sam3-preview-fails)
- [Running locally](#running-locally)
- [Architecture](#architecture)
  - [Pipeline stages](#pipeline-stages)
  - [Project structure](#project-structure)
- [Configuration reference](#configuration-reference)
  - [Segmenters](#segmenters)
  - [Fitters](#fitters)
  - [Compositors](#compositors)
  - [Court geometry](#court-geometry)
  - [Stabilization](#stabilization)
  - [Surface types and geometry models](#surface-types-and-geometry-models)
- [Metrics](#metrics)
- [Experiments and reproducibility](#experiments-and-reproducibility)
- [Benchmarking](#benchmarking)
  - [Single-config benchmarks](#benchmarking-across-gpus)
  - [Benchmark matrix](#benchmark-matrix-multiple-prompt-counts--multiple-gpus)
- [GPU reference](#available-gpus)
- [Tests](#tests)
- [Adding new components](#adding-new-components)

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd homography-fitting

# 2. Install all dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Authenticate with Modal (one-time, for GPU runs)
uv run modal setup
```

SAM2 setup is only needed for **local** runs. Modal builds SAM2 from source automatically.

```bash
# Only if running locally (not needed for Modal)
git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
```

### Dependencies

The project is managed with [uv](https://docs.astral.sh/uv/) and built with [Hatchling](https://hatch.pypa.io/). Core runtime dependencies:

| Package | Purpose |
|---------|---------|
| `opencv-python` | Frame I/O, contour analysis, homography, inpainting, video encoding |
| `numpy`, `scipy` | Numerical operations, linear programming (LP fitter) |
| `torch`, `torchvision` | SAM2/SAM3 model inference |
| `pyyaml` | Config loading/saving |
| `matplotlib`, `Pillow` | Visualisation and image manipulation |
| `modal` | Remote GPU execution |

Dev dependencies: `pytest`, `ruff`, `mypy`, `pre-commit`.

## Running the pipeline

Two-step process: collect clicks locally, then run on a remote GPU.

### Step 1: Select banner regions (local, no GPU needed)

```bash
uv run python scripts/collect_prompts.py --config configs/default.yaml
uv run python scripts/collect_prompts.py --config configs/sam3_default.yaml
uv run python scripts/collect_prompts.py --config configs/sam3_court_eval.yaml
```

This opens the selected frame of the video and saves the prompt points into the config automatically.

- SAM2: left-click positive points as usual.
- SAM3: left-click positive points, right-click negative points, `U` undo, `N` next object.
- SAM3 prompting works best with 1 to 2 positive clicks inside the banner plus negative clicks on nearby background. Do not outline the whole perimeter.

The `--frame` flag selects which frame to display (e.g. `--frame 20`). The `--video` flag overrides the video path from the config.

### Step 2: Run on a GPU via Modal

```bash
# Video mode (processes all frames, outputs .mp4)
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --mode video
uv run modal run scripts/modal_run.py --config configs/sam3_default.yaml --gpu A100 --mode video
uv run modal run scripts/modal_run.py --config configs/sam3_court_eval.yaml --gpu A100 --mode video

# Image mode (processes single frame, outputs .png)
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --mode image
uv run modal run scripts/modal_run.py --config configs/sam3_default.yaml --gpu A100 --mode image
uv run modal run scripts/modal_run.py --config configs/sam3_court_eval.yaml --gpu A100 --mode image
```

For SAM3, use `--mode image` first to preview the prompt-stage masks and geometry-constrained quads on the selected frame. If the preview looks wrong, or the preview metrics do not include `geometry_*` and `compositor_*`, adjust the clicks or config before running `--mode video`.
The shipped SAM3 configs now use `temporal_rectified`, which caches a rectified clean plate for wall banners and a shaded rectified plate for the `blua.` court eval.

`configs/sam3_default.yaml` must not be run on `T4`. The launcher rejects
that combination locally before any remote build starts because SAM3 requires
FlashAttention and `T4` is not supported for that path.

### SAM3 Quick Check

Use this loop to validate that SAM3 is working before launching a full video run:

```bash
# 1. Collect or recollect prompts on a chosen frame
uv run python scripts/collect_prompts.py --config configs/sam3_default.yaml --frame 0
uv run python scripts/collect_prompts.py --config configs/sam3_court_eval.yaml --frame 0

# 2. Preview the wall-banner config and inspect the metrics
uv run modal run scripts/modal_run.py --config configs/sam3_default.yaml --gpu A100 --mode image

# 3. Preview the court-plane eval config and inspect the metrics
uv run modal run scripts/modal_run.py --config configs/sam3_court_eval.yaml --gpu A100 --mode image

# 4. Confirm the preview metrics include geometry_* and compositor_* and show the intended fit method

# 5. If the previews look good, run the full video
uv run modal run scripts/modal_run.py --config configs/sam3_default.yaml --gpu A100 --mode video
uv run modal run scripts/modal_run.py --config configs/sam3_court_eval.yaml --gpu A100 --mode video
```

The preview run writes a single composited image to `experiments/.../outputs/composited.png`.
Inspect that PNG and the saved `metrics.json` before running `--mode video`.

### SAM3 Prompting Rules

- Use 1 to 2 positive clicks inside each banner.
- Add 1 negative click on adjacent background if the mask bleeds.
- Do not outline the whole banner perimeter with many positive points.
- When validating a new setup, start with one banner before adding more objects.
- `configs/sam3_default.yaml` is the wall-banner config. `back_wall_banner` objects use a fronto-parallel wall solver plus the `temporal_rectified` wall-plate compositor, and the shipped `side_wall_banner` prompt is explicitly forced onto `mask_free_quad` until a dedicated side-wall model is ready.
- `configs/sam3_court_eval.yaml` is the court-plane validation config for the `blua.` left-court ad. It uses `court_plane` plus the `temporal_rectified` court-plate compositor so you can validate court stability separately from the wall run.

### If SAM3 Preview Fails

- Try a different seed frame with `--frame 10`, `--frame 20`, or another clearer frame.
- Use 2 positive clicks plus 1 negative click instead of a single positive click.
- Reduce the test to one object and verify that first.
- If the log shows `usable_outputs=False parsed_nonempty_masks=0`, interpret it as:
  the prompt request was accepted, but SAM3 returned no usable mask for that preview frame.
- If the preview metrics are missing `geometry_*` or `stabilization_*` despite those features being enabled in the config, treat the run as invalid. The pipeline now fails loudly for that case instead of silently saving a contour-only experiment.

### Current SAM3 Preview Limitation

`--mode image` returns a single composited preview image, but the current SAM3 implementation still loads the extracted frame set to initialize the predictor session. In other words, it is a preview of the selected frame's output, not yet a truly cheap first-frame-only execution path.

## Running locally

```bash
# Interactive (opens UI for clicking + runs SAM2 locally)
uv run python scripts/run_pipeline.py --config configs/default.yaml --save result.png

# Run experiment with saved outputs + metrics
uv run python scripts/run_experiment.py --config configs/default.yaml

# With per-operation profiling (adds composite_breakdown_ms to metrics)
uv run python scripts/run_experiment.py --config configs/default.yaml --profile
```

## Architecture

### Pipeline stages

The pipeline processes video through five sequential stages, each implemented as a swappable component:

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│ 1. Segment   │───▶│ 2. Stabilize │───▶│ 3. Geometry   │───▶│ 4. Fit quads   │───▶│ 5. Composite │
│ (SAM2/SAM3)  │    │ (optical     │    │ (vanishing    │    │ (PCA / LP /   │    │ (inpaint /   │
│              │    │  flow masks) │    │  points +     │    │  hull /       │    │  alpha /     │
│              │    │              │    │  court lines) │    │  fronto /     │    │  temporal    │
│              │    │              │    │               │    │  VP-constr.)  │    │  rectified)  │
└─────────────┘    └──────────────┘    └──────────────┘    └───────────────┘    └─────────────┘
```

1. **Segmentation** — SAM2 (single-frame or video) or SAM3.1 (video with Object Multiplex) produces per-object binary masks for every frame. SAM3.1 uses a streamed `propagate_in_video` API with automatic reanchoring when tracking quality degrades.

2. **Stabilization** — A hybrid temporal mask stabilization pass (optional, `stabilization.enabled: true`). Estimates inter-frame motion via sparse optical flow (Shi-Tomasi corners + Lucas-Kanade pyramidal tracking), warps previous masks forward, and fuses them with the raw tracker output. Supports hard-hold for static frames, weighted blending for moving frames, and mask carry-forward when the tracker drops an object. Gated by IoU between predicted and raw masks.

3. **Court geometry estimation** — Detects court lines via edge detection and Hough transforms, classifies them into width (horizontal) and depth (perspective) families, estimates vanishing points for each family, and smooths the estimates temporally with exponential moving averages. Produces a per-frame `CourtGeometryEstimate` with vanishing points, dominant directions, court boundary lines, and an image-to-court homography.

4. **Quad fitting** — Converts each object's binary mask into a 4-corner quadrilateral. Five fitting strategies are available, selected per-object based on the `geometry_model` and `surface_type`:

   | Strategy | When used | Algorithm |
   |----------|-----------|-----------|
   | `pca` / `lp` / `hull` | `mask_free_quad` or geometry disabled | Mask-only geometric fitting |
   | `fronto_parallel_wall_banner` | `back_wall_banner` surfaces | Oriented rectangle from mask contour + smoothed court width direction |
   | `vp_constrained_horizontal_banner` | Horizontal banners with VP | Support lines + VP rays from the depth vanishing point |
   | `vp_constrained_vertical_banner` | `side_wall_banner` surfaces | Support lines + VP rays from the width vanishing point |
   | `court_plane` | `court_marking` surfaces | Quad projected via the court homography from a stored local-plane template |

   All geometry-constrained fitters smooth their parameters temporally (support offsets, lateral offsets, ray angles) and support hold-last-good + fallback-to-mask-free-quad when the geometry estimate is unavailable.

5. **Compositing** — Warps the replacement logo into each detected quad region. Three strategies:

   | Compositor | Algorithm |
   |------------|-----------|
   | `inpaint` | Inpaints the old logo away (Telea), builds an aspect-aware logo canvas, matches luminosity in LAB space, alpha-blends with soft Gaussian edges |
   | `alpha` | Uses camera-intrinsics-based oriented homography decomposition to recover the physical aspect ratio, estimates background fill colour from the rectified border band, warps and alpha-composites |
   | `temporal_rectified` | Stateful compositor that rectifies the quad region to a canonical plate, caches a clean plate (wall) or updates a shading field (court), composites the logo in rectified space, and warps back. Supports wall-plate freezing after initialisation and court-plane shading adaptation via per-frame luminosity ratio estimation |

### Project structure

```
src/banner_pipeline/
  pipeline.py                              # Orchestration: config loading, factories, run_pipeline(), run_pipeline_video()
  court_geometry.py                        # Court line detection, VP estimation, GeometryFittingEngine
  stabilization.py                         # Hybrid temporal mask stabilization via optical flow
  quality.py                               # Mask/quad quality validation, geometry flags, fallback fitting
  reporting.py                             # Metrics report builder for persisted experiments
  sam3_attention.py                        # GPU-family FlashAttention backend selection for SAM3
  device.py                                # Torch device detection, SAM2/SAM3 model loading
  io.py                                    # Frame extraction (ffmpeg), StreamingVideoWriter, video FPS
  geometry.py                              # Line intersection (parametric + implicit), corner sorting
  viz.py                                   # Visualisation helpers
  ui.py                                    # Interactive click collection UI (matplotlib)
  diff.py                                  # Frame differencing utilities
  _perf.py                                 # Optional per-operation timing (enabled with --profile)

  segment/
    base.py                                # ObjectPrompt dataclass, SegmentationModel ABC
    sam2_image.py                           # SAM2 single-frame segmenter
    sam2_video.py                           # SAM2 video tracker
    sam3_video.py                           # SAM3.1 video tracker (Object Multiplex, reanchoring)

  fitting/
    base.py                                # QuadFitter ABC
    pca_fit.py                             # Weighted-PCA with Hann windows
    lp_fit.py                              # Linear programming supporting lines
    hull_fit.py                            # Hull vertex deduction (handles off-screen corners)
    fronto_parallel.py                     # Fronto-parallel wall banner fitting
    vp_constrained.py                      # Vanishing-point-constrained banner fitting

  homography/
    camera.py                              # Camera intrinsics, oriented homography decomposition
    court.py                               # Court-specific homography helpers

  composite/
    base.py                                # Compositor ABC
    inpaint.py                             # Inpaint + LAB luminosity matching
    alpha.py                               # Oriented-homography alpha-blend
    temporal_rectified.py                  # Stateful rectified-plane compositor

configs/
  default.yaml                             # SAM2 single-frame config
  sam3_default.yaml                        # SAM3 wall-banner config (geometry + stabilization enabled)
  sam3_court_eval.yaml                     # SAM3 court-plane validation config
  experiments/                             # Custom experiment configs
  matrix/                                  # Benchmark matrix configs (1/5/11 prompts × SAM2/SAM3)

scripts/
  collect_prompts.py                       # Interactive prompt collection (no GPU)
  modal_run.py                             # Remote GPU execution via Modal
  run_pipeline.py                          # Local single-frame pipeline
  run_experiment.py                        # Local experiment runner with metrics
  run_matrix.sh                            # Sequential benchmark matrix
  run_matrix_parallel.py                   # Parallel benchmark matrix
  benchmark_fps.py                         # FPS benchmarking
  analyze_matrix.py                        # Matrix results analysis
  compare_baselines.py                     # Baseline comparison
  perf_summary.py                          # Performance summary

tests/                                     # Unit and integration tests
archive/                                   # Legacy scripts from before the refactor
```

See [MIGRATION.md](MIGRATION.md) for how the old files map to this structure.

## Configuration reference

All configuration is done through YAML files. The top-level structure:

```yaml
pipeline:
  mode: video              # "image" (single frame) or "video" (full video)
  segmenter: { ... }
  fitter: { ... }
  compositor: { ... }
  camera: { ... }
  geometry: { ... }        # Court geometry (optional)
  stabilization: { ... }   # Mask stabilization (optional)
input:
  video: data/tennis-clip.mp4
  logo: data/sponsor_logo.png
  prompts: [ ... ]
output:
  dir: experiments/
```

### Segmenters

| Type | Model | Notes |
|------|-------|-------|
| `sam2_image` | SAM 2.1 | Single-frame segmentation. Works on all GPUs including T4. |
| `sam2_video` | SAM 2.1 | Multi-frame video tracking. Works on all GPUs including T4. |
| `sam3_video` | SAM 3.1 Object Multiplex | Multi-frame video tracking with joint multi-object tracking. Requires FlashAttention (L4+ GPUs). Supports automatic reanchoring when tracking degrades. |

SAM3 segmenter config:
```yaml
segmenter:
  type: sam3_video
  checkpoint: sam3/checkpoints/sam3.1_multiplex.pt
```

SAM2 segmenter config:
```yaml
segmenter:
  type: sam2_image  # or sam2_video
  checkpoint: sam2/checkpoints/sam2.1_hiera_tiny.pt
  model_cfg: configs/sam2.1/sam2.1_hiera_t.yaml
```

### Fitters

| Fitter | Algorithm | Best for |
|--------|-----------|----------|
| `pca` | Weighted PCA with Hann windows on split contour edges | Rectangular banners viewed at moderate perspective |
| `lp` | Four supporting-line linear programmes intersected | Tight convex bounds |
| `hull` | Hull vertex deduction with boundary classification | Regions extending off-screen (1-2 corners outside the frame) |

Fitter config:
```yaml
fitter:
  type: pca           # pca | lp | hull
  params:
    axis: short        # PCA-specific: "short" (default) or "long"
```

The geometry engine adds two additional fitting strategies that are selected per-object via `geometry_model`, not via the fitter config:

| Geometry fitter | Description |
|-----------------|-------------|
| `fronto_parallel_wall_banner` | Oriented rectangle whose parallel edges follow the smoothed court width direction. Parameters (support + lateral offsets) are blended temporally. |
| `vp_constrained_horizontal_banner` / `vp_constrained_vertical_banner` | Quad whose converging edges pass through the scene vanishing point. Parameters (support offsets + ray angles) are blended temporally with circular-angle averaging. |

### Compositors

| Compositor | Algorithm | Best for |
|------------|-----------|----------|
| `inpaint` | Inpaints old logo (Telea), builds aspect-aware canvas, matches luminosity in LAB space, soft-edge alpha blend | General use, single-frame or video |
| `alpha` | Camera-intrinsics oriented homography for physical aspect ratio, border-band background fill estimation, alpha composite | Aspect-ratio-correct warping |
| `temporal_rectified` | Stateful rectified-plane compositor with cached clean plates (wall) or shading fields (court) | Video mode with geometry-constrained banners |

Compositor config for `temporal_rectified`:
```yaml
compositor:
  type: temporal_rectified
  params:
    padding: 0.05                  # Logo padding as fraction of plate size
    rectified_min_size_px: 500     # Minimum dimension of the rectified plate
    erase_mask_dilate_px: 7        # Dilation for the inpaint erase mask
    wall_freeze_after_init: true   # Freeze the wall clean plate after first frame
    court_shading_enabled: true    # Enable per-frame shading adaptation for court plates
    court_shading_blur_px: 41      # Gaussian blur kernel for shading field
    court_shading_alpha: 0.9       # EMA alpha for shading field updates
```

The `temporal_rectified` compositor automatically selects per-object compositing models based on `surface_type`:
- `back_wall_banner` → `temporal_wall_plate` (frozen clean plate)
- `court_marking` → `temporal_court_plate` (shading-adapted plate)
- Anything else → delegates to `InpaintCompositor`

### Court geometry

Court geometry estimation detects court lines from the video frames and uses them to constrain banner quad fitting. This produces temporally stable quads that follow the scene's perspective geometry.

```yaml
geometry:
  enabled: true
  court_backend: classical_lines_v1   # Line detection backend
  vp_smoothing_alpha: 0.7             # EMA alpha for vanishing point smoothing
  back_wall_line_smoothing_alpha: 0.9  # EMA alpha for back-wall fit parameter smoothing
  line_smoothing_alpha: 0.65           # EMA alpha for VP-constrained fit parameter smoothing
  hold_frames: 8                       # Max frames to hold the last good quad when fit fails
  fallback_after_frames: 3             # Fall back to mask_free_quad after this many failed frames
  vp_confidence_min: 0.35             # Minimum VP confidence to use geometry-constrained fitting
  tangent_margin_px: 2.0              # Margin added to mask tangent support lines
```

The `classical_lines_v1` backend pipeline:
1. Convert to grayscale, blur, threshold bright regions, morphological open
2. Canny edge detection
3. Probabilistic Hough line detection
4. Classify lines into width family (near-horizontal) and depth families (positive/negative slope)
5. Filter angle inliers within each family
6. Estimate vanishing points from line intersections (within-family for width, cross-family for depth)
7. Select boundary lines (top/bottom width, left/right depth)
8. Estimate court homography from 4 boundary line intersections

Per-frame geometry estimates are smoothed via exponential moving average. Scene cuts (detected by high inter-frame mean absolute difference > 18.0) reset all smoothed state.

### Stabilization

Hybrid temporal mask stabilization reduces jitter in tracked masks:

```yaml
stabilization:
  enabled: true
  mode: hybrid                         # Only "hybrid" is supported
  static_motion_threshold_px: 0.75     # Below this corner displacement → near-static frame
  hold_corner_rms_px: 1.25             # Hard-hold threshold for quad corner RMS displacement
  mask_iou_gate: 0.55                  # Minimum IoU between raw and predicted mask to blend
  max_hold_frames: 6                   # Max frames to carry forward an empty mask
  predicted_mask_weight: 0.65          # Weight of predicted mask in fusion blend
  morph_kernel_px: 3                   # Morphological cleanup kernel size
```

Stabilization pipeline per frame:
1. Estimate inter-frame homography from sparse optical flow (Shi-Tomasi + Lucas-Kanade)
2. Warp previous stabilized mask forward using the estimated homography
3. Compare warped prediction with raw tracker mask:
   - If raw mask is empty and within hold budget → carry forward predicted mask
   - If IoU ≥ gate and frame is near-static and quad corners are close → hard-hold (use predicted mask as-is)
   - If IoU ≥ gate → fuse predicted + raw mask with weighted blend + morphological cleanup
   - If IoU < gate → accept raw mask (scene change or re-detection)

### Surface types and geometry models

Each prompt can specify a `surface_type` and optionally a `geometry_model` override:

```yaml
prompts:
  - obj_id: 2
    points: [[899, 45], [983, 43], [963, 78]]
    labels: [1, 1, 0]
    surface_type: back_wall_banner       # Surface type for routing
    # geometry_model: mask_free_quad     # Optional override
```

| Surface type | Default geometry model | Compositor model |
|-------------|----------------------|------------------|
| `banner` (default) | `mask_free_quad` | `inpaint` (delegated) |
| `back_wall_banner` | `fronto_parallel_wall_banner` | `temporal_wall_plate` |
| `side_wall_banner` | `vp_constrained_vertical_banner` | `inpaint` (delegated) |
| `court_marking` | `court_plane` | `temporal_court_plate` |

Available `geometry_model` overrides: `mask_free_quad`, `fronto_parallel_wall_banner`, `vp_constrained_horizontal_banner`, `vp_constrained_vertical_banner`, `court_plane`.

## Metrics

Each run produces a `metrics.json` in the experiment directory. The metrics cover every pipeline stage and vary depending on the mode (image vs. video) and which features are enabled.

### Core timing metrics

| Metric | Description |
|--------|-------------|
| `segment_total_s` | Time for segmentation and tracking across all frames |
| `fit_mean_ms` / `fit_std_ms` | Average and std of quad fitting time per frame |
| `composite_mean_ms` / `composite_std_ms` | Average and std of compositing time per frame |
| `write_video_s` | Time to encode the output video via ffmpeg |
| `total_s` | End-to-end wall time |
| `output_fps` | Processing speed (`num_frames / total_s`) |
| `run_total_s` | Total wall time including overhead (Modal runs) |

### Coverage metrics (video mode)

| Metric | Description |
|--------|-------------|
| `num_frames` | Total frames in the video |
| `input_fps` | Original video framerate |
| `duration_s` | Video duration in seconds |
| `frames_with_masks` | Frames where at least one object has a non-empty mask |
| `frames_with_valid_objects` | Frames with at least one valid fitted quad |
| `frames_with_quads` | Frames where at least one quad passed quality validation |
| `frames_composited` | Frames where compositing was performed |
| `object_masks_total` | Total object-mask pairs across all frames |
| `first_frame_with_mask` / `last_frame_with_mask` | Tracking extent |
| `max_consecutive_mask_gap` | Longest stretch without any masks |
| `object_frame_coverage` | Per-object mask presence ratio |
| `object_valid_frame_coverage` | Per-object valid quad ratio |
| `object_rejection_counts` / `object_rejection_reasons` | Per-object fit rejection breakdown |

### Geometry metrics

| Metric | Description |
|--------|-------------|
| `geometry_config_enabled` / `geometry_runtime_enabled` | Whether geometry was configured and actually ran |
| `geometry_total_s` | Total geometry fitting time |
| `geometry_active_objects` | Object IDs using geometry-constrained fitting |
| `geometry_frames_held` | Frames where hold-last-good was used |
| `geometry_fallback_frames` | Frames where mask-free-quad fallback was used |
| `vp_width_valid_ratio` / `vp_depth_valid_ratio` | Fraction of frames with confident VP estimates |
| `court_width_candidate_count` / `court_depth_candidate_count` | Average detected court lines per frame |
| `object_geometry_model` | Per-object geometry model assignment |
| `geometry_object_jitter_stats` | Per-object corner jitter (median and p95 RMS in px) |
| `geometry_fit_method_counts` | Per-object breakdown of fit methods used |

### Stabilization metrics

| Metric | Description |
|--------|-------------|
| `stabilization_total_s` | Total stabilization time |
| `stabilization_static_frame_ratio` | Fraction of frames classified as near-static |
| `stabilization_frames_held` / `stabilization_frames_blended` / `stabilization_frames_raw_accepted` | Aggregate action counts |
| `stabilization_object_stats` | Per-object breakdown (held, blended, raw, dropped, max empty hold streak) |

### Compositor metrics (temporal_rectified)

| Metric | Description |
|--------|-------------|
| `compositor_total_s` | Total compositing time |
| `compositor_object_model` | Per-object compositor model (`temporal_wall_plate`, `temporal_court_plate`, `delegated_inpaint`) |
| `compositor_object_stats` | Per-object stats (plate init frame, reuse count, reset count, delegated frames, shading updates) |

### Preview diagnostics (image mode)

| Metric | Description |
|--------|-------------|
| `preview_ok` | Whether all objects passed preview validation |
| `preview_failure_reasons` | List of failure descriptions |
| `preview_objects_with_masks` / `preview_objects_with_quads` | Object counts |
| `preview_object_diagnostics` | Per-object detailed diagnostics (mask area, bbox, fit status, fit method, geometry flags, composite status, background fill analysis) |

### Reproducibility metadata (Modal runs)

| Metric | Description |
|--------|-------------|
| `gpu` / `gpu_memory_gb` | GPU name and VRAM |
| `git_branch` / `git_commit_sha` / `git_dirty` | Git state at run time |
| `workspace_diff_sha256` | Hash of uncommitted changes |
| `frozen_config_path` / `frozen_config_sha256` | Frozen config path and hash |

Example output (video mode):

```json
{
  "gpu": "Tesla T4",
  "gpu_memory_gb": 14.6,
  "mode": "video",
  "num_frames": 202,
  "input_fps": 25.0,
  "segment_total_s": 95.68,
  "fit_mean_ms": 10.25,
  "composite_mean_ms": 202.53,
  "write_video_s": 2.54,
  "total_s": 141.21,
  "output_fps": 1.43
}
```

## Experiments and reproducibility

Each run saves to `experiments/<timestamp>_<name>/`:

```
experiments/2026-04-07_20-38-28_pca_T4/
  config.yaml      # frozen config with exact click coordinates + all settings
  metrics.json      # timing, FPS, GPU info, coverage, geometry, stabilization
  outputs/
    composited.mp4   # output video (or .png for image mode)
    preview_prompts.png    # annotated prompt markers (SAM3 image mode)
    preview_masks.png      # annotated mask overlay (SAM3 image mode)
    preview_geometry.png   # court line + VP overlay (when geometry enabled)
    compositor_rectified_obj_*.png  # rectified plate triptych (temporal_rectified)
```

Everything is tracked in git — configs, metrics, and outputs. For long videos that exceed GitHub's file size limit, the output will be rejected by git; in that case, just add the specific output to `.gitignore` and let teammates reproduce it from the saved config:

```bash
# Reproduce an experiment exactly
uv run modal run scripts/modal_run.py --config experiments/2026-04-07_20-38-28_pca_T4/config.yaml --gpu T4

# Reuse same coordinates with different settings
cp experiments/2026-04-07_20-38-28_pca_T4/config.yaml configs/experiments/my_test.yaml
# edit fitter.type, compositor.type, etc.
uv run modal run scripts/modal_run.py --config configs/experiments/my_test.yaml --gpu A100 --mode video
```

## Benchmarking

### Benchmarking across GPUs

Single config, single GPU, multiple averaged runs:

```bash
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --mode video --benchmark 5
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu A100 --mode video --benchmark 5
```

When `--benchmark N` with N > 1, numeric metrics are aggregated as `{mean, std, min, max}` objects in the report.

### Benchmark matrix (multiple prompt counts × multiple GPUs)

For systematic comparison, use the matrix runner. It executes every (config, GPU) combination and saves each as its own experiment directory.

**Step 1: Set up configs in `configs/matrix/`**

The repo ships with SAM2 and SAM3 templates that use the same input video but different numbers of tracked objects:

- `configs/matrix/1prompt.yaml`, `configs/matrix/5prompts.yaml`, `configs/matrix/11prompts.yaml`
- `configs/matrix/sam3_1prompt.yaml`, `configs/matrix/sam3_5prompts.yaml`, `configs/matrix/sam3_11prompts.yaml`

You can reuse the shipped prompts as-is, or recollect them for either SAM2 or SAM3:

```bash
uv run python scripts/collect_prompts.py --config configs/matrix/1prompt.yaml
uv run python scripts/collect_prompts.py --config configs/matrix/5prompts.yaml
uv run python scripts/collect_prompts.py --config configs/matrix/sam3_1prompt.yaml
uv run python scripts/collect_prompts.py --config configs/matrix/sam3_5prompts.yaml
```

The SAM3 matrix templates now use sparse positive/negative click seeds instead of SAM2-style outline prompts. If you recollect SAM3 prompts, preview them with `--mode image` before launching the full matrix run.

You can also create your own matrix configs (different videos, fitters, compositors, etc.) — just `cp` an existing one and edit.

**Step 2: Run the matrix**

Two options:

```bash
# Sequential — runs one at a time, simple output
./scripts/run_matrix.sh

# Parallel — runs all combinations simultaneously, ~10x faster
uv run python scripts/run_matrix_parallel.py

# SAM3 matrix example
uv run python scripts/run_matrix_parallel.py \
  --configs configs/matrix/sam3_1prompt.yaml configs/matrix/sam3_5prompts.yaml configs/matrix/sam3_11prompts.yaml \
  --gpus A100 H100 B200
```

Defaults: `T4 A100 H100 B200` × 3 configs × `--benchmark 3` = 12 jobs.

If a config uses `sam3_video`, any `T4` pairing is skipped before remote execution starts. The valid SAM3 jobs still run.

**Modal concurrency limit:** Starter accounts have a limit of 10 concurrent GPUs. Throttle the parallel runner accordingly:

```bash
uv run python scripts/run_matrix_parallel.py --max-parallel 10
```

Excess jobs queue automatically and start as soon as a slot frees up. All combinations still run.

**Customize the matrix:**

```bash
# Run only specific GPUs
uv run python scripts/run_matrix_parallel.py --gpus T4 A100

# Run only specific configs
uv run python scripts/run_matrix_parallel.py --configs configs/matrix/1prompt.yaml configs/matrix/11prompts.yaml

# Lower benchmark count for quick test
uv run python scripts/run_matrix_parallel.py --benchmark 1
```

Each combination produces an experiment directory named `<config>_<gpu>` (e.g. `5prompts_A100`), so they're easy to compare.

## Available GPUs

Pass any of these to `--gpu`:

| GPU | VRAM | Cost/hr |
|-----|------|---------|
| `T4` | 16 GB | $0.59 |
| `L4` | 24 GB | $0.80 |
| `A10G` | 24 GB | $1.10 |
| `L40S` | 48 GB | $1.95 |
| `A100` | 40 GB | $2.10 |
| `A100-80GB` | 80 GB | $2.50 |
| `H100` | 80 GB | $3.95 |
| `H200` | 141 GB | $4.54 |
| `B200` | 192 GB | $6.25 |

### SAM3 GPU and FlashAttention support

SAM3 requires FlashAttention. The Modal build image selects the correct FlashAttention version based on the GPU:

| GPU | SAM support | FlashAttention | Modal image |
|-----|-------------|----------------|-------------|
| `T4` | SAM2 only | Not supported (sm75) | `t4_image` |
| `L4`, `A10G`, `L40S`, `A100`, `A100-80GB`, `H100`, `H200` | SAM2 + SAM3 | FlashAttention-2 | `fa2_image` |
| `B200` | SAM2 + SAM3 | FlashAttention-4 | `fa4_image` |

The attention backend is selected at runtime by `sam3_attention.py`, which inspects the GPU name and CUDA compute capability to choose between FA2 and FA4 wrappers. The wrappers patch SAM3's internal attention modules (`sam3.perflib.fa3`, `sam3.model.vitdet`, and optionally `sam3.perflib.fa2`).

As of April 10, 2026, PyPI only publishes `flash-attn-4` as prereleases, so the Modal B200 image pins `flash-attn-4==4.0.0b8` instead of relying on pip to resolve a final release. See [PyPI](https://pypi.org/project/flash-attn-4/) and the [upstream README](https://github.com/Dao-AILab/flash-attention).

### SAM3 checkpoint download

SAM2 checkpoints are downloaded from Meta's CDN. SAM3 checkpoints are downloaded from HuggingFace Hub (requires `HF_TOKEN` environment variable or cached credentials). The `modal_run.py` launcher caches downloaded checkpoints in a Modal persistent volume (`banner-pipeline-checkpoints`) to avoid re-downloading on every run.

## Tests

```bash
# Run the full test suite
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_court_geometry.py -v
```

Test coverage:

| Test file | What it covers |
|-----------|----------------|
| `test_court_geometry.py` | Court line detection, VP estimation, geometry engine, hold/fallback, surface type routing |
| `test_stabilization.py` | Optical flow stabilization, mask fusion, hold/blend/raw decisions |
| `test_pipeline_prompts.py` | Prompt loading, validation, legacy SAM2 prompt detection, duplicate/near-identical click rejection |
| `test_pipeline_video.py` | Video pipeline integration, coverage validation, compositor metrics enforcement |
| `test_prompt_workflows.py` | End-to-end prompt workflows, surface type filtering, geometry model resolution |
| `test_modal_run_routing.py` | GPU↔image routing, SAM3+T4 rejection, config validation |
| `test_sam3_video.py` | SAM3 video segmenter response parsing, reanchoring, mask extraction |
| `test_sam3_attention.py` | FlashAttention backend selection, FA2/FA4 wrapper logic, GPU family detection |
| `test_sam3_loader.py` | SAM3 model builder detection, signature inspection |
| `test_alpha_compositor.py` | Alpha compositor, border fill estimation |
| `test_temporal_rectified_compositor.py` | Temporal rectified compositor, plate caching, shading, state reset |
| `test_reporting.py` | Metrics report aggregation, benchmark averaging |

## Adding new components

### Adding a new segmentation model

1. Create `src/banner_pipeline/segment/my_model.py`
2. Implement the `SegmentationModel` interface (see `segment/base.py`):
   ```python
   class MySegmenter(SegmentationModel):
       def segment(self, frame_bgr, prompts) -> dict[int, np.ndarray]: ...
       @property
       def name(self) -> str: ...
   ```
3. Register it in `pipeline.py`: `SEGMENTERS["my_model"] = MySegmenter`
4. Set `segmenter.type: my_model` in your config

### Adding a new quad fitter

1. Create `src/banner_pipeline/fitting/my_fitter.py`
2. Implement the `QuadFitter` interface (see `fitting/base.py`):
   ```python
   class MyFitter(QuadFitter):
       def fit(self, mask, **kwargs) -> np.ndarray | None: ...
       @property
       def name(self) -> str: ...
   ```
   The `fit` method returns a `(4, 2)` float32 array ordered `[TL, TR, BR, BL]`, or `None` on failure.
3. Register it in `pipeline.py`: `FITTERS["my_fitter"] = MyFitter`
4. Set `fitter.type: my_fitter` in your config

### Adding a new compositor

1. Create `src/banner_pipeline/composite/my_compositor.py`
2. Implement the `Compositor` interface (see `composite/base.py`):
   ```python
   class MyCompositor(Compositor):
       def composite(self, frame, corners, overlay, mask=None, **kwargs) -> np.ndarray: ...
       @property
       def name(self) -> str: ...
   ```
3. Register it in `pipeline.py`: `COMPOSITORS["my_compositor"] = MyCompositor`
4. Set `compositor.type: my_compositor` in your config
