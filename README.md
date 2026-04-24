# Banner Pipeline

Video banner/logo replacement using SAM2 or SAM3.1 segmentation. Detects billboard regions in video frames, tracks them across all frames, fits perspective-aware quadrilaterals, and composites new logos with correct aspect ratio and luminosity matching.

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

For SAM3, use `--mode image` first to preview the prompt-stage masks and geometry-constrained quads on the selected frame. If the preview looks wrong, or the preview metrics do not include `geometry_*`, adjust the clicks or config before running `--mode video`.
The shipped SAM3 parity configs use `inpaint`, which is the same compositor family as the stronger SAM2 baseline.

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

# 4. Confirm the preview metrics include geometry_* and show the intended fit method

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
- `configs/sam3_default.yaml` is the wall-banner config. `back_wall_banner` objects now use a fronto-parallel wall solver, while `side_wall_banner` keeps the VP-constrained path when court geometry is confident.
- `configs/sam3_court_eval.yaml` is the court-plane validation config for `court_marking` prompts. Use it to verify that `court_plane` is active before mixing court ads into a larger run.

### If SAM3 Preview Fails

- Try a different seed frame with `--frame 10`, `--frame 20`, or another clearer frame.
- Use 2 positive clicks plus 1 negative click instead of a single positive click.
- Reduce the test to one object and verify that first.
- If the log shows `usable_outputs=False parsed_nonempty_masks=0`, interpret it as:
  the prompt request was accepted, but SAM3 returned no usable mask for that preview frame.
- If the preview metrics are missing `geometry_*` or `stabilization_*` despite those features being enabled in the config, treat the run as invalid. The pipeline now fails loudly for that case instead of silently saving a contour-only experiment.

### Current SAM3 Preview Limitation

`--mode image` returns a single composited preview image, but the current SAM3 implementation still loads the extracted frame set to initialize the predictor session. In other words, it is a preview of the selected frame's output, not yet a truly cheap first-frame-only execution path.

### Available GPUs

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

SAM3 GPU support:

- `T4`: `SAM2` only
- `L4`, `A10G`, `L40S`, `A100`, `A100-80GB`, `H100`, `H200`: `SAM3` via FlashAttention-2
- `B200`: `SAM3` via FlashAttention-4

As of April 10, 2026, PyPI only publishes `flash-attn-4` as prereleases, so the Modal B200 image pins `flash-attn-4==4.0.0b8` instead of relying on pip to resolve a final release. See [PyPI](https://pypi.org/project/flash-attn-4/) and the [upstream README](https://github.com/Dao-AILab/flash-attention).

### Benchmarking across GPUs

Single config, single GPU, multiple averaged runs:

```bash
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --mode video --benchmark 5
uv run modal run scripts/modal_run.py --config configs/default.yaml --gpu A100 --mode video --benchmark 5
```

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

## Metrics

Each run produces a `metrics.json` in the experiment directory. Example output (video mode, T4):

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

| Metric | Description |
|--------|-------------|
| `num_frames` | Total frames in the video |
| `input_fps` | Original video framerate |
| `segment_total_s` | Time for the configured SAM video tracker to segment and track objects across all frames |
| `fit_mean_ms` | Average time to fit a quad per frame |
| `composite_mean_ms` | Average time to composite logo per frame |
| `write_video_s` | Time to encode the output video |
| `total_s` | End-to-end wall time |
| `output_fps` | Processing speed (`num_frames / total_s`) — compare this to `input_fps` to gauge how far from real-time |

## Experiments and reproducibility

Each run saves to `experiments/<timestamp>_<name>/`:

```
experiments/2026-04-07_20-38-28_pca_T4/
  config.yaml      # frozen config with exact click coordinates + all settings
  metrics.json      # timing, FPS, GPU info
  outputs/
    composited.mp4   # output video (or .png for image mode)
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

## Running locally

```bash
# Interactive (opens UI for clicking + runs SAM2 locally)
uv run python scripts/run_pipeline.py --config configs/default.yaml --save result.png

# Run experiment with saved outputs + metrics
uv run python scripts/run_experiment.py --config configs/default.yaml
```

## Swapping components

Change the config to use different algorithms:

```yaml
pipeline:
  fitter:
    type: lp           # pca | lp | hull
  compositor:
    type: alpha         # inpaint | alpha
```

| Fitter | Algorithm | Best for |
|--------|-----------|----------|
| `pca` | Weighted PCA with Hann windows | Rectangular banners |
| `lp` | Linear programming supporting lines | Tight convex bounds |
| `hull` | Hull vertex deduction | Regions extending off-screen |

## Adding a new segmentation model

1. Create `src/banner_pipeline/segment/sam3_image.py`
2. Implement the `SegmentationModel` interface (see `segment/base.py`)
3. Register it in `pipeline.py`: `SEGMENTERS["sam3"] = SAM3ImageSegmenter`
4. Set `segmenter.type: sam3` in your config

## Project structure

```
src/banner_pipeline/
  io.py, device.py, geometry.py, ui.py, viz.py   # shared utilities
  segment/    sam2_image.py, sam2_video.py         # segmentation models
  fitting/    pca_fit.py, lp_fit.py, hull_fit.py   # quad fitting algorithms
  homography/ camera.py, court.py                  # camera intrinsics
  composite/  inpaint.py, alpha.py                 # compositing strategies
  pipeline.py                                      # orchestration + config
configs/      default.yaml, experiments/            # experiment configs
scripts/      collect_prompts.py, modal_run.py,     # main workflow
              run_pipeline.py, run_experiment.py,    # local alternatives
              benchmark_fps.py
```

See [MIGRATION.md](MIGRATION.md) for how the old files map to this structure.
