# Banner Pipeline

Video banner/logo replacement using SAM2 segmentation. Detects billboard regions in video frames, fits perspective-aware quadrilaterals, and composites new logos with correct aspect ratio and luminosity matching.

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd homography-fitting

# 2. Install all dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Authenticate with Modal (one-time, for GPU runs)
modal setup
```

SAM2 setup is only needed for **local** runs. Modal downloads checkpoints automatically.

```bash
# Only if running locally (not needed for Modal)
git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
```

## Pipeline stages

```
Input frame → [Segment] → [Fit quad] → [Composite] → Output frame
                SAM2        PCA/LP/Hull   Inpaint/Alpha
```

Each stage is swappable via the YAML config.

## Running on Modal (GPU)

Two-step process: collect clicks locally, then run on a remote GPU.

```bash
# Step 1: Click on banner regions (local, no GPU needed)
python scripts/collect_prompts.py --config configs/default.yaml

# Step 2: Run on a GPU via Modal
modal run scripts/modal_run.py --config configs/default.yaml --gpu T4
```

Available GPUs: `T4` (cheapest), `L4`, `A10G`, `A100`, `H100`.

Benchmark across GPU tiers:

```bash
modal run scripts/modal_run.py --config configs/default.yaml --gpu T4 --benchmark 5
modal run scripts/modal_run.py --config configs/default.yaml --gpu A10G --benchmark 5
```

Results are saved to `experiments/<timestamp>_<name>/` with `metrics.json`.

## Running locally

```bash
# Interactive (opens UI for clicking + runs SAM2 locally)
python scripts/run_pipeline.py --config configs/default.yaml --save result.png

# Run experiment with saved outputs + metrics
python scripts/run_experiment.py --config configs/default.yaml

# Benchmark FPS locally
python scripts/benchmark_fps.py --config configs/default.yaml --runs 5
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
scripts/      run_pipeline.py, run_experiment.py, benchmark_fps.py,
              collect_prompts.py, modal_run.py
```

See [MIGRATION.md](MIGRATION.md) for how the old files map to this structure.
