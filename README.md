# Banner Pipeline — Virtual Ad Insertion in Tennis Broadcasts

Capstone hand-off. SAM2-based banner / logo replacement on real broadcast footage: detects placement regions, tracks them across all frames, fits perspective-aware quadrilaterals via court-plane homography, and composites new logos with correct aspect ratio, brightness response, and player occlusion.

## What's in this repo

This is the final commit of a capstone project on virtual ad insertion. The Melbourne walkover demo clip is the canonical test case — a 13-second broadcast clip with five simultaneous virtual ad placements (3 back banners, 1 left side banner, 1 court-floor Red Bull walkover logo) over a player walking across the court.

**Final delivered output:** `experiments/2026-05-05_18-38-39_hull_H200/outputs/composited.mp4`

**Recipe** (config `configs/experiments/eval_walkover_p3_a1_ball_tracker_net_v1.yaml`):
- V68 manually-clicked court corners as the seed homography
- BallTrackerNet learned 14-keypoint detector for per-frame court geometry estimation
- Hybrid lock at 30-px tolerance — stays pixel-locked at the seed when the camera is static, ramps to the BTN estimate when motion exceeds tolerance
- V68's compositor settings (median_fill inpaint, LED brightness re-baking, MatAnyone2 person-mask occlusion)

**Side-by-side vs the V68 static-clicked gold:** `experiments/2026-05-05_18-38-39_hull_H200/eval/vs_reference_side_by_side.mp4`

## Where to start reading

| You want to | Read |
|---|---|
| Read the canonical project narrative — problem, approach, every phase, why the final approach won | **`docs/FINAL_REPORT.md`** |
| Reproduce the final result | "Reproduce the final" below |
| Understand the eval framework (gates, walkover detection, side-by-side video) | `docs/EVALUATION.md` |
| Read the raw append-only experiment log | `docs/EXPERIMENT_LEDGER.md` |
| Continue the autonomous experimentation work | `docs/AGENT_BRIEFING.md` |

## Reproduce the final

```bash
# 1. Install dependencies (uv: https://docs.astral.sh/uv/)
uv sync

# 2. Authenticate Modal (one-time)
uv run modal setup

# 3. Run the final config on H200
uv run modal run scripts/modal_run.py \
    --config configs/experiments/eval_walkover_p3_a1_ball_tracker_net_v1.yaml \
    --gpu H200 --mode video_hybrid

# 4. Score the deterministic eval framework
uv run python -m banner_pipeline.eval \
    --experiment experiments/<your_run_dir>/ --reference auto
```

This produces `outputs/composited.mp4`, `eval/quality_metrics.json`, `eval/report.md`, per-region crop strips, walkover forensic sheets, and a side-by-side regression video against the V68 gold.

## Setup

```bash
# Clone and enter the repo
git clone <repo-url> && cd homography-fitting

# Install all dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Authenticate with Modal (one-time, for GPU runs)
uv run modal setup
```

SAM2 is built from source automatically on the Modal worker. Local runs need it locally:

```bash
# Only if running locally (not needed for Modal)
git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
```

## Running the pipeline (general)

Two-step process: collect clicks locally on a chosen seed frame, then run on a remote GPU via Modal.

### Step 1 — Select banner regions (local, no GPU)

```bash
uv run python scripts/collect_prompts.py --config configs/default.yaml
```

This opens the seed frame and saves the prompt points into the config automatically. SAM2: left-click positive points. SAM3: left-click positive, right-click negative, `U` to undo, `N` for next object.

### Step 2 — Run on a Modal GPU

```bash
# Video mode (full clip)
uv run modal run scripts/modal_run.py --config <config>.yaml --gpu H200 --mode video_hybrid

# Image mode (single frame preview)
uv run modal run scripts/modal_run.py --config <config>.yaml --gpu H200 --mode image
```

For walking-over renders (long clips with player occlusion), use `H200` or `B200` and `--mode video_hybrid`. For preview / debugging, `--mode image` produces a single composited PNG.

### Available GPUs

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

`T4` does not support SAM3 (FlashAttention requirement). `B200` requires FlashAttention-4 (`flash-attn-4==4.0.0b8` pinned in the Modal image as of April 2026).

## Repository layout

```
homography-fitting/
  README.md                               ← you are here
  CHANGELOG.md                            milestone log
  docs/
    FINAL_REPORT.md                       canonical narrative (start here)
    EVALUATION.md                         eval framework spec
    EXPERIMENT_LEDGER.md                  append-only experiment log
    AGENT_BRIEFING.md                     autonomous worker contract (internal)
  src/banner_pipeline/
    pipeline.py                           orchestration
    segment/sam2_image.py, sam2_video.py  segmentation
    fitting/hull_fit.py, lp_fit.py, …     quad fitters (final = hull)
    court_geometry.py                     classical_lines_v1 + HybridLockState
    court_geometry_ball_tracker.py        ball_tracker_net_v1 (FINAL)
    composite/painted.py                  inpaint compositor + LED-blend
    eval/                                 deterministic eval framework
      report.py, walkover.py, contact_sheets.py, side_by_side.py
  scripts/
    collect_prompts.py, modal_run.py      main workflow
    run_pipeline.py, run_experiment.py    local alternatives
  configs/
    default.yaml                          starter SAM2 config
    sam3_default.yaml, sam3_court_eval.yaml
    experiments/                          268 experiment configs
      eval_walkover_v68_*.yaml            Phase 1 / 2 (clicked / dynamic)
      eval_walkover_p3_a1_*.yaml          Phase 3 BTN port (FINAL)
      …
    eval/reference.yaml                   gold-mapping for --reference auto
    matrix/                               benchmark matrix templates
  data/                                   test clips
  experiments/                            timestamped runs (configs + outputs + eval)
    2026-05-05_18-38-39_hull_H200/        ← FINAL deliverable
  tests/                                  pytest suite
```

## Pipeline overview

End-to-end:

```
data/<input>.mov
  → SAM2 segmenter      (segment/sam2_image.py)
  → hull quad fitter    (fitting/hull_fit.py)
  → court geometry      (court_geometry_ball_tracker.py — FINAL)
  → hybrid lock         (court_geometry.py:HybridLockState)
  → MatAnyone2 mask     (occlusion alpha matting)
  → inpaint compositor  (composite/painted.py — median_fill + LED-blend)
  → outputs/composited.mp4
```

For details on each module and why each was chosen, see `docs/FINAL_REPORT.md` §3.

## Configuration

All behaviour is config-driven. To change algorithms, edit the YAML:

```yaml
pipeline:
  fitter:
    type: hull          # pca | lp | hull
  compositor:
    type: inpaint        # alpha | inpaint
  geometry:
    enabled: true
    court_backend: ball_tracker_net_v1   # classical_lines_v1 | ball_tracker_net_v1
    hybrid_lock:
      enabled: true
      tolerance_px: 30.0
```

| Fitter | Algorithm | Best for |
|--------|-----------|----------|
| `pca` | Weighted PCA with Hann windows | Rectangular banners |
| `lp` | Linear programming supporting lines | Tight convex bounds |
| `hull` | Hull vertex deduction | Regions extending off-screen (FINAL) |

| Court backend | Source | Notes |
|---|---|---|
| `classical_lines_v1` | `court_geometry.py` | Default. Hough-line detector. Phase 2 found this too noisy to gate on dynamically. |
| `ball_tracker_net_v1` | `court_geometry_ball_tracker.py` | FINAL. Learned 14-keypoint detector. Stable enough for `hybrid_lock` at `tolerance_px: 30`. |

## Evaluation framework

Every run can be scored post-hoc by the deterministic eval framework:

```bash
uv run python -m banner_pipeline.eval \
    --experiment experiments/<run_dir>/ \
    [--reference auto] \
    [--regions back,left,floor,full,walkover] \
    [--walkover-window 690:745]
```

Exit codes: `0` = pass, `2` = scorecard fail, `3` = pass-but-regression-vs-gold, `1` = framework error.

Per-region hard gates (defined in `src/banner_pipeline/eval/report.py`):
- `corner_max_jump_px < 2.0`, `corner_accel_p95_px < 1.0`, `quad_area_cv < 0.05`
- `roi_jitter_ratio ≤ 1.05`, `roi_temporal_ssim_mean > 0.95`
- Floor-only walkover: `walkover_logo_visible_pct > 0.10`, `walkover_occlusion_iou > 0.80`

Outputs: `eval/quality_metrics.json` (machine-readable, schema-versioned), `eval/report.md` (human rollup), per-region crop strips and motion strips (PNG), walkover forensic sheets (PNG), and a side-by-side regression video against the gold reference (MP4, when `--reference auto`).

Adding a new clip = one entry in `configs/eval/reference.yaml` mapping the input-video basename to its gold dir. No code change.

Full spec: `docs/EVALUATION.md`. Final-run metrics with discussion: `docs/FINAL_REPORT.md` §7.

## Metrics (per-run timing)

Each run also writes a top-level `metrics.json` with timing / GPU info:

```json
{
  "gpu": "Tesla H200",
  "gpu_memory_gb": 139.8,
  "mode": "video_hybrid",
  "num_frames": 767,
  "input_fps": 59.0,
  "segment_total_s": 95.68,
  "fit_mean_ms": 10.25,
  "composite_mean_ms": 202.53,
  "write_video_s": 2.54,
  "total_s": 286.5,
  "output_fps": 2.68
}
```

| Metric | Description |
|--------|-------------|
| `num_frames` | Total frames in the video |
| `input_fps` | Original framerate |
| `segment_total_s` | Time for SAM video tracker to segment + track |
| `fit_mean_ms` | Average per-frame quad fit time |
| `composite_mean_ms` | Average per-frame composite time |
| `write_video_s` | Video encoding |
| `total_s` | End-to-end wall time |
| `output_fps` | `num_frames / total_s` |

## Reproducibility

Each run saves `experiments/<timestamp>_<name>_<gpu>/` with:
- `config.yaml` — frozen config with exact click coordinates and all settings
- `metrics.json` — timing + GPU info
- `outputs/composited.mp4` — output
- `eval/` — full eval framework output (after running `python -m banner_pipeline.eval`)

To reproduce any run exactly: `uv run modal run scripts/modal_run.py --config <run_dir>/config.yaml --gpu H200 --mode video_hybrid`.

To explore variants: `cp <run_dir>/config.yaml configs/experiments/my_test.yaml`, edit, then run.

## Known limits and future work

The final has three known ceilings, documented in `docs/FINAL_REPORT.md` §9:

1. **Texture-match.** The smoothed inpaint micro-grain is visible vs the gritty real court paint at close zoom. Would need real texture transfer (noise injection / GAN-based inpaint).
2. **Single-clip eval.** Only `melbourne-walking-over-logo.mov` is wired into `configs/eval/reference.yaml`. Adding `data/tennis-clip.mp4` and `data/zoom-clip-melbourne.mov` would catch clip-specific regressions.
3. **Adaptive vp_smoothing.** Code shipped (P3-A2) but the parameter sweep didn't conclude. Worth completing.

## Internal-only experimentation framework

For the autonomous-worker iteration loop (Phase 3 produced ~50 H200 GPU runs across 14 waves of self-experimenting agents): see `docs/AGENT_BRIEFING.md`. Defines the per-cycle worker contract, parallelism patterns, and the "lessons learned" knowledge-sharing protocol. Internal — not part of the production pipeline.

## Adding a new segmentation model

1. Create `src/banner_pipeline/segment/sam3_image.py`
2. Implement the `SegmentationModel` interface (see `segment/base.py`)
3. Register it in `pipeline.py`: `SEGMENTERS["sam3"] = SAM3ImageSegmenter`
4. Set `segmenter.type: sam3` in your config

## Benchmarking across GPUs

Single config across multiple averaged runs:

```bash
uv run modal run scripts/modal_run.py --config <config>.yaml --gpu H200 --mode video --benchmark 5
```

Matrix runner for systematic (config, GPU) comparison:

```bash
# Sequential
./scripts/run_matrix.sh

# Parallel (respects 10-concurrent-GPU Modal limit)
uv run python scripts/run_matrix_parallel.py --max-parallel 10
```

Templates in `configs/matrix/`. Each combination produces an experiment dir named `<config>_<gpu>` for easy comparison.

## License

Capstone submission. Not yet released open-source.
