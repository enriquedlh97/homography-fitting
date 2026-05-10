# Branch: `feat/sam3-v2`

## Goal

Integrate **SAM3** into the banner-replacement pipeline to remove the manual logo-selection step. Logos are detected automatically from a text prompt (e.g. `"logo"`), then segmented, tracked, and composited like in the SAM2 pipeline.

## What changed vs. `main`

- New segmenter type `sam3_video` driven by text prompts.
- New configs:
  - `configs/experiments/sam3_auto.yaml` — generic auto-detection config.
  - `configs/experiments/sam3_auto_zoom.yaml` — tuned for `data/zoom_in_camera_change.mp4` (zoom + camera change).
- New runner: `scripts/modal_run_sam3.py`.
- Detection filtering, EMA tracking, and hybrid stabilization wired into the SAM3 path.

## How to run

All experiments were executed on **A100-80GB** via Modal.

```bash
# Generic auto-detection
modal run scripts/modal_run_sam3.py --config configs/experiments/sam3_auto.yaml --mode video

# Zoom + camera-change scenario
modal run scripts/modal_run_sam3.py --config configs/experiments/sam3_auto_zoom.yaml --mode video
```

Outputs land in `experiments/<timestamp>_sam3_pca_A100-80GB/outputs/composited.mp4`.

## Results

| Scenario | Run | Detected | Segmented | Output FPS |
|---|---|---|---|---|
| Standard clip | `2026-04-29_10-11-10_sam3_pca_A100-80GB` | — | 21 | **0.93** |
| Zoom + camera change | `2026-05-04_23-41-58_sam3_pca_A100-80GB` | 38 | 27 | **0.99** |

## Takeaway

Detection quality is acceptable, but throughput is low (~1 fps). The lighter variant on `feat/sam3-light-v1` reaches up to **~4.09 fps** on the same hardware with comparable quality, so the light version is currently the preferred path forward. This branch is kept as the reference full-quality implementation.
