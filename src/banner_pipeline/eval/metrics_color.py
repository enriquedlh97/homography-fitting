"""Color, jitter-ratio, noise-variance, and edge-sharpness metrics.

All region-parameterized: ROI is supplied as (x0, y0, x1, y1) in image
coordinates. Math for jitter_ratio is lifted from
`scripts/quality_eval.py:40-80` and generalized to any ROI.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Jitter ratio: composite ROI mean abs frame-diff vs original ROI same metric
# ---------------------------------------------------------------------------


def roi_jitter_ratio(
    composite_path: str,
    original_path: str,
    roi: tuple[int, int, int, int],
) -> dict[str, float]:
    """Frame-to-frame mean abs pixel diff inside ROI, composite vs original.

    Returns ratio = composite_mean / original_mean. <= 1.05 means the
    composite is no jumpier than the underlying footage in that region.
    """
    x0, y0, x1, y1 = roi
    comp_mean = _mean_frame_diff(composite_path, y0, y1, x0, x1)
    orig_mean = _mean_frame_diff(original_path, y0, y1, x0, x1)
    ratio = comp_mean / orig_mean if orig_mean > 0 else 0.0
    return {
        "roi_jitter_composite_mean": round(comp_mean, 4),
        "roi_jitter_original_mean": round(orig_mean, 4),
        "roi_jitter_ratio": round(ratio, 4),
    }


def _mean_frame_diff(video_path: str, y0: int, y1: int, x0: int, x1: int) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    prev = None
    diffs: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = frame[y0:y1, x0:x1].astype(np.float32)
        if prev is not None and cur.shape == prev.shape:
            diffs.append(float(np.abs(cur - prev).mean()))
        prev = cur
    cap.release()
    return float(np.mean(diffs)) if diffs else 0.0


# ---------------------------------------------------------------------------
# ROI vs neighbor-patch ΔE in Lab space (surface-agnostic color match)
# ---------------------------------------------------------------------------


def roi_delta_e_vs_neighbor(
    composite_path: str,
    roi: tuple[int, int, int, int],
    neighbor_roi: tuple[int, int, int, int],
    n_samples: int = 10,
) -> dict[str, float]:
    """Mean Lab ΔE between the placed-region average color and a neighbor patch.

    Surface-agnostic — works for the dark back banner AND the green court floor,
    unlike the legacy `compute_inpaint_color_match` which assumed dark pixels.
    """
    cap = cv2.VideoCapture(composite_path)
    if not cap.isOpened():
        return {"roi_delta_E_lab": 0.0}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return {"roi_delta_E_lab": 0.0}
    indices = np.linspace(0, max(n - 1, 0), n_samples, dtype=int)
    deltas: list[float] = []
    rx0, ry0, rx1, ry1 = roi
    nx0, ny0, nx1, ny1 = neighbor_roi
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        roi_patch = frame[ry0:ry1, rx0:rx1]
        neigh_patch = frame[ny0:ny1, nx0:nx1]
        if roi_patch.size == 0 or neigh_patch.size == 0:
            continue
        lab_r = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_n = cv2.cvtColor(neigh_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta = float(np.sqrt(((lab_r.mean(axis=(0, 1)) - lab_n.mean(axis=(0, 1))) ** 2).sum()))
        deltas.append(delta)
    cap.release()
    return {"roi_delta_E_lab": round(float(np.mean(deltas)) if deltas else 0.0, 3)}


# ---------------------------------------------------------------------------
# Noise variance ratio ("too clean" warning detector)
# ---------------------------------------------------------------------------


def roi_noise_variance_ratio(
    composite_path: str,
    roi: tuple[int, int, int, int],
    neighbor_roi: tuple[int, int, int, int],
    n_samples: int = 10,
) -> dict[str, float]:
    """Luminance variance(ROI) / variance(neighbor patch).

    A real surface has texture. If the placed region's variance is much lower
    than the neighboring patch's (ratio < 0.3), the logo looks pasted-on /
    too clean. Computed on the high-frequency residual (image - blur) so
    overall brightness doesn't dominate.
    """
    cap = cv2.VideoCapture(composite_path)
    if not cap.isOpened():
        return {"noise_variance_ratio": 1.0}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return {"noise_variance_ratio": 1.0}
    indices = np.linspace(0, max(n - 1, 0), n_samples, dtype=int)
    ratios: list[float] = []
    rx0, ry0, rx1, ry1 = roi
    nx0, ny0, nx1, ny1 = neighbor_roi
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi_patch = gray[ry0:ry1, rx0:rx1]
        neigh_patch = gray[ny0:ny1, nx0:nx1]
        if roi_patch.size == 0 or neigh_patch.size == 0:
            continue
        roi_res = roi_patch - cv2.GaussianBlur(roi_patch, (7, 7), 0)
        neigh_res = neigh_patch - cv2.GaussianBlur(neigh_patch, (7, 7), 0)
        roi_var = float(roi_res.var())
        neigh_var = float(neigh_res.var())
        if neigh_var > 1e-3:
            ratios.append(roi_var / neigh_var)
    cap.release()
    return {"noise_variance_ratio": round(float(np.mean(ratios)) if ratios else 1.0, 4)}


# ---------------------------------------------------------------------------
# Edge sharpness ratio ("pasted-on edge" warning detector)
# ---------------------------------------------------------------------------


def roi_edge_sharpness_ratio(
    composite_path: str,
    roi: tuple[int, int, int, int],
    n_samples: int = 6,
    edge_band_px: int = 8,
) -> dict[str, float]:
    """Mean Sobel-gradient magnitude on the ROI border vs the rest of the frame.

    A pasted-on logo introduces unnaturally sharp gradients along its bounding
    rectangle. ratio > 1.8 = warning.
    """
    cap = cv2.VideoCapture(composite_path)
    if not cap.isOpened():
        return {"edge_sharpness_ratio": 1.0}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return {"edge_sharpness_ratio": 1.0}
    indices = np.linspace(0, max(n - 1, 0), n_samples, dtype=int)
    ratios: list[float] = []
    x0, y0, x1, y1 = roi
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)

        h, w = gray.shape
        b = edge_band_px
        # Border band of the ROI (a rectangular ring `b` pixels wide on each side).
        border_mask = np.zeros_like(mag, dtype=bool)
        bx0, bx1 = max(0, x0 - b), min(w, x1 + b)
        by0, by1 = max(0, y0 - b), min(h, y1 + b)
        border_mask[by0:by1, bx0:bx1] = True
        inner_mask = np.zeros_like(mag, dtype=bool)
        inner_mask[max(0, y0 + b):max(0, y1 - b), max(0, x0 + b):max(0, x1 - b)] = True
        ring = border_mask & ~inner_mask

        elsewhere = np.ones_like(mag, dtype=bool)
        elsewhere[by0:by1, bx0:bx1] = False
        if not ring.any() or not elsewhere.any():
            continue
        ratios.append(float(mag[ring].mean()) / max(float(mag[elsewhere].mean()), 1e-6))
    cap.release()
    return {"edge_sharpness_ratio": round(float(np.mean(ratios)) if ratios else 1.0, 4)}
