"""Temporal SSIM, region-parameterized.

Math lifted from `scripts/quality_eval.py:318-355`. ROI is the only parameter.
"""

from __future__ import annotations

import cv2
import numpy as np


def roi_temporal_ssim(
    video_path: str,
    roi: tuple[int, int, int, int],
) -> dict[str, float]:
    """Mean SSIM between consecutive frames inside `roi`.

    Returns mean and 5th-percentile SSIM. > 0.95 = stable / no flicker.
    """
    x0, y0, x1, y1 = roi
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"roi_temporal_ssim_mean": 0.0, "roi_temporal_ssim_p5": 0.0}
    prev_gray: np.ndarray | None = None
    ssim_values: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
        if prev_gray is not None and gray.shape == prev_gray.shape:
            mu1, mu2 = gray.mean(), prev_gray.mean()
            s1, s2 = gray.std(), prev_gray.std()
            s12 = ((gray - mu1) * (prev_gray - mu2)).mean()
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
                (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
            )
            ssim_values.append(float(ssim))
        prev_gray = gray
    cap.release()
    if not ssim_values:
        return {"roi_temporal_ssim_mean": 0.0, "roi_temporal_ssim_p5": 0.0}
    arr = np.asarray(ssim_values)
    return {
        "roi_temporal_ssim_mean": round(float(arr.mean()), 4),
        "roi_temporal_ssim_p5": round(float(np.percentile(arr, 5)), 4),
    }


def roi_ssim_vs_reference(
    video_a: str,
    video_b: str,
    roi: tuple[int, int, int, int],
) -> dict[str, float]:
    """Mean SSIM between corresponding frames of two videos inside `roi`.

    Used by the reference comparison: how close does the current run look to
    the gold inside each region?
    """
    x0, y0, x1, y1 = roi
    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)
    if not (cap_a.isOpened() and cap_b.isOpened()):
        cap_a.release()
        cap_b.release()
        return {"roi_ssim_vs_reference_mean": 0.0, "roi_ssim_vs_reference_p5": 0.0}
    ssims: list[float] = []
    while True:
        oa, fa = cap_a.read()
        ob, fb = cap_b.read()
        if not (oa and ob):
            break
        ga = cv2.cvtColor(fa[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
        gb = cv2.cvtColor(fb[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
        if ga.shape != gb.shape or ga.size == 0:
            continue
        mu1, mu2 = ga.mean(), gb.mean()
        s1, s2 = ga.std(), gb.std()
        s12 = ((ga - mu1) * (gb - mu2)).mean()
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
        )
        ssims.append(float(ssim))
    cap_a.release()
    cap_b.release()
    if not ssims:
        return {"roi_ssim_vs_reference_mean": 0.0, "roi_ssim_vs_reference_p5": 0.0}
    arr = np.asarray(ssims)
    return {
        "roi_ssim_vs_reference_mean": round(float(arr.mean()), 4),
        "roi_ssim_vs_reference_p5": round(float(np.percentile(arr, 5)), 4),
    }
