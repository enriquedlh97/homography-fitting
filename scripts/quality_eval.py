#!/usr/bin/env python3
"""Quality evaluation script for composited banner replacement videos.

Computes objective quality metrics by comparing a composited video to the
original input video. Metrics cover jitter, logo size stability, inpaint
color matching, perspective correctness, and temporal consistency.

Usage
-----
    uv run python scripts/quality_eval.py \\
        --experiment experiments/<timestamp>/ \\
        --original data/tennis-clip.mp4

    # Or compare two experiments:
    uv run python scripts/quality_eval.py \\
        --experiment experiments/run_A/ \\
        --baseline experiments/run_B/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Metric 1: Jitter ratio (frame-to-frame pixel instability)
# ---------------------------------------------------------------------------


def compute_jitter_ratio(
    composited_path: str,
    original_path: str,
    roi: tuple[int, int, int, int] = (20, 80, 250, 1350),
) -> dict:
    """Compare frame-to-frame pixel differences in a banner region.

    roi = (y_start, y_end, x_start, x_end) for the banner area.
    Returns jitter_ratio (composited/original). Target: <= 1.05.
    """
    y0, y1, x0, x1 = roi

    def _frame_diffs(video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        prev_roi = None
        diffs = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cur_roi = frame[y0:y1, x0:x1].astype(np.float32)
            if prev_roi is not None:
                diffs.append(np.abs(cur_roi - prev_roi).mean())
            prev_roi = cur_roi
        cap.release()
        return np.array(diffs)

    comp_diffs = _frame_diffs(composited_path)
    orig_diffs = _frame_diffs(original_path)

    comp_mean = float(comp_diffs.mean()) if len(comp_diffs) else 0.0
    orig_mean = float(orig_diffs.mean()) if len(orig_diffs) else 1.0
    ratio = comp_mean / orig_mean if orig_mean > 0 else 0.0

    return {
        "jitter_composited_mean": round(comp_mean, 4),
        "jitter_original_mean": round(orig_mean, 4),
        "jitter_ratio": round(ratio, 4),
        "jitter_target": "<=1.05",
        "jitter_pass": ratio <= 1.05,
    }


# ---------------------------------------------------------------------------
# Metric 2 + 4 + 6: Corner tracking analysis
# ---------------------------------------------------------------------------


def compute_corner_metrics(composited_path: str) -> dict:
    """Analyze logo quad corners across frames for stability.

    Detects bright regions (logos) in the top banner area and tracks their
    bounding boxes. Measures:
    - corner_std_px: positional jitter of detected regions
    - logo_area_cv: coefficient of variation of region area
    - overlay_accel_mean/p95: acceleration of region centroids (jitter tracker)
    """
    cap = cv2.VideoCapture(composited_path)
    centroids: list[np.ndarray] = []
    areas: list[float] = []

    # Track the top banner region
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        banner = frame[20:80, 250:1350]
        gray = cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)
        # Detect bright spots (logo text is white on dark banner)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Use the largest contour as the primary logo
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroids.append(np.array([cx, cy]))
                areas.append(area)
    cap.release()

    result: dict = {}

    if len(centroids) < 3:
        result["corner_analysis"] = "insufficient_data"
        return result

    centroids_arr = np.array(centroids)
    areas_arr = np.array(areas)

    # Corner std (positional jitter)
    result["corner_std_px"] = round(float(centroids_arr.std(axis=0).mean()), 3)

    # Max frame-to-frame jump
    jumps = np.linalg.norm(np.diff(centroids_arr, axis=0), axis=1)
    result["corner_max_jump_px"] = round(float(jumps.max()), 3)
    result["corner_mean_jump_px"] = round(float(jumps.mean()), 3)
    result["corner_jump_target"] = "<2.0"
    result["corner_jump_pass"] = float(jumps.max()) < 2.0

    # Logo area stability
    area_mean = areas_arr.mean()
    area_std = areas_arr.std()
    cv = area_std / area_mean if area_mean > 0 else 0.0
    result["logo_area_mean_px2"] = round(float(area_mean), 1)
    result["logo_area_cv"] = round(cv, 4)
    result["logo_area_cv_target"] = "<0.05"
    result["logo_area_cv_pass"] = cv < 0.05

    # Overlay acceleration (2nd derivative of position, from tennis-virtual-ads)
    if len(centroids_arr) >= 3:
        velocity = np.diff(centroids_arr, axis=0)
        accel = np.diff(velocity, axis=0)
        accel_mag = np.linalg.norm(accel, axis=1)
        result["overlay_accel_mean_px"] = round(float(accel_mag.mean()), 3)
        result["overlay_accel_p95_px"] = round(float(np.percentile(accel_mag, 95)), 3)
        result["overlay_accel_max_px"] = round(float(accel_mag.max()), 3)
        result["overlay_accel_target"] = "p95<1.0"
        result["overlay_accel_pass"] = float(np.percentile(accel_mag, 95)) < 1.0

    return result


# ---------------------------------------------------------------------------
# Metric 3: Inpaint color match
# ---------------------------------------------------------------------------


def compute_inpaint_color_match(
    composited_path: str,
    original_path: str,
    roi: tuple[int, int, int, int] = (20, 80, 250, 1350),
    n_samples: int = 10,
) -> dict:
    """Measure inpaint background color uniformity in the banner region.

    Focuses on DARK pixels (the banner background) in the composited video.
    Measures the standard deviation of the dark-pixel colors across the
    banner ROI. If inpainting creates uniform-but-different-tone patches,
    the std will be higher than the original's natural dark-pixel variance.

    Also computes delta_E between the dark pixels of composited and original.
    """
    cap_comp = cv2.VideoCapture(composited_path)
    cap_orig = cv2.VideoCapture(original_path)
    n_frames = int(cap_comp.get(cv2.CAP_PROP_FRAME_COUNT))
    y0, y1, x0, x1 = roi

    sample_indices = np.linspace(0, n_frames - 1, n_samples, dtype=int)
    dark_delta_es = []
    comp_dark_stds = []
    orig_dark_stds = []

    for idx in sample_indices:
        cap_comp.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_c, frame_c = cap_comp.read()
        ok_o, frame_o = cap_orig.read()
        if not ok_c or not ok_o:
            continue

        roi_c = frame_c[y0:y1, x0:x1]
        roi_o = frame_o[y0:y1, x0:x1]

        # Focus on dark pixels (banner background, not logo text)
        gray_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
        gray_o = cv2.cvtColor(roi_o, cv2.COLOR_BGR2GRAY)
        dark_mask_c = gray_c < 100  # dark pixels in composited
        dark_mask_o = gray_o < 100  # dark pixels in original

        if np.sum(dark_mask_c) > 100 and np.sum(dark_mask_o) > 100:
            # Color variance of dark pixels (higher = more patchy inpaint)
            lab_c = cv2.cvtColor(roi_c, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_o = cv2.cvtColor(roi_o, cv2.COLOR_BGR2LAB).astype(np.float32)

            comp_dark_stds.append(float(lab_c[dark_mask_c].std()))
            orig_dark_stds.append(float(lab_o[dark_mask_o].std()))

            # Delta E of dark pixels only
            # Use the mean dark color from each frame
            mean_c = lab_c[dark_mask_c].mean(axis=0)
            mean_o = lab_o[dark_mask_o].mean(axis=0)
            de = float(np.sqrt(((mean_c - mean_o) ** 2).sum()))
            dark_delta_es.append(de)

    cap_comp.release()
    cap_orig.release()

    mean_de = float(np.mean(dark_delta_es)) if dark_delta_es else 0.0
    comp_std = float(np.mean(comp_dark_stds)) if comp_dark_stds else 0.0
    orig_std = float(np.mean(orig_dark_stds)) if orig_dark_stds else 0.0
    uniformity_ratio = comp_std / orig_std if orig_std > 0 else 1.0

    return {
        "inpaint_dark_delta_E": round(mean_de, 3),
        "inpaint_dark_std_composited": round(comp_std, 3),
        "inpaint_dark_std_original": round(orig_std, 3),
        "inpaint_uniformity_ratio": round(uniformity_ratio, 3),
        "inpaint_color_delta_E_mean": round(mean_de, 3),
        "inpaint_color_target": "<5.0",
        "inpaint_color_pass": mean_de < 5.0,
    }


# ---------------------------------------------------------------------------
# Metric 5: Perspective correctness (quad angle deviation)
# ---------------------------------------------------------------------------


def compute_perspective_metrics(composited_path: str) -> dict:
    """Measure perspective correctness by checking quad angles.

    Detects logo quads in the banner region and measures how close
    the internal angles are to 90 degrees (a perfect rectangle in
    perspective would still have ~90 degree angles when viewed
    from the correct viewpoint).
    """
    cap = cv2.VideoCapture(composited_path)
    angle_errors: list[float] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Only sample every 10th frame
        if frame_idx % 10 != 0:
            frame_idx += 1
            continue

        banner = frame[20:80, 250:1350]
        gray = cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float64)
                # Compute angles at each corner
                for i in range(4):
                    p0 = pts[(i - 1) % 4]
                    p1 = pts[i]
                    p2 = pts[(i + 1) % 4]
                    v1 = p0 - p1
                    v2 = p2 - p1
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                    angle_errors.append(abs(angle - 90.0))
        frame_idx += 1
    cap.release()

    if not angle_errors:
        return {"perspective_analysis": "no_quads_detected"}

    errors_arr = np.array(angle_errors)
    return {
        "perspective_angle_error_mean_deg": round(float(errors_arr.mean()), 2),
        "perspective_angle_error_max_deg": round(float(errors_arr.max()), 2),
        "perspective_target": "<5.0",
        "perspective_pass": float(errors_arr.mean()) < 5.0,
    }


# ---------------------------------------------------------------------------
# Metric 7: Temporal consistency (SSIM on banner crops)
# ---------------------------------------------------------------------------


def compute_temporal_ssim(
    composited_path: str,
    roi: tuple[int, int, int, int] = (20, 80, 250, 1350),
) -> dict:
    """Compute mean SSIM between consecutive frames in the banner region."""
    cap = cv2.VideoCapture(composited_path)
    y0, y1, x0, x1 = roi
    prev_gray = None
    ssim_values = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
        if prev_gray is not None:
            # Simplified SSIM (mean-based, no window)
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
        return {"temporal_ssim": "insufficient_data"}

    return {
        "temporal_ssim_mean": round(float(np.mean(ssim_values)), 4),
        "temporal_ssim_min": round(float(np.min(ssim_values)), 4),
        "temporal_ssim_target": ">0.95",
        "temporal_ssim_pass": float(np.mean(ssim_values)) > 0.95,
    }


# ---------------------------------------------------------------------------
# Crop extraction for visual inspection
# ---------------------------------------------------------------------------


def extract_crops(
    composited_path: str,
    output_dir: str,
    roi: tuple[int, int, int, int] = (20, 80, 250, 1350),
    frame_indices: list[int] | None = None,
) -> list[str]:
    """Extract zoomed banner crops at specific frames for visual inspection."""
    if frame_indices is None:
        cap = cv2.VideoCapture(composited_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_indices = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]

    os.makedirs(output_dir, exist_ok=True)
    y0, y1, x0, x1 = roi
    saved = []

    cap = cv2.VideoCapture(composited_path)
    frame_idx = 0
    target_set = set(frame_indices)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in target_set:
            crop = frame[y0:y1, x0:x1]
            # Scale up 3x for easier visual inspection
            crop_large = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            path = os.path.join(output_dir, f"banner_crop_frame_{frame_idx:04d}.png")
            cv2.imwrite(path, crop_large)
            saved.append(path)
        frame_idx += 1
    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation(
    experiment_dir: str,
    original_video: str,
) -> dict:
    """Run all quality metrics on an experiment."""
    composited = os.path.join(experiment_dir, "outputs", "composited.mp4")
    if not os.path.exists(composited):
        raise FileNotFoundError(f"No composited.mp4 in {experiment_dir}/outputs/")

    print(f"Evaluating: {experiment_dir}")
    print(f"Original:   {original_video}")
    print()

    results: dict = {}

    print("Computing jitter ratio...")
    results.update(compute_jitter_ratio(composited, original_video))

    print("Computing corner metrics...")
    results.update(compute_corner_metrics(composited))

    print("Computing inpaint color match...")
    results.update(compute_inpaint_color_match(composited, original_video))

    print("Computing perspective metrics...")
    results.update(compute_perspective_metrics(composited))

    print("Computing temporal SSIM...")
    results.update(compute_temporal_ssim(composited))

    # Extract crops for visual inspection
    crops_dir = os.path.join(experiment_dir, "crops")
    print(f"Extracting banner crops to {crops_dir}...")
    crop_paths = extract_crops(composited, crops_dir)
    results["crop_paths"] = crop_paths

    # Summary
    pass_count = sum(1 for k, v in results.items() if k.endswith("_pass") and v is True)
    fail_count = sum(1 for k, v in results.items() if k.endswith("_pass") and v is False)
    results["metrics_passed"] = pass_count
    results["metrics_failed"] = fail_count
    results["metrics_total"] = pass_count + fail_count

    # Save
    # Convert numpy bools to Python bools for JSON serialization.
    serializable = {}
    for k, v in results.items():
        if k == "crop_paths":
            continue
        if isinstance(v, np.bool_ | np.generic):
            serializable[k] = v.item()
        else:
            serializable[k] = v

    out_path = os.path.join(experiment_dir, "quality_metrics.json")
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return results


def print_summary(results: dict) -> None:
    """Print a human-readable summary."""
    print()
    print("=" * 60)
    print("QUALITY METRICS SUMMARY")
    print("=" * 60)

    metrics = [
        ("Jitter ratio", "jitter_ratio", "jitter_pass"),
        ("Corner max jump (px)", "corner_max_jump_px", "corner_jump_pass"),
        ("Logo area CV", "logo_area_cv", "logo_area_cv_pass"),
        ("Overlay accel p95 (px)", "overlay_accel_p95_px", "overlay_accel_pass"),
        ("Inpaint color dE", "inpaint_color_delta_E_mean", "inpaint_color_pass"),
        ("Perspective angle err (deg)", "perspective_angle_error_mean_deg", "perspective_pass"),
        ("Temporal SSIM", "temporal_ssim_mean", "temporal_ssim_pass"),
    ]

    for label, key, pass_key in metrics:
        value = results.get(key, "N/A")
        passed = results.get(pass_key)
        status = "PASS" if passed else "FAIL" if passed is False else "N/A"
        marker = "+" if passed else "x" if passed is False else "?"
        if isinstance(value, float):
            print(f"  [{marker}] {label:30s} {value:>10.4f}  {status}")
        else:
            print(f"  [{marker}] {label:30s} {str(value):>10s}  {status}")

    p = results.get("metrics_passed", 0)
    f = results.get("metrics_failed", 0)
    t = results.get("metrics_total", 0)
    print()
    print(f"  Result: {p}/{t} passed, {f}/{t} failed")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate composited video quality")
    parser.add_argument("--experiment", required=True, help="Path to experiment directory")
    parser.add_argument(
        "--original", default="data/tennis-clip.mp4", help="Path to original input video"
    )
    args = parser.parse_args()

    results = run_evaluation(args.experiment, args.original)
    print_summary(results)


if __name__ == "__main__":
    main()
