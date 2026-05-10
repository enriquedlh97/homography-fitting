"""Walkover-window detection and occlusion correctness metrics.

The court-floor logo (obj_id 3) is the most demanding placement — the player
physically walks over it. We auto-detect the window by computing the per-frame
mean abs luminance delta between the original video and the clean plate inside
the floor placement_quad (padded). The longest contiguous super-threshold run,
padded ±10 frames, is the walkover window.

Inside the window, two metrics quantify how well the pipeline handles
occlusion:

- `logo_visible_pct` : fraction of placement_quad pixels showing logo signal
  in player-absent regions of the frame.
- `occlusion_iou`    : IoU between (|original − clean| > T) and
  (|composite − baked_logo| > T) inside placement_quad. A "baked logo" plate
  is built once from the gold composited.mp4 at frame 0 (logo on clean court
  with no player); for the simpler v0 we use the gold's frame at the same
  index since both runs share the input video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class WalkoverWindow:
    start: int
    end: int
    method: str  # "delta_threshold" | "manual_override" | "no_clean_video"


def detect_walkover_window(
    original_video: str | Path,
    clean_video: str | Path | None,
    floor_quad: np.ndarray,
    pad_x: int = 30,
    pad_y: int = 60,
    smoothing: int = 5,
    sigma_k: float = 2.0,
    pad_frames: int = 10,
) -> WalkoverWindow | None:
    """Auto-detect the walkover window via clean-vs-original luminance delta.

    Returns None if no window can be detected (e.g. no clean video and the
    fallback temporal-median heuristic also fails).
    """
    cap_o = cv2.VideoCapture(str(original_video))
    if not cap_o.isOpened():
        return None
    n = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap_o.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_o.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:
        cap_o.release()
        return None

    x0 = max(0, int(np.floor(floor_quad[:, 0].min())) - pad_x)
    y0 = max(0, int(np.floor(floor_quad[:, 1].min())) - pad_y)
    x1 = min(fw, int(np.ceil(floor_quad[:, 0].max())) + pad_x)
    y1 = min(fh, int(np.ceil(floor_quad[:, 1].max())) + pad_y)
    if x1 <= x0 or y1 <= y0:
        cap_o.release()
        return None

    cap_c = cv2.VideoCapture(str(clean_video)) if clean_video and Path(clean_video).is_file() else None
    use_clean = cap_c is not None and cap_c.isOpened()

    deltas: list[float] = []
    if use_clean:
        method = "delta_threshold"
        while True:
            oa, fo = cap_o.read()
            oc, fc = cap_c.read()
            if not (oa and oc):
                break
            # Resize clean to original frame size BEFORE cropping the ROI —
            # the clean plate is often a lower-resolution rendering covering
            # only the inpaint region.
            if fc.shape[:2] != fo.shape[:2]:
                fc = cv2.resize(fc, (fo.shape[1], fo.shape[0]))
            roi_o = fo[y0:y1, x0:x1].astype(np.float32)
            roi_c = fc[y0:y1, x0:x1].astype(np.float32)
            if roi_o.shape != roi_c.shape or roi_o.size == 0:
                continue
            deltas.append(float(np.abs(roi_o - roi_c).mean()))
    else:
        # Temporal-median fallback: build a per-pixel median over the clip
        # as the local "no-player" reference, then compute deltas vs that.
        method = "no_clean_video"
        # Collect a sparse stack to estimate the median to keep memory bounded.
        sample_indices = np.linspace(0, n - 1, min(n, 24), dtype=int)
        stack: list[np.ndarray] = []
        for idx in sample_indices:
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap_o.read()
            if not ok:
                continue
            stack.append(frame[y0:y1, x0:x1].astype(np.float32))
        if not stack:
            cap_o.release()
            if cap_c is not None:
                cap_c.release()
            return None
        median_roi = np.median(np.stack(stack, axis=0), axis=0)

        cap_o.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap_o.read()
            if not ok:
                break
            roi = frame[y0:y1, x0:x1].astype(np.float32)
            deltas.append(float(np.abs(roi - median_roi).mean()))

    cap_o.release()
    if cap_c is not None:
        cap_c.release()
    if not deltas:
        return None

    arr = np.asarray(deltas)
    # Box-smooth.
    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        arr = np.convolve(arr, kernel, mode="same")

    threshold = float(arr.mean() + sigma_k * arr.std())
    above = arr > threshold
    if not above.any():
        return None

    # Longest contiguous run.
    runs: list[tuple[int, int]] = []
    in_run = False
    s = 0
    for i, hit in enumerate(above):
        if hit and not in_run:
            s = i
            in_run = True
        elif not hit and in_run:
            runs.append((s, i - 1))
            in_run = False
    if in_run:
        runs.append((s, len(above) - 1))
    if not runs:
        return None
    s, e = max(runs, key=lambda r: r[1] - r[0])
    s = max(0, s - pad_frames)
    e = min(len(above) - 1, e + pad_frames)
    return WalkoverWindow(start=int(s), end=int(e), method=method)


def occlusion_metrics_in_window(
    composite_path: str | Path,
    original_path: str | Path,
    clean_path: str | Path | None,
    reference_composite_path: str | Path | None,
    floor_quad: np.ndarray,
    window: WalkoverWindow,
    delta_thresh: int = 20,
) -> dict[str, float]:
    """Per-window aggregate occlusion correctness signals.

    `logo_visible_pct` averages, across walkover frames, the fraction of
    placement_quad pixels where the composite differs meaningfully from the
    clean plate but the original does NOT (i.e. the logo is present in
    composite, the player is not in original at that pixel).

    `occlusion_iou` averages, across walkover frames, the IoU between
    "where the player is" (|original - clean| > T) and the corresponding
    region in the composite. Without a reference, only player-presence is
    available so the metric falls back to logo_visible_pct alone.
    """
    cap_co = cv2.VideoCapture(str(composite_path))
    cap_o = cv2.VideoCapture(str(original_path))
    cap_c = cv2.VideoCapture(str(clean_path)) if clean_path and Path(clean_path).is_file() else None
    cap_r = (
        cv2.VideoCapture(str(reference_composite_path))
        if reference_composite_path and Path(reference_composite_path).is_file()
        else None
    )
    if not cap_co.isOpened() or not cap_o.isOpened():
        for c in (cap_co, cap_o, cap_c, cap_r):
            if c is not None:
                c.release()
        return {"walkover_logo_visible_pct": 0.0, "walkover_occlusion_iou": 0.0}

    fw = int(cap_co.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_co.get(cv2.CAP_PROP_FRAME_HEIGHT))

    quad_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(quad_mask, [floor_quad.astype(np.int32)], 255)
    quad_bool = quad_mask > 0
    quad_size = int(quad_bool.sum())
    if quad_size == 0:
        for c in (cap_co, cap_o, cap_c, cap_r):
            if c is not None:
                c.release()
        return {"walkover_logo_visible_pct": 0.0, "walkover_occlusion_iou": 0.0}

    visible_pcts: list[float] = []
    ious: list[float] = []
    for fid in range(window.start, window.end + 1):
        cap_co.set(cv2.CAP_PROP_POS_FRAMES, fid)
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok_co, fco = cap_co.read()
        ok_o, fo = cap_o.read()
        if not (ok_co and ok_o):
            continue
        clean_frame = None
        if cap_c is not None:
            cap_c.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok_c, fc = cap_c.read()
            if ok_c:
                clean_frame = fc if fc.shape[:2] == fo.shape[:2] else cv2.resize(fc, (fw, fh))
        if clean_frame is None:
            continue
        gray_co = cv2.cvtColor(fco, cv2.COLOR_BGR2GRAY).astype(np.int16)
        gray_o = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY).astype(np.int16)
        gray_c = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY).astype(np.int16)

        player_present = (np.abs(gray_o - gray_c) > delta_thresh)
        logo_signal = (np.abs(gray_co - gray_c) > delta_thresh)

        # Pixels inside the placement quad where the player is NOT in the
        # original — those are the pixels we expect to show logo.
        non_player_in_quad = quad_bool & ~player_present
        if non_player_in_quad.any():
            visible_pcts.append(float(logo_signal[non_player_in_quad].mean()))

        # Reference IoU: where the gold composite differs from clean (logo or
        # rendered player), compared to current composite. Otherwise fall back
        # to player-presence-vs-composite-edits.
        if cap_r is not None:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok_r, fr = cap_r.read()
            if ok_r and fr.shape == fco.shape:
                gray_r = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)
                ref_signal = (np.abs(gray_r - gray_c) > delta_thresh) & quad_bool
                cur_signal = logo_signal & quad_bool
                inter = int((ref_signal & cur_signal).sum())
                union = int((ref_signal | cur_signal).sum())
                if union > 0:
                    ious.append(inter / union)
        else:
            # No reference: how well does the composite preserve the player?
            ref_signal = player_present & quad_bool
            cur_signal = logo_signal & quad_bool
            # If composite hides the player where it shouldn't, IoU drops.
            inter = int((ref_signal & cur_signal).sum())
            union = int((ref_signal | cur_signal).sum())
            if union > 0:
                # Invert: high overlap = composite tracks player edits well.
                # We report (1 - over_paint_ratio) where over_paint_ratio is
                # (player-pixels-not-in-composite-edit) / player-pixels.
                player_total = int(ref_signal.sum())
                if player_total > 0:
                    over_paint = 1.0 - (inter / player_total)
                    ious.append(max(0.0, 1.0 - over_paint))

    for c in (cap_co, cap_o, cap_c, cap_r):
        if c is not None:
            c.release()

    return {
        "walkover_logo_visible_pct": round(
            float(np.mean(visible_pcts)) if visible_pcts else 0.0, 4
        ),
        "walkover_occlusion_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
    }
