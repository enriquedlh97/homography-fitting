"""Visual artifact generation: per-region crop strips + forensic contact sheets.

`build_forensic_sheet` is refactored from
`scripts/diagnose_underfoot_text_leak.py:118-169` and parameterized by ROI.

`crops_strip` and `consecutive_frames_strip` produce the per-region PNGs the
eval framework writes under `eval/<region>/`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _label(image: np.ndarray, text: str) -> np.ndarray:
    """Add a title bar to an image."""
    h, w = image.shape[:2]
    band_h = 28
    out = np.zeros((h + band_h, w, 3), dtype=np.uint8)
    out[band_h:] = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (w, band_h), (32, 32, 32), -1)
    cv2.putText(
        out,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def crops_strip(
    composite_path: str | Path,
    output_path: str | Path,
    roi: tuple[int, int, int, int] | None,
    n_frames: int = 6,
    upscale: int = 1,
) -> None:
    """Save a horizontal strip of `n_frames` evenly-spaced frames.

    If `roi` is None, full frames are used (used for the `full/` region).
    """
    cap = cv2.VideoCapture(str(composite_path))
    if not cap.isOpened():
        return
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return
    indices = np.linspace(0, n - 1, n_frames, dtype=int)
    panels: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        img = frame if roi is None else frame[roi[1]:roi[3], roi[0]:roi[2]]
        if upscale > 1:
            img = cv2.resize(
                img,
                (img.shape[1] * upscale, img.shape[0] * upscale),
                interpolation=cv2.INTER_LANCZOS4,
            )
        panels.append(_label(img, f"f{int(idx):04d}"))
    cap.release()
    if not panels:
        return
    h = max(p.shape[0] for p in panels)
    panels = [_pad_to(p, h) for p in panels]
    sheet = np.hstack(panels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def consecutive_frames_strip(
    composite_path: str | Path,
    output_path: str | Path,
    roi: tuple[int, int, int, int] | None,
    start_frame: int,
    n_frames: int = 8,
    upscale: int = 1,
) -> None:
    """Save a horizontal strip of `n_frames` consecutive frames starting at `start_frame`."""
    cap = cv2.VideoCapture(str(composite_path))
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    panels: list[np.ndarray] = []
    for offset in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        img = frame if roi is None else frame[roi[1]:roi[3], roi[0]:roi[2]]
        if upscale > 1:
            img = cv2.resize(
                img,
                (img.shape[1] * upscale, img.shape[0] * upscale),
                interpolation=cv2.INTER_LANCZOS4,
            )
        panels.append(_label(img, f"f{start_frame + offset:04d}"))
    cap.release()
    if not panels:
        return
    h = max(p.shape[0] for p in panels)
    panels = [_pad_to(p, h) for p in panels]
    sheet = np.hstack(panels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def build_forensic_sheet(
    *,
    original_path: str | Path,
    clean_path: str | Path,
    composite_path: str | Path,
    output_path: str | Path,
    frame_index: int,
    roi: tuple[int, int, int, int],
    text_delta_threshold: int = 18,
    survival_threshold: float = 0.45,
) -> dict[str, float] | None:
    """One forensic frame: 6 columns (orig | clean | composite | deltas | leak overlay).

    Refactored from scripts/diagnose_underfoot_text_leak.py:118-169.
    Returns per-frame metrics for CSV aggregation.
    """
    x0, y0, x1, y1 = roi
    orig = _read_frame(original_path, frame_index)
    clean = _read_frame(clean_path, frame_index)
    comp = _read_frame(composite_path, frame_index)
    if orig is None or clean is None or comp is None:
        return None
    if clean.shape != orig.shape:
        clean = cv2.resize(clean, (orig.shape[1], orig.shape[0]))
    if comp.shape != orig.shape:
        comp = cv2.resize(comp, (orig.shape[1], orig.shape[0]))

    orig_c = orig[y0:y1, x0:x1]
    clean_c = clean[y0:y1, x0:x1]
    comp_c = comp[y0:y1, x0:x1]

    orig_clean_delta = np.mean(np.abs(orig_c.astype(np.int16) - clean_c.astype(np.int16)), axis=2)
    comp_clean_delta = np.mean(np.abs(comp_c.astype(np.int16) - clean_c.astype(np.int16)), axis=2)

    survival = _original_survival(orig_c, clean_c, comp_c)
    leak_mask = (orig_clean_delta > text_delta_threshold) & (survival > survival_threshold)

    diff_heat = _heatmap(np.clip(orig_clean_delta / 80.0, 0.0, 1.0))
    surv_heat = _heatmap(survival)
    leak_overlay = _overlay(comp_c, leak_mask.astype(np.uint8), (0, 0, 255))

    panels = [
        _label(orig_c, f"original f{frame_index}"),
        _label(clean_c, "clean plate"),
        _label(comp_c, "composite"),
        _label(diff_heat, "original-clean delta"),
        _label(surv_heat, "original survival"),
        _label(leak_overlay, "suspected leak overlay"),
    ]
    h = max(p.shape[0] for p in panels)
    panels = [_pad_to(p, h) for p in panels]
    sheet = np.hstack(panels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)

    return {
        "frame_index": float(frame_index),
        "mean_original_clean_delta": float(orig_clean_delta.mean()),
        "mean_composite_clean_delta": float(comp_clean_delta.mean()),
        "mean_original_survival": float(survival.mean()),
        "suspected_leak_pixels": float(leak_mask.sum()),
        "suspected_leak_ratio": float(leak_mask.mean()),
    }


def _read_frame(path: str | Path, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _heatmap(values: np.ndarray) -> np.ndarray:
    arr = np.clip(values, 0.0, 1.0)
    return cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _overlay(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    bool_mask = mask > 0
    if bool_mask.any():
        color = np.array(color_bgr, dtype=np.float32)
        out[bool_mask] = (
            out[bool_mask].astype(np.float32) * 0.35 + color[None, :] * 0.65
        ).astype(np.uint8)
    return out


def _original_survival(
    original: np.ndarray, clean: np.ndarray, composite: np.ndarray
) -> np.ndarray:
    """Measure how much the composite still resembles the original (vs the clean plate)."""
    o = original.astype(np.float32)
    cl = clean.astype(np.float32)
    co = composite.astype(np.float32)
    orig_vec = o - cl
    comp_vec = co - cl
    num = np.sum(orig_vec * comp_vec, axis=2)
    den = np.sum(orig_vec * orig_vec, axis=2) + 1e-6
    return np.clip(num / den, 0.0, 1.0)


def _pad_to(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    pad = np.zeros((target_h - h, w, 3), dtype=image.dtype)
    return np.vstack([image, pad])
