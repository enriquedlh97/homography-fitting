#!/usr/bin/env python3
"""Build forensic contact sheets for under-foot MELBOURNE text leakage.

The remaining artifact is brief: text-colored pixels appear under the moving
shoe during contact. This diagnostic compares original, clean plate, and
composite frames to find pixels where the composite still behaves like the
original frame inside a review crop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_int_list(text: str) -> list[int]:
    """Parse comma-separated integer values."""
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_crop_xyxy(text: str) -> tuple[int, int, int, int]:
    """Parse x1,y1,x2,y2 crop coordinates."""
    values = parse_int_list(text)
    if len(values) != 4:
        raise ValueError("crop must be x1,y1,x2,y2")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("crop must have x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Read one BGR frame from a video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame_bgr


def crop_frame(frame_bgr: np.ndarray, crop_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame with clipping to image bounds."""
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = crop_xyxy
    x1 = int(np.clip(x1, 0, width))
    x2 = int(np.clip(x2, 0, width))
    y1 = int(np.clip(y1, 0, height))
    y2 = int(np.clip(y2, 0, height))
    return frame_bgr[y1:y2, x1:x2].copy()


def resize_like(frame_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    """Resize frame to match reference dimensions if needed."""
    reference_height, reference_width = reference_bgr.shape[:2]
    if frame_bgr.shape[:2] == (reference_height, reference_width):
        return frame_bgr
    return cv2.resize(frame_bgr, (reference_width, reference_height), interpolation=cv2.INTER_CUBIC)


def make_heatmap(values: np.ndarray) -> np.ndarray:
    """Convert a scalar map to a colored heatmap."""
    clipped = np.clip(values, 0.0, 1.0)
    image = (clipped * 255.0).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def add_label(image_bgr: np.ndarray, text: str) -> np.ndarray:
    """Add a small readable label to an image."""
    output = image_bgr.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def compute_original_survival_map(
    original_crop: np.ndarray,
    clean_crop: np.ndarray,
    composite_crop: np.ndarray,
) -> np.ndarray:
    """Estimate how much each composite pixel follows original-vs-clean delta."""
    original_vector = original_crop.astype(np.float32) - clean_crop.astype(np.float32)
    composite_vector = composite_crop.astype(np.float32) - clean_crop.astype(np.float32)
    numerator = np.sum(original_vector * composite_vector, axis=2)
    denominator = np.sum(original_vector * original_vector, axis=2) + 1e-6
    survival = numerator / denominator
    return np.clip(survival, 0.0, 1.0)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Overlay a binary mask on an image."""
    output = image_bgr.copy()
    mask_bool = mask > 0
    if np.any(mask_bool):
        color = np.array(color_bgr, dtype=np.float32)
        output[mask_bool] = (
            output[mask_bool].astype(np.float32) * 0.35 + color[None, :] * 0.65
        ).astype(np.uint8)
    return output


def build_contact_sheet(
    *,
    original_crop: np.ndarray,
    clean_crop: np.ndarray,
    composite_crop: np.ndarray,
    frame_index: int,
    variant_label: str,
    original_survival_threshold: float,
    text_delta_threshold: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build one forensic row and return metrics for the crop."""
    original_clean_delta = np.mean(
        np.abs(original_crop.astype(np.int16) - clean_crop.astype(np.int16)),
        axis=2,
    )
    composite_clean_delta = np.mean(
        np.abs(composite_crop.astype(np.int16) - clean_crop.astype(np.int16)),
        axis=2,
    )
    original_survival = compute_original_survival_map(
        original_crop,
        clean_crop,
        composite_crop,
    )

    original_text_candidate = original_clean_delta > float(text_delta_threshold)
    composite_original_like = original_survival > original_survival_threshold
    suspected_leak_mask = original_text_candidate & composite_original_like

    diff_heatmap = make_heatmap(np.clip(original_clean_delta / 80.0, 0.0, 1.0))
    survival_heatmap = make_heatmap(original_survival)
    leak_overlay = overlay_mask(composite_crop, suspected_leak_mask.astype(np.uint8), (0, 0, 255))

    columns = [
        add_label(original_crop, f"original f{frame_index}"),
        add_label(clean_crop, "clean plate"),
        add_label(composite_crop, variant_label),
        add_label(diff_heatmap, "original-clean delta"),
        add_label(survival_heatmap, "original survival"),
        add_label(leak_overlay, "suspected leak overlay"),
    ]
    sheet = np.hstack(columns)

    metrics = {
        "frame_index": float(frame_index),
        "mean_original_clean_delta": float(np.mean(original_clean_delta)),
        "mean_composite_clean_delta": float(np.mean(composite_clean_delta)),
        "mean_original_survival": float(np.mean(original_survival)),
        "suspected_leak_pixels": float(np.count_nonzero(suspected_leak_mask)),
        "suspected_leak_ratio": float(np.mean(suspected_leak_mask)),
    }
    return sheet, metrics


def parse_composite_arg(value: str) -> tuple[str, Path]:
    """Parse label=path for a composite video."""
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path_text = value.split("=", 1)
    return label.strip(), Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose under-foot text leakage by comparing original, clean, and composite videos.",
    )
    parser.add_argument("--original-video", required=True, type=Path)
    parser.add_argument("--clean-video", required=True, type=Path)
    parser.add_argument(
        "--composite",
        action="append",
        required=True,
        help="Composite video as label=path. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", default="690,700,710,720,730,740")
    parser.add_argument("--crop", default="520,735,1360,1045", help="x1,y1,x2,y2")
    parser.add_argument("--original-survival-threshold", type=float, default=0.45)
    parser.add_argument("--text-delta-threshold", type=int, default=18)
    args = parser.parse_args()

    frame_indices = parse_int_list(args.frames)
    crop_xyxy = parse_crop_xyxy(args.crop)
    composites = [parse_composite_arg(value) for value in args.composite]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, str | float]] = []

    for frame_index in frame_indices:
        original_frame = read_video_frame(args.original_video, frame_index)
        clean_frame = resize_like(read_video_frame(args.clean_video, frame_index), original_frame)
        original_crop = crop_frame(original_frame, crop_xyxy)
        clean_crop = crop_frame(clean_frame, crop_xyxy)

        frame_rows: list[np.ndarray] = []
        for variant_label, composite_path in composites:
            composite_frame = resize_like(read_video_frame(composite_path, frame_index), original_frame)
            composite_crop = crop_frame(composite_frame, crop_xyxy)
            sheet, metrics = build_contact_sheet(
                original_crop=original_crop,
                clean_crop=clean_crop,
                composite_crop=composite_crop,
                frame_index=frame_index,
                variant_label=variant_label,
                original_survival_threshold=args.original_survival_threshold,
                text_delta_threshold=args.text_delta_threshold,
            )
            frame_rows.append(sheet)
            metrics_rows.append(
                {
                    "variant": variant_label,
                    **metrics,
                }
            )

        frame_sheet = np.vstack(frame_rows)
        cv2.imwrite(str(args.output_dir / f"underfoot_diagnostic_frame_{frame_index}.png"), frame_sheet)

    metrics_path = args.output_dir / "underfoot_diagnostic_metrics.csv"
    with metrics_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "variant",
                "frame_index",
                "mean_original_clean_delta",
                "mean_composite_clean_delta",
                "mean_original_survival",
                "suspected_leak_pixels",
                "suspected_leak_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"Wrote diagnostics to {args.output_dir}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
