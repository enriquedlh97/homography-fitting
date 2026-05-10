"""Side-by-side comparison video: current | gold | diff.

Only invoked when `--reference` is provided. Each frame is a horizontal
stack of three half-width panels with annotated frame index.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_side_by_side(
    current_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    walkover_window: tuple[int, int] | None = None,
) -> None:
    """Stack current and reference videos with a per-frame absolute-diff heatmap.

    The output is half-width per panel so the total resolution matches the
    input. Frame index and (optionally) walkover-window highlight are drawn
    on the bottom-left corner.
    """
    cap_a = cv2.VideoCapture(str(current_path))
    cap_b = cv2.VideoCapture(str(reference_path))
    if not (cap_a.isOpened() and cap_b.isOpened()):
        cap_a.release()
        cap_b.release()
        return
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

    half_w = fw // 2
    out_w = half_w * 3
    out_h = fh // 2 if fh % 2 == 0 else fh // 2

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        oa, fa = cap_a.read()
        ob, fb = cap_b.read()
        if not oa or not ob:
            break
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fw, fh))
        diff = cv2.absdiff(fa, fb)
        heat = cv2.applyColorMap(
            cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_TURBO
        )

        a_small = cv2.resize(fa, (half_w, out_h))
        b_small = cv2.resize(fb, (half_w, out_h))
        d_small = cv2.resize(heat, (half_w, out_h))

        for img, label in [(a_small, "current"), (b_small, "reference"), (d_small, "diff")]:
            cv2.putText(
                img,
                f"{label}  f{frame_idx:04d}",
                (8, out_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if walkover_window is not None and walkover_window[0] <= frame_idx <= walkover_window[1]:
            for img in (a_small, b_small, d_small):
                cv2.rectangle(img, (1, 1), (img.shape[1] - 2, img.shape[0] - 2), (0, 255, 255), 2)

        writer.write(np.hstack([a_small, b_small, d_small]))
        frame_idx += 1

    writer.release()
    cap_a.release()
    cap_b.release()
