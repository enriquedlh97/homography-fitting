"""Interactive click-collection UI (local / non-headless only)."""

from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline.segment.base import ObjectPrompt

OBJ_COLORS_UI = [
    (0, 255, 0),
    (0, 100, 255),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
]


def collect_clicks(frame: np.ndarray, frame_idx: int = 0) -> list[ObjectPrompt]:
    """Show *frame* in an OpenCV window and collect grouped seed clicks.

    Controls
    --------
    - Left-click : add positive point to current object
    - Right-click: add negative point to current object
    - U          : undo last point in current object
    - N          : finish current object, start a new one
    - Enter/Space: finish all
    - Escape     : cancel (returns empty list)

    Returns a list of ``ObjectPrompt`` instances with explicit labels and
    the selected ``frame_idx``.
    """
    groups: list[list[tuple[int, int, int]]] = [[]]
    win = "Click prompts (LMB=+, RMB=-, U=undo, N=next, Enter=done)"

    def _current_color():
        return OBJ_COLORS_UI[(len(groups) - 1) % len(OBJ_COLORS_UI)]

    def _redraw():
        display = frame.copy()
        for obj_idx, group in enumerate(groups, start=1):
            color = OBJ_COLORS_UI[(obj_idx - 1) % len(OBJ_COLORS_UI)]
            for pt_idx, (x, y, label) in enumerate(group, start=1):
                marker = cv2.MARKER_STAR if label == 1 else cv2.MARKER_TILTED_CROSS
                suffix = "+" if label == 1 else "-"
                cv2.drawMarker(display, (x, y), color, marker, 20, 2)
                cv2.putText(
                    display,
                    f"{obj_idx}.{pt_idx}{suffix}",
                    (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        obj_idx = len(groups)
        current = groups[-1]
        n_pos = sum(label == 1 for *_xy, label in current)
        n_neg = sum(label == 0 for *_xy, label in current)
        label = (
            f"Object {obj_idx}  ({n_pos}+/{n_neg}-)  |  LMB+=  RMB-=  U=undo  N=next  Enter=done"
        )
        cv2.rectangle(display, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(
            display,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(win, display)

    def _on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            groups[-1].append((x, y, 1))
            _redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            groups[-1].append((x, y, 0))
            _redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, _on_mouse)
    _redraw()

    print("[UI] Left-click = positive point, right-click = negative point.")
    print("     U = undo last point | N = next object | Enter/Space = done | Esc = cancel")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # Enter / Space
            break
        if key == 27:  # Escape
            groups.clear()
            break
        if key in (ord("u"), ord("U")) and groups[-1]:
            groups[-1].pop()
            _redraw()
        if key in (ord("n"), ord("N")) and groups[-1]:
            print(
                f"  Object {len(groups)} done ({len(groups[-1])} pts). "
                f"Starting object {len(groups) + 1}…"
            )
            groups.append([])
            _redraw()

    cv2.destroyAllWindows()
    prompts: list[ObjectPrompt] = []
    for obj_id, group in enumerate((g for g in groups if g), start=1):
        points = np.array([[x, y] for x, y, _label in group], dtype=np.float32)
        labels = np.array([label for _x, _y, label in group], dtype=np.int32)
        prompts.append(
            ObjectPrompt(
                obj_id=obj_id,
                points=points,
                labels=labels,
                frame_idx=frame_idx,
            )
        )
    return prompts


def select_polygon(frame: np.ndarray) -> np.ndarray | None:
    """Interactive polygon vertex selection.

    Controls
    --------
    - Left-click  : add a vertex
    - Right-click / Enter / Space : close polygon and accept
    - Escape      : cancel

    Returns an ``(N, 2)`` float32 array, or ``None`` if cancelled.
    """
    pts: list[tuple[int, int]] = []
    display = frame.copy()
    win = "Click corners (right-click/Enter=done, Esc=cancel)"

    def _on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.drawMarker(display, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            if len(pts) > 1:
                cv2.line(display, pts[-2], pts[-1], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(
                display,
                str(len(pts)),
                (x + 8, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow(win, display)
        elif event == cv2.EVENT_RBUTTONDOWN and len(pts) >= 3:
            cv2.line(display, pts[-1], pts[0], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, _on_mouse)
    cv2.imshow(win, display)

    print("[UI] Left-click to add polygon vertices.")
    print("     Right-click or Enter/Space to finish | Esc to cancel")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):
            break
        if key == 27:
            pts.clear()
            break

    cv2.destroyAllWindows()
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.float32)
