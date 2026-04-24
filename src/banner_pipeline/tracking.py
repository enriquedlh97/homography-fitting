"""Optical-flow corner tracker for temporally stable video compositing.

Instead of re-running SAM2 segmentation + quad fitting on every frame,
track the 4 quad corners from frame 0 across subsequent frames using
Lucas-Kanade pyramidal optical flow with EMA smoothing.

Ported from Raghav's ``CornerTracker`` in the original banner_segment.py.
"""

from __future__ import annotations

import cv2
import numpy as np


class CornerTracker:
    """Track quad corners across frames using sparse optical flow.

    Usage
    -----
    1. Segment + fit on frame 0 to get initial corners per object.
    2. Call ``init(obj_id, corners, frame_gray)`` for each object.
    3. For each subsequent frame, call ``update(frame_gray)`` to get
       smoothed corners for all tracked objects.

    Parameters
    ----------
    ema_alpha : float
        Exponential moving average weight for temporal smoothing.
        Higher = more responsive to new positions; lower = smoother.
        Default 0.3 (Raghav's tuned value).
    fb_threshold : float
        Forward-backward consistency threshold in pixels.  Tracks with
        higher error are rejected and the previous position is kept.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        fb_threshold: float = 2.0,
        lk_win_size: int = 21,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.fb_threshold = fb_threshold
        self.LK_PARAMS = dict(
            winSize=(lk_win_size, lk_win_size),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._corners: dict[int, np.ndarray] = {}  # obj_id -> (4, 2) float32
        self._smoothed: dict[int, np.ndarray] = {}  # EMA-smoothed version
        self._prev_gray: np.ndarray | None = None

    def init(self, obj_id: int, corners: np.ndarray, frame_gray: np.ndarray) -> None:
        """Seed an object's corners from the frame-0 quad fit."""
        c = corners.astype(np.float32).reshape(4, 2)
        self._corners[obj_id] = c
        self._smoothed[obj_id] = c.copy()
        self._prev_gray = frame_gray

    def update(self, frame_gray: np.ndarray) -> dict[int, np.ndarray]:
        """Track all objects to a new frame.

        Returns ``{obj_id: corners}`` where corners are EMA-smoothed
        ``(4, 2) float32`` arrays in ``[TL, TR, BR, BL]`` order.
        """
        if self._prev_gray is None or not self._corners:
            self._prev_gray = frame_gray
            return {oid: c.copy() for oid, c in self._smoothed.items()}

        # Concatenate all object corners into one array for a single
        # optical flow call (more efficient than per-object calls).
        all_pts = []
        obj_ids = []
        for oid, c in self._corners.items():
            all_pts.append(c)
            obj_ids.append(oid)

        pts_old = np.vstack(all_pts).reshape(-1, 1, 2)

        # Scene cut detection: if the frame difference is very large,
        # the camera angle has changed drastically. Reset EMA state
        # to prevent smoothing across the cut.
        frame_diff = cv2.absdiff(self._prev_gray, frame_gray)
        mean_diff = float(frame_diff.mean())
        if mean_diff > 30:  # hard cut threshold
            # Reset EMA: use raw optical flow without smoothing
            for oid in obj_ids:
                self._smoothed[oid] = self._corners[oid].copy()

        # Forward flow: prev -> current
        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, frame_gray, pts_old, None, **self.LK_PARAMS
        )

        # Backward flow: current -> prev (forward-backward consistency check)
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            frame_gray, self._prev_gray, pts_new, None, **self.LK_PARAMS
        )
        fb_dist = np.linalg.norm((pts_old - pts_back).reshape(-1, 2), axis=1)
        good = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_dist < self.fb_threshold)

        # Distribute tracked points back to per-object corners.
        idx = 0
        result: dict[int, np.ndarray] = {}
        for oid in obj_ids:
            n_good = sum(1 for j in range(4) if good[idx + j])

            if n_good >= 2:
                # Enough good tracks: update corners.
                new_c = np.empty((4, 2), dtype=np.float32)
                for j in range(4):
                    if good[idx + j]:
                        new_c[j] = pts_new[idx + j].ravel()
                    else:
                        new_c[j] = self._corners[oid][j]
                self._corners[oid] = new_c
            else:
                # Too few good tracks: hold previous corners entirely.
                # This prevents jitter from bad optical flow frames.
                new_c = self._corners[oid]

            # EMA smooth for temporal stability.
            sm = self.ema_alpha * new_c + (1 - self.ema_alpha) * self._smoothed[oid]
            self._smoothed[oid] = sm
            result[oid] = sm.copy()
            idx += 4

        self._prev_gray = frame_gray
        return result

    @property
    def tracked_ids(self) -> list[int]:
        """Return the list of currently tracked object IDs."""
        return list(self._corners.keys())
