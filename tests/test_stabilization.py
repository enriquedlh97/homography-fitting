from __future__ import annotations

import cv2
import numpy as np

from banner_pipeline import stabilization as stabilization_mod


def _base_mask() -> np.ndarray:
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[10:18, 12:30] = 255
    return mask


def _translate_mask(mask: np.ndarray, tx: float, ty: float) -> np.ndarray:
    height, width = mask.shape[:2]
    homography = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)
    return cv2.warpPerspective(mask, homography, (width, height), flags=cv2.INTER_NEAREST)


def _identity_motion() -> stabilization_mod.MotionEstimate:
    return stabilization_mod.MotionEstimate(
        homography=np.eye(3, dtype=np.float32),
        median_corner_disp_px=0.0,
        near_static=True,
        estimation_ok=True,
    )


def test_stabilizer_hard_holds_near_static_jitter() -> None:
    base_mask = _base_mask()
    jittered_mask = _translate_mask(base_mask, tx=1.0, ty=0.0)
    stabilized_segments, object_stats = stabilization_mod._stabilize_masks_with_motion(
        video_segments={
            0: {1: base_mask},
            1: {1: jittered_mask},
        },
        tracked_obj_ids=[1],
        motion_steps=[_identity_motion(), _identity_motion()],
        frame_shape=base_mask.shape,
        config=stabilization_mod.StabilizationConfig(enabled=True),
    )

    assert np.array_equal(stabilized_segments[0][1], base_mask)
    assert np.array_equal(stabilized_segments[1][1], base_mask)
    assert object_stats["1"]["frames_held"] == 1
    assert object_stats["1"]["frames_raw_accepted"] == 1


def test_stabilizer_bridges_short_empty_mask_gaps() -> None:
    base_mask = _base_mask()
    stabilized_segments, object_stats = stabilization_mod._stabilize_masks_with_motion(
        video_segments={
            0: {1: base_mask},
            1: {},
            2: {},
            3: {},
        },
        tracked_obj_ids=[1],
        motion_steps=[
            _identity_motion(),
            _identity_motion(),
            _identity_motion(),
            _identity_motion(),
        ],
        frame_shape=base_mask.shape,
        config=stabilization_mod.StabilizationConfig(enabled=True),
    )

    for frame_idx in (1, 2, 3):
        assert np.array_equal(stabilized_segments[frame_idx][1], base_mask)
    assert object_stats["1"]["frames_held"] == 3
    assert object_stats["1"]["frames_empty_reused"] == 3
    assert object_stats["1"]["max_empty_hold_streak"] == 3


def test_stabilizer_warps_previous_mask_for_nontrivial_motion() -> None:
    base_mask = _base_mask()
    translation = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    motion_steps = [
        _identity_motion(),
        stabilization_mod.MotionEstimate(
            homography=translation,
            median_corner_disp_px=5.0,
            near_static=False,
            estimation_ok=True,
        ),
    ]
    stabilized_segments, object_stats = stabilization_mod._stabilize_masks_with_motion(
        video_segments={
            0: {1: base_mask},
            1: {},
        },
        tracked_obj_ids=[1],
        motion_steps=motion_steps,
        frame_shape=base_mask.shape,
        config=stabilization_mod.StabilizationConfig(enabled=True),
    )

    expected = _translate_mask(base_mask, tx=5.0, ty=2.0)
    assert np.array_equal(stabilized_segments[1][1], expected)
    assert object_stats["1"]["frames_held"] == 1
    assert object_stats["1"]["frames_empty_reused"] == 1


def test_stabilizer_accepts_low_iou_raw_mask_and_resets_anchor() -> None:
    base_mask = _base_mask()
    new_raw_mask = _translate_mask(base_mask, tx=18.0, ty=8.0)
    stabilized_segments, object_stats = stabilization_mod._stabilize_masks_with_motion(
        video_segments={
            0: {1: base_mask},
            1: {1: new_raw_mask},
            2: {},
        },
        tracked_obj_ids=[1],
        motion_steps=[_identity_motion(), _identity_motion(), _identity_motion()],
        frame_shape=base_mask.shape,
        config=stabilization_mod.StabilizationConfig(enabled=True),
    )

    assert np.array_equal(stabilized_segments[1][1], new_raw_mask)
    assert np.array_equal(stabilized_segments[2][1], new_raw_mask)
    assert object_stats["1"]["frames_raw_accepted"] == 2
    assert object_stats["1"]["frames_held"] == 1
    assert object_stats["1"]["frames_empty_reused"] == 1
