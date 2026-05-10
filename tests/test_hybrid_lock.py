"""Unit tests for HybridLockState — hybrid lock-with-tolerance state machine."""

from __future__ import annotations

import numpy as np

from banner_pipeline.court_geometry import (
    HybridLockState,
    corner_displacement_px,
)


def _seed_quad() -> np.ndarray:
    """A simple axis-aligned 100x100 quad for synthetic tests."""
    return np.array(
        [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]],
        dtype=np.float32,
    )


def test_locked_when_estimate_within_tolerance() -> None:
    """Estimate within tolerance → returns seed, decision 'locked'."""
    seed = _seed_quad()
    estimate = seed + np.array([3.0, 0.0], dtype=np.float32)  # 3 px shift
    state = HybridLockState(seed_corners=seed, tolerance_px=6.0)

    corners, decision, disp = state.step(estimate)

    assert decision == "locked"
    np.testing.assert_allclose(corners, seed, atol=1e-5)
    assert disp < 6.0
    assert state.ramp_frames_remaining == 0
    assert state.ramp_target is None


def test_ramp_kicks_off_beyond_tolerance() -> None:
    """Estimate ~20 px from seed → returns interpolated corners, decision 'ramp'."""
    seed = _seed_quad()
    estimate = seed + np.array([20.0, 0.0], dtype=np.float32)
    state = HybridLockState(
        seed_corners=seed,
        tolerance_px=6.0,
        ramp_min_frames=3,
        ramp_motion_px_per_frame=2.0,
    )

    corners, decision, disp = state.step(estimate)

    assert decision == "ramp"
    # Should have started a ramp toward `estimate` and consumed step #1 already.
    # Because ramp duration was 10 frames, after first step we have 9 remaining.
    assert state.ramp_frames_remaining == 9
    assert state.ramp_target is not None
    np.testing.assert_allclose(state.ramp_target, estimate, atol=1e-5)
    # First-step corners are NOT yet at the target.
    assert not np.allclose(corners, estimate, atol=1e-3)
    # Displacement reported is the trigger displacement (estimate vs seed).
    assert disp > 6.0


def test_ramp_completes_over_n_frames() -> None:
    """20 px / 2 px-per-frame ⇒ 10 ramp frames; last_committed should land on target."""
    seed = _seed_quad()
    estimate = seed + np.array([20.0, 0.0], dtype=np.float32)
    state = HybridLockState(
        seed_corners=seed,
        tolerance_px=6.0,
        ramp_min_frames=3,
        ramp_motion_px_per_frame=2.0,
    )

    # Frame 1: triggers ramp + first step.
    state.step(estimate)
    assert state.ramp_frames_remaining == 9

    # Drive nine more steps (estimate stays the same for this test).
    for _ in range(9):
        corners, decision, _ = state.step(estimate)
        assert decision == "ramp"

    np.testing.assert_allclose(state.last_committed, estimate, atol=1e-4)
    assert state.ramp_frames_remaining == 0


def test_no_estimate_keeps_lock() -> None:
    """estimated_corners=None → returns seed, decision 'locked'."""
    seed = _seed_quad()
    state = HybridLockState(seed_corners=seed, tolerance_px=6.0)

    corners, decision, disp = state.step(None)

    assert decision == "locked"
    np.testing.assert_allclose(corners, seed, atol=1e-5)
    assert disp == 0.0
    assert state.ramp_frames_remaining == 0


def test_displacement_returned() -> None:
    """displacement_px is the seed-vs-estimate L2 mean — non-zero out of tolerance."""
    seed = _seed_quad()
    estimate = seed + np.array([15.0, 0.0], dtype=np.float32)
    state = HybridLockState(
        seed_corners=seed,
        tolerance_px=6.0,
        ramp_min_frames=3,
        ramp_motion_px_per_frame=2.0,
    )

    _, decision, disp = state.step(estimate)

    assert decision == "ramp"
    assert disp > 0.0
    # Mean L2 of [15, 0] over 4 corners == 15.0.
    assert abs(disp - 15.0) < 1e-4

    # Sanity: corner_displacement_px helper computes the same thing directly.
    assert abs(corner_displacement_px(estimate, seed) - 15.0) < 1e-4
