"""Lightweight per-operation timer for profiling.

Disabled by default (zero overhead). Enable by setting ``PERF_ENABLED = True``
or by calling ``enable()`` before the code path you want to measure.

Usage
-----
    from banner_pipeline._perf import Timer, snapshot, reset, enable

    enable()
    reset()
    with Timer("inpaint.warp"):
        cv2.warpPerspective(...)
    print(snapshot())  # {"inpaint.warp": 0.0123}
"""

from __future__ import annotations

import time

PERF: dict[str, float] = {}
PERF_ENABLED: bool = False


def enable() -> None:
    """Enable timing collection."""
    global PERF_ENABLED
    PERF_ENABLED = True


def disable() -> None:
    """Disable timing collection."""
    global PERF_ENABLED
    PERF_ENABLED = False


def reset() -> None:
    """Clear all accumulated timings."""
    PERF.clear()


def snapshot() -> dict[str, float]:
    """Return a copy of the current timings (in seconds)."""
    return dict(PERF)


def snapshot_ms(divisor: int = 1) -> dict[str, float]:
    """Return timings in milliseconds, optionally divided by *divisor* (e.g. frame count)."""
    return {k: round(v * 1000.0 / divisor, 3) for k, v in PERF.items()}


class Timer:
    """Context manager that accumulates wall-clock time under *key*.

    Zero overhead when ``PERF_ENABLED`` is False (just two attribute reads).
    """

    __slots__ = ("key", "_t0")

    def __init__(self, key: str) -> None:
        self.key = key

    def __enter__(self) -> Timer:
        if PERF_ENABLED:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_args) -> None:
        if PERF_ENABLED:
            elapsed = time.perf_counter() - self._t0
            PERF[self.key] = PERF.get(self.key, 0.0) + elapsed
