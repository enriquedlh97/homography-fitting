#!/usr/bin/env python3
"""Thin CLI wrapper around `python -m banner_pipeline.eval`.

Identical behavior; exists so `scripts/run_and_eval.sh` and ad-hoc CLI users
have a discoverable entry point under `scripts/`. See docs/EVALUATION.md.
"""

from __future__ import annotations

import sys

from banner_pipeline.eval.__main__ import main


if __name__ == "__main__":
    sys.exit(main())
