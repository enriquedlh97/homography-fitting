"""CLI entry point: `python -m banner_pipeline.eval --experiment <dir> ...`."""

from __future__ import annotations

import argparse
import sys

from banner_pipeline.eval.runner import run_eval


def parse_window(value: str) -> tuple[int, int]:
    """Parse `start:end` into (int, int)."""
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected start:end (e.g. 690:745)")
    a, b = value.split(":", 1)
    return int(a), int(b)


def main() -> int:
    parser = argparse.ArgumentParser(prog="banner_pipeline.eval")
    parser.add_argument("--experiment", required=True, help="Path to an experiment directory")
    parser.add_argument(
        "--reference",
        default=None,
        help='"auto" to resolve via configs/eval/reference.yaml, "off" to disable, '
        "or an explicit path to a gold experiment dir",
    )
    parser.add_argument(
        "--regions",
        default=None,
        help="Comma-separated subset of {back,left,floor,full,walkover}; default = all",
    )
    parser.add_argument(
        "--walkover-window",
        type=parse_window,
        default=None,
        help="Override walkover-window auto-detect, e.g. 690:745",
    )
    parser.add_argument("--original", default=None, help="Override the original video path")
    parser.add_argument("--clean", default=None, help="Override the clean-plate video path")
    args = parser.parse_args()

    regions_subset = (
        [r.strip() for r in args.regions.split(",") if r.strip()] if args.regions else None
    )

    payload, exit_code = run_eval(
        experiment_dir=args.experiment,
        reference_arg=args.reference,
        regions_subset=regions_subset,
        walkover_window_override=args.walkover_window,
        original_video=args.original,
        clean_video=args.clean,
    )

    # Brief stdout summary so the CLI output is informative.
    if payload:
        for region in ("back", "left", "floor", "full"):
            key = f"{region}_pass"
            if key in payload:
                fail_keys = payload.get(f"{region}_failed_metrics") or []
                status = "PASS" if payload[key] else f"FAIL ({', '.join(fail_keys)})"
                print(f"  [{region:5s}] {status}")
        if payload.get("any_regression"):
            print("  REGRESSION vs reference detected.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
