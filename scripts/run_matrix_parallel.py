#!/usr/bin/env python3
"""Run the benchmark matrix in parallel.

Launches every (config, gpu) combination as an independent Modal job
simultaneously. Each job becomes its own subprocess running modal_run.py,
so Modal spawns parallel containers (one per GPU).

Same result as run_matrix.sh — 12 experiment directories — but ~10x faster.

Usage
-----
    uv run python scripts/run_matrix_parallel.py

    # Override GPU list
    uv run python scripts/run_matrix_parallel.py --gpus T4 A100

    # Override config list
    uv run python scripts/run_matrix_parallel.py \\
        --configs configs/matrix/1prompt.yaml configs/matrix/11prompts.yaml

    # Lower benchmark count
    uv run python scripts/run_matrix_parallel.py --benchmark 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CONFIGS = [
    "configs/matrix/1prompt.yaml",
    "configs/matrix/5prompts.yaml",
    "configs/matrix/11prompts.yaml",
]
DEFAULT_GPUS = ["T4", "A100", "H100", "B200"]
DEFAULT_BENCHMARK = 3


def run_one(config: str, gpu: str, benchmark: int) -> tuple[str, str, int, str]:
    """Run one (config, gpu) combination via modal_run.py subprocess."""
    config_name = Path(config).stem
    name = f"{config_name}_{gpu}"
    cmd = [
        "uv",
        "run",
        "modal",
        "run",
        "scripts/modal_run.py",
        "--config",
        config,
        "--gpu",
        gpu,
        "--mode",
        "video",
        "--benchmark",
        str(benchmark),
        "--name",
        name,
    ]
    print(f"[start] {name}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"[done ] {name}  {status}  {elapsed:.1f}s", flush=True)
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[-500:]}", flush=True)
    return config, gpu, result.returncode, name


def main():
    parser = argparse.ArgumentParser(description="Run benchmark matrix in parallel")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--gpus", nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--benchmark", type=int, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--max-parallel", type=int, default=None, help="Max concurrent jobs (default: all)"
    )
    args = parser.parse_args()

    # Verify all configs exist.
    for cfg in args.configs:
        if not os.path.exists(cfg):
            print(f"ERROR: Config not found: {cfg}", file=sys.stderr)
            print(f"Run: uv run python scripts/collect_prompts.py --config {cfg}", file=sys.stderr)
            sys.exit(1)

    combos = [(cfg, gpu) for cfg in args.configs for gpu in args.gpus]
    n = len(combos)
    max_workers = args.max_parallel or n

    print("=" * 60)
    print(f"Launching {n} jobs in parallel (max_workers={max_workers})")
    print(f"Configs: {args.configs}")
    print(f"GPUs:    {args.gpus}")
    print(f"Benchmark runs per job: {args.benchmark}")
    print("=" * 60)

    t_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, cfg, gpu, args.benchmark) for cfg, gpu in combos]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    elapsed = time.perf_counter() - t_start

    successes = sum(1 for _, _, rc, _ in results if rc == 0)
    failures = n - successes

    print()
    print("=" * 60)
    print(f"Matrix complete in {elapsed:.1f}s")
    print(f"  Success: {successes}/{n}")
    print(f"  Failed:  {failures}/{n}")
    print("=" * 60)
    print("Results in experiments/")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
