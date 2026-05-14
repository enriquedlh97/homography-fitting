#!/usr/bin/env bash
# Chain the pipeline run with the eval framework. Leaves run_experiment.py
# and modal_run.py untouched (per docs/EVALUATION.md).
#
# Usage:
#   scripts/run_and_eval.sh configs/experiments/<your_yaml>.yaml
#
# Env vars:
#   MODAL=1            -> dispatch through scripts/modal_run.py instead of run_experiment.py
#   GPU=B200|H200|...  -> Modal GPU type (default B200)
#   EVAL_REFERENCE=auto|off|<path>  -> --reference value (default: auto)

set -euo pipefail

CONFIG="${1:?usage: run_and_eval.sh <config.yaml>}"
REFERENCE="${EVAL_REFERENCE:-auto}"
GPU="${GPU:-B200}"

if [[ "${MODAL:-0}" == "1" ]]; then
    OUTPUT=$(uv run modal run scripts/modal_run.py \
        --config "$CONFIG" --gpu "$GPU" --mode video_hybrid \
        2>&1 | tee /tmp/modal_run.log | grep -oE 'experiments/[^ ]+' | tail -1)
else
    OUTPUT=$(uv run python scripts/run_experiment.py --config "$CONFIG" \
        2>&1 | tee /tmp/local_run.log | grep -oE 'experiments/[^ ]+' | tail -1)
fi

if [[ -z "${OUTPUT}" ]]; then
    echo "[run_and_eval] could not locate experiment dir from run output" >&2
    exit 1
fi

uv run python -m banner_pipeline.eval --experiment "$OUTPUT" --reference "$REFERENCE"
