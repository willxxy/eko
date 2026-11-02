#!/usr/bin/env bash
set -euo pipefail

for leads in single-lead-I single-lead-II; do
  uv run python -m benchmark.run \
    --config "benchmark/experiments/${leads}.json" \
    --provider sglang \
    --model Qwen/Qwen3.8-27B \
    --sglang-url http://127.0.0.1:30001 \
    --classifier-checkpoint /p01/whan/ecg_agent/ECG-Agent-Play/outputs/2026-10-30/14-45-06/checkpoints/checkpoint_best.pt \
    --ecg-dir /p01/whan/ecg_agent/ECG-Agent-Play/data/ptbxl_preprocessed/mat_files \
    "$@"
done
