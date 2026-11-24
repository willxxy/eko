#!/usr/bin/env bash

uv run python -m benchmark.run \
  --config benchmark/experiments/single-lead-I-gpt.json \
  --provider sglang \
  --model Qwen/Qwen3.8-27B \
  --sglang-url http://127.0.0.1:30001 \
  --output benchmark-results/single-lead-I/qwen3.8-27b/responses.jsonl \
  --classifier-checkpoint /p01/whan/ecg_agent/ECG-Agent-Play/outputs/2026-10-30/14-45-06/checkpoints/checkpoint_best.pt \
  --ecg-dir /p01/whan/ecg_agent/ECG-Agent-Play/data/ptbxl_preprocessed/mat_files \
  "$@"

uv run python -m benchmark.run \
  --config benchmark/experiments/single-lead-II-gpt.json \
  --provider sglang \
  --model Qwen/Qwen3.8-27B \
  --sglang-url http://127.0.0.1:30001 \
  --output benchmark-results/single-lead-II/qwen3.8-27b/responses.jsonl \
  --classifier-checkpoint /p01/whan/ecg_agent/ECG-Agent-Play/outputs/2026-10-30/14-45-06/checkpoints/checkpoint_best.pt \
  --ecg-dir /p01/whan/ecg_agent/ECG-Agent-Play/data/ptbxl_preprocessed/mat_files \
  "$@"

uv run python -m benchmark.run \
  --config benchmark/experiments/12-lead-gpt.json \
  --provider sglang \
  --model Qwen/Qwen3.8-27B \
  --sglang-url http://127.0.0.1:30001 \
  --output benchmark-results/12-lead/qwen3.8-27b/responses.jsonl \
  --classifier-checkpoint /p01/whan/ecg_agent/ECG-Agent-Play/outputs/2026-10-30/14-45-06/checkpoints/checkpoint_best.pt \
  --ecg-dir /p01/whan/ecg_agent/ECG-Agent-Play/data/ptbxl_preprocessed/mat_files \
  "$@"
