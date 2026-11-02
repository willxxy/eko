# uv run python host.py --provider sglang \
# --model Qwen/Qwen3.8-27B \
# --sglang-url http://127.0.0.1:30001 \
# --classifier-checkpoint /p01/whan/ecg_agent/ECG-Agent-Play/outputs/2026-10-30/14-45-06/checkpoints/checkpoint_best.pt \
# --sandbox \
# --cwd /p01/whan/eko


uv run python -m benchmark.run --provider sglang \
--model Qwen/Qwen3.8-27B \
--sglang-url http://127.0.0.1:30001 \
--classifier-checkpoint /p01/whan/ecg_agent/ECG-Agent-Play/outputs/2026-10-30/14-45-06/checkpoints/checkpoint_best.pt \
--ecg-dir /p01/whan/ecg_agent/ECG-Agent-Play/data/ptbxl_preprocessed/mat_files \
--config /p01/whan/eko/benchmark/experiments/12-lead.json \
--max-samples 1