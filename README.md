# Agentic ECG Interpretation

## System Configuration

### WILLIAM's Configs:

RHEL system with 8xH100 NVL GPUs.

- `nvcc --version` == cuda132
- `nvidia-smi` Driver Version == 570.158.01
- `nvidia-smi` CUDA Version == 13.2
- I had to upgrade my gcc version to 13. If you run into any kind of gcc bug, let me know and I can try to help.

For SGLANG hosting:
- `sglang` version is 0.5.14. We built it locally via cloning the repo.
- `sglang_kernel` version is 0.4.4+cu129. Somehow this doesn't break (12 and 13 are I think bidirectionally compatible so maybe this is the case? But not sure.). We should make it consistent, but for now, lets not worry about this since it works lol.
- `SGL DeepGEMM` 0.1.3+cu129. Similar situation to above but works! I think this one is the one coupled together with `sglang_kernel`.
- `triton` version is 3.6.0. Make sure to delete the `.triton` folder before installation if you used triton before. Triton has this nasty behavior of using old cached things even on fresh new installs on new environment.
- `transformers` version 5.8.1. This is important so make sure to get this exact version. If you don't you may end up to some kind of "config doesn't exist" bug (I did).

For EKO:
Just use the uv.lock and pyprojec.toml files as reference. I just `uv venv` then `source .venv/bin/activate` then `uv sync` in side the eko repo.

## Backends
We support a couple of backends for the agent.
- huggingface
- sglang
- openrouter

### SGLANG
Honestly, SGLANG reminds me of old pytorch where just installing it is a pain in the ***. BUT, after doing it a couple times I found many workarounds and got a working MACHINE specific config. Since SGLANG touches a lot of low level stuff, having a machine specific installation process is quite common. Therefore, what I suggest is to go to sglang and just try to install it using their documented way. If it doesn't work, just let me know and I can try to help debug.

For hosting sglang, this is done completely in another folder and environment. Just set it up, and follow their direction of hosting a llm. It should look something like this in their repo

```
#!/usr/bin/env bash

export FLASHINFER_USE_CUDA_NORM=1
MULTINODE=${NNODES:+--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $DIST_INIT_ADDR}
sglang serve \
    --model-path "${MODEL:-Qwen/Qwen3.8-27B}" \
    --host 0.0.0.0 --port "${PORT:-30001}" \
    --tp-size "${TP:-1}" --dp-size "${DP:-1}" \
    --cuda-graph-max-bs "${CUDA_GRAPH_BS:-256}" \
    ${MEM_FRAC:+--mem-fraction-static $MEM_FRAC} $MULTINODE
```

If you have questions, explore the sglang documentation/code or take a look at this [repo](https://github.com/willxxy/sglang-play) where i document some common failures and interesting observations (e.g., why we need export FLASHINFER_USE_CUDA_NORM=1 lol).

## Running Eko

The default provider is `sglang`. Every provider requires `--model`.

The pipeline is `host.py` (provider, sandbox) → `eko.py` (model → Python
→ result → model). `ui.py` handles the terminal. The model and its credentials
stay in the host. Background processes send text or images through `EKO_SESSION`;
`EKO_WORLD` connects an optional external world socket.

Use OpenRouter with `OPENROUTER_API_KEY` set:

```bash
OPENROUTER_API_KEY=$API_KEY uv run python host.py --provider openrouter --model nvidia/nemotron-3-ultra-550b-a55b:free
```

Load a Hugging Face chat model by repository ID or local directory:

```bash
uv run python host.py --provider huggingface --model Qwen/Qwen3.5-0.8B
```

Connect to an SGLang server you already run in a separate environment:

```bash
uv run python host.py --provider sglang \
--model Qwen/Qwen3.8-27B \
--sglang-url http://127.0.0.1:30001
```

For Qwen3 reasoning models, launch SGLang with `--reasoning-parser qwen3` to
[separate reasoning from final content](https://docs.sglang.io/docs/advanced_features/separate_reasoning).

## ECG-Agent adaptation

For all current measurements in `toolbox/measurements/`, we follow [ECG-Agent](https://github.com/gustmd0121/ECG-Agent) exactly in calculating them. We also maintain the exact tools used in ECG-Agent, and shown to the LLM.

## Benchmark experiments

The three main configs cover ECG-Agent's response accuracy/completeness,
tool faithfulness, dialogue naturalness, and CEFR adherence. NAP and explanation
TIoU are excluded. Each uses the original test questions, generated history, and
the corresponding published `-gt` dataset as judge references.

| Dialogue leads | Config |
| --- | --- |
| 12-lead | [`12-lead.json`](benchmark/experiments/12-lead.json) |
| Lead I | [`single-lead-I.json`](benchmark/experiments/single-lead-I.json) |
| Lead II | [`single-lead-II.json`](benchmark/experiments/single-lead-II.json) |

Run from the repository root with SGLang already running. The generation script
contains the ECG directory and classifier checkpoint paths; edit them if needed.
Both scripts process all three lead configurations.

```bash
uv sync
bash scripts/run_sglang.sh

export OPENROUTER_API_KEY="your-key"
bash scripts/eval.sh
```

For a small check, append `--max-samples 3` to both script commands. Other command
options are forwarded to every configuration. Outputs are in
`benchmark-results/{12-lead,single-lead-I,single-lead-II}/`;
each contains `responses.jsonl`, `scores.jsonl`, and `scores.summary.json`.

Shared keys: `dataset`, `split`, `max_samples`, `leads`.
`run` and `score` hold command options using underscores. CLI overrides JSON;
paths are relative to the working directory. Use separate outputs when changing settings.

`dataset` selects inference questions. The score-only
`reference_dataset` and `reference_split` select judge answers. References are
matched by ECG ID and exact user question, independently of row or turn order.
`max_samples` limits inference cases; the full reference dataset is searched.
Unmatched answers are reported as `missing_reference` and excluded from score means.
Duplicate reference questions and mismatched generated questions stop scoring.

The agent keeps its generated conversation and tool results throughout each case.
Reference answers are loaded only by the scorer. Each turn receives tool examples
and executes tools for new findings.
Saved answers and judge inputs exclude Qwen `<think>` reasoning, including older
response files. Execution traces retain the original model output.

Summary fields correspond to the paper's metrics:

| Paper result | Summary field |
| --- | --- |
| Table 1: post-tool accuracy/completeness | `tool_response` (pooled post-classification and post-measurement responses) |
| Table 2: direct accuracy/completeness | `metrics.direct_response` |
| Table 4: faithfulness | `metrics.faithfulness` (binary mean; multiply by 100 for percent) |
| Table 5: naturalness/CEFR adherence | `metrics.dialogue` |

Each mean includes only successfully scored tasks and is accompanied by its count.
Per-category post-classification and post-measurement scores are also retained.
The paper averages Tables 2 and 5 across lead configurations; these files report
each configuration separately. See the [evaluation protocol](https://arxiv.org/html/2601.20323v1#S3.SS3).

Input: ECG-Agent's preprocessed 12-lead MAT files (`HR25.mat` for dataset filename
`HR00025.mat`). Tools currently receive all 12 leads in every dialogue configuration;
measurements use lead II. Classification requires a checkpoint.

Scoring requires `OPENROUTER_API_KEY`; `--dry-run` saves prompts only.
Execution defaults to Linux Bubblewrap; use `--execution local` to run locally.
