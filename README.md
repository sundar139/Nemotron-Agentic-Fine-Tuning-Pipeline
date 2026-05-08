# Nemotron-Agentic-Fine-Tuning-Pipeline

Reproducible ML pipeline for fine-tuning a tool-calling LLM, exporting a GGUF artifact, serving it locally with Ollama, and evaluating behavior with an LLM judge.

![Python](https://img.shields.io/badge/Python-3.11%20to%203.12-3776AB)
![Orchestration](https://img.shields.io/badge/Orchestration-Apache%20Airflow-017CEE)
![Training](https://img.shields.io/badge/Training-Unsloth%20QLoRA-0F766E)
![Serving](https://img.shields.io/badge/Serving-Ollama-111111)
![Runtime](https://img.shields.io/badge/Runtime-Docker-2496ED)
![Model%20Format](https://img.shields.io/badge/Model%20Format-GGUF-6B7280)

## Overview

Fine-tuning tool-calling models is often fragmented across notebooks, ad hoc scripts, and manual deployment steps. This repository packages that lifecycle into one Airflow DAG: ingest and format agentic data, fine-tune with Unsloth QLoRA, export GGUF, register in Ollama, and run LLM-as-judge evaluation. It is designed as an engineering portfolio project that emphasizes reproducibility and operational clarity.

## Why this project exists

Agentic model work is most useful when training, packaging, serving, and evaluation are connected in a repeatable pipeline. This project demonstrates that end-to-end workflow using practical, local-first infrastructure.

## Who this is for

- ML engineers iterating on tool-calling behavior
- MLOps engineers building reproducible local training/evaluation workflows
- Hiring managers and recruiters reviewing production-oriented AI infrastructure work

## Key features

- Single Airflow DAG for the full model lifecycle
- Data ingestion/formatting from `nvidia/Nemotron-Agentic-v1`
- Unsloth + TRL `SFTTrainer` QLoRA fine-tuning
- LoRA merge and GGUF export for efficient local inference
- Ollama model registration and serving
- Two-pass LLM-as-judge evaluation with JSONL outputs
- Dockerized services and GPU training/export tasks

## Architecture overview

The DAG `agentic_finetuning_pipeline` executes:

1. `ingest_and_format_nemotron`
2. `prepare_eval_prompts`
3. `train_model` (DockerOperator)
4. `export_gguf` (DockerOperator)
5. `load_ollama_model`
6. `evaluate_model`

Primary run artifacts:

- `data/training_data.jsonl`
- `data/eval_prompts.jsonl`
- `models/training_outputs/`
- `models/merged_gguf/agentic-phi3.gguf`
- `data/eval_results.jsonl`

## Training and evaluation metrics

The following metrics are from the current repository artifacts (`checkpoint-625`, `eval_results.jsonl`, and local Ollama runtime) as of 2026-05-07.

### A) Nemotron-Agentic-v1 training sample count

- Training samples used: **5,000** (`data/training_data.jsonl` line count)

### B) Training loss (curve + final)

Representative points from `models/training_outputs/checkpoint-625/trainer_state.json`:

| Step | Train loss |
| ---: | ---: |
| 5 | 1.6739 |
| 25 | 0.5430 |
| 50 | 0.0681 |
| 100 | 0.0064 |
| 200 | 0.0026 |
| 300 | 0.0004 |
| 400 | 0.0004 |
| 500 | 0.0005 |
| 625 | **0.0002** |

### C) LLM judge score statistics across eval samples

Computed from `data/eval_results.jsonl` (`passed` mapped to 1, `failed` mapped to 0):

- Eval samples: **50**
- Mean judge score: **0.3200**
- Std. dev. (population): **0.4665**
- Pass rate: **32.0%** (16/50)

### D) GGUF artifact size

- `models/merged_gguf/agentic-phi3.gguf`: **2,318,919,552 bytes**
- Approx size: **2,211.49 MiB** (**2.16 GiB**)

### E) Ollama inference latency (tokens/sec)

Measured on local Ollama with model `agentic-phi3` over 3 non-streaming generations:

- Run 1: 96.50 tokens/sec
- Run 2: 112.59 tokens/sec
- Run 3: 77.18 tokens/sec
- Mean throughput: **95.42 tokens/sec**
- Std. dev.: **14.47 tokens/sec**

## Quickstart

Requirements:

- Docker Desktop with GPU support (WSL2 + NVIDIA runtime on Windows)
- NVIDIA GPU
- Hugging Face token

### 1) Clone and configure

```powershell
git clone <repo-url>
cd Nemotron-Agentic-Fine-Tuning-Pipeline

Copy-Item docker/.env.example docker/.env
# Set AIRFLOW_SECRET_KEY, HF_TOKEN, HOST_PROJECT_ROOT in docker/.env
```

### 2) Build trainer image

```powershell
docker build -f docker/Dockerfile.trainer -t agentic-finetune-trainer:latest .
```

### 3) Start services

```powershell
cd docker
docker compose up -d
```

Airflow UI: `http://localhost:8080` (default: `airflow` / `airflow`)

### 4) Trigger pipeline

```powershell
docker compose exec airflow-scheduler airflow dags trigger agentic_finetuning_pipeline
```

### 5) Inspect evaluation output

```powershell
Get-Content ../data/eval_results.jsonl -TotalCount 5
```

## Example workflow

1. Download and format Nemotron trajectories into training/eval JSONL files.
2. Fine-tune `unsloth/Phi-3-mini-4k-instruct` using QLoRA.
3. Export quantized GGUF (`q4_k_m`) from the fine-tuned adapter.
4. Register model in Ollama as `agentic-phi3:latest`.
5. Evaluate generated tool calls with a judge model and persist results.

## Repository structure

```text
.
├── dags/
│   └── agentic_pipeline_dag.py
├── scripts/
│   ├── train_model.py
│   ├── export_gguf.py
│   └── evaluate.py
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.airflow
│   ├── Dockerfile.trainer
│   └── .env.example
├── models/
│   ├── Modelfile
│   ├── merged_gguf/
│   └── training_outputs/
├── data/
├── pyproject.toml
└── uv.lock
```

## Evaluation approach

The evaluator runs in two passes:

1. Agent generation pass using `agentic-phi3:latest`
2. Judge scoring pass using `llama3.2:3b` (default)

Each sample is scored for tool-call correctness and appended to `data/eval_results.jsonl`.

## Local development

For script-level debugging outside the DAG, run the same trainer image directly:

```powershell
docker run --rm --gpus all `
    -v "${PWD}:/workspace" `
    -e HF_TOKEN="$env:HF_TOKEN" `
    -e TRAIN_DATA_PATH=/workspace/data/training_data.jsonl `
    -e OUTPUT_DIR=/workspace/models/training_outputs `
    -e LORA_SAVE_PATH=/workspace/models/lora_agentic_model `
    agentic-finetune-trainer:latest `
    python scripts/train_model.py
```

Use equivalent `docker run` calls for `scripts/export_gguf.py` and `scripts/evaluate.py` when isolating specific stages.

## Roadmap

- [ ] Scale training data beyond the current subset size
- [ ] Add per-tool-category evaluation breakdown
- [ ] Support alternate base models as configurable options
- [ ] Add experiment tracking integration
- [ ] Add automated smoke checks for pipeline setup

## References

- [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1)
- [Unsloth](https://github.com/unslothai/unsloth)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Ollama](https://ollama.ai)
- [Apache Airflow](https://airflow.apache.org)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
