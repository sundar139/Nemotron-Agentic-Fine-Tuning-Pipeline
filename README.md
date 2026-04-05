# Agentic LLM Fine-Tuning Pipeline

An end-to-end pipeline for fine-tuning a tool-calling language model, from raw data to local deployment and automated evaluation — orchestrated by Apache Airflow, trained with Unsloth QLoRA, and served through Ollama.

## Why This Project Exists

Large language models can answer questions, but making them *act* — calling the right tool with the right arguments in a multi-turn conversation — requires targeted fine-tuning on agentic data. This pipeline automates the entire workflow so that every iteration from data prep through evaluation is a single DAG trigger.

## Tech Stack

| Layer | Tool | Role |
|---|---|---|
| **Orchestration** | Apache Airflow 2.9 | Sequences the six pipeline stages |
| **Data** | `nvidia/Nemotron-Agentic-v1` | 5 000 multi-turn tool-use trajectories |
| **Training** | Unsloth + TRL `SFTTrainer` | 4-bit QLoRA on Phi-3-mini (3.8 B) |
| **Export** | llama.cpp `llama-quantize` | Merge LoRA → Q4_K_M GGUF (~2 GB) |
| **Serving** | Ollama | Local inference from the GGUF |
| **Evaluation** | LLM-as-judge (Llama 3.2 3B) | Automated tool-call accuracy scoring |
| **Infra** | Docker Compose + NVIDIA Container Toolkit | GPU-accelerated containers on Windows/WSL2 |

---

## Architecture

The Airflow DAG executes six sequential tasks:

```
ingest_and_format_nemotron
        │
        ▼
 prepare_eval_prompts
        │
        ▼
    train_model          ← DockerOperator (GPU container)
        │
        ▼
    export_gguf          ← DockerOperator (GPU container)
        │
        ▼
  load_ollama_model      ← Blob upload + /api/create
        │
        ▼
   evaluate_model        ← 2-pass: agent generation → judge scoring
```

**Data flow:**

```
HuggingFace Hub ──→ data/training_data.jsonl ──→ Unsloth trainer ──→ models/lora_agentic_model/
                ──→ data/eval_prompts.jsonl                     ──→ models/merged_gguf/*.gguf
                                                                ──→ Ollama (agentic-phi3:latest)
                                                                ──→ data/eval_results.jsonl
```

---

## Design Decisions

### Why Phi-3-mini over Llama 3 8B?

Llama 3 8B in 4-bit consumes ~5–6 GB of VRAM for inference alone, leaving almost no headroom for optimizer states on an 8 GB card. Phi-3-mini (3.8 B params) needs only ~1.8 GB at int4, giving ample room for gradient checkpointing, LoRA adapters, and activation memory during training.

### Why QLoRA through Unsloth?

Unsloth patches the attention kernels with fused Triton implementations, cutting training VRAM by ~43% and wall-clock time by ~44% compared to stock HuggingFace PEFT. Combined with 4-bit NormalFloat quantization, this makes full fine-tuning on consumer GPUs practical.

### Why Docker-based training tasks?

Training and GGUF export run inside a `DockerOperator` container with the NVIDIA runtime. This isolates CUDA/PyTorch/Unsloth dependencies from Airflow's own Python environment and ensures bit-for-bit reproducibility across machines.

### Why LLM-as-judge instead of BLEU/ROUGE?

Tool-call correctness is binary (right tool + right arguments), not a text-similarity problem. A small judge model (Llama 3.2 3B) can reliably parse the agent's JSON output and answer YES/NO, giving a meaningful accuracy metric without human labeling.

---

## Tuned Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `MAX_SEQ_LENGTH` | 2048 | Prevents truncation of multi-turn tool-call conversations |
| `LORA_R` | 16 | Good capacity vs. memory trade-off; r=32 caused overfitting |
| `LORA_ALPHA` | 16 | Equal to r for stable LoRA scaling |
| `MAX_STEPS` | 625 | ~1 epoch over 5 000 examples (batch 1 × grad accum 8) |
| `LEARNING_RATE` | 1e-4 | Moderate rate; 2e-4 was too aggressive for this data size |
| `WARMUP_STEPS` | 50 | Gradual ramp avoids early instability |
| `QUANT_METHOD` | q4_k_m | Best quality-to-size ratio for 8 GB VRAM inference |
| `EVAL_MAX_ROWS` | 50 | Enough for signal; keeps eval pass under 15 minutes |

---

## Challenges & Solutions

### Data truncation at 1024 tokens

**Problem:** Assistant tool-call responses in Nemotron conversations frequently exceed 1024 tokens. The model was training on truncated examples, learning incomplete tool calls.

**Fix:** Increased `MAX_SEQ_LENGTH` from 1024 → 2048. This doubled context without exceeding VRAM thanks to Unsloth's memory-efficient attention.

### Overfitting at 2 epochs

**Problem:** Training for 2 epochs (1250 steps, r=32) dropped accuracy from 44% to 32%. The model memorized training patterns instead of generalizing.

**Fix:** Reverted to 1 epoch (625 steps) with r=16. The smaller adapter and single pass gave the best generalization.

### HTTP connection overhead during evaluation

**Problem:** 400+ sequential Ollama API calls created a new TCP connection each time, adding ~200 ms of overhead per request.

**Fix:** Replaced `requests.post()` with a module-level `requests.Session()` for connection pooling, cutting total eval time significantly.

### VRAM management across model swaps

**Problem:** The agent model and judge model can't coexist on 8 GB VRAM. Failing to evict one before loading the other caused OOM errors.

**Fix:** Evaluation runs in two passes — agent generation first, then explicit eviction (`keepalive=0`), then judge scoring — with VRAM checks after each model load.

---

## Getting Started

### Prerequisites

| Tool | Notes |
|---|---|
| **WSL2** | Required for Docker Desktop GPU passthrough on Windows |
| **Docker Desktop** | Enable WSL2 backend + NVIDIA Container Toolkit |
| **NVIDIA GPU** | 8 GB VRAM minimum (RTX 3070 / 4060 Ti or better) |
| **uv** | `winget install astral-sh.uv` |

### 1. Clone and configure

```powershell
git clone <repo-url>
cd "Nemotron Agentic Fine-Tuning Pipeline"

# Create your local environment file
Copy-Item docker\.env.example docker\.env
# Edit docker\.env — fill in HF_TOKEN and HOST_PROJECT_ROOT
```

### 2. Build the trainer image

```powershell
docker build --no-cache -f docker/Dockerfile.trainer -t agentic-finetune-trainer:latest .
```

### 3. Start the Airflow stack

```powershell
cd docker
docker compose up -d
```

Open **http://localhost:8080** (credentials: `airflow` / `airflow`).

### 4. Trigger the pipeline

From the Airflow UI, unpause and trigger `agentic_finetuning_pipeline`. Or via CLI:

```powershell
docker compose exec airflow-scheduler airflow dags trigger agentic_finetuning_pipeline
```

The full pipeline takes **45–60 minutes** on an RTX-class GPU:
- Data ingestion: ~2 min
- Training (625 steps): ~25–35 min
- GGUF export: ~5 min
- Ollama registration: ~2 min
- Evaluation (50 samples): ~10 min

### 5. Check results

```powershell
# View evaluation accuracy from Airflow task logs, or:
Get-Content data\eval_results.jsonl | ConvertFrom-Json | Select-Object -First 5
```

---

## Project Structure

```
├── dags/
│   └── agentic_pipeline_dag.py     # Airflow DAG — orchestrates all 6 stages
├── scripts/
│   ├── train_model.py              # Unsloth QLoRA fine-tuning
│   ├── export_gguf.py              # LoRA merge + GGUF quantization
│   └── evaluate.py                 # 2-pass LLM-as-judge evaluation
├── docker/
│   ├── docker-compose.yml          # Airflow + Postgres + Ollama stack
│   ├── Dockerfile.airflow          # Airflow image with HF Hub + Docker provider
│   ├── Dockerfile.trainer          # CUDA 12.1 + Unsloth + llama.cpp
│   └── .env.example                # Template for secrets and config
├── models/
│   └── Modelfile                   # Ollama model definition (Phi-3 chat template)
├── data/                           # Generated at runtime (gitignored)
├── pyproject.toml                  # Python dependencies (uv-managed)
├── uv.lock                         # Locked dependency versions
└── .gitignore
```

---

## Local Development (without Docker)

```powershell
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[gpu]"

# Train
python scripts/train_model.py

# Export
python scripts/export_gguf.py

# Register with Ollama
cd models && ollama create agentic-phi3 -f Modelfile

# Evaluate
python scripts/evaluate.py
```

---

## Roadmap

- [ ] Scale training data beyond 5 000 examples for higher accuracy
- [ ] Add per-tool-type accuracy breakdown in evaluation
- [ ] Support Llama 3.1 8B as an alternate base model (requires ≥12 GB VRAM)
- [ ] Integrate Weights & Biases for experiment tracking
- [ ] Add CI pipeline for automated smoke tests

---

## References

- [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) — Training dataset
- [Unsloth](https://github.com/unslothai/unsloth) — Memory-efficient fine-tuning
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) — Supervised fine-tuning trainer
- [Ollama](https://ollama.ai) — Local model serving
- [Apache Airflow](https://airflow.apache.org) — Workflow orchestration
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF quantization tooling
