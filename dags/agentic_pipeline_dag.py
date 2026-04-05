"""Airflow DAG: end-to-end agentic fine-tuning pipeline."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount

DATA_DIR = "/opt/airflow/data"
TRAIN_PATH = os.path.join(DATA_DIR, "training_data.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "eval_prompts.jsonl")

TRAIN_MAX_ROWS = 5_000
EVAL_MAX_ROWS = 50

_HOST_ROOT = os.environ.get("HOST_PROJECT_ROOT", "")

TRAINER_IMAGE = "agentic-finetune-trainer:latest"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "agentic-phi3:latest")
GGUF_PATH = os.environ.get(
    "OLLAMA_GGUF_PATH",
    "/opt/airflow/models/merged_gguf/agentic-phi3.gguf",
)
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "1800"))


def _iter_jsonl(path: str):
    """Yield parsed dicts from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def ingest_and_format_data(max_rows: int = TRAIN_MAX_ROWS) -> None:
    """Download Nemotron interactive_agent data and format as ChatML JSONL."""
    if os.path.exists(TRAIN_PATH) and os.path.getsize(TRAIN_PATH) > 0:
        print(f"Training data already exists at {TRAIN_PATH}, skipping download.")
        return

    from huggingface_hub import hf_hub_download

    jsonl_path = hf_hub_download(
        repo_id="nvidia/Nemotron-Agentic-v1",
        filename="data/interactive_agent.jsonl",
        repo_type="dataset",
    )

    os.makedirs(DATA_DIR, exist_ok=True)

    written = 0
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for row in _iter_jsonl(jsonl_path):
            if written >= max_rows:
                break

            tools_json = json.dumps(row.get("tools", []), ensure_ascii=False)
            system_prompt = (
                "You are a helpful agent with access to the following tools: "
                f"{tools_json}. Use them when helpful to accomplish the user's goal."
            )

            conversation = [{"role": "system", "content": system_prompt}]
            for msg in row.get("messages", []):
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    content += "\nTool Call: " + json.dumps(tool_calls, ensure_ascii=False)
                conversation.append({"role": msg["role"], "content": content})

            f.write(json.dumps({"messages": conversation}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} training examples to {TRAIN_PATH}")


def prepare_eval_prompts(max_rows: int = EVAL_MAX_ROWS) -> None:
    """Extract held-out eval samples from the tail of the dataset."""
    if os.path.exists(EVAL_PATH) and os.path.getsize(EVAL_PATH) > 0:
        print(f"Eval data already exists at {EVAL_PATH}, skipping download.")
        return

    from huggingface_hub import hf_hub_download

    jsonl_path = hf_hub_download(
        repo_id="nvidia/Nemotron-Agentic-v1",
        filename="data/interactive_agent.jsonl",
        repo_type="dataset",
    )

    eval_rows = collections.deque(maxlen=max_rows)
    for row in _iter_jsonl(jsonl_path):
        eval_rows.append(row)

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(eval_rows)} eval examples to {EVAL_PATH}")


def _sha256_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _get_existing_model_names(base_url: str) -> set[str]:
    import requests

    resp = requests.get(f"{base_url}/api/tags", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return {model.get("name", "") for model in payload.get("models", [])}


def _ensure_blob_uploaded(base_url: str, file_path: Path, digest: str) -> None:
    import requests

    blob_url = f"{base_url}/api/blobs/{digest}"

    head_resp = requests.head(blob_url, timeout=60)
    if head_resp.status_code == 200:
        print(f"Blob already exists in Ollama: {digest}")
        return
    if head_resp.status_code not in (404,):
        head_resp.raise_for_status()

    with file_path.open("rb") as fh:
        upload_resp = requests.post(
            blob_url,
            data=fh,
            headers={"Content-Type": "application/octet-stream"},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )

    print(f"Ollama blob upload status: {upload_resp.status_code}")
    if upload_resp.text:
        print(upload_resp.text)
    upload_resp.raise_for_status()


def load_ollama_model() -> None:
    """Upload the GGUF blob and register it as an Ollama model."""
    gguf_path = Path(GGUF_PATH)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    print(f"Ollama base URL: {OLLAMA_BASE_URL}")
    print(f"Target model name: {MODEL_NAME}")
    print(f"GGUF path: {gguf_path}")

    existing_before = _get_existing_model_names(OLLAMA_BASE_URL)
    print(f"Existing Ollama models before import: {sorted(existing_before)}")

    digest = _sha256_digest(gguf_path)
    print(f"Computed GGUF digest: {digest}")

    _ensure_blob_uploaded(OLLAMA_BASE_URL, gguf_path, digest)

    create_payload = {
        "model": MODEL_NAME,
        "files": {
            gguf_path.name: digest,
        },
        "stream": False,
    }

    print("Creating Ollama model with payload:")
    print(json.dumps(create_payload, indent=2))

    import requests

    create_resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/create",
        json=create_payload,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    print(f"Ollama create status: {create_resp.status_code}")
    print(create_resp.text)
    create_resp.raise_for_status()

    existing_after = _get_existing_model_names(OLLAMA_BASE_URL)
    print(f"Existing Ollama models after import: {sorted(existing_after)}")

    if MODEL_NAME not in existing_after:
        raise RuntimeError(
            f"Import request finished, but '{MODEL_NAME}' is not present in /api/tags."
        )

    print(f"Ollama model registered successfully: {MODEL_NAME}")


def run_evaluate() -> None:
    """Run evaluate.py as a subprocess with real-time log streaming."""
    env = {
        **os.environ,
        "OLLAMA_URL": f"{OLLAMA_BASE_URL}/api/generate",
        "EVAL_PATH": EVAL_PATH,
        "AGENT_MODEL": MODEL_NAME,
        "JUDGE_MODEL": os.environ.get("JUDGE_MODEL", "llama3.2:3b"),
    }
    process = subprocess.Popen(
        ["python", "-u", "/opt/airflow/scripts/evaluate.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in process.stdout:
        print(line, end="")
    rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, "evaluate.py")


def _trainer_mounts() -> list[Mount]:
    if not _HOST_ROOT:
        raise RuntimeError(
            "HOST_PROJECT_ROOT is not set. Add it to docker/.env, e.g.:\n"
            "HOST_PROJECT_ROOT=C:/Users/you/Nemotron Agentic Fine-Tuning Pipeline"
        )

    return [
        Mount(source=f"{_HOST_ROOT}/data", target="/workspace/data", type="bind"),
        Mount(source=f"{_HOST_ROOT}/models", target="/workspace/models", type="bind"),
    ]


_DEFAULT_ARGS = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="agentic_finetuning_pipeline",
    default_args=_DEFAULT_ARGS,
    schedule=None,
    catchup=False,
    tags=["llm", "agentic", "finetune"],
) as dag:

    task_prepare_data = PythonOperator(
        task_id="ingest_and_format_nemotron",
        python_callable=ingest_and_format_data,
    )

    task_prepare_eval = PythonOperator(
        task_id="prepare_eval_prompts",
        python_callable=prepare_eval_prompts,
    )

    _mounts = _trainer_mounts() if _HOST_ROOT else []

    task_train_model = DockerOperator(
        task_id="train_model",
        image=TRAINER_IMAGE,
        command="python scripts/train_model.py",
        mounts=_mounts,
        mount_tmp_dir=False,
        environment={
            "TRAIN_DATA_PATH": "/workspace/data/training_data.jsonl",
            "OUTPUT_DIR": "/workspace/models/training_outputs",
            "LORA_SAVE_PATH": "/workspace/models/lora_agentic_model",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
        device_requests=[DeviceRequest(capabilities=[["gpu"]], count=-1)],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
    )

    task_export_gguf = DockerOperator(
        task_id="export_gguf",
        image=TRAINER_IMAGE,
        command="python scripts/export_gguf.py",
        mounts=_mounts,
        mount_tmp_dir=False,
        environment={
            "LORA_SAVE_PATH": "/workspace/models/lora_agentic_model",
            "GGUF_DIR": "/workspace/models/merged_gguf",
            "TMP_GGUF_DIR": "/workspace/gguf_tmp",
            "FINAL_GGUF_NAME": "agentic-phi3.gguf",
            "QUANT_METHOD": "q4_k_m",
            "LLAMA_CPP_DIR": "/opt/llama.cpp",
        },
        device_requests=[DeviceRequest(capabilities=[["gpu"]], count=-1)],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
    )

    task_load_ollama = PythonOperator(
        task_id="load_ollama_model",
        python_callable=load_ollama_model,
    )

    task_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluate,
    )

    (
        task_prepare_data
        >> task_prepare_eval
        >> task_train_model
        >> task_export_gguf
        >> task_load_ollama
        >> task_evaluate
    )