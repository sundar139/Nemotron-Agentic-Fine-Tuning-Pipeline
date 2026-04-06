"""Evaluate tool-calling accuracy using Ollama as an LLM judge.

Uses a persistent requests.Session for HTTP connection pooling across
hundreds of sequential Ollama API calls.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

_session = requests.Session()

RAW_OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_BASE_URL = RAW_OLLAMA_URL.rstrip("/").removesuffix("/api/generate")
OLLAMA_URL      = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_PS_URL   = f"{OLLAMA_BASE_URL}/api/ps"

AGENT_MODEL  = os.environ.get("AGENT_MODEL", "agentic-phi3:latest")
JUDGE_MODEL  = os.environ.get("JUDGE_MODEL", "llama3.2:3b")
EVAL_PATH    = os.environ.get("EVAL_PATH",   "data/eval_prompts.jsonl")

RESULTS_PATH = os.environ.get(
    "EVAL_RESULTS_PATH",
    str(Path(EVAL_PATH).parent / "eval_results.jsonl"),
)

WARMUP_TIMEOUT  = int(os.environ.get("EVAL_WARMUP_TIMEOUT",  "300"))
REQUEST_TIMEOUT = int(os.environ.get("EVAL_REQUEST_TIMEOUT", "120"))
MAX_RETRIES     = int(os.environ.get("EVAL_MAX_RETRIES",     "1"))


def _gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"


def check_vram(model: str) -> None:
    """Query /api/ps and warn if the model is running in split CPU/GPU mode."""
    try:
        resp = _session.get(OLLAMA_PS_URL, timeout=10)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size      = m.get("size", 0)
                size_vram = m.get("size_vram", 0)
                cpu_part  = size - size_vram
                print(f"  {model}: total={_gb(size)}  vram={_gb(size_vram)}  cpu={_gb(cpu_part)}")
                if cpu_part > 200 * 1024 ** 2:
                    print(
                        f"  WARNING: {_gb(cpu_part)} of model is running on CPU. "
                        f"Inference will be slow. Re-export the GGUF with q4_k_m "
                        f"quantization so the full model fits in VRAM."
                    )
                else:
                    print(f"  OK: {model} fully resident in VRAM.")
                return
        print(f"  {model} not found in /api/ps (not yet loaded or already evicted).")
    except Exception as exc:
        print(f"  WARN: could not reach /api/ps: {exc}")


def evict_model(model: str) -> None:
    """Tell Ollama to unload a model immediately (keepalive=0)."""
    print(f"Evicting {model} from VRAM...")
    try:
        resp = _session.post(
            OLLAMA_URL,
            json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"  {model} evicted.")
    except Exception as exc:
        print(f"  WARN: eviction request failed for {model}: {exc}")


def warmup_model(model: str) -> bool:
    """Send a lightweight prompt to force Ollama to load the model into VRAM."""
    print(f"Warming up {model} (timeout={WARMUP_TIMEOUT}s)...")
    try:
        resp = _session.post(
            OLLAMA_URL,
            json={"model": model, "prompt": "hi", "stream": False},
            timeout=WARMUP_TIMEOUT,
        )
        resp.raise_for_status()
        print(f"  {model} ready.")
        check_vram(model)
        return True
    except Exception as exc:
        print(f"  ERROR: warm-up failed for {model}: {exc}")
        return False


def call_ollama(
    model: str,
    prompt: str,
    *,
    retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT,
) -> str | None:
    """Call Ollama generate API with retry logic."""
    payload   = {"model": model, "prompt": prompt, "stream": False}
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = _session.post(OLLAMA_URL, json=payload, timeout=timeout)
            if not resp.ok:
                print(f"  WARN: HTTP {resp.status_code} {resp.text[:200]}")
                resp.raise_for_status()
            data = resp.json()
            if "response" not in data:
                raise KeyError(f"Missing 'response' in Ollama output: {list(data.keys())}")
            return data["response"].strip()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = 5 * (attempt + 1)
                print(f"  WARN: attempt {attempt + 1} failed: {exc}. Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  ERROR: all {retries + 1} attempts failed: {last_exc}")
    return None


def build_agent_prompt(tools: list, user_content: str) -> str:
    return (
        f"You are an agent with access to the following tools:\n{json.dumps(tools)}\n\n"
        f"User request: {user_content}\n\n"
        f"Respond with a JSON tool call if a tool is needed."
    )


def build_judge_prompt(tools: list, user_content: str, agent_output: str) -> str:
    return (
        f"You are a strict evaluator.\n"
        f"User request: {user_content}\n"
        f"Available tools: {json.dumps(tools)}\n"
        f"Assistant output: {agent_output}\n\n"
        f"Did the assistant output a valid JSON tool call that correctly uses "
        f"an appropriate tool to satisfy the user request? "
        f"Answer ONLY with YES or NO."
    )


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_result(results_path: Path, **kwargs) -> None:
    """Append a single result dict to the JSONL results file."""
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kwargs, ensure_ascii=False) + "\n")


def main() -> None:
    dataset_path = Path(EVAL_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Eval file not found: {dataset_path}. "
            f"Run the Airflow DAG first to generate eval_prompts.jsonl."
        )

    samples = load_samples(dataset_path)
    total   = len(samples)

    print("=" * 60)
    print("EVALUATION SETUP")
    print("=" * 60)
    print(f"Eval samples : {total}")
    print(f"Dataset path : {dataset_path}")
    print(f"Agent model  : {AGENT_MODEL}")
    print(f"Judge model  : {JUDGE_MODEL}")
    print(f"Ollama URL   : {OLLAMA_URL}")
    print(f"Warmup timeout  : {WARMUP_TIMEOUT}s")
    print(f"Request timeout : {REQUEST_TIMEOUT}s")
    print(f"Max retries     : {MAX_RETRIES}")
    print(f"Results path : {RESULTS_PATH}")

    results_path = Path(RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("")

    print("=" * 60)
    print(f"PASS 1/2 — Agent generations ({total} samples)")
    print("=" * 60)

    if not warmup_model(AGENT_MODEL):
        raise RuntimeError(f"Could not load agent model {AGENT_MODEL}. Aborting.")

    agent_outputs:  list[str | None] = []
    agent_prompts:  list[str]        = []
    skipped_indices: set[int]        = set()

    pass1_start = time.time()

    for i, sample in enumerate(samples):
        tools    = sample.get("tools", [])
        messages = sample.get("messages", [])
        user_turns = [m for m in messages if m.get("role") == "user"]

        if not user_turns:
            print(f"  [{i:>4}/{total}] SKIP — no user turn")
            agent_outputs.append(None)
            agent_prompts.append("")
            skipped_indices.add(i)
            continue

        user_content = user_turns[0].get("content", "")
        prompt       = build_agent_prompt(tools, user_content)
        agent_prompts.append(prompt)

        t0     = time.time()
        output = call_ollama(AGENT_MODEL, prompt)
        elapsed = time.time() - t0

        agent_outputs.append(output)
        status = "OK" if output else "EMPTY"
        print(f"  [{i:>4}/{total}] {status}  {elapsed:.1f}s")

    pass1_elapsed = time.time() - pass1_start
    generated     = total - len(skipped_indices)
    print(
        f"\nPass 1 done in {pass1_elapsed:.1f}s — "
        f"{generated} generated, {len(skipped_indices)} skipped."
    )

    print("\nEvicting agent model before judge pass...")
    evict_model(AGENT_MODEL)
    time.sleep(2)

    print("=" * 60)
    print(f"PASS 2/2 — Judge evaluations ({total} samples)")
    print("=" * 60)

    if not warmup_model(JUDGE_MODEL):
        raise RuntimeError(f"Could not load judge model {JUDGE_MODEL}. Aborting.")

    correct    = 0
    evaluated  = 0
    results_list: list[dict] = []
    pass2_start = time.time()

    for i, sample in enumerate(samples):
        if i in skipped_indices or not agent_outputs[i]:
            status_str = (
                "SKIP — no user turn" if i in skipped_indices
                else "SKIP — empty agent output"
            )
            print(f"  [{i:>4}/{total}] {status_str}")
            result = dict(
                index=i,
                passed=False,
                agent_output=agent_outputs[i],
                judge_answer="",
                skipped=True,
            )
            results_list.append(result)
            write_result(results_path, **result)
            continue

        tools      = sample.get("tools", [])
        messages   = sample.get("messages", [])
        user_turns = [m for m in messages if m.get("role") == "user"]
        user_content = user_turns[0].get("content", "")

        judge_prompt = build_judge_prompt(tools, user_content, agent_outputs[i] or "")

        t0           = time.time()
        judge_answer = call_ollama(JUDGE_MODEL, judge_prompt)
        elapsed      = time.time() - t0

        passed      = bool(judge_answer and judge_answer.strip().upper().split()[0] in ("YES", "YES.", "YES,"))
        correct    += int(passed)
        evaluated  += 1

        running_acc = correct / evaluated
        status      = "PASS" if passed else "FAIL"
        print(
            f"  [{i:>4}/{total}] {status}  {elapsed:.1f}s  "
            f"running_acc={running_acc:.3f} ({correct}/{evaluated})"
        )

        result = dict(
            index=i,
            passed=passed,
            agent_output=agent_outputs[i],
            judge_answer=judge_answer,
            skipped=False,
        )
        results_list.append(result)
        write_result(results_path, **result)

    pass2_elapsed = time.time() - pass2_start

    print("\nFinal VRAM check:")
    check_vram(JUDGE_MODEL)

    accuracy      = correct / evaluated if evaluated else 0.0
    total_elapsed = pass1_elapsed + pass2_elapsed

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total samples : {total}")
    print(f"Evaluated     : {evaluated}")
    print(f"Skipped       : {total - evaluated}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.3f}  ({correct}/{evaluated})")
    print(f"Pass 1 time   : {pass1_elapsed:.1f}s")
    print(f"Pass 2 time   : {pass2_elapsed:.1f}s")
    print(f"Total time    : {total_elapsed:.1f}s")
    print("=" * 60)
    print(f"Results written to: {results_path}")

    _session.close()

    if accuracy < 0.1 and evaluated >= 10:
        print(
            "\nNOTE: Accuracy is below 10%. This may indicate the model needs "
            "more training steps, a larger dataset, or the eval prompts are "
            "out of distribution."
        )


if __name__ == "__main__":
    main()