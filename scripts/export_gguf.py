"""Merge LoRA adapter into base model and export a quantized GGUF for Ollama."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import torch
from unsloth import FastLanguageModel

WORKSPACE_DIR = Path("/workspace")

LORA_PATH_RAW    = os.environ.get("LORA_SAVE_PATH",  "/workspace/models/lora_agentic_model")
GGUF_DIR_RAW     = os.environ.get("GGUF_DIR",        "/workspace/models/merged_gguf")
TMP_GGUF_DIR_RAW = os.environ.get("TMP_GGUF_DIR",    "/workspace/gguf_tmp")
FINAL_GGUF_NAME  = os.environ.get("FINAL_GGUF_NAME", "agentic-phi3.gguf")
QUANT_METHOD     = os.environ.get("QUANT_METHOD",    "q4_k_m")
LLAMA_CPP_DIR    = Path(os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp"))

MAX_SEQ_LENGTH = 2048  # Must match training seq length

QUANT_CANDIDATES = [
    LLAMA_CPP_DIR / "llama-quantize",
    LLAMA_CPP_DIR / "quantize",
    LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
    LLAMA_CPP_DIR / "build" / "bin" / "quantize",
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = WORKSPACE_DIR / path
    return path.resolve()


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_dir(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(p.name for p in path.iterdir())


def find_quantize_binary() -> Path | None:
    for path in QUANT_CANDIDATES:
        if path.exists() and path.is_file():
            return path
    return None


def build_llama_cpp_if_needed() -> Path:
    """Locate or build the llama-quantize binary."""
    existing = find_quantize_binary()
    if existing is not None:
        return existing

    if not LLAMA_CPP_DIR.exists():
        raise RuntimeError(f"llama.cpp directory does not exist: {LLAMA_CPP_DIR}")
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is not installed in the trainer container.")

    build_dir = LLAMA_CPP_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake", "-S", str(LLAMA_CPP_DIR), "-B", str(build_dir),
            "-DGGML_CUDA=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release",
         "--target", "llama-quantize", "-j"],
        check=True,
    )
    built = find_quantize_binary()
    if built is None:
        raise RuntimeError(
            "Could not build llama.cpp quantize binary. "
            f"Checked: {', '.join(str(p) for p in QUANT_CANDIDATES)}"
        )
    return built


def pick_quantized_gguf(tmp_dir: Path) -> Path:
    """Select the quantized GGUF, skipping the BF16 intermediate.

    Unsloth writes two files: a BF16 intermediate (~7 GB) and the quantized
    output (~2 GB for Q4_K_M). We must pick the quantized one.

    Priority: exact match > any non-BF16 (smallest) > smallest overall.
    """
    quant_upper   = QUANT_METHOD.upper().replace("-", "_")
    expected_name = f"unsloth.{quant_upper}.gguf"

    all_ggufs: list[Path] = sorted(tmp_dir.glob("*.gguf")) if tmp_dir.exists() else []

    if not all_ggufs:
        raise RuntimeError(
            f"No GGUF file was produced. Checked directory: {tmp_dir}"
        )

    for p in all_ggufs:
        if p.name == expected_name:
            print(f"Found expected quantized GGUF: {p.name}")
            return p

    non_bf16 = [p for p in all_ggufs if "BF16" not in p.name.upper()]
    if non_bf16:
        non_bf16.sort(key=lambda p: p.stat().st_size)
        print(f"Expected '{expected_name}' not found; using '{non_bf16[0].name}' (non-BF16 fallback)")
        return non_bf16[0]

    all_ggufs.sort(key=lambda p: p.stat().st_size)
    print(
        f"WARNING: Only BF16 GGUFs found. Returning smallest: {all_ggufs[0].name}. "
        f"This is likely a full-precision model and will be slow in Ollama!"
    )
    return all_ggufs[0]


def main() -> None:
    os.chdir("/")

    lora_path      = resolve_path(LORA_PATH_RAW)
    final_gguf_dir = resolve_path(GGUF_DIR_RAW)
    tmp_gguf_dir   = resolve_path(TMP_GGUF_DIR_RAW)

    print(f"LLAMA_CPP_DIR={LLAMA_CPP_DIR}")
    print(f"LORA_PATH={lora_path}")
    print(f"FINAL_GGUF_DIR={final_gguf_dir}")
    print(f"TMP_GGUF_DIR={tmp_gguf_dir}")
    print(f"FINAL_GGUF_NAME={FINAL_GGUF_NAME}")
    print(f"QUANT_METHOD={QUANT_METHOD}")

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA model path not found: {lora_path}")

    quantize_binary = build_llama_cpp_if_needed()
    print(f"Using quantize binary: {quantize_binary}")

    clean_dir(tmp_gguf_dir)
    final_gguf_dir.mkdir(parents=True, exist_ok=True)

    # 4-bit loading is safe here: Unsloth's save_pretrained_gguf internally
    # dequantizes to BF16 before writing GGUF, so no quality loss occurs.
    print(f"Loading fine-tuned model from {lora_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Saving merged GGUF ({QUANT_METHOD}) to temp directory {tmp_gguf_dir}")
    model.save_pretrained_gguf(
        str(tmp_gguf_dir),
        tokenizer,
        quantization_method=QUANT_METHOD,
    )

    produced = list_dir(tmp_gguf_dir)
    print(f"Temp GGUF directory contents: {produced}")

    source_gguf = pick_quantized_gguf(tmp_gguf_dir)
    size_gb = source_gguf.stat().st_size / (1024 ** 3)
    print(f"Selected GGUF: {source_gguf}  ({size_gb:.2f} GB)")

    # Q4_K_M of a 3.8B model should be ~2 GB; BF16 would be ~7-8 GB
    if size_gb > 5.0:
        raise RuntimeError(
            f"Selected GGUF is {size_gb:.2f} GB — this looks like the BF16 "
            f"intermediate, not the quantized model. "
            f"Temp dir contents: {produced}. Aborting to avoid loading a "
            f"full-precision model into Ollama."
        )

    final_gguf_path = final_gguf_dir / FINAL_GGUF_NAME
    if final_gguf_path.exists():
        final_gguf_path.unlink()

    shutil.move(str(source_gguf), str(final_gguf_path))
    print(f"Moved GGUF from {source_gguf} to {final_gguf_path}")

    if tmp_gguf_dir.exists():
        for f in tmp_gguf_dir.glob("*.gguf"):
            if "BF16" in f.name.upper():
                print(f"Removing BF16 intermediate to free disk space: {f}")
                f.unlink()

    print(f"Final GGUF exists: {final_gguf_path.exists()}")
    print("Export complete.")


if __name__ == "__main__":
    main()