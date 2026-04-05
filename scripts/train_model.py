"""Fine-tune Phi-3-mini on Nemotron interactive_agent data using Unsloth + LoRA."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastLanguageModel

WORKSPACE_DIR = Path("/workspace")

MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct"

LORA_R = 16
LORA_ALPHA = 16

TRAIN_DATA_PATH_RAW = os.environ.get("TRAIN_DATA_PATH", "/workspace/data/training_data.jsonl")
OUTPUT_DIR_RAW = os.environ.get("OUTPUT_DIR", "/workspace/models/training_outputs")
LORA_SAVE_PATH_RAW = os.environ.get("LORA_SAVE_PATH", "/workspace/models/lora_agentic_model")

PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_STEPS = 625


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = WORKSPACE_DIR / path
    return path.resolve()


TRAIN_DATA_PATH = _resolve_path(TRAIN_DATA_PATH_RAW)
OUTPUT_DIR = _resolve_path(OUTPUT_DIR_RAW)
LORA_SAVE_PATH = _resolve_path(LORA_SAVE_PATH_RAW)


def load_model_and_tokenizer():
    """Load base model in 4-bit and attach a LoRA adapter."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer


def load_and_format_dataset(tokenizer, path: str):
    """Load JSONL training data and apply the chat template."""
    dataset = load_dataset("json", data_files={"train": path})["train"]

    def format_chat(batch):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(format_chat, batched=True)
    return dataset


def _list_dir(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(p.name for p in path.iterdir())


def _validate_lora_artifacts(path: Path) -> None:
    """Verify the LoRA adapter directory contains expected files."""
    existing = set(_list_dir(path))

    if "adapter_config.json" not in existing:
        raise RuntimeError(
            f"LoRA save appears incomplete. Missing adapter_config.json in {path}. "
            f"Found: {sorted(existing)}"
        )

    if not ({"adapter_model.safetensors", "adapter_model.bin"} & existing):
        raise RuntimeError(
            f"LoRA weights were not saved in {path}. "
            f"Found: {sorted(existing)}"
        )


def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"TRAIN_DATA_PATH={TRAIN_DATA_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"LORA_SAVE_PATH={LORA_SAVE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LORA_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_DATA_PATH}")

    model, tokenizer = load_model_and_tokenizer()

    print(f"Loading dataset from: {TRAIN_DATA_PATH}")
    dataset = load_and_format_dataset(tokenizer, str(TRAIN_DATA_PATH))
    print(f"Dataset size: {len(dataset)} examples")

    use_bf16 = torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        max_steps=MAX_STEPS,
        learning_rate=1e-4,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=5,
        save_steps=100,
        output_dir=str(OUTPUT_DIR),
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {LORA_SAVE_PATH}")
    model.save_pretrained(str(LORA_SAVE_PATH))
    tokenizer.save_pretrained(str(LORA_SAVE_PATH))
    print(f"LORA_SAVE_PATH contents after direct save: {_list_dir(LORA_SAVE_PATH)}")

    if "adapter_config.json" not in _list_dir(LORA_SAVE_PATH):
        print("Direct adapter save did not produce expected files; trying trainer.save_model on LoRA path")
        trainer.save_model(str(LORA_SAVE_PATH))
        tokenizer.save_pretrained(str(LORA_SAVE_PATH))
        print(f"LORA_SAVE_PATH contents after trainer.save_model fallback: {_list_dir(LORA_SAVE_PATH)}")

    _validate_lora_artifacts(LORA_SAVE_PATH)

    print("Training complete.")
    print(f"Verified LoRA adapter directory: {LORA_SAVE_PATH}")
    print(f"Final LORA_SAVE_PATH contents: {_list_dir(LORA_SAVE_PATH)}")


if __name__ == "__main__":
    main()