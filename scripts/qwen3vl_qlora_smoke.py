#!/usr/bin/env python3
"""Run a one-step QLoRA smoke train for a Qwen3-VL candidate verifier.

This intentionally trains an adapter on pseudo labels. It is a pipeline check,
not a quality evaluation and not a deployment command.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from qwen_training_provenance import (
    _configure_processor,
    validate_human_gold_dataset,
    validate_supervision_lengths,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _messages(row: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    images = [str((root / image).resolve()) for image in row["image"]]
    pool = iter(images)
    result: list[dict[str, Any]] = []
    for turn in row["conversations"]:
        if turn["from"] == "human":
            content: list[dict[str, Any]] = []
            for segment in re.split(r"(<image>)", str(turn["value"])):
                if segment == "<image>":
                    content.append({"type": "image", "image": next(pool)})
                elif segment.strip():
                    content.append({"type": "text", "text": segment.strip()})
            result.append({"role": "user", "content": content})
        else:
            result.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(turn["value"])}],
                }
            )
    try:
        next(pool)
    except StopIteration:
        return result
    raise ValueError(f"unused image in row {row.get('id')}")


class QwenVerifierDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: Path,
        processor: Any,
        max_length: int,
        *,
        require_human_gold: bool = False,
        provenance_root: Path | None = None,
        rows: Sequence[dict[str, Any]] | None = None,
    ):
        self.rows = list(rows) if rows is not None else _read_jsonl(path)
        self.root = path.parent
        self.processor = processor
        self.max_length = max_length
        self.provenance_summary = None
        if require_human_gold:
            self.provenance_summary = validate_human_gold_dataset(
                self.rows,
                dataset_path=path,
                provenance_root=provenance_root or Path.cwd(),
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        messages = _messages(row, self.root)
        if len(messages) < 2:
            raise ValueError(f"row {row.get('id')} has no user/assistant training pair")
        full = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = full["input_ids"]
        prompt_ids = prompt["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        supervised_tokens = validate_supervision_lengths(
            sequence_length=int(input_ids.shape[-1]),
            prompt_length=int(prompt_ids.shape[-1]),
            max_length=self.max_length,
            row_id=str(row.get("id", index)),
        )
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[-1]] = -100
        if int((labels != -100).sum().item()) != supervised_tokens:
            raise ValueError(f"row {row.get('id')} lost assistant supervision tokens")
        full["labels"] = labels
        return {key: value for key, value in full.items() if torch.is_tensor(value)}


@dataclass
class SingleSampleCollator:
    """Keep batches at one sample to avoid variable image patch padding."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("QLoRA smoke collator expects batch size 1")
        return features[0]




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-pixels", type=int, default=28 * 28 * 64)
    parser.add_argument("--max-pixels", type=int, default=28 * 28 * 256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this smoke run must execute on the GPU server")
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    _configure_processor(processor, args.min_pixels, args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    dataset = QwenVerifierDataset(args.train, processor, args.max_length)
    if not dataset:
        raise ValueError("training dataset is empty")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-5,
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.max_steps),
        save_total_limit=1,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SingleSampleCollator(),
        processing_class=processor.tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    summary = {
        "model": args.model,
        "task": "candidate_roi_binary_verifier",
        "quantization": "4bit_nf4_double_quant",
        "train_file": str(args.train),
        "train_rows": len(dataset),
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_loss": result.training_loss,
        "pseudo_labels_only": True,
        "deployable": False,
    }
    (args.output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
