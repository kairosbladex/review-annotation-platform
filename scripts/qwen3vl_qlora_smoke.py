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

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - local CPU test env without torch
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: Any) -> Any:
            return cls

try:
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
except ModuleNotFoundError:  # pragma: no cover - local CPU test env without transformers
    AutoProcessor = None  # type: ignore[assignment]
    BitsAndBytesConfig = None  # type: ignore[assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
    Trainer = object  # type: ignore[assignment,misc]
    TrainingArguments = None  # type: ignore[assignment]

from qwen_training_provenance import (
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


DEFAULT_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
VISION_PROJ_MODULES: list[str] = [
    "merger.mlp.0",
    "merger.mlp.2",
]


def resolve_lora_target_modules(include_vision_proj: bool = False) -> list[str]:
    modules = list(DEFAULT_LORA_TARGET_MODULES)
    if include_vision_proj:
        for mod in VISION_PROJ_MODULES:
            if mod not in modules:
                modules.append(mod)
    return modules


def _detect_gpu_capabilities(device_index: int = 0) -> dict[str, Any]:
    """Detect GPU SM capability and memory to select optimal precision, attention, and DataLoader parameters."""
    import importlib.util

    if torch is None or not torch.cuda.is_available():
        return {
            "device_name": "cpu",
            "sm_major": 0,
            "sm_minor": 0,
            "total_gb": 0.0,
            "free_gb": 0.0,
            "compute_dtype": getattr(torch, "float32", "float32"),
            "bf16": False,
            "fp16": False,
            "attn_implementation": "sdpa",
            "is_l20_class": False,
            "recommended_batch_size": 1,
            "recommended_grad_accum": 1,
            "recommended_num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }

    major, minor = torch.cuda.get_device_capability(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    has_fa2 = importlib.util.find_spec("flash_attn") is not None

    if major >= 8:
        compute_dtype = torch.bfloat16
        bf16 = True
        fp16 = False
        attn_impl = "flash_attention_2" if has_fa2 else "sdpa"
    else:
        compute_dtype = torch.float16
        bf16 = False
        fp16 = True
        attn_impl = "sdpa"

    is_l20_class = major >= 8 and total_gb >= 40.0
    if is_l20_class:
        batch_size = 4
        grad_accum = 4
        num_workers = 8
        pin_memory = True
        persistent_workers = True
    else:
        batch_size = 1
        grad_accum = 16
        num_workers = 2
        pin_memory = True
        persistent_workers = False

    return {
        "device_name": device_name,
        "sm_major": major,
        "sm_minor": minor,
        "total_gb": total_gb,
        "free_gb": free_gb,
        "compute_dtype": compute_dtype,
        "bf16": bf16,
        "fp16": fp16,
        "attn_implementation": attn_impl,
        "is_l20_class": is_l20_class,
        "recommended_batch_size": batch_size,
        "recommended_grad_accum": grad_accum,
        "recommended_num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }


@dataclass
class SingleSampleCollator:
    """Keep batches at one sample to avoid variable image patch padding."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("QLoRA smoke collator expects batch size 1")
        return features[0]


@dataclass
class MultiSamplePadCollator:
    """Collate single or multi-sample Qwen3-VL batches with right-padded sequences and concatenated vision patches."""

    pad_token_id: int = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("MultiSamplePadCollator received empty features")
        if len(features) == 1:
            return features[0]

        if torch is None:
            import numpy as np

            max_seq_len = max(int(np.asarray(f["input_ids"]).shape[-1]) for f in features)
            b_ids, b_mask, b_lbl, pv_list, grid_list = [], [], [], [], []
            for f in features:
                ids = np.asarray(f["input_ids"]).reshape(-1)
                mask = (
                    np.asarray(f["attention_mask"]).reshape(-1)
                    if "attention_mask" in f
                    else np.ones_like(ids)
                )
                lbl = np.asarray(f["labels"]).reshape(-1)
                pad_len = max_seq_len - int(ids.shape[-1])
                if pad_len > 0:
                    ids = np.pad(ids, (0, pad_len), constant_values=self.pad_token_id)
                    mask = np.pad(mask, (0, pad_len), constant_values=0)
                    lbl = np.pad(lbl, (0, pad_len), constant_values=-100)
                b_ids.append(ids)
                b_mask.append(mask)
                b_lbl.append(lbl)
                if "pixel_values" in f:
                    pv_list.append(np.asarray(f["pixel_values"]))
                if "image_grid_thw" in f:
                    grid_list.append(np.asarray(f["image_grid_thw"]))
            out: dict[str, Any] = {
                "input_ids": np.stack(b_ids, axis=0),
                "attention_mask": np.stack(b_mask, axis=0),
                "labels": np.stack(b_lbl, axis=0),
            }
            if pv_list:
                out["pixel_values"] = np.concatenate(pv_list, axis=0)
            if grid_list:
                out["image_grid_thw"] = np.concatenate(grid_list, axis=0)
            return out

        max_seq_len = max(int(f["input_ids"].shape[-1]) for f in features)
        batch_input_ids: list[Any] = []
        batch_attention_mask: list[Any] = []
        batch_labels: list[Any] = []
        pixel_values_list: list[Any] = []
        image_grid_thw_list: list[Any] = []

        for f in features:
            ids = f["input_ids"].squeeze(0) if f["input_ids"].ndim == 2 else f["input_ids"]
            mask = (
                f["attention_mask"].squeeze(0)
                if "attention_mask" in f and f["attention_mask"].ndim == 2
                else torch.ones_like(ids)
            )
            lbl = f["labels"].squeeze(0) if f["labels"].ndim == 2 else f["labels"]
            pad_len = max_seq_len - int(ids.shape[-1])
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=-100)
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            batch_labels.append(lbl)
            if "pixel_values" in f and torch.is_tensor(f["pixel_values"]):
                pixel_values_list.append(f["pixel_values"])
            if "image_grid_thw" in f and torch.is_tensor(f["image_grid_thw"]):
                image_grid_thw_list.append(f["image_grid_thw"])

        batch: dict[str, Any] = {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
        }
        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
        return batch


def _configure_processor(processor: Any, min_pixels: int, max_pixels: int) -> None:
    """Apply and verify the pixel budget across dict and SizeDict processors."""
    image_processor = processor.image_processor
    size = getattr(image_processor, "size", None)
    configured_min = False
    configured_max = False

    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min_pixels
        configured_min = True
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
        configured_max = True

    if size is not None:
        if hasattr(size, "shortest_edge"):
            size.shortest_edge = min_pixels
            configured_min = True
        elif hasattr(size, "__setitem__"):
            size["shortest_edge"] = min_pixels
            configured_min = True
        if hasattr(size, "longest_edge"):
            size.longest_edge = max_pixels
            configured_max = True
        elif hasattr(size, "__setitem__"):
            size["longest_edge"] = max_pixels
            configured_max = True

    observed_min = getattr(image_processor, "min_pixels", None)
    observed_max = getattr(image_processor, "max_pixels", None)
    if observed_min is None and size is not None:
        observed_min = getattr(size, "shortest_edge", None)
        if observed_min is None and hasattr(size, "get"):
            observed_min = size.get("shortest_edge")
    if observed_max is None and size is not None:
        observed_max = getattr(size, "longest_edge", None)
        if observed_max is None and hasattr(size, "get"):
            observed_max = size.get("longest_edge")

    if not configured_min or not configured_max:
        raise RuntimeError("image processor exposes no supported pixel-budget fields")
    if observed_min != min_pixels or observed_max != max_pixels:
        raise RuntimeError(
            "image processor pixel budget differs after configuration: "
            f"min={observed_min!r}, max={observed_max!r}"
        )


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
    parser.add_argument(
        "--include-vision-proj",
        action="store_true",
        help="Include vision merger MLP projection layers in LoRA target modules.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this smoke run must execute on the GPU server")
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu_caps = _detect_gpu_capabilities(0)
    compute_dtype = gpu_caps["compute_dtype"]
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
        attn_implementation=gpu_caps["attn_implementation"],
    )
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    target_modules = resolve_lora_target_modules(args.include_vision_proj)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
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
        bf16=bool(gpu_caps["bf16"]),
        fp16=bool(gpu_caps["fp16"]),
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
