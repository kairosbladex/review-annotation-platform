#!/usr/bin/env python3
"""Train a QLoRA adapter for the Qwen3-VL candidate ROI verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from qwen3vl_qlora_smoke import (
    QwenVerifierDataset,
    SingleSampleCollator,
    _configure_processor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-pixels", type=int, default=28 * 28 * 64)
    parser.add_argument("--max-pixels", type=int, default=28 * 28 * 256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; formal training must execute on the GPU server")
    for path in (args.train, args.val):
        if not path.is_file():
            raise FileNotFoundError(path)
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

    train_dataset = QwenVerifierDataset(args.train, processor, args.max_length)
    val_dataset = QwenVerifierDataset(args.val, processor, args.max_length)
    if not train_dataset or not val_dataset:
        raise ValueError("train and validation datasets must both be non-empty")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
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
        train_dataset=train_dataset,
        data_collator=SingleSampleCollator(),
        processing_class=processor.tokenizer,
    )
    result = trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    summary = {
        "model": args.model,
        "task": "candidate_roi_binary_verifier",
        "quantization": "4bit_nf4_double_quant",
        "train_file": str(args.train),
        "val_file": str(args.val),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "epochs": args.epochs,
        "train_loss": result.training_loss,
        "eval_loss": eval_metrics.get("eval_loss"),
        "label_source": "human_gold",
        "deployable": False,
        "note": "Adapter requires independent generation evaluation before any service rollout.",
    }
    (args.output_dir / "formal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
