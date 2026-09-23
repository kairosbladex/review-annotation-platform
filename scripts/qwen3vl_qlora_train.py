#!/usr/bin/env python3
"""Train a QLoRA adapter for the Qwen3-VL candidate ROI verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
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
    _read_jsonl,
)
from qwen_training_provenance import (
    deduplicate_human_gold_rows,
    validate_disjoint_image_splits,
    validate_human_gold_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_provenance_summary(summary: dict) -> dict:
    result = {
        key: value
        for key, value in summary.items()
        if key not in {"image_sha256s", "source_group_ids"}
    }
    result["unique_source_event_task_groups"] = len(summary["source_group_ids"])
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-free-gb", type=float, default=24.0)
    parser.add_argument(
        "--provenance-root",
        type=Path,
        default=Path.cwd(),
        help="Root containing the declared human annotation exports and source images.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; formal training must execute on the GPU server")
    if not 0.0 < args.epochs <= 3.0:
        raise ValueError("formal training epochs must be greater than 0 and at most 3")
    if args.min_free_gb < 0:
        raise ValueError("--min-free-gb must be non-negative")
    for path in (args.train, args.val):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Audit labels, source exports, image hashes, and split isolation before loading weights.
    raw_train_rows = _read_jsonl(args.train)
    raw_val_rows = _read_jsonl(args.val)
    raw_train_provenance = validate_human_gold_dataset(
        raw_train_rows,
        dataset_path=args.train,
        provenance_root=args.provenance_root,
        allow_duplicates=True,
        allow_conflicts=True,
    )
    raw_val_provenance = validate_human_gold_dataset(
        raw_val_rows,
        dataset_path=args.val,
        provenance_root=args.provenance_root,
        allow_duplicates=True,
        allow_conflicts=True,
    )
    validate_disjoint_image_splits(
        raw_train_provenance,
        raw_val_provenance,
    )

    train_rows, train_duplicate_audit, train_conflicts = deduplicate_human_gold_rows(
        raw_train_rows
    )
    val_rows, val_duplicate_audit, val_conflicts = deduplicate_human_gold_rows(
        raw_val_rows
    )
    if not train_rows or not val_rows:
        raise ValueError("train and validation need rows after duplicate/conflict exclusion")
    train_dataset = QwenVerifierDataset(
        args.train,
        None,
        args.max_length,
        require_human_gold=True,
        provenance_root=args.provenance_root,
        rows=train_rows,
    )
    val_dataset = QwenVerifierDataset(
        args.val,
        None,
        args.max_length,
        require_human_gold=True,
        provenance_root=args.provenance_root,
        rows=val_rows,
    )
    validate_disjoint_image_splits(
        train_dataset.provenance_summary,
        val_dataset.provenance_summary,
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"GPU 0 has {free_gb:.1f} GiB free of {total_gb:.1f} GiB; "
            f"requires at least {args.min_free_gb:.1f} GiB. "
            "No process was stopped; release GPU resources or choose a suitable host."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "train_duplicate_audit.jsonl", train_duplicate_audit)
    _write_jsonl(args.output_dir / "val_duplicate_audit.jsonl", val_duplicate_audit)
    _write_jsonl(args.output_dir / "train_conflicts_excluded.jsonl", train_conflicts)
    _write_jsonl(args.output_dir / "val_conflicts_excluded.jsonl", val_conflicts)

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
    train_dataset.processor = processor
    val_dataset.processor = processor

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

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        seed=args.seed,
        data_seed=args.seed,
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
        "model_revision": getattr(model.config, "_commit_hash", None),
        "processor_name_or_path": getattr(processor, "name_or_path", None),
        "task": "candidate_roi_binary_verifier",
        "quantization": "4bit_nf4_double_quant",
        "train_file": str(args.train),
        "val_file": str(args.val),
        "train_file_sha256": _file_sha256(args.train),
        "val_file_sha256": _file_sha256(args.val),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "raw_train_rows": len(raw_train_rows),
        "raw_val_rows": len(raw_val_rows),
        "train_conflicts_excluded": len(train_conflicts),
        "val_conflicts_excluded": len(val_conflicts),
        "train_duplicate_audit_rows": len(train_duplicate_audit),
        "val_duplicate_audit_rows": len(val_duplicate_audit),
        "raw_train_provenance": _json_provenance_summary(raw_train_provenance),
        "raw_val_provenance": _json_provenance_summary(raw_val_provenance),
        "epochs": args.epochs,
        "seed": args.seed,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gpu": {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gib": round(total_gb, 2),
            "free_memory_before_model_load_gib": round(free_gb, 2),
            "required_free_memory_gib": args.min_free_gb,
        },
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "train_loss": result.training_loss,
        "eval_loss": eval_metrics.get("eval_loss"),
        "train_provenance": _json_provenance_summary(
            train_dataset.provenance_summary
        ),
        "val_provenance": _json_provenance_summary(val_dataset.provenance_summary),
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
