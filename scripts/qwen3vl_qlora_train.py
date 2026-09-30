#!/usr/bin/env python3
"""Train a QLoRA adapter for the Qwen3-VL candidate ROI verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import WeightedRandomSampler

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



_QUANTIZATION_CONTRACT = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}
_STRICT_OUTPUT_CONTRACT = {
    "json_object_keys": ["decision"],
    "allowed_decisions": ["CONFIRMED", "REJECTED"],
    "invalid_output": "uncertain",
}

def _training_stratum(row: dict[str, Any]) -> str:
    meta = row.get("meta")
    if not isinstance(meta, dict) or not meta.get("class_name"):
        raise ValueError("balanced sampling requires class_name metadata")
    label = meta.get("human_label")
    if label not in {"true", "false"}:
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("balanced sampling requires a binary human target")
        try:
            target = json.loads(str(conversations[-1]["value"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("balanced sampling requires a JSON assistant target") from exc
        label = {"CONFIRMED": "true", "REJECTED": "false"}.get(target.get("decision"))
    if label not in {"true", "false"}:
        raise ValueError("balanced sampling requires a binary human target")
    return str(meta["class_name"]) + "|" + str(label)


def _balanced_sampling_manifest(rows: list[dict[str, Any]], seed: int) -> tuple[list[float], dict[str, Any]]:
    strata = [_training_stratum(row) for row in rows]
    counts = Counter(strata)
    weights = [len(rows) / (len(counts) * counts[stratum]) for stratum in strata]
    return weights, {
        "strategy": "deterministic_weighted_replacement_by_class_and_human_label",
        "seed": seed,
        "replacement": True,
        "samples_per_epoch": len(rows),
        "validation_sampling": "natural_original_partition",
        "stratum_counts": dict(sorted(counts.items())),
        "per_row_weight_by_stratum": {
            stratum: len(rows) / (len(counts) * count)
            for stratum, count in sorted(counts.items())
        },
    }


def _capped_sampling_manifest(
    rows: list[dict[str, Any]], seed: int, cap: float
) -> tuple[list[float], dict[str, Any]]:
    """等权抽样的折中：每个“类别|标签”分组的放大倍数不超过 cap，再整体归一化到平均权重 1。

    cap 越小越接近自然比例，越大越接近完全等权（balanced_stratum）。
    """
    if not cap > 0:
        raise ValueError("stratum weight cap must be positive")
    strata = [_training_stratum(row) for row in rows]
    counts = Counter(strata)
    raw = {
        stratum: min(len(rows) / (len(counts) * count), cap)
        for stratum, count in counts.items()
    }
    scale = len(rows) / sum(raw[stratum] for stratum in strata)
    per_stratum = {stratum: raw[stratum] * scale for stratum in raw}
    weights = [per_stratum[stratum] for stratum in strata]
    return weights, {
        "strategy": "deterministic_weighted_replacement_by_class_and_human_label_capped",
        "seed": seed,
        "replacement": True,
        "stratum_weight_cap": cap,
        "samples_per_epoch": len(rows),
        "validation_sampling": "natural_original_partition",
        "stratum_counts": dict(sorted(counts.items())),
        "per_row_weight_by_stratum": dict(sorted(per_stratum.items())),
    }


class DeterministicBalancedTrainer(Trainer):
    """Use an audited seeded sampler without changing the held-out distribution."""

    def __init__(
        self,
        *args: Any,
        train_sample_weights: list[float],
        sampler_seed: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._train_sample_weights = train_sample_weights
        self._sampler_seed = sampler_seed

    def _get_train_sampler(self, train_dataset: Any = None) -> WeightedRandomSampler:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or len(dataset) != len(self._train_sample_weights):
            raise ValueError("balanced sampler weights do not match the training dataset")
        generator = torch.Generator()
        generator.manual_seed(self._sampler_seed)
        return WeightedRandomSampler(
            weights=torch.as_tensor(self._train_sample_weights, dtype=torch.double),
            num_samples=len(self._train_sample_weights),
            replacement=True,
            generator=generator,
        )


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "release_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return artifacts


def _checkpoint_manifest(root: Path) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for path in sorted(root.glob("checkpoint-*"), key=lambda item: int(item.name.split("-")[-1])):
        trainer_state = path / "trainer_state.json"
        state = json.loads(trainer_state.read_text(encoding="utf-8")) if trainer_state.is_file() else {}
        checkpoints.append(
            {
                "path": path.name,
                "global_step": state.get("global_step"),
                "epoch": state.get("epoch"),
                "artifacts": _artifact_manifest(path),
            }
        )
    return checkpoints


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_provenance_summary(summary: dict) -> dict:
    set_fields = {
        "image_sha256s",
        "view_sha256s",
        "annotated_frame_sha256s",
        "clean_frame_sha256s",
        "source_group_ids",
        "source_event_ids",
        "source_task_ids",
    }
    result = {key: value for key, value in summary.items() if key not in set_fields}
    result.update(
        unique_source_images=len(summary.get("image_sha256s", ())),
        unique_training_views=len(summary.get("view_sha256s", ())),
        unique_annotated_frames=len(summary.get("annotated_frame_sha256s", ())),
        unique_clean_frames=len(summary.get("clean_frame_sha256s", ())),
        unique_source_event_task_groups=len(summary.get("source_group_ids", ())),
        unique_source_events=len(summary.get("source_event_ids", ())),
        unique_source_tasks=len(summary.get("source_task_ids", ())),
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(value: Any) -> str:
    basis = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(basis).hexdigest()


def _base_model_identity(model_reference: str) -> dict[str, Any]:
    path = Path(model_reference).expanduser()
    if not path.is_dir():
        return {
            "reference": model_reference,
            "resolved_path": None,
            "tree_sha256": None,
            "artifacts": [],
            "limitation": "base model reference is not a local directory; file-tree SHA unavailable",
        }
    root = path.resolve()
    artifacts: list[dict[str, Any]] = []
    for artifact in sorted(root.rglob("*")):
        if not artifact.is_file():
            continue
        artifacts.append(
            {
                "path": str(artifact.relative_to(root)),
                "bytes": artifact.stat().st_size,
                "sha256": _file_sha256(artifact),
            }
        )
    return {
        "reference": model_reference,
        "resolved_path": str(root),
        "tree_hash_basis": "sha256(canonical UTF-8 JSON of sorted path,bytes,sha256 artifacts)",
        "tree_sha256": _manifest_sha256(artifacts),
        "file_count": len(artifacts),
        "total_bytes": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }


def _training_code_identity() -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    names = (
        "qwen3vl_qlora_train.py",
        "qwen3vl_qlora_smoke.py",
        "qwen_training_provenance.py",
        "prepare_ktd_clean_source_qwen_dataset.py",
        "qwen3vl_verify.py",
    )
    files = []
    for name in names:
        path = script_dir / name
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return {
        "git_commit": None,
        "tree_hash_basis": "sha256(canonical UTF-8 JSON of sorted path,bytes,sha256 files)",
        "tree_sha256": _manifest_sha256(files),
        "files": files,
        "limitation": "git commit is intentionally null until these exact code files are committed",
    }


def _expected_epoch_checkpoints(epochs: float) -> int:
    return math.ceil(epochs)


def _resolve_resume_checkpoint(
    output_dir: Path, resume_from_checkpoint: Path | None
) -> tuple[Path | None, dict[str, Any]]:
    if resume_from_checkpoint is None:
        return None, {
            "enabled": False,
            "checkpoint_path": None,
            "starting_global_step": None,
            "starting_epoch": None,
        }
    resolved_output = output_dir.expanduser().resolve()
    checkpoint = resume_from_checkpoint.expanduser().resolve()
    if checkpoint.parent != resolved_output or re.fullmatch(
        r"checkpoint-(\d+)", checkpoint.name
    ) is None:
        raise ValueError(
            "--resume-from-checkpoint must be an output-dir direct child named checkpoint-N"
        )
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise ValueError(
            "resume checkpoint is missing trainer_state.json: "
            + str(trainer_state_path)
        )
    try:
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "resume checkpoint trainer_state.json is invalid JSON"
        ) from exc
    if not isinstance(trainer_state, dict):
        raise ValueError("resume checkpoint trainer_state.json must be an object")
    global_step = trainer_state.get("global_step")
    epoch = trainer_state.get("epoch")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("resume checkpoint has an invalid global_step")
    checkpoint_step = int(checkpoint.name.removeprefix("checkpoint-"))
    if checkpoint_step != global_step:
        raise ValueError(
            "resume checkpoint directory step differs from trainer_state global_step"
        )
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or epoch < 0:
        raise ValueError("resume checkpoint has an invalid epoch")
    return checkpoint, {
        "enabled": True,
        "checkpoint_path": str(checkpoint),
        "trainer_state_path": str(trainer_state_path),
        "trainer_state_sha256": _file_sha256(trainer_state_path),
        "starting_global_step": global_step,
        "starting_epoch": epoch,
    }


def _load_dataset_contract(
    train_path: Path,
    val_path: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_schema = "annotated_locator_plus_clean_original_v1"
    schemas = {
        str(row.get("meta", {}).get("view_schema") or "")
        for row in train_rows + val_rows
    }
    if clean_schema not in schemas:
        schemas.discard("")
        return {
            "view_schema": sorted(schemas) or ["legacy_context_plus_roi"],
            "strict_output": dict(_STRICT_OUTPUT_CONTRACT),
            "processor": None,
            "quantization": dict(_QUANTIZATION_CONTRACT),
            "contract_source": "derived_from_legacy_dataset",
        }
    if schemas != {clean_schema}:
        raise ValueError(
            "clean-source train and validation must contain only the four-view schema"
        )
    if train_path.resolve().parent != val_path.resolve().parent:
        raise ValueError("clean-source train and validation files must share a dataset manifest")
    manifest_path = train_path.resolve().parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("clean-source dataset is missing dataset_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("deployment_contract")
    if not isinstance(contract, dict):
        raise ValueError("clean-source dataset manifest lacks deployment_contract")
    template = contract.get("prompt_template")
    template_sha = contract.get("prompt_template_sha256")
    if not isinstance(template, str) or hashlib.sha256(template.encode("utf-8")).hexdigest() != template_sha:
        raise ValueError("dataset prompt template SHA-256 is invalid")
    if contract.get("view_schema") != clean_schema:
        raise ValueError("dataset view schema differs from clean-source training schema")
    if contract.get("strict_output") != _STRICT_OUTPUT_CONTRACT:
        raise ValueError("dataset strict-output contract is unsupported")
    if contract.get("quantization") != _QUANTIZATION_CONTRACT:
        raise ValueError("dataset quantization contract is unsupported")
    for split, path, rows in (
        ("train", train_path, train_rows),
        ("validation", val_path, val_rows),
    ):
        declared = manifest.get("outputs", {}).get(split, {})
        if declared.get("sha256") != _file_sha256(path) or declared.get("rows") != len(rows):
            raise ValueError(f"{split} file differs from dataset manifest")
        for row in rows:
            meta = row.get("meta", {})
            if meta.get("view_schema") != clean_schema:
                raise ValueError("row view schema differs from the clean-source contract")
            if meta.get("prompt_template_sha256") != template_sha:
                raise ValueError("row prompt template differs from dataset manifest")
            conversations = row.get("conversations")
            prompt = None
            if isinstance(conversations, list):
                for turn in conversations:
                    if isinstance(turn, dict) and turn.get("from") in {"human", "user"}:
                        prompt = turn.get("value")
                        break
            expected_prompt = template.replace(
                "{class_name}", str(meta.get("class_name") or "")
            )
            if prompt != expected_prompt:
                raise ValueError(
                    "row human prompt differs from the rendered deployment template"
                )
    return {**contract, "contract_source": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume from an output-dir direct child named checkpoint-N.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-pixels", type=int, default=28 * 28 * 64)
    parser.add_argument("--max-pixels", type=int, default=28 * 28 * 256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-sampling",
        choices=("balanced_stratum", "capped_stratum", "natural"),
        default="balanced_stratum",
        help="Balance class_name x human_label strata for train only, cap the balancing boost, or keep natural order.",
    )
    parser.add_argument(
        "--stratum-weight-cap",
        type=float,
        default=2.0,
        help="capped_stratum only: maximum per-stratum weight before renormalization.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--min-free-gb", type=float, default=24.0)
    parser.add_argument(
        "--provenance-root",
        type=Path,
        default=Path.cwd(),
        help="Root containing the declared human annotation exports and source images.",
    )
    args = parser.parse_args()
    resume_checkpoint, resume_manifest = _resolve_resume_checkpoint(
        args.output_dir, args.resume_from_checkpoint
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; formal training must execute on the GPU server")
    if not 0.0 < args.epochs <= 3.0:
        raise ValueError("formal training epochs must be greater than 0 and at most 3")
    if not 1e-6 <= args.learning_rate <= 1e-4:
        raise ValueError("--learning-rate must be within [1e-6, 1e-4]")
    if args.min_free_gb < 0:
        raise ValueError("--min-free-gb must be non-negative")
    for path in (args.train, args.val):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Audit labels, source exports, image hashes, and split isolation before loading weights.
    raw_train_rows = _read_jsonl(args.train)
    raw_val_rows = _read_jsonl(args.val)
    dataset_contract = _load_dataset_contract(
        args.train, args.val, raw_train_rows, raw_val_rows
    )
    declared_processor = dataset_contract.get("processor")
    if isinstance(declared_processor, dict) and (
        declared_processor.get("min_pixels") != args.min_pixels
        or declared_processor.get("max_pixels") != args.max_pixels
        or declared_processor.get("max_sequence_length") != args.max_length
    ):
        raise ValueError(
            "training processor or sequence length differs from the dataset deployment contract"
        )
    runtime_contract = {
        **dataset_contract,
        "processor": {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_sequence_length": args.max_length,
        },
        "quantization": dict(_QUANTIZATION_CONTRACT),
        "strict_output": dict(_STRICT_OUTPUT_CONTRACT),
    }
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
    if (
        resume_checkpoint is None
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "train_duplicate_audit.jsonl", train_duplicate_audit)
    _write_jsonl(args.output_dir / "val_duplicate_audit.jsonl", val_duplicate_audit)
    _write_jsonl(args.output_dir / "train_conflicts_excluded.jsonl", train_conflicts)
    _write_jsonl(args.output_dir / "val_conflicts_excluded.jsonl", val_conflicts)
    if args.train_sampling == "balanced_stratum":
        train_sample_weights, sampling_manifest = _balanced_sampling_manifest(
            train_rows, args.seed
        )
    elif args.train_sampling == "capped_stratum":
        train_sample_weights, sampling_manifest = _capped_sampling_manifest(
            train_rows, args.seed, args.stratum_weight_cap
        )
    else:
        train_sample_weights = []
        sampling_manifest = {
            "strategy": "natural_original_partition",
            "seed": args.seed,
            "replacement": False,
            "samples_per_epoch": len(train_rows),
            "validation_sampling": "natural_original_partition",
            "stratum_counts": dict(
                sorted(Counter(_training_stratum(row) for row in train_rows).items())
            ),
        }
    (args.output_dir / "train_sampling_manifest.json").write_text(
        json.dumps(sampling_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
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
    weighted = args.train_sampling in {"balanced_stratum", "capped_stratum"}
    trainer_class = DeterministicBalancedTrainer if weighted else Trainer
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": SingleSampleCollator(),
        "processing_class": processor.tokenizer,
    }
    if weighted:
        trainer_kwargs.update(
            train_sample_weights=train_sample_weights,
            sampler_seed=args.seed,
        )
    trainer = trainer_class(**trainer_kwargs)
    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    summary = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "processor_name_or_path": getattr(processor, "name_or_path", None),
        "task": "candidate_roi_binary_verifier",
        "deployment_contract": runtime_contract,
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
        "train_sampling": sampling_manifest,
        "learning_rate": args.learning_rate,
        "checkpoint_policy": {
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "maximum_epochs": 3,
            "purpose": "retain every epoch checkpoint for comparison and rollback",
        },
        "resume": resume_manifest,
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
    formal_summary_path = args.output_dir / "formal_summary.json"
    formal_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checkpoints = _checkpoint_manifest(args.output_dir)
    expected_checkpoint_count = _expected_epoch_checkpoints(args.epochs)
    if len(checkpoints) != expected_checkpoint_count:
        raise RuntimeError(
            "epoch checkpoint count mismatch: "
            f"expected {expected_checkpoint_count}, found {len(checkpoints)}"
        )
    dataset_manifest_path = args.train.resolve().parent / "dataset_manifest.json"
    base_model_identity = _base_model_identity(args.model)
    code_identity = _training_code_identity()
    release_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_status": "candidate_requires_generation_evaluation",
        "task": "candidate_roi_binary_verifier",
        "deployment_contract": runtime_contract,
        "base_model": {
            **base_model_identity,
            "revision": getattr(model.config, "_commit_hash", None),
        },
        "training_code": code_identity,
        "adapter_directory": str(args.output_dir.resolve()),
        "datasets": {
            "manifest": {
                "path": str(dataset_manifest_path),
                "sha256": _file_sha256(dataset_manifest_path),
            },
            "train": {
                "path": str(args.train.resolve()),
                "sha256": _file_sha256(args.train),
                "rows": len(train_dataset),
                "provenance": _json_provenance_summary(train_dataset.provenance_summary),
            },
            "validation": {
                "path": str(args.val.resolve()),
                "sha256": _file_sha256(args.val),
                "rows": len(val_dataset),
                "provenance": _json_provenance_summary(val_dataset.provenance_summary),
                "sampling": "natural_original_partition",
            },
        },
        "training": {
            "epochs": args.epochs,
            "seed": args.seed,
            "quantization": "4bit_nf4_double_quant",
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
            },
            "sampling": sampling_manifest,
            "checkpoint_policy": {
                **summary["checkpoint_policy"],
                "expected_checkpoint_count": expected_checkpoint_count,
                "actual_checkpoint_count": len(checkpoints),
            },
            "resume": resume_manifest,
            "train_loss": result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss"),
        },
        "checkpoints": checkpoints,
        "artifacts": _artifact_manifest(args.output_dir),
        "deployment_gate": {
            "deployable": False,
            "required_next_step": "strict generation evaluation on the untouched validation partition",
        },
        "limitations": [
            "normalized gold_labels.jsonl does not retain labeled_by or labeled_at; this release pins the normalized gold and candidate-pair hashes but does not independently re-attest reviewer identity or timestamp",
            "git commit remains null until these exact training code files are committed",
        ],
        "release_manifest_hash_basis": {
            "algorithm": "sha256",
            "encoding": "UTF-8",
            "canonicalization": "JSON with ensure_ascii=false, sort_keys=true, separators=(',', ':')",
            "excluded_fields": ["release_manifest_content_sha256"],
        },
    }
    release_basis = json.dumps(
        release_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    release_manifest["release_manifest_content_sha256"] = hashlib.sha256(
        release_basis
    ).hexdigest()
    (args.output_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
