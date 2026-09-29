#!/usr/bin/env python3
"""Generate strict-JSON candidate decisions and audited metrics on a held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


_DECISION_MAP = {
    "CONFIRMED": "true",
    "REJECTED": "false",
}


def parse_strict_decision(text: str, *, allow_uncertain: bool = True) -> tuple[str, str | None]:
    """Parse exactly one JSON verdict; fail closed to uncertain on any violation."""
    duplicate_key = False

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate_key
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate_key = True
            result[key] = value
        return result

    try:
        payload = json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, json.JSONDecodeError):
        return "uncertain", "invalid_json"
    if duplicate_key:
        return "uncertain", "duplicate_json_key"
    if not isinstance(payload, dict):
        return "uncertain", "response_is_not_an_object"
    if set(payload) != {"decision"}:
        return "uncertain", "unexpected_json_fields"
    decision = payload.get("decision")
    if decision not in _DECISION_MAP:
        return "uncertain", "unsupported_decision"
    return _DECISION_MAP[str(decision)], None


def _metric_block(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    total = len(values)
    tp = sum(row["gold"] == "true" and row["prediction"] == "true" for row in values)
    tn = sum(row["gold"] == "false" and row["prediction"] == "false" for row in values)
    fp = sum(row["gold"] == "false" and row["prediction"] == "true" for row in values)
    fn = sum(row["gold"] == "true" and row["prediction"] == "false" for row in values)
    true_uncertain = sum(
        row["gold"] == "true" and row["prediction"] == "uncertain" for row in values
    )
    false_uncertain = sum(
        row["gold"] == "false" and row["prediction"] == "uncertain" for row in values
    )
    valid = tp + tn + fp + fn
    actual_positive = tp + fn + true_uncertain
    actual_negative = tn + fp + false_uncertain
    false_accept_rate = fp / actual_negative if actual_negative else None
    false_reject_rate = fn / actual_positive if actual_positive else None
    return {
        "n": total,
        "correct": tp + tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "coverage_n": valid,
        "coverage": valid / total if total else 0.0,
        "actual_positive_n": actual_positive,
        "actual_negative_n": actual_negative,
        "selective_accuracy": (tp + tn) / valid if valid else 0.0,
        "far": false_accept_rate,
        "frr": false_reject_rate,
        "false_accept_rate_actual_negative": false_accept_rate,
        "false_reject_rate_actual_positive": false_reject_rate,
        "manual_review_rate": (true_uncertain + false_uncertain) / total if total else 0.0,
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "true_uncertain": true_uncertain,
            "false_uncertain": false_uncertain,
        },
    }


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_class_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_class_rows[str(row.get("class_name") or "__missing__")].append(row)
    return {
        "overall": _metric_block(predictions),
        "by_class": {
            class_name: _metric_block(rows)
            for class_name, rows in sorted(by_class_rows.items())
        },
        "parse_failures": sum(bool(row.get("parse_error")) for row in predictions),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_output_path(root: Path, manifest: Mapping[str, Any], split: str) -> Path:
    declared = manifest.get("outputs", {}).get(split, {})
    value = declared.get("path") if isinstance(declared, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"dataset manifest has no {split} output path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_audited_validation(
    data_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load the declared validation split and re-run dataset/provenance checks."""
    from qwen3vl_qlora_smoke import _read_jsonl
    from qwen3vl_qlora_train import _json_provenance_summary, _load_dataset_contract
    from qwen_training_provenance import (
        validate_disjoint_image_splits,
        validate_human_gold_dataset,
    )

    data_path = data_path.expanduser().resolve()
    manifest_path = data_path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("validation dataset is missing dataset_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = _declared_output_path(data_path.parent, manifest, "train")
    validation_path = _declared_output_path(data_path.parent, manifest, "validation")
    if data_path != validation_path:
        raise ValueError("generation evaluation must use the manifest-declared validation split")
    if not train_path.is_file() or not validation_path.is_file():
        raise ValueError("manifest-declared train or validation file is unavailable")

    train_rows = _read_jsonl(train_path)
    validation_rows = _read_jsonl(validation_path)
    contract = _load_dataset_contract(
        train_path, validation_path, train_rows, validation_rows
    )
    train_provenance = validate_human_gold_dataset(
        train_rows,
        dataset_path=train_path,
        provenance_root=data_path.parent,
    )
    validation_provenance = validate_human_gold_dataset(
        validation_rows,
        dataset_path=validation_path,
        provenance_root=data_path.parent,
    )
    validate_disjoint_image_splits(train_provenance, validation_provenance)
    audit = {
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _file_sha256(manifest_path),
        },
        "train": {
            "path": str(train_path),
            "sha256": _file_sha256(train_path),
            "rows": len(train_rows),
            "provenance": _json_provenance_summary(train_provenance),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": _file_sha256(validation_path),
            "rows": len(validation_rows),
            "provenance": _json_provenance_summary(validation_provenance),
        },
    }
    return validation_rows, contract, audit


def _load_model(
    model_name: str,
    adapter: Path | None,
    *,
    processor_name: str | None = None,
    min_pixels: int = 28 * 28 * 64,
    max_pixels: int = 28 * 28 * 256,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    from qwen3vl_qlora_smoke import _configure_processor

    dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        processor_name or model_name,
        local_files_only=True,
    )
    _configure_processor(processor, min_pixels, max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    model.eval()
    return processor, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--processor",
        help="Optional local processor path; defaults to --model and never to the adapter.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch

    from qwen3vl_qlora_smoke import _messages

    rows, deployment_contract, data_audit = _load_audited_validation(args.data)
    if not rows:
        raise ValueError("validation dataset is empty")
    declared_processor = deployment_contract.get("processor")
    if not isinstance(declared_processor, Mapping):
        raise ValueError("validation deployment contract has no processor configuration")
    min_pixels = declared_processor.get("min_pixels")
    max_pixels = declared_processor.get("max_pixels")
    max_sequence_length = declared_processor.get("max_sequence_length")
    if (
        not isinstance(min_pixels, int)
        or not isinstance(max_pixels, int)
        or not isinstance(max_sequence_length, int)
    ):
        raise ValueError("validation processor contract is invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; evaluation must execute on the GPU server")
    processor, model = _load_model(
        args.model,
        args.adapter,
        processor_name=args.processor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    device = torch.device("cuda:0")
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        messages = _messages(row, args.data.expanduser().resolve().parent)
        inputs = processor.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        prompt_len = int(inputs["input_ids"].shape[-1])
        if prompt_len > max_sequence_length:
            raise ValueError(
                f"row {row.get('id')} exceeds the validation sequence-length contract"
            )
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
        gold_text = str(row["conversations"][-1]["value"])
        gold, gold_error = parse_strict_decision(gold_text, allow_uncertain=False)
        if gold_error or gold not in {"true", "false"}:
            raise ValueError(f"row {row.get('id')} has an invalid binary gold target: {gold_error}")
        predicted, parse_error = parse_strict_decision(text, allow_uncertain=True)
        predictions.append(
            {
                "id": row.get("id"),
                "image_id": row.get("meta", {}).get("image_id"),
                "class_name": row.get("meta", {}).get("class_name"),
                "gold": gold,
                "prediction": predicted,
                "correct": predicted == gold,
                "parse_error": parse_error,
                "raw_response": text,
            }
        )
        if index % 25 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)

    metrics = summarize_predictions(predictions)
    summary = {
        "schema_version": 2,
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "processor": args.processor or args.model,
        "data": str(args.data.resolve()),
        "deployment_contract": deployment_contract,
        "data_audit": data_audit,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "strict_json_policy": {
            "required_shape": '{"decision":"CONFIRMED|REJECTED"}',
            "invalid_output": "uncertain",
        },
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
