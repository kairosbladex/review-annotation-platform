#!/usr/bin/env python3
"""Generate candidate decisions on a held-out Qwen3-VL verifier split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from qwen3vl_qlora_smoke import _configure_processor, _messages, _read_jsonl


def _decision(text: str) -> str | None:
    upper = text.upper()
    if "CONFIRMED" in upper:
        return "true"
    if "REJECTED" in upper:
        return "false"
    return None


def _load_model(model_name: str, adapter: Path | None) -> tuple[Any, Any]:
    dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        str(adapter) if adapter else model_name,
        local_files_only=True,
    )
    _configure_processor(processor, 28 * 28 * 64, 28 * 28 * 256)
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; evaluation must execute on the GPU server")
    rows = _read_jsonl(args.data)
    if not rows:
        raise ValueError("validation dataset is empty")
    processor, model = _load_model(args.model, args.adapter)
    device = torch.device("cuda:0")
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        messages = _messages(row, args.data.parent)
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
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        prompt_len = int(inputs["input_ids"].shape[-1])
        text = processor.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
        gold_text = str(row["conversations"][-1]["value"])
        gold = _decision(gold_text)
        predicted = _decision(text)
        predictions.append({
            "id": row.get("id"),
            "image_id": row.get("meta", {}).get("image_id"),
            "class_name": row.get("meta", {}).get("class_name"),
            "gold": gold,
            "prediction": predicted,
            "correct": predicted == gold,
            "raw_response": text,
        })
        if index % 25 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)

    tp = sum(r["gold"] == "true" and r["prediction"] == "true" for r in predictions)
    fp = sum(r["gold"] == "false" and r["prediction"] == "true" for r in predictions)
    fn = sum(r["gold"] == "true" and r["prediction"] == "false" for r in predictions)
    tn = sum(r["gold"] == "false" and r["prediction"] == "false" for r in predictions)
    valid = sum(r["prediction"] in {"true", "false"} for r in predictions)
    total = len(predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "data": str(args.data),
        "total": total,
        "valid_predictions": valid,
        "valid_prediction_rate": valid / total if total else 0.0,
        "accuracy": sum(r["correct"] for r in predictions) / total if total else 0.0,
        "precision_true": precision,
        "recall_true": recall,
        "f1_true": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions),
        encoding="utf-8",
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
