from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen3vl_qlora_train import _balanced_sampling_manifest


def _row(class_name: str, label: str) -> dict:
    return {
        "meta": {"class_name": class_name, "human_label": label},
        "conversations": [],
    }


def test_balanced_sampling_equalizes_class_label_stratum_weight_mass():
    rows = (
        [_row("车", "true") for _ in range(59)]
        + [_row("车", "false") for _ in range(7)]
        + [_row("人员", "true") for _ in range(8)]
        + [_row("人员", "false") for _ in range(17)]
    )

    weights, manifest = _balanced_sampling_manifest(rows, seed=42)

    assert manifest["stratum_counts"] == {
        "人员|false": 17,
        "人员|true": 8,
        "车|false": 7,
        "车|true": 59,
    }
    assert manifest["validation_sampling"] == "natural_original_partition"
    masses = {}
    for row, weight in zip(rows, weights):
        key = row["meta"]["class_name"] + "|" + row["meta"]["human_label"]
        masses[key] = masses.get(key, 0.0) + weight
    assert len(set(round(value, 10) for value in masses.values())) == 1
    assert sum(weights) == pytest.approx(len(rows))


def test_loads_and_hash_checks_clean_source_deployment_contract(tmp_path: Path):
    import hashlib
    import json

    from qwen3vl_qlora_train import (
        _QUANTIZATION_CONTRACT,
        _STRICT_OUTPUT_CONTRACT,
        _load_dataset_contract,
    )

    prompt_template = "class={class_name}"
    prompt_template_sha = hashlib.sha256(prompt_template.encode()).hexdigest()
    rows = [
        {
            "meta": {
                "view_schema": "annotated_locator_plus_clean_original_v1",
                "prompt_template_sha256": prompt_template_sha,
                "class_name": "车",
            },
            "conversations": [
                {"from": "human", "value": "class=车"},
                {"from": "gpt", "value": '{"decision":"CONFIRMED"}'},
            ],
        }
    ]
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    payload = json.dumps(rows[0]) + "\n"
    train.write_text(payload)
    val.write_text(payload)
    file_sha = hashlib.sha256(payload.encode()).hexdigest()
    contract = {
        "view_schema": "annotated_locator_plus_clean_original_v1",
        "view_roles": [
            "annotated_context",
            "annotated_roi",
            "clean_context",
            "clean_roi",
        ],
        "prompt_template": prompt_template,
        "prompt_template_sha256": prompt_template_sha,
        "strict_output": _STRICT_OUTPUT_CONTRACT,
        "processor": {
            "min_pixels": 50176,
            "max_pixels": 200704,
            "max_sequence_length": 2048,
        },
        "quantization": _QUANTIZATION_CONTRACT,
    }
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "deployment_contract": contract,
                "outputs": {
                    "train": {"rows": 1, "sha256": file_sha},
                    "validation": {"rows": 1, "sha256": file_sha},
                },
            }
        )
    )

    loaded = _load_dataset_contract(train, val, rows, rows)

    assert loaded["prompt_template_sha256"] == prompt_template_sha
    assert loaded["strict_output"] == _STRICT_OUTPUT_CONTRACT
    assert loaded["contract_source"].endswith("dataset_manifest.json")


def test_configure_processor_supports_transformers_size_dict_shape():
    from qwen3vl_qlora_smoke import _configure_processor

    class SizeLike:
        def __init__(self) -> None:
            self.shortest_edge = 65536
            self.longest_edge = 16777216

    class ImageProcessor:
        def __init__(self) -> None:
            self.size = SizeLike()

    class Processor:
        def __init__(self) -> None:
            self.image_processor = ImageProcessor()

    processor = Processor()
    _configure_processor(processor, 50176, 200704)

    assert processor.image_processor.size.shortest_edge == 50176
    assert processor.image_processor.size.longest_edge == 200704


def test_configure_processor_rejects_unverifiable_pixel_budget():
    from qwen3vl_qlora_smoke import _configure_processor

    class Processor:
        image_processor = object()

    with pytest.raises(RuntimeError, match="no supported pixel-budget fields"):
        _configure_processor(Processor(), 50176, 200704)


def test_capped_sampling_limits_boost_and_keeps_mean_weight():
    from qwen3vl_qlora_train import _capped_sampling_manifest

    rows = (
        [_row("车", "true") for _ in range(92)]
        + [_row("车", "false") for _ in range(18)]
        + [_row("人员", "true") for _ in range(8)]
        + [_row("人员", "false") for _ in range(17)]
    )
    weights, manifest = _capped_sampling_manifest(rows, seed=42, cap=1.0)
    assert sum(weights) == pytest.approx(len(rows))
    per = manifest["per_row_weight_by_stratum"]
    # cap=1：小组不再被放大，三个小组权重相同，大组被压低
    assert per["车|false"] == pytest.approx(per["人员|true"])
    assert per["人员|false"] == pytest.approx(per["人员|true"])
    assert per["车|true"] < per["车|false"]
    assert manifest["stratum_weight_cap"] == 1.0

    balanced, _ = _balanced_sampling_manifest(rows, seed=42)
    capped_large, _ = _capped_sampling_manifest(rows, seed=42, cap=1000.0)
    assert capped_large == pytest.approx(balanced)


def test_capped_sampling_rejects_non_positive_cap():
    from qwen3vl_qlora_train import _capped_sampling_manifest

    with pytest.raises(ValueError):
        _capped_sampling_manifest([_row("车", "true")], seed=42, cap=0)


def test_negative_oversample_ratio_boosts_false_strata_and_preserves_total_weight():
    rows = (
        [_row("车", "true") for _ in range(40)]
        + [_row("车", "false") for _ in range(10)]
        + [_row("人员", "true") for _ in range(20)]
        + [_row("人员", "false") for _ in range(10)]
    )
    weights, manifest = _balanced_sampling_manifest(
        rows, seed=42, negative_oversample_ratio=1.3
    )
    assert manifest["negative_oversample_ratio"] == 1.3
    assert sum(weights) == pytest.approx(len(rows))
    masses = {}
    for row, weight in zip(rows, weights):
        key = row["meta"]["class_name"] + "|" + row["meta"]["human_label"]
        masses[key] = masses.get(key, 0.0) + weight
    assert masses["车|false"] / masses["车|true"] == pytest.approx(1.3)
    assert masses["人员|false"] / masses["人员|true"] == pytest.approx(1.3)


def test_lora_target_modules_and_multi_sample_pad_collator():
    import numpy as np
    from qwen3vl_qlora_smoke import (
        DEFAULT_LORA_TARGET_MODULES,
        MultiSamplePadCollator,
        _detect_gpu_capabilities,
        resolve_lora_target_modules,
    )

    assert DEFAULT_LORA_TARGET_MODULES == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    with_vision = resolve_lora_target_modules(include_vision_proj=True)
    assert "merger.mlp.0" in with_vision
    assert "merger.mlp.2" in with_vision

    caps = _detect_gpu_capabilities(0)
    assert "attn_implementation" in caps
    assert "recommended_batch_size" in caps

    collator = MultiSamplePadCollator(pad_token_id=0)
    f1 = {
        "input_ids": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]]),
        "labels": np.array([[-100, 2, 3]]),
        "pixel_values": np.zeros((4, 16)),
        "image_grid_thw": np.array([[1, 2, 2]]),
    }
    f2 = {
        "input_ids": np.array([[4, 5, 6, 7, 8]]),
        "attention_mask": np.array([[1, 1, 1, 1, 1]]),
        "labels": np.array([[-100, -100, 6, 7, 8]]),
        "pixel_values": np.ones((6, 16)),
        "image_grid_thw": np.array([[1, 2, 3]]),
    }
    batch = collator([f1, f2])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["labels"][0, 3:].tolist() == [-100, -100]
    assert batch["pixel_values"].shape == (10, 16)
    assert batch["image_grid_thw"].shape == (2, 3)
