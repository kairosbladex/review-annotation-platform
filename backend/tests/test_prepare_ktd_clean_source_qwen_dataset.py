from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from prepare_ktd_clean_source_qwen_dataset import VIEW_SCHEMA, build_dataset
from qwen_training_provenance import (
    validate_disjoint_image_splits,
    validate_human_gold_dataset,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    pairs: list[dict] = []
    gold: list[dict] = []
    specs = (
        ("train-1", "train", "event-train", "task-train", "车", "true", "strong", True),
        ("val-1", "validation", "event-val", "task-val", "人员", "false", "usable", True),
        ("excluded", "train", "event-missing", "task-missing", "人员", "false", None, False),
    )
    for candidate_id, partition, event_id, task_id, class_name, label, status, has_clean in specs:
        row = {
            "candidate_id": candidate_id,
            "partition": partition,
            "event_id": event_id,
            "source_task_id": task_id,
            "class_name": class_name,
            "human_label": label,
            "bbox_px": [1, 2, 30, 40],
            "has_clean_source": has_clean,
            "clean_match_status": status,
            "clean_source_kind": "primary" if has_clean else None,
        }
        for role in ("annotated_context", "annotated_roi", "clean_context", "clean_roi"):
            path = source / f"{candidate_id}-{role}.jpg"
            path.write_bytes(f"{candidate_id}:{role}".encode())
            row[role] = path.name
            row[role + "_sha256"] = _sha(path)
        clean_frame = source / f"{candidate_id}-clean-frame.jpg"
        clean_frame.write_bytes(f"{candidate_id}:clean-frame".encode())
        row["clean_frame"] = clean_frame.name
        pairs.append(row)
        gold.append(
            {
                "candidate_id": candidate_id,
                "partition": partition,
                "event_id": event_id,
                "class_name": class_name,
                "human_label": label,
            }
        )
    pairs_path = source / "candidate_pairs.jsonl"
    gold_path = source / "gold_labels.jsonl"
    _write_jsonl(pairs_path, pairs)
    _write_jsonl(gold_path, gold)
    return pairs_path, gold_path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_builds_self_contained_four_view_dataset_and_preserves_partition(tmp_path: Path):
    pairs, gold = _source_fixture(tmp_path)
    output = tmp_path / "output"

    summary = build_dataset(candidate_pairs_path=pairs, gold_path=gold, output_dir=output)

    train = _read(output / "train.jsonl")
    validation = _read(output / "validation.jsonl")
    lineage = _read(output / "lineage_manifest.jsonl")
    assert summary["eligible_candidates"] == 2
    assert summary["excluded_candidates"] == 1
    contract = summary["deployment_contract"]
    assert contract["view_roles"] == [
        "annotated_context",
        "annotated_roi",
        "clean_context",
        "clean_roi",
    ]
    assert contract["strict_output"]["allowed_decisions"] == [
        "CONFIRMED",
        "REJECTED",
    ]
    assert contract["processor"] == {
        "min_pixels": 50176,
        "max_pixels": 200704,
        "max_sequence_length": 2048,
    }
    assert len(contract["prompt_template_sha256"]) == 64
    assert summary["partition_rows"] == {"train": 1, "validation": 1}
    assert summary["isolation"] == {
        "clean_frame_sha256_overlap": [],
        "source_task_id_overlap": [],
        "event_id_overlap": [],
    }
    assert train[0]["meta"]["partition"] == "train"
    assert validation[0]["meta"]["partition"] == "validation"
    assert train[0]["meta"]["view_schema"] == VIEW_SCHEMA
    assert len(train[0]["image"]) == 4
    assert all((output / path).is_file() for path in train[0]["image"])
    assert sum(row["eligible"] for row in lineage) == 2
    assert next(row for row in lineage if row["candidate_id"] == "excluded")["exclusion_reason"]


def test_four_view_provenance_hashes_and_split_isolation_are_enforced(tmp_path: Path):
    pairs, gold = _source_fixture(tmp_path)
    output = tmp_path / "output"
    build_dataset(candidate_pairs_path=pairs, gold_path=gold, output_dir=output)
    train_rows = _read(output / "train.jsonl")
    val_rows = _read(output / "validation.jsonl")

    train_summary = validate_human_gold_dataset(
        train_rows, dataset_path=output / "train.jsonl", provenance_root=output
    )
    val_summary = validate_human_gold_dataset(
        val_rows, dataset_path=output / "validation.jsonl", provenance_root=output
    )
    validate_disjoint_image_splits(train_summary, val_summary)
    assert train_summary["view_schemas"] == [VIEW_SCHEMA]
    assert train_summary["unique_image_sha256"] == 1
    assert len(train_summary["view_sha256s"]) == 4
    assert len(train_summary["clean_frame_sha256s"]) == 1

    (output / train_rows[0]["image"][2]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="clean_context training-view SHA-256 mismatch"):
        validate_human_gold_dataset(
            train_rows, dataset_path=output / "train.jsonl", provenance_root=output
        )


def test_rejects_candidate_gold_mismatch(tmp_path: Path):
    pairs, gold = _source_fixture(tmp_path)
    rows = _read(gold)
    rows[0]["human_label"] = "false"
    _write_jsonl(gold, rows)

    with pytest.raises(ValueError, match="differs between pair and gold"):
        build_dataset(candidate_pairs_path=pairs, gold_path=gold, output_dir=tmp_path / "output")
