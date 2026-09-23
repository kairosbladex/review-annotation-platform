from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_training_provenance import (
    deduplicate_human_gold_rows,
    validate_disjoint_image_splits,
    validate_human_gold_dataset,
    validate_supervision_lengths,
)


def _fixture(tmp_path: Path, labels: tuple[str, ...] = ("true", "true")):
    dataset = tmp_path / "train.jsonl"
    image = tmp_path / "source.jpg"
    image.write_bytes(b"stable source image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    source_file = tmp_path / "annotations.jsonl"
    context = tmp_path / "context.jpg"
    roi = tmp_path / "roi.jpg"
    context.write_bytes(b"context view")
    roi.write_bytes(b"roi view")
    source_rows = []
    training_rows = []
    for index, label in enumerate(labels):
        image_id = f"img-{index}"
        source_rows.append(
            {
                "sample_id": image_id,
                "target_index": str(index),
                "source_event_id": f"event-{index}",
                "human_label": label,
                "labeled_by": "reviewer@example.test",
                "labeled_at": "2026-09-22T12:00:00Z",
                "class_name": "车",
                "bbox": [1, 2, 30, 40],
                "image_object_key": "bucket/shared.jpg",
            }
        )
        verdict = "CONFIRMED" if label == "true" else "REJECTED"
        training_rows.append(
            {
                "id": f"row-{index}",
                "image": [context.name, roi.name],
                "conversations": [
                    {"from": "human", "value": "<image> <image> Candidate"},
                    {
                        "from": "gpt",
                        "value": json.dumps({"decision": verdict}),
                    },
                ],
                "meta": {
                    "label_source": "human_gold",
                    "label_field": "human_label",
                    "labeled_by": "reviewer@example.test",
                    "image_id": image_id,
                    "candidate_index": str(index),
                    "class_name": "车",
                    "bbox": [1, 2, 30, 40],
                    "image_sha256": digest,
                    "image_object_key": "bucket/shared.jpg",
                    "source_file": str(source_file),
                    "source_image": str(image),
                },
            }
        )
    source_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training_rows),
        encoding="utf-8",
    )
    return dataset, training_rows, image


def test_validates_human_gold_source_image_and_training_target(tmp_path: Path):
    dataset, rows, _ = _fixture(tmp_path, labels=("true",))

    summary = validate_human_gold_dataset(
        rows, dataset_path=dataset, provenance_root=tmp_path
    )

    assert summary["human_gold_rows"] == 1
    assert summary["unique_image_sha256"] == 1
    assert summary["duplicate_groups"] == 0


def test_rejects_pseudo_labels_and_training_target_mismatch(tmp_path: Path):
    dataset, rows, _ = _fixture(tmp_path, labels=("true",))
    rows[0]["meta"]["label_source"] = "codex_pseudo_prelabel"
    with pytest.raises(ValueError, match="not declared human gold"):
        validate_human_gold_dataset(rows, dataset_path=dataset, provenance_root=tmp_path)

    rows[0]["meta"]["label_source"] = "human_gold"
    rows[0]["conversations"][-1]["value"] = json.dumps({"decision": "REJECTED"})
    with pytest.raises(ValueError, match="training target differs"):
        validate_human_gold_dataset(rows, dataset_path=dataset, provenance_root=tmp_path)


def test_rejects_source_image_hash_and_box_mismatch(tmp_path: Path):
    dataset, rows, image = _fixture(tmp_path, labels=("true",))
    rows[0]["meta"]["bbox"] = [2, 2, 30, 40]
    with pytest.raises(ValueError, match="box differs"):
        validate_human_gold_dataset(rows, dataset_path=dataset, provenance_root=tmp_path)

    rows[0]["meta"]["bbox"] = [1, 2, 30, 40]
    image.write_bytes(b"changed source image")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_human_gold_dataset(rows, dataset_path=dataset, provenance_root=tmp_path)


def test_rejects_duplicates_then_collapses_consistent_copies(tmp_path: Path):
    dataset, rows, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate image/class/box"):
        validate_human_gold_dataset(rows, dataset_path=dataset, provenance_root=tmp_path)

    raw_summary = validate_human_gold_dataset(
        rows,
        dataset_path=dataset,
        provenance_root=tmp_path,
        allow_duplicates=True,
        allow_conflicts=True,
    )
    canonical, duplicate_audit, conflicts = deduplicate_human_gold_rows(rows)

    assert raw_summary["duplicate_extra_rows"] == 1
    assert len(canonical) == 1
    assert duplicate_audit[0]["status"] == "consistent_duplicate_collapsed"
    assert conflicts == []


def test_excludes_conflicting_duplicate_labels_for_adjudication(tmp_path: Path):
    dataset, rows, _ = _fixture(tmp_path, labels=("true", "false"))
    with pytest.raises(ValueError, match="conflicting human-label"):
        validate_human_gold_dataset(
            rows,
            dataset_path=dataset,
            provenance_root=tmp_path,
            allow_duplicates=True,
        )

    canonical, duplicate_audit, conflicts = deduplicate_human_gold_rows(rows)
    assert canonical == []
    assert len(conflicts) == 2
    assert duplicate_audit[0]["status"] == "conflicting_human_labels"


def test_rejects_same_source_image_across_train_and_validation():
    with pytest.raises(ValueError, match="share 1 source image"):
        validate_disjoint_image_splits(
            {
                "image_sha256s": {"shared"},
                "source_group_ids": {"event-a"},
            },
            {
                "image_sha256s": {"shared"},
                "source_group_ids": {"event-b"},
            },
        )
    with pytest.raises(ValueError, match="source event/task group"):
        validate_disjoint_image_splits(
            {"image_sha256s": {"image-a"}, "source_group_ids": {"shared-event"}},
            {"image_sha256s": {"image-b"}, "source_group_ids": {"shared-event"}},
        )


def test_supervision_guard_rejects_silent_truncation_and_empty_answer():
    assert validate_supervision_lengths(
        sequence_length=20, prompt_length=15, max_length=20, row_id="ok"
    ) == 5
    with pytest.raises(ValueError, match="refusing to truncate"):
        validate_supervision_lengths(
            sequence_length=21, prompt_length=15, max_length=20, row_id="long"
        )
    with pytest.raises(ValueError, match="no supervised assistant"):
        validate_supervision_lengths(
            sequence_length=15, prompt_length=15, max_length=20, row_id="empty"
        )
