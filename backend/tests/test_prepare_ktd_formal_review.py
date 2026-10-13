from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_ktd_formal_review import build_human_confirmation_queue, split_by_content_sha


def test_split_by_content_sha_keeps_duplicate_images_together():
    rows = [
        {"image_id": "a", "image_sha256": "same"},
        {"image_id": "b", "image_sha256": "same"},
        {"image_id": "c", "image_sha256": "other-1"},
        {"image_id": "d", "image_sha256": "other-2"},
    ]

    split = split_by_content_sha(rows, val_ratio=0.5, seed=42)

    assert split["a"] == split["b"]
    assert set(split.values()) == {"train", "val"}


def test_human_queue_keeps_machine_suggestion_separate(tmp_path: Path):
    candidate_path = tmp_path / "candidate_results.jsonl"
    candidate_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "run_id": "run-1",
                    "run_stage": "uncertain_recheck",
                    "image_id": "img-1",
                    "candidate_index": 0,
                    "file_path": "images/img-1.jpg",
                    "class_name": "车",
                    "auto_label": "uncertain",
                    "label_source": "model_pseudo",
                },
                {
                    "run_id": "run-1",
                    "run_stage": "uncertain_recheck",
                    "image_id": "img-2",
                    "candidate_index": 1,
                    "file_path": "images/img-2.jpg",
                    "class_name": "车",
                    "auto_label": "true",
                    "label_source": "model_pseudo",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "human_confirmation_queue.jsonl"

    summary = build_human_confirmation_queue(
        candidate_results_path=candidate_path,
        output_path=output,
        image_bucket="minio-src",
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["queue_rows"] == 2
    assert summary["human_labels_present"] == 0
    assert rows[0]["model_suggestion"] == "uncertain"
    assert rows[0]["human_label"] is None
    assert rows[0]["priority"] == "high"
    assert rows[1]["model_suggestion"] == "true"
    assert rows[1]["human_label"] is None
