from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from audit_codex_pilot_review import audit_review_export
from qwen_training_provenance import validate_human_gold_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pilot(tmp_path: Path) -> tuple[Path, Path]:
    pilot = tmp_path / "pilot"
    media = pilot / "media"
    media.mkdir(parents=True)
    original = media / "source-original-01.jpg"
    original.write_bytes(b"original-image")
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    # 溯源必需字段：由 build_codex_review_batch.py 写入，审计时落库为 gold 行的
    # image_object_key。夹具必须显式声明，否则审计会因缺少对象键而拒绝冻结。
    source_object_key = "ktd/2026/09/24/image-1/original.jpg"

    manifests = []
    prelabels = []
    reviews = []
    for index, suggestion in enumerate(("true", "uncertain"), start=1):
        candidate_id = f"P{index:03d}"
        bbox_norm = [0.1 * index, 0.2, 0.1 * index + 0.1, 0.4]
        bbox_px = [10 * index, 20, 10 * index + 10, 40]
        view_hashes = {}
        view_paths = {}
        for view_name in ("context", "roi", "roi-marked"):
            view = media / f"{candidate_id}-{view_name}.jpg"
            view.write_bytes(f"{candidate_id}-{view_name}".encode())
            view_paths[view_name] = str(view.relative_to(pilot))
            view_hashes[view_name] = hashlib.sha256(view.read_bytes()).hexdigest()
        manifests.append(
            {
                "candidate_id": candidate_id,
                "event_id": "image-1",
                "source_group": "event-group-1",
                "class_name": "车",
                "bbox_norm": bbox_norm,
                "bbox_px": bbox_px,
                "image_size": [100, 100],
                "original_sha256": original_sha,
                "source_object_key": source_object_key,
                "source_original": "source/original.jpg",
                "full_original_copy": str(original.relative_to(pilot)),
                "media": view_paths,
                "derived_sha256": view_hashes,
                "source_row_ordinals": [index - 1],
            }
        )
        prelabels.append(
            {
                "batch_id": "pilot-batch",
                "candidate_id": candidate_id,
                "codex_suggestion": suggestion,
                "visual_evidence": f"evidence-{index}",
                "prompt_version": "prompt-v1",
                "codex_model": "model",
            }
        )
        action = "confirm" if suggestion == "true" else "false"
        label = "true" if action == "confirm" else "false"
        reviews.append(
            {
                "batch_id": "pilot-batch",
                "candidate_id": candidate_id,
                "source_image_sha256": original_sha,
                "candidate_bbox_px": bbox_px,
                "candidate_class": "车",
                "source_group": "event-group-1",
                "codex_suggestion": suggestion,
                "codex_evidence": f"evidence-{index}",
                "human_action": action,
                "human_label": label,
                "human_comment": "",
                "human_reviewed": True,
                "reviewed_at": f"2026-09-24T02:48:0{index}.000Z",
                "label_source": "human_review_pending_audit",
                "is_training_eligible": False,
            }
        )

    _write_jsonl(pilot / "pilot_manifest.jsonl", manifests)
    _write_jsonl(pilot / "codex_prelabels.jsonl", prelabels)
    (pilot / "summary.json").write_text("{}\n", encoding="utf-8")
    review_export = tmp_path / "review.jsonl"
    _write_jsonl(review_export, reviews)
    return pilot, review_export


def test_audit_freezes_binary_human_gold_without_splitting_source_group(
    tmp_path: Path,
):
    pilot, review_export = _pilot(tmp_path)
    output = tmp_path / "audited"

    summary = audit_review_export(
        review_export=review_export,
        pilot_root=pilot,
        output_dir=output,
        reviewer_id="workspace-user:test",
        expected_count=2,
    )

    gold = [
        json.loads(line)
        for line in (output / "human_gold_annotations.jsonl").read_text().splitlines()
    ]
    assert summary["audit_status"] == "passed"
    assert summary["human_gold_rows"] == 2
    assert summary["excluded_rows"] == 0
    assert summary["comparison"] == {
        "confirmed_binary_suggestion": 1,
        "human_resolved_uncertain": 1,
    }
    assert {row["source_event_id"] for row in gold} == {"event-group-1"}
    assert all(row["split"] == "train_candidate_single_source_group" for row in gold)
    assert all(row["training_eligible"] for row in gold)
    # 溯源链不得断：每条 gold 行都必须携带非空 image_object_key
    assert all(row["image_object_key"] for row in gold)
    assert (output / "excluded_from_training.jsonl").read_text() == ""
    qwen_rows = [
        json.loads(line)
        for line in (output / "qwen_training_fragment.jsonl").read_text().splitlines()
    ]
    qwen_summary = validate_human_gold_dataset(
        qwen_rows,
        dataset_path=output / "qwen_training_fragment.jsonl",
        provenance_root=output,
    )
    assert qwen_summary["human_gold_rows"] == 2
    assert hashlib.sha256(
        (output / "source_human_review_export.jsonl").read_bytes()
    ).hexdigest() == hashlib.sha256(review_export.read_bytes()).hexdigest()


def test_audit_rejects_review_that_differs_from_frozen_candidate(tmp_path: Path):
    pilot, review_export = _pilot(tmp_path)
    rows = [json.loads(line) for line in review_export.read_text().splitlines()]
    rows[0]["candidate_bbox_px"] = [0, 0, 1, 1]
    _write_jsonl(review_export, rows)

    with pytest.raises(ValueError, match="candidate_bbox_px differs"):
        audit_review_export(
            review_export=review_export,
            pilot_root=pilot,
            output_dir=tmp_path / "audited",
            reviewer_id="workspace-user:test",
            expected_count=2,
        )


def test_audit_rejects_confirming_an_uncertain_machine_suggestion(tmp_path: Path):
    pilot, review_export = _pilot(tmp_path)
    rows = [json.loads(line) for line in review_export.read_text().splitlines()]
    rows[1]["human_action"] = "confirm"
    rows[1]["human_label"] = "uncertain"
    _write_jsonl(review_export, rows)

    with pytest.raises(ValueError, match="invalid confirm action"):
        audit_review_export(
            review_export=review_export,
            pilot_root=pilot,
            output_dir=tmp_path / "audited",
            reviewer_id="workspace-user:test",
            expected_count=2,
        )
