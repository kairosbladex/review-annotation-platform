from __future__ import annotations

import json
import hashlib
from pathlib import Path

from PIL import Image

from scripts.prepare_ktd_canonical_crop_dataset import build_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(canonical_id: str, image_sha256: str, object_key: str) -> dict:
    return {
        "canonical_id": canonical_id,
        "class_id": 8,
        "class_name": "车",
        "bbox_norm": [0.25, 0.25, 0.75, 0.75],
        "bbox_px": [2, 2, 6, 6],
        "image_sha256": image_sha256,
        "image_object_key": object_key,
        "image_width": 8,
        "image_height": 8,
        "detector_confidence": 0.8,
        "has_human_gold": False,
        "prior_gpt": {"passes": [{"label": "true", "status": "succeeded"}]},
        "prior_yolo": {},
        "prior_qwen": None,
        "aliases": [],
        "alias_count": 0,
    }


def test_build_dataset_keeps_gold_sources_separate_and_excludes_eval_images(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    image_root = tmp_path / "images"
    object_keys = ["a/one.jpg", "a/two.jpg", "a/three.jpg"]
    records = [
        _record("gold-selection", "selection-sha", object_keys[0]),
        _record("gold-locked", "locked-sha", object_keys[1]),
        _record("teacher-train", "train-sha", object_keys[2]),
    ]
    for key in object_keys:
        image_path = image_root / key
        image_path.parent.mkdir(parents=True, exist_ok=True)
        color = (40 + 20 * object_keys.index(key), 80, 120)
        Image.new("RGB", (8, 8), color).save(image_path)
    image_sha = hashlib.sha256((image_root / object_keys[0]).read_bytes()).hexdigest()
    locked_sha = hashlib.sha256((image_root / object_keys[1]).read_bytes()).hexdigest()
    train_sha = hashlib.sha256((image_root / object_keys[2]).read_bytes()).hexdigest()
    records[0]["image_sha256"] = image_sha
    records[1]["image_sha256"] = locked_sha
    records[2]["image_sha256"] = train_sha
    _write_jsonl(snapshot / "inference_records.jsonl", records)
    _write_jsonl(snapshot / "selection_inputs.jsonl", [
        {"canonical_id": "gold-selection", "image_sha256": image_sha, "image_object_key": object_keys[0]}
    ])
    _write_jsonl(snapshot / "locked_test_inputs.jsonl", [
        {"canonical_id": "gold-locked", "image_sha256": locked_sha, "image_object_key": object_keys[1]}
    ])
    _write_jsonl(snapshot / "gold_records.jsonl", [
        {"canonical_id": "gold-selection", "class_id": 8, "class_name": "车", "label": "false", "split": "selection", "image_sha256": image_sha, "image_object_key": object_keys[0], "bbox_norm": [0.25, 0.25, 0.75, 0.75], "bbox_px": [2, 2, 6, 6]},
        {"canonical_id": "gold-locked", "class_id": 8, "class_name": "车", "label": "true", "split": "locked_test", "image_sha256": locked_sha, "image_object_key": object_keys[1], "bbox_norm": [0.25, 0.25, 0.75, 0.75], "bbox_px": [2, 2, 6, 6]},
    ])
    (snapshot / "snapshot_manifest.json").write_text(
        json.dumps({"snapshot_id": "test-snapshot"}) + "\n", encoding="utf-8"
    )

    output = tmp_path / "output"
    output.mkdir()
    (output / "image_errors.jsonl").write_text("stale failure\n", encoding="utf-8")
    summary = build_dataset(snapshot_dir=snapshot, image_root=image_root, output_dir=output, image_size=32, val_ratio=0.5, seed=42)

    assert not (output / "image_errors.jsonl").exists()
    assert summary["canonical_rows"] == 3
    assert summary["selection_gold_rows"] == 1
    assert summary["locked_gold_rows"] == 1
    assert summary["leakage"]["train_vs_eval_sha"] == []
    assert summary["leakage"]["val_vs_eval_sha"] == []
    teacher_rows = [json.loads(line) for line in (output / "teacher_all.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["label_source"] for row in teacher_rows} == {"gpt_teacher"}
    train_rows = [json.loads(line) for line in (output / "teacher_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["image_sha256"] == train_sha for row in train_rows)
    assert all(row["teacher_label"] in {"true", "false"} for row in train_rows)
    manifest_rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["human_label"] is None for row in manifest_rows if row["split"] == "locked_test")
    assert len(list((output / "media").glob("*-roi.jpg"))) == 3
    with Image.open(next((output / "media").glob("*-roi.jpg"))) as crop:
        assert crop.size == (32, 32)


def test_build_dataset_rejects_overlapping_eval_image_sets(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_jsonl(snapshot / "inference_records.jsonl", [])
    _write_jsonl(snapshot / "selection_inputs.jsonl", [{"image_sha256": "same"}])
    _write_jsonl(snapshot / "locked_test_inputs.jsonl", [{"image_sha256": "same"}])
    _write_jsonl(snapshot / "gold_records.jsonl", [])

    try:
        build_dataset(snapshot_dir=snapshot, image_root=tmp_path, output_dir=tmp_path / "output", image_size=32, val_ratio=0.2, seed=42)
    except ValueError as exc:
        assert "selection and locked_test" in str(exc)
    else:
        raise AssertionError("overlapping evaluation image sets must fail")
