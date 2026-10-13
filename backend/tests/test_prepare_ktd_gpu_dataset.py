from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_ktd_gpu_dataset import prepare


def test_prepare_uses_deterministic_split_and_class_map(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    class_map = source / "class_map.json"
    class_map.write_text(json.dumps({"classes": {"车": 0, "人员": 1}}), encoding="utf-8")
    rows = []
    for index in range(5):
        image_id = f"img-{index}"
        label_file = source / "labels" / f"{image_id}.txt"
        label_file.parent.mkdir(exist_ok=True)
        label_file.write_text(f"{index % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        rows.append({
            "image_id": image_id,
            "image_object_key": f"images/{image_id}.jpg",
            "label_file": f"labels/{image_id}.txt",
        })
    manifest = source / "yolo_image_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output = tmp_path / "dataset"
    summary = prepare(
        manifest_path=manifest,
        class_map_path=class_map,
        output_dir=output,
        image_base_url="",
        download=False,
        val_ratio=0.2,
        seed=42,
        timeout=1,
    )

    assert summary["images"] == 5
    assert summary["train_images"] == 4
    assert summary["val_images"] == 1
    assert summary["label_rows"] == 5
    assert (output / "dataset.yaml").read_text(encoding="utf-8").endswith("  1: 人员\n")
    assert len(list((output / "labels" / "train").glob("*.txt"))) == 4
    assert len(list((output / "labels" / "val").glob("*.txt"))) == 1
