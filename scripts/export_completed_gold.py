#!/usr/bin/env python3
"""Dump completed human gold labels to a local folder.

The platform export API queues Celery and writes to object storage. This
machine uses a read-only HTTP image prefix and has no Redis, so this script
writes JSONL + YOLO labels locally. False boxes never enter positive labels.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import httpx

CLASS_MAP = {
    "三轮车": 0,
    "两轮车": 1,
    "人员": 2,
    "光伏板": 3,
    "光斑": 4,
    "公交车": 5,
    "卡车": 6,
    "没戴头盔": 7,
    "车": 8,
    "违建": 9,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/ktd-local.sqlite3")
    parser.add_argument("--output-dir", default="data/gold-export-20260903")
    parser.add_argument("--source-base-url", default="https://lxtest.zhuoyilingxi.com:8088/minio-src")
    parser.add_argument("--download-images", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.database)
    con.row_factory = sqlite3.Row
    tasks = []
    for task in con.execute("SELECT id, sample_id, model_run_id, status FROM annotation_tasks ORDER BY updated_at"):
        candidates = list(con.execute(
            "SELECT id FROM candidates WHERE sample_id=? AND model_run_id=?",
            (task["sample_id"], task["model_run_id"]),
        ))
        if not candidates:
            continue
        unlabeled = 0
        for candidate in candidates:
            annotation = con.execute("SELECT label FROM annotations WHERE candidate_id=?", (candidate["id"],)).fetchone()
            if annotation is None or annotation["label"] is None:
                unlabeled += 1
        if unlabeled == 0:
            tasks.append(task)
    rows = []
    hard_negatives = []
    for task in tasks:
        sample = con.execute("SELECT * FROM samples WHERE id=?", (task["sample_id"],)).fetchone()
        asset = con.execute("SELECT * FROM assets WHERE sample_id=? AND kind='original'", (task["sample_id"],)).fetchone()
        candidates = list(con.execute(
            "SELECT * FROM candidates WHERE sample_id=? AND model_run_id=? ORDER BY target_index",
            (task["sample_id"], task["model_run_id"]),
        ))
        yolo_lines = []
        for candidate in candidates:
            annotation = con.execute("SELECT * FROM annotations WHERE candidate_id=?", (candidate["id"],)).fetchone()
            if annotation is None or annotation["label"] is None:
                raise SystemExit(f"completed task {task['id']} has unlabeled candidate {candidate['id']}")
            class_id = CLASS_MAP.get(candidate["class_name"])
            if class_id is None:
                raise SystemExit(f"unknown class_name {candidate['class_name']}")
            record = {
                "sample_id": sample["id"],
                "source_event_id": sample["source_event_id"],
                "candidate_id": candidate["id"],
                "target_index": candidate["target_index"],
                "bbox": [candidate["x1"], candidate["y1"], candidate["x2"], candidate["y2"]],
                "class_id": class_id,
                "class_name": candidate["class_name"],
                "human_label": annotation["label"],
                "human_note": annotation["note"],
                "labeled_by": annotation["labeled_by"],
                "labeled_at": annotation["labeled_at"],
                "image_object_key": asset["object_key"] if asset else None,
            }
            rows.append(record)
            if annotation["label"] == "false":
                hard_negatives.append(record)
                continue
            if annotation["label"] != "true":
                continue
            x1, y1, x2, y2 = record["bbox"]
            yolo_lines.append(f"{class_id} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}")
        (output / "labels" / f"{sample['id']}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        if args.download_images and asset is not None:
            url = f"{args.source_base_url.rstrip('/')}/{asset['object_key'].lstrip('/')}"
            response = httpx.get(url, timeout=60, follow_redirects=True, verify=True, trust_env=False)
            response.raise_for_status()
            suffix = Path(asset["object_key"]).suffix.lower() or ".jpg"
            (output / "images" / f"{sample['id']}{suffix}").write_bytes(response.content)

    (output / "annotations.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "hard_negative.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hard_negatives), encoding="utf-8")
    (output / "class_map.json").write_text(json.dumps({"classes": CLASS_MAP, "id_source": "class_name_sorted_dataset_local"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "completed_samples": len(tasks),
        "rows": len(rows),
        "true": sum(1 for row in rows if row["human_label"] == "true"),
        "false": sum(1 for row in rows if row["human_label"] == "false"),
        "human_labeled_rows": sum(1 for row in rows if row["labeled_by"]),
        "auto_labeled_rows": sum(1 for row in rows if not row["labeled_by"]),
        "downloaded_images": args.download_images,
        "note": "Only completed samples. Platform /exports is blocked locally (Celery + read-only HTTP store).",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
