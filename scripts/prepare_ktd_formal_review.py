#!/usr/bin/env python3
"""Prepare a content-grouped KTD dataset and a human-confirmation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_class_map(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("class_map.json must contain a non-empty classes object")
    mapping: dict[str, int] = {}
    for name, class_id in classes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("class names must be non-empty strings")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
            raise ValueError(f"invalid class_id for {name!r}")
        mapping[name] = class_id
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("class_id values must be contiguous and start at 0")
    return mapping


def _validate_label_file(path: Path, class_count: int) -> int:
    count = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"invalid detection label at {path}:{line_no}")
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"non-numeric detection label at {path}:{line_no}") from exc
        if not 0 <= class_id < class_count or not (
            0 <= center_x <= 1
            and 0 <= center_y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
            and center_x - width / 2 >= -1e-8
            and center_x + width / 2 <= 1 + 1e-8
            and center_y - height / 2 >= -1e-8
            and center_y + height / 2 <= 1 + 1e-8
        ):
            raise ValueError(f"detection label outside bounds at {path}:{line_no}")
        count += 1
    if count == 0:
        raise ValueError(f"empty detection label file: {path}")
    return count


def split_by_content_sha(
    rows: list[dict[str, Any]], *, val_ratio: float, seed: int
) -> dict[str, str]:
    """Assign complete SHA-256 groups to train or val deterministically."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    groups: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        image_id = str(row.get("image_id") or "")
        digest = str(row.get("image_sha256") or "")
        if not image_id or image_id in seen_ids:
            raise ValueError("dataset manifest must have unique non-empty image_id values")
        if not digest:
            raise ValueError(f"missing image_sha256 for {image_id}")
        seen_ids.add(image_id)
        groups[digest].append(image_id)

    target = min(max(1, round(len(rows) * val_ratio)), len(rows) - 1)
    ordered = sorted(
        groups,
        key=lambda digest: hashlib.sha256(f"{seed}:{digest}".encode("utf-8")).hexdigest(),
    )
    val_groups: set[str] = set()
    val_count = 0
    for digest in ordered:
        group_size = len(groups[digest])
        if val_count + group_size <= target:
            val_groups.add(digest)
            val_count += group_size
    if not val_groups:
        val_groups.add(ordered[0])
        val_count = len(groups[ordered[0]])
    if val_count == len(rows):
        val_groups.remove(ordered[-1])

    return {
        image_id: ("val" if digest in val_groups else "train")
        for digest, image_ids in groups.items()
        for image_id in image_ids
    }


def _write_dataset_yaml(path: Path, dataset_root: Path, class_map: dict[str, int]) -> None:
    names = {class_id: name for name, class_id in class_map.items()}
    lines = [f"path: {dataset_root}", "train: images/train", "val: images/val", "names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in sorted(names.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_grouped_dataset(
    *,
    source_root: Path,
    manifest_path: Path,
    class_map_path: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    rows = _read_jsonl(manifest_path)
    class_map = _load_class_map(class_map_path)
    split_by_id = split_by_content_sha(rows, val_ratio=val_ratio, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row["image_id"])
        split = split_by_id[image_id]
        source_image = source_root / str(row["image_file"])
        source_label = source_root / str(row["label_file"])
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"missing prepared source for {image_id}")
        label_rows = _validate_label_file(source_label, len(class_map))
        suffix = source_image.suffix.lower() or ".jpg"
        image_dest = output_dir / "images" / split / f"{image_id}{suffix}"
        label_dest = output_dir / "labels" / split / f"{image_id}.txt"
        shutil.copy2(source_image, image_dest)
        shutil.copy2(source_label, label_dest)
        prepared.append({
            "image_id": image_id,
            "split": split,
            "content_group": str(row["image_sha256"]),
            "image_object_key": row.get("image_object_key"),
            "image_sha256": row["image_sha256"],
            "source_image_file": str(row["image_file"]),
            "source_label_file": str(row["label_file"]),
            "image_file": str(image_dest.relative_to(output_dir)),
            "label_file": str(label_dest.relative_to(output_dir)),
            "label_rows": label_rows,
        })

    _write_jsonl(output_dir / "dataset_manifest.jsonl", prepared)
    (output_dir / "classes.txt").write_text(
        "\n".join(f"{class_id}\t{name}" for name, class_id in sorted(class_map.items(), key=lambda item: item[1])) + "\n",
        encoding="utf-8",
    )
    _write_dataset_yaml(output_dir / "dataset.yaml", output_dir.resolve(), class_map)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        groups[row["content_group"]].append(row)
    cross_split = [
        digest for digest, members in groups.items()
        if len({member["split"] for member in members}) > 1
    ]
    audit = {
        "schema_version": 1,
        "split_policy": "content_sha256_grouped",
        "seed": seed,
        "val_ratio": val_ratio,
        "target_val_images": min(max(1, round(len(rows) * val_ratio)), len(rows) - 1),
        "images": len(prepared),
        "train_images": sum(row["split"] == "train" for row in prepared),
        "val_images": sum(row["split"] == "val" for row in prepared),
        "label_rows": sum(row["label_rows"] for row in prepared),
        "content_groups": len(groups),
        "duplicate_content_groups": sum(len(members) > 1 for members in groups.values()),
        "cross_split_content_groups": cross_split,
    }
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if cross_split:
        raise AssertionError(f"content groups cross train/val: {cross_split}")
    return audit


def build_human_confirmation_queue(
    *, candidate_results_path: Path, output_path: Path, image_bucket: str
) -> dict[str, Any]:
    candidates = _read_jsonl(candidate_results_path)
    queue: list[dict[str, Any]] = []
    for row in candidates:
        run_id = str(row.get("run_id") or "")
        image_id = str(row.get("image_id") or "")
        candidate_index = row.get("candidate_index")
        queue_id = f"{run_id}:{image_id}:{candidate_index}"
        suggested_label = row.get("auto_label")
        queue.append({
            "queue_id": queue_id,
            "queue_status": "pending",
            "priority": "high" if suggested_label == "uncertain" else "normal",
            "run_id": run_id,
            "run_stage": row.get("run_stage"),
            "parent_run_id": row.get("parent_run_id"),
            "image_id": image_id,
            "image": {
                "bucket": image_bucket,
                "object_key": row.get("file_path"),
            },
            "file_path": row.get("file_path"),
            "source_row_ids": row.get("source_row_ids", []),
            "candidate_index": candidate_index,
            "source_candidate_index": row.get("source_candidate_index"),
            "class_name": row.get("class_name"),
            "confidence": row.get("confidence"),
            "bbox": row.get("bbox"),
            "model_suggestion": suggested_label,
            "human_label": None,
            "human_note": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "provenance": {
                "label_source": row.get("label_source"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "decision": row.get("decision"),
                "first_pass_label": row.get("first_pass_label"),
                "first_pass_decision": row.get("first_pass_decision"),
                "second_pass_label": row.get("second_pass_label"),
                "second_pass_decision": row.get("second_pass_decision"),
                "candidate_results_file": str(candidate_results_path),
                "attempts_file": str(candidate_results_path.with_name("attempts.jsonl")),
            },
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, queue)
    suggestions = defaultdict(int)
    for row in queue:
        suggestions[str(row["model_suggestion"])] += 1
    return {
        "queue_rows": len(queue),
        "high_priority_uncertain": sum(row["priority"] == "high" for row in queue),
        "model_suggestions": dict(sorted(suggestions.items())),
        "human_labels_present": sum(row["human_label"] is not None for row in queue),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-bucket", default="minio-src")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_audit = build_grouped_dataset(
        source_root=args.source_root,
        manifest_path=args.dataset_manifest,
        class_map_path=args.class_map,
        output_dir=args.output_dir / "content-grouped-dataset",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    queue_audit = build_human_confirmation_queue(
        candidate_results_path=args.candidate_results,
        output_path=args.output_dir / "human_confirmation_queue.jsonl",
        image_bucket=args.image_bucket,
    )
    summary = {"dataset": dataset_audit, "queue": queue_audit}
    (args.output_dir / "review_preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
