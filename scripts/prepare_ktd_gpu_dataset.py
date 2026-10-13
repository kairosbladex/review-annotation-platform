#!/usr/bin/env python3
"""Prepare a class-name-mapped KTD YOLO dataset on a GPU host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
        rows.append(value)
    return rows


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
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("class_id values must be unique")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("class_id values must be contiguous and start at 0")
    return mapping


def _split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> dict[str, str]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    image_ids = [str(row.get("image_id") or "") for row in rows]
    if any(not image_id for image_id in image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("yolo_image_manifest must have unique non-empty image_id values")
    ordered = sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest(),
    )
    val_count = min(max(1, round(len(ordered) * val_ratio)), len(ordered) - 1)
    return {image_id: ("val" if index < val_count else "train") for index, image_id in enumerate(ordered)}


def _validate_label_file(path: Path, class_count: int) -> int:
    count = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label at {path}:{line_no}")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"non-numeric YOLO label at {path}:{line_no}") from exc
        center_x, center_y, width, height = values
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
            raise ValueError(f"YOLO label outside bounds at {path}:{line_no}")
        count += 1
    if count == 0:
        raise ValueError(f"empty YOLO label file: {path}")
    return count


def _download(url: str, destination: Path, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "ktd-yolo-dataset-preparer/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"empty image response: {url}")
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    digest = hashlib.sha256(payload).hexdigest()
    return len(payload), digest


def _write_dataset_yaml(path: Path, dataset_root: Path, class_map: dict[str, int]) -> None:
    names = {class_id: name for name, class_id in class_map.items()}
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {class_id}: {name}" for class_id, name in sorted(names.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(
    *,
    manifest_path: Path,
    class_map_path: Path,
    output_dir: Path,
    image_base_url: str,
    download: bool,
    val_ratio: float,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    rows = _jsonl(manifest_path)
    class_map = _load_class_map(class_map_path)
    split_by_image = _split_rows(rows, val_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    prepared_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in rows:
        image_id = str(row["image_id"])
        split = split_by_image[image_id]
        object_key = str(row.get("image_object_key") or row.get("file_path") or "")
        if not object_key:
            raise ValueError(f"missing image object key for {image_id}")
        suffix = Path(object_key).suffix.lower() or ".jpg"
        label_source = manifest_path.parent / str(row["label_file"])
        label_destination = output_dir / "labels" / split / f"{image_id}.txt"
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        label_count = _validate_label_file(label_source, len(class_map))
        shutil.copy2(label_source, label_destination)
        image_destination = output_dir / "images" / split / f"{image_id}{suffix}"
        image_url = f"{image_base_url.rstrip('/')}/{object_key.lstrip('/')}"
        record: dict[str, Any] = {
            "image_id": image_id,
            "split": split,
            "image_object_key": object_key,
            "image_file": str(image_destination.relative_to(output_dir)),
            "label_file": str(label_destination.relative_to(output_dir)),
            "label_rows": label_count,
            "downloaded": False,
            "image_bytes": None,
            "image_sha256": None,
        }
        if download:
            try:
                image_bytes, image_sha256 = _download(image_url, image_destination, timeout)
                record.update(downloaded=True, image_bytes=image_bytes, image_sha256=image_sha256)
            except Exception as exc:  # retain per-image evidence before failing the batch
                errors.append({"image_id": image_id, "url": image_url, "error": str(exc)})
        prepared_rows.append(record)

    (output_dir / "dataset_manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in prepared_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "classes.txt").write_text(
        "\n".join(f"{class_id}\t{name}" for name, class_id in sorted(class_map.items(), key=lambda item: item[1])) + "\n",
        encoding="utf-8",
    )
    _write_dataset_yaml(output_dir / "dataset.yaml", output_dir, class_map)
    if errors:
        (output_dir / "download_errors.jsonl").write_text(
            "\n".join(json.dumps(error, ensure_ascii=False) for error in errors) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"failed to download {len(errors)} of {len(rows)} images")
    return {
        "images": len(rows),
        "train_images": sum(row["split"] == "train" for row in prepared_rows),
        "val_images": sum(row["split"] == "val" for row in prepared_rows),
        "label_rows": sum(row["label_rows"] for row in prepared_rows),
        "classes": len(class_map),
        "downloaded": sum(row["downloaded"] for row in prepared_rows),
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="yolo_image_manifest.jsonl")
    parser.add_argument("--class-map", type=Path, required=True, help="class_map.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-base-url", default="", help="public image URL prefix")
    parser.add_argument("--download", action="store_true", help="download source images")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.download and not args.image_base_url:
        parser.error("--image-base-url is required with --download")
    print(json.dumps(prepare(
        manifest_path=args.manifest,
        class_map_path=args.class_map,
        output_dir=args.output_dir,
        image_base_url=args.image_base_url,
        download=args.download,
        val_ratio=args.val_ratio,
        seed=args.seed,
        timeout=args.timeout,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
