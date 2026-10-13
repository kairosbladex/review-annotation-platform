#!/usr/bin/env python3
"""Prepare ROI/context images for a Qwen3-VL verifier smoke run.

The input is the existing qwen_vl_training.jsonl. Its labels are model-derived
and are deliberately recorded as pseudo labels in the generated metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image, ImageDraw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"row {line_no} is not an object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_url(base_url: str, object_key: str) -> str:
    return f"{base_url.rstrip('/')}/{quote(object_key.lstrip('/'), safe='/')}"


def _download_image(
    client: httpx.Client, url: str, destination: Path
) -> tuple[Image.Image, str]:
    response = client.get(url)
    response.raise_for_status()
    payload = response.content
    if not payload:
        raise ValueError(f"empty image response: {url}")
    digest = hashlib.sha256(payload).hexdigest()
    destination.write_bytes(payload)
    with Image.open(destination) as image:
        return image.convert("RGB"), digest


def _render_crop(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    size: int,
) -> Image.Image:
    left, top, right, bottom = crop_box
    crop = image.crop((left, top, right, bottom))
    crop_w, crop_h = crop.size
    scale = min(size / crop_w, size / crop_h)
    resized = crop.resize(
        (max(1, round(crop_w * scale)), max(1, round(crop_h * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), (230, 230, 230))
    offset_x = (size - resized.width) // 2
    offset_y = (size - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    tx1, ty1, tx2, ty2 = target_box
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (
            offset_x + (tx1 - left) * scale,
            offset_y + (ty1 - top) * scale,
            offset_x + (tx2 - left) * scale,
            offset_y + (ty2 - top) * scale,
        ),
        outline=(0, 180, 80),
        width=max(2, round(size / 112)),
    )
    return canvas


def _square_context(
    image_size: tuple[int, int], target: tuple[int, int, int, int], scale: float
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = target
    side = max(2, round(max(x2 - x1, y2 - y1) * scale))
    side = min(max(side, 32), max(width, height))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    left = round(cx - side / 2)
    top = round(cy - side / 2)
    left = max(0, min(left, width - min(side, width)))
    top = max(0, min(top, height - min(side, height)))
    right = min(width, left + side)
    bottom = min(height, top + side)
    return left, top, right, bottom


def _split_by_content(
    records: list[dict[str, Any]], val_ratio: float, seed: int
) -> dict[str, str]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        meta = record["meta"]
        groups[str(meta["image_sha256"])].add(str(meta["image_id"]))
    ordered = sorted(
        groups,
        key=lambda digest: hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest(),
    )
    image_ids = {
        str(record["meta"]["image_id"])
        for record in records
    }
    target = min(max(1, round(len(image_ids) * val_ratio)), len(image_ids) - 1)
    val_groups: set[str] = set()
    val_records = 0
    for digest in ordered:
        if val_records + len(groups[digest]) <= target:
            val_groups.add(digest)
            val_records += len(groups[digest])
    if not val_groups:
        val_groups.add(ordered[0])
    if len(val_groups) == len(ordered):
        val_groups.remove(ordered[-1])
    return {
        image_id: ("val" if digest in val_groups else "train")
        for digest, image_ids in groups.items()
        for image_id in image_ids
    }


def prepare(
    *,
    input_path: Path,
    source_base_url: str,
    output_dir: Path,
    no_verify_tls: bool,
    timeout: float,
    val_ratio: float,
    seed: int,
    image_size: int,
    label_field: str = "training_label",
    label_source: str = "model_pseudo",
    require_human: bool = False,
) -> dict[str, Any]:
    rows = [
        row
        for row in _read_jsonl(input_path)
        if str(row.get(label_field)) in {"true", "false"}
        and (not require_human or bool(row.get("labeled_by")))
    ]
    if not rows:
        raise ValueError("no true/false rows found")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_images"
    media_dir = output_dir / "media"
    source_dir.mkdir(exist_ok=True)
    media_dir.mkdir(exist_ok=True)

    by_object: dict[str, dict[str, Any]] = {}
    for row in rows:
        object_key = str(row.get("image_object_key") or row.get("file_path") or "")
        image_id = str(row.get("image_id") or row.get("sample_id") or "")
        if not object_key or not image_id:
            raise ValueError("every row needs image_id and image_object_key")
        by_object.setdefault(object_key, {"image_id": image_id, "object_key": object_key})

    client = httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        trust_env=False,
        verify=not no_verify_tls,
        headers={"User-Agent": "ktd-qwen3vl-qlora-smoke/1.0"},
    )
    source_images: dict[str, tuple[Image.Image, str, str]] = {}
    try:
        for object_key, info in by_object.items():
            image_id = info["image_id"]
            destination = source_dir / f"{image_id}.jpg"
            image, digest = _download_image(
                client, _source_url(source_base_url, object_key), destination
            )
            source_images[object_key] = (image, digest, str(destination))
    finally:
        client.close()

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        object_key = str(row.get("image_object_key") or row.get("file_path"))
        image, digest, source_path = source_images[object_key]
        image_id = str(row.get("image_id") or row.get("sample_id") or "")
        candidate_index = row.get("candidate_index", row.get("target_index", row_index))
        width, height = image.size
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"invalid bbox at row {row_index}")
        x1, y1, x2, y2 = (
            max(0, min(width, round(float(bbox[0]) * width))),
            max(0, min(height, round(float(bbox[1]) * height))),
            max(0, min(width, round(float(bbox[2]) * width))),
            max(0, min(height, round(float(bbox[3]) * height))),
        )
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"empty bbox at row {row_index}")
        target = (x1, y1, x2, y2)
        context_box = _square_context(image.size, target, 8.0)
        roi_box = _square_context(image.size, target, 3.5)
        stem = f"{image_id}-{int(candidate_index)}"
        context_path = media_dir / f"{stem}-context.jpg"
        roi_path = media_dir / f"{stem}-roi.jpg"
        _render_crop(image, context_box, target, image_size).save(context_path, quality=92)
        _render_crop(image, roi_box, target, image_size).save(roi_path, quality=92)

        label = str(row[label_field])
        decision = "CONFIRMED" if label == "true" else "REJECTED"
        prompt = (
            "<image>\n<image>\n"
            "第一张图是候选框上下文，第二张图是候选区域放大图。"
            f"候选类别：{row['class_name']}。"
            "请判断候选区域内是否存在且符合该类别的目标。"
            '只输出 JSON：{"decision":"CONFIRMED"} 或 '
            '{"decision":"REJECTED"}。'
        )
        records.append(
            {
                "id": str(row.get("id") or row.get("candidate_id") or f"row-{row_index}"),
                "image": [
                    str(context_path.relative_to(output_dir)),
                    str(roi_path.relative_to(output_dir)),
                ],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": json.dumps({"decision": decision})},
                ],
                "meta": {
                    "label_source": label_source,
                    "label_field": label_field,
                    "labeled_by": row.get("labeled_by"),
                    "source_run_id": row.get("run_id"),
                    "image_id": image_id,
                    "image_object_key": object_key,
                    "image_sha256": digest,
                    "candidate_index": candidate_index,
                    "class_name": row.get("class_name"),
                    "bbox": bbox,
                    "source_image": source_path,
                    "source_file": str(input_path),
                },
            }
        )

    split_by_image = _split_by_content(records, val_ratio, seed)
    train_rows = [
        row for row in records if split_by_image[str(row["meta"]["image_id"])] == "train"
    ]
    val_rows = [
        row for row in records if split_by_image[str(row["meta"]["image_id"])] == "val"
    ]
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    _write_jsonl(output_dir / "all.jsonl", records)
    summary = {
        "schema_version": 1,
        "task": "candidate_roi_binary_verifier",
        "label_source": label_source,
        "label_field": label_field,
        "human_only": require_human,
        "source_file": str(input_path),
        "rows": len(records),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "images": len(source_images),
        "train_images": len({row["meta"]["image_id"] for row in train_rows}),
        "val_images": len({row["meta"]["image_id"] for row in val_rows}),
        "labels": Counter(
            row["conversations"][1]["value"] for row in records
        ),
        "split_policy": "content_sha256_grouped",
        "seed": seed,
        "val_ratio": val_ratio,
        "image_size": image_size,
        "uncertain_excluded": True,
    }
    summary["labels"] = dict(summary["labels"])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/recheck-runs/ktd-uncertain-recheck-20260903/qwen_vl_training.jsonl"),
    )
    parser.add_argument(
        "--source-base-url",
        default="https://lxtest.zhuoyilingxi.com:8088/minio-src",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--label-field", default="training_label")
    parser.add_argument("--label-source", default="model_pseudo")
    parser.add_argument(
        "--require-human",
        action="store_true",
        help="only include rows with a non-empty labeled_by provenance field",
    )
    args = parser.parse_args()
    summary = prepare(
        input_path=args.input,
        source_base_url=args.source_base_url,
        output_dir=args.output_dir,
        no_verify_tls=args.no_verify_tls,
        timeout=args.timeout,
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_size=args.image_size,
        label_field=args.label_field,
        label_source=args.label_source,
        require_human=args.require_human,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
