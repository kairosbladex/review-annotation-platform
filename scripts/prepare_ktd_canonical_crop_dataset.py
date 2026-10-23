#!/usr/bin/env python3
"""Export an auditable canonical crop dataset for a candidate-box verifier.

The prompt-gold snapshot is the source of truth for canonical boxes and the
selection/locked-test image split. GPT labels remain teacher metadata; only
binary teacher rows from non-evaluation images enter train/val manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
from PIL import Image

try:
    from scripts.prepare_ktd_clean_source_qwen_dataset import (
        CATEGORY_ADAPTIVE_SPECS,
        NEUTRAL_GRAY_RGB,
        VALID_BBOX_STYLES,
        VALID_VIEW_MODES,
        draw_bbox_overlay,
        get_category_crop_spec,
        get_processor_pixel_budget,
    )
except ImportError:
    from prepare_ktd_clean_source_qwen_dataset import (  # type: ignore[no-redef]
        CATEGORY_ADAPTIVE_SPECS,
        NEUTRAL_GRAY_RGB,
        VALID_BBOX_STYLES,
        VALID_VIEW_MODES,
        draw_bbox_overlay,
        get_category_crop_spec,
        get_processor_pixel_budget,
    )


VALID_LABELS = {"true", "false", "uncertain"}
OUTPUT_FILES = (
    "manifest.jsonl",
    "teacher_all.jsonl",
    "teacher_train.jsonl",
    "teacher_val.jsonl",
    "human_gold_selection.jsonl",
    "selection_inputs.jsonl",
    "selection_gold.jsonl",
    "locked_test_inputs.jsonl",
    "summary.json",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _latest_teacher_label(record: dict[str, Any]) -> str | None:
    prior = record.get("prior_gpt") or {}
    passes = prior.get("passes") if isinstance(prior, dict) else None
    if not isinstance(passes, list):
        return None
    labels = []
    for item in passes:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        status = str(item.get("status") or "succeeded").strip().lower()
        if label in VALID_LABELS and status == "succeeded":
            labels.append(label)
    return labels[-1] if labels else None


def _split_non_eval(shas: set[str], *, val_ratio: float, seed: int) -> set[str]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if len(shas) < 2 or val_ratio == 0:
        return set()
    target = min(max(1, round(len(shas) * val_ratio)), len(shas) - 1)
    ordered = sorted(
        shas,
        key=lambda digest: hashlib.sha256(
            f"canonical-crop-split-v1:{seed}:{digest}".encode("utf-8")
        ).hexdigest(),
    )
    return set(ordered[:target])


def _square_context(
    image_size: tuple[int, int],
    target: tuple[int, int, int, int],
    scale: float,
    *,
    class_name: str | None = None,
    is_context: bool = False,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = target
    cat_spec = get_category_crop_spec(class_name) if class_name is not None else None
    effective_scale = float(scale)
    if is_context and cat_spec and cat_spec.get("clamp_context_to_image"):
        effective_scale = max(effective_scale, float(cat_spec.get("min_context_scale", 3.0)))
    side = max(2, round(max(x2 - x1, y2 - y1) * effective_scale))
    side = min(max(side, 32), max(width, height))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    left = round(cx - side / 2)
    top = round(cy - side / 2)
    left = max(0, min(left, width - min(side, width)))
    top = max(0, min(top, height - min(side, height)))
    return left, top, min(width, left + side), min(height, top + side)


def _render_crop(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    size: int,
    *,
    target: tuple[int, int, int, int] | None = None,
    bbox_style: str = "none",
    class_name: str | None = None,
    is_clean_roi: bool = False,
) -> Image.Image:
    left, top, right, bottom = crop_box
    crop = image.crop((left, top, right, bottom))
    scale = min(size / crop.width, size / crop.height)
    cat_spec = get_category_crop_spec(class_name) if class_name is not None else None
    resample = (
        Image.Resampling.BICUBIC
        if is_clean_roi and cat_spec and cat_spec.get("clean_roi_upscale") == "cv2.INTER_CUBIC" and scale > 1.0
        else Image.Resampling.LANCZOS
    )
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        resample,
    )
    canvas = Image.new("RGB", (size, size), NEUTRAL_GRAY_RGB)
    offset_x = (size - resized.width) // 2
    offset_y = (size - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    if target is not None and bbox_style != "none":
        tx1, ty1, tx2, ty2 = target
        shifted_bbox = (
            offset_x + (tx1 - left) * scale,
            offset_y + (ty1 - top) * scale,
            offset_x + (tx2 - left) * scale,
            offset_y + (ty2 - top) * scale,
        )
        drawn = draw_bbox_overlay(canvas, shifted_bbox, style=bbox_style)
        return Image.fromarray(drawn)
    return canvas


def _image_url(base_url: str, object_key: str) -> str:
    return f"{base_url.rstrip('/')}/{quote(object_key.lstrip('/'), safe='/')}"


def _load_image_bytes(
    object_key: str,
    *,
    image_root: Path | None,
    image_base_url: str,
    timeout: float,
    no_verify_tls: bool,
) -> bytes:
    if image_root is not None:
        path = image_root / object_key
        if not path.is_file():
            raise FileNotFoundError(f"missing local image: {path}")
        return path.read_bytes()
    response = httpx.get(
        _image_url(image_base_url, object_key),
        timeout=timeout,
        follow_redirects=True,
        verify=not no_verify_tls,
        trust_env=False,
        headers={"User-Agent": "ktd-canonical-crop-export/1.0"},
    )
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"empty image response: {object_key}")
    return response.content


def _load_images(
    records: list[dict[str, Any]],
    *,
    image_root: Path | None,
    image_base_url: str,
    timeout: float,
    no_verify_tls: bool,
    workers: int,
) -> tuple[dict[str, tuple[Image.Image, bytes]], list[dict[str, str]]]:
    keys = sorted({str(row["image_object_key"]) for row in records})

    def load(key: str) -> tuple[str, tuple[Image.Image, bytes] | None, str | None]:
        try:
            payload = _load_image_bytes(
                key,
                image_root=image_root,
                image_base_url=image_base_url,
                timeout=timeout,
                no_verify_tls=no_verify_tls,
            )
            with Image.open(io.BytesIO(payload)) as image:
                return key, (image.convert("RGB"), payload), None
        except Exception as exc:  # retain every failure for the audit
            return key, None, str(exc)

    images: dict[str, tuple[Image.Image, bytes]] = {}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for key, image, error in pool.map(load, keys):
            if image is not None:
                images[key] = image
            else:
                errors.append({"image_object_key": key, "error": error or "unknown error"})
    return images, errors


def _input_view(row: dict[str, Any], *, split: str, context_path: str, roi_path: str) -> dict[str, Any]:
    return {
        "canonical_id": row["canonical_id"],
        "class_id": row["class_id"],
        "class_name": row["class_name"],
        "bbox_norm": row["bbox_norm"],
        "bbox_px": row["bbox_px"],
        "image_sha256": row["image_sha256"],
        "image_object_key": row["image_object_key"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "detector_confidence": row.get("detector_confidence"),
        "split": split,
        "context_image": context_path,
        "roi_image": roi_path,
    }


def build_dataset(
    *,
    snapshot_dir: Path,
    output_dir: Path,
    image_root: Path | None = None,
    image_base_url: str = "https://lxtest.zhuoyilingxi.com:8088/minio-src",
    timeout: float = 60.0,
    no_verify_tls: bool = False,
    workers: int = 8,
    image_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    bbox_style: str = "corner_ticks",
    view_mode: str = "2view",
) -> dict[str, Any]:
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if bbox_style not in VALID_BBOX_STYLES:
        raise ValueError(f"unsupported bbox_style: {bbox_style!r}")
    if view_mode not in VALID_VIEW_MODES:
        raise ValueError(f"unsupported view_mode: {view_mode!r}")
    inference = _read_jsonl(snapshot_dir / "inference_records.jsonl")
    gold = _read_jsonl(snapshot_dir / "gold_records.jsonl")
    selection_inputs = _read_jsonl(snapshot_dir / "selection_inputs.jsonl")
    locked_inputs = _read_jsonl(snapshot_dir / "locked_test_inputs.jsonl")

    selection_sha = {str(row.get("image_sha256") or "") for row in selection_inputs}
    locked_sha = {str(row.get("image_sha256") or "") for row in locked_inputs}
    selection_sha.discard("")
    locked_sha.discard("")
    overlap = sorted(selection_sha & locked_sha)
    if overlap:
        raise ValueError(f"selection and locked_test image SHA overlap: {overlap[:3]}")
    if not inference:
        raise ValueError("inference_records.jsonl is empty")

    by_id: dict[str, dict[str, Any]] = {}
    for row in inference:
        canonical_id = str(row.get("canonical_id") or "")
        if not canonical_id or canonical_id in by_id:
            raise ValueError(f"duplicate or missing canonical_id: {canonical_id!r}")
        for key in ("image_sha256", "image_object_key", "bbox_norm", "bbox_px"):
            if not row.get(key):
                raise ValueError(f"missing {key} for {canonical_id}")
        by_id[canonical_id] = row

    gold_by_id: dict[str, dict[str, Any]] = {}
    for row in gold:
        canonical_id = str(row.get("canonical_id") or "")
        if not canonical_id or canonical_id in gold_by_id:
            raise ValueError(f"duplicate or missing gold canonical_id: {canonical_id!r}")
        if canonical_id not in by_id:
            raise ValueError(f"gold canonical_id missing from inference records: {canonical_id}")
        gold_by_id[canonical_id] = row

    non_eval_shas = {
        str(row["image_sha256"])
        for row in inference
        if str(row["image_sha256"]) not in selection_sha | locked_sha
    }
    val_shas = _split_non_eval(non_eval_shas, val_ratio=val_ratio, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    source_dir = output_dir / "source_images"
    image_errors_path = output_dir / "image_errors.jsonl"
    media_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    images, errors = _load_images(
        inference,
        image_root=image_root,
        image_base_url=image_base_url,
        timeout=timeout,
        no_verify_tls=no_verify_tls,
        workers=workers,
    )
    if errors:
        image_errors_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in errors),
            encoding="utf-8",
        )
        raise RuntimeError(f"failed to load {len(errors)} of {len({row['image_object_key'] for row in inference})} images")
    image_errors_path.unlink(missing_ok=True)

    image_sha_by_key: dict[str, str] = {}
    for object_key, (image, payload) in images.items():
        digest = hashlib.sha256(payload).hexdigest()
        image_sha_by_key[object_key] = digest
        expected = {str(row["image_sha256"]) for row in inference if row["image_object_key"] == object_key}
        if expected != {digest}:
            raise ValueError(f"image SHA mismatch for {object_key}: expected {sorted(expected)}, got {digest}")
        image.save(source_dir / f"{digest}.jpg", format="JPEG", quality=95)

    manifest_rows: list[dict[str, Any]] = []
    teacher_all: list[dict[str, Any]] = []
    teacher_train: list[dict[str, Any]] = []
    teacher_val: list[dict[str, Any]] = []
    human_gold_selection: list[dict[str, Any]] = []
    selection_view: list[dict[str, Any]] = []
    selection_gold: list[dict[str, Any]] = []
    locked_view: list[dict[str, Any]] = []

    for row in sorted(inference, key=lambda item: str(item["canonical_id"])):
        canonical_id = str(row["canonical_id"])
        digest = str(row["image_sha256"])
        if digest in locked_sha:
            split = "locked_test"
        elif digest in selection_sha:
            split = "selection"
        elif digest in val_shas:
            split = "val"
        else:
            split = "train"
        image, _ = images[str(row["image_object_key"])]
        width, height = image.size
        x1, y1, x2, y2 = (int(value) for value in row["bbox_px"])
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"bbox_px outside image for {canonical_id}")
        target = (x1, y1, x2, y2)
        class_name_str = str(row.get("class_name") or "")
        cat_spec = get_category_crop_spec(class_name_str)
        context_scale = float(cat_spec["context_scale"])
        roi_scale = float(cat_spec["roi_scale"])
        context_path = media_dir / f"{canonical_id}-context.jpg"
        roi_path = media_dir / f"{canonical_id}-roi.jpg"
        _render_crop(
            image,
            _square_context(image.size, target, context_scale, class_name=class_name_str, is_context=True),
            image_size,
            target=target,
            bbox_style=bbox_style,
            class_name=class_name_str,
            is_clean_roi=False,
        ).save(context_path, quality=95)
        _render_crop(
            image,
            _square_context(image.size, target, roi_scale, class_name=class_name_str, is_context=False),
            image_size,
            target=target,
            bbox_style="none",
            class_name=class_name_str,
            is_clean_roi=True,
        ).save(roi_path, quality=95)
        context_rel = str(context_path.relative_to(output_dir))
        roi_rel = str(roi_path.relative_to(output_dir))
        teacher_label = _latest_teacher_label(row)
        gold_row = gold_by_id.get(canonical_id)
        source = "human_gold" if gold_row is not None else "gpt_teacher"
        input_row = _input_view(row, split=split, context_path=context_rel, roi_path=roi_rel)
        manifest_row = {
            **input_row,
            "label_source": source,
            "human_label": gold_row.get("label") if gold_row is not None and split == "selection" else None,
            "teacher_label": teacher_label,
            "eligible_for_training": (
                split in {"train", "val"}
                and source == "gpt_teacher"
                and teacher_label in {"true", "false"}
            ),
        }
        manifest_rows.append(manifest_row)

        teacher_row = {**input_row, "label_source": "gpt_teacher", "teacher_label": teacher_label}
        teacher_all.append(teacher_row)
        if manifest_row["eligible_for_training"]:
            (teacher_train if split == "train" else teacher_val).append(teacher_row)

        if gold_row is not None and split == "selection":
            human_gold_selection.append({**input_row, "label_source": "human_gold", "human_label": gold_row["label"]})
            selection_view.append(input_row)
            selection_gold.append({
                "canonical_id": canonical_id,
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "label": gold_row["label"],
            })
        elif gold_row is not None and split == "locked_test":
            locked_view.append(input_row)

    train_sha = {row["image_sha256"] for row in teacher_train}
    val_sha = {row["image_sha256"] for row in teacher_val}
    eval_sha = selection_sha | locked_sha
    leakage = {
        "train_vs_eval_sha": sorted(train_sha & eval_sha),
        "val_vs_eval_sha": sorted(val_sha & eval_sha),
        "train_vs_val_sha": sorted(train_sha & val_sha),
        "selection_vs_locked_sha": overlap,
    }
    if any(leakage.values()):
        raise AssertionError(f"crop dataset leakage detected: {leakage}")

    _write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    _write_jsonl(output_dir / "teacher_all.jsonl", teacher_all)
    _write_jsonl(output_dir / "teacher_train.jsonl", teacher_train)
    _write_jsonl(output_dir / "teacher_val.jsonl", teacher_val)
    _write_jsonl(output_dir / "human_gold_selection.jsonl", human_gold_selection)
    _write_jsonl(output_dir / "selection_inputs.jsonl", selection_view)
    _write_jsonl(output_dir / "selection_gold.jsonl", selection_gold)
    _write_jsonl(output_dir / "locked_test_inputs.jsonl", locked_view)
    summary = {
        "schema_version": 1,
        "task": "canonical_candidate_crop_verifier",
        "snapshot_dir": str(snapshot_dir.resolve()),
        "snapshot_id": json.loads((snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")).get("snapshot_id"),
        "canonical_rows": len(inference),
        "human_gold_rows": len(gold),
        "selection_gold_rows": len(selection_gold),
        "locked_gold_rows": len(gold) - len(selection_gold),
        "teacher_rows": len(teacher_all),
        "teacher_labels": dict(Counter(str(row["teacher_label"]) for row in teacher_all)),
        "teacher_train_rows": len(teacher_train),
        "teacher_val_rows": len(teacher_val),
        "teacher_train_images": len(train_sha),
        "teacher_val_images": len(val_sha),
        "selection_images": len(selection_sha),
        "locked_test_images": len(locked_sha),
        "source_images": len(images),
        "crop_size": image_size,
        "view_mode": view_mode,
        "bbox_style": bbox_style,
        "processor": get_processor_pixel_budget(view_mode),
        "crop_views": {
            "context_scale": 8.0,
            "roi_scale": 3.5,
            "overlay": bbox_style != "none",
            "bbox_style": bbox_style,
            "category_adaptive": True,
        },
        "split_policy": "selection/locked SHA from pg1; non-eval SHA grouped deterministic train/val",
        "seed": seed,
        "val_ratio": val_ratio,
        "uncertain_teacher_excluded_from_training": sum(row["teacher_label"] == "uncertain" for row in teacher_all),
        "locked_test_labels_written": sum(
            row["human_label"] is not None
            for row in manifest_rows
            if row["split"] == "locked_test"
        ),
        "leakage": leakage,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None, help="local root keyed by image_object_key")
    parser.add_argument("--image-base-url", default="https://lxtest.zhuoyilingxi.com:8088/minio-src")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bbox-style", choices=VALID_BBOX_STYLES, default="corner_ticks")
    parser.add_argument("--view-mode", choices=VALID_VIEW_MODES, default="2view")
    args = parser.parse_args()
    summary = build_dataset(
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        image_root=args.image_root,
        image_base_url=args.image_base_url,
        timeout=args.timeout,
        no_verify_tls=args.no_verify_tls,
        workers=args.workers,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        bbox_style=args.bbox_style,
        view_mode=args.view_mode,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
