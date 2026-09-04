"""Crop-image training dataset builder (contract section 12).

Turns a prompt-gold snapshot into a crop-level training dataset: every canonical
box is cut from its source image with context padding, small crops are upscaled,
and labels are written in strictly separated files — human gold and GPT teacher
labels are never mixed into one row or one file.

Split discipline (extends contract section 5, image-level, zero intersection):
    sha256("prompt-gold-split-v1|" + image_sha256) mod 5
        == 0  -> locked_test   -> EXCLUDED from this dataset entirely (untouched)
        == 4  -> val           (human gold only; teacher labels never enter val)
        else  -> train         (human gold and/or teacher labels, separate files)

Determinism: same rules as the snapshot — rows sorted by canonical_id, canonical
JSON, no wall-clock timestamps; dataset_id is content-addressed
("cd1-" + sha256 of sorted file hashes, first 12 hex). Crops are JPEG
(quality=95, 4:4:4) which is byte-deterministic for a fixed Pillow build; the
Pillow version is recorded in the manifest.

Requires Pillow (isolated venv `.venv-crop`); everything else is stdlib.

Usage:
    .venv-crop/bin/python -m tools.prompt_eval_data.crop_dataset \
        --snapshot data/prompt-eval/snapshots/pg1-bc8c818184d8 \
        --output-root data/prompt-eval/crop-datasets
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import CLASS_MAP
from .common import jdumps, jsonl_bytes, sha256_bytes, sha256_text
from .imaging import DEFAULT_IMAGE_BASE_URL, fetch_bytes

DATASET_SCHEMA_VERSION = "prompt-gold-crop-dataset/1"

TRAIN = "train"
VAL = "val"
EXCLUDED = "excluded_locked"

#: Split salt identical to tools/prompt_eval_data/splits.py (contract section 5).
SPLIT_SALT = "prompt-gold-split-v1"

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0  # 4:4:4 — keep small text / plate edges intact


def assign_train_val(image_sha256: str) -> str:
    """Image-level split: mod5==0 -> excluded_locked, ==4 -> val, else train."""
    digest = int(sha256_text(f"{SPLIT_SALT}|{image_sha256}"), 16)
    mod = digest % 5
    if mod == 0:
        return EXCLUDED
    return VAL if mod == 4 else TRAIN


def padded_crop_box(
    bbox: list[int], width: int, height: int, pad_ratio: float
) -> tuple[int, int, int, int]:
    """Expand [x1,y1,x2,y2] by pad_ratio of its own size per side, clamp to image."""
    x1, y1, x2, y2 = bbox
    pad_x = int(round((x2 - x1) * pad_ratio))
    pad_y = int(round((y2 - y1) * pad_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def render_crop(image_bytes: bytes, crop_box: tuple[int, int, int, int], min_long_edge: int) -> bytes:
    """Crop and (if smaller than min_long_edge) upscale; return JPEG bytes."""
    from PIL import Image  # Pillow is a hard dependency of this module only

    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB")
        crop = rgb.crop(crop_box)
        w, h = crop.size
        long_edge = max(w, h)
        if long_edge < min_long_edge and long_edge > 0:
            scale = min_long_edge / long_edge
            crop = crop.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
        return buf.getvalue()


def teacher_label_of(record: dict) -> tuple[str, dict] | None:
    """Latest GPT pass as a teacher label; None when unusable."""
    info = record.get("prior_gpt") or {}
    passes = info.get("passes") or []
    if not passes:
        return None
    latest = passes[-1]
    label = latest.get("label")
    if latest.get("status") != "succeeded" or label not in ("true", "false", "uncertain"):
        return None
    return label, {"model": latest.get("model"), "run_id": latest.get("run_id")}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_crop_dataset(
    snapshot_dir: str,
    output_root: str,
    *,
    image_root: str | None = None,
    image_base_url: str = DEFAULT_IMAGE_BASE_URL,
    pad_ratio: float = 0.25,
    min_long_edge: int = 224,
    workers: int = 8,
) -> dict:
    """Build the dataset; returns the manifest. Idempotent (content-addressed)."""
    snapshot = Path(snapshot_dir)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    inference = _load_jsonl(snapshot / "inference_records.jsonl")
    gold_rows = _load_jsonl(snapshot / "gold_records.jsonl")
    gold_by_cid = {row["canonical_id"]: row for row in gold_rows}

    counts = {
        "canonical_boxes": len(inference),
        "images_total": 0,
        "images_excluded_locked": 0,
        "images_train": 0,
        "images_val": 0,
        "boxes_excluded_locked": 0,
        "boxes_no_label": 0,
        "crops_written": 0,
        "human_train": 0,
        "human_val": 0,
        "teacher_train": 0,
        "teacher_skipped_val_image": 0,
        "teacher_label_true": 0,
        "teacher_label_false": 0,
        "teacher_label_uncertain": 0,
        "image_fetch_failures": 0,
    }

    # --- plan: which boxes enter the dataset -----------------------------------
    planned: list[dict] = []
    images: dict[str, dict] = {}
    for rec in inference:
        sha = rec["image_sha256"]
        split = assign_train_val(sha)
        images.setdefault(
            sha,
            {
                "split": split,
                "object_key": rec["image_object_key"],
                "width": rec["image_width"],
                "height": rec["image_height"],
            },
        )
        if split == EXCLUDED:
            counts["boxes_excluded_locked"] += 1
            continue
        teacher = teacher_label_of(rec)
        has_human = rec["canonical_id"] in gold_by_cid
        if teacher is None and not has_human:
            counts["boxes_no_label"] += 1
            continue
        planned.append({"record": rec, "split": split, "teacher": teacher, "has_human": has_human})

    for info in images.values():
        counts["images_total"] += 1
        key = {"excluded_locked": "images_excluded_locked", "train": "images_train", "val": "images_val"}[info["split"]]
        counts[key] += 1

    # --- fetch images once each, verify against the snapshot sha256 -------------
    def _fetch(sha: str) -> tuple[str, bytes | None]:
        info = images[sha]
        try:
            data = fetch_bytes(info["object_key"], image_root=image_root, base_url=image_base_url)
        except RuntimeError:
            return sha, None
        if sha256_bytes(data) != sha:
            return sha, None
        return sha, data

    needed = [sha for sha, info in images.items() if info["split"] != EXCLUDED]
    image_bytes: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sha, data in pool.map(_fetch, sorted(needed)):
            if data is None:
                counts["image_fetch_failures"] += 1
            else:
                image_bytes[sha] = data

    # --- render crops and assemble label rows -----------------------------------
    tmp_dir = Path(tempfile.mkdtemp(prefix="cd1-build-", dir=output_root))
    files: dict[str, bytes] = {}
    human_rows: list[dict] = []
    teacher_rows: list[dict] = []

    for item in sorted(planned, key=lambda it: it["record"]["canonical_id"]):
        rec = item["record"]
        cid = rec["canonical_id"]
        sha = rec["image_sha256"]
        data = image_bytes.get(sha)
        if data is None:
            continue
        box = padded_crop_box(rec["bbox_px"], rec["image_width"], rec["image_height"], pad_ratio)
        crop_rel = f"crops/{item['split']}/{cid}.jpg"
        files[crop_rel] = render_crop(data, box, min_long_edge)
        counts["crops_written"] += 1

        base = {
            "canonical_id": cid,
            "class_id": rec["class_id"],
            "class_name": rec["class_name"],
            "crop_path": crop_rel,
            "image_sha256": sha,
            "bbox_px": rec["bbox_px"],
            "crop_box_px": list(box),
            "detector_confidence": rec["detector_confidence"],
        }
        if item["has_human"]:
            gold = gold_by_cid[cid]
            human_rows.append({**base, "label": gold["label"], "split": item["split"], "has_teacher_gpt": item["teacher"] is not None})
            counts["human_train" if item["split"] == TRAIN else "human_val"] += 1
        if item["teacher"] is not None:
            # Teacher labels on val images are dropped: training on a val image
            # (even with teacher labels) would leak validation images into train.
            if item["split"] != TRAIN:
                counts["teacher_skipped_val_image"] += 1
            else:
                label, meta = item["teacher"]
                teacher_rows.append(
                    {
                        **base,
                        "teacher_label": label,
                        "teacher_model": meta["model"],
                        "teacher_run_id": meta["run_id"],
                        "split": TRAIN,
                        "has_human_gold": item["has_human"],
                    }
                )
                counts["teacher_train"] += 1
                counts[f"teacher_label_{label}"] += 1

    human_rows.sort(key=lambda r: r["canonical_id"])
    teacher_rows.sort(key=lambda r: r["canonical_id"])
    files["labels_human.jsonl"] = jsonl_bytes(human_rows)
    files["labels_teacher_gpt.jsonl"] = jsonl_bytes(teacher_rows)

    # --- manifest + SHA256SUMS, then content-addressed rename -------------------
    file_entries = []
    for rel in sorted(files):
        file_entries.append({"path": rel, "bytes": len(files[rel]), "sha256": sha256_bytes(files[rel])})
    sums_text = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in file_entries)

    import PIL

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_snapshot": json.loads((snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))["snapshot_id"],
        "class_map": {str(k): v for k, v in CLASS_MAP},
        "params": {
            "pad_ratio": pad_ratio,
            "min_long_edge": min_long_edge,
            "crop_format": f"jpeg q={JPEG_QUALITY} subsampling={JPEG_SUBSAMPLING}",
            "pillow_version": PIL.__version__,
        },
        "split_rule": "sha256('prompt-gold-split-v1|' + image_sha256) mod 5: 0 -> excluded_locked (untouched), 4 -> val (human only), else train",
        "label_files": {
            "labels_human.jsonl": "人工金标 true/false，train+val；与教师标签永不混行混文件",
            "labels_teacher_gpt.jsonl": "GPT 教师 true/false/uncertain，仅 train；不进入 val",
        },
        "counts": counts,
        "files": file_entries,
    }
    sums_bytes = sums_text.encode("utf-8")
    manifest_bytes = (jdumps(manifest) + "\n").encode("utf-8")
    dataset_id = "cd1-" + sha256_text(
        "".join(entry["sha256"] for entry in file_entries) + sha256_bytes(manifest_bytes)
    )[:12]
    manifest["dataset_id"] = dataset_id
    manifest_bytes = (jdumps(manifest) + "\n").encode("utf-8")

    for rel, data in files.items():
        target = tmp_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (tmp_dir / "SHA256SUMS").write_bytes(sums_bytes)
    (tmp_dir / "dataset_manifest.json").write_bytes(manifest_bytes)

    final_dir = Path(output_root) / dataset_id
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        shutil.rmtree(tmp_dir)
    else:
        tmp_dir.rename(final_dir)
    manifest["dataset_dir"] = str(final_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a crop-level training dataset from a prompt-gold snapshot.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-root", default="data/prompt-eval/crop-datasets")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    parser.add_argument("--pad-ratio", type=float, default=0.25)
    parser.add_argument("--min-long-edge", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    manifest = build_crop_dataset(
        args.snapshot,
        args.output_root,
        image_root=args.image_root,
        image_base_url=args.image_base_url,
        pad_ratio=args.pad_ratio,
        min_long_edge=args.min_long_edge,
        workers=args.workers,
    )
    sys.stdout.write(jdumps({"dataset_id": manifest["dataset_id"], "counts": manifest["counts"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
