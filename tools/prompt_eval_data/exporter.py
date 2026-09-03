"""Snapshot exporter: DB (read-only) + images -> versioned prompt-gold snapshot.

Usage:
    python3 -m tools.prompt_eval_data.exporter --db /path/ktd-local.sqlite3 \
        [--output-root data/prompt-eval/snapshots] \
        [--image-root DIR | --image-base-url URL] \
        [--prior-gpt PATH ...] [--prior-qwen PATH ...] [--workers N]

Same logical input always yields byte-identical output (contract section 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import CLASS_MAP, SCHEMA_VERSION
from .canon import build_canonical
from .common import json_bytes, jsonl_bytes, sha256_bytes
from .dbload import load_rows
from .imaging import DEFAULT_IMAGE_BASE_URL, resolve_images
from .priors import load_gpt_priors, load_qwen_priors
from .splits import assign_split

INPUT_ALLOWED_KEYS = [
    "canonical_id",
    "class_id",
    "class_name",
    "bbox_norm",
    "bbox_px",
    "image_sha256",
    "image_object_key",
    "image_width",
    "image_height",
    "detector_confidence",
]
SPLIT_GOLD_KEYS = ["canonical_id", "class_id", "class_name", "label"]


def _input_row(box: dict) -> dict:
    """Model-input view of a box: no label/note/reviewer/priors (contract section 6)."""
    return {key: box[key] for key in INPUT_ALLOWED_KEYS}


def _inference_row(box: dict) -> dict:
    return {
        "canonical_id": box["canonical_id"],
        "class_id": box["class_id"],
        "class_name": box["class_name"],
        "bbox_norm": box["bbox_norm"],
        "bbox_px": box["bbox_px"],
        "image_sha256": box["image_sha256"],
        "image_object_key": box["image_object_key"],
        "image_width": box["image_width"],
        "image_height": box["image_height"],
        "detector_confidence": box["detector_confidence"],
        "prior_yolo": box["prior_yolo"],
        "prior_gpt": box["prior_gpt"],
        "prior_qwen": box["prior_qwen"],
        "alias_count": len(box["aliases"]),
        "aliases": box["aliases"],
        "has_human_gold": box["verdict"] == "gold",
    }


def _gold_row(box: dict, split: str) -> dict:
    return {
        "canonical_id": box["canonical_id"],
        "class_id": box["class_id"],
        "class_name": box["class_name"],
        "label": box["label"],
        "note": box["note"],
        "reviewer": box["reviewer"],
        "reviewer_id": box["reviewer_id"],
        "revision": box["revision"],
        "labeled_at": box["labeled_at"],
        "source": box["source"],
        "aliases": box["aliases"],
        "prior_yolo": box["prior_yolo"],
        "prior_gpt": box["prior_gpt"],
        "prior_qwen": box["prior_qwen"],
        "image_sha256": box["image_sha256"],
        "image_object_key": box["image_object_key"],
        "bbox_norm": box["bbox_norm"],
        "bbox_px": box["bbox_px"],
        "split": split,
    }


def _quarantine_row(box: dict) -> dict:
    return {
        "canonical_id": box["canonical_id"],
        "class_id": box["class_id"],
        "class_name": box["class_name"],
        "quarantine_reasons": box["quarantine_reasons"],
        "suspect_hits": box["suspect_hits"],
        "evidence": box["evidence"],
        "image_sha256": box["image_sha256"],
        "image_object_key": box["image_object_key"],
        "bbox_norm": box["bbox_norm"],
        "bbox_px": box["bbox_px"],
    }


def build_snapshot(
    *,
    db_path: str,
    image_root: str | None = None,
    image_base_url: str = DEFAULT_IMAGE_BASE_URL,
    prior_gpt_paths: list[str] | None = None,
    prior_qwen_paths: list[str] | None = None,
    workers: int = 8,
) -> tuple[str, dict[str, bytes], dict]:
    """Return (snapshot_id, {relpath: bytes}, manifest). Pure and deterministic."""
    rows = load_rows(db_path)
    object_keys = sorted({row["object_key"] for row in rows if row["object_key"]})
    images, image_errors = resolve_images(
        object_keys, image_root=image_root, base_url=image_base_url, workers=workers
    )
    built = build_canonical(
        rows,
        images,
        image_errors,
        gpt_priors=load_gpt_priors(prior_gpt_paths or []),
        qwen_priors=load_qwen_priors(prior_qwen_paths or []),
    )
    boxes, authoring, counts = built["boxes"], built["authoring"], built["counts"]

    gold_rows, quarantine_rows, inference_rows = [], [], []
    selection_inputs, selection_gold, locked_inputs, locked_gold = [], [], [], []
    split_images = {"selection": set(), "locked_test": set()}

    for cid in sorted(boxes):
        box = boxes[cid]
        inference_rows.append(_inference_row(box))
        if box["verdict"] == "quarantine":
            quarantine_rows.append(_quarantine_row(box))
        elif box["verdict"] == "gold":
            split = assign_split(box["image_sha256"])
            split_images[split].add(box["image_sha256"])
            gold_rows.append(_gold_row(box, split))
            input_row = _input_row(box)
            gold_ref = {key: box[key] for key in SPLIT_GOLD_KEYS}
            if split == "selection":
                selection_inputs.append(input_row)
                selection_gold.append(gold_ref)
            else:
                locked_inputs.append(input_row)
                locked_gold.append(gold_ref)

    counts["unique_images"] = len(object_keys)
    counts["images_downloaded"] = len(images)
    counts["images_failed"] = len(image_errors)
    counts["selection_images"] = len(split_images["selection"])
    counts["selection_boxes"] = len(selection_gold)
    counts["locked_test_images"] = len(split_images["locked_test"])
    counts["locked_test_boxes"] = len(locked_gold)

    files: dict[str, bytes] = {
        "inference_records.jsonl": jsonl_bytes(inference_rows),
        "gold_records.jsonl": jsonl_bytes(gold_rows),
        "quarantine.jsonl": jsonl_bytes(quarantine_rows),
        "authoring.jsonl": jsonl_bytes(authoring),
        "selection_inputs.jsonl": jsonl_bytes(selection_inputs),
        "selection_gold.jsonl": jsonl_bytes(selection_gold),
        "locked_test_inputs.jsonl": jsonl_bytes(locked_inputs),
        "locked_test_gold.jsonl": jsonl_bytes(locked_gold),
    }
    if image_errors:
        files["image_errors.jsonl"] = jsonl_bytes(
            {"image_object_key": key, "error": image_errors[key]} for key in sorted(image_errors)
        )

    snapshot_id = "pg1-" + sha256_bytes(
        "".join(f"{path}:{sha256_bytes(data)}\n" for path, data in sorted(files.items())).encode("utf-8")
    )[:12]

    sums = "".join(
        f"{sha256_bytes(data)}  {path}\n" for path, data in sorted(files.items())
    ).encode("utf-8")
    files["SHA256SUMS"] = sums

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "class_map": {str(cid): name for cid, name in CLASS_MAP},
        "source": {
            "database": str(Path(db_path).resolve()),
            "image_base_url": None if image_root else image_base_url,
            "image_root": str(Path(image_root).resolve()) if image_root else None,
            "prior_gpt_paths": [str(Path(p).resolve()) for p in prior_gpt_paths or []],
            "prior_qwen_paths": [str(Path(p).resolve()) for p in prior_qwen_paths or []],
            "max_labeled_at": counts.pop("max_labeled_at"),
        },
        "counts": counts,
        "determinism": {
            "bbox_pixel_rule": "px = floor(norm * size + 0.5), clamp [0, size]",
            "canonical_id": "sha256(image_sha256|class_id|x1|y1|x2|y2) over integer pixel bbox",
            "json": "utf-8, sort_keys, compact separators, LF endings, rows sorted by canonical_id",
            "split_rule": "sha256('prompt-gold-split-v1|' + image_sha256) mod 5 == 0 -> locked_test",
            "wall_clock": "no wall-clock timestamps are written anywhere",
        },
        "files": [
            {"path": path, "sha256": sha256_bytes(data), "bytes": len(data)}
            for path, data in sorted(files.items())
        ],
    }
    files["snapshot_manifest.json"] = json_bytes(manifest)
    return snapshot_id, files, manifest


def write_snapshot(files: dict[str, bytes], output_root: str, snapshot_id: str) -> Path:
    target = Path(output_root) / snapshot_id
    target.mkdir(parents=True, exist_ok=True)
    for path, data in sorted(files.items()):
        (target / path).write_bytes(data)
    return target


def explain(counts: dict) -> str:
    """Human-readable funnel, contract section 7."""
    return "\n".join(
        [
            f"raw_candidates={counts['raw_candidates']} (DB candidates 全量)",
            f"  excluded: image_unavailable={counts['excluded_image_unavailable']}, "
            f"unknown_class={counts['excluded_unknown_class']}",
            f"canonical_boxes={counts['canonical_boxes']} (image_sha256+class+整数像素bbox 合并 aliases 后)",
            f"  ├─ unlabeled={counts['unlabeled_canonical']} (无人工标注)",
            f"  ├─ gold_clean={counts['gold_records']}",
            f"  └─ quarantine={counts['quarantine_records']} "
            f"(label_conflict={counts['quarantine_label_conflict']}, "
            f"note_label_suspect={counts['quarantine_note_label_suspect']})",
            f"authoring: human={counts['authoring_human']}, "
            f"auto_excluded={counts['authoring_auto_excluded']} (auto_annotation.applied 不进 gold)",
            f"images: unique={counts['unique_images']}, downloaded={counts['images_downloaded']}, "
            f"failed={counts['images_failed']}",
            f"priors: gpt_matched={counts['prior_gpt_matched']}, qwen_matched={counts['prior_qwen_matched']}",
            f"splits: selection={counts['selection_boxes']} boxes/{counts['selection_images']} images, "
            f"locked_test={counts['locked_test_boxes']} boxes/{counts['locked_test_images']} images "
            f"(image SHA 交集恒为 0)",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a versioned prompt-gold snapshot (read-only).")
    parser.add_argument("--db", required=True, help="SQLite DB path (opened read-only)")
    parser.add_argument("--output-root", default="data/prompt-eval/snapshots")
    parser.add_argument("--image-root", default=None, help="local directory keyed by object_key (tests)")
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    parser.add_argument("--prior-gpt", action="append", default=[], dest="prior_gpt")
    parser.add_argument("--prior-qwen", action="append", default=[], dest="prior_qwen")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    snapshot_id, files, manifest = build_snapshot(
        db_path=args.db,
        image_root=args.image_root,
        image_base_url=args.image_base_url,
        prior_gpt_paths=args.prior_gpt,
        prior_qwen_paths=args.prior_qwen,
        workers=args.workers,
    )
    target = write_snapshot(files, args.output_root, snapshot_id)
    print(explain(manifest["counts"]))
    print(f"snapshot_id={snapshot_id}")
    print(f"snapshot_dir={target.resolve()}")
    manifest_sha = sha256_bytes(files["snapshot_manifest.json"])
    print(f"manifest_sha256={manifest_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
