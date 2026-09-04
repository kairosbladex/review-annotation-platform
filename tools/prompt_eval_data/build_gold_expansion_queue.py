"""Build a deterministic human-gold expansion queue from a frozen snapshot.

The queue contains model inputs only. It deliberately omits all prior model
decisions so that the annotator is not primed by teacher or service output.
Locked-test images and already labeled or quarantined canonical boxes are
excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SIZE_BINS = ("lt_0.02", "0.02_to_0.05", "0.05_to_0.10", "gte_0.10")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _size_bin(row: dict[str, Any]) -> str:
    bbox = row["bbox_norm"]
    long_edge = max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
    if long_edge < 0.02:
        return SIZE_BINS[0]
    if long_edge < 0.05:
        return SIZE_BINS[1]
    if long_edge < 0.10:
        return SIZE_BINS[2]
    return SIZE_BINS[3]


def _order(row: dict[str, Any]) -> tuple[str, int, str, str]:
    seed = hashlib.sha256(f"gold-expansion-v1|{row['canonical_id']}".encode("utf-8")).hexdigest()
    return (row["class_name"], SIZE_BINS.index(row["size_bin"]), seed, row["canonical_id"])


def build_queue(snapshot: Path, *, max_per_class: int = 20, max_per_image: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inference = _jsonl(snapshot / "inference_records.jsonl")
    gold = _jsonl(snapshot / "gold_records.jsonl")
    quarantine = _jsonl(snapshot / "quarantine.jsonl")
    locked_inputs = _jsonl(snapshot / "locked_test_inputs.jsonl")
    excluded_ids = {row["canonical_id"] for row in gold} | {row["canonical_id"] for row in quarantine}
    locked_images = {row["image_sha256"] for row in locked_inputs}
    existing_gold_images = {row["image_sha256"] for row in gold} | {
        row["image_sha256"] for row in quarantine
    }
    excluded_images = locked_images | existing_gold_images

    eligible: list[dict[str, Any]] = []
    for row in inference:
        if row["canonical_id"] in excluded_ids or row["image_sha256"] in excluded_images:
            continue
        item = {
            "canonical_id": row["canonical_id"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "bbox_norm": row["bbox_norm"],
            "bbox_px": row["bbox_px"],
            "image_sha256": row["image_sha256"],
            "image_object_key": row["image_object_key"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
        }
        item["size_bin"] = _size_bin(item)
        eligible.append(item)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(eligible, key=_order):
        buckets[(row["class_name"], row["size_bin"])].append(row)

    selected: list[dict[str, Any]] = []
    image_counts: Counter[str] = Counter()
    classes = sorted({row["class_name"] for row in eligible})
    for class_name in classes:
        class_selected = 0
        while class_selected < max_per_class:
            added = False
            for size_bin in SIZE_BINS:
                bucket = buckets[(class_name, size_bin)]
                while bucket and image_counts[bucket[0]["image_sha256"]] >= max_per_image:
                    bucket.pop(0)
                if not bucket:
                    continue
                row = bucket.pop(0)
                image_counts[row["image_sha256"]] += 1
                selected.append({**row, "queue": "gold_expansion", "label": None})
                class_selected += 1
                added = True
                if class_selected >= max_per_class:
                    break
            if not added:
                break

    selected.sort(key=_order)
    summary = {
        "schema_version": "gold-expansion-queue/1",
        "snapshot_id": snapshot.name,
        "selection_policy": {
            "max_per_class": max_per_class,
            "max_per_image": max_per_image,
            "size_bins": list(SIZE_BINS),
            "ordering": "class, size_bin, sha256(gold-expansion-v1|canonical_id), canonical_id",
            "locked_test": "all locked-test image_sha256 excluded",
            "existing_gold_images": "all images with gold or quarantine excluded for image-level independence",
            "model_priors_in_queue": False,
        },
        "eligible_unlabeled_boxes": len(eligible),
        "eligible_by_class": dict(sorted(Counter(row["class_name"] for row in eligible).items())),
        "selected_boxes": len(selected),
        "selected_images": len({row["image_sha256"] for row in selected}),
        "selected_by_class": dict(sorted(Counter(row["class_name"] for row in selected).items())),
        "selected_by_size_bin": dict(sorted(Counter(row["size_bin"] for row in selected).items())),
        "existing_gold_boxes": len(gold),
        "quarantine_boxes": len(quarantine),
        "locked_test_images_excluded": len(locked_images),
        "existing_gold_images_excluded": len(existing_gold_images),
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--max-per-image", type=int, default=2)
    args = parser.parse_args()
    if args.max_per_class < 1 or args.max_per_image < 1:
        raise ValueError("queue limits must be positive")
    if not args.snapshot.is_dir():
        raise FileNotFoundError(args.snapshot)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty output: {args.output_dir}")
    rows, summary = build_queue(
        args.snapshot,
        max_per_class=args.max_per_class,
        max_per_image=args.max_per_image,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
