"""疑似标签错误复核：挑选、裁图和更正清单。

模型原始结论只进入 selection.json，供核对挑选规则。
复核页读取的 adjudication.csv 不含模型结论。本模块不修改任何金标文件。
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

RUN_IDS = ("p000", "p000b", "p001")
RULE_HUMAN_FALSE_GE2_TRUE = "human_false_at_least_two_true"
RULE_HUMAN_TRUE_ALL_FALSE = "human_true_all_three_false"
RULE_TEXT = {
    RULE_HUMAN_FALSE_GE2_TRUE: "人工标签为 false，且 p000、p000b、p001 三次中至少两次 raw 为 true",
    RULE_HUMAN_TRUE_ALL_FALSE: "人工标签为 true，且 p000、p000b、p001 三次 raw 都是 false",
}
ORDER_SEED = 20261009
RAW_VALUES = {"true", "false", "uncertain"}
DECISIONS = ("维持原标签", "改为是", "改为不是", "看不清")
LABEL_ZH = {"true": "是", "false": "不是"}
CSV_FIELDS = (
    "key",
    "class_name",
    "human_label",
    "unit",
    "bbox",
    "image_file",
    "crop_file",
    "thumb_file",
    "image_w",
    "image_h",
    "box_w_px",
    "box_h_px",
    "decision",
    "note",
    "reviewed_at",
)
API_FIELDS = (
    "key",
    "class_name",
    "human_label",
    "image_file",
    "crop_file",
    "thumb_file",
    "image_w",
    "image_h",
    "box_w_px",
    "box_h_px",
    "decision",
    "note",
    "reviewed_at",
)
CORRECTION_FIELDS = (
    "key",
    "class_name",
    "unit",
    "bbox",
    "original_label",
    "corrected_label",
    "note",
    "reviewed_at",
)
DECISION_TO_LABEL = {"改为是": "true", "改为不是": "false"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_BOX_COLOR = (255, 214, 0)


def file_slug(key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._")
    if not slug or not _SAFE_NAME.fullmatch(slug):
        raise ValueError(f"无法为框生成安全文件名：{key}")
    return slug


def normalize_bbox(bbox) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox 必须是 4 个数：{bbox}")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ValueError(f"bbox 宽高必须为正：{bbox}")
    if min(x1, y1) < -1e-6 or max(x2, y2) > 1 + 1e-3:
        raise ValueError(f"bbox 超出 0 到 1：{bbox}")
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_paths(root: Path) -> tuple[Path, Path, dict[str, Path]]:
    labels = root / "assets" / "labels.json"
    items = root / "assets" / "items.json"
    runs = {name: root / "runs" / name / "out" for name in RUN_IDS}
    missing = [str(path) for path in (labels, items, *runs.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError("数据集缺少文件：\n" + "\n".join(missing))
    return labels, items, runs


def load_run_dir(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"没有运行结果：{path}")
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for item in payload.get("results") or []:
            key = str(item["key"])
            raw = item.get("raw")
            if key in found:
                raise ValueError(f"运行结果里框重复：{path.parent.name} {key}")
            if raw in RAW_VALUES:
                found[key] = raw
    return found


def index_candidates(items: list[dict]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for item in items:
        unit = str(item["unit"])
        if not _SAFE_NAME.fullmatch(unit):
            raise ValueError(f"图片单元名含有不安全字符：{unit}")
        for candidate in item["candidates"]:
            key = str(candidate["key"])
            if key in found:
                raise ValueError(f"items.json 里框重复：{key}")
            found[key] = {"unit": unit, "bbox": normalize_bbox(candidate["bbox"])}
    return found


def classify_box(human_label: str, votes: dict[str, str | None]) -> str | None:
    ordered = [votes.get(run_id) for run_id in RUN_IDS]
    if human_label == "false" and sum(value == "true" for value in ordered) >= 2:
        return RULE_HUMAN_FALSE_GE2_TRUE
    if human_label == "true" and all(value == "false" for value in ordered):
        return RULE_HUMAN_TRUE_ALL_FALSE
    return None


def select_records(labels: dict, items: list[dict], runs: dict[str, dict[str, str]]) -> list[dict]:
    candidates = index_candidates(items)
    records = []
    missing_items = []
    for key, label in labels.items():
        human = label.get("label")
        if human not in {"true", "false"}:
            continue
        votes = {run_id: runs[run_id].get(key) for run_id in RUN_IDS}
        rule = classify_box(human, votes)
        if rule is None:
            continue
        candidate = candidates.get(key)
        if candidate is None:
            missing_items.append(key)
            continue
        slug = file_slug(key)
        records.append({
            "key": key,
            "class_name": label["class_name"],
            "human_label": human,
            "source": label.get("source", ""),
            "split": label.get("split", ""),
            "unit": candidate["unit"],
            "bbox": candidate["bbox"],
            "rule": rule,
            "raw": votes,
            "image_file": f"images/{candidate['unit']}.jpg",
            "crop_file": f"crops/{slug}.jpg",
            "thumb_file": f"thumbs/{slug}.jpg",
        })
    if missing_items:
        raise ValueError("入选框在 items.json 中不存在：" + ", ".join(missing_items[:10]))
    records.sort(key=lambda record: record["key"])
    random.Random(ORDER_SEED).shuffle(records)
    return records


def queue_row(record: dict) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "key": record["key"],
        "class_name": record["class_name"],
        "human_label": record["human_label"],
        "unit": record["unit"],
        "bbox": json.dumps(record["bbox"], ensure_ascii=False),
        "image_file": record["image_file"],
        "crop_file": record["crop_file"],
        "thumb_file": record["thumb_file"],
    })
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        handle.flush()
    temp_path.replace(path)


def public_row(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in API_FIELDS}


def vote_pattern(raw: dict[str, str | None]) -> str:
    return ",".join(f"{run_id}={raw.get(run_id) or 'missing'}" for run_id in RUN_IDS)


def build_summary(labels: dict, records: list[dict], runs: dict[str, dict[str, str]]) -> dict:
    by_split = Counter((label.get("split") or "unknown") for label in labels.values())
    by_rule = Counter(record["rule"] for record in records)
    by_class = Counter(record["class_name"] for record in records)
    by_class_rule = Counter((record["class_name"], record["rule"]) for record in records)
    labeled_classes = Counter(label["class_name"] for label in labels.values())
    patterns = Counter((record["rule"], vote_pattern(record["raw"])) for record in records)

    def covered(split: str) -> int:
        keys = [key for key, label in labels.items() if label.get("split") == split]
        return sum(all(key in runs[run_id] for run_id in RUN_IDS) for key in keys)

    return {
        "purpose": "人工复核疑似因高清帧时间差错位而被标错的框。复核页不展示模型结论，本文件只记录挑选数量。",
        "seed": ORDER_SEED,
        "runs": list(RUN_IDS),
        "raw_values": "只认 true、false、uncertain。uncertain 既不算判是，也不算判不是。缺少某次结果时，不能算作三次都是 false。",
        "rules": RULE_TEXT,
        "scope": {
            "labeled_boxes": len(labels),
            "by_split": dict(sorted(by_split.items())),
            "dev_with_three_runs": covered("dev"),
            "holdout_with_three_runs": covered("holdout"),
            "labeled_by_class": dict(sorted(labeled_classes.items())),
            "classes_with_labels_but_none_selected": sorted(
                name for name in labeled_classes if by_class[name] == 0
            ),
        },
        "selected_boxes": len(records),
        "selected_images": len({record["unit"] for record in records}),
        "by_rule": dict(by_rule),
        "by_class": dict(sorted(by_class.items())),
        "by_class_and_rule": {
            f"{class_name}/{rule}": count
            for (class_name, rule), count in sorted(by_class_rule.items())
        },
        "vote_patterns": {f"{rule}/{pattern}": count for (rule, pattern), count in sorted(patterns.items())},
    }


def write_selection(path: Path, records: list[dict]) -> None:
    payload = {
        "note": "仅用于核对挑选规则。复核页面和接口不读取本文件。",
        "seed": ORDER_SEED,
        "runs": list(RUN_IDS),
        "rules": RULE_TEXT,
        "boxes": [
            {
                "key": record["key"],
                "class_name": record["class_name"],
                "human_label": record["human_label"],
                "source": record["source"],
                "split": record["split"],
                "unit": record["unit"],
                "bbox": record["bbox"],
                "rule": record["rule"],
                "raw": record["raw"],
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reviewed_count(rows: list[dict[str, str]]) -> int:
    return sum(bool((row.get("decision") or "").strip()) for row in rows)


def build_package(dataset: Path, out_dir: Path) -> dict:
    labels_path, items_path, run_paths = dataset_paths(dataset)
    labels = load_json(labels_path)
    items = load_json(items_path)
    runs = {name: load_run_dir(path) for name, path in run_paths.items()}
    records = select_records(labels, items, runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "adjudication.csv"
    if csv_path.exists() and reviewed_count(read_csv(csv_path)):
        raise FileExistsError(f"{csv_path} 已有复核结果，拒绝覆盖")
    write_csv(csv_path, [queue_row(record) for record in records])
    write_selection(out_dir / "selection.json", records)
    summary = build_summary(labels, records, runs)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def zoom_window(width: int, height: int, bbox: list[float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    side = max(160.0, box_w * 6.0, box_h * 6.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top, right, bottom = cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(width, int(round(right)))
    bottom = min(height, int(round(bottom)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def _fit(image: Image.Image, long_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, long_side / max(width, height)) if max(width, height) > long_side else long_side / max(width, height)
    if abs(scale - 1.0) < 0.02:
        return image
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    return resized


def _draw_box(image: Image.Image, rect: tuple[float, float, float, float]) -> None:
    draw = ImageDraw.Draw(image)
    width = 4 if max(image.size) >= 500 else 2
    draw.rectangle(rect, outline=_BOX_COLOR, width=width)


def render_views(image: Image.Image, bbox: list[float]) -> tuple[Image.Image, Image.Image, dict[str, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    left, top, right, bottom = zoom_window(width, height, bbox)
    crop = rgb.crop((left, top, right, bottom))
    crop = _fit(crop, 720)
    scale_x = crop.size[0] / max(1, right - left)
    scale_y = crop.size[1] / max(1, bottom - top)
    _draw_box(crop, ((x1 - left) * scale_x, (y1 - top) * scale_y, (x2 - left) * scale_x, (y2 - top) * scale_y))
    thumb = _fit(rgb, 880)
    thumb_scale_x = thumb.size[0] / width
    thumb_scale_y = thumb.size[1] / height
    _draw_box(thumb, (x1 * thumb_scale_x, y1 * thumb_scale_y, x2 * thumb_scale_x, y2 * thumb_scale_y))
    meta = {
        "image_w": width,
        "image_h": height,
        "box_w_px": max(1, round(x2 - x1)),
        "box_h_px": max(1, round(y2 - y1)),
    }
    return crop, thumb, meta


def prepare_views(package: Path) -> dict[str, int]:
    rows = read_csv(package / "adjudication.csv")
    (package / "crops").mkdir(parents=True, exist_ok=True)
    (package / "thumbs").mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, Image.Image] = {}
    for row in rows:
        image_path = package / row["image_file"]
        if not image_path.is_file():
            raise FileNotFoundError(f"缺少原图：{image_path}")
        cache_key = str(image_path)
        if cache_key not in image_cache:
            with Image.open(image_path) as opened:
                image_cache[cache_key] = opened.convert("RGB")
        crop, thumb, meta = render_views(image_cache[cache_key], json.loads(row["bbox"]))
        crop.save(package / row["crop_file"], "JPEG", quality=90)
        thumb.save(package / row["thumb_file"], "JPEG", quality=90)
        for name, value in meta.items():
            row[name] = str(value)
    write_csv(package / "adjudication.csv", rows)
    summary_path = package / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        units = sorted({row["unit"] for row in rows})
        manifest = {}
        total_bytes = 0
        for unit in units:
            file_path = package / "images" / f"{unit}.jpg"
            data = file_path.read_bytes()
            total_bytes += len(data)
            manifest[unit] = hashlib.sha256(data).hexdigest()
        summary["image_files"] = len(manifest)
        summary["image_bytes"] = total_bytes
        summary["image_sha256"] = manifest
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"boxes": len(rows), "images": len(image_cache)}


def correction_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    changed = []
    for row in rows:
        decision = (row.get("decision") or "").strip()
        corrected = DECISION_TO_LABEL.get(decision)
        original = (row.get("human_label") or "").strip()
        if corrected is None or corrected == original:
            continue
        changed.append({
            "key": row.get("key", ""),
            "class_name": row.get("class_name", ""),
            "unit": row.get("unit", ""),
            "bbox": row.get("bbox", ""),
            "original_label": original,
            "corrected_label": corrected,
            "note": row.get("note", ""),
            "reviewed_at": row.get("reviewed_at", ""),
        })
    return changed


def assert_correction_output(path: Path) -> None:
    if path.name == "labels.json" or path.name.endswith(".jsonl"):
        raise ValueError(f"拒绝写入金标或标签文件：{path}")
    blocked = {"labels", "gold", "annotations.jsonl"}
    if path.stem in blocked or "labels.json" in path.name:
        raise ValueError(f"拒绝写入金标或标签文件：{path}")


def write_corrections(rows: list[dict[str, str]], path: Path) -> list[dict[str, str]]:
    assert_correction_output(path)
    changed = correction_rows(rows)
    write_csv(path, changed, CORRECTION_FIELDS)
    return changed
