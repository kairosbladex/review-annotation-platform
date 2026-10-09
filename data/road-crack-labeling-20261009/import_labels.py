#!/usr/bin/env python3
"""把 annotations.csv 写成与 luna-prompt-autotune assets/labels.json 相同的结构。

是裂纹 -> label true，不是裂纹 -> label false。看不清和未标注不导入。
只写本目录 labels-road-crack.json，不读取、不修改服务器上的 assets。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
LABEL_OF = {"是裂纹": "true", "不是裂纹": "false"}
SKIP = {"", "看不清"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_labels(rows: list[dict[str, str]], manifest: dict) -> tuple[dict, dict[str, int]]:
    by_key = {item["key"]: item for item in manifest["candidates"]}
    seen: set[str] = set()
    chosen: dict[str, dict] = {}
    counts = {"true": 0, "false": 0, "看不清": 0, "未标注": 0}
    for row in rows:
        key = (row.get("key") or "").strip()
        label = (row.get("label") or "").strip()
        if not key or key not in by_key:
            raise ValueError(f"未知框：{key or '(空)'}")
        if key in seen:
            raise ValueError(f"重复框：{key}")
        seen.add(key)
        if label in SKIP:
            counts["看不清" if label == "看不清" else "未标注"] += 1
            continue
        if label not in LABEL_OF:
            raise ValueError(f"{key} 的标注无法导入：{label}")
        candidate = by_key[key]
        mapped = LABEL_OF[label]
        chosen[key] = {
            "label": mapped,
            "class_name": candidate["class_name"],
            "source": candidate["source"],
            "split": candidate["split"],
        }
        counts[mapped] += 1
    missing = set(by_key) - seen
    if missing:
        raise ValueError(f"annotations.csv 缺少 {len(missing)} 个框，例如 {sorted(missing)[:3]}")
    ordered = {key: chosen[key] for key in (item["key"] for item in manifest["candidates"]) if key in chosen}
    return ordered, counts


def write_json(path: Path, payload: dict) -> None:
    fd, temp_name = tempfile.mkstemp(prefix="labels.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="把道路裂纹人工标注转成 labels.json 格式")
    parser.add_argument("--annotations", type=Path, default=PACKAGE / "annotations.csv")
    parser.add_argument("--manifest", type=Path, default=PACKAGE / "manifest.json")
    parser.add_argument("--out", type=Path, default=PACKAGE / "labels-road-crack.json")
    args = parser.parse_args()
    if not args.annotations.is_file():
        raise SystemExit(f"找不到标注文件：{args.annotations}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels, counts = build_labels(load_rows(args.annotations), manifest)
    write_json(args.out, labels)
    print(
        f"已写入 {args.out}：是裂纹 {counts['true']}，不是裂纹 {counts['false']}，"
        f"跳过看不清 {counts['看不清']}，未标注 {counts['未标注']}"
    )
    print("未修改服务器 assets。合并进自动迭代前，需另行决定这些框进入 dev 还是 holdout。")


if __name__ == "__main__":
    main()
