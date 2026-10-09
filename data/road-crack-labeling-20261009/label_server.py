#!/usr/bin/env python3
"""道路裂纹候选框标注页（仅本机 127.0.0.1）。只写本目录 annotations.csv。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PACKAGE = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE / "manifest.json"
CSV_PATH = PACKAGE / "annotations.csv"
INDEX_PATH = PACKAGE / "index.html"
ALLOWED_LABELS = {"是裂纹", "不是裂纹", "看不清"}
FIELDNAMES = ["key", "label", "note", "labeled_by", "labeled_at"]
LOCK = threading.Lock()


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    images = {item["unit"]: item for item in manifest["images"]}
    candidates = {item["key"]: item for item in manifest["candidates"]}
    order = manifest["order"]
    if len(order) != len(set(order)) or set(order) != set(candidates):
        raise SystemExit("manifest.json 的 order 与 candidates 不一致")
    missing = [item["file"] for item in manifest["images"] if not (PACKAGE / item["file"]).is_file()]
    if missing:
        sample = "、".join(missing[:5])
        raise SystemExit(f"缺少 {len(missing)} 张图片（例如 {sample}）。图片不入库，需按 README 复制到 images/。")
    manifest["image_by_unit"] = images
    manifest["candidate_by_key"] = candidates
    return manifest


MANIFEST = load_manifest()


def empty_row(key: str) -> dict[str, str]:
    return {"key": key, "label": "", "note": "", "labeled_by": "", "labeled_at": ""}


def read_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        return [empty_row(key) for key in MANIFEST["order"]]
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [row.get("key", "") for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(MANIFEST["order"]):
        raise ValueError("annotations.csv 的框与 manifest.json 不一致，未覆盖该文件")
    for row in rows:
        for name in FIELDNAMES:
            row.setdefault(name, "")
    return rows


def write_rows(rows: list[dict[str, str]]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix="annotations.", suffix=".csv", dir=PACKAGE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, CSV_PATH)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def public_row(row: dict[str, str]) -> dict:
    candidate = MANIFEST["candidate_by_key"][row["key"]]
    image = MANIFEST["image_by_unit"][candidate["unit"]]
    return {
        "key": row["key"],
        "unit": candidate["unit"],
        "image": image["file"],
        "width": image["width"],
        "height": image["height"],
        "bbox": candidate["bbox"],
        "label": row.get("label", ""),
        "note": row.get("note", ""),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PACKAGE), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/rows":
            try:
                with LOCK:
                    rows = read_rows()
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
                return
            self.send_json({"rows": [public_row(row) for row in rows], "total": len(rows)})
            return
        if path in {"/", "/index.html"}:
            body = INDEX_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("请求体无效")
            payload = json.loads(self.rfile.read(length))
            key = str(payload.get("key", ""))
            label = str(payload.get("label", ""))
            note = str(payload.get("note", ""))[:500]
            if key not in MANIFEST["candidate_by_key"] or label not in ALLOWED_LABELS:
                raise ValueError("key 或 label 无效")
            with LOCK:
                rows = read_rows()
                matches = [row for row in rows if row.get("key") == key]
                if len(matches) != 1:
                    raise ValueError("找不到唯一候选框")
                matches[0]["label"] = label
                matches[0]["note"] = note
                matches[0]["labeled_by"] = "wangxiaolin"
                matches[0]["labeled_at"] = datetime.now(timezone.utc).isoformat()
                write_rows(rows)
            self.send_json({"ok": True, "key": key, "label": label})
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动道路裂纹候选框标注页")
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    with LOCK:
        rows = read_rows()
        if not CSV_PATH.exists():
            write_rows(rows)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"道路裂纹标注页: http://127.0.0.1:{args.port}/")
    print("数据文件:", CSV_PATH)
    print(f"共 {MANIFEST['box_count']} 框，顺序种子 {MANIFEST['seed']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n标注页已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
