#!/usr/bin/env python3
"""新事件高清补标页（仅本机）。只写本目录 annotations.csv。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PACKAGE = Path(__file__).resolve().parent
# 静态文件根设为上一级，便于同时展示原低清裁图作对照
STATIC_ROOT = PACKAGE.parent
CSV_PATH = PACKAGE / "annotations.csv"
ALLOWED_LABELS = {"真车", "屋顶设施", "看不清"}


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    fd, temp_name = tempfile.mkstemp(prefix="annotations.", suffix=".csv", dir=PACKAGE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, CSV_PATH)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        pass

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/rows":
            self.send_json({"rows": read_rows()})
            return
        if path in {"/", "/index.html"}:
            self.path = f"/{PACKAGE.name}/index.html"
        range_header = self.headers.get("Range")
        if range_header and self.path.endswith(".mp4"):
            self.send_range(range_header)
            return
        super().do_GET()

    def send_range(self, range_header: str) -> None:
        # Safari 播放 mp4 需要 Range 支持
        file_path = Path(self.translate_path(self.path))
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = file_path.stat().st_size
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not match:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        start = int(match.group(1)) if match.group(1) else max(0, size - int(match.group(2) or 0))
        end = int(match.group(2)) if match.group(1) and match.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with file_path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(end - start + 1))

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            candidate_id = str(payload.get("candidate_id", ""))
            label = str(payload.get("label", ""))
            note = str(payload.get("note", ""))[:500]
            if not candidate_id or label not in ALLOWED_LABELS:
                raise ValueError("candidate_id 或 label 无效")
            rows = read_rows()
            matches = [row for row in rows if row.get("candidate_id") == candidate_id]
            if len(matches) != 1:
                raise ValueError("找不到唯一候选框")
            matches[0]["label"] = label
            matches[0]["note"] = note
            write_rows(rows)
            self.send_json({"ok": True, "candidate_id": candidate_id, "label": label})
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动新事件高清补标页")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"新事件高清补标页: http://127.0.0.1:{args.port}/")
    print("数据文件:", CSV_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n标注页已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
