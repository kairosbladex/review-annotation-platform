#!/usr/bin/env python3
"""本机人工标签复核页。只监听 127.0.0.1，只写本目录 adjudication.csv。"""

from __future__ import annotations

import argparse
import json
import re
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from adjudication import API_FIELDS, DECISIONS, CSV_FIELDS, public_row, read_csv, write_csv

PACKAGE = Path(__file__).resolve().parent
CSV_PATH = PACKAGE / "adjudication.csv"
_WRITE_LOCK = threading.Lock()
_STATIC_NAME = re.compile(r"^(images|crops|thumbs)/[A-Za-z0-9._-]+$")
_BLOCKED = {"selection.json", "summary.json", "adjudication.csv", "corrections.csv"}


def validate_queue(rows: list[dict[str, str]], package: Path) -> None:
    keys = [row.get("key", "") for row in rows]
    if len(keys) != len(set(keys)) or any(not key for key in keys):
        raise ValueError("adjudication.csv 存在空的或重复的框编号")
    for row in rows:
        decision = (row.get("decision") or "").strip()
        if decision and decision not in DECISIONS:
            raise ValueError(f"未知复核选项：{decision}")
        for field in ("image_file", "crop_file", "thumb_file"):
            relative = row.get(field, "")
            if not _STATIC_NAME.fullmatch(relative):
                raise ValueError(f"图片路径不安全：{relative}")
            if not (package / relative).is_file():
                raise ValueError(f"缺少图片：{relative}")


class Handler(BaseHTTPRequestHandler):
    package = PACKAGE
    csv_path = CSV_PATH

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        name = path.lstrip("/")
        if path in {"/", "/index.html"}:
            page = (self.package / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if path == "/api/rows":
            rows = [public_row(row) for row in read_csv(self.csv_path)]
            extra = set().union(*(row.keys() for row in rows)) - set(API_FIELDS)
            if extra:
                self.send_json({"error": "接口字段超出复核页允许范围"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json({"rows": rows})
            return
        if name in _BLOCKED or not _STATIC_NAME.fullmatch(name):
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        file_path = (self.package / name).resolve()
        if not file_path.is_relative_to(self.package.resolve()) or not file_path.is_file():
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self.send_file(file_path)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("请求体无效")
            payload = json.loads(self.rfile.read(length))
            key = str(payload.get("key", "")).strip()
            decision = str(payload.get("decision", "")).strip()
            note = str(payload.get("note", "")).replace("\r\n", "\n").strip()[:500]
            if decision not in DECISIONS:
                raise ValueError("复核选项无效")
            with _WRITE_LOCK:
                rows = read_csv(self.csv_path)
                matches = [row for row in rows if row.get("key") == key]
                if len(matches) != 1:
                    raise ValueError("找不到唯一的框")
                matches[0]["decision"] = decision
                matches[0]["note"] = note
                matches[0]["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                write_csv(self.csv_path, rows, CSV_FIELDS)
            self.send_json({"ok": True, "key": key, "decision": decision})
        except (ValueError, json.JSONDecodeError, OSError, KeyError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本机人工标签复核页")
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    rows = read_csv(CSV_PATH)
    validate_queue(rows, PACKAGE)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    reviewed = sum(bool((row.get("decision") or "").strip()) for row in rows)
    print(f"人工标签复核页: http://127.0.0.1:{args.port}/")
    print(f"共 {len(rows)} 框，已复核 {reviewed}。结果只写入 {CSV_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n复核页已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
