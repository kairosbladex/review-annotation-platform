#!/usr/bin/env python3
"""Local-only browser labeler for the prerelease annotation package."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "annotations.csv"
ALLOWED_LABELS = {"真车", "屋顶设施", "看不清"}


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict[str, str]]) -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    if not fieldnames:
        raise RuntimeError("annotations.csv 没有表头")
    fd, temp_name = tempfile.mkstemp(prefix="annotations.", suffix=".csv", dir=ROOT)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print("[labeler] " + format % args)

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
            rows = read_rows()
            self.send_json({"rows": rows})
            return
        if path in {"/", "/index.html"}:
            self.path = "/index.html"
        super().do_GET()

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
        except (ValueError, json.JSONDecodeError, OSError, RuntimeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8786), Handler)
    print("标注页: http://127.0.0.1:8786/")
    print("数据文件:", CSV_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n标注页已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
