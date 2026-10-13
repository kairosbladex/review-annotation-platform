#!/usr/bin/env python3
"""未曝光人员框的本机待标页。没有样本时不接受标签，也不预填。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

PACKAGE = Path(__file__).resolve().parent
LABELS = ("是人员", "不是人员", "看不清")
POISON = ("model_decision", "rawDecision", "gold", "human_label", "reasoning", "confidence")


class QueueError(ValueError):
    pass


def load_queue(path: Path | None = None) -> dict:
    payload = json.loads((path or PACKAGE / "samples.json").read_text(encoding="utf-8"))
    if payload.get("show_model_answer") is not False or payload.get("prefill") is not False:
        raise QueueError("队列不允许展示模型答案或预填标签")
    if list(payload.get("labels") or []) != list(LABELS):
        raise QueueError("标注选项必须是：是人员、不是人员、看不清")
    eligible = payload.get("eligible")
    manual = payload.get("manual_boxes")
    if not isinstance(eligible, list) or not isinstance(manual, list):
        raise QueueError("eligible 和 manual_boxes 必须是列表")
    seen = set()
    for item in eligible:
        if not isinstance(item, dict):
            raise QueueError("待标样本必须是对象")
        if item.get("label") not in (None, ""):
            raise QueueError("待标样本不能预填标签")
        if item.get("box_origin") != "initial_review":
            raise QueueError("待标框必须是初审原框")
        if item.get("class_name") not in {"人员", "人"}:
            raise QueueError("待标框的初审类别必须是人员")
        if any(key in item for key in POISON):
            raise QueueError("待标样本含有模型结论或旧标签")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise QueueError("待标样本缺少唯一编号")
        seen.add(candidate_id)
    for item in manual:
        if not isinstance(item, dict) or item.get("box_origin") != "manual":
            raise QueueError("人工补框必须单独标记，不能进入初审队列")
        if item.get("candidate_id") in seen:
            raise QueueError("人工补框不能与初审框使用同一编号")
    return payload


def public_queue(payload: dict) -> dict:
    eligible = []
    for item in payload["eligible"]:
        eligible.append({
            "candidate_id": item["candidate_id"],
            "group": item.get("group") or "",
            "class_name": item["class_name"],
            "box_origin": "initial_review",
            "bbox": item.get("bbox"),
            "evidence": item.get("evidence") or {},
            "label": "",
        })
    return {
        "eligible": eligible,
        "manual_boxes": [],
        "labels": list(LABELS),
        "blocked_groups": payload.get("blocked_groups") or [],
        "show_model_answer": False,
    }


def read_annotations(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["candidate_id", "label", "note", "labeled_at"]
    fd, temp_name = tempfile.mkstemp(prefix="annotations.", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class Handler(SimpleHTTPRequestHandler):
    package = PACKAGE
    csv_path = PACKAGE / "annotations.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.package), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/queue":
            self._json(public_queue(load_queue(self.package / "samples.json")))
            return
        if path in {"/", "/index.html"}:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/label":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            candidate_id = str(payload.get("candidate_id") or "")
            label = str(payload.get("label") or "")
            note = str(payload.get("note") or "")[:500]
            queue = load_queue(self.package / "samples.json")
            known = {item["candidate_id"] for item in queue["eligible"]}
            if candidate_id not in known or label not in LABELS:
                raise QueueError("没有这个初审框，或标签不在允许范围内")
            rows = read_annotations(self.csv_path)
            rows = [row for row in rows if row.get("candidate_id") != candidate_id]
            rows.append({
                "candidate_id": candidate_id,
                "label": label,
                "note": note,
                "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            write_annotations(self.csv_path, rows)
        except (QueueError, json.JSONDecodeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        self._json({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args()
    load_queue()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
