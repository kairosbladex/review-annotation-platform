#!/usr/bin/env python3
"""未曝光人员框的本机待标页。没有样本时不接受标签，也不预填。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

PACKAGE = Path(__file__).resolve().parent
LABELS = ("是人员", "不是人员", "看不清")
POISON = ("model_decision", "rawDecision", "gold", "human_label", "reasoning", "confidence")
# 这些字段即使填了，也不能单独证明图片和初审框来自同一帧。
_NOT_BINDING = ("verified", "frame_binding", "same_frame_verified", "bytes_differ", "byte_different",
                "nearby_time", "time_delta", "mad", "similarity", "image_similarity")
_IMAGE_NAME = re.compile(r"^[A-Za-z0-9._/-]+$")
_WRITE_LOCK = threading.Lock()


class QueueError(ValueError):
    pass


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _box(item):
    raw = item.get("bbox")
    if isinstance(raw, dict):
        values = [raw.get(name) for name in ("x", "y", "width", "height")]
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = list(raw)
    else:
        return None
    numbers = [_positive_int(value) if name in {"width", "height"} else value
               for name, value in zip(("x", "y", "width", "height"), values)]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in numbers):
        return None
    x, y, width, height = numbers
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _image_path(root: Path, item):
    relative = item.get("image_file")
    if not isinstance(relative, str) or not _IMAGE_NAME.fullmatch(relative) or ".." in relative.split("/"):
        return None
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    return path


def _image_size(path: Path):
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, UnidentifiedImageError):
        return None


def _binding(evidence):
    if not isinstance(evidence, dict):
        return None
    evidence = {key: value for key, value in evidence.items() if key not in _NOT_BINDING}
    task_id = evidence.get("task_id")
    stream_id = evidence.get("stream_id")
    binding_id = evidence.get("binding_id")
    frame_id = evidence.get("frame_id")
    pts_ns = evidence.get("pts_ns")
    texts = (task_id, stream_id, binding_id)
    if not all(isinstance(value, str) and value.strip() for value in texts):
        return None
    if binding_id.strip().lower() in {"verified", "same_frame_verified", "true"}:
        return None
    frame_ok = isinstance(frame_id, str) and frame_id.strip() and frame_id.strip().lower() not in {"verified", "true"}
    pts_ok = isinstance(pts_ns, int) and not isinstance(pts_ns, bool) and pts_ns >= 0
    if not frame_ok and not pts_ok:
        return None
    return {
        "task_id": task_id.strip(),
        "stream_id": stream_id.strip(),
        "frame_id": frame_id.strip() if frame_ok else "",
        "pts_ns": pts_ns if pts_ok else None,
        "binding_id": binding_id.strip(),
    }


def admission_errors(item: dict, root: Path) -> list[str]:
    """缺图、缺框、缺任务分组或缺来源绑定时，样本不能进入待标队列。"""
    errors = []
    if not isinstance(item.get("group"), str) or not item.get("group").strip():
        errors.append("缺少任务分组")
    box = _box(item)
    width = _positive_int(item.get("frame_width"))
    height = _positive_int(item.get("frame_height"))
    if box is None or width is None or height is None or box["x"] + box["width"] > width or box["y"] + box["height"] > height:
        errors.append("框坐标或画面尺寸无效")
    image = _image_path(root, item)
    size = _image_size(image) if image is not None else None
    if size is None:
        errors.append("缺少可读取图片")
    elif width is not None and height is not None and (width, height) != size:
        errors.append("画面尺寸与图片不一致")
    if _binding(item.get("source_evidence")) is None:
        errors.append("缺少可核验的来源绑定")
    return errors


def load_queue(path: Path | None = None) -> dict:
    path = path or PACKAGE / "samples.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        if not isinstance(candidate_id, str) or not candidate_id.strip() or candidate_id in seen:
            raise QueueError("待标样本缺少唯一编号")
        seen.add(candidate_id)
    for item in manual:
        if not isinstance(item, dict) or item.get("box_origin") != "manual":
            raise QueueError("人工补框必须单独标记，不能进入初审队列")
        if item.get("label") not in (None, ""):
            raise QueueError("人工补框不能预填标签")
        if any(key in item for key in POISON):
            raise QueueError("人工补框含有模型结论或旧标签")
        if item.get("candidate_id") in seen:
            raise QueueError("人工补框不能与初审框使用同一编号")
    root = path.parent
    admitted = []
    rejected = []
    for item in eligible:
        errors = admission_errors(item, root)
        if errors:
            rejected.append({"candidate_id": item["candidate_id"], "reasons": errors})
        else:
            admitted.append(item)
    payload["eligible"] = admitted
    payload["rejected"] = rejected
    payload["manual_boxes"] = manual
    return payload


def public_queue(payload: dict, annotations_path: Path | None = None) -> dict:
    saved = {}
    if annotations_path is not None:
        for row in read_annotations(annotations_path):
            if row.get("label") in LABELS:
                saved[row.get("candidate_id") or ""] = row["label"]
    eligible = []
    for item in payload["eligible"]:
        binding = _binding(item.get("source_evidence"))
        eligible.append({
            "candidate_id": item["candidate_id"],
            "group": item["group"].strip(),
            "class_name": item["class_name"],
            "box_origin": "initial_review",
            "bbox": _box(item),
            "frame_width": item["frame_width"],
            "frame_height": item["frame_height"],
            "image_file": item["image_file"],
            "source": binding,
            "label": saved.get(item["candidate_id"], ""),
        })
    manual = []
    for item in payload["manual_boxes"]:
        manual.append({
            "candidate_id": item.get("candidate_id") or "",
            "group": item.get("group") or "",
            "box_origin": "manual",
        })
    return {
        "eligible": eligible,
        "manual_boxes": manual,
        "rejected": payload.get("rejected") or [],
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
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/queue":
            self._json(public_queue(load_queue(self.package / "samples.json"), self.csv_path))
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
            rejected = {item["candidate_id"] for item in queue["rejected"]}
            if candidate_id in rejected:
                raise QueueError("样本缺少必要证据，不能标注")
            if candidate_id not in known or label not in LABELS:
                raise QueueError("没有这个初审框，或标签不在允许范围内")
            with _WRITE_LOCK:
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
