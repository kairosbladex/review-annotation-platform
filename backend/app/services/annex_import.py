from __future__ import annotations

import json
from typing import Any

from ..schemas import CandidateInput, ImageReference, ReviewEventInput


def _as_items(imgxy: Any) -> list[dict[str, Any]]:
    if isinstance(imgxy, str):
        imgxy = json.loads(imgxy)
    if not isinstance(imgxy, list):
        raise ValueError("imgxy must be a JSON list of boxes")
    return [item for item in imgxy if isinstance(item, dict)]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def normalize_imgxy_targets(imgxy: Any) -> list[dict[str, Any]]:
    """Convert ktd `imgxy` pixel xywh boxes into normalized [x1, y1, x2, y2] targets."""
    targets: list[dict[str, Any]] = []
    for item in _as_items(imgxy):
        frame_width = float(item.get("frame_width") or 0)
        frame_height = float(item.get("frame_height") or 0)
        width = float(item.get("width") or 0)
        height = float(item.get("height") or 0)
        class_name = str(item.get("class_name") or "").strip()
        if frame_width <= 0 or frame_height <= 0 or width <= 0 or height <= 0 or not class_name:
            continue
        x1 = _clamp(float(item.get("x") or 0) / frame_width)
        y1 = _clamp(float(item.get("y") or 0) / frame_height)
        x2 = _clamp((float(item.get("x") or 0) + width) / frame_width)
        y2 = _clamp((float(item.get("y") or 0) + height) / frame_height)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = item.get("confidence")
        if confidence is not None:
            confidence = _clamp(float(confidence))
        targets.append({
            "target_index": len(targets),
            "class_id": None,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "raw_decision": "uncertain",
            "reason_code": "ktd_annex_import",
        })
    return targets


def public_object_key(file_path: str) -> str:
    return str(file_path or "").lstrip("/")


def build_review_event_from_annex(
    row: dict[str, Any],
    *,
    image_bucket: str = "minio-src",
    source_service: str = "ktd-lx",
) -> ReviewEventInput:
    targets = [CandidateInput.model_validate(item) for item in normalize_imgxy_targets(row.get("imgxy"))]
    if not targets:
        raise ValueError("annex row has no valid imgxy targets")
    object_key = public_object_key(str(row.get("file_path") or ""))
    if not object_key:
        raise ValueError("annex row is missing file_path")
    annex_id = str(row.get("id") or "").strip()
    if not annex_id:
        raise ValueError("annex row is missing id")
    return ReviewEventInput(
        event_id=f"ktd-annex-{annex_id}",
        source_service=source_service,
        capture_time=row.get("create_time") or row.get("date_time_original"),
        image=ImageReference(bucket=image_bucket, object_key=object_key),
        targets=targets,
        metadata={
            "annex_id": annex_id,
            "source_file_path": row.get("file_path"),
        },
    )
