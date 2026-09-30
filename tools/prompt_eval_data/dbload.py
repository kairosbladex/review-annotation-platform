"""Read-only loading of candidate/annotation rows from the platform SQLite DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_QUERY = """
SELECT c.id AS candidate_id,
       c.sample_id AS sample_id,
       c.model_run_id AS model_run_id,
       c.target_index AS target_index,
       c.class_name AS class_name,
       c.confidence AS confidence,
       c.x1 AS x1,
       c.y1 AS y1,
       c.x2 AS x2,
       c.y2 AS y2,
       c.raw_decision AS raw_decision,
       c.yolo_decision AS yolo_decision,
       c.reason_code AS reason_code,
       s.source_type AS source_type,
       s.source_event_id AS source_event_id,
       s.source_service AS source_service,
       s.session_id AS session_id,
       s.capture_time AS capture_time,
       s.camera_id AS camera_id,
       s.rule_code AS rule_code,
       s.metadata_json AS metadata_json,
       a.bucket AS asset_bucket,
       a.object_key AS object_key,
       a.sha256 AS asset_sha256,
       a.kind AS asset_kind,
       an.id AS annotation_id,
       an.label AS annotation_label,
       an.note AS annotation_note,
       an.labeled_by AS labeled_by,
       an.revision AS annotation_revision,
       an.labeled_at AS labeled_at,
       u.username AS reviewer
FROM candidates c
JOIN samples s ON s.id = c.sample_id
LEFT JOIN assets a ON a.sample_id = c.sample_id AND a.kind = 'original'
LEFT JOIN annotations an ON an.candidate_id = c.id
LEFT JOIN users u ON u.id = an.labeled_by
ORDER BY c.id
"""


def _metadata(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if raw in (None, ""):
        return {}, None
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        return None, f"invalid_json:{exc.__class__.__name__}"
    if not isinstance(value, dict):
        return None, "metadata_not_object"
    return value, None


def load_rows(db_path: str) -> list[dict[str, Any]]:
    """Load every candidate with its (optional) annotation. Never writes to the DB."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = []
        for r in con.execute(_QUERY):
            parsed_metadata, metadata_error = _metadata(r["metadata_json"])
            annotation = None
            if r["annotation_id"] is not None:
                annotation = {
                    "annotation_id": r["annotation_id"],
                    "label": r["annotation_label"],
                    "note": r["annotation_note"] or "",
                    "labeled_by": r["labeled_by"],
                    "revision": r["annotation_revision"],
                    "labeled_at": str(r["labeled_at"]) if r["labeled_at"] is not None else None,
                    "reviewer": r["reviewer"],
                }
            rows.append(
                {
                    "candidate_id": r["candidate_id"],
                    "sample_id": r["sample_id"],
                    "model_run_id": r["model_run_id"],
                    "target_index": r["target_index"],
                    "class_name": r["class_name"],
                    "confidence": r["confidence"],
                    "bbox_norm": [r["x1"], r["y1"], r["x2"], r["y2"]],
                    "raw_decision": r["raw_decision"],
                    "yolo_decision": r["yolo_decision"],
                    "reason_code": r["reason_code"],
                    "source_type": r["source_type"],
                    "source_event_id": r["source_event_id"],
                    "source_service": r["source_service"],
                    "session_id": r["session_id"],
                    "capture_time": str(r["capture_time"]) if r["capture_time"] is not None else None,
                    "camera_id": r["camera_id"],
                    "rule_code": r["rule_code"],
                    "metadata": parsed_metadata,
                    "metadata_json": r["metadata_json"],
                    "metadata_error": metadata_error,
                    "asset_bucket": r["asset_bucket"],
                    "object_key": r["object_key"],
                    "asset_sha256": r["asset_sha256"],
                    "asset_kind": r["asset_kind"],
                    "annotation": annotation,
                }
            )
        return rows
    finally:
        con.close()
