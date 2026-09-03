"""Read-only loading of candidate/annotation rows from the platform SQLite DB."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_QUERY = """
SELECT c.id, c.sample_id, c.model_run_id, c.target_index, c.class_name, c.confidence,
       c.x1, c.y1, c.x2, c.y2, c.raw_decision, c.yolo_decision, c.reason_code,
       s.source_event_id, s.rule_code,
       a.object_key,
       an.id, an.label, an.note, an.labeled_by, an.revision, an.labeled_at,
       u.username
FROM candidates c
JOIN samples s ON s.id = c.sample_id
LEFT JOIN assets a ON a.sample_id = c.sample_id AND a.kind = 'original'
LEFT JOIN annotations an ON an.candidate_id = c.id
LEFT JOIN users u ON u.id = an.labeled_by
ORDER BY c.id
"""


def load_rows(db_path: str) -> list[dict[str, Any]]:
    """Load every candidate with its (optional) annotation. Never writes to the DB."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = []
        for r in con.execute(_QUERY):
            annotation = None
            if r[16] is not None:
                annotation = {
                    "annotation_id": r[16],
                    "label": r[17],
                    "note": r[18] or "",
                    "labeled_by": r[19],
                    "revision": r[20],
                    "labeled_at": str(r[21]) if r[21] is not None else None,
                    "reviewer": r[22],
                }
            rows.append(
                {
                    "candidate_id": r[0],
                    "sample_id": r[1],
                    "model_run_id": r[2],
                    "target_index": r[3],
                    "class_name": r[4],
                    "confidence": r[5],
                    "bbox_norm": [r[6], r[7], r[8], r[9]],
                    "raw_decision": r[10],
                    "yolo_decision": r[11],
                    "reason_code": r[12],
                    "source_event_id": r[13],
                    "rule_code": r[14],
                    "object_key": r[15],
                    "annotation": annotation,
                }
            )
        return rows
    finally:
        con.close()
