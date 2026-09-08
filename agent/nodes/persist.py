"""Persist stage: write stage outputs into the local agent store.

Reference implementation over stdlib sqlite3 — demonstrates durable
persistence with lineage (run, pipeline version, content hash) and
idempotent re-runs. P3 swaps this for the platform database; the contract
(same input + same pipeline version => same rows, no duplicates) stays.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.core.registry import register
from agent.core.state import RunState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT NOT NULL,
    pipeline_id TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kind TEXT,
    sha256 TEXT,
    payload_json TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (item_id, pipeline_id, pipeline_version)
)
"""


def _input_rows(state: RunState, path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@register("persist_store")
def persist_store(state: RunState) -> RunState:
    params = state["params"]
    # default input: the most recent manifest/analysis artifact
    source = params.get("input_file")
    if source:
        input_path = Path(str(source).format(run_id=state["run_id"]))
    else:
        candidates = [
            a
            for a in state["artifacts"]
            if a["kind"] in ("analysis", "ingest_manifest") and a["uri"].startswith("file://")
        ]
        if not candidates:
            raise RuntimeError("persist_store: no input_file and no manifest/analysis artifact")
        input_path = Path(candidates[-1]["uri"][len("file://"):])
    db_path = Path(
        str(params.get("store_db", "data/agent-runs/agent-store.sqlite3")).format(
            run_id=state["run_id"]
        )
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _input_rows(state, input_path)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_SCHEMA)
        conn.executemany(
            "INSERT OR REPLACE INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(row.get("item_id") or row.get("image_id") or row.get("id") or ""),
                    state["pipeline_id"],
                    state["pipeline_version"],
                    state["run_id"],
                    str(row.get("kind", "")),
                    str(row.get("sha256", "")),
                    json.dumps(row, ensure_ascii=False),
                    now,
                )
                for row in rows
            ],
        )
        total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    state["artifacts"].append(
        {
            "kind": "store_db",
            "uri": f"file://{db_path.resolve()}",
            "producer": "persist_store",
            "meta": {"input": str(input_path)},
        }
    )
    state["metrics"]["persist"] = {"written": len(rows), "store_total": total}
    return state
