"""Ingest stage: normalize source items (images / text) into a manifest.

Reference implementation over the local filesystem. Each item gets a
content sha256 — the basis for dedup and downstream idempotency. Duplicate
content is kept once and recorded; unreadable items go to quarantine and
never fail the batch.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path

from agent.core.registry import register
from agent.core.state import RunState

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".log"}


def _kind_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    guessed, _ = mimetypes.guess_type(str(path))
    return "unknown" if guessed is None else guessed.split("/", 1)[0]


@register("ingest_files")
def ingest_files(state: RunState) -> RunState:
    params = state["params"]
    source_dir = Path(str(params["source_dir"]))
    pattern = str(params.get("glob", "**/*"))
    output_dir = Path(
        str(params.get("output_dir", "data/agent-runs/{run_id}/ingest")).format(
            run_id=state["run_id"]
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: dict[str, str] = {}  # sha256 -> first item_id
    duplicates: list[dict] = []
    quarantine: list[dict] = []
    for path in sorted(source_dir.glob(pattern)):
        if not path.is_file():
            continue
        try:
            payload = path.read_bytes()
            if not payload:
                raise ValueError("empty file")
            sha = hashlib.sha256(payload).hexdigest()
        except Exception as exc:
            quarantine.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        item_id = path.stem
        if sha in seen:
            duplicates.append({"path": str(path), "sha256": sha, "kept": seen[sha]})
            continue
        seen[sha] = item_id
        rows.append(
            {
                "item_id": item_id,
                "kind": _kind_of(path),
                "uri": f"file://{path.resolve()}",
                "sha256": sha,
                "bytes": len(payload),
            }
        )

    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    if duplicates:
        (output_dir / "duplicates.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in duplicates) + "\n",
            encoding="utf-8",
        )
    if quarantine:
        (output_dir / "quarantine.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in quarantine) + "\n",
            encoding="utf-8",
        )
    state["artifacts"].append(
        {
            "kind": "ingest_manifest",
            "uri": f"file://{manifest.resolve()}",
            "producer": "ingest_files",
            "meta": {"source_dir": str(source_dir), "glob": pattern},
        }
    )
    state["metrics"]["ingest"] = {
        "items": len(rows),
        "duplicates": len(duplicates),
        "quarantined": len(quarantine),
    }
    return state
