"""Deterministic serialization and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jdumps(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, UTF-8 (no ASCII escaping)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jsonl_bytes(rows: Iterable[Any]) -> bytes:
    lines = [jdumps(row) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def json_bytes(obj: Any) -> bytes:
    return (jdumps(obj) + "\n").encode("utf-8")
