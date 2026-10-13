"""Pipeline definitions.

A pipeline is versioned YAML: which nodes run, in what order, with what
params. Business iteration happens here and in node implementations, not in
the runner. pipeline_id + version are recorded on every run for lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineDef:
    pipeline_id: str
    version: str
    nodes: list[str]
    params: dict[str, Any] = field(default_factory=dict)


def load(path: Path) -> PipelineDef:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline file must be a mapping: {path}")
    for key in ("pipeline_id", "version"):
        if not payload.get(key):
            raise ValueError(f"pipeline missing {key!r}: {path}")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(n, str) for n in nodes):
        raise ValueError(f"pipeline must declare a non-empty nodes list: {path}")
    return PipelineDef(
        pipeline_id=str(payload["pipeline_id"]),
        version=str(payload["version"]),
        nodes=list(nodes),
        params=dict(payload.get("params") or {}),
    )
