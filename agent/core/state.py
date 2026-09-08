"""Run state shared by all pipeline nodes.

The state is the single source of truth passed between nodes. It must stay
JSON-serializable so a checkpointer (LangGraph, P1) can persist it at any
step: large payloads live in object storage or the filesystem and enter the
state only as ArtifactRef (uri + sha256), never as bytes. Secrets never
enter the state.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ArtifactRef(TypedDict, total=False):
    kind: str  # dataset | checkpoint | report | manifest | ...
    uri: str  # file://... | minio://bucket/key
    sha256: str
    producer: str  # name of the node that created it
    meta: dict[str, Any]


class RunState(TypedDict, total=False):
    run_id: str
    pipeline_id: str
    pipeline_version: str
    params: dict[str, Any]
    artifacts: list[ArtifactRef]
    metrics: dict[str, Any]
    gates: dict[str, Any]  # gate name -> {passed, reasons, decided_by}
    remote_job: dict[str, Any]  # handle for SSH jobs: run_dir, state, ...
    errors: list[dict[str, str]]
    status: str  # running | waiting_human | completed | failed
