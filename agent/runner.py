"""Graph runner (P1).

LangGraph + SQLite checkpointer, keyed by run_id. A crashed local process
resumes with the same run_id: completed nodes are skipped from the
checkpoint, and the remote-submit/monitor nodes re-attach to the remote job
instead of resubmitting it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.core import graph as graph_mod
from agent.core.pipeline import PipelineDef
from agent.core.state import RunState

DEFAULT_CHECKPOINT_DB = Path("data/agent-runs/agent-checkpoints.sqlite3")


def new_run_id(pipeline_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{pipeline_id}-{stamp}-{uuid4().hex[:8]}"


def _initial_state(
    pipeline: PipelineDef, run_id: str, params: dict[str, Any] | None
) -> RunState:
    return {
        "run_id": run_id,
        "pipeline_id": pipeline.pipeline_id,
        "pipeline_version": pipeline.version,
        "params": {**pipeline.params, **(params or {})},
        "artifacts": [],
        "metrics": {},
        "gates": {},
        "errors": [],
        "status": "running",
    }


def _finalize(final: RunState) -> RunState:
    if final.get("__interrupt__"):  # type: ignore[typeddict-item]
        final["status"] = "waiting_human"
    elif final.get("status") == "running":
        final["status"] = "completed"
    return final


def run(
    pipeline: PipelineDef,
    *,
    run_id: str | None = None,
    params: dict[str, Any] | None = None,
    checkpoint_db: Path = DEFAULT_CHECKPOINT_DB,
) -> RunState:
    rid = run_id or new_run_id(pipeline.pipeline_id)
    graph = graph_mod.build(pipeline, graph_mod.open_checkpointer(checkpoint_db))
    final = graph.invoke(
        _initial_state(pipeline, rid, params),
        {"configurable": {"thread_id": rid}},
    )
    return _finalize(final)


def resume(
    pipeline: PipelineDef,
    *,
    run_id: str,
    decision: Any | None = None,
    checkpoint_db: Path = DEFAULT_CHECKPOINT_DB,
) -> RunState:
    """Resume a run.

    Three cases on the same thread:
    - waiting at a human gate: pass decision to answer the interrupt;
    - crashed mid-graph: completed nodes are skipped from the checkpoint;
    - reached END with status=failed: start a fresh attempt from START.
      Nodes stay idempotent for a fixed run_id (remote_submit attaches to
      the existing remote job instead of resubmitting), so a fresh attempt
      is cheap and safe.
    """
    graph = graph_mod.build(pipeline, graph_mod.open_checkpointer(checkpoint_db))
    config = {"configurable": {"thread_id": run_id}}
    if decision is not None:
        from langgraph.types import Command

        final = graph.invoke(Command(resume=decision), config)
    elif graph.get_state(config).next:
        final = graph.invoke(None, config)
    else:
        final = graph.invoke(_initial_state(pipeline, run_id, None), config)
    return _finalize(final)
