"""LangGraph assembly (P1).

Builds a StateGraph from a PipelineDef: linear chain, fail-fast conditional
edges, per-node exception capture, and a SQLite checkpointer keyed by
run_id. Node and pipeline contracts are unchanged from the P0 runner, so
existing nodes and YAML keep working.
"""

from __future__ import annotations

import sqlite3
import traceback
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph

from agent.core import registry
from agent.core.pipeline import PipelineDef
from agent.core.state import RunState


def open_checkpointer(db_path: Path) -> SqliteSaver:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def _guarded(name: str, fn: Any) -> Any:
    def call(state: RunState) -> dict[str, Any]:
        try:
            return fn(state)
        except GraphInterrupt:
            raise  # human gates pause the run; never convert to failure
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(
                {
                    "node": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            return {"status": "failed", "errors": errors}

    call.__name__ = name
    return call


def build(pipeline: PipelineDef, checkpointer: SqliteSaver) -> Any:
    builder = StateGraph(RunState)
    node_names = list(pipeline.nodes)
    for node_name in node_names:
        builder.add_node(node_name, _guarded(node_name, registry.get(node_name)))
    for index, node_name in enumerate(node_names):
        if index == 0:
            builder.add_edge(START, node_name)
        nxt = node_names[index + 1] if index + 1 < len(node_names) else END

        def route(state: RunState, _nxt: Any = nxt) -> Any:
            return END if state.get("status") == "failed" else _nxt

        builder.add_conditional_edges(node_name, route, {nxt: nxt, END: END})
    return builder.compile(checkpointer=checkpointer)
