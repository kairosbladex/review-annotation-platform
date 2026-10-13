"""Node registry.

Pipelines reference nodes by name. Business iteration means adding or
swapping node implementations here, without touching the runner or the
(P1) graph builder. A node is a pure-ish function RunState -> RunState:
idempotent for the same run_id, artifacts written by reference.
"""

from __future__ import annotations

from collections.abc import Callable

from agent.core.state import RunState

NodeFn = Callable[[RunState], RunState]

_REGISTRY: dict[str, NodeFn] = {}


def register(name: str) -> Callable[[NodeFn], NodeFn]:
    def decorator(fn: NodeFn) -> NodeFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate node registration: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get(name: str) -> NodeFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown node {name!r}; registered: {sorted(_REGISTRY)}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)
