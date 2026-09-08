"""LangSmith tracing with a hard no-op fallback.

Observability must never take a run down: when LANGCHAIN_API_KEY is absent
(or a self-hosted LANGCHAIN_ENDPOINT is unreachable), tracing stays disabled
and execution continues untraced. Every node call goes through run_traced,
so the fallback policy lives in exactly one place.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from agent.core.state import RunState

F = TypeVar("F", bound=Callable[[RunState], RunState])

_ENABLED = False


def setup(project: str) -> bool:
    """Enable LangSmith tracing when credentials are configured.

    Returns True when tracing is active. Never raises.
    """
    global _ENABLED
    if _ENABLED:
        return True
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return False
    try:
        import langsmith  # noqa: F401  (import failure must not kill runs)
    except Exception:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    _ENABLED = True
    return True


def enabled() -> bool:
    return _ENABLED


def run_traced(name: str, fn: F, state: RunState) -> RunState:
    if _ENABLED:
        from langsmith import traceable

        return traceable(name=name)(fn)(state)
    return fn(state)
