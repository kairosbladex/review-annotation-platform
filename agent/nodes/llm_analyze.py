"""Analyze stage: generic per-item LLM analysis (OpenAI-compatible HTTP).

The only stage where an LLM lives. Endpoint, model and key come from env
(AGENT_LLM_BASE_URL / AGENT_LLM_API_KEY / AGENT_LLM_MODEL) — never from
RunState. The prompt template is pipeline config. Every item is analyzed
independently: transport errors, timeouts and non-JSON answers go to a
review queue for a human, they never fail the batch and never get
auto-filled.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import httpx

from agent.core.registry import register
from agent.core.state import RunState


def _load_items(state: RunState, params: dict[str, Any]) -> tuple[list[dict], Path]:
    source = params.get("manifest")
    if source:
        path = Path(str(source).format(run_id=state["run_id"]))
    else:
        candidates = [
            a
            for a in state["artifacts"]
            if a["kind"] == "ingest_manifest" and a["uri"].startswith("file://")
        ]
        if not candidates:
            raise RuntimeError("llm_analyze: no manifest param and no ingest_manifest artifact")
        path = Path(candidates[-1]["uri"][len("file://"):])
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows, path


def _content_for(item: dict, prompt_template: str, text_bytes: int) -> list[dict]:
    uri = str(item.get("uri", ""))
    path = Path(uri[len("file://"):]) if uri.startswith("file://") else None
    if item.get("kind") == "image" and path is not None:
        encoded = base64.b64encode(path.read_bytes()).decode()
        suffix = path.suffix.lstrip(".").lower() or "jpeg"
        return [
            {"type": "text", "text": prompt_template},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{suffix};base64,{encoded}"},
            },
        ]
    text = ""
    if path is not None:
        text = path.read_text(encoding="utf-8", errors="replace")[:text_bytes]
    return [{"type": "text", "text": prompt_template.format(item_text=text, **item)}]


def _call_llm(content: list[dict], *, timeout: float) -> dict:
    base_url = os.environ.get("AGENT_LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("AGENT_LLM_API_KEY", "")
    model = os.environ.get("AGENT_LLM_MODEL", "")
    if not base_url or not api_key or not model:
        raise RuntimeError("AGENT_LLM_BASE_URL / AGENT_LLM_API_KEY / AGENT_LLM_MODEL not set")
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    return json.loads(text)  # non-JSON answers raise -> review queue


@register("llm_analyze")
def llm_analyze(state: RunState) -> RunState:
    params = state["params"]
    prompt_template = str(params["prompt_template"])
    text_bytes = int(params.get("text_bytes", 4000))
    timeout = float(params.get("timeout", 60.0))
    limit = params.get("limit")
    output_dir = Path(
        str(params.get("output_dir", "data/agent-runs/{run_id}/analysis")).format(
            run_id=state["run_id"]
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    items, _ = _load_items(state, params)
    if limit is not None:
        items = items[: int(limit)]

    results: list[dict] = []
    review: list[dict] = []
    for item in items:
        try:
            content = _content_for(item, prompt_template, text_bytes)
            answer = _call_llm(content, timeout=timeout)
            results.append({**item, "analysis": answer})
        except Exception as exc:
            review.append({**item, "error": f"{type(exc).__name__}: {exc}"})

    results_path = output_dir / "analysis.jsonl"
    results_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results)
        + ("\n" if results else ""),
        encoding="utf-8",
    )
    if review:
        (output_dir / "review_queue.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in review) + "\n",
            encoding="utf-8",
        )
    state["artifacts"].append(
        {
            "kind": "analysis",
            "uri": f"file://{results_path.resolve()}",
            "producer": "llm_analyze",
        }
    )
    state["metrics"]["analyze"] = {"ok": len(results), "review_queue": len(review)}
    return state
