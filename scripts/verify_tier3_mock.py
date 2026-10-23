#!/usr/bin/env python3
"""Deterministic Tier 3 Responses fixture for isolated API verification.

This is a contract fixture, not a Qwen3-VL model-quality or business-accuracy
oracle.  It lets an isolated backend exercise the real ingest/auto-annotation
HTTP path and SQLite persistence when no independent Tier 3 model service is
available.  It deliberately returns a decision for every candidate in the
prompt so a missing or malformed candidate fails loudly instead of being
silently auto-confirmed.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARKER = "<UNTRUSTED_CANDIDATE_DATA>"
END_MARKER = "</UNTRUSTED_CANDIDATE_DATA>"


def _candidates(prompt: str) -> list[dict]:
    start = prompt.find(MARKER)
    end = prompt.find(END_MARKER)
    if start < 0 or end <= start:
        raise ValueError("prompt has no candidate data block")
    body = prompt[start + len(MARKER) : end].strip()
    parsed = json.loads(body)
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate data block has no candidates list")
    return candidates


def _decision(candidate: dict) -> tuple[str, str]:
    class_name = str(candidate.get("class_name") or "")
    if candidate.get("geometric_warning"):
        return "REJECTED", "fixture: geometric conflict must go through Tier 3"
    if "违建" in class_name or "roof" in class_name.lower():
        return "CONFIRMED", "fixture: rooftop scene evidence"
    return "CONFIRMED", "fixture: target evidence present"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        return

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/healthz":
            self._json(200, {"status": "ok", "fixture": True})
            return
        self._json(404, {"error": "unknown route"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/responses":
            self._json(404, {"error": "unknown route"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        content = payload.get("input", [{}])[0].get("content", [])
        prompt = next(
            (
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "input_text"
            ),
            "",
        )
        try:
            candidates = _candidates(prompt)
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": str(exc)})
            return
        decisions = []
        scene = "ground"
        for position, candidate in enumerate(candidates):
            decision, reason = _decision(candidate)
            if "违建" in str(candidate.get("class_name") or ""):
                scene = "rooftop"
            decisions.append(
                {
                    "target_index": int(candidate.get("target_index", position)),
                    "decision": decision,
                    "confidence": 0.95,
                    "reason": reason,
                }
            )
        self._json(
            200,
            {
                "scene": {
                    "label": scene,
                    "confidence": 0.95,
                    "reason": "deterministic Tier 3 fixture",
                },
                "decisions": decisions,
            },
        )


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18393), Handler).serve_forever()
