from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

import pytest

from app.adapters.model_provider import HttpModelAdapter, LegacyReviewAgentAdapter, ModelAdapterRouter

from .test_imports import MemoryObjectStore


class ContractHandler(BaseHTTPRequestHandler):
    mode = "success"
    last_request: dict = {}

    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        ContractHandler.last_request = {
            "path": self.path,
            "api_key": self.headers.get("X-API-Key"),
            "content_type": self.headers.get("Content-Type"),
            "body_size": len(body),
        }
        if self.mode == "error":
            self._send_json({"success": False, "error": "dependency unavailable"})
        elif self.mode == "malformed":
            self._send_json({"success": True, "detections": [{"bbox": [0.1, 0.2]}]})
        else:
            self._send_json({
                "success": True,
                "detections": [{
                    "class_id": 2,
                    "class_name": "车",
                    "confidence": 0.61,
                    "bbox": [0.1, 0.1, 0.24, 0.25],
                    "raw_decision": "uncertain",
                    "yolo_decision": "uncertain",
                    "llm_decision": None,
                    "reason_code": "dependency_error",
                    "error_code": "dependency_error",
                    "reasoning": "依赖服务不可用，保留存疑",
                }],
                "preview_url": "/preview.jpg",
            })

    def do_GET(self):
        if urlparse(self.path).path != "/preview.jpg":
            self.send_response(404)
            self.end_headers()
            return
        payload = b"preview-image"
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def contract_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ContractHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_legacy_yolo_detect_contract_preserves_review_fields_and_preview(contract_server):
    ContractHandler.mode = "success"
    store = MemoryObjectStore()
    store.put_bytes("review-images", "raw/sample.jpg", b"raw-image", "image/jpeg")
    adapter = HttpModelAdapter(object_store=store, default_base_url=contract_server, default_api_key='REDACTED_FOR_GITHUB')

    result = adapter.infer(
        {"bucket": "review-images", "object_key": "raw/sample.jpg"},
        {"confidence_threshold": 0.42, "_model_profile": {"endpoint": "/api/v1/review-agent/yolo-detect", "api_key": 'REDACTED_FOR_GITHUB'}},
    )

    request = ContractHandler.last_request
    assert parse_qs(urlparse(request["path"]).query)["conf"] == ["0.42"]
    assert request["api_key"] == "profile-key"
    assert "multipart/form-data" in request["content_type"]
    assert request["body_size"] > len(b"raw-image")
    assert result["predictions"] == [{
        "target_index": 0,
        "class_id": 2,
        "class_name": "车",
        "confidence": 0.61,
        "bbox": [0.1, 0.1, 0.24, 0.25],
        "raw_decision": "uncertain",
        "yolo_decision": "uncertain",
        "llm_decision": None,
        "reason_code": "dependency_error",
        "error_code": "dependency_error",
        "reasoning": "依赖服务不可用，保留存疑",
    }]
    assert result["artifact"] == f"{contract_server}/preview.jpg"
    assert adapter.fetch_artifact(result["artifact"]) == (b"preview-image", "image/jpeg")


@pytest.mark.parametrize("mode, expected", [("error", "dependency unavailable"), ("malformed", "invalid bbox")])
def test_legacy_yolo_detect_contract_surfaces_invalid_or_failed_responses(contract_server, mode, expected):
    ContractHandler.mode = mode
    store = MemoryObjectStore()
    store.put_bytes("review-images", "raw/sample.jpg", b"raw-image", "image/jpeg")
    adapter = HttpModelAdapter(object_store=store, default_base_url=contract_server)
    with pytest.raises(RuntimeError, match=expected):
        adapter.infer({"bucket": "review-images", "object_key": "raw/sample.jpg"}, {})


class ReviewContractHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        self.__class__.requests.append({
            "path": urlparse(self.path).path,
            "api_key": self.headers.get("X-API-Key"),
            "body": body,
        })
        if urlparse(self.path).path.endswith("/yolo-detect"):
            self._send_json({
                "success": True,
                "detections": [{
                    "class_id": 2,
                    "class_name": "车",
                    "confidence": 0.61,
                    "bbox": [0.1, 0.1, 0.24, 0.25],
                }],
            })
            return
        self._send_json({
            "success": True,
            "results": [{
                "original_class": "车",
                "original_bbox": [0.1, 0.1, 0.24, 0.25],
                "original_confidence": 0.61,
                "decision": "confirmed",
                "rawDecision": "uncertain",
                "reasoning": "依赖服务不可用，保留存疑",
                "confidence_score": 0.55,
                "yolo_decision": "uncertain",
                "llm_decision": None,
                "reasonCode": "dependency_error",
                "errorCode": "dependency_error",
            }],
            "image_url": "/review-static/comparison.jpg",
        })

    def do_GET(self):
        self.__class__.requests.append({
            "path": urlparse(self.path).path,
            "api_key": self.headers.get("X-API-Key"),
            "body": b"",
        })
        if urlparse(self.path).path != "/review-static/comparison.jpg":
            self.send_response(404)
            self.end_headers()
            return
        payload = b"review-preview"
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def review_contract_server():
    ReviewContractHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReviewContractHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_legacy_review_agent_adapter_runs_detection_then_review(review_contract_server):
    store = MemoryObjectStore()
    store.put_bytes("review-images", "raw/sample.jpg", b"raw-image", "image/jpeg")
    adapter = LegacyReviewAgentAdapter(
        object_store=store,
        default_base_url=review_contract_server,
        default_api_key='REDACTED_FOR_GITHUB',
    )

    result = adapter.infer(
        {"bucket": "review-images", "object_key": "raw/sample.jpg"},
        {
            "confidence_threshold": 0.42,
            "rule_code": "waterline-v1",
            "_model_profile": {
                "adapter_type": "legacy_review_agent",
                "api_key": 'REDACTED_FOR_GITHUB',
                "detect_endpoint": "/api/v1/review-agent/yolo-detect",
                "review_endpoint": "/api/v1/review-agent/review",
            },
        },
    )

    assert [request["path"] for request in ReviewContractHandler.requests] == [
        "/api/v1/review-agent/yolo-detect",
        "/api/v1/review-agent/review",
    ]
    assert all(request["api_key"] == "profile-key" for request in ReviewContractHandler.requests)
    assert b'"detections"' in ReviewContractHandler.requests[1]["body"]
    assert b'name="rule_code"' in ReviewContractHandler.requests[1]["body"]
    assert result["predictions"] == [{
        "target_index": 0,
        "class_id": 2,
        "class_name": "车",
        "confidence": 0.61,
        "bbox": [0.1, 0.1, 0.24, 0.25],
        "raw_decision": "uncertain",
        "yolo_decision": "uncertain",
        "llm_decision": None,
        "reason_code": "dependency_error",
        "error_code": "dependency_error",
        "reasoning": "依赖服务不可用，保留存疑",
    }]
    assert result["artifact"] == f"{review_contract_server}/review-static/comparison.jpg"
    assert adapter.fetch_artifact(
        result["artifact"],
        {"_model_profile": {"api_key": 'REDACTED_FOR_GITHUB', "verify_tls": False}},
    ) == (b"review-preview", "image/jpeg")
    assert ReviewContractHandler.requests[-1]["api_key"] == "artifact-profile-key"


def test_router_applies_production_defaults_to_legacy_adapter(monkeypatch):
    router = ModelAdapterRouter(
        object_store=MemoryObjectStore(),
        default_base_url="http://review-service:8188",
        default_api_key='REDACTED_FOR_GITHUB',
        default_adapter_type="legacy_review_agent",
        default_detect_endpoint="/detect-default",
        default_review_endpoint="/review-default",
        default_verify_tls=False,
    )

    adapter = router._adapter({"_model_profile": {}})

    assert isinstance(adapter, LegacyReviewAgentAdapter)
    assert adapter.default_detect_endpoint == "/detect-default"
    assert adapter.default_review_endpoint == "/review-default"
    assert adapter.default_verify_tls is False
