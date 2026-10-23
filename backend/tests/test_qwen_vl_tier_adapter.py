from __future__ import annotations

import json

from backend.app.adapters.qwen_vl_tier import QwenVLTierAdapter
from backend.app.services.auto_annotation_prompts import ROLE_ORDER


class FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class FakeClient:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, *, files, data, headers):
        self.calls.append({"url": url, "files": files, "data": data, "headers": headers})
        return FakeResponse(
            {
                "success": True,
                "session_id": "session-1",
                "results": [
                    {
                        "decision": "confirmed",
                        "modelProbability": 0.93,
                        "reasonCode": "clean_source_confirmed",
                    },
                    {
                        "decision": "rejected",
                        "corrected_confidence": 0.88,
                        "reasonCode": "clean_source_rejected",
                    },
                    {"decision": "uncertain", "reasonCode": "clean_source_uncertain"},
                ],
            }
        )


class FakeStore:
    def get_bytes(self, bucket: str, key: str) -> bytes:
        assert bucket == "vision-test"
        assert key == "verify-source/frame.jpg"
        return b"jpeg-bytes"


def test_qwen_vl_tier_adapter_maps_real_review_results(monkeypatch):
    import backend.app.adapters.qwen_vl_tier as module

    FakeClient.calls = []
    monkeypatch.setattr(module.httpx, "Client", FakeClient)
    adapter = QwenVLTierAdapter(
        object_store=FakeStore(),
        base_url="http://algorithm:8193",
        api_key="test-key",
        endpoint="/api/v1/review-agent/review",
    )
    candidates = [
        {"target_index": 4, "class_id": 1, "class_name": "车", "confidence": 0.7, "bbox": [0, 0, 1, 1]},
        {"target_index": 5, "class_id": 2, "class_name": "人员", "confidence": 0.6, "bbox": [1, 1, 2, 2]},
        {"target_index": 6, "class_id": 3, "class_name": "三轮车", "confidence": 0.5, "bbox": [2, 2, 3, 3]},
    ]

    result = adapter.review(
        {"bucket": "vision-test", "object_key": "verify-source/frame.jpg"},
        candidates,
        {"tier2_threshold": 0.12},
    )

    call = FakeClient.calls[0]
    detections = json.loads(call["data"]["detections"])
    assert [item["target_index"] for item in detections] == [4, 5, 6]
    assert call["headers"] == {"X-API-Key": "test-key"}
    assert set(call["files"]) == {"image", "clean_image"}
    decisions = result["decisions"]
    assert [decision["target_index"] for decision in decisions] == [4, 5, 6]
    assert [decision["consensus"] for decision in decisions] == ["true", "false", "uncertain"]
    assert all(len(decision["votes"]) == len(ROLE_ORDER) for decision in decisions)
    assert decisions[0]["votes"][0]["confidence"] == 0.93
    assert decisions[1]["votes"][0]["confidence"] == 0.88
