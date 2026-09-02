from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app

from .test_annotation_flow import _login
from .test_imports import MemoryObjectStore
from .test_model_runs import _settings


def test_manifest_import_preserves_external_context_and_comparison_asset(tmp_path):
    object_store = MemoryObjectStore()
    object_store.put_bytes("review-images", "incoming/event-001.jpg", b"image", "image/jpeg")
    object_store.put_bytes("review-images", "incoming/event-001-compare.jpg", b"compare", "image/jpeg")
    manifest = [{
        "external_id": "event-001",
        "object_key": "incoming/event-001.jpg",
        "comparison_object_key": "incoming/event-001-compare.jpg",
        "session_id": "session-001",
        "metadata": {"source": "manifest", "priority": "high"},
    }]
    object_store.put_bytes("review-images", "manifests/batch.json", json.dumps(manifest).encode(), "application/json")
    app = create_app(_settings(tmp_path), object_store=object_store)
    client = TestClient(app)
    auth = _login(client)

    response = client.post(
        "/api/v1/imports",
        json={"source_type": "minio", "bucket": "review-images", "manifest_key": "manifests/batch.json"},
        headers=auth,
    )
    assert response.status_code == 201, response.text
    assert response.json()["imported"] == 1

    samples = client.get("/api/v1/samples", headers=auth).json()["samples"]
    assert len(samples) == 1
    assert samples[0]["source_event_id"] == "event-001"
    assert samples[0]["session_id"] == "session-001"
    assert samples[0]["metadata"] == {"source": "manifest", "priority": "high"}
    assert {asset["kind"] for asset in samples[0]["assets"]} == {"original", "comparison"}
