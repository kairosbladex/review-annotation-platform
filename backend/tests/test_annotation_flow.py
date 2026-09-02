from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
        service_api_key="service-secret",
        admin_username="admin",
        admin_password="admin-password",
        jwt_secret="test-jwt-secret-with-at-least-32-bytes",
        minio_endpoint="",
    )


def _login(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "admin-password"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _review_event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "source_service": "yoloe-review-service",
        "session_id": "session-001",
        "capture_time": datetime.now(timezone.utc).isoformat(),
        "camera_id": "camera-7",
        "rule_code": "waterline-v1",
        "image": {
            "bucket": "review-images",
            "object_key": "review/2026/09/02/session-001/original.jpg",
            "sha256": "abc123",
        },
        "comparison_image": {
            "bucket": "review-images",
            "object_key": "review/2026/09/02/session-001/comparison.jpg",
        },
        "targets": [
            {
                "target_index": 0,
                "class_id": 2,
                "class_name": "车",
                "confidence": 0.61,
                "bbox": [0.1, 0.1, 0.24, 0.25],
                "raw_decision": "uncertain",
                "yolo_decision": "uncertain",
                "llm_decision": None,
                "reason_code": "model_conflict",
                "error_code": None,
                "reasoning": "两路判断不一致",
            },
            {
                "target_index": 1,
                "class_id": 2,
                "class_name": "车",
                "confidence": 0.48,
                "bbox": [0.5, 0.2, 0.7, 0.45],
                "raw_decision": "uncertain",
                "yolo_decision": "uncertain",
                "llm_decision": None,
                "reason_code": "low_confidence",
                "error_code": "dependency_error",
                "reasoning": "低置信度",
            },
        ],
    }


def test_review_event_is_idempotent_and_labels_complete_sample(tmp_path):
    client = TestClient(create_app(_settings(tmp_path)))
    event = _review_event("event-001")
    headers = {"X-Service-Key": "service-secret"}

    first = client.post("/api/v1/ingest/review-events", json=event, headers=headers)
    assert first.status_code == 201, first.text
    sample_id = first.json()["sample_id"]
    assert first.json()["annotation_task_id"]

    duplicate = client.post("/api/v1/ingest/review-events", json=event, headers=headers)
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["sample_id"] == sample_id
    assert duplicate.json()["created"] is False

    auth = _login(client)
    detail = client.get(f"/api/v1/samples/{sample_id}", headers=auth)
    assert detail.status_code == 200, detail.text
    sample = detail.json()
    assert sample["status"] == "pending"
    assert len(sample["candidates"]) == 2
    assert sample["candidates"][0]["bbox"] == [0.1, 0.1, 0.24, 0.25]

    first_candidate = sample["candidates"][0]["candidate_id"]
    second_candidate = sample["candidates"][1]["candidate_id"]
    labeled = client.patch(
        f"/api/v1/candidates/{first_candidate}/annotation",
        json={"label": "true", "note": "确认是目标"},
        headers=auth,
    )
    assert labeled.status_code == 200, labeled.text
    assert labeled.json()["sample"]["status"] == "in_progress"

    completed = client.patch(
        f"/api/v1/candidates/{second_candidate}/annotation",
        json={"label": "false", "note": "背景纹理"},
        headers=auth,
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["sample"]["status"] == "completed"
    assert completed.json()["sample"]["counts"] == {
        "total": 2,
        "labeled": 2,
        "pending": 0,
        "true": 1,
        "false": 1,
    }
