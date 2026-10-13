from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.adapters.object_store import ObjectStore
from app.config import Settings
from app.main import create_app
from app.models import AgentReviewRequest, Annotation, Candidate


class MemoryStore(ObjectStore):
    def __init__(self):
        self.objects = {
            ("source", "frames/one.jpg"),
            ("source", "frames/two.jpg"),
        }

    def exists(self, bucket: str, object_key: str) -> bool:
        return (bucket, object_key) in self.objects

    def presigned_get_url(self, bucket, object_key, expires_seconds=900):
        return f"memory://{bucket}/{object_key}"

    def put_bytes(self, bucket, object_key, data, content_type):
        self.objects.add((bucket, object_key))

    def get_bytes(self, bucket, object_key):
        return b"image"

    def list_keys(self, bucket, prefix):
        return []


def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'agent-api.sqlite3'}",
        service_api_key='REDACTED_FOR_GITHUB',
        admin_username="admin",
        admin_password='REDACTED_FOR_GITHUB',
        jwt_secret='REDACTED_FOR_GITHUB',
        minio_endpoint="",
    )


def event(index: int) -> dict:
    return {
        "event_id": f"source-event-{index}",
        "source_service": "review-service",
        "capture_time": datetime.now(timezone.utc).isoformat(),
        "image": {
            "bucket": "source",
            "object_key": f"frames/{'one' if index == 1 else 'two'}.jpg",
            "sha256": f"{index:064x}",
        },
        "targets": [
            {
                "target_index": 0,
                "class_id": 999 + index,
                "class_name": "车辆" if index == 1 else "人员",
                "confidence": 0.75,
                "bbox": [0.1, 0.2, 0.5, 0.8],
                "raw_decision": "uncertain",
                "yolo_decision": "true",
                "llm_decision": None,
                "reason_code": "model_conflict",
                "reasoning": "source model evidence",
            }
        ],
    }


def login(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": 'REDACTED_FOR_GITHUB'},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def seeded_app(tmp_path):
    app = create_app(settings(tmp_path), object_store=MemoryStore())
    client = TestClient(app)
    service_headers = {"X-Service-Key": "agent-service-key"}
    for index in (1, 2):
        response = client.post(
            "/api/v1/ingest/review-events",
            json=event(index),
            headers=service_headers,
        )
        assert response.status_code == 201, response.text
    return app, client, service_headers


def test_candidates_are_service_authenticated_and_cursor_paginated(tmp_path):
    _app, client, headers = seeded_app(tmp_path)

    assert client.get("/api/v1/agent/candidates").status_code == 401
    first = client.get(
        "/api/v1/agent/candidates", params={"limit": 1}, headers=headers
    )
    assert first.status_code == 200, first.text
    assert len(first.json()["items"]) == 1
    assert first.json()["next_cursor"]
    second = client.get(
        "/api/v1/agent/candidates",
        params={"limit": 1, "cursor": first.json()["next_cursor"]},
        headers=headers,
    )
    assert second.status_code == 200, second.text
    assert len(second.json()["items"]) == 1
    assert second.json()["items"][0]["candidate_id"] != first.json()["items"][0][
        "candidate_id"
    ]
    envelope = first.json()["items"][0]
    assert set(envelope) == {
        "candidate_id",
        "source_event_id",
        "source_service",
        "image",
        "class_name",
        "source_class_id",
        "bbox",
        "initial_evidence",
        "created_at",
    }
    assert len(envelope["image"]["sha256"]) == 64
    assert envelope["initial_evidence"]["model_run_id"]


def test_review_request_is_idempotent_and_never_writes_annotation(tmp_path):
    app, client, headers = seeded_app(tmp_path)
    candidate_id = client.get(
        "/api/v1/agent/candidates", headers=headers
    ).json()["items"][0]["candidate_id"]
    payload = {
        "cycle_id": "cycle-1",
        "candidate_id": candidate_id,
        "reason_code": "model_conflict",
        "reason": "GPU and GPT disagree",
        "attempt_refs": ["attempt-1", "attempt-2"],
    }

    first = client.post("/api/v1/agent/review-requests", json=payload, headers=headers)
    repeated = client.post(
        "/api/v1/agent/review-requests", json=payload, headers=headers
    )
    assert first.status_code == 201, first.text
    assert repeated.status_code == 200, repeated.text
    assert first.json()["review_request_id"] == repeated.json()["review_request_id"]
    assert repeated.json()["created"] is False

    conflict = client.post(
        "/api/v1/agent/review-requests",
        json={**payload, "reason_code": "different"},
        headers=headers,
    )
    assert conflict.status_code == 409
    with app.state.session_factory() as session:
        assert session.query(AgentReviewRequest).count() == 1
        assert session.scalar(
            select(Annotation).where(Annotation.candidate_id == candidate_id)
        ) is None


def test_human_gold_excludes_automatic_labels_and_notes(tmp_path):
    app, client, headers = seeded_app(tmp_path)
    items = client.get("/api/v1/agent/candidates", headers=headers).json()["items"]
    human_candidate = items[0]["candidate_id"]
    automatic_candidate = items[1]["candidate_id"]
    auth = login(client)
    labeled = client.patch(
        f"/api/v1/candidates/{human_candidate}/annotation",
        json={"label": "true", "note": "private human note"},
        headers=auth,
    )
    assert labeled.status_code == 200, labeled.text
    with app.state.session_factory() as session:
        session.add(
            Annotation(
                candidate_id=automatic_candidate,
                label="false",
                note="automatic note",
                labeled_by=None,
                labeled_at=datetime.now(timezone.utc),
                revision=1,
            )
        )
        session.commit()

    gold = client.get("/api/v1/agent/human-gold", headers=headers)
    assert gold.status_code == 200, gold.text
    assert len(gold.json()["items"]) == 1
    record = gold.json()["items"][0]
    assert record["candidate_id"] == human_candidate
    assert record["label"] == "true"
    assert record["labeled_by"]
    assert record["revision"] == 1
    assert "note" not in record
    assert record["image"]["sha256"]


def test_human_label_completes_agent_review_request_and_clear_reopens_it(tmp_path):
    app, client, headers = seeded_app(tmp_path)
    candidate_id = client.get(
        "/api/v1/agent/candidates", headers=headers
    ).json()["items"][0]["candidate_id"]
    requested = client.post(
        "/api/v1/agent/review-requests",
        headers=headers,
        json={
            "cycle_id": "cycle-review-status",
            "candidate_id": candidate_id,
            "reason_code": "model_conflict",
            "reason": "GPU and GPT disagree",
            "attempt_refs": ["gpu-1", "gpt-1"],
        },
    )
    assert requested.status_code == 201
    auth = login(client)

    labeled = client.patch(
        f"/api/v1/candidates/{candidate_id}/annotation",
        json={"label": "true"},
        headers=auth,
    )
    assert labeled.status_code == 200
    with app.state.session_factory() as session:
        review = session.get(AgentReviewRequest, requested.json()["review_request_id"])
        assert review.status == "completed"

    cleared = client.patch(
        f"/api/v1/candidates/{candidate_id}/annotation",
        json={"label": None, "expected_revision": 1},
        headers=auth,
    )
    assert cleared.status_code == 200
    with app.state.session_factory() as session:
        review = session.get(AgentReviewRequest, requested.json()["review_request_id"])
        assert review.status == "pending"
