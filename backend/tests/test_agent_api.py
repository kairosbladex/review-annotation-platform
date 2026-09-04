from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import Annotation, Asset, Candidate, ModelRun, Sample, User
from app.security import hash_password


class StubObjectStore:
    pass


def build_client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'agent-api.sqlite3'}",
        service_api_key="agent-secret",
        admin_password="admin-secret",
        jwt_secret="test-jwt-secret",
        minio_auto_create_bucket=False,
    )
    app = create_app(settings, object_store=StubObjectStore())
    return app, TestClient(app)


def seed_candidate(app, *, suffix: str, sha256: str | None, created_at: datetime):
    with app.state.session_factory() as session:
        sample = Sample(
            source_type="review_service",
            source_event_id=f"event-{suffix}",
            source_service="ktd-lx",
            metadata_json='{"rule_code":"vehicle"}',
            created_at=created_at,
        )
        session.add(sample)
        session.flush()
        session.add(
            Asset(
                sample_id=sample.id,
                kind="original",
                bucket="source-images",
                object_key=f"camera/{suffix}.jpg",
                sha256=sha256,
                status="ready",
                created_at=created_at,
            )
        )
        run = ModelRun(
            model_name="yoloe",
            model_version="prod-20260901",
            status="succeeded",
            source_type="review_service",
            created_at=created_at,
        )
        session.add(run)
        session.flush()
        candidate = Candidate(
            sample_id=sample.id,
            model_run_id=run.id,
            target_index=0,
            class_id=917,
            class_name="车辆",
            confidence=0.72,
            x1=0.1,
            y1=0.2,
            x2=0.4,
            y2=0.6,
            raw_decision="true",
            yolo_decision="true",
            reasoning="initial detector evidence",
            created_at=created_at,
        )
        session.add(candidate)
        session.commit()
        return candidate.id


def test_candidates_require_service_key_and_page_with_opaque_cursor(tmp_path):
    app, client = build_client(tmp_path)
    now = datetime.now(timezone.utc)
    first_id = seed_candidate(app, suffix="001", sha256="a" * 64, created_at=now)
    second_id = seed_candidate(app, suffix="002", sha256="b" * 64, created_at=now + timedelta(seconds=1))
    seed_candidate(app, suffix="missing-sha", sha256=None, created_at=now + timedelta(seconds=2))
    seed_candidate(app, suffix="invalid-sha", sha256="short", created_at=now + timedelta(seconds=3))

    assert client.get("/api/v1/agent/candidates").status_code == 401
    first_page = client.get(
        "/api/v1/agent/candidates",
        params={"limit": 1},
        headers={"X-Service-Key": "agent-secret"},
    )

    assert first_page.status_code == 200
    body = first_page.json()
    assert [item["candidate_id"] for item in body["items"]] == [first_id]
    assert body["items"][0]["image"] == {
        "bucket": "source-images",
        "object_key": "camera/001.jpg",
        "sha256": "a" * 64,
    }
    assert body["items"][0]["class_name"] == "车辆"
    assert body["items"][0]["source_class_id"] == 917
    assert body["items"][0]["bbox"] == [0.1, 0.2, 0.4, 0.6]
    assert body["items"][0]["initial_evidence"]["model_version"] == "prod-20260901"
    assert body["next_cursor"]

    second_page = client.get(
        "/api/v1/agent/candidates",
        params={"limit": 10, "cursor": body["next_cursor"]},
        headers={"X-Service-Key": "agent-secret"},
    )
    assert [item["candidate_id"] for item in second_page.json()["items"]] == [second_id]
    assert second_page.json()["next_cursor"] is None
    invalid_cursor = client.get(
        "/api/v1/agent/candidates",
        params={"cursor": "not-a-cursor"},
        headers={"X-Service-Key": "agent-secret"},
    )
    assert invalid_cursor.status_code == 400


def test_review_request_is_idempotent_and_never_sets_annotation_label(tmp_path):
    app, client = build_client(tmp_path)
    candidate_id = seed_candidate(
        app,
        suffix="review",
        sha256="c" * 64,
        created_at=datetime.now(timezone.utc),
    )
    payload = {
        "cycle_id": "cycle-20260904-001",
        "candidate_id": candidate_id,
        "reason_code": "model_disagreement",
        "reason": "GPU and GPT disagree",
        "attempt_refs": ["attempt-gpu-1", "attempt-gpt-1"],
    }

    created = client.post(
        "/api/v1/agent/review-requests",
        json=payload,
        headers={"X-Service-Key": "agent-secret"},
    )
    repeated = client.post(
        "/api/v1/agent/review-requests",
        json=payload,
        headers={"X-Service-Key": "agent-secret"},
    )

    assert created.status_code == 201
    assert repeated.status_code == 200
    assert created.json()["review_request_id"] == repeated.json()["review_request_id"]
    assert created.json()["created"] is True
    assert repeated.json()["created"] is False
    with app.state.session_factory() as session:
        annotation = session.query(Annotation).filter(Annotation.candidate_id == candidate_id).one_or_none()
        assert annotation is None or annotation.label is None


def test_human_gold_exports_only_human_true_false_with_lineage(tmp_path):
    app, client = build_client(tmp_path)
    now = datetime.now(timezone.utc)
    true_id = seed_candidate(app, suffix="gold", sha256="d" * 64, created_at=now)
    pseudo_id = seed_candidate(app, suffix="pseudo", sha256="e" * 64, created_at=now + timedelta(seconds=1))
    uncertain_id = seed_candidate(app, suffix="uncertain", sha256="f" * 64, created_at=now + timedelta(seconds=2))

    with app.state.session_factory() as session:
        human = User(username="annotator", password_hash=hash_password("secret"))
        session.add(human)
        session.flush()
        session.add_all(
            [
                Annotation(
                    candidate_id=true_id,
                    label="true",
                    labeled_by=human.id,
                    labeled_at=now + timedelta(minutes=1),
                    revision=3,
                ),
                Annotation(candidate_id=pseudo_id, label="false", labeled_by=None, revision=1),
                Annotation(
                    candidate_id=uncertain_id,
                    label=None,
                    labeled_by=human.id,
                    labeled_at=now + timedelta(minutes=2),
                    revision=2,
                ),
            ]
        )
        session.commit()

    response = client.get(
        "/api/v1/agent/human-gold",
        headers={"X-Service-Key": "agent-secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["candidate_id"] == true_id
    assert item["label"] == "true"
    assert item["revision"] == 3
    assert item["labeled_by"]
    assert item["source_event_id"] == "event-gold"
    assert item["image"]["sha256"] == "d" * 64
    assert item["class_name"] == "车辆"
    assert item["source_class_id"] == 917
