from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .test_annotation_flow import _login, _review_event


@dataclass
class FakePrediction:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class FakeModelAdapter:
    def infer(self, image, options):
        return {
            "predictions": [
                FakePrediction(2, "车", 0.91, [0.2, 0.2, 0.4, 0.45]),
                FakePrediction(2, "车", 0.37, [0.6, 0.3, 0.75, 0.5]),
            ],
            "artifact": None,
        }


class InlineDispatcher:
    def __init__(self, runner):
        self.runner = runner

    def dispatch(self, job_id: str) -> None:
        self.runner(job_id)


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
        service_api_key="service-secret",
        admin_username="admin",
        admin_password="admin-password",
        jwt_secret="a-test-jwt-secret-that-is-long-enough",
        minio_endpoint="",
    )


def test_model_run_is_versioned_and_creates_a_new_annotation_task(tmp_path):
    dispatcher_holder = {}

    def build_app(settings):
        app = create_app(settings, model_adapter=FakeModelAdapter(), task_dispatcher=dispatcher_holder.get("dispatcher"))
        dispatcher_holder["dispatcher"] = InlineDispatcher(lambda job_id: app.state.run_model_job(job_id))
        app.state.task_dispatcher = dispatcher_holder["dispatcher"]
        return app

    app = build_app(_settings(tmp_path))
    client = TestClient(app)
    event_response = client.post(
        "/api/v1/ingest/review-events",
        json=_review_event("event-model-run-001"),
        headers={"X-Service-Key": "service-secret"},
    )
    assert event_response.status_code == 201, event_response.text
    sample_id = event_response.json()["sample_id"]
    auth = _login(client)

    profile = client.post(
        "/api/v1/model-profiles",
        json={
            "profile_key": "fake-v2",
            "name": "Fake v2",
            "model_name": "fake-detector",
            "model_version": "v2",
            "adapter_type": "fake",
            "config": {},
        },
        headers=auth,
    )
    assert profile.status_code == 201, profile.text

    run_response = client.post(
        "/api/v1/model-runs",
        json={
            "sample_ids": [sample_id],
            "model_profile_id": profile.json()["model_profile_id"],
            "options": {"confidence_threshold": 0.25},
        },
        headers=auth,
    )
    assert run_response.status_code == 202, run_response.text
    run_body = run_response.json()
    assert run_body["status"] == "succeeded"
    assert run_body["model_version"] == "v2"

    run_detail = client.get(f"/api/v1/model-runs/{run_body['run_id']}", headers=auth)
    assert run_detail.status_code == 200, run_detail.text
    assert run_detail.json()["candidate_count"] == 2

    sample_detail = client.get(f"/api/v1/samples/{sample_id}", headers=auth)
    assert sample_detail.status_code == 200, sample_detail.text
    sample = sample_detail.json()
    assert len(sample["runs"]) == 2
    assert {run["model_run"]["model_version"] for run in sample["runs"]} == {"review-event", "v2"}


def test_annotation_counts_are_scoped_to_sample_and_model_run(tmp_path):
    app = create_app(_settings(tmp_path), model_adapter=FakeModelAdapter())
    app.state.task_dispatcher = InlineDispatcher(lambda job_id: app.state.run_model_job(job_id))
    client = TestClient(app)
    headers = {"X-Service-Key": "service-secret"}
    first = client.post("/api/v1/ingest/review-events", json=_review_event("event-scope-001"), headers=headers)
    second = client.post("/api/v1/ingest/review-events", json=_review_event("event-scope-002"), headers=headers)
    assert first.status_code == 201 and second.status_code == 201
    auth = _login(client)
    profile = client.post("/api/v1/model-profiles", json={"profile_key": "scope-v2", "name": "Scope v2", "model_name": "fake", "model_version": "v2", "adapter_type": "fake", "config": {}}, headers=auth)
    run = client.post("/api/v1/model-runs", json={"sample_ids": [first.json()["sample_id"], second.json()["sample_id"]], "model_profile_id": profile.json()["model_profile_id"], "options": {}}, headers=auth)
    assert run.status_code == 202, run.text

    first_detail = client.get(f"/api/v1/samples/{first.json()['sample_id']}", headers=auth).json()
    second_detail = client.get(f"/api/v1/samples/{second.json()['sample_id']}", headers=auth).json()
    for candidate in first_detail["candidates"]:
        response = client.patch(f"/api/v1/candidates/{candidate['candidate_id']}/annotation", json={"label": "true"}, headers=auth)
        assert response.status_code == 200, response.text
    assert client.get(f"/api/v1/samples/{first.json()['sample_id']}", headers=auth).json()["status"] == "completed"
    assert client.get(f"/api/v1/samples/{second.json()['sample_id']}", headers=auth).json()["status"] == "pending"
