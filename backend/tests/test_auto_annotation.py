from __future__ import annotations

import json

from sqlalchemy import select

from fastapi.testclient import TestClient

from app.adapters.codex_responses import CodexResponsesVisionAdapter
from app.config import Settings
from app.main import create_app
from app.models import AuditEvent

from .test_annotation_flow import _login, _review_event
from .test_imports import MemoryObjectStore


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
        service_api_key='REDACTED_FOR_GITHUB',
        admin_username="admin",
        admin_password='REDACTED_FOR_GITHUB',
        jwt_secret='REDACTED_FOR_GITHUB',
        minio_endpoint="",
        auto_annotation_api_key='REDACTED_FOR_GITHUB',
    )


def test_settings_reads_codex_responses_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("AUTO_ANNOTATION_API_KEY", "secret")
    monkeypatch.setenv("AUTO_ANNOTATION_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("AUTO_ANNOTATION_MODEL", "test-model")
    monkeypatch.setenv("AUTO_ANNOTATION_ENDPOINT", "/responses")
    monkeypatch.setenv("AUTO_ANNOTATION_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("AUTO_ANNOTATION_VERIFY_TLS", "false")

    settings = Settings.from_env()

    assert settings.auto_annotation_api_key == "secret"
    assert settings.auto_annotation_base_url == "https://example.invalid"
    assert settings.auto_annotation_model == "test-model"
    assert settings.auto_annotation_endpoint == "/responses"
    assert settings.auto_annotation_timeout_seconds == 17
    assert settings.auto_annotation_verify_tls is False


def test_codex_responses_adapter_sends_one_image_request_and_parses_votes(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps({
                        "decisions": [{
                            "target_index": 0,
                            "votes": [
                                {"role": "target_presence_fit", "label": "true", "confidence": 0.8, "reason": "car"},
                                {"role": "class_scene", "label": "true", "confidence": 0.7, "reason": "box"},
                                {"role": "adversarial", "label": "false", "confidence": 0.9, "reason": "none"},
                            ],
                            "consensus": "true",
                        }],
                    })}],
                }],
            }

    class Client:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, *, json, headers):
            captured.update(url=url, json=json, headers=headers)
            return Response()

    monkeypatch.setattr("app.adapters.codex_responses.httpx.Client", Client)
    store = MemoryObjectStore()
    store.put_bytes("images", "raw/car.jpg", b"raw-image", "image/jpeg")
    adapter = CodexResponsesVisionAdapter(
        object_store=store,
        api_key='REDACTED_FOR_GITHUB',
        base_url="https://codex.example",
        endpoint="/v1/responses",
        model="gpt-test",
        reasoning_effort="low",
    )

    result = adapter.infer(
        {"bucket": "images", "object_key": "raw/car.jpg"},
        {"candidates": [{"target_index": 0, "class_name": "car", "bbox": [0.1, 0.1, 0.2, 0.2]}]},
    )

    assert captured["url"] == "https://codex.example/v1/responses"
    assert captured["headers"] == {"Authorization": "Bearer secret", "Content-Type": "application/json"}
    assert captured["json"]["model"] == "gpt-test"
    assert captured["json"]["reasoning"] == {"effort": "low"}
    assert captured["json"]["input"][0]["content"][1]["type"] == "input_image"
    assert result["decisions"][0]["votes"][0]["label"] == "true"


class FakeAutoAnnotationAdapter:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []

    def infer(self, image, options):
        self.calls.append((image, options))
        return {"decisions": self.decisions}


class InlineDispatcher:
    def __init__(self, runner):
        self.runner = runner

    def dispatch(self, job_id: str) -> None:
        self.runner(job_id)


def test_auto_annotation_requires_two_confident_votes_and_is_inline(tmp_path):
    adapter = FakeAutoAnnotationAdapter([
        {
            "target_index": 0,
            "votes": [
                {"role": "target_presence_fit", "label": "true", "confidence": 0.8, "reason": "car"},
                {"role": "class_scene", "label": "true", "confidence": 0.7, "reason": "box"},
                {"role": "adversarial", "label": "false", "confidence": 0.9, "reason": "background"},
            ],
            "consensus": "true",
        },
        {
            "target_index": 1,
            "votes": [
                {"role": "target_presence_fit", "label": "true", "confidence": 0.8, "reason": "car"},
                {"role": "class_scene", "label": "false", "confidence": 0.8, "reason": "box"},
                {"role": "adversarial", "label": "uncertain", "confidence": 0.9, "reason": "occluded"},
            ],
            "consensus": "uncertain",
        },
    ])
    store = MemoryObjectStore()
    event = _review_event("auto-annotation-001")
    store.put_bytes(event["image"]["bucket"], event["image"]["object_key"], b"image", "image/jpeg")
    app = create_app(
        _settings(tmp_path),
        object_store=store,
        auto_annotation_adapter=adapter,
    )
    app.state.auto_annotation_dispatcher = InlineDispatcher(app.state.run_auto_annotation_job)
    client = TestClient(app)
    ingest = client.post("/api/v1/ingest/review-events", json=event, headers={"X-Service-Key": "service-secret"})
    assert ingest.status_code == 201, ingest.text
    sample_id = ingest.json()["sample_id"]

    response = client.post("/api/v1/auto-annotations", json={"sample_ids": [sample_id]}, headers=_login(client))

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["total"] == body["completed"] == 1
    assert body["failed"] == 0
    assert len(adapter.calls) == 1
    sample = client.get(f"/api/v1/samples/{sample_id}", headers=_login(client)).json()
    assert sample["candidates"][0]["annotation"]["label"] == "true"
    assert sample["candidates"][1]["annotation"]["label"] is None


def test_auto_annotation_applies_shared_rooftop_and_red_overlay_policy_and_audits_it(tmp_path):
    def votes(label: str):
        return [
            {"role": "target_presence_fit", "label": label, "confidence": 0.9, "reason": "visual evidence"},
            {"role": "class_scene", "label": label, "confidence": 0.9, "reason": "class evidence"},
            {"role": "adversarial", "label": label, "confidence": 0.9, "reason": "adversarial check"},
        ]

    adapter = FakeAutoAnnotationAdapter([
        {
            "target_index": 0,
            "votes": votes("uncertain"),
            "consensus": "uncertain",
            "scene": {"label": "ground", "confidence": 0.95, "reason": "road"},
            "evidence": {
                "red_overlay_occlusion": True,
                "overlay_obscures": True,
                "background_only": False,
                "target_visible": "occluded",
                "occlusion_type": "red_overlay",
            },
        },
        {
            "target_index": 1,
            "votes": votes("true"),
            "consensus": "true",
            "scene": {"label": "rooftop", "confidence": 0.95, "reason": "roof"},
            "evidence": {
                "red_overlay_occlusion": True,
                "overlay_obscures": True,
                "background_only": False,
                "target_visible": "occluded",
                "occlusion_type": "red_overlay",
            },
        },
    ])
    store = MemoryObjectStore()
    event = _review_event("auto-annotation-policy-001")
    store.put_bytes(event["image"]["bucket"], event["image"]["object_key"], b"image", "image/jpeg")
    app = create_app(_settings(tmp_path), object_store=store, auto_annotation_adapter=adapter)
    app.state.auto_annotation_dispatcher = InlineDispatcher(app.state.run_auto_annotation_job)
    client = TestClient(app)
    ingest = client.post(
        "/api/v1/ingest/review-events",
        json=event,
        headers={"X-Service-Key": "service-secret"},
    )
    assert ingest.status_code == 201, ingest.text
    sample_id = ingest.json()["sample_id"]

    response = client.post(
        "/api/v1/auto-annotations",
        json={"sample_ids": [sample_id]},
        headers=_login(client),
    )

    assert response.status_code == 202, response.text
    sample = client.get(f"/api/v1/samples/{sample_id}", headers=_login(client)).json()
    assert sample["candidates"][0]["annotation"]["label"] == "true"
    assert sample["candidates"][1]["annotation"]["label"] == "false"
    with app.state.session_factory() as session:
        audits = session.scalars(
            select(AuditEvent).where(AuditEvent.action == "auto_annotation.applied")
        ).all()
    assert len(audits) == 2
    payloads = [json.loads(item.payload_json) for item in audits]
    assert {payload["policy"]["reason_code"] for payload in payloads} == {
        "red_overlay_occlusion_exempt",
        "rooftop_vehicle_rule",
    }
    assert all("votes" in payload and "model_label" in payload and "local_decision" in payload for payload in payloads)
