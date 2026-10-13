from __future__ import annotations

import json
import io
import zipfile

from fastapi.testclient import TestClient

from app.main import create_app

from .test_annotation_flow import _login, _review_event
from .test_imports import MemoryObjectStore
from .test_model_runs import _settings


class InlineExportDispatcher:
    def __init__(self, runner):
        self.runner = runner

    def dispatch_export(self, job_id: str) -> None:
        self.runner(job_id)


def test_completed_model_run_can_export_provenance_rich_jsonl(tmp_path):
    object_store = MemoryObjectStore()
    object_store.put_bytes("review-images", "review/2026/09/02/session-001/original.jpg", b"image", "image/jpeg")
    app = create_app(_settings(tmp_path), object_store=object_store)
    app.state.export_dispatcher = InlineExportDispatcher(lambda job_id: app.state.run_export_job(job_id))
    client = TestClient(app)
    service_headers = {"X-Service-Key": "service-secret"}
    event_response = client.post("/api/v1/ingest/review-events", json=_review_event("event-export-001"), headers=service_headers)
    assert event_response.status_code == 201, event_response.text
    sample_id = event_response.json()["sample_id"]
    auth = _login(client)

    detail = client.get(f"/api/v1/samples/{sample_id}", headers=auth).json()
    run_id = detail["runs"][0]["model_run_id"]
    for candidate in detail["candidates"]:
        response = client.patch(
            f"/api/v1/candidates/{candidate['candidate_id']}/annotation",
            json={"label": "true", "note": "人工确认"},
            headers=auth,
        )
    assert response.status_code == 200, response.text

    evaluation = client.get(f"/api/v1/model-runs/{run_id}/evaluation", headers=auth)
    assert evaluation.status_code == 200, evaluation.text
    assert evaluation.json()["candidate_precision"] == 1.0
    assert evaluation.json()["recall_available"] is False

    exported = client.post(
        "/api/v1/exports",
        json={"run_id": run_id, "formats": ["jsonl"], "filters": {}},
        headers=auth,
    )
    assert exported.status_code == 202, exported.text
    assert exported.json()["status"] == "succeeded"

    export_id = exported.json()["export_id"]
    content = client.get(f"/api/v1/exports/{export_id}/content", headers=auth)
    assert content.status_code == 200, content.text
    rows = [json.loads(line) for line in content.text.splitlines() if line]
    assert len(rows) == 2
    assert {row["human_label"] for row in rows} == {"true"}
    assert {row["model_version"] for row in rows} == {"review-event"}
    assert rows[0]["sample_id"] == sample_id

    zipped_export = client.post(
        "/api/v1/exports",
        json={"run_id": run_id, "formats": ["yolo"], "filters": {}},
        headers=auth,
    )
    assert zipped_export.status_code == 202, zipped_export.text
    zipped_content = client.get(f"/api/v1/exports/{zipped_export.json()['export_id']}/download", headers=auth)
    assert zipped_content.status_code == 200, zipped_content.text
    with zipfile.ZipFile(io.BytesIO(zipped_content.content)) as archive:
        names = set(archive.namelist())
        assert f"images/{sample_id}.jpg" in names
        assert f"labels/{sample_id}.txt" in names
        assert "hard_negative.jsonl" in names
        assert len(archive.read(f"labels/{sample_id}.txt").decode().splitlines()) == 2
