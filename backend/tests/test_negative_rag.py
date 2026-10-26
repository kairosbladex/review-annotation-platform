from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import Annotation, Candidate, ModelRun, Sample, User


class StubObjectStore:
    pass


def test_negative_rag_returns_only_human_confirmed_false_samples(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'negative-rag.sqlite3'}",
        service_api_key="service-secret",
        admin_password="admin-password",
        jwt_secret="test-jwt-secret",
        minio_auto_create_bucket=False,
    )
    app = create_app(settings, object_store=StubObjectStore())
    with app.state.session_factory() as session:
        sample = Sample(source_type="review", metadata_json='{"scene_tags":["water"]}')
        session.add(sample)
        session.flush()
        run = ModelRun(model_name="yoloe", model_version="test", source_type="review")
        session.add(run)
        session.flush()
        candidate = Candidate(
            sample_id=sample.id,
            model_run_id=run.id,
            target_index=0,
            class_name="车辆",
            x1=.1,
            y1=.1,
            x2=.3,
            y2=.3,
            reasoning="反光水面形成车状高光",
        )
        session.add(candidate)
        session.flush()
        user = session.query(User).filter_by(username="admin").one()
        session.add(Annotation(
            candidate_id=candidate.id,
            label="false",
            note="反光水面和亮斑拼成车状轮廓",
            labeled_by=user.id,
            labeled_at=datetime.now(timezone.utc),
            revision=1,
        ))
        session.commit()
    client = TestClient(app)
    response = client.get(
        "/api/v1/negative-rag",
        params=[("category", "vehicle"), ("scene_tags", "water"), ("top_k", "3")],
        headers={"X-Service-Key": "service-secret"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["items"][0]["description"].startswith("反光水面")
    assert "strictly" not in payload["negative_rag_text"]
