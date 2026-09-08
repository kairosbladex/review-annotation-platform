from __future__ import annotations

from fastapi.testclient import TestClient

from app.adapters.object_store import HttpPrefixObjectStore, build_object_store
from app.config import Settings
from app.main import create_app
from app.services.annex_import import build_review_event_from_annex, normalize_imgxy_targets


SAMPLE_IMGXY = [
    {
        "frame_width": 960,
        "isHotspotArea": True,
        "confidence": 0.7853084802627563,
        "x": 0,
        "width": 10,
        "y": 312,
        "frame_height": 720,
        "class_name": "车",
        "height": 38,
    },
    {
        "frame_width": 960,
        "isHotspotArea": True,
        "confidence": 0.8519556522369385,
        "x": 202,
        "width": 31,
        "y": 60,
        "frame_height": 720,
        "class_name": "车",
        "height": 21,
    },
]


def test_imgxy_pixel_boxes_normalize_to_unit_xyxy():
    targets = normalize_imgxy_targets(SAMPLE_IMGXY)

    assert len(targets) == 2
    first = targets[0]
    assert first["class_name"] == "车"
    assert first["confidence"] == 0.7853084802627563
    assert first["raw_decision"] == "uncertain"
    assert first["bbox"][0] == 0.0
    assert first["bbox"][1] == 312 / 720
    assert first["bbox"][2] == 10 / 960
    assert first["bbox"][3] == 350 / 720


def test_imgxy_skips_zero_size_and_clamps_to_frame():
    targets = normalize_imgxy_targets(
        [
            {"frame_width": 100, "frame_height": 100, "x": 10, "y": 10, "width": 0, "height": 8, "class_name": "车"},
            {"frame_width": 100, "frame_height": 100, "x": 90, "y": 90, "width": 30, "height": 30, "class_name": "车", "confidence": 0.4},
        ]
    )

    assert len(targets) == 1
    assert targets[0]["bbox"] == [0.9, 0.9, 1.0, 1.0]


def test_http_prefix_store_joins_public_image_url():
    store = HttpPrefixObjectStore(base_url="https://lxtest.zhuoyilingxi.com:8088/minio-src")

    assert store.exists("minio-src", "/zm-zq-test/ktdTest/a.jpg") is True
    assert store.exists("review-annotation", "raw/a.jpg") is False
    assert store.presigned_get_url("minio-src", "/zm-zq-test/ktdTest/a.jpg") == (
        "https://lxtest.zhuoyilingxi.com:8088/minio-src/zm-zq-test/ktdTest/a.jpg"
    )


def test_build_object_store_uses_http_prefix_when_minio_is_absent(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    settings = Settings(
        minio_endpoint="",
        minio_access_key="",
        minio_secret_key="",
        source_image_base_url="https://lxtest.zhuoyilingxi.com:8088/minio-src",
    )
    store = build_object_store(settings)
    assert store.presigned_get_url("minio-src", "zm-zq-test/a.jpg") == (
        "https://lxtest.zhuoyilingxi.com:8088/minio-src/zm-zq-test/a.jpg"
    )


def test_annex_row_ingests_as_ready_sample_with_public_image_url(tmp_path):
    event = build_review_event_from_annex(
        {
            "id": "69c83b9359ed4e4396200ff3e832d82a",
            "file_path": "/zm-zq-test/ktdTest/aiEventImg/20260826/69c83b9359ed4e4396200ff3e832d82a.jpg",
            "imgxy": SAMPLE_IMGXY,
            "create_time": None,
        }
    )
    assert event.event_id == "ktd-annex-69c83b9359ed4e4396200ff3e832d82a"
    assert event.image.bucket == "minio-src"
    assert event.image.object_key == "zm-zq-test/ktdTest/aiEventImg/20260826/69c83b9359ed4e4396200ff3e832d82a.jpg"
    assert event.targets[0].raw_decision == "uncertain"

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
        service_api_key="service-secret",
        admin_username="admin",
        admin_password="admin-password",
        jwt_secret="test-jwt-secret-with-at-least-32-bytes",
        minio_endpoint="",
        source_image_base_url="https://lxtest.zhuoyilingxi.com:8088/minio-src",
    )
    client = TestClient(create_app(settings, object_store=build_object_store(settings)))
    created = client.post(
        "/api/v1/ingest/review-events",
        json=event.model_dump(mode="json"),
        headers={"X-Service-Key": "service-secret"},
    )
    assert created.status_code == 201, created.text
    sample_id = created.json()["sample_id"]

    login = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin-password"})
    auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
    sample = client.get(f"/api/v1/samples/{sample_id}", headers=auth)
    assert sample.status_code == 200, sample.text
    body = sample.json()
    assert body["asset_status"] == "ready"
    assert body["counts"]["total"] == 2
    original = next(asset for asset in body["assets"] if asset["kind"] == "original")
    url = client.get(f"/api/v1/assets/{original['asset_id']}/url", headers=auth)
    assert url.status_code == 200, url.text
    assert url.json()["url"] == (
        "https://lxtest.zhuoyilingxi.com:8088/minio-src/"
        "zm-zq-test/ktdTest/aiEventImg/20260826/69c83b9359ed4e4396200ff3e832d82a.jpg"
    )
