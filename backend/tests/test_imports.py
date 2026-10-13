from __future__ import annotations

from app.adapters.object_store import ObjectStore
from app.main import create_app
from fastapi.testclient import TestClient

from .test_annotation_flow import _login
from .test_model_runs import _settings


class MemoryObjectStore(ObjectStore):
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def exists(self, bucket: str, object_key: str) -> bool:
        return (bucket, object_key) in self.objects

    def presigned_get_url(self, bucket: str, object_key: str, expires_seconds: int = 900) -> str | None:
        return f"memory://{bucket}/{object_key}"

    def put_bytes(self, bucket: str, object_key: str, data: bytes, content_type: str) -> None:
        self.objects[(bucket, object_key)] = data

    def get_bytes(self, bucket: str, object_key: str) -> bytes:
        return self.objects[(bucket, object_key)]

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        return [key for current_bucket, key in self.objects if current_bucket == bucket and key.startswith(prefix)]


def test_minio_prefix_and_manual_upload_create_deduplicated_samples(tmp_path):
    object_store = MemoryObjectStore()
    object_store.put_bytes("review-images", "incoming/one.jpg", b"image-1", "image/jpeg")
    object_store.put_bytes("review-images", "incoming/ignore.txt", b"ignore", "text/plain")
    app = create_app(_settings(tmp_path), object_store=object_store)
    client = TestClient(app)
    auth = _login(client)

    imported = client.post(
        "/api/v1/imports",
        json={"source_type": "minio", "bucket": "review-images", "prefix": "incoming/"},
        headers=auth,
    )
    assert imported.status_code == 201, imported.text
    assert imported.json()["imported"] == 1

    duplicate = client.post(
        "/api/v1/imports",
        json={"source_type": "minio", "bucket": "review-images", "prefix": "incoming/"},
        headers=auth,
    )
    assert duplicate.status_code == 201, duplicate.text
    assert duplicate.json()["imported"] == 0
    assert duplicate.json()["skipped"] == 1

    uploaded = client.post(
        "/api/v1/imports/uploads",
        files=[("files", ("manual.png", b"png-bytes", "image/png"))],
        headers=auth,
    )
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["imported"] == 1

    samples = client.get("/api/v1/samples", headers=auth)
    assert samples.status_code == 200, samples.text
    assert samples.json()["pagination"]["total"] == 2
