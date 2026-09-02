from __future__ import annotations

import sys
from types import SimpleNamespace

from app.adapters.object_store import MinioObjectStore


def test_presigned_url_uses_browser_reachable_endpoint(monkeypatch):
    class FakeMinio:
        def __init__(self, endpoint, **_kwargs):
            self.endpoint = endpoint

        def presigned_get_object(self, bucket, object_key, *, expires):
            return f"http://{self.endpoint}/{bucket}/{object_key}?expires={expires.total_seconds()}"

    monkeypatch.setitem(sys.modules, "minio", SimpleNamespace(Minio=FakeMinio))
    store = MinioObjectStore(
        endpoint="minio:9000",
        public_endpoint="127.0.0.1:9000",
        access_key="access",
        secret_key="secret",
        secure=False,
    )

    assert store.presigned_get_url("review-annotation", "raw/sample.png").startswith("http://127.0.0.1:9000/")
