from __future__ import annotations

import sys
from types import SimpleNamespace

from app.adapters.object_store import FallbackObjectStore, HttpPrefixObjectStore, MinioObjectStore, NullObjectStore
from app.config import Settings


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
        access_key='REDACTED_FOR_GITHUB',
        secret_key='REDACTED_FOR_GITHUB',
        secure=False,
    )

    assert store.presigned_get_url("review-annotation", "raw/sample.png").startswith("http://127.0.0.1:9000/")


def test_production_store_does_not_create_missing_bucket(monkeypatch):
    class FakeMinio:
        def __init__(self, *_args, **_kwargs):
            self.make_bucket_called = False

        def bucket_exists(self, _bucket):
            return False

        def make_bucket(self, _bucket):
            self.make_bucket_called = True

    monkeypatch.setitem(sys.modules, "minio", SimpleNamespace(Minio=FakeMinio))
    store = MinioObjectStore(
        endpoint="minio.example.internal:9000",
        public_endpoint="minio.example.internal:9000",
        access_key='REDACTED_FOR_GITHUB',
        secret_key='REDACTED_FOR_GITHUB',
        secure=True,
        auto_create_bucket=False,
    )

    try:
        store.ensure_bucket("review-annotation")
    except RuntimeError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing production bucket must fail")
    assert not store.client.make_bucket_called


def test_required_object_store_does_not_silently_downgrade_when_config_is_missing():
    from app.adapters.object_store import build_object_store

    settings = Settings(
        minio_endpoint="",
        minio_access_key="",
        minio_secret_key="",
        object_store_required=True,
    )

    try:
        build_object_store(settings)
    except RuntimeError as exc:
        assert "required" in str(exc)
    else:
        raise AssertionError("required object storage must fail when configuration is missing")


def test_fallback_store_uses_http_prefix_when_minio_object_is_absent():
    http_store = HttpPrefixObjectStore(base_url="https://lxtest.zhuoyilingxi.com:8088/minio-src")
    store = FallbackObjectStore(NullObjectStore(), http_store)

    assert store.exists("minio-src", "zm-zq-test/a.jpg") is True
    assert store.presigned_get_url("minio-src", "zm-zq-test/a.jpg") == (
        "https://lxtest.zhuoyilingxi.com:8088/minio-src/zm-zq-test/a.jpg"
    )
    assert store.exists("review-annotation", "raw/a.jpg") is False


def test_fallback_store_does_not_probe_primary_for_http_source_bucket():
    class FailingPrimary(NullObjectStore):
        def exists(self, _bucket, _object_key):
            raise AssertionError("source bucket must not probe the unavailable primary store")

    http_store = HttpPrefixObjectStore(base_url="https://lxtest.zhuoyilingxi.com:8088/minio-src")
    store = FallbackObjectStore(FailingPrimary(), http_store)

    assert store.presigned_get_url("minio-src", "zm-zq-test/a.jpg") == (
        "https://lxtest.zhuoyilingxi.com:8088/minio-src/zm-zq-test/a.jpg"
    )
    assert store.exists("minio-src", "zm-zq-test/a.jpg") is True
