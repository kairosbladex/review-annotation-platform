from __future__ import annotations


def test_settings_reads_production_adapter_and_minio_controls(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("MODEL_ADAPTER_TYPE", "legacy_review_agent")
    monkeypatch.setenv("MINIO_REGION", "cn-test-1")
    monkeypatch.setenv("MINIO_AUTO_CREATE_BUCKET", "false")
    monkeypatch.setenv("OBJECT_STORE_REQUIRED", "true")
    monkeypatch.setenv("MODEL_SERVICE_DETECT_ENDPOINT", "/detect")
    monkeypatch.setenv("MODEL_SERVICE_REVIEW_ENDPOINT", "/review")
    monkeypatch.setenv("MODEL_SERVICE_VERIFY_TLS", "false")
    monkeypatch.setenv("SOURCE_IMAGE_BASE_URL", "https://REDACTED_TEST_HOST.invalid:8088/minio-src/")
    monkeypatch.setenv("SOURCE_IMAGE_BUCKET", "minio-src")

    settings = Settings.from_env()

    assert settings.model_adapter_type == "legacy_review_agent"
    assert settings.minio_region == "cn-test-1"
    assert settings.minio_auto_create_bucket is False
    assert settings.object_store_required is True
    assert settings.model_service_detect_endpoint == "/detect"
    assert settings.model_service_review_endpoint == "/review"
    assert settings.model_service_verify_tls is False
    assert settings.source_image_base_url == "https://REDACTED_TEST_HOST.invalid:8088/minio-src"
    assert settings.source_image_bucket == "minio-src"
