from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Local development stays runnable without a database driver or Docker.
    # Compose/.env overrides this with PostgreSQL for every deployed instance.
    database_url: str = "sqlite:///./data/dev.sqlite3"
    redis_url: str = "redis://redis:6379/0"
    service_api_key: str = "change-me-service-key"
    admin_username: str = "admin"
    admin_password: str = "change-me-admin-password"
    jwt_secret: str = "change-me-jwt-secret"
    jwt_exp_minutes: int = 720
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "review-annotation"
    minio_public_endpoint: str = ""
    minio_secure: bool = False
    minio_region: str = "us-east-1"
    minio_auto_create_bucket: bool = True
    object_store_required: bool = False
    model_adapter_type: str = "http"
    model_service_base_url: str = ""
    model_service_api_key: str = ""
    model_service_detect_endpoint: str = "/api/v1/review-agent/yolo-detect"
    model_service_review_endpoint: str = "/api/v1/review-agent/review"
    model_service_verify_tls: bool = True
    # The auto-annotation path talks to an OpenAI Responses-compatible vision
    # endpoint.  Its credential intentionally has no non-empty default and is
    # never copied into a model profile or job payload.
    auto_annotation_api_key: str = ""
    auto_annotation_base_url: str = "https://sub.butterfly2sea.us"
    auto_annotation_model: str = "gpt-5.6-luna"
    auto_annotation_endpoint: str = "/responses"
    auto_annotation_reasoning_effort: str = "low"
    auto_annotation_timeout_seconds: int = 120
    auto_annotation_verify_tls: bool = True
    # Company Codex gateways may reject raw HTTP callers and require the
    # official CLI client headers/handshake. Keep that path enabled by default
    # while retaining the direct Responses adapter for generic gateways.
    auto_annotation_use_codex_cli: bool = True
    auto_annotation_codex_binary: str = "codex"
    auto_annotation_actor_authorization: str = "local-image-extension"
    source_image_base_url: str = ""
    source_image_bucket: str = "minio-src"
    source_image_verify_tls: bool = True
    v2_queue_map_path: str = "data/prompt-eval/gold-expansion-20260904-v2/index.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            redis_url=os.getenv("REDIS_URL", cls.redis_url),
            service_api_key=os.getenv("SERVICE_API_KEY", cls.service_api_key),
            admin_username=os.getenv("ADMIN_USERNAME", cls.admin_username),
            admin_password=os.getenv("ADMIN_PASSWORD", cls.admin_password),
            jwt_secret=os.getenv("JWT_SECRET", cls.jwt_secret),
            jwt_exp_minutes=int(os.getenv("JWT_EXP_MINUTES", str(cls.jwt_exp_minutes))),
            minio_endpoint=os.getenv("MINIO_ENDPOINT", cls.minio_endpoint),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY", cls.minio_access_key),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY", cls.minio_secret_key),
            minio_bucket=os.getenv("MINIO_BUCKET", cls.minio_bucket),
            minio_public_endpoint=os.getenv("MINIO_PUBLIC_ENDPOINT", cls.minio_public_endpoint),
            minio_secure=os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes", "on"},
            minio_region=os.getenv("MINIO_REGION", cls.minio_region),
            minio_auto_create_bucket=os.getenv("MINIO_AUTO_CREATE_BUCKET", "true").lower() in {"1", "true", "yes", "on"},
            object_store_required=os.getenv("OBJECT_STORE_REQUIRED", "false").lower() in {"1", "true", "yes", "on"},
            model_adapter_type=os.getenv("MODEL_ADAPTER_TYPE", cls.model_adapter_type),
            model_service_base_url=os.getenv("MODEL_SERVICE_BASE_URL", cls.model_service_base_url),
            model_service_api_key=os.getenv("MODEL_SERVICE_API_KEY", cls.model_service_api_key),
            model_service_detect_endpoint=os.getenv("MODEL_SERVICE_DETECT_ENDPOINT", cls.model_service_detect_endpoint),
            model_service_review_endpoint=os.getenv("MODEL_SERVICE_REVIEW_ENDPOINT", cls.model_service_review_endpoint),
            model_service_verify_tls=os.getenv("MODEL_SERVICE_VERIFY_TLS", "true").lower() in {"1", "true", "yes", "on"},
            auto_annotation_api_key=os.getenv("AUTO_ANNOTATION_API_KEY", ""),
            auto_annotation_base_url=os.getenv("AUTO_ANNOTATION_BASE_URL", cls.auto_annotation_base_url).rstrip("/"),
            auto_annotation_model=os.getenv("AUTO_ANNOTATION_MODEL", cls.auto_annotation_model),
            auto_annotation_endpoint=os.getenv("AUTO_ANNOTATION_ENDPOINT", cls.auto_annotation_endpoint),
            auto_annotation_reasoning_effort=os.getenv("AUTO_ANNOTATION_REASONING_EFFORT", cls.auto_annotation_reasoning_effort),
            auto_annotation_timeout_seconds=int(os.getenv("AUTO_ANNOTATION_TIMEOUT_SECONDS", str(cls.auto_annotation_timeout_seconds))),
            auto_annotation_verify_tls=os.getenv("AUTO_ANNOTATION_VERIFY_TLS", "true").lower() in {"1", "true", "yes", "on"},
            auto_annotation_use_codex_cli=os.getenv("AUTO_ANNOTATION_USE_CODEX_CLI", "true").lower() in {"1", "true", "yes", "on"},
            auto_annotation_codex_binary=os.getenv("AUTO_ANNOTATION_CODEX_BINARY", cls.auto_annotation_codex_binary),
            auto_annotation_actor_authorization=os.getenv("AUTO_ANNOTATION_ACTOR_AUTHORIZATION", cls.auto_annotation_actor_authorization),
            source_image_base_url=os.getenv("SOURCE_IMAGE_BASE_URL", cls.source_image_base_url).rstrip("/"),
            source_image_bucket=os.getenv("SOURCE_IMAGE_BUCKET", cls.source_image_bucket),
            source_image_verify_tls=os.getenv("SOURCE_IMAGE_VERIFY_TLS", "true").lower() in {"1", "true", "yes", "on"},
            v2_queue_map_path=os.getenv("V2_QUEUE_MAP_PATH", cls.v2_queue_map_path),
        )
