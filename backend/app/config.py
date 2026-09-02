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
    minio_secure: bool = False
    model_service_base_url: str = ""
    model_service_api_key: str = ""

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
            minio_secure=os.getenv("MINIO_SECURE", "false").lower() in {"1", "true", "yes", "on"},
            model_service_base_url=os.getenv("MODEL_SERVICE_BASE_URL", cls.model_service_base_url),
            model_service_api_key=os.getenv("MODEL_SERVICE_API_KEY", cls.model_service_api_key),
        )
