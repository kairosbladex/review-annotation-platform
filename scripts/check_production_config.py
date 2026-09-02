from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse


REQUIRED = (
    "DATABASE_URL",
    "REDIS_URL",
    "ADMIN_PASSWORD",
    "JWT_SECRET",
    "SERVICE_API_KEY",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_BUCKET",
    "MINIO_PUBLIC_ENDPOINT",
    "MODEL_SERVICE_BASE_URL",
    "MODEL_SERVICE_API_KEY",
)
PLACEHOLDER_MARKERS = ("REPLACE_ME", "change-me", ".example.")


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _is_placeholder(value: str) -> bool:
    return any(marker.lower() in value.lower() for marker in PLACEHOLDER_MARKERS)


def validate() -> list[str]:
    errors = []
    missing = [name for name in REQUIRED if not os.getenv(name, "").strip()]
    if missing:
        errors.append("missing required variables: " + ", ".join(missing))

    placeholders = [name for name in REQUIRED if _is_placeholder(os.getenv(name, ""))]
    if placeholders:
        errors.append("replace example values for: " + ", ".join(placeholders))

    if len(os.getenv("JWT_SECRET", "")) < 32:
        errors.append("JWT_SECRET must contain at least 32 characters")
    if os.getenv("MINIO_SECURE", "").lower() not in {"1", "true", "yes", "on"}:
        errors.append("MINIO_SECURE must be true for production")
    if os.getenv("MINIO_AUTO_CREATE_BUCKET", "true").lower() in {"1", "true", "yes", "on"}:
        errors.append("MINIO_AUTO_CREATE_BUCKET must be false for production")
    if os.getenv("OBJECT_STORE_REQUIRED", "false").lower() not in {"1", "true", "yes", "on"}:
        errors.append("OBJECT_STORE_REQUIRED must be true for production")
    if os.getenv("MODEL_ADAPTER_TYPE", "") != "legacy_review_agent":
        errors.append("MODEL_ADAPTER_TYPE must be legacy_review_agent")

    for name in ("MINIO_ENDPOINT", "MINIO_PUBLIC_ENDPOINT"):
        value = os.getenv(name, "")
        if "://" in value:
            errors.append(f"{name} must be host:port without a URL scheme")
    service_url = os.getenv("MODEL_SERVICE_BASE_URL", "")
    if service_url and urlparse(service_url).scheme not in {"http", "https"}:
        errors.append("MODEL_SERVICE_BASE_URL must start with http:// or https://")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production platform configuration without printing secrets")
    parser.add_argument("--env-file", type=Path, help="dotenv file to load before validation")
    args = parser.parse_args()
    if args.env_file:
        _load_env_file(args.env_file)

    errors = validate()
    if errors:
        print("production config invalid:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("production config OK: strict MinIO, legacy review adapter, and required secrets are enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
