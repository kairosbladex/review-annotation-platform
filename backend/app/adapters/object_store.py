from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO


class ObjectStore:
    def exists(self, bucket: str, object_key: str) -> bool:
        raise NotImplementedError

    def presigned_get_url(self, bucket: str, object_key: str, expires_seconds: int = 900) -> str | None:
        raise NotImplementedError

    def put_bytes(self, bucket: str, object_key: str, data: bytes, content_type: str) -> None:
        raise NotImplementedError

    def get_bytes(self, bucket: str, object_key: str) -> bytes:
        raise NotImplementedError

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        raise NotImplementedError


def public_image_url(base_url: str, object_key: str) -> str:
    return f"{base_url.rstrip('/')}/{str(object_key).lstrip('/')}"


class NullObjectStore(ObjectStore):
    """No-op adapter used for tests and local API-only development."""

    def exists(self, bucket: str, object_key: str) -> bool:
        return False

    def presigned_get_url(self, bucket: str, object_key: str, expires_seconds: int = 900) -> str | None:
        return None

    def put_bytes(self, bucket: str, object_key: str, data: bytes, content_type: str) -> None:
        raise RuntimeError("object storage is not configured")

    def get_bytes(self, bucket: str, object_key: str) -> bytes:
        raise RuntimeError("object storage is not configured")

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        return []


class MinioObjectStore(ObjectStore):
    def __init__(self, *, endpoint: str, access_key: str, secret_key: str, secure: bool, public_endpoint: str = "", region: str = "us-east-1", auto_create_bucket: bool = True):
        from minio import Minio

        self.auto_create_bucket = auto_create_bucket
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure, region=region or None)
        self.presign_client = self.client
        if public_endpoint and public_endpoint != endpoint:
            # The browser-facing endpoint may be unreachable from this
            # container. Pin MinIO's default region so presigning does not
            # need a bucket-location request through that public endpoint.
            self.presign_client = Minio(
                public_endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                region=region or "us-east-1",
            )

    def ensure_bucket(self, bucket: str) -> None:
        if self.client.bucket_exists(bucket):
            return
        if not self.auto_create_bucket:
            raise RuntimeError(f"MinIO bucket does not exist: {bucket}")
        self.client.make_bucket(bucket)

    def exists(self, bucket: str, object_key: str) -> bool:
        try:
            self.client.stat_object(bucket, object_key)
            return True
        except Exception:
            return False

    def presigned_get_url(self, bucket: str, object_key: str, expires_seconds: int = 900) -> str | None:
        from datetime import timedelta

        try:
            return self.presign_client.presigned_get_object(bucket, object_key, expires=timedelta(seconds=expires_seconds))
        except Exception:
            return None

    def put_bytes(self, bucket: str, object_key: str, data: bytes, content_type: str) -> None:
        stream = BytesIO(data)
        self.client.put_object(bucket, object_key, stream, length=len(data), content_type=content_type)

    def get_bytes(self, bucket: str, object_key: str) -> bytes:
        response = self.client.get_object(bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        return [item.object_name for item in self.client.list_objects(bucket, prefix=prefix, recursive=True)]


class HttpPrefixObjectStore(ObjectStore):
    """Read-only store that serves existing public image URLs without copying bytes."""

    def __init__(self, *, base_url: str, bucket: str = "minio-src", verify_tls: bool = True):
        self.base_url = base_url.rstrip("/")
        self.bucket = bucket
        self.verify_tls = verify_tls

    def exists(self, bucket: str, object_key: str) -> bool:
        return bucket == self.bucket and bool(object_key)

    def presigned_get_url(self, bucket: str, object_key: str, expires_seconds: int = 900) -> str | None:
        if not self.exists(bucket, object_key):
            return None
        return public_image_url(self.base_url, object_key)

    def put_bytes(self, bucket: str, object_key: str, data: bytes, content_type: str) -> None:
        raise RuntimeError("http prefix object store is read-only")

    def get_bytes(self, bucket: str, object_key: str) -> bytes:
        if not self.exists(bucket, object_key):
            raise FileNotFoundError(object_key)
        import httpx

        response = httpx.get(
            public_image_url(self.base_url, object_key),
            timeout=60,
            follow_redirects=True,
            trust_env=False,
            verify=self.verify_tls,
        )
        response.raise_for_status()
        return response.content

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        return []


class FallbackObjectStore(ObjectStore):
    """Prefer local/MinIO objects, then fall back to a public HTTP image prefix."""

    def __init__(self, primary: ObjectStore, fallback: ObjectStore):
        self.primary = primary
        self.fallback = fallback

    def _fallback_handles_bucket(self, bucket: str) -> bool:
        # The configured HTTP prefix is authoritative for its source bucket.
        # Probing the default Docker MinIO endpoint first adds several seconds
        # when this API is running directly on the host.
        return isinstance(self.fallback, HttpPrefixObjectStore) and self.fallback.bucket == bucket

    def exists(self, bucket: str, object_key: str) -> bool:
        if self._fallback_handles_bucket(bucket):
            return self.fallback.exists(bucket, object_key)
        return self.primary.exists(bucket, object_key) or self.fallback.exists(bucket, object_key)

    def presigned_get_url(self, bucket: str, object_key: str, expires_seconds: int = 900) -> str | None:
        if self._fallback_handles_bucket(bucket):
            return self.fallback.presigned_get_url(bucket, object_key, expires_seconds)
        if self.primary.exists(bucket, object_key):
            return self.primary.presigned_get_url(bucket, object_key, expires_seconds)
        return self.fallback.presigned_get_url(bucket, object_key, expires_seconds)

    def put_bytes(self, bucket: str, object_key: str, data: bytes, content_type: str) -> None:
        self.primary.put_bytes(bucket, object_key, data, content_type)

    def get_bytes(self, bucket: str, object_key: str) -> bytes:
        if self._fallback_handles_bucket(bucket):
            return self.fallback.get_bytes(bucket, object_key)
        if self.primary.exists(bucket, object_key):
            return self.primary.get_bytes(bucket, object_key)
        return self.fallback.get_bytes(bucket, object_key)

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        return self.primary.list_keys(bucket, prefix)

    def ensure_bucket(self, bucket: str) -> None:
        if hasattr(self.primary, "ensure_bucket"):
            self.primary.ensure_bucket(bucket)


def _http_prefix_store(settings) -> HttpPrefixObjectStore | None:
    base_url = getattr(settings, "source_image_base_url", "") or ""
    if not base_url:
        return None
    return HttpPrefixObjectStore(
        base_url=base_url,
        bucket=getattr(settings, "source_image_bucket", "minio-src") or "minio-src",
        verify_tls=getattr(settings, "source_image_verify_tls", True),
    )


class LocalReadonlyObjectStore(ObjectStore):
    """Isolated SQLite replay only. Resolves files beneath an explicit root."""
    def __init__(self,root):
        from pathlib import Path
        self.root=Path(root).resolve()
    def path(self,key):
        p=(self.root/key).resolve()
        if not p.is_relative_to(self.root):raise ValueError("image path outside local replay root")
        return p
    def exists(self,bucket,key):
        return self.path(key).is_file()
    def get_bytes(self,bucket,key):
        p=self.path(key)
        if p.stat().st_size>20_000_000:raise ValueError("image too large")
        return p.read_bytes()
    def presigned_get_url(self,bucket,key,expires_seconds=900):
        import base64
        return "data:image/jpeg;base64,"+base64.b64encode(self.get_bytes(bucket,key)).decode()
    def put_bytes(self,*args,**kwargs):raise RuntimeError("local replay image source is read-only")


def build_object_store(settings) -> ObjectStore:
    if settings.local_readonly_image_root:
        if not settings.database_url.startswith("sqlite:") or settings.minio_endpoint:
            raise ValueError("local replay requires SQLite and disabled MinIO")
        return LocalReadonlyObjectStore(settings.local_readonly_image_root)
    http_store = _http_prefix_store(settings)
    if not settings.minio_endpoint or not settings.minio_access_key or not settings.minio_secret_key:
        if settings.object_store_required:
            raise RuntimeError("object storage is required but MinIO endpoint and credentials are incomplete")
        return http_store or NullObjectStore()
    try:
        primary = MinioObjectStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            public_endpoint=settings.minio_public_endpoint,
            region=settings.minio_region,
            auto_create_bucket=settings.minio_auto_create_bucket,
        )
    except Exception:
        if settings.object_store_required:
            raise
        return http_store or NullObjectStore()
    if http_store is None:
        return primary
    return FallbackObjectStore(primary, http_store)
