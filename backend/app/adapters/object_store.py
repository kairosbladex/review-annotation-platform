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
    def __init__(self, *, endpoint: str, access_key: str, secret_key: str, secure: bool, public_endpoint: str = ""):
        from minio import Minio

        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
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
                region="us-east-1",
            )

    def ensure_bucket(self, bucket: str) -> None:
        if not self.client.bucket_exists(bucket):
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


def build_object_store(settings) -> ObjectStore:
    if not settings.minio_endpoint or not settings.minio_access_key or not settings.minio_secret_key:
        return NullObjectStore()
    try:
        return MinioObjectStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            public_endpoint=settings.minio_public_endpoint,
        )
    except Exception:
        return NullObjectStore()
