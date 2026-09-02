"""Run a non-destructive read/write smoke check against configured MinIO."""

from __future__ import annotations

import argparse
import json
import uuid

from app.adapters.object_store import build_object_store
from app.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=None)
    args = parser.parse_args()
    settings = Settings.from_env()
    bucket = args.bucket or settings.minio_bucket
    store = build_object_store(settings)
    if not hasattr(store, "client"):
        raise SystemExit("MinIO is not configured or is unreachable; set MINIO_ENDPOINT and credentials")
    if hasattr(store, "ensure_bucket"):
        store.ensure_bucket(bucket)
    key = f"_smoke/{uuid.uuid4().hex}.txt"
    payload = b"review-annotation-platform minio smoke check"
    store.put_bytes(bucket, key, payload, "text/plain")
    round_trip = store.get_bytes(bucket, key)
    print(json.dumps({"bucket": bucket, "object_key": key, "exists": store.exists(bucket, key), "round_trip_ok": round_trip == payload, "presigned_url": bool(store.presigned_get_url(bucket, key))}, ensure_ascii=False))
    if round_trip != payload:
        raise SystemExit("MinIO round-trip verification failed")


if __name__ == "__main__":
    main()
