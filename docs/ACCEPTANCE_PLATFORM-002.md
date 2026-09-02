# ACCEPTANCE_PLATFORM-002

## Delivered

- Review-service callback with `event_id` idempotency and service API key.
- MinIO prefix and JSON/JSONL manifest import with external/session metadata.
- Multipart page upload, image extension and size checks, object/hash deduplication.
- Missing original/comparison assets remain visible as explicit `asset_missing` / `missing` records.
- `scripts/migrate_legacy_sqlite.py` reads the old SQLite schema, uploads local assets, normalizes pixel boxes, preserves existing labels, and never mutates the source database.

## Evidence

`backend/tests/test_annotation_flow.py`, `backend/tests/test_imports.py`, and `backend/tests/test_manifest_import.py` cover callback idempotency, prefix import, upload, deduplication, manifest context, and comparison assets.

## Boundary

Import batch UI/history and downloadable import-error reports are follow-up work. Production migration requires a reachable writable MinIO and Pillow for legacy pixel-space boxes.
