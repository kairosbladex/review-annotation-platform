# ACCEPTANCE_ADAPTER-CONTRACT

## Delivered behavior

- `HttpModelAdapter` calls the current review service `POST /api/v1/review-agent/yolo-detect` with multipart image data and `conf` threshold.
- Profile-level endpoint/API key overrides take precedence over defaults.
- `detections` and `predictions` response shapes both normalize into the platform Candidate contract.
- Candidate normalization preserves class, confidence, normalized bbox, raw decision, YOLO/VLM decisions, reason/error codes and reasoning.
- `success: false`, non-2xx responses and malformed bbox responses surface as explicit failures to the Worker.
- Environment HTTP/SOCKS proxy variables are ignored for internal service-to-service requests (`trust_env=False`).
- Relative preview paths from the legacy service resolve against the configured service base URL; absolute/data URLs can be fetched into MinIO by the Worker.
- `scripts/check_minio.py` verifies target bucket creation, write, read, existence and presigned URL generation without touching business objects.

## Validation evidence

`backend/tests/test_adapter_contract.py` starts a local HTTP contract fixture and verifies multipart request shape, `conf`, API key precedence, all review fields, relative preview resolution, dependency error and malformed bbox behavior. Full backend suite: 9 passed.

## Not falsely claimed

- The current host has no Docker daemon and no reachable real MinIO instance, so this task does not claim a live MinIO round-trip.
- A real deployment still needs `MODEL_SERVICE_BASE_URL`, `MODEL_SERVICE_API_KEY`, `MINIO_ENDPOINT`, credentials and bucket values in `.env`, followed by `PYTHONPATH=backend python3 scripts/check_minio.py`.
- The old service's preview URL must be reachable from the Worker network; an inaccessible preview is recorded as an artifact warning while valid predictions remain successful.

## Handoff

```bash
PYTHONPATH=backend python3 -m pytest backend/tests -q
PYTHONPATH=backend python3 scripts/check_minio.py
```

The first command is runnable locally. The second is the deployment smoke command and intentionally exits with a clear message when MinIO is not configured.
