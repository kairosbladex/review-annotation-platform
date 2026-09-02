# ACCEPTANCE_ADAPTER-CONTRACT

## Delivered behavior

- `HttpModelAdapter` calls a compatible detection service with multipart image data and a `conf` threshold.
- `LegacyReviewAgentAdapter` calls the existing service in the required two-stage order: `POST /api/v1/review-agent/yolo-detect`, then `POST /api/v1/review-agent/review` with the detection JSON.
- Profile-level endpoint/API key overrides take precedence over defaults.
- `detections` and `predictions` response shapes both normalize into the platform Candidate contract.
- Candidate normalization preserves class, confidence, normalized bbox, raw decision, YOLO/VLM decisions, reason/error codes and reasoning. For the legacy review response, `rawDecision` takes precedence over the product-facing `decision`, so an uncertain result is not silently persisted as false.
- `success: false`, non-2xx responses and malformed bbox responses surface as explicit failures to the Worker.
- Environment HTTP/SOCKS proxy variables are ignored for internal service-to-service requests (`trust_env=False`).
- Relative preview paths from the legacy service resolve against the configured service base URL; absolute/data URLs can be fetched into MinIO by the Worker. The configured service API key and TLS verification setting are also applied when fetching HTTP artifacts.
- `scripts/check_minio.py` verifies target bucket creation, write, read, existence and presigned URL generation without touching business objects.

## Validation evidence

`backend/tests/test_adapter_contract.py` starts local HTTP contract fixtures and verifies multipart request shape, `conf`, API key precedence, detection→review call order, all review fields, relative preview resolution, dependency error and malformed bbox behavior. The full backend suite currently passes 15 tests.

## Not falsely claimed

- The current host has no Docker daemon and no reachable real MinIO instance, so this task does not claim a live MinIO round-trip.
- A real deployment still needs `MODEL_SERVICE_BASE_URL`, `MODEL_SERVICE_API_KEY`, `MINIO_ENDPOINT`, credentials and bucket values in `.env`, followed by `python3 scripts/check_production_config.py --env-file deploy/production.env`, `PYTHONPATH=backend python3 scripts/check_minio.py --env-file deploy/production.env`, and a model-service probe using a representative image.
- The old service's preview URL must be reachable from the Worker network; an inaccessible preview is recorded as an artifact warning while valid predictions remain successful.

## Production controls

- `deploy/production.env.example` is a safe template only. The real `deploy/production.env` is ignored by Git and must be supplied through the deployment host or a secret manager.
- `docker-compose.production.yml` uses external PostgreSQL, Redis, MinIO and model-service endpoints; it intentionally does not start local storage services.
- `MINIO_AUTO_CREATE_BUCKET=false` and `OBJECT_STORE_REQUIRED=true` make startup fail when the production bucket or required credentials are unavailable. Production never silently downgrades to `NullObjectStore`.
- `MODEL_SERVICE_DETECT_ENDPOINT`, `MODEL_SERVICE_REVIEW_ENDPOINT` and `MODEL_SERVICE_VERIFY_TLS` are wired into the API and Worker defaults. A ModelProfile may still override them per model version.

## Handoff

```bash
PYTHONPATH=backend python3 -m pytest backend/tests -q
python3 scripts/check_production_config.py --env-file deploy/production.env
PYTHONPATH=backend python3 scripts/check_minio.py --env-file deploy/production.env
PYTHONPATH=backend python3 scripts/check_model_service.py --env-file deploy/production.env /path/to/review-sample.jpg
```

The first command is runnable locally. The remaining commands are deployment checks and intentionally require real production configuration.
