# ACCEPTANCE_PLATFORM-004

## Delivered

- `ModelProfile`, `ModelRun`, `ModelRunSample`, and `Job` persistence.
- HTTP `ModelAdapter` contract with configurable endpoint, API key, timeout, and prediction normalization.
- Celery/Redis dispatcher and worker task definitions.
- Per-sample success/failure isolation, progress counters, retry wrapper, model version history, and new annotation task per run.
- Optional adapter result previews are copied into `model-runs/{run_id}/{sample_id}/result.*` when the adapter can resolve them.

## Evidence

`backend/tests/test_model_runs.py` uses a fake adapter and inline dispatcher to verify a second version creates a separate run, candidate set, and task without replacing the review-event run.

## Boundary

The real legacy service still needs a live Adapter Contract Test and deployment-level retry/timeout verification. No local GPU inference or automatic model training is included.
