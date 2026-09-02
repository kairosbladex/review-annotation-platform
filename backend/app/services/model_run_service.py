from __future__ import annotations

import json
import hashlib
from typing import Any

from sqlalchemy import select

from ..models import Asset, AnnotationTask, Candidate, Job, ModelProfile, ModelRun, ModelRunSample, Sample, utc_now


def _prediction_value(prediction: Any, name: str, default=None):
    if isinstance(prediction, dict):
        return prediction.get(name, default)
    return getattr(prediction, name, default)


def execute_model_job(*, session_factory, object_store, adapter, job_id: str) -> None:
    """Execute one model run; each sample failure is isolated and persisted."""
    with session_factory() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        run = session.get(ModelRun, job.reference_id)
        if run is None:
            job.status = "failed"
            job.error_summary = "model run not found"
            session.commit()
            return
        job.status = "running"
        job.started_at = utc_now()
        run.status = "running"
        session.commit()
        inputs = session.scalars(select(ModelRunSample).where(ModelRunSample.model_run_id == run.id)).all()
        profile = session.get(ModelProfile, run.model_profile_id) if run.model_profile_id else None
        profile_options = json.loads(run.parameters_json or "{}")
        profile_options["_model_profile"] = {
            **(json.loads(profile.config_json or "{}") if profile else {}),
            "adapter_type": profile.adapter_type if profile else "http",
            "model_name": profile.model_name if profile else run.model_name,
            "model_version": profile.model_version if profile else run.model_version,
        }

        for run_input in inputs:
            try:
                sample = session.get(Sample, run_input.sample_id)
                if sample is None:
                    raise ValueError("sample not found")
                asset = session.scalar(select(Asset).where(Asset.sample_id == sample.id, Asset.kind == "original"))
                if asset is None:
                    raise ValueError("original asset not found")
                image = {"bucket": asset.bucket, "object_key": asset.object_key, "sha256": asset.sha256}
                result = adapter.infer(image, profile_options)
                predictions = result.get("predictions", []) if isinstance(result, dict) else getattr(result, "predictions", [])
                for index, prediction in enumerate(predictions):
                    bbox = _prediction_value(prediction, "bbox", [])
                    if len(bbox) != 4:
                        raise ValueError("model returned invalid bbox")
                    session.add(Candidate(
                        sample_id=sample.id,
                        model_run_id=run.id,
                        target_index=index,
                        class_id=_prediction_value(prediction, "class_id"),
                        class_name=str(_prediction_value(prediction, "class_name", "unknown")),
                        confidence=_prediction_value(prediction, "confidence"),
                        x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]),
                        raw_decision=_prediction_value(prediction, "raw_decision"),
                        yolo_decision=_prediction_value(prediction, "yolo_decision"),
                        llm_decision=_prediction_value(prediction, "llm_decision"),
                        reason_code=_prediction_value(prediction, "reason_code"),
                        error_code=_prediction_value(prediction, "error_code"),
                        reasoning=str(_prediction_value(prediction, "reasoning", "") or ""),
                    ))
                artifact = result.get("artifact") if isinstance(result, dict) else getattr(result, "artifact", None)
                if artifact and hasattr(adapter, "fetch_artifact"):
                    try:
                        resolved = adapter.fetch_artifact(artifact, profile_options)
                        if resolved:
                            artifact_bytes, content_type = resolved
                            suffix = ".png" if content_type == "image/png" else ".jpg"
                            object_key = f"model-runs/{run.id}/{sample.id}/result{suffix}"
                            object_store.put_bytes(asset.bucket, object_key, artifact_bytes, content_type)
                            session.add(Asset(sample_id=sample.id, kind="result", bucket=asset.bucket, object_key=object_key, sha256=hashlib.sha256(artifact_bytes).hexdigest(), status="ready"))
                    except Exception as artifact_error:
                        # A preview is useful but must not turn a valid model
                        # prediction into a failed sample.
                        run_input.error_message = f"result artifact unavailable: {artifact_error}"
                session.add(AnnotationTask(sample_id=sample.id, model_run_id=run.id, status="pending" if predictions else "no_candidates"))
                run_input.status = "succeeded"
                job.completed += 1
            except Exception as exc:
                run_input.status = "failed"
                run_input.error_message = str(exc)
                job.failed += 1
            session.commit()

        run.status = "partial_failed" if job.failed and job.completed else ("failed" if job.failed else "succeeded")
        run.finished_at = utc_now()
        job.status = "partial_failed" if job.failed and job.completed else ("failed" if job.failed else "succeeded")
        job.finished_at = utc_now()
        session.commit()
