from __future__ import annotations

import json
import hashlib
import mimetypes
from collections.abc import Iterable

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from ..models import Annotation, AnnotationTask, Asset, AuditEvent, Candidate, ImportBatch, IngestionEvent, ModelRun, ModelRunSample, Sample, User, new_id, utc_now


def _counts(session: Session, task_id: str) -> dict[str, int]:
    rows = session.execute(
        select(
            func.count(Candidate.id),
            func.sum(case((Annotation.label.is_not(None), 1), else_=0)),
            func.sum(case((Annotation.label.is_(None), 1), else_=0)),
            func.sum(case((Annotation.label == "true", 1), else_=0)),
            func.sum(case((Annotation.label == "false", 1), else_=0)),
        )
        .select_from(Candidate)
        .join(AnnotationTask, AnnotationTask.model_run_id == Candidate.model_run_id)
        .outerjoin(Annotation, Annotation.candidate_id == Candidate.id)
        .where(AnnotationTask.id == task_id)
    ).one()
    values = [int(value or 0) for value in rows]
    return {"total": values[0], "labeled": values[1], "pending": values[2], "true": values[3], "false": values[4]}


def refresh_task_status(session: Session, task: AnnotationTask) -> dict[str, int]:
    counts = _counts(session, task.id)
    task.status = task_status_from_counts(counts)
    task.updated_at = utc_now()
    return counts


def task_status_from_counts(counts: dict[str, int]) -> str:
    """Derive a read-only task status from its current candidate labels."""
    if counts["total"] == 0:
        return "no_candidates"
    if counts["pending"] == 0:
        return "completed"
    if counts["labeled"] == 0:
        return "pending"
    return "in_progress"


def _candidate_payload(candidate: Candidate, annotation: Annotation | None) -> dict:
    return {
        "candidate_id": candidate.id,
        "target_index": candidate.target_index,
        "class_id": candidate.class_id,
        "class_name": candidate.class_name,
        "confidence": candidate.confidence,
        "bbox": [candidate.x1, candidate.y1, candidate.x2, candidate.y2],
        "raw_decision": candidate.raw_decision,
        "yolo_decision": candidate.yolo_decision,
        "llm_decision": candidate.llm_decision,
        "reason_code": candidate.reason_code,
        "error_code": candidate.error_code,
        "reasoning": candidate.reasoning,
        "annotation": {
            "annotation_id": annotation.id,
            "label": annotation.label,
            "note": annotation.note,
            "labeled_by": annotation.labeled_by,
            "labeled_at": annotation.labeled_at,
            "revision": annotation.revision,
        } if annotation else {"annotation_id": None, "label": None, "note": None, "labeled_by": None, "labeled_at": None, "revision": 0},
    }


def serialize_sample(session: Session, sample: Sample) -> dict:
    tasks = session.scalars(select(AnnotationTask).where(AnnotationTask.sample_id == sample.id).order_by(AnnotationTask.created_at.desc())).all()
    task_payloads = []
    for task in tasks:
        # Serialization must not mutate task state.  In particular, a GET after
        # a committed label update must not create an uncommitted status update
        # that can roll back or overwrite the value just persisted.
        counts = _counts(session, task.id)
        run = session.get(ModelRun, task.model_run_id)
        candidates = session.scalars(select(Candidate).where(Candidate.model_run_id == task.model_run_id).order_by(Candidate.target_index)).all()
        annotations = {
            annotation.candidate_id: annotation
            for annotation in session.scalars(select(Annotation).where(Annotation.candidate_id.in_([candidate.id for candidate in candidates]))).all()
        } if candidates else {}
        task_payloads.append({
            "annotation_task_id": task.id,
            "model_run_id": task.model_run_id,
            "status": task_status_from_counts(counts),
            "counts": counts,
            "model_run": {
                "model_name": run.model_name if run else None,
                "model_version": run.model_version if run else None,
                "source_type": run.source_type if run else None,
                "status": run.status if run else None,
                "parameters": json.loads(run.parameters_json or "{}") if run else {},
            },
            "candidates": [_candidate_payload(candidate, annotations.get(candidate.id)) for candidate in candidates],
        })
    active = task_payloads[0] if task_payloads else {"annotation_task_id": None, "status": "imported", "counts": {"total": 0, "labeled": 0, "pending": 0, "true": 0, "false": 0}, "candidates": []}
    return {
        "sample_id": sample.id,
        "source_type": sample.source_type,
        "source_event_id": sample.source_event_id,
        "source_service": sample.source_service,
        "session_id": sample.session_id,
        "capture_time": sample.capture_time,
        "camera_id": sample.camera_id,
        "rule_code": sample.rule_code,
        "metadata": json.loads(sample.metadata_json or "{}"),
        "asset_status": sample.asset_status,
        "created_at": sample.created_at,
        "status": active["status"],
        "counts": active["counts"],
        "annotation_task_id": active.get("annotation_task_id"),
        "assets": [
            {"asset_id": asset.id, "kind": asset.kind, "bucket": asset.bucket, "object_key": asset.object_key, "sha256": asset.sha256, "status": asset.status}
            for asset in session.scalars(select(Asset).where(Asset.sample_id == sample.id).order_by(Asset.kind)).all()
        ],
        "runs": task_payloads,
        "candidates": active.get("candidates", []),
    }


def ingest_review_event(session: Session, payload, object_store) -> tuple[Sample, bool]:
    existing = session.scalar(select(IngestionEvent).where(IngestionEvent.event_id == payload.event_id))
    if existing:
        return session.get(Sample, existing.sample_id), False

    image_exists = object_store.exists(payload.image.bucket, payload.image.object_key)
    comparison_exists = bool(payload.comparison_image and object_store.exists(payload.comparison_image.bucket, payload.comparison_image.object_key))
    sample = Sample(
        source_type="review_service",
        source_event_id=payload.event_id,
        source_service=payload.source_service,
        session_id=payload.session_id,
        capture_time=payload.capture_time,
        camera_id=payload.camera_id,
        rule_code=payload.rule_code,
        metadata_json=json.dumps(payload.metadata, ensure_ascii=False),
        asset_status="ready" if image_exists else "asset_missing",
    )
    session.add(sample)
    session.flush()
    session.add(Asset(sample_id=sample.id, kind="original", bucket=payload.image.bucket, object_key=payload.image.object_key, sha256=payload.image.sha256, status="ready" if image_exists else "missing"))
    if payload.comparison_image:
        session.add(Asset(sample_id=sample.id, kind="comparison", bucket=payload.comparison_image.bucket, object_key=payload.comparison_image.object_key, sha256=payload.comparison_image.sha256, status="ready" if comparison_exists else "missing"))

    run = ModelRun(
        model_name=payload.source_service,
        model_version="review-event",
        source_type="review_service",
        status="succeeded",
        input_sample_count=1,
        parameters_json="{}",
        started_at=payload.capture_time,
        finished_at=payload.capture_time,
    )
    session.add(run)
    session.flush()
    session.add(ModelRunSample(model_run_id=run.id, sample_id=sample.id, status="succeeded"))
    task = AnnotationTask(sample_id=sample.id, model_run_id=run.id, status="pending")
    session.add(task)
    session.flush()
    for item in payload.targets:
        session.add(Candidate(
            sample_id=sample.id,
            model_run_id=run.id,
            target_index=item.target_index,
            class_id=item.class_id,
            class_name=item.class_name,
            confidence=item.confidence,
            x1=item.bbox[0], y1=item.bbox[1], x2=item.bbox[2], y2=item.bbox[3],
            raw_decision=item.raw_decision,
            yolo_decision=item.yolo_decision,
            llm_decision=item.llm_decision,
            reason_code=item.reason_code,
            error_code=item.error_code,
            reasoning=item.reasoning,
        ))
    session.add(IngestionEvent(event_id=payload.event_id, sample_id=sample.id, payload_json=payload.model_dump_json()))
    return sample, True


def update_annotation(session: Session, candidate: Candidate, user: User, payload) -> Sample:
    annotation = session.scalar(select(Annotation).where(Annotation.candidate_id == candidate.id))
    if annotation is None:
        annotation = Annotation(candidate_id=candidate.id)
        session.add(annotation)
        session.flush()
    if payload.expected_revision is not None and annotation.revision != payload.expected_revision:
        raise ValueError("annotation revision conflict")
    annotation.label = payload.label
    annotation.note = payload.note
    annotation.labeled_by = user.id if payload.label is not None else None
    annotation.labeled_at = utc_now() if payload.label is not None else None
    annotation.revision += 1
    # The application deliberately uses autoflush=False.  Flush the changed
    # label before counting so the task transition is based on the new state,
    # not on the previous transaction snapshot.
    session.flush()
    task = session.scalar(select(AnnotationTask).where(AnnotationTask.model_run_id == candidate.model_run_id))
    if task is None:
        raise ValueError("annotation task not found")
    counts = refresh_task_status(session, task)
    session.add(AuditEvent(action="annotation.updated", user_id=user.id, entity_type="candidate", entity_id=candidate.id, payload_json=json.dumps({"label": payload.label, "note": payload.note, "revision": annotation.revision}, ensure_ascii=False)))
    return session.get(Sample, candidate.sample_id)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def _create_asset_sample(
    session: Session,
    *,
    source_type: str,
    bucket: str,
    object_key: str,
    sha256: str | None = None,
    source_event_id: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,
    comparison_object_key: str | None = None,
    comparison_exists: bool = False,
    object_exists: bool = True,
) -> tuple[Sample | None, bool]:
    if source_event_id:
        duplicate_event = session.scalar(select(Sample).where(Sample.source_event_id == source_event_id))
        if duplicate_event:
            return duplicate_event, False
    duplicate = session.scalar(select(Asset).where(Asset.bucket == bucket, Asset.object_key == object_key))
    if duplicate:
        return session.get(Sample, duplicate.sample_id), False
    if sha256:
        duplicate = session.scalar(select(Asset).where(Asset.bucket == bucket, Asset.sha256 == sha256))
        if duplicate:
            return session.get(Sample, duplicate.sample_id), False
    sample = Sample(
        source_type=source_type,
        source_event_id=source_event_id,
        session_id=session_id,
        metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
        asset_status="ready" if object_exists else "asset_missing",
    )
    session.add(sample)
    session.flush()
    session.add(Asset(sample_id=sample.id, kind="original", bucket=bucket, object_key=object_key, sha256=sha256, status="ready" if object_exists else "missing"))
    if comparison_object_key:
        session.add(Asset(sample_id=sample.id, kind="comparison", bucket=bucket, object_key=comparison_object_key, status="ready" if comparison_exists else "missing"))
    return sample, True


def _load_manifest_entries(data: bytes) -> list[dict]:
    text = data.decode("utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("items", parsed.get("samples", []))
        if not isinstance(parsed, list):
            raise ValueError("manifest JSON must be an array or contain items")
        return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        entries = []
        for line in text.splitlines():
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("manifest JSONL entries must be objects")
                entries.append(item)
        return entries


def import_minio_prefix(session: Session, *, bucket: str, prefix: str, object_store, manifest_key: str | None = None) -> ImportBatch:
    batch = ImportBatch(source_type="minio", bucket=bucket, prefix=prefix, status="running")
    session.add(batch)
    session.flush()
    if manifest_key:
        entries = _load_manifest_entries(object_store.get_bytes(bucket, manifest_key))
        batch.total = len(entries)
        for item in entries:
            object_key = str(item.get("object_key") or "")
            if not object_key:
                batch.failed += 1
                batch.error_summary = "manifest entry missing object_key"
                continue
            try:
                original_exists = object_store.exists(bucket, object_key)
                comparison_key = item.get("comparison_object_key")
                comparison_exists = bool(comparison_key and object_store.exists(bucket, comparison_key))
                _, created = _create_asset_sample(
                    session,
                    source_type="minio_import",
                    bucket=bucket,
                    object_key=object_key,
                    sha256=item.get("sha256"),
                    source_event_id=item.get("external_id"),
                    session_id=item.get("session_id"),
                    metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    comparison_object_key=comparison_key,
                    comparison_exists=comparison_exists,
                    object_exists=original_exists,
                )
                if not original_exists:
                    batch.failed += 1
                    batch.error_summary = f"asset missing: {object_key}"
                elif created:
                    batch.imported += 1
                else:
                    batch.skipped += 1
            except Exception as exc:
                batch.failed += 1
                batch.error_summary = str(exc)
        batch.status = "partial_failed" if batch.failed else "succeeded"
        return batch
    keys = [key for key in object_store.list_keys(bucket, prefix) if any(key.lower().endswith(extension) for extension in IMAGE_EXTENSIONS)]
    batch.total = len(keys)
    for key in keys:
        try:
            content = object_store.get_bytes(bucket, key)
            _, created = _create_asset_sample(session, source_type="minio_import", bucket=bucket, object_key=key, sha256=hashlib.sha256(content).hexdigest())
            if created:
                batch.imported += 1
            else:
                batch.skipped += 1
        except Exception as exc:
            batch.failed += 1
            batch.error_summary = str(exc)
    batch.status = "partial_failed" if batch.failed else "succeeded"
    return batch


def import_uploaded_bytes(session: Session, *, bucket: str, object_key: str, data: bytes, content_type: str, object_store) -> tuple[Sample, bool]:
    sha256 = hashlib.sha256(data).hexdigest()
    object_store.put_bytes(bucket, object_key, data, content_type or mimetypes.guess_type(object_key)[0] or "application/octet-stream")
    return _create_asset_sample(session, source_type="manual_upload", bucket=bucket, object_key=object_key, sha256=sha256)
