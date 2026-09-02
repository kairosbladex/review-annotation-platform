from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from sqlalchemy import select

from ..models import Annotation, AnnotationTask, Asset, Candidate, ExportRecord, Job, ModelRun, ModelRunSample, Sample, utc_now


def _jsonl_rows(session, run_id: str) -> list[dict]:
    run = session.get(ModelRun, run_id)
    if run is None:
        raise ValueError("model run not found")
    inputs = session.scalars(select(ModelRunSample).where(ModelRunSample.model_run_id == run_id)).all()
    rows: list[dict] = []
    for run_input in inputs:
        task = session.scalar(select(AnnotationTask).where(AnnotationTask.model_run_id == run_id, AnnotationTask.sample_id == run_input.sample_id))
        if task is None or task.status != "completed":
            raise ValueError("only completed annotation tasks can be exported")
        sample = session.get(Sample, run_input.sample_id)
        candidates = session.scalars(select(Candidate).where(Candidate.model_run_id == run_id, Candidate.sample_id == run_input.sample_id).order_by(Candidate.target_index)).all()
        for candidate in candidates:
            annotation = session.scalar(select(Annotation).where(Annotation.candidate_id == candidate.id))
            if annotation is None or annotation.label is None:
                raise ValueError("all candidates must be labeled before export")
            rows.append({
                "sample_id": sample.id,
                "source_type": sample.source_type,
                "sample_created_at": sample.created_at,
                "run_id": run_id,
                "image_object_key": None,
                "image_bucket": None,
                "candidate_id": candidate.id,
                "bbox": [candidate.x1, candidate.y1, candidate.x2, candidate.y2],
                "class_id": candidate.class_id,
                "class_name": candidate.class_name,
                "model_confidence": candidate.confidence,
                "human_label": annotation.label,
                "human_note": annotation.note,
                "raw_decision": candidate.raw_decision,
                "reason_code": candidate.reason_code,
                "error_code": candidate.error_code,
                "model_name": run.model_name,
                "model_version": run.model_version,
                "labeled_by": annotation.labeled_by,
                "labeled_at": annotation.labeled_at.isoformat() if annotation.labeled_at else None,
            })
        original = session.scalar(select(Asset).where(Asset.sample_id == sample.id, Asset.kind == "original"))
        for row in rows:
            if row["sample_id"] == sample.id:
                row["image_object_key"] = original.object_key if original else None
                row["image_bucket"] = original.bucket if original else None
    return rows


def _apply_filters(rows: list[dict], filters: dict) -> list[dict]:
    class_names = set(filters.get("class_names") or [])
    source_type = filters.get("source_type")
    created_from = filters.get("created_from")
    created_to = filters.get("created_to")
    # Creation timestamps are intentionally parsed only when supplied. The
    # row-level export contract remains stable for consumers.
    if class_names:
        rows = [row for row in rows if row["class_name"] in class_names]
    if source_type:
        rows = [row for row in rows if row.get("source_type") == source_type]
    if created_from or created_to:
        # _jsonl_rows can be extended with sample timestamps without changing
        # the public JSONL schema; malformed filters fail loudly here.
        from datetime import datetime

        lower = datetime.fromisoformat(created_from.replace("Z", "+00:00")) if created_from else None
        upper = datetime.fromisoformat(created_to.replace("Z", "+00:00")) if created_to else None
        rows = [row for row in rows if (not lower or row.get("sample_created_at") >= lower) and (not upper or row.get("sample_created_at") <= upper)]
    return rows


def execute_export_job(*, session_factory, object_store, bucket: str, job_id: str) -> None:
    with session_factory() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        record = session.get(ExportRecord, job.reference_id)
        if record is None:
            job.status = "failed"
            job.error_summary = "export record not found"
            session.commit()
            return
        try:
            rows = _apply_filters(_jsonl_rows(session, record.run_id), json.loads(record.filters_json or "{}"))
            for row in rows:
                row.pop("source_type", None)
                row.pop("sample_created_at", None)
            formats = json.loads(record.formats_json or "[]")
            payload = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows).encode("utf-8")
            extension = "jsonl"
            if "yolo" in formats:
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
                    zipped.writestr("annotations.jsonl", payload)
                    hard_negative = b"\n".join(
                        json.dumps(row, ensure_ascii=False, default=str).encode("utf-8")
                        for row in rows
                        if row["human_label"] == "false"
                    )
                    if hard_negative:
                        hard_negative += b"\n"
                    zipped.writestr("hard_negative.jsonl", hard_negative)
                    grouped: dict[str, list[dict]] = {}
                    for row in rows:
                        grouped.setdefault(row["sample_id"], []).append(row)
                    for sample_id, sample_rows in grouped.items():
                        positive_lines = []
                        for row in sample_rows:
                            if row["human_label"] != "true":
                                continue
                            class_id = row["class_id"] if row["class_id"] is not None else 0
                            x1, y1, x2, y2 = row["bbox"]
                            positive_lines.append(f"{class_id} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}")
                        zipped.writestr(f"labels/{sample_id}.txt", "\n".join(positive_lines) + ("\n" if positive_lines else ""))
                        image_key = next((row.get("image_object_key") for row in sample_rows if row.get("image_object_key")), None)
                        image_bucket = next((row.get("image_bucket") for row in sample_rows if row.get("image_bucket")), None)
                        if not image_key or not image_bucket:
                            raise ValueError(f"original asset missing for sample {sample_id}")
                        image_bytes = object_store.get_bytes(image_bucket, image_key)
                        suffix = Path(image_key).suffix.lower() or ".jpg"
                        zipped.writestr(f"images/{sample_id}{suffix}", image_bytes)
                payload = archive.getvalue()
                extension = "zip"
            object_key = f"exports/{record.id}/dataset.{extension}"
            object_store.put_bytes(bucket, object_key, payload, "application/zip" if extension == "zip" else "application/x-ndjson")
            record.status = "succeeded"
            record.bucket = bucket
            record.object_key = object_key
            record.row_count = len(rows)
            record.finished_at = utc_now()
            job.status = "succeeded"
            job.completed = 1
            job.finished_at = utc_now()
        except Exception as exc:
            record.status = "failed"
            record.error_summary = str(exc)
            job.status = "failed"
            job.error_summary = str(exc)
            record.finished_at = utc_now()
            job.finished_at = utc_now()
        session.commit()
