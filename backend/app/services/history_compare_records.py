"""Authenticated, durable history-comparison records."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from datetime import datetime, timezone

from fastapi import Depends, File, HTTPException, Query, UploadFile
from PIL import Image
from sqlalchemy import DateTime, ForeignKey, String, Text, func, select
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base
from ..models import AuditEvent, User, new_id, utc_now
from .history_compare import MAX_IMAGE_BYTES, compare_images


class HistoryComparison(Base):
    __tablename__ = 'history_comparisons'

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_by: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    bucket: Mapped[str] = mapped_column(String(255))
    baseline_object_key: Mapped[str] = mapped_column(String(1024))
    current_object_key: Mapped[str] = mapped_column(String(1024))
    baseline_sha256: Mapped[str] = mapped_column(String(64))
    current_sha256: Mapped[str] = mapped_column(String(64))
    result_json: Mapped[str] = mapped_column(Text)


def _timestamp(value: datetime) -> str:
    # SQLite drops the offset from timezone-aware DateTime columns.
    return value.replace(tzinfo=timezone.utc).isoformat() if value.tzinfo is None else value.isoformat()


def _content_type(data: bytes) -> str:
    # The bytes have already passed the comparison decoder; distrust upload MIME.
    with Image.open(io.BytesIO(data)) as image:
        return Image.MIME.get(image.format, 'application/octet-stream')


def _detail(record: HistoryComparison, store) -> dict:
    try:
        if not store.exists(record.bucket, record.baseline_object_key) or not store.exists(record.bucket, record.current_object_key):
            raise FileNotFoundError('stored comparison image unavailable')
        baseline_url = store.presigned_get_url(record.bucket, record.baseline_object_key)
        current_url = store.presigned_get_url(record.bucket, record.current_object_key)
        if not baseline_url or not current_url:
            raise RuntimeError('object URL unavailable')
    except Exception as exc:
        raise HTTPException(503, '对比图片暂时无法读取，请稍后重试') from exc
    return {
        'comparison_id': record.id,
        'result': json.loads(record.result_json),
        'baseline_url': baseline_url,
        'current_url': current_url,
        'created_at': _timestamp(record.created_at),
    }


def register_history_record_routes(app, current_user):
    """Keep comparison ownership, stored media and result retrieval together."""
    slots = threading.BoundedSemaphore(2)

    @app.post('/api/v1/history-comparisons', status_code=201)
    def create_history_comparison(
        baseline: UploadFile = File(...),
        current: UploadFile = File(...),
        user: User = Depends(current_user),
    ):
        if not slots.acquire(blocking=False):
            raise HTTPException(503, '历史对比繁忙，请稍后重试')
        try:
            before = baseline.file.read(MAX_IMAGE_BYTES + 1)
            after = current.file.read(MAX_IMAGE_BYTES + 1)
            if max(len(before), len(after)) > MAX_IMAGE_BYTES:
                raise HTTPException(413, '单张图片不能超过20MB')
            settings = app.state.settings
            try:
                result = compare_images(
                    before, after,
                    base_url=settings.history_compare_ollama_url,
                    model=settings.history_compare_model,
                    timeout=settings.history_compare_timeout_seconds,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

            identifier = new_id()
            record = HistoryComparison(
                id=identifier,
                created_by=user.id,
                created_at=utc_now(),
                bucket=settings.minio_bucket,
                baseline_object_key=f'history-comparisons/{user.id}/{identifier}/baseline',
                current_object_key=f'history-comparisons/{user.id}/{identifier}/current',
                baseline_sha256=hashlib.sha256(before).hexdigest(),
                current_sha256=hashlib.sha256(after).hexdigest(),
                result_json=json.dumps(result, ensure_ascii=False, allow_nan=False),
            )
            store = app.state.object_store
            try:
                store.put_bytes(record.bucket, record.baseline_object_key, before, _content_type(before))
                store.put_bytes(record.bucket, record.current_object_key, after, _content_type(after))
            except Exception as exc:
                raise HTTPException(503, '对比图片保存失败，请稍后重试') from exc
            detail = _detail(record, store)
            with app.state.session_factory() as session:
                session.add(record)
                session.add(AuditEvent(action='history_comparison.created', user_id=user.id,
                                       entity_type='history_comparison', entity_id=record.id,
                                       payload_json=json.dumps({'baseline_sha256': record.baseline_sha256,
                                                                'current_sha256': record.current_sha256,
                                                                'model': settings.history_compare_model})))
                session.commit()
            return detail
        finally:
            slots.release()

    @app.get('/api/v1/history-comparisons')
    def list_history_comparisons(
        page: int = Query(1, ge=1),
        limit: int = Query(20, ge=1, le=100),
        user: User = Depends(current_user),
    ):
        with app.state.session_factory() as session:
            owned = HistoryComparison.created_by == user.id
            total = session.scalar(select(func.count()).select_from(HistoryComparison).where(owned))
            records = session.scalars(select(HistoryComparison).where(owned)
                                      .order_by(HistoryComparison.created_at.desc(), HistoryComparison.id.desc())
                                      .offset((page - 1) * limit).limit(limit)).all()
            comparisons = []
            for record in records:
                result = json.loads(record.result_json)
                comparisons.append({
                    'comparison_id': record.id, 'created_at': _timestamp(record.created_at),
                    'status': result['status'], 'change_count': result['change_count'],
                    'diff_summary': result['diff_summary'],
                })
            return {'comparisons': comparisons, 'total': total}

    @app.get('/api/v1/history-comparisons/{comparison_id}')
    def get_history_comparison(comparison_id: str, user: User = Depends(current_user)):
        with app.state.session_factory() as session:
            record = session.scalar(select(HistoryComparison).where(
                HistoryComparison.id == comparison_id, HistoryComparison.created_by == user.id))
            if record is None:
                raise HTTPException(404, '对比记录不存在')
            return _detail(record, app.state.object_store)
