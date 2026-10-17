from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import threading
from datetime import timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from fastapi import Depends, File, Header, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from ..adapters.retrieval import RetrievalAdapter, RetrievalServiceError
from ..db import Base
from ..models import AuditEvent, Job, new_id, utc_now

TERMINAL = frozenset({'completed', 'partial', 'failed', 'cancelled', 'submission_unknown'})


class RetrievalMedia(Base):
    __tablename__ = 'retrieval_media'
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_by: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(16))
    bucket: Mapped[str] = mapped_column(String(255))
    object_key: Mapped[str] = mapped_column(String(1024))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), default=utc_now)


class RetrievalTask(Base):
    __tablename__ = 'retrieval_tasks'
    __table_args__ = (UniqueConstraint('created_by', 'idempotency_key', name='uq_retrieval_submission'),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_by: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey('jobs.id'), unique=True)
    idempotency_key: Mapped[str] = mapped_column(String(100))
    request_hash: Mapped[str] = mapped_column(String(64))
    input_json: Mapped[str] = mapped_column(Text)
    remote_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    snapshot_json: Mapped[str] = mapped_column(Text, default='{}')
    results_json: Mapped[str] = mapped_column(Text, default='[]')
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True, allow_inf_nan=False)
    query: str = Field(default='', max_length=256)
    media_ids: list[str] = Field(min_length=1, max_length=100)
    prompt_media_ids: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(default=0.2, ge=0.02, le=1.0)
    sample_interval_seconds: float = Field(default=3.0, ge=0.3, le=600.0)

    @model_validator(mode='after')
    def valid_query(self):
        if not self.query and not self.prompt_media_ids:
            raise ValueError('请输入检索目标或上传参考图片')
        if len(set(self.media_ids)) != len(self.media_ids):
            raise ValueError('检索素材不能重复')
        return self


def _public_url(value):
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme not in {'http', 'https', 'minio'} or not parts.hostname or parts.username or parts.password:
            return None
        return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
    except ValueError:
        return None


def _clean_snapshot(raw, media_keys=()):
    if not isinstance(raw, dict) or raw.get('status') not in {'running', 'stopping', 'completed', 'cancelled', 'error'}:
        raise RetrievalServiceError('invalid_response')
    rows, errors = raw.get('results', []), raw.get('file_errors', [])
    if not isinstance(rows, list) or not isinstance(errors, list):
        raise RetrievalServiceError('invalid_response')
    results = []
    def file_index(row):
        source_path = unquote(urlsplit(str(row.get('file_url') or '')).path)
        for index, key in enumerate(media_keys):
            if source_path.endswith('/' + key):
                return index
        value = row.get('file_index')
        return value - 1 if isinstance(value, int) and value >= 1 else None
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('detections', []), list):
            raise RetrievalServiceError('invalid_response')
        detections = []
        for detection in row.get('detections', []):
            if not isinstance(detection, dict):
                raise RetrievalServiceError('invalid_response')
            bbox, confidence = detection.get('bbox'), detection.get('confidence')
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bbox):
                raise RetrievalServiceError('invalid_response')
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise RetrievalServiceError('invalid_response')
            detections.append({'class_name': str(detection.get('class_name') or '目标')[:120],
                               'class_id': detection.get('class_id'), 'confidence': confidence, 'bbox': bbox})
        results.append({'file_index': file_index(row),
                        'frame_index': row.get('video_frame_id', row.get('frame_index')),
                        'timestamp_seconds': row.get('timestamp_sec_approx', row.get('time_seconds')),
                        'image_url': _public_url(row.get('result_image_minio_url') or row.get('image_url') or row.get('sample_frame_url')),
                        'thumbnail_url': _public_url(row.get('thumbnail_minio_url') or row.get('thumbnail_url') or row.get('sample_frame_thumbnail_url')),
                        'detections': detections})
    status = {'error': 'failed'}.get(raw['status'], raw['status'])
    if status == 'completed' and errors:
        status = 'partial' if results else 'failed'
    count = sum(len(row['detections']) for row in results)
    hits = raw.get('total_hits', count)
    if not isinstance(hits, int) or hits < count or (status in TERMINAL and hits != count):
        raise RetrievalServiceError('incomplete_results')
    progress = raw.get('video_progress')
    progress = {key: value for key, value in progress.items() if key in {
        'file_index', 'phase', 'media_type', 'duration_seconds', 'processed_frames', 'total_frames',
        'current_seconds', 'progress', 'sample_index', 'sample_count', 'frame_index', 'sampled_frames',
        'estimated_total_samples', 'processed_samples', 'video_sample_index', 'video_frame_id', 'timestamp_sec_approx',
    }} if isinstance(progress, dict) else None
    if progress:
        if 'timestamp_sec_approx' in progress:
            progress['current_seconds'] = progress['timestamp_sec_approx']
        if isinstance(progress.get('video_sample_index'), int):
            progress['sampled_frames'] = progress['video_sample_index'] + 1
        if 'video_frame_id' in progress:
            progress['frame_index'] = progress['video_frame_id']
    return {'status': status, 'total_hits': hits, 'processed_count': raw.get('processed_count', 0),
            'total_count': raw.get('total_count', 0), 'video_progress': progress, 'results': results,
            'file_errors': [{'file_index': file_index(error), 'phase': str(error.get('phase') or 'failed')[:80],
                             'error': '该素材处理失败，请检查文件是否完整及格式是否支持'}
                            for error in errors if isinstance(error, dict)],
            'error_code': 'algorithm_failed' if status == 'failed' else None,
            'message': {'completed': '检索完成' if hits else '检索完成，未发现匹配目标',
                        'partial': '部分素材处理失败，已保留成功结果', 'cancelled': '已停止，已保留检出结果',
                        'failed': '检索失败，请检查素材或稍后重试', 'stopping': '正在停止，等待算法确认',
                        'running': '正在分析素材'}.get(status, '')}


def _save_snapshot(task, job, snapshot):
    snapshot = dict(snapshot)
    results = snapshot.pop('results', [])
    task.snapshot_json = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    task.results_json = json.dumps(results, ensure_ascii=False, allow_nan=False)
    task.revision = (task.revision or 0) + 1
    job.status = snapshot['status']
    job.completed = int(snapshot.get('processed_count') or 0)
    job.total = int(snapshot.get('total_count') or job.total)
    job.failed = len(snapshot.get('file_errors', []))
    if job.status in TERMINAL:
        job.finished_at = utc_now()


def _serialize(task, job, *, detail=True, sync_error=None, object_store=None, bucket=''):
    snapshot, inputs = json.loads(task.snapshot_json or '{}'), json.loads(task.input_json)
    def timestamp(value):
        return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value).isoformat()
    results = json.loads(task.results_json or '[]') if detail else []
    if object_store:
        for result in results:
            for key in ('image_url', 'thumbnail_url'):
                value = result.get(key)
                if value and value.startswith('minio://'):
                    parsed = urlsplit(value)
                    object_key = unquote(parsed.path).lstrip('/')
                    if parsed.netloc != bucket or not task.remote_id or not object_key.startswith(task.remote_id + '/') or '..' in object_key.split('/'):
                        result[key] = None
                    else:
                        result[key] = object_store.presigned_get_url(bucket, object_key)
    return {**snapshot, 'task_id': task.id, 'status': job.status, 'query': inputs['query'],
            'media_count': len(inputs['media_ids']), 'created_at': timestamp(task.created_at),
            'finished_at': timestamp(job.finished_at) if job.finished_at else None, 'sync_error': sync_error,
            **({'results': results} if detail else {})}


def register_retrieval_routes(app, current_user):
    app.state.retrieval_adapter = RetrievalAdapter(app.state.settings)
    upload_slot = threading.BoundedSemaphore(1)

    def enabled():
        if not app.state.settings.retrieval_enabled:
            raise HTTPException(403, '智能检索尚未启用')

    def serialize(task, job, **kwargs):
        return _serialize(task, job, object_store=app.state.object_store,
                          bucket=app.state.settings.minio_bucket, **kwargs)

    def claim_snapshot(session, task, expected_revision):
        # The conditional UPDATE also serializes writers on SQLite, where
        # SELECT FOR UPDATE does not take a row lock.
        result = session.execute(update(RetrievalTask).where(
            RetrievalTask.id == task.id, RetrievalTask.revision == expected_revision
        ).values(revision=expected_revision + 1).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            session.rollback()
            return False
        return True

    def owned(session, task_id, user_id):
        task = session.scalar(select(RetrievalTask).where(RetrievalTask.id == task_id, RetrievalTask.created_by == user_id))
        if task is None:
            raise HTTPException(404, '任务不存在')
        return task, session.get(Job, task.job_id)

    @app.get('/api/v1/capabilities')
    def capabilities(user=Depends(current_user)):
        return {'retrieval_enabled': app.state.settings.retrieval_enabled}

    @app.post('/api/v1/retrieval/media', status_code=201)
    def upload_media(file: UploadFile = File(...), user=Depends(current_user)):
        enabled()
        if not upload_slot.acquire(blocking=False):
            raise HTTPException(503, '素材上传繁忙，请稍后再试')
        try:
            limit = app.state.settings.retrieval_max_upload_bytes
            payload = file.file.read(limit + 1)
            if len(payload) > limit:
                raise HTTPException(413, '单个素材不能超过256MB')
            filename = Path((file.filename or 'media').replace('\\', '/')).name[:255]
            suffix = Path(filename).suffix.lower()
            if suffix in {'.png', '.jpg', '.jpeg', '.webp'}:
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        image.verify()
                    media_type, mime = 'image', 'image/' + ('jpeg' if suffix in {'.jpg', '.jpeg'} else suffix[1:])
                except Exception:
                    raise HTTPException(422, '无法读取图片') from None
            elif suffix in {'.mp4', '.mov', '.avi', '.webm'}:
                import cv2
                with tempfile.TemporaryDirectory(prefix='retrieval-upload-') as directory:
                    video_path = Path(directory) / ('media' + suffix)
                    video_path.write_bytes(payload)
                    capture = cv2.VideoCapture(str(video_path))
                    try:
                        valid, frame = capture.read()
                    finally:
                        capture.release()
                if not valid or frame is None:
                    raise HTTPException(422, '无法读取视频，请检查格式和文件完整性')
                media_type, mime = 'video', {'.mp4': 'video/mp4', '.mov': 'video/quicktime',
                                            '.avi': 'video/x-msvideo', '.webm': 'video/webm'}[suffix]
            else:
                raise HTTPException(422, '仅支持 JPG、PNG、WebP、MP4、MOV、AVI、WebM')
            media_id = new_id()
            key = f'retrieval-inputs/{user.id}/{media_id}{suffix}'
            try:
                app.state.object_store.put_bytes(app.state.settings.minio_bucket, key, payload, mime)
            except Exception:
                raise HTTPException(503, '素材存储暂时不可用，请稍后再试') from None
            with app.state.session_factory() as session:
                session.add(RetrievalMedia(id=media_id, created_by=user.id, filename=filename, media_type=media_type,
                                          bucket=app.state.settings.minio_bucket, object_key=key,
                                          size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
                session.add(AuditEvent(action='retrieval.media_uploaded', user_id=user.id,
                                       entity_type='retrieval_media', entity_id=media_id,
                                       payload_json=json.dumps({'size_bytes': len(payload), 'media_type': media_type})))
                session.commit()
            return {'media_id': media_id, 'filename': filename, 'media_type': media_type, 'size_bytes': len(payload)}
        finally:
            upload_slot.release()

    @app.get('/api/v1/retrieval/tasks')
    def list_tasks(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100), user=Depends(current_user)):
        with app.state.session_factory() as session:
            condition = RetrievalTask.created_by == user.id
            total = session.scalar(select(func.count()).select_from(RetrievalTask).where(condition))
            rows = session.scalars(select(RetrievalTask).where(condition).order_by(RetrievalTask.created_at.desc()).offset(offset).limit(limit)).all()
            return {'tasks': [serialize(task, session.get(Job, task.job_id), detail=False) for task in rows], 'total': total}

    @app.post('/api/v1/retrieval/tasks', status_code=202)
    def create_task(payload: RetrievalRequest, idempotency_key: str = Header(..., min_length=8, max_length=100), user=Depends(current_user)):
        enabled()
        if not app.state.settings.retrieval_service_base_url:
            raise HTTPException(503, '智能检索服务尚未配置')
        inputs = payload.model_dump()
        fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with app.state.session_factory() as session:
            previous = session.scalar(select(RetrievalTask).where(RetrievalTask.created_by == user.id,
                                                                  RetrievalTask.idempotency_key == idempotency_key))
            if previous:
                if previous.request_hash != fingerprint:
                    raise HTTPException(409, '同一提交标识对应的参数已改变，请重新发起')
                return serialize(previous, session.get(Job, previous.job_id))
            ids = set(payload.media_ids + payload.prompt_media_ids)
            media = {item.id: item for item in session.scalars(select(RetrievalMedia).where(
                RetrievalMedia.id.in_(ids), RetrievalMedia.created_by == user.id)).all()}
            if set(media) != ids:
                raise HTTPException(404, '部分素材不存在或无权访问')
            if any(media[item].media_type != 'image' for item in payload.prompt_media_ids):
                raise HTTPException(422, '参考素材必须为图片')
            def media_url(item_id):
                item = media[item_id]
                if not app.state.object_store.exists(item.bucket, item.object_key):
                    raise HTTPException(503, '素材暂时无法读取，请重新上传或稍后重试')
                url = app.state.object_store.presigned_get_url(item.bucket, item.object_key, expires_seconds=86400)
                if not url:
                    raise HTTPException(503, '素材链接暂时不可用，请稍后重试')
                return url
            outgoing = {'file_urls': [media_url(item) for item in payload.media_ids],
                        'confidence': payload.confidence, 'sample_interval_seconds': payload.sample_interval_seconds}
            if payload.prompt_media_ids:
                outgoing.update(prompt_image_urls=[media_url(item) for item in payload.prompt_media_ids], text_query=payload.query or None)
            else:
                outgoing['text'] = payload.query
            task_id, job_id = new_id(), new_id()
            job = Job(id=job_id, job_type='retrieval', reference_id=task_id, total=len(payload.media_ids), status='submitting', started_at=utc_now())
            task = RetrievalTask(id=task_id, created_by=user.id, job_id=job_id, idempotency_key=idempotency_key,
                                 request_hash=fingerprint, input_json=json.dumps(inputs, ensure_ascii=False),
                                 snapshot_json=json.dumps({'message': '正在提交检索任务', 'total_hits': 0,
                                                          'processed_count': 0, 'total_count': len(payload.media_ids),
                                                          'video_progress': None, 'file_errors': [], 'error_code': None}))
            session.add(job)
            session.flush()
            session.add(task)
            session.add(AuditEvent(action='retrieval.created', user_id=user.id, entity_type='retrieval_task',
                                   entity_id=task_id, payload_json=json.dumps({'request_hash': fingerprint, 'job_id': job_id})))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                previous = session.scalar(select(RetrievalTask).where(RetrievalTask.created_by == user.id,
                                                                      RetrievalTask.idempotency_key == idempotency_key))
                if previous and previous.request_hash == fingerprint:
                    return serialize(previous, session.get(Job, previous.job_id))
                raise HTTPException(409, '提交发生冲突，请重新读取任务记录') from None
        try:
            remote_id = app.state.retrieval_adapter.submit(outgoing)
            new_status, message, error_code = 'running', '任务已提交，正在分析素材', None
        except httpx.HTTPStatusError as exc:
            remote_id = None
            if 400 <= exc.response.status_code < 500:
                new_status, message, error_code = 'failed', '算法拒绝创建任务，请检查服务授权或参数后重新发起', 'submission_rejected'
            else:
                new_status, message, error_code = 'submission_unknown', '创建结果尚未确认，请勿重复提交；请联系管理员核对算法任务', 'submission_unknown'
        except (httpx.HTTPError, RetrievalServiceError):
            remote_id = None
            new_status, message, error_code = 'submission_unknown', '创建结果尚未确认，请勿重复提交；请联系管理员核对算法任务', 'submission_unknown'
        with app.state.session_factory() as session:
            task, job = owned(session, task_id, user.id)
            task.remote_id = remote_id
            snapshot = json.loads(task.snapshot_json)
            snapshot.update(status=new_status, message=message, error_code=error_code)
            _save_snapshot(task, job, snapshot)
            session.commit()
            return serialize(task, job)

    @app.get('/api/v1/retrieval/tasks/{task_id}')
    def get_task(task_id: str, user=Depends(current_user)):
        with app.state.session_factory() as session:
            task, job = owned(session, task_id, user.id)
            if job.status in TERMINAL or not task.remote_id:
                started = job.started_at
                if job.status == 'submitting' and not task.remote_id and started:
                    started = started.replace(tzinfo=timezone.utc) if started.tzinfo is None else started
                    if (utc_now() - started).total_seconds() > max(120, app.state.settings.retrieval_timeout_seconds * 2):
                        snapshot = json.loads(task.snapshot_json)
                        snapshot.update(status='submission_unknown', error_code='submission_interrupted',
                                        message='服务重启前的提交结果尚未确认，请勿重复提交；请联系管理员核对算法任务')
                        _save_snapshot(task, job, snapshot)
                        session.commit()
                return serialize(task, job)
            remote_id = task.remote_id
            expected_revision = task.revision
            inputs = json.loads(task.input_json)
            media_keys = [session.get(RetrievalMedia, identifier).object_key for identifier in inputs['media_ids']]
        try:
            snapshot = _clean_snapshot(app.state.retrieval_adapter.get(remote_id), media_keys)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                with app.state.session_factory() as session:
                    task, job = owned(session, task_id, user.id)
                    return serialize(task, job, sync_error='算法进度暂时无法读取，请检查服务连接或授权后重新读取')
            with app.state.session_factory() as session:
                task, job = owned(session, task_id, user.id)
                snapshot = json.loads(task.snapshot_json)
                snapshot.update(status='failed', error_code='algorithm_task_missing',
                                message='算法任务已不存在，无法继续读取；已保存的结果仍可查看')
                snapshot['results'] = json.loads(task.results_json)
        except (httpx.HTTPError, RetrievalServiceError, ValueError, TypeError):
            with app.state.session_factory() as session:
                task, job = owned(session, task_id, user.id)
                return serialize(task, job, sync_error='暂时无法读取算法进度，请稍后重新读取；任务不会被重新提交')
        with app.state.session_factory() as session:
            task = session.scalar(select(RetrievalTask).where(RetrievalTask.id == task_id).with_for_update())
            job = session.get(Job, task.job_id)
            if job.status not in TERMINAL and claim_snapshot(session, task, expected_revision):
                if job.status == 'stopping' and snapshot['status'] == 'running':
                    snapshot.update(status='stopping', message='正在停止，等待算法确认')
                _save_snapshot(task, job, snapshot)
                session.commit()
            else:
                session.rollback()
                task, job = owned(session, task_id, user.id)
            return serialize(task, job)

    @app.post('/api/v1/retrieval/tasks/{task_id}/cancel')
    def cancel_task(task_id: str, user=Depends(current_user)):
        enabled()
        with app.state.session_factory() as session:
            task, job = owned(session, task_id, user.id)
            if job.status in TERMINAL or job.status == 'stopping':
                return serialize(task, job)
            if not task.remote_id:
                raise HTTPException(409, '任务提交尚未确认，暂时无法停止')
            remote_id = task.remote_id
        try:
            app.state.retrieval_adapter.cancel(remote_id)
        except (httpx.HTTPError, RetrievalServiceError):
            raise HTTPException(503, '停止请求尚未确认，请先重新读取任务状态') from None
        with app.state.session_factory() as session:
            task = session.scalar(select(RetrievalTask).where(RetrievalTask.id == task_id).with_for_update())
            job = session.get(Job, task.job_id)
            if job.status in TERMINAL or job.status == 'stopping':
                return serialize(task, job)
            if not claim_snapshot(session, task, task.revision):
                task, job = owned(session, task_id, user.id)
                return serialize(task, job)
            snapshot = json.loads(task.snapshot_json)
            snapshot.update(status='stopping', message='正在停止，等待算法确认')
            snapshot['results'] = json.loads(task.results_json)
            _save_snapshot(task, job, snapshot)
            session.add(AuditEvent(action='retrieval.cancel_requested', user_id=user.id,
                                   entity_type='retrieval_task', entity_id=task.id, payload_json='{}'))
            session.commit()
            return serialize(task, job)
