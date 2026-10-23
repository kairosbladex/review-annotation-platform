from __future__ import annotations

import json
import hmac
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select

from .adapters.object_store import ObjectStore, build_object_store
from .adapters.codex_responses import build_codex_vision_adapter
from .adapters.model_provider import build_model_adapter
from .config import Settings
from .db import Base, build_engine, build_session_factory
from .models import Annotation, AnnotationTask, AuditEvent, Candidate, ExportRecord, Job, ModelProfile, ModelRun, ModelRunSample, Sample, User, AutoAnnotationRun
from .schemas import AgentReviewRequestInput, AnnotationUpdate, AutoAnnotationRequest, ExportRequest, LoginRequest, MinioImportRequest, ModelProfileCreate, ModelRunRequest, ReviewEventInput
from .security import decode_token, hash_password, issue_token, verify_password
from .services.annotation_service import import_minio_prefix, import_uploaded_bytes, ingest_review_event, serialize_sample, update_annotation
from .services.dataset_queue import DatasetQueueIndex, V2_DATASET_TAG
from .services.model_run_service import execute_model_job
from .services.export_service import execute_export_job
from .services.auto_annotation_service import build_auto_annotation_options, create_auto_annotation_run, execute_auto_annotation_job
from .services.agent_api import create_agent_review_request, list_agent_candidates, list_human_gold
from .tasks.dispatcher import CeleryDispatcher


def create_app(settings: Settings | None = None, *, object_store: ObjectStore | None = None, model_adapter=None, task_dispatcher=None, export_dispatcher=None, auto_annotation_adapter=None, auto_annotation_dispatcher=None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    object_store = object_store or build_object_store(settings)
    if hasattr(object_store, "ensure_bucket"):
        try:
            object_store.ensure_bucket(settings.minio_bucket)
        except Exception:
            # A missing/unavailable object store is reflected as an asset error
            # by the ingestion path; API startup should stay inspectable.
            if settings.object_store_required:
                raise

    with session_factory() as session:
        admin = session.scalar(select(User).where(User.username == settings.admin_username))
        if admin is None:
            session.add(User(username=settings.admin_username, password_hash=hash_password(settings.admin_password)))
            session.commit()

    app = FastAPI(title="Review Annotation Platform", version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.object_store = object_store
    app.state.dataset_queue = DatasetQueueIndex.from_jsonl(settings.v2_queue_map_path)
    app.state.model_adapter = model_adapter or build_model_adapter(settings, object_store)
    app.state.auto_annotation_adapter = auto_annotation_adapter or build_codex_vision_adapter(settings, object_store)
    app.state.task_dispatcher = task_dispatcher or CeleryDispatcher(settings.redis_url)
    app.state.export_dispatcher = export_dispatcher or app.state.task_dispatcher
    app.state.auto_annotation_dispatcher = auto_annotation_dispatcher or app.state.task_dispatcher

    def run_model_job(job_id: str) -> None:
        execute_model_job(
            session_factory=app.state.session_factory,
            object_store=app.state.object_store,
            adapter=app.state.model_adapter,
            job_id=job_id,
        )

    app.state.run_model_job = run_model_job

    def run_auto_annotation_job(job_id: str) -> None:
        execute_auto_annotation_job(
            session_factory=app.state.session_factory,
            object_store=app.state.object_store,
            adapter=app.state.auto_annotation_adapter,
            job_id=job_id,
        )

    app.state.run_auto_annotation_job = run_auto_annotation_job

    def run_export_job(job_id: str) -> None:
        execute_export_job(
            session_factory=app.state.session_factory,
            object_store=app.state.object_store,
            bucket=settings.minio_bucket,
            job_id=job_id,
        )

    app.state.run_export_job = run_export_job

    @contextmanager
    def session_scope():
        session = app.state.session_factory()
        try:
            yield session
        finally:
            session.close()

    def current_user(request: Request, authorization: str | None = Header(default=None)) -> User:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要管理员登录")
        try:
            claims = decode_token(authorization.split(" ", 1)[1], settings.jwt_secret)
            user_id = claims["sub"]
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证无效") from exc
        with session_scope() as session:
            user = session.get(User, user_id)
            if user is None or not user.is_active:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已停用")
            return user

    def service_auth(x_service_key: str | None = Header(default=None)) -> None:
        if (
            not settings.service_api_key
            or not x_service_key
            or not hmac.compare_digest(
                x_service_key.encode("utf-8"), settings.service_api_key.encode("utf-8")
            )
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="服务密钥无效")

    history_compare_slots = threading.BoundedSemaphore(2)

    @app.post("/api/v1/history-compare")
    def history_compare(
        baseline: UploadFile = File(...),
        current: UploadFile = File(...),
        _: User = Depends(current_user),
    ):
        from .services.history_compare import MAX_IMAGE_BYTES, compare_images

        if not history_compare_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="历史对比繁忙，请稍后重试")
        try:
            reference_bytes = baseline.file.read(MAX_IMAGE_BYTES + 1)
            current_bytes = current.file.read(MAX_IMAGE_BYTES + 1)
            if max(len(reference_bytes), len(current_bytes)) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="单张图片不能超过20MB")
            return compare_images(
                reference_bytes, current_bytes,
                base_url=settings.history_compare_ollama_url,
                model=settings.history_compare_model,
                timeout=settings.history_compare_timeout_seconds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            history_compare_slots.release()

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/agent/candidates")
    def agent_candidates(
        cursor: str | None = None,
        limit: int = 100,
        _: None = Depends(service_auth),
    ):
        if limit < 1 or limit > 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        with session_scope() as session:
            try:
                return list_agent_candidates(session, cursor=cursor, limit=limit)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/agent/review-requests")
    def agent_review_request(
        payload: AgentReviewRequestInput,
        _: None = Depends(service_auth),
    ):
        with session_scope() as session:
            try:
                review_request, created = create_agent_review_request(session, payload)
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if created:
                session.add(
                    AuditEvent(
                        action="agent.review_requested",
                        user_id=None,
                        entity_type="candidate",
                        entity_id=review_request.candidate_id,
                        payload_json=json.dumps(
                            {
                                "cycle_id": review_request.cycle_id,
                                "reason_code": review_request.reason_code,
                                "review_request_id": review_request.id,
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
            session.commit()
            return JSONResponse(
                status_code=201 if created else 200,
                content={
                    "review_request_id": review_request.id,
                    "candidate_id": review_request.candidate_id,
                    "cycle_id": review_request.cycle_id,
                    "status": review_request.status,
                    "created": created,
                },
            )

    @app.get("/api/v1/agent/human-gold")
    def agent_human_gold(
        cursor: str | None = None,
        limit: int = 100,
        _: None = Depends(service_auth),
    ):
        if limit < 1 or limit > 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        with session_scope() as session:
            try:
                return list_human_gold(session, cursor=cursor, limit=limit)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/auth/login")
    def login(payload: LoginRequest):
        with session_scope() as session:
            user = session.scalar(select(User).where(User.username == payload.username))
            if user is None or not verify_password(payload.password, user.password_hash) or not user.is_active:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
            token = issue_token(user_id=user.id, username=user.username, secret=settings.jwt_secret, expires_minutes=settings.jwt_exp_minutes)
            session.add(AuditEvent(action="auth.login", user_id=user.id, entity_type="user", entity_id=user.id, payload_json="{}"))
            session.commit()
            return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "username": user.username}}

    @app.post("/api/v1/ingest/review-events", status_code=status.HTTP_201_CREATED)
    def ingest(payload: ReviewEventInput, _: None = Depends(service_auth)):
        with session_scope() as session:
            sample, created = ingest_review_event(session, payload, app.state.object_store)
            if created:
                session.add(AuditEvent(action="ingest.review_event", user_id=None, entity_type="sample", entity_id=sample.id, payload_json=json.dumps({"event_id": payload.event_id, "source_service": payload.source_service}, ensure_ascii=False)))
            session.commit()
            result = serialize_sample(session, sample, app.state.dataset_queue)
            return JSONResponse(
                status_code=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
                content={
                    "created": created,
                    "sample_id": sample.id,
                    "annotation_task_id": result["annotation_task_id"],
                    "status": result["status"],
                },
            )

    @app.post("/api/v1/model-profiles", status_code=status.HTTP_201_CREATED)
    def create_model_profile(payload: ModelProfileCreate, _: User = Depends(current_user)):
        with session_scope() as session:
            if session.scalar(select(ModelProfile).where(ModelProfile.profile_key == payload.profile_key)):
                raise HTTPException(status_code=409, detail="model profile 已存在")
            profile = ModelProfile(
                profile_key=payload.profile_key,
                name=payload.name,
                model_name=payload.model_name,
                model_version=payload.model_version,
                adapter_type=payload.adapter_type,
                config_json=json.dumps(payload.config, ensure_ascii=False),
            )
            session.add(profile)
            session.commit()
            return {"model_profile_id": profile.id, "profile_key": profile.profile_key, "model_version": profile.model_version}

    @app.post("/api/v1/imports", status_code=status.HTTP_201_CREATED)
    def import_from_minio(payload: MinioImportRequest, _: User = Depends(current_user)):
        with session_scope() as session:
            batch = import_minio_prefix(session, bucket=payload.bucket, prefix=payload.prefix, manifest_key=payload.manifest_key, object_store=app.state.object_store)
            session.commit()
            return {"import_id": batch.id, "status": batch.status, "total": batch.total, "imported": batch.imported, "skipped": batch.skipped, "failed": batch.failed}

    @app.post("/api/v1/imports/uploads", status_code=status.HTTP_201_CREATED)
    async def import_uploads(files: list[UploadFile] = File(...), _: User = Depends(current_user)):
        imported = 0
        skipped = 0
        errors: list[str] = []
        bucket = settings.minio_bucket
        with session_scope() as session:
            for file in files:
                filename = (file.filename or "upload.bin").replace("/", "_").replace("\\", "_")
                if not any(filename.lower().endswith(extension) for extension in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}):
                    errors.append(f"{filename}: unsupported image format")
                    continue
                data = await file.read()
                if len(data) > 50 * 1024 * 1024:
                    errors.append(f"{filename}: file too large")
                    continue
                key = f"raw/uploads/{uuid.uuid4().hex}/{filename}"
                try:
                    _, created = import_uploaded_bytes(session, bucket=bucket, object_key=key, data=data, content_type=file.content_type or "", object_store=app.state.object_store)
                    imported += int(created)
                    skipped += int(not created)
                except Exception as exc:
                    errors.append(f"{filename}: {exc}")
            session.commit()
        return {"imported": imported, "skipped": skipped, "failed": len(errors), "errors": errors}

    @app.get("/api/v1/model-profiles")
    def list_model_profiles(_: User = Depends(current_user)):
        with session_scope() as session:
            profiles = session.scalars(select(ModelProfile).where(ModelProfile.is_active.is_(True)).order_by(ModelProfile.created_at.desc())).all()
            return {"profiles": [{"model_profile_id": p.id, "profile_key": p.profile_key, "name": p.name, "model_name": p.model_name, "model_version": p.model_version, "adapter_type": p.adapter_type} for p in profiles]}

    @app.post("/api/v1/model-runs", status_code=status.HTTP_202_ACCEPTED)
    def create_model_run(payload: ModelRunRequest, user: User = Depends(current_user)):
        with session_scope() as session:
            profile = session.get(ModelProfile, payload.model_profile_id)
            if profile is None or not profile.is_active:
                raise HTTPException(status_code=404, detail="model profile 不存在")
            samples = session.scalars(select(Sample).where(Sample.id.in_(payload.sample_ids))).all()
            if len(samples) != len(set(payload.sample_ids)):
                raise HTTPException(status_code=404, detail="存在不存在的样本")
            run = ModelRun(
                model_profile_id=profile.id,
                model_name=profile.model_name,
                model_version=profile.model_version,
                source_type="model",
                status="pending",
                input_sample_count=len(samples),
                parameters_json=json.dumps(payload.options, ensure_ascii=False),
            )
            session.add(run)
            session.flush()
            for sample in samples:
                session.add(ModelRunSample(model_run_id=run.id, sample_id=sample.id))
            job = Job(job_type="model_run", reference_id=run.id, total=len(samples))
            session.add(job)
            session.add(AuditEvent(action="model_run.created", user_id=user.id, entity_type="model_run", entity_id=run.id, payload_json=json.dumps({"profile_key": profile.profile_key}, ensure_ascii=False)))
            session.commit()
            try:
                app.state.task_dispatcher.dispatch(job.id)
            except Exception as exc:
                job.status = "failed"
                job.error_summary = str(exc)
                run.status = "failed"
                session.commit()
                raise HTTPException(status_code=503, detail="模型任务无法入队") from exc
            session.expire_all()
            refreshed_run = session.get(ModelRun, run.id)
            return {"run_id": run.id, "job_id": job.id, "status": refreshed_run.status, "model_version": run.model_version}

    @app.get("/api/v1/model-runs/{run_id}")
    def get_model_run(run_id: str, _: User = Depends(current_user)):
        with session_scope() as session:
            run = session.get(ModelRun, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="model run 不存在")
            job = session.scalar(select(Job).where(Job.reference_id == run.id).order_by(Job.created_at.desc()))
            candidate_count = session.query(Candidate).filter(Candidate.model_run_id == run.id).count()
            return {"run_id": run.id, "model_name": run.model_name, "model_version": run.model_version, "status": run.status, "input_sample_count": run.input_sample_count, "candidate_count": candidate_count, "job": {"job_id": job.id, "status": job.status, "total": job.total, "completed": job.completed, "failed": job.failed} if job else None}

    @app.post("/api/v1/auto-annotations", status_code=status.HTTP_202_ACCEPTED)
    def create_auto_annotations(payload: AutoAnnotationRequest, user: User = Depends(current_user)):
        # Keep first occurrence order while preventing duplicate work items.
        sample_ids = list(dict.fromkeys(payload.sample_ids))
        with session_scope() as session:
            samples = session.scalars(select(Sample).where(Sample.id.in_(sample_ids))).all()
            if len(samples) != len(sample_ids):
                raise HTTPException(status_code=404, detail="存在不存在的样本")
            run, job = create_auto_annotation_run(
                session,
                sample_ids=sample_ids,
                options=build_auto_annotation_options(payload.options, settings),
                model_name=settings.auto_annotation_model,
                created_by=user.id,
            )
            session.commit()
            try:
                dispatcher = app.state.auto_annotation_dispatcher
                if hasattr(dispatcher, "dispatch_auto_annotation"):
                    dispatcher.dispatch_auto_annotation(job.id)
                else:
                    dispatcher.dispatch(job.id)
            except Exception as exc:
                run.status = "failed"
                run.error_summary = str(exc)
                job.status = "failed"
                job.error_summary = str(exc)
                session.commit()
                raise HTTPException(status_code=503, detail="自动标注任务无法入队") from exc
            session.expire_all()
            refreshed = session.get(Job, job.id)
            return {"job_id": refreshed.id, "status": refreshed.status, "total": refreshed.total, "completed": refreshed.completed, "failed": refreshed.failed}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, _: User = Depends(current_user)):
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="任务不存在")
            return {
                "job_id": job.id,
                "job_type": job.job_type,
                "reference_id": job.reference_id,
                "status": job.status,
                "total": job.total,
                "completed": job.completed,
                "failed": job.failed,
                "error_summary": job.error_summary,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            }

    @app.post("/api/v1/exports", status_code=status.HTTP_202_ACCEPTED)
    def create_export(payload: ExportRequest, user: User = Depends(current_user)):
        request_payload = payload
        with session_scope() as session:
            run = session.get(ModelRun, request_payload.run_id) if request_payload.run_id else None
            if run is None:
                raise HTTPException(status_code=404, detail="model run 不存在")
            record = ExportRecord(
                run_id=run.id,
                formats_json=json.dumps(request_payload.formats),
                filters_json=json.dumps(request_payload.filters, ensure_ascii=False),
                status="pending",
                created_by=user.id,
            )
            session.add(record)
            session.flush()
            job = Job(job_type="export", reference_id=record.id, total=1)
            session.add(job)
            session.add(AuditEvent(action="export.created", user_id=user.id, entity_type="export", entity_id=record.id, payload_json=json.dumps({"formats": request_payload.formats}, ensure_ascii=False)))
            session.commit()
            try:
                if hasattr(app.state.export_dispatcher, "dispatch_export"):
                    app.state.export_dispatcher.dispatch_export(job.id)
                else:
                    app.state.export_dispatcher.dispatch(job.id)
            except Exception as exc:
                record.status = "failed"
                record.error_summary = str(exc)
                job.status = "failed"
                job.error_summary = str(exc)
                session.commit()
                raise HTTPException(status_code=503, detail="导出任务无法入队") from exc
            session.expire_all()
            refreshed = session.get(ExportRecord, record.id)
            return {"export_id": refreshed.id, "job_id": job.id, "status": refreshed.status, "row_count": refreshed.row_count}

    @app.get("/api/v1/exports/{export_id}")
    def get_export(export_id: str, _: User = Depends(current_user)):
        with session_scope() as session:
            record = session.get(ExportRecord, export_id)
            if record is None:
                raise HTTPException(status_code=404, detail="导出不存在")
            job = session.scalar(select(Job).where(Job.reference_id == record.id).order_by(Job.created_at.desc()))
            return {"export_id": record.id, "run_id": record.run_id, "status": record.status, "formats": json.loads(record.formats_json or "[]"), "row_count": record.row_count, "object_key": record.object_key, "error_summary": record.error_summary, "job_id": job.id if job else None}

    @app.get("/api/v1/exports/{export_id}/content")
    @app.get("/api/v1/exports/{export_id}/download")
    def get_export_content(export_id: str, _: User = Depends(current_user)):
        from fastapi.responses import Response

        with session_scope() as session:
            record = session.get(ExportRecord, export_id)
            if record is None or record.status != "succeeded" or not record.object_key or not record.bucket:
                raise HTTPException(status_code=404, detail="导出内容不可用")
            try:
                content = app.state.object_store.get_bytes(record.bucket, record.object_key)
            except Exception as exc:
                raise HTTPException(status_code=404, detail="导出文件不可用") from exc
            media_type = "application/zip" if record.object_key.endswith(".zip") else "application/x-ndjson"
            return Response(content=content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="export-{record.id}{Path(record.object_key).suffix}"'})

    @app.get("/api/v1/model-runs/{run_id}/evaluation")
    def evaluate_model_run(run_id: str, _: User = Depends(current_user)):
        with session_scope() as session:
            run = session.get(ModelRun, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="model run 不存在")
            candidates = session.scalars(select(Candidate).where(Candidate.model_run_id == run_id)).all()
            annotations = {annotation.candidate_id: annotation for annotation in session.scalars(select(Annotation).where(Annotation.candidate_id.in_([candidate.id for candidate in candidates]))).all()} if candidates else {}
            labeled = [candidate for candidate in candidates if annotations.get(candidate.id) and annotations[candidate.id].label is not None]

            def precision(items: list[Candidate]) -> float | None:
                judged = [item for item in items if annotations.get(item.id) and annotations[item.id].label is not None]
                if not judged:
                    return None
                return round(sum(annotations[item.id].label == "true" for item in judged) / len(judged), 6)

            by_class = {}
            for class_name in sorted({candidate.class_name for candidate in candidates}):
                items = [candidate for candidate in candidates if candidate.class_name == class_name]
                by_class[class_name] = {"total_candidates": len(items), "labeled": sum(1 for item in items if annotations.get(item.id) and annotations[item.id].label is not None), "precision": precision(items)}
            buckets = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
            by_confidence = []
            for lower, upper in buckets:
                items = [candidate for candidate in candidates if candidate.confidence is not None and lower <= candidate.confidence < upper]
                by_confidence.append({"range": f"{lower:.2f}-{min(upper, 1.0):.2f}", "total_candidates": len(items), "labeled": sum(1 for item in items if annotations.get(item.id) and annotations[item.id].label is not None), "precision": precision(items)})
            by_reason = {}
            for reason_code in sorted({candidate.reason_code or "unknown" for candidate in candidates}):
                items = [candidate for candidate in candidates if (candidate.reason_code or "unknown") == reason_code]
                by_reason[reason_code] = {"total_candidates": len(items), "labeled": sum(1 for item in items if annotations.get(item.id) and annotations[item.id].label is not None), "precision": precision(items)}
            return {
                "run_id": run.id,
                "model_name": run.model_name,
                "model_version": run.model_version,
                "total_candidates": len(candidates),
                "labeled_candidates": len(labeled),
                "true_candidates": sum(annotations[item.id].label == "true" for item in labeled),
                "false_candidates": sum(annotations[item.id].label == "false" for item in labeled),
                "pending_candidates": len(candidates) - len(labeled),
                "candidate_precision": precision(candidates),
                "recall_available": False,
                "recall_note": "一期只标注模型已产生的候选框，未覆盖漏检目标，因此不能计算完整 recall。",
                "by_class": by_class,
                "by_confidence": by_confidence,
                "by_reason_code": by_reason,
            }

    @app.get("/api/v1/samples")
    def list_samples(
        status_filter: str = "all",
        limit: int = 50,
        offset: int = 0,
        source_type: str | None = None,
        model_version: str | None = None,
        class_name: str | None = None,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        raw_decision: str | None = None,
        reason_code: str | None = None,
        asset_missing: bool | None = None,
        model_failed: bool | None = None,
        dataset_filter: str = "all",
        _: User = Depends(current_user),
    ):
        if status_filter not in {"all", "imported", "pending", "in_progress", "completed", "no_candidates", "needs_review"}:
            raise HTTPException(status_code=400, detail="status_filter 无效")
        if dataset_filter not in {"all", V2_DATASET_TAG, "other"}:
            raise HTTPException(status_code=400, detail="dataset_filter 无效")
        with session_scope() as session:
            all_samples = session.scalars(select(Sample).order_by(Sample.created_at.desc())).all()
            samples: list[dict] = []
            for sample in all_samples:
                serialized = serialize_sample(session, sample, app.state.dataset_queue)
                has_v2_target = serialized["v2_target_count"] > 0
                v2_pending = serialized["v2_pending_count"]
                other_pending = sum(
                    candidate["annotation"]["label"] is None
                    for candidate in serialized["candidates"]
                    if candidate["dataset_tag"] != V2_DATASET_TAG
                )
                if dataset_filter == V2_DATASET_TAG and not has_v2_target:
                    continue
                if dataset_filter == "other" and has_v2_target:
                    continue
                if status_filter == "needs_review":
                    if dataset_filter == V2_DATASET_TAG:
                        if v2_pending == 0:
                            continue
                    elif dataset_filter == "other":
                        if other_pending == 0:
                            continue
                    elif serialized["status"] not in {"pending", "in_progress"}:
                        continue
                elif status_filter != "all" and serialized["status"] != status_filter:
                    continue
                if source_type and serialized["source_type"] != source_type:
                    continue
                if asset_missing is not None and (serialized["asset_status"] == "asset_missing") != asset_missing:
                    continue
                runs = serialized["runs"]
                if model_version and not any(run["model_run"].get("model_version") == model_version for run in runs):
                    continue
                candidates = [candidate for run in runs for candidate in run["candidates"]]
                scoped_candidates = [
                    candidate
                    for candidate in candidates
                    if dataset_filter == "all"
                    or (dataset_filter == V2_DATASET_TAG and candidate["dataset_tag"] == V2_DATASET_TAG)
                    or (dataset_filter == "other" and candidate["dataset_tag"] != V2_DATASET_TAG)
                ]
                if class_name and not any(candidate["class_name"] == class_name for candidate in scoped_candidates):
                    continue
                if confidence_min is not None and not any(candidate["confidence"] is not None and candidate["confidence"] >= confidence_min for candidate in scoped_candidates):
                    continue
                if confidence_max is not None and not any(candidate["confidence"] is not None and candidate["confidence"] <= confidence_max for candidate in scoped_candidates):
                    continue
                if raw_decision and not any(candidate["raw_decision"] == raw_decision for candidate in scoped_candidates):
                    continue
                if reason_code and not any(candidate["reason_code"] == reason_code for candidate in scoped_candidates):
                    continue
                if model_failed is not None and any(run["model_run"].get("status") in {"failed", "partial_failed"} for run in runs) != model_failed:
                    continue
                samples.append(serialized)
            total = len(samples)
            page_limit = max(1, min(limit, 200))
            page_offset = max(offset, 0)
            samples = samples[page_offset: page_offset + page_limit]
            return {"samples": samples, "pagination": {"limit": page_limit, "offset": page_offset, "total": total}}

    @app.get("/api/v1/samples/{sample_id}")
    def get_sample(sample_id: str, _: User = Depends(current_user)):
        with session_scope() as session:
            sample = session.get(Sample, sample_id)
            if sample is None:
                raise HTTPException(status_code=404, detail="样本不存在")
            return serialize_sample(session, sample, app.state.dataset_queue)

    @app.get("/api/v1/assets/{asset_id}/url")
    def get_asset_url(asset_id: str, _: User = Depends(current_user)):
        from .models import Asset

        with session_scope() as session:
            asset = session.get(Asset, asset_id)
            if asset is None or asset.status == "missing":
                raise HTTPException(status_code=404, detail="资源不存在")
            url = app.state.object_store.presigned_get_url(asset.bucket, asset.object_key)
            if not url:
                raise HTTPException(status_code=404, detail="资源暂不可访问")
            return {"asset_id": asset.id, "url": url, "expires_in": 900}

    @app.patch("/api/v1/candidates/{candidate_id}/annotation")
    def label_candidate(candidate_id: str, payload: AnnotationUpdate, user: User = Depends(current_user)):
        with session_scope() as session:
            candidate = session.get(Candidate, candidate_id)
            if candidate is None:
                raise HTTPException(status_code=404, detail="候选目标不存在")
            try:
                sample = update_annotation(session, candidate, user, payload)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            session.commit()
            return {"sample": serialize_sample(session, sample, app.state.dataset_queue)}

    return app


app = create_app()


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8190")))
