from __future__ import annotations

from celery import Celery

from ..adapters.model_provider import build_model_adapter
from ..adapters.codex_responses import build_codex_vision_adapter
from ..adapters.object_store import build_object_store
from ..config import Settings
from ..db import build_engine, build_session_factory
from ..services.model_run_service import execute_model_job
from ..services.export_service import execute_export_job
from ..services.auto_annotation_service import execute_auto_annotation_job


settings = Settings.from_env()
celery_app = Celery("review_annotation_platform", broker=settings.redis_url)


@celery_app.task(name="review_annotation_platform.execute_model_job", bind=True, max_retries=2)
def execute_model_job_task(self, job_id: str):
    try:
        engine = build_engine(settings.database_url)
        factory = build_session_factory(engine)
        object_store = build_object_store(settings)
        adapter = build_model_adapter(settings, object_store)
        execute_model_job(session_factory=factory, object_store=object_store, adapter=adapter, job_id=job_id)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(name="review_annotation_platform.execute_export_job", bind=True, max_retries=2)
def execute_export_job_task(self, job_id: str):
    try:
        engine = build_engine(settings.database_url)
        factory = build_session_factory(engine)
        execute_export_job(session_factory=factory, object_store=build_object_store(settings), bucket=settings.minio_bucket, job_id=job_id)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(name="review_annotation_platform.execute_auto_annotation_job", bind=True, max_retries=2)
def execute_auto_annotation_job_task(self, job_id: str):
    try:
        engine = build_engine(settings.database_url)
        factory = build_session_factory(engine)
        object_store = build_object_store(settings)
        adapter = build_codex_vision_adapter(settings, object_store)
        execute_auto_annotation_job(
            session_factory=factory,
            object_store=object_store,
            adapter=adapter,
            job_id=job_id,
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)


# Dedicated queue: normal inference/export workers do not consume this queue.
@celery_app.task(name="review_annotation_platform.execute_recheck", acks_late=True, reject_on_worker_lost=True)
def execute_recheck_task(record_id: str):
    from ..services.online_recheck import execute
    from ..adapters.recheck_relay import RecheckRelay
    from .dispatcher import CeleryDispatcher
    if not settings.recheck_enabled:return
    factory=build_session_factory(build_engine(settings.database_url))
    delay=execute(record_id,factory=factory,settings=settings,store=build_object_store(settings),relay=_recheck_relay())
    if delay is not None:CeleryDispatcher(settings.redis_url).dispatch_recheck(record_id,countdown=delay)


_relay_instance=None

def _recheck_relay():
    global _relay_instance
    if _relay_instance is None:
        from ..adapters.recheck_relay import RecheckRelay
        _relay_instance=RecheckRelay(settings)
    return _relay_instance
