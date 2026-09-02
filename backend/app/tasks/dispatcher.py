from __future__ import annotations


class CeleryDispatcher:
    def __init__(self, redis_url: str):
        from celery import Celery

        self.celery = Celery("review_annotation_platform", broker=redis_url)

    def dispatch(self, job_id: str) -> None:
        self.celery.send_task("review_annotation_platform.execute_model_job", args=[job_id])

    def dispatch_export(self, job_id: str) -> None:
        self.celery.send_task("review_annotation_platform.execute_export_job", args=[job_id])
