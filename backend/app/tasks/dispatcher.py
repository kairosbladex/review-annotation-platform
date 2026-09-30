from __future__ import annotations


class CeleryDispatcher:
    def __init__(self, redis_url: str):
        from celery import Celery

        self.celery = Celery("review_annotation_platform", broker=redis_url)

    def dispatch(self, job_id: str) -> None:
        self.celery.send_task("review_annotation_platform.execute_model_job", args=[job_id])

    def dispatch_recheck(self, record_id: str, countdown: float = 0) -> None:
        self.celery.send_task("review_annotation_platform.execute_recheck", args=[record_id], countdown=countdown, queue="recheck", retry=False)

    def dispatch_export(self, job_id: str) -> None:
        self.celery.send_task("review_annotation_platform.execute_export_job", args=[job_id])

    def dispatch_auto_annotation(self, job_id: str) -> None:
        self.celery.send_task("review_annotation_platform.execute_auto_annotation_job", args=[job_id])


class InlineDispatcher:
    """Synchronous dispatcher for API/integration tests and local tooling."""

    def __init__(self, runner):
        self.runner = runner

    def dispatch(self, job_id: str) -> None:
        self.runner(job_id)

    def dispatch_auto_annotation(self, job_id: str) -> None:
        self.runner(job_id)

    def dispatch_export(self, job_id: str) -> None:
        self.runner(job_id)
