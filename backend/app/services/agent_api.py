from __future__ import annotations

import base64
import json
from datetime import datetime

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    AgentReviewRequest,
    Annotation,
    Asset,
    Candidate,
    ModelRun,
    Sample,
    User,
)
from ..schemas import AgentReviewRequestInput


def _encode_cursor(timestamp: datetime, row_id: str) -> str:
    raw = json.dumps(
        {"timestamp": timestamp.isoformat(), "id": row_id},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(cursor + padding))
        return datetime.fromisoformat(value["timestamp"]), str(value["id"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid cursor") from exc


def _after(column, id_column, cursor: str | None):
    if not cursor:
        return None
    timestamp, row_id = _decode_cursor(cursor)
    return or_(column > timestamp, and_(column == timestamp, id_column > row_id))


def _image(asset: Asset) -> dict[str, str]:
    return {
        "bucket": asset.bucket,
        "object_key": asset.object_key,
        "sha256": asset.sha256,
    }


def _candidate_lineage(
    candidate: Candidate,
    sample: Sample,
    asset: Asset,
    run: ModelRun,
) -> dict:
    return {
        "candidate_id": candidate.id,
        "source_event_id": sample.source_event_id,
        "source_service": sample.source_service,
        "image": _image(asset),
        "class_name": candidate.class_name,
        "source_class_id": candidate.class_id,
        "bbox": [candidate.x1, candidate.y1, candidate.x2, candidate.y2],
        "initial_evidence": {
            "model_run_id": run.id,
            "model_name": run.model_name,
            "model_version": run.model_version,
            "confidence": candidate.confidence,
            "raw_decision": candidate.raw_decision,
            "yolo_decision": candidate.yolo_decision,
            "llm_decision": candidate.llm_decision,
            "reason_code": candidate.reason_code,
            "error_code": candidate.error_code,
            "reasoning": candidate.reasoning,
        },
    }


def list_agent_candidates(
    session: Session,
    *,
    cursor: str | None,
    limit: int,
) -> dict:
    query = (
        select(Candidate, Sample, Asset, ModelRun)
        .join(Sample, Sample.id == Candidate.sample_id)
        .join(
            Asset,
            and_(
                Asset.sample_id == Sample.id,
                Asset.kind == "original",
                Asset.status == "ready",
                Asset.sha256.is_not(None),
                func.length(Asset.sha256) == 64,
            ),
        )
        .join(ModelRun, ModelRun.id == Candidate.model_run_id)
        .order_by(Candidate.created_at, Candidate.id)
    )
    after = _after(Candidate.created_at, Candidate.id, cursor)
    if after is not None:
        query = query.where(after)
    rows = session.execute(query.limit(limit + 1)).all()
    page = rows[:limit]
    items = []
    for candidate, sample, asset, run in page:
        item = _candidate_lineage(candidate, sample, asset, run)
        item.update(
            {
                "created_at": candidate.created_at.isoformat(),
            }
        )
        items.append(item)
    next_cursor = None
    if page:
        candidate = page[-1][0]
        next_cursor = _encode_cursor(candidate.created_at, candidate.id)
    return {"items": items, "next_cursor": next_cursor}


def create_agent_review_request(
    session: Session,
    payload: AgentReviewRequestInput,
) -> tuple[AgentReviewRequest, bool]:
    candidate = session.get(Candidate, payload.candidate_id)
    if candidate is None:
        raise LookupError("candidate not found")
    existing = session.scalar(
        select(AgentReviewRequest).where(
            AgentReviewRequest.cycle_id == payload.cycle_id,
            AgentReviewRequest.candidate_id == payload.candidate_id,
        )
    )
    if existing is not None:
        return existing, False
    request = AgentReviewRequest(
        cycle_id=payload.cycle_id,
        candidate_id=payload.candidate_id,
        reason_code=payload.reason_code,
        reason=payload.reason,
        attempt_refs_json=json.dumps(payload.attempt_refs, ensure_ascii=False),
    )
    try:
        with session.begin_nested():
            session.add(request)
            session.flush()
    except IntegrityError:
        existing = session.scalar(
            select(AgentReviewRequest).where(
                AgentReviewRequest.cycle_id == payload.cycle_id,
                AgentReviewRequest.candidate_id == payload.candidate_id,
            )
        )
        if existing is None:
            raise
        return existing, False
    return request, True


def list_human_gold(
    session: Session,
    *,
    cursor: str | None,
    limit: int,
) -> dict:
    query = (
        select(Annotation, Candidate, Sample, Asset, User, ModelRun)
        .join(Candidate, Candidate.id == Annotation.candidate_id)
        .join(Sample, Sample.id == Candidate.sample_id)
        .join(User, User.id == Annotation.labeled_by)
        .join(ModelRun, ModelRun.id == Candidate.model_run_id)
        .join(
            Asset,
            and_(
                Asset.sample_id == Sample.id,
                Asset.kind == "original",
                Asset.status == "ready",
                Asset.sha256.is_not(None),
                func.length(Asset.sha256) == 64,
            ),
        )
        .where(
            Annotation.labeled_by.is_not(None),
            Annotation.labeled_at.is_not(None),
            Annotation.label.in_(("true", "false")),
        )
        .order_by(Annotation.labeled_at, Annotation.id)
    )
    after = _after(Annotation.labeled_at, Annotation.id, cursor)
    if after is not None:
        query = query.where(after)
    rows = session.execute(query.limit(limit + 1)).all()
    page = rows[:limit]
    items = []
    for annotation, candidate, sample, asset, user, run in page:
        item = _candidate_lineage(candidate, sample, asset, run)
        item.update(
            {
                "label": annotation.label,
                "revision": annotation.revision,
                "labeled_by": user.username,
                "labeled_at": annotation.labeled_at.isoformat(),
                "annotation_id": annotation.id,
            }
        )
        items.append(item)
    next_cursor = None
    if page:
        annotation = page[-1][0]
        next_cursor = _encode_cursor(annotation.labeled_at, annotation.id)
    return {"items": items, "next_cursor": next_cursor}
