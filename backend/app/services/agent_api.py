from __future__ import annotations

import base64
import hashlib
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


def _encode_cursor(
    kind: str,
    timestamp: datetime,
    row_id: str,
    revision: int | None = None,
) -> str:
    raw = json.dumps(
        {
            "kind": kind,
            "timestamp": timestamp.isoformat(),
            "id": row_id,
            "revision": revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str, *, kind: str) -> tuple[datetime, str, int | None]:
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.b64decode(cursor + padding, altchars=b"-_", validate=True))
        if not isinstance(value, dict) or value.get("kind") != kind:
            raise ValueError("cursor kind does not match endpoint")
        timestamp = datetime.fromisoformat(value["timestamp"])
        row_id = value["id"]
        revision = value.get("revision")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("cursor id is invalid")
        if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int)):
            raise ValueError("cursor revision is invalid")
        return timestamp, row_id, revision
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid cursor") from exc


def _after_candidate(column, id_column, cursor: str | None):
    if not cursor:
        return None
    timestamp, row_id, _revision = _decode_cursor(cursor, kind="candidate")
    return or_(column > timestamp, and_(column == timestamp, id_column > row_id))


def _after_gold(cursor: str | None):
    if not cursor:
        return None
    timestamp, candidate_id, revision = _decode_cursor(cursor, kind="human_gold")
    if revision is None:
        raise ValueError("invalid cursor")
    return or_(
        Annotation.labeled_at > timestamp,
        and_(
            Annotation.labeled_at == timestamp,
            Candidate.id > candidate_id,
        ),
        and_(
            Annotation.labeled_at == timestamp,
            Candidate.id == candidate_id,
            Annotation.revision > revision,
        ),
    )


def _is_sha256(value: str | None) -> bool:
    return bool(
        value
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        "created_at": candidate.created_at.isoformat(),
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
    after = _after_candidate(Candidate.created_at, Candidate.id, cursor)
    if after is not None:
        query = query.where(after)
    rows = session.execute(query.limit(limit * 2 + 1)).all()
    page = []
    last_scanned = None
    for row in rows:
        last_scanned = row[0]
        if _is_sha256(row[2].sha256):
            page.append(row)
        if len(page) == limit:
            break
    items = []
    for candidate, sample, asset, run in page:
        items.append(_candidate_lineage(candidate, sample, asset, run))
    next_cursor = None
    if last_scanned is not None:
        next_cursor = _encode_cursor(
            "candidate",
            last_scanned.created_at,
            last_scanned.id,
        )
    return {"items": items, "next_cursor": next_cursor}


def _request_fingerprint(payload: AgentReviewRequestInput) -> str:
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _same_request(existing: AgentReviewRequest, fingerprint: str) -> bool:
    return existing.request_fingerprint == fingerprint


def create_agent_review_request(
    session: Session,
    payload: AgentReviewRequestInput,
) -> tuple[AgentReviewRequest, bool]:
    candidate = session.get(Candidate, payload.candidate_id)
    if candidate is None:
        raise LookupError("candidate not found")
    fingerprint = _request_fingerprint(payload)
    existing = session.scalar(
        select(AgentReviewRequest).where(
            AgentReviewRequest.cycle_id == payload.cycle_id,
            AgentReviewRequest.candidate_id == payload.candidate_id,
        )
    )
    if existing is not None:
        if not _same_request(existing, fingerprint):
            raise ValueError("review request idempotency conflict")
        return existing, False
    request = AgentReviewRequest(
        cycle_id=payload.cycle_id,
        candidate_id=payload.candidate_id,
        reason_code=payload.reason_code,
        reason=payload.reason,
        attempt_refs_json=json.dumps(payload.attempt_refs, ensure_ascii=False),
        request_fingerprint=fingerprint,
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
        if not _same_request(existing, fingerprint):
            raise ValueError("review request idempotency conflict")
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
        .order_by(Annotation.labeled_at, Candidate.id, Annotation.revision)
    )
    after = _after_gold(cursor)
    if after is not None:
        query = query.where(after)
    rows = session.execute(query.limit(limit * 2 + 1)).all()
    page = []
    last_scanned = None
    for row in rows:
        last_scanned = row
        if _is_sha256(row[3].sha256):
            page.append(row)
        if len(page) == limit:
            break
    items = []
    for annotation, candidate, sample, asset, user, run in page:
        item = _candidate_lineage(candidate, sample, asset, run)
        item.update(
            {
                "label": annotation.label,
                "revision": annotation.revision,
                "labeled_by": user.username,
                "labeled_at": annotation.labeled_at.isoformat(),
                "label_source": "human",
            }
        )
        items.append(item)
    next_cursor = None
    if last_scanned is not None:
        annotation, candidate = last_scanned[0], last_scanned[1]
        next_cursor = _encode_cursor(
            "human_gold",
            annotation.labeled_at,
            candidate.id,
            annotation.revision,
        )
    return {"items": items, "next_cursor": next_cursor}
