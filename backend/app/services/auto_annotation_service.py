"""Run one-request-per-sample visual consensus annotation jobs."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Annotation, AnnotationTask, Asset, AuditEvent, AutoAnnotationItem, AutoAnnotationRun, Candidate, Job, Sample, new_id, utc_now
from .auto_annotation_prompts import ROLE_ORDER, merge_agent_results


AUTO_ANNOTATION_CONFIDENCE_FLOOR = 0.6
_SECRET_KEY_RE = re.compile(r"(?i)(bearer\s+)[^\s,;]+")


def _safe_error(error: Exception | str) -> str:
    """Keep dependency errors useful without accidentally storing a token."""
    return _SECRET_KEY_RE.sub(r"\1[redacted]", str(error))[:2000]


def sanitize_options(value: Any) -> Any:
    """Remove credential-shaped keys before options enter the database/audit."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(secret_word in key_text for secret_word in ("api_key", "apikey", "authorization", "access_token", "secret", "password")):
                continue
            sanitized[str(key)] = sanitize_options(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_options(item) for item in value]
    return value


def create_auto_annotation_run(
    session: Session,
    *,
    sample_ids: list[str],
    options: dict[str, Any],
    model_name: str,
    created_by: str | None = None,
) -> tuple[AutoAnnotationRun, Job]:
    """Persist a run, its sample work items, and the generic job tracker."""
    run = AutoAnnotationRun(
        model_name=model_name,
        status="pending",
        input_sample_count=len(sample_ids),
        options_json=json.dumps(sanitize_options(options), ensure_ascii=False),
        created_by=created_by,
    )
    session.add(run)
    session.flush()
    for sample_id in sample_ids:
        session.add(AutoAnnotationItem(run_id=run.id, sample_id=sample_id, status="pending"))
    job = Job(job_type="auto_annotation", reference_id=run.id, total=len(sample_ids))
    session.add(job)
    session.add(AuditEvent(
        action="auto_annotation.created",
        user_id=created_by,
        entity_type="auto_annotation_run",
        entity_id=run.id,
        payload_json=json.dumps({"sample_count": len(sample_ids), "model_name": model_name}, ensure_ascii=False),
    ))
    session.flush()
    return run, job


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
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
    }


def _latest_candidates(session: Session, sample: Sample) -> tuple[Asset | None, list[Candidate]]:
    asset = session.scalar(select(Asset).where(Asset.sample_id == sample.id, Asset.kind == "original"))
    task = session.scalar(select(AnnotationTask).where(AnnotationTask.sample_id == sample.id).order_by(AnnotationTask.created_at.desc()))
    if task is None:
        return asset, []
    candidates = session.scalars(
        select(Candidate).where(
            Candidate.sample_id == sample.id,
            Candidate.model_run_id == task.model_run_id,
        ).order_by(Candidate.target_index)
    ).all()
    return asset, candidates


def _normalise_decisions(result: Any, candidates: list[Candidate]) -> dict[int, dict[str, Any]]:
    if not isinstance(result, dict):
        raise ValueError("auto annotation adapter returned an invalid object")
    decisions = result.get("decisions")
    if decisions is None and isinstance(result.get("results"), list):
        decisions = result["results"]
    if decisions is None and isinstance(result.get("votes"), list):
        decisions = [{"target_index": 0, "votes": result["votes"]}]
    if not isinstance(decisions, list):
        raise ValueError("auto annotation response has no decisions")
    expected = {candidate.target_index for candidate in candidates}
    parsed: dict[int, dict[str, Any]] = {}
    for position, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"decision {position} is invalid")
        try:
            target_index = int(decision.get("target_index", decision.get("candidate_index", position)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decision {position} has invalid target_index") from exc
        if target_index in parsed:
            raise ValueError(f"duplicate decision for target {target_index}")
        votes = decision.get("votes")
        if isinstance(votes, dict):
            # Preferred contract from auto_annotation_prompts.py: role names
            # are keys, making it impossible to silently reorder a vote.
            if set(votes) != set(ROLE_ORDER):
                raise ValueError(f"target {target_index} must have exactly the three role votes")
            vote_values = [votes[role] for role in ROLE_ORDER]
            vote_roles = list(ROLE_ORDER)
        elif isinstance(votes, list):
            if len(votes) != 3:
                raise ValueError(f"target {target_index} must have exactly three votes")
            vote_values = votes
            vote_roles = []
        else:
            raise ValueError(f"target {target_index} has invalid votes")
        normalized_votes = []
        for vote_position, vote in enumerate(vote_values):
            if not isinstance(vote, dict):
                raise ValueError(f"target {target_index} vote {vote_position} is invalid")
            if vote_roles and set(vote) != {"label", "confidence", "reason"}:
                raise ValueError(f"target {target_index} role vote {vote_position} has invalid fields")
            if not vote_roles:
                supplied_role = vote.get("role")
                expected_role = ROLE_ORDER[vote_position]
                if supplied_role is not None and str(supplied_role) != expected_role:
                    raise ValueError(f"target {target_index} vote {vote_position} has the wrong role")
            label = str(vote.get("label", "")).strip().lower()
            if label not in {"true", "false", "uncertain"}:
                raise ValueError(f"target {target_index} vote {vote_position} has invalid label")
            try:
                confidence = float(vote.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"target {target_index} vote {vote_position} has invalid confidence") from exc
            if not 0 <= confidence <= 1:
                raise ValueError(f"target {target_index} vote {vote_position} has invalid confidence")
            normalized_votes.append({
                "role": str(vote.get("role") or (vote_roles[vote_position] if vote_roles else ROLE_ORDER[vote_position])),
                "label": label,
                "confidence": confidence,
                "reason": str(vote.get("reason") or "")[:1000],
            })
        parsed[target_index] = {
            "target_index": target_index,
            "votes": normalized_votes,
            "votes_by_role": dict(zip(ROLE_ORDER, normalized_votes, strict=True)) if vote_roles else None,
            "consensus": decision.get("consensus"),
        }
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        extra = sorted(set(parsed) - expected)
        raise ValueError(f"decisions do not match candidates (missing={missing}, extra={extra})")
    return parsed


def decision_from_votes(votes: list[dict[str, Any]], *, votes_by_role: dict[str, dict[str, Any]] | None = None, consensus: Any = None) -> str:
    """Apply local fail-open consensus: only 2 confident votes can decide."""
    if votes_by_role is not None:
        strict_votes = {
            role: {key: vote[key] for key in ("label", "confidence", "reason")}
            for role, vote in votes_by_role.items()
        }
        merged = merge_agent_results({
            "votes": strict_votes,
            "consensus": consensus if isinstance(consensus, dict) else {"label": None, "confidence": 0.0, "reason": "local recomputation"},
        })
        return merged["label"] if merged is not None else "uncertain"
    eligible = [vote["label"] for vote in votes if vote["label"] in {"true", "false"} and vote["confidence"] >= AUTO_ANNOTATION_CONFIDENCE_FLOOR]
    counts = Counter(eligible)
    if counts["true"] >= 2 and counts["true"] > counts["false"]:
        return "true"
    if counts["false"] >= 2 and counts["false"] > counts["true"]:
        return "false"
    return "uncertain"


def _audit(session: Session, *, action: str, sample_id: str, candidate_id: str | None, payload: dict[str, Any]) -> None:
    session.add(AuditEvent(
        action=action,
        user_id=None,
        entity_type="candidate" if candidate_id else "sample",
        entity_id=candidate_id or sample_id,
        payload_json=json.dumps(sanitize_options(payload), ensure_ascii=False),
    ))


def _invoke_adapter(adapter: Any, image: dict[str, Any], candidates: list[dict[str, Any]], options: dict[str, Any]) -> Any:
    if hasattr(adapter, "review"):
        return adapter.review(image, candidates, options)
    adapter_options = dict(options)
    adapter_options["candidates"] = candidates
    return adapter.infer(image, adapter_options)


def _process_item(session: Session, item: AutoAnnotationItem, *, adapter: Any, options: dict[str, Any]) -> None:
    sample = session.get(Sample, item.sample_id)
    if sample is None:
        raise ValueError("sample not found")
    asset, candidates = _latest_candidates(session, sample)
    if asset is None or asset.status == "missing":
        raise ValueError("original asset not found")
    if not candidates:
        # There is no candidate to label. This is a successful no-op and does
        # not spend a network request on an image that has nothing to review.
        _audit(session, action="auto_annotation.no_candidates", sample_id=sample.id, candidate_id=None, payload={"status": "no_candidates"})
        return
    image = {"bucket": asset.bucket, "object_key": asset.object_key, "sha256": asset.sha256}
    candidate_payloads = [_candidate_payload(candidate) for candidate in candidates]
    result = _invoke_adapter(adapter, image, candidate_payloads, options)
    decisions = _normalise_decisions(result, candidates)
    for candidate in candidates:
        decision = decisions[candidate.target_index]
        votes = decision["votes"]
        label = decision_from_votes(votes, votes_by_role=decision.get("votes_by_role"), consensus=decision.get("consensus"))
        audit_payload = {
            "target_index": candidate.target_index,
            "votes": votes,
            "model_consensus": decision.get("consensus"),
            "local_decision": label,
            "confidence_floor": AUTO_ANNOTATION_CONFIDENCE_FLOOR,
        }
        if label == "uncertain":
            _audit(session, action="auto_annotation.uncertain", sample_id=sample.id, candidate_id=candidate.id, payload=audit_payload)
            continue
        annotation = session.scalar(select(Annotation).where(Annotation.candidate_id == candidate.id))
        if annotation is not None and annotation.label is not None:
            _audit(session, action="auto_annotation.skipped_existing", sample_id=sample.id, candidate_id=candidate.id, payload={**audit_payload, "existing_label": annotation.label})
            continue
        if annotation is None:
            annotation = Annotation(candidate_id=candidate.id, revision=0)
            session.add(annotation)
        else:
            annotation.revision = (annotation.revision or 0) + 1
        annotation.label = label
        annotation.note = "auto annotation consensus: " + "; ".join(
            f"{vote['role']}={vote['label']} ({vote['confidence']:.2f})" for vote in votes
        )
        annotation.labeled_by = None
        annotation.labeled_at = utc_now()
        session.flush()
        _audit(session, action="auto_annotation.applied", sample_id=sample.id, candidate_id=candidate.id, payload=audit_payload)


def execute_auto_annotation_job(*, session_factory, object_store, adapter, job_id: str) -> None:
    """Execute a job, isolating each sample's network/parse failure."""
    del object_store  # Kept in the signature for worker parity and injection.
    with session_factory() as session:
        job = session.get(Job, job_id)
        if job is None or job.job_type != "auto_annotation":
            return
        run = session.get(AutoAnnotationRun, job.reference_id)
        if run is None:
            job.status = "failed"
            job.error_summary = "auto annotation run not found"
            session.commit()
            return
        if job.status == "succeeded":
            return
        job.status = "running"
        job.started_at = utc_now()
        run.status = "running"
        session.commit()
        options = json.loads(run.options_json or "{}")
        items = session.scalars(select(AutoAnnotationItem).where(AutoAnnotationItem.run_id == run.id).order_by(AutoAnnotationItem.created_at)).all()
        for item in items:
            if item.status == "succeeded":
                continue
            try:
                _process_item(session, item, adapter=adapter, options=options)
                item.status = "succeeded"
                job.completed += 1
            except Exception as exc:
                # A database/serialization failure after an earlier candidate
                # in this sample was written must not leave a half-labelled
                # sample behind. Roll back the item transaction, then record
                # the isolated failure in a fresh session state.
                item_id = item.id
                session.rollback()
                job = session.get(Job, job_id)
                item = session.get(AutoAnnotationItem, item_id)
                error = _safe_error(exc)
                item.status = "failed"
                item.error_message = error
                job.failed += 1
                _audit(session, action="auto_annotation.failed", sample_id=item.sample_id, candidate_id=None, payload={"error": error})
            session.commit()

        run.status = "partial_failed" if job.failed and job.completed else ("failed" if job.failed else "succeeded")
        run.error_summary = "one or more samples failed" if job.failed else None
        run.finished_at = utc_now()
        job.status = run.status
        job.finished_at = utc_now()
        session.commit()
