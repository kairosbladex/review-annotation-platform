"""Read-only retrieval of recently human-confirmed false positives.

The endpoint intentionally uses only committed human annotations.  Model
predictions, pending labels, and unreviewed notes never become prompt memory.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Annotation, Candidate, Sample, User


_ALIASES = {
    "vehicle": {"vehicle", "car", "truck", "van", "车辆", "车", "汽车", "机动车", "多类型车辆"},
    "person": {"person", "personnel", "pedestrian", "人员", "人", "行人"},
    "tricycle": {"tricycle", "三轮车", "三轮"},
    "blue_canopy_roof": {"blue_canopy_roof", "blue canopy roof", "彩钢棚", "违建", "蓝色违建彩钢棚"},
}


def _canonical(value: Any) -> str:
    text = re.sub(r"[_\-\s]*[vV]\d+(?:\.\d+)?$", "", str(value or "").strip().casefold())
    for canonical, aliases in _ALIASES.items():
        if text in {str(item).casefold() for item in aliases}:
            return canonical
    return text


def _scene_tags(metadata_json: str | None) -> set[str]:
    try:
        payload = json.loads(metadata_json or "{}")
    except (TypeError, ValueError):
        return set()
    raw = payload.get("scene_tags", payload.get("scene", payload.get("scene_tag", [])))
    if isinstance(raw, str):
        raw = re.split(r"[,;，；|]", raw)
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {str(item).strip().casefold() for item in raw if str(item).strip()}


def _description(candidate: Candidate, annotation: Annotation) -> str:
    # Notes are the human-authored feature description.  Reasoning is useful
    # for legacy rows that predate notes, while the code is a last-resort cue.
    for value in (annotation.note, candidate.reasoning, candidate.reason_code):
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text:
            return text[:500]
    return f"False positive visual lookalike for {candidate.class_name}"


def retrieve_negative_samples(
    session: Session,
    *,
    category: str,
    scene_tags: list[str] | tuple[str, ...] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Return a deterministic, recent, manually verified negative memory."""
    requested = _canonical(category)
    requested_scenes = {str(item).strip().casefold() for item in (scene_tags or []) if str(item).strip()}
    limit = max(1, min(int(top_k), 20))
    rows = session.execute(
        select(Candidate, Annotation, Sample, User)
        .join(Annotation, Annotation.candidate_id == Candidate.id)
        .join(Sample, Sample.id == Candidate.sample_id)
        .outerjoin(User, User.id == Annotation.labeled_by)
        .where(
            Annotation.label == "false",
            Annotation.labeled_by.is_not(None),
            Annotation.labeled_at.is_not(None),
        )
        .order_by(Annotation.labeled_at.desc(), Candidate.id)
        .limit(200)
    ).all()
    ranked: list[tuple[tuple[int, int, float], dict[str, Any]]] = []
    for candidate, annotation, sample, _user in rows:
        if requested and _canonical(candidate.class_name) != requested:
            continue
        tags = _scene_tags(sample.metadata_json)
        scene_match = bool(requested_scenes and tags.intersection(requested_scenes))
        labeled_at = annotation.labeled_at
        timestamp = labeled_at.timestamp() if isinstance(labeled_at, datetime) else 0.0
        item = {
            "description": _description(candidate, annotation),
            "class_name": candidate.class_name,
            "scene_tags": sorted(tags),
            "labeled_at": labeled_at.isoformat() if labeled_at else None,
            "candidate_id": candidate.id,
        }
        # Scene matches first, then newest labels.  Candidate ID makes ties stable.
        ranked.append(((1 if scene_match else 0, 1, timestamp), item))
    ranked.sort(key=lambda row: row[0], reverse=True)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, item in ranked:
        key = re.sub(r"\s+", " ", item["description"]).casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= limit:
            break
    text = "; ".join(item["description"] for item in items)
    return {
        "category": category,
        "canonical_category": requested,
        "scene_tags": sorted(requested_scenes),
        "top_k": limit,
        "items": items,
        "negative_rag_text": text,
        "source": "human_false_annotations",
    }
