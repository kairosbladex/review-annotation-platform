"""Import the legacy review_annotation_store SQLite data without touching it.

The legacy store keeps filesystem paths and pixel-space boxes. This script
copies local assets to the configured object store, normalizes boxes, and
creates immutable review-event samples in the new PostgreSQL/SQLite schema.
It never updates or deletes the legacy database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from app.adapters.object_store import build_object_store
from app.config import Settings
from app.db import Base, build_engine, build_session_factory
from app.models import Annotation, Candidate, User
from app.schemas import ReviewEventInput
from app.security import hash_password
from app.services.annotation_service import ingest_review_event
from sqlalchemy import select


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except ImportError as exc:
        raise RuntimeError("pixel-space legacy boxes require Pillow: pip install Pillow") from exc


def _normalized_bbox(raw: str, image_path: Path) -> list[float]:
    bbox = json.loads(raw)
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("legacy bbox must contain four coordinates")
    values = [float(value) for value in bbox]
    if max(values) <= 1:
        normalized = values
    else:
        width, height = _image_size(image_path)
        if width <= 0 or height <= 0:
            raise ValueError("legacy image has invalid dimensions")
        normalized = [values[0] / width, values[1] / height, values[2] / width, values[3] / height]
    if any(value < 0 or value > 1 for value in normalized) or normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        raise ValueError("legacy bbox is outside image bounds")
    return normalized


def _timestamp(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _asset_key(sample_key: str, path_value: str | None, kind: str) -> tuple[str, str | None]:
    if not path_value:
        return "", None
    path = Path(path_value)
    suffix = path.suffix.lower() or ".jpg"
    return f"raw/legacy/{sample_key}/{kind}{suffix}", str(path)


def migrate(legacy_db: Path, settings: Settings) -> dict[str, int]:
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    store = build_object_store(settings)
    imported = skipped = failed = 0
    with sqlite3.connect(legacy_db) as legacy:
        legacy.row_factory = sqlite3.Row
        samples = legacy.execute("SELECT * FROM review_annotation_samples ORDER BY created_at, id").fetchall()
        with factory() as session:
            admin = session.scalar(select(User).where(User.username == settings.admin_username))
            if admin is None:
                admin = User(username=settings.admin_username, password_hash=hash_password(settings.admin_password))
                session.add(admin)
                session.flush()
            for old_sample in samples:
                old_targets = legacy.execute("SELECT * FROM review_annotation_targets WHERE sample_id = ? ORDER BY target_index", (old_sample["id"],)).fetchall()
                original_key, original_path_value = _asset_key(old_sample["id"], old_sample["original_path"], "original")
                comparison_key, comparison_path_value = _asset_key(old_sample["id"], old_sample["comparison_path"], "comparison")
                targets = []
                try:
                    if original_path_value and Path(original_path_value).is_file():
                        original_bytes = Path(original_path_value).read_bytes()
                        store.put_bytes(settings.minio_bucket, original_key, original_bytes, "image/jpeg")
                    if comparison_path_value and Path(comparison_path_value).is_file():
                        comparison_bytes = Path(comparison_path_value).read_bytes()
                        store.put_bytes(settings.minio_bucket, comparison_key, comparison_bytes, "image/jpeg")
                    for target in old_targets:
                        targets.append({
                            "target_index": int(target["target_index"]),
                            "class_name": target["class_name"],
                            "confidence": target["confidence"],
                            "bbox": _normalized_bbox(target["bbox_json"], Path(original_path_value)),
                            "raw_decision": target["raw_decision"],
                            "reasoning": target["reasoning"],
                            "yolo_decision": target["yolo_decision"],
                            "llm_decision": target["llm_decision"],
                            "reason_code": target["reason_code"],
                            "error_code": target["error_code"],
                        })
                    event = ReviewEventInput.model_validate({
                        "event_id": f"legacy:{old_sample['id']}",
                        "source_service": "legacy-sqlite",
                        "session_id": old_sample["session_id"],
                        "capture_time": _timestamp(old_sample["created_at"]),
                        "rule_code": old_sample["rule_code"],
                        "image": {"bucket": settings.minio_bucket, "object_key": original_key or old_sample["original_path"]},
                        "comparison_image": {"bucket": settings.minio_bucket, "object_key": comparison_key} if comparison_key else None,
                        "targets": targets,
                        "metadata": {"legacy_sample_id": old_sample["id"], "legacy_database": str(legacy_db)},
                    })
                    sample, created = ingest_review_event(session, event, store)
                    if not created:
                        skipped += 1
                        session.rollback()
                        continue
                    session.flush()
                    candidates = session.scalars(select(Candidate).where(Candidate.sample_id == sample.id).order_by(Candidate.target_index)).all()
                    for candidate, target in zip(candidates, old_targets):
                        if target["label"] is not None or target["note"] is not None:
                            session.add(Annotation(candidate_id=candidate.id, label=target["label"], note=target["note"], labeled_by=admin.id if target["label"] else None, labeled_at=_timestamp(target["labeled_at"]), revision=1 if target["label"] else 0))
                    session.commit()
                    imported += 1
                except Exception as exc:
                    session.rollback()
                    failed += 1
                    print(f"failed legacy sample {old_sample['id']}: {exc}")
    return {"total": len(samples), "imported": imported, "skipped": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_db", type=Path, help="path to the old annotations.sqlite3")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.database_url:
        settings = replace(settings, database_url=args.database_url)
    print(json.dumps(migrate(args.legacy_db, settings), ensure_ascii=False))


if __name__ == "__main__":
    main()
