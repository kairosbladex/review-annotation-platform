"""Prefill existing KTD candidate decisions from the GPU 8189 review service.

The script sends the existing boxes to the review endpoint. It never asks the
GPU service to rediscover boxes and it never writes human annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sqlite3
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


DEFAULT_SOURCE_BASE_URL = "https://lxtest.zhuoyilingxi.com:8088/minio-src"
DEFAULT_GPU_BASE_URL = "https://192.168.130.238:8189"
DEFAULT_REVIEW_ENDPOINT = "/api/v1/review-agent/review"
MAX_IMAGE_BYTES = 50 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_database_path(value: str) -> Path:
    prefix = "sqlite:///"
    if value.startswith(prefix):
        value = value[len(prefix) :]
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_bundles(database: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                s.id AS sample_id,
                s.created_at AS sample_created_at,
                asset.bucket AS bucket,
                asset.object_key AS object_key,
                candidate.id AS candidate_id,
                candidate.target_index AS target_index,
                candidate.class_id AS class_id,
                candidate.class_name AS class_name,
                candidate.confidence AS confidence,
                candidate.x1 AS x1,
                candidate.y1 AS y1,
                candidate.x2 AS x2,
                candidate.y2 AS y2
            FROM samples AS s
            JOIN assets AS asset
              ON asset.sample_id = s.id AND asset.kind = 'original'
            JOIN candidates AS candidate
              ON candidate.sample_id = s.id
             AND candidate.model_run_id = (
                 SELECT c2.model_run_id
                 FROM candidates AS c2
                 JOIN model_runs AS r2 ON r2.id = c2.model_run_id
                 WHERE c2.sample_id = s.id
                 ORDER BY r2.created_at DESC, r2.id DESC
                 LIMIT 1
             )
            ORDER BY s.created_at DESC, s.id, candidate.target_index
            """
        ).fetchall()
    finally:
        connection.close()

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        bundle = grouped.setdefault(
            row["sample_id"],
            {
                "sample_id": row["sample_id"],
                "sample_created_at": row["sample_created_at"],
                "bucket": row["bucket"],
                "object_key": row["object_key"],
                "candidates": [],
            },
        )
        bundle["candidates"].append(
            {
                "candidate_id": row["candidate_id"],
                "target_index": int(row["target_index"]),
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "confidence": row["confidence"],
                "bbox": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            }
        )
    return list(grouped.values())


def load_successful_samples(path: Path) -> set[str]:
    if not path.exists():
        return set()
    successful: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "succeeded" and record.get("applied") is True:
                successful.add(str(record.get("sample_id")))
    return successful


def source_url(base_url: str, object_key: str) -> str:
    return f"{base_url.rstrip('/')}/{quote(object_key.lstrip('/'), safe='/')}"


def normalize_decision(result: dict[str, Any]) -> tuple[str, str]:
    raw = result.get("rawDecision", result.get("raw_decision"))
    decision = result.get("decision")
    raw_text = str(raw).strip().lower() if raw is not None else ""
    decision_text = str(decision).strip().lower() if decision is not None else ""
    effective = raw_text or decision_text
    if effective in {"confirmed", "true", "keep"}:
        return "true", effective
    if effective in {"rejected", "false", "reject", "candidate_drop"}:
        return "false", effective
    if effective in {"uncertain", "corrected", "manual_review", "keep_manual_review"}:
        return "uncertain", effective
    return "uncertain", effective or "missing"


def close_enough(left: Any, right: list[float], tolerance: float = 1e-5) -> bool:
    return (
        isinstance(left, list)
        and len(left) == 4
        and all(abs(float(value) - expected) <= tolerance for value, expected in zip(left, right))
    )


def validate_results(bundle: dict[str, Any], body: dict[str, Any]) -> list[dict[str, Any]]:
    if body.get("success") is False:
        raise RuntimeError(str(body.get("error") or "review service returned failure"))
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("review service returned no results list")
    candidates = bundle["candidates"]
    if len(results) != len(candidates):
        raise RuntimeError(f"result count mismatch: sent {len(candidates)}, received {len(results)}")

    decisions = []
    for candidate, result in zip(candidates, results):
        if not isinstance(result, dict):
            raise RuntimeError(f"result {candidate['target_index']} is not an object")
        result_bbox = result.get("original_bbox") or result.get("bbox") or result.get("box")
        if not close_enough(result_bbox, candidate["bbox"]):
            raise RuntimeError(f"bbox mismatch at target_index {candidate['target_index']}")
        original_class = result.get("original_class")
        if original_class is not None and str(original_class) != str(candidate["class_name"]):
            raise RuntimeError(f"class mismatch at target_index {candidate['target_index']}")
        label, effective_decision = normalize_decision(result)
        decisions.append(
            {
                "candidate_id": candidate["candidate_id"],
                "target_index": candidate["target_index"],
                "class_name": candidate["class_name"],
                "bbox": candidate["bbox"],
                "model_label": label,
                "effective_decision": effective_decision,
                "raw_decision": result.get("rawDecision", result.get("raw_decision")),
                "decision": result.get("decision"),
                "yolo_decision": result.get("yolo_decision"),
                "llm_decision": result.get("llm_decision"),
                "reason_code": result.get("reasonCode", result.get("reason_code")),
                "error_code": result.get("errorCode", result.get("error_code")),
                "reasoning": result.get("reasoning") or "",
                "corrected_class": result.get("corrected_class"),
                "corrected_confidence": result.get("corrected_confidence"),
            }
        )
    return decisions


def apply_decisions(database: Path, decisions: list[dict[str, Any]]) -> None:
    connection = sqlite3.connect(database, timeout=30)
    try:
        connection.execute("BEGIN")
        for decision in decisions:
            cursor = connection.execute(
                """
                UPDATE candidates
                   SET raw_decision = ?,
                       yolo_decision = ?,
                       llm_decision = ?,
                       reason_code = ?,
                       error_code = ?,
                       reasoning = ?
                 WHERE id = ?
                """,
                (
                    decision["effective_decision"],
                    decision["yolo_decision"],
                    decision["llm_decision"],
                    decision["reason_code"],
                    decision["error_code"],
                    decision["reasoning"],
                    decision["candidate_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"candidate not found: {decision['candidate_id']}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/ktd-local.sqlite3", help="SQLite path or sqlite:/// URL")
    parser.add_argument("--source-base-url", default=os.getenv("KTD_SOURCE_IMAGE_BASE_URL", DEFAULT_SOURCE_BASE_URL))
    parser.add_argument("--gpu-base-url", default=os.getenv("GPU_REVIEW_BASE_URL", DEFAULT_GPU_BASE_URL))
    parser.add_argument("--review-endpoint", default=DEFAULT_REVIEW_ENDPOINT)
    parser.add_argument("--api-key", default=os.getenv("GPU_REVIEW_API_KEY", ""))
    parser.add_argument("--output-dir", default="data/ktd-8189-prefill-20260903")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None, help="process at most this many images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="collect responses without updating candidates")
    parser.add_argument("--no-verify-tls", action="store_true", help="allow the internal self-signed HTTPS certificates")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database = parse_database_path(args.database)
    if not database.is_file():
        print(f"database does not exist: {database}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "attempts.jsonl"
    summary_path = output_dir / "summary.json"
    bundles = load_bundles(database)
    if args.limit is not None:
        bundles = bundles[: max(0, args.limit)]
    successful = load_successful_samples(attempts_path) if args.resume else set()
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    verify = not args.no_verify_tls
    stats = {
        "database": str(database),
        "gpu_base_url": args.gpu_base_url.rstrip("/"),
        "review_endpoint": args.review_endpoint,
        "source_base_url": args.source_base_url.rstrip("/"),
        "started_at": utc_now(),
        "images_selected": len(bundles),
        "images_skipped_resume": 0,
        "images_succeeded": 0,
        "images_failed": 0,
        "candidates_sent": 0,
        "candidates_applied": 0,
        "model_true": 0,
        "model_false": 0,
        "model_uncertain": 0,
        "dry_run": args.dry_run,
    }

    with httpx.Client(
        timeout=httpx.Timeout(args.timeout_seconds),
        follow_redirects=True,
        trust_env=False,
        verify=verify,
        headers={"User-Agent": "ktd-8189-prefill/1.0"},
    ) as client:
        for position, bundle in enumerate(bundles, start=1):
            sample_id = bundle["sample_id"]
            if sample_id in successful:
                stats["images_skipped_resume"] += 1
                continue
            candidates = bundle["candidates"]
            detections = [
                {
                    "target_index": candidate["target_index"],
                    "class_id": int(candidate["class_id"] or 0),
                    "class_name": candidate["class_name"],
                    "confidence": float(candidate["confidence"] or 0.0),
                    "bbox": candidate["bbox"],
                }
                for candidate in candidates
            ]
            request_fingerprint = hashlib.sha256(
                json.dumps(detections, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            started_at = utc_now()
            started = time.monotonic()
            record: dict[str, Any] = {
                "attempt_no": position,
                "sample_id": sample_id,
                "bucket": bundle["bucket"],
                "object_key": bundle["object_key"],
                "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
                "target_indices": [candidate["target_index"] for candidate in candidates],
                "candidate_count": len(candidates),
                "request_fingerprint": request_fingerprint,
                "started_at": started_at,
            }
            stats["candidates_sent"] += len(candidates)
            try:
                image_url = source_url(args.source_base_url, bundle["object_key"])
                image_response = client.get(image_url)
                image_response.raise_for_status()
                image_bytes = image_response.content
                if len(image_bytes) > MAX_IMAGE_BYTES:
                    raise RuntimeError("source image is larger than 50 MiB")
                filename = Path(bundle["object_key"]).name or f"{sample_id}.jpg"
                media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                response = client.post(
                    f"{args.gpu_base_url.rstrip('/')}{args.review_endpoint}",
                    files={"image": (filename, image_bytes, media_type)},
                    data={
                        "detections": json.dumps(detections, ensure_ascii=False),
                        "yolo_conf_threshold": str(args.confidence),
                    },
                    headers=headers,
                )
                body_text = response.text
                try:
                    body = response.json()
                except ValueError as exc:
                    raise RuntimeError(f"invalid JSON response, HTTP {response.status_code}") from exc
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {body.get('error', body_text[:300])}")
                decisions = validate_results(bundle, body)
                if not args.dry_run:
                    apply_decisions(database, decisions)
                record.update(
                    {
                        "status": "succeeded",
                        "applied": not args.dry_run,
                        "http_status": response.status_code,
                        "duration_ms": round((time.monotonic() - started) * 1000, 1),
                        "finished_at": utc_now(),
                        "response": body,
                        "raw_response": body_text,
                        "decisions": decisions,
                    }
                )
                stats["images_succeeded"] += 1
                if not args.dry_run:
                    stats["candidates_applied"] += len(decisions)
                stats["model_true"] += sum(item["model_label"] == "true" for item in decisions)
                stats["model_false"] += sum(item["model_label"] == "false" for item in decisions)
                stats["model_uncertain"] += sum(item["model_label"] == "uncertain" for item in decisions)
                print(
                    f"[{position}/{len(bundles)}] {sample_id} succeeded "
                    f"candidates={len(decisions)} duration_ms={record['duration_ms']}",
                    flush=True,
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "failed",
                        "applied": False,
                        "duration_ms": round((time.monotonic() - started) * 1000, 1),
                        "finished_at": utc_now(),
                        "error": str(exc),
                    }
                )
                stats["images_failed"] += 1
                print(f"[{position}/{len(bundles)}] {sample_id} failed: {exc}", file=sys.stderr, flush=True)
            append_jsonl(attempts_path, record)
            stats["finished_at"] = utc_now()
            write_json(summary_path, stats)

    stats["finished_at"] = utc_now()
    write_json(summary_path, stats)
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if stats["images_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
