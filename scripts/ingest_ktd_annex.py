"""Ingest local ktd annex rows into the annotation platform without copying images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
import pymysql

from app.services.annex_import import build_review_event_from_annex


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.ktd.local"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--api-url", default=os.getenv("ANNOTATION_API_URL", "http://127.0.0.1:8190"))
    args = parser.parse_args()
    if args.env_file.is_file():
        _load_env_file(args.env_file)

    api_key = os.environ.get("SERVICE_API_KEY", 'REDACTED_FOR_GITHUB')
    image_bucket = os.environ.get("SOURCE_IMAGE_BUCKET", "minio-src")
    conn = pymysql.connect(
        host=os.environ.get("KTD_LOCAL_MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("KTD_LOCAL_MYSQL_PORT", "3307")),
        user=os.environ["KTD_LOCAL_MYSQL_USER"],
        password=os.environ["KTD_LOCAL_MYSQL_PASSWORD"],
        database=os.environ["KTD_LOCAL_MYSQL_DATABASE"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15,
        read_timeout=60,
    )
    created = 0
    skipped = 0
    failed = 0
    errors: list[str] = []
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, file_path, imgxy, create_time, date_time_original
                FROM sys_annex_file
                WHERE file_path LIKE %s
                  AND imgxy LIKE %s
                  AND imgxy NOT LIKE %s
                ORDER BY create_time DESC, id
                LIMIT %s OFFSET %s
                """,
                ("/zm-zq-test/%", "%width%", "%type%", args.limit, args.offset),
            )
            rows = cur.fetchall()

    with httpx.Client(timeout=30, trust_env=False) as client:
        for row in rows:
            try:
                event = build_review_event_from_annex(row, image_bucket=image_bucket)
                response = client.post(
                    f"{args.api_url.rstrip('/')}/api/v1/ingest/review-events",
                    json=event.model_dump(mode="json"),
                    headers={"X-Service-Key": api_key},
                )
                response.raise_for_status()
                if response.json().get("created"):
                    created += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{row.get('id')}: {exc}")
    print({"read": len(rows), "created": created, "skipped": skipped, "failed": failed, "errors": errors[:10]})
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
