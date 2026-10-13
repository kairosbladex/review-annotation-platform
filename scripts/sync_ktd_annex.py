"""Copy matching ktd sys_annex_file rows onto a local MySQL replica.

Images stay at the public HTTP prefix; this script only copies metadata.
Credentials come from the environment or an env file, never from git.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pymysql


MATCH_PREFIX = "/zm-zq-test/%"
WIDTH_PATTERN = "%width%"
TYPE_PATTERN = "%type%"


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _connect(prefix: str) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.environ[f"{prefix}_HOST"],
        port=int(os.environ.get(f"{prefix}_PORT", "3306")),
        user=os.environ[f"{prefix}_USER"],
        password=os.environ[f"{prefix}_PASSWORD"],
        database=os.environ[f"{prefix}_DATABASE"],
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=15,
        read_timeout=120,
        write_timeout=120,
    )


def _create_local_table(source, dest) -> None:
    with source.cursor() as cur:
        cur.execute("SHOW CREATE TABLE sys_annex_file")
        ddl = cur.fetchone()[1]
    with dest.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        cur.execute("DROP TABLE IF EXISTS sys_annex_file")
        cur.execute(ddl)
    dest.commit()


def _copy_rows(source, dest, *, limit: int | None) -> int:
    sql = """
        SELECT * FROM sys_annex_file
        WHERE file_path LIKE %s
          AND imgxy LIKE %s
          AND imgxy NOT LIKE %s
    """
    params: list = [MATCH_PREFIX, WIDTH_PATTERN, TYPE_PATTERN]
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with source.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [item[0] for item in cur.description]
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(f"`{name}`" for name in columns)
    update_sql = ", ".join(f"`{name}`=VALUES(`{name}`)" for name in columns if name != "id")
    insert_sql = (
        f"INSERT INTO sys_annex_file ({column_sql}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_sql}"
    )
    with dest.cursor() as cur:
        cur.executemany(insert_sql, rows)
    dest.commit()
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.ktd.local"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.env_file.is_file():
        _load_env_file(args.env_file)

    source = _connect("KTD_SOURCE_MYSQL")
    dest = _connect("KTD_LOCAL_MYSQL")
    try:
        _create_local_table(source, dest)
        copied = _copy_rows(source, dest, limit=args.limit)
    finally:
        source.close()
        dest.close()
    print({"copied": copied, "limit": args.limit})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
