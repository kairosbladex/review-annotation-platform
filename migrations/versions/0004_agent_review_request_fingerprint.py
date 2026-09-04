"""Upgrade legacy agent review requests with idempotency fingerprints."""

from __future__ import annotations

import hashlib
import json

import sqlalchemy as sa
from alembic import op


revision = "0004_agent_review_fingerprint"
down_revision = "0003_agent_review_api"
branch_labels = None
depends_on = None


def _fingerprint(row: sa.RowMapping) -> str:
    try:
        attempt_refs = json.loads(row["attempt_refs_json"] or "[]")
    except (TypeError, json.JSONDecodeError):
        attempt_refs = []
    payload = {
        "attempt_refs": attempt_refs,
        "candidate_id": row["candidate_id"],
        "cycle_id": row["cycle_id"],
        "reason": row["reason"],
        "reason_code": row["reason_code"],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("agent_review_requests"):
        return

    columns = {column["name"] for column in inspector.get_columns("agent_review_requests")}
    if "request_fingerprint" not in columns:
        op.add_column(
            "agent_review_requests",
            sa.Column("request_fingerprint", sa.String(length=64), nullable=True),
        )
        requests = sa.table(
            "agent_review_requests",
            sa.column("id", sa.String(length=32)),
            sa.column("cycle_id", sa.String(length=120)),
            sa.column("candidate_id", sa.String(length=32)),
            sa.column("reason_code", sa.String(length=120)),
            sa.column("reason", sa.Text()),
            sa.column("attempt_refs_json", sa.Text()),
            sa.column("request_fingerprint", sa.String(length=64)),
        )
        rows = bind.execute(
            sa.select(
                requests.c.id,
                requests.c.cycle_id,
                requests.c.candidate_id,
                requests.c.reason_code,
                requests.c.reason,
                requests.c.attempt_refs_json,
            )
        ).mappings()
        for row in rows:
            bind.execute(
                requests.update()
                .where(requests.c.id == row["id"])
                .values(request_fingerprint=_fingerprint(row))
            )
        with op.batch_alter_table("agent_review_requests") as batch_op:
            batch_op.alter_column(
                "request_fingerprint",
                existing_type=sa.String(length=64),
                nullable=False,
            )

    inspector = sa.inspect(bind)
    indexes = {
        index["name"] for index in inspector.get_indexes("agent_review_requests")
    }
    if "ix_agent_review_requests_created_at" not in indexes:
        op.create_index(
            "ix_agent_review_requests_created_at",
            "agent_review_requests",
            ["created_at"],
        )


def downgrade() -> None:
    # Review-request idempotency evidence is intentionally retained.
    pass
