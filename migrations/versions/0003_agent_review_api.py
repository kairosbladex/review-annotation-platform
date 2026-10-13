"""Add the idempotent learning-agent human-review handoff table."""

import sqlalchemy as sa
from alembic import op


revision = "0003_agent_review_api"
down_revision = "0002_auto_annotation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("agent_review_requests"):
        return
    op.create_table(
        "agent_review_requests",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("cycle_id", sa.String(length=120), nullable=False),
        sa.Column("candidate_id", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=120), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("attempt_refs_json", sa.Text(), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["candidate_id"], ["candidates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "cycle_id",
            "candidate_id",
            name="uq_agent_review_cycle_candidate",
        ),
    )
    op.create_index(
        op.f("ix_agent_review_requests_candidate_id"),
        "agent_review_requests",
        ["candidate_id"],
    )
    op.create_index(
        op.f("ix_agent_review_requests_created_at"),
        "agent_review_requests",
        ["created_at"],
    )
    op.create_index(
        op.f("ix_agent_review_requests_cycle_id"),
        "agent_review_requests",
        ["cycle_id"],
    )
    op.create_index(
        op.f("ix_agent_review_requests_reason_code"),
        "agent_review_requests",
        ["reason_code"],
    )
    op.create_index(
        op.f("ix_agent_review_requests_status"),
        "agent_review_requests",
        ["status"],
    )


def downgrade() -> None:
    # Review handoffs are audit records and are intentionally retained.
    pass
