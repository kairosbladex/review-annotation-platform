"""Add the idempotent learning-agent human-review handoff table."""

from alembic import op

from app import models


revision = "0003_agent_review_api"
down_revision = "0002_auto_annotation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    models.AgentReviewRequest.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    # Review handoffs are audit records and are intentionally retained.
    pass
