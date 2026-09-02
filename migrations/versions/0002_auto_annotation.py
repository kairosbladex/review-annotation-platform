"""Add persisted one-request-per-sample auto-annotation jobs."""

from alembic import op

from app.db import Base
from app import models  # noqa: F401 - register all model tables


revision = "0002_auto_annotation"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The application uses create_all for local/dev bootstrapping. Creating
    # the metadata here keeps the migration compatible with that bootstrap
    # path while adding only the newly registered tables in an existing DB.
    bind = op.get_bind()
    for table in (models.AutoAnnotationRun.__table__, models.AutoAnnotationItem.__table__):
        table.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    # Data-bearing jobs are intentionally not dropped by downgrade.
    pass
