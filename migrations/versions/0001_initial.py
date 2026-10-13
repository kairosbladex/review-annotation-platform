"""Create the review annotation platform schema."""

from alembic import op

from app.db import Base
from app import models  # noqa: F401 - register all model tables

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The first migration intentionally uses the domain metadata as the source
    # of truth. Later migrations should be explicit ALTER operations.
    Base.metadata.create_all(op.get_bind())


def downgrade() -> None:
    # No destructive downgrade for a data-bearing platform.
    pass
