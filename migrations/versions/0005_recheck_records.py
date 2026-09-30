"""Dedicated GPT advice records, separate from human annotations."""
from alembic import op
from app.models import RecheckRecord
revision='0005_recheck_records'
down_revision='0004_agent_review_fingerprint'
branch_labels=None
depends_on=None

def upgrade():
    RecheckRecord.__table__.create(bind=op.get_bind(),checkfirst=True)

def downgrade():
    # Keep paid inference evidence on rollback. Disable the feature instead.
    pass
