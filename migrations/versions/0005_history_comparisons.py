"""Persist independently owned image-comparison inputs and full results."""

import sqlalchemy as sa
from alembic import op


revision = '0005_history_comparisons'
down_revision = '0004_agent_review_fingerprint'
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table('history_comparisons'):
        return
    op.create_table(
        'history_comparisons',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('created_by', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bucket', sa.String(length=255), nullable=False),
        sa.Column('baseline_object_key', sa.String(length=1024), nullable=False),
        sa.Column('current_object_key', sa.String(length=1024), nullable=False),
        sa.Column('baseline_sha256', sa.String(length=64), nullable=False),
        sa.Column('current_sha256', sa.String(length=64), nullable=False),
        sa.Column('result_json', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['created_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_history_comparisons_created_by', 'history_comparisons', ['created_by'])
    op.create_index('ix_history_comparisons_created_at', 'history_comparisons', ['created_at'])


def downgrade() -> None:
    # Retain comparison evidence and original-media references during rollback.
    pass
