"""Add owned media and durable independent retrieval jobs."""
import sqlalchemy as sa
from alembic import op

revision = '0006_independent_retrieval'
down_revision = '0005_history_comparisons'
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table('retrieval_media'):
        op.create_table('retrieval_media',
            sa.Column('id', sa.String(32), primary_key=True),
            sa.Column('created_by', sa.String(32), sa.ForeignKey('users.id'), nullable=False),
            sa.Column('filename', sa.String(255), nullable=False),
            sa.Column('media_type', sa.String(16), nullable=False),
            sa.Column('bucket', sa.String(255), nullable=False),
            sa.Column('object_key', sa.String(1024), nullable=False),
            sa.Column('size_bytes', sa.Integer(), nullable=False),
            sa.Column('sha256', sa.String(64), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False))
        op.create_index('ix_retrieval_media_created_by', 'retrieval_media', ['created_by'])
    if not inspector.has_table('retrieval_tasks'):
        op.create_table('retrieval_tasks',
            sa.Column('id', sa.String(32), primary_key=True),
            sa.Column('created_by', sa.String(32), sa.ForeignKey('users.id'), nullable=False),
            sa.Column('job_id', sa.String(32), sa.ForeignKey('jobs.id'), nullable=False, unique=True),
            sa.Column('idempotency_key', sa.String(100), nullable=False),
            sa.Column('request_hash', sa.String(64), nullable=False),
            sa.Column('input_json', sa.Text(), nullable=False),
            sa.Column('remote_id', sa.String(64), nullable=True),
            sa.Column('snapshot_json', sa.Text(), nullable=False),
            sa.Column('results_json', sa.Text(), nullable=False),
            sa.Column('revision', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint('created_by', 'idempotency_key', name='uq_retrieval_submission'))
        op.create_index('ix_retrieval_tasks_created_by', 'retrieval_tasks', ['created_by'])
        op.create_index('ix_retrieval_tasks_created_at', 'retrieval_tasks', ['created_at'])


def downgrade():
    # Preserve job evidence and source lineage when reverting the application.
    pass
