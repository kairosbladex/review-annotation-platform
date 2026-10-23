from alembic import command
from alembic.config import Config
import sqlalchemy as sa

from app.db import Base
from app import models  # noqa: F401


def test_existing_database_gets_retrieval_tables_without_losing_jobs(tmp_path, monkeypatch):
    database_url = f'sqlite:///{tmp_path / "migration.sqlite3"}'
    engine = sa.create_engine(database_url)
    # Simulate an already deployed database, not create_app's create_all path.
    models.User.__table__.create(engine)
    models.Job.__table__.create(engine)
    monkeypatch.setenv('DATABASE_URL', database_url)
    config = Config('alembic.ini')
    command.stamp(config, '0004_agent_review_fingerprint')
    command.upgrade(config, 'head')
    tables = set(sa.inspect(engine).get_table_names())
    assert {'retrieval_tasks', 'retrieval_media', 'history_comparisons'} <= tables
    with engine.begin() as connection:
        connection.execute(sa.text("INSERT INTO users (id,username,password_hash,is_active,created_at) VALUES ('owner','owner','unused',1,CURRENT_TIMESTAMP)"))
        connection.execute(sa.text("INSERT INTO jobs (id,job_type,reference_id,status,total,completed,failed,created_at) VALUES ('old-job','model_run','old-run','succeeded',1,1,0,CURRENT_TIMESTAMP)"))
    command.downgrade(config, '0004_agent_review_fingerprint')
    with engine.connect() as connection:
        assert connection.scalar(sa.text("SELECT status FROM jobs WHERE id='old-job'")) == 'succeeded'
    command.upgrade(config, 'head')
    assert 'retrieval_tasks' in sa.inspect(engine).get_table_names()
