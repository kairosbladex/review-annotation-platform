from __future__ import annotations

from alembic import command
from alembic.config import Config
import sqlalchemy as sa

from app.schemas import AgentReviewRequestInput
from app.services.agent_api import _request_fingerprint


def test_legacy_agent_review_table_is_upgraded_without_losing_requests(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "legacy-agent-review.sqlite3"
    database_url = f"sqlite:///{database_path}"
    engine = sa.create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                """
                CREATE TABLE agent_review_requests (
                    id VARCHAR(32) PRIMARY KEY,
                    cycle_id VARCHAR(120) NOT NULL,
                    candidate_id VARCHAR(32) NOT NULL,
                    reason_code VARCHAR(120) NOT NULL,
                    reason TEXT NOT NULL,
                    attempt_refs_json TEXT NOT NULL,
                    status VARCHAR(30) NOT NULL,
                    created_at DATETIME NOT NULL,
                    CONSTRAINT uq_agent_review_cycle_candidate
                        UNIQUE (cycle_id, candidate_id)
                )
                """
            )
        )
        connection.execute(
            sa.text(
                """
                CREATE TABLE alembic_version (
                    version_num VARCHAR(32) NOT NULL PRIMARY KEY
                )
                """
            )
        )
        connection.execute(
            sa.text(
                """
                INSERT INTO agent_review_requests (
                    id, cycle_id, candidate_id, reason_code, reason,
                    attempt_refs_json, status, created_at
                ) VALUES (
                    'request-1', 'cycle-1', 'candidate-1', 'model_conflict',
                    'GPU and GPT disagree', '["gpu-1","gpt-1"]', 'pending',
                    '2026-09-04 08:00:00'
                )
                """
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO alembic_version (version_num) "
                "VALUES ('0003_agent_review_api')"
            )
        )

    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")
    command.upgrade(config, "head")

    expected = _request_fingerprint(
        AgentReviewRequestInput(
            cycle_id="cycle-1",
            candidate_id="candidate-1",
            reason_code="model_conflict",
            reason="GPU and GPT disagree",
            attempt_refs=["gpu-1", "gpt-1"],
        )
    )
    inspector = sa.inspect(engine)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("agent_review_requests")
    }
    indexes = {index["name"] for index in inspector.get_indexes("agent_review_requests")}
    with engine.connect() as connection:
        row = connection.execute(
            sa.text(
                "SELECT id, request_fingerprint FROM agent_review_requests"
            )
        ).mappings().one()
        revision = connection.scalar(
            sa.text("SELECT version_num FROM alembic_version")
        )

    assert row == {"id": "request-1", "request_fingerprint": expected}
    assert columns["request_fingerprint"]["nullable"] is False
    assert "ix_agent_review_requests_created_at" in indexes
    assert revision == "0004_agent_review_fingerprint"
