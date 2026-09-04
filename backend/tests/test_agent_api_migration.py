from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from alembic import command
from alembic.config import Config
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import make_url

from app.schemas import AgentReviewRequestInput
from app.services.agent_api import _request_fingerprint


def _seed_legacy_schema(engine: sa.Engine) -> None:
    metadata = sa.MetaData()
    requests = sa.Table(
        "agent_review_requests",
        metadata,
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("cycle_id", sa.String(length=120), nullable=False),
        sa.Column("candidate_id", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=120), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("attempt_refs_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "cycle_id",
            "candidate_id",
            name="uq_agent_review_cycle_candidate",
        ),
    )
    versions = sa.Table(
        "alembic_version",
        metadata,
        sa.Column("version_num", sa.String(length=32), primary_key=True),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            requests.insert().values(
                id="request-1",
                cycle_id="cycle-1",
                candidate_id="candidate-1",
                reason_code="model_conflict",
                reason="GPU and GPT disagree",
                attempt_refs_json='["gpu-1","gpt-1"]',
                status="pending",
                created_at=datetime(2026, 9, 4, 8, tzinfo=timezone.utc),
            )
        )
        connection.execute(
            versions.insert().values(version_num="0003_agent_review_api")
        )


def _assert_upgraded(engine: sa.Engine) -> None:
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


def test_legacy_agent_review_table_is_upgraded_without_losing_requests(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "legacy-agent-review.sqlite3"
    database_url = f"sqlite:///{database_path}"
    engine = sa.create_engine(database_url)
    _seed_legacy_schema(engine)
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")
    command.upgrade(config, "head")

    _assert_upgraded(engine)


def test_postgres_legacy_agent_review_table_upgrades_in_isolated_schema(
    monkeypatch,
):
    postgres_url = os.getenv("TEST_POSTGRES_URL")
    if not postgres_url or make_url(postgres_url).get_backend_name() != "postgresql":
        pytest.skip("TEST_POSTGRES_URL must point to PostgreSQL")

    schema = f"platform_agent_migration_{uuid.uuid4().hex}"
    base_url = make_url(postgres_url)
    admin_engine = sa.create_engine(base_url, pool_pre_ping=True)
    with admin_engine.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))

    query = dict(base_url.query)
    query["options"] = f"-csearch_path={schema}"
    schema_url = base_url.set(query=query).render_as_string(hide_password=False)
    engine = sa.create_engine(schema_url, pool_pre_ping=True)
    try:
        _seed_legacy_schema(engine)
        monkeypatch.setenv("DATABASE_URL", schema_url)
        config = Config("alembic.ini")
        command.upgrade(config, "head")
        _assert_upgraded(engine)
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin_engine.dispose()
