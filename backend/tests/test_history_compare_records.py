from __future__ import annotations

import io
import hashlib

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.adapters.object_store import ObjectStore
from app.config import Settings
from app.db import Base, build_engine, build_session_factory
from app.models import User
from app.services.history_compare_records import HistoryComparison, register_history_record_routes


class MemoryObjectStore(ObjectStore):
    """Fake the external object-storage boundary, including expiring URLs."""

    def __init__(self):
        self.objects = {}
        self.signatures = 0

    def exists(self, bucket, object_key):
        return (bucket, object_key) in self.objects

    def put_bytes(self, bucket, object_key, data, content_type):
        self.objects[bucket, object_key] = (data, content_type)

    def get_bytes(self, bucket, object_key):
        return self.objects[bucket, object_key][0]

    def presigned_get_url(self, bucket, object_key, expires_seconds=900):
        self.signatures += 1
        return f'https://media.invalid/{bucket}/{object_key}?generation={self.signatures}'


def make_client(database_path, store):
    settings = Settings(database_url=f'sqlite:///{database_path}', history_compare_ollama_url='')
    app = FastAPI()
    app.state.settings = settings
    app.state.session_factory = build_session_factory(build_engine(settings.database_url))
    app.state.object_store = store
    Base.metadata.create_all(app.state.session_factory.kw['bind'])
    with app.state.session_factory() as session:
        for identifier in ('owner', 'other'):
            if session.get(User, identifier) is None:
                session.add(User(id=identifier, username=identifier, password_hash='unused-test-hash'))
        session.commit()

    def current_user(x_test_user: str | None = Header(default=None)):
        with app.state.session_factory() as session:
            user = session.get(User, x_test_user) if x_test_user else None
            if user is None:
                raise HTTPException(401, 'login required')
            return user

    register_history_record_routes(app, current_user)
    return TestClient(app)


def image_bytes(color=120, size=(48, 48)):
    stream = io.BytesIO()
    Image.new('RGB', size, (color, color, color)).save(stream, 'PNG')
    return stream.getvalue()


def create_comparison(client, baseline=None, current=None, user='owner'):
    return client.post('/api/v1/history-comparisons', headers={'X-Test-User': user}, files={
        'baseline': ('baseline.png', baseline if baseline is not None else image_bytes(), 'image/png'),
        'current': ('current.png', current if current is not None else image_bytes(), 'image/png'),
    })


def test_identical_pair_persists_and_reopens_with_fresh_media_urls(tmp_path):
    path = tmp_path / 'history.sqlite3'
    store = MemoryObjectStore()
    client = make_client(path, store)
    response = create_comparison(client)
    assert response.status_code == 201
    created = response.json()
    assert created['result']['status'] == 'matched'
    assert created['result']['change_count'] == 0
    assert created['result']['semantic']['status'] == 'not_needed'
    assert created['comparison_id'] and created['created_at']
    assert len(store.objects) == 2
    assert all(value[0] == image_bytes() for value in store.objects.values())

    replacement = make_client(path, store)
    reopened = replacement.get('/api/v1/history-comparisons/' + created['comparison_id'],
                               headers={'X-Test-User': 'owner'})
    assert reopened.status_code == 200
    restored = reopened.json()
    assert restored['result'] == created['result']
    assert restored['created_at'] == created['created_at']
    assert restored['baseline_url'] != created['baseline_url']
    assert restored['current_url'] != created['current_url']
    with replacement.app.state.session_factory() as session:
        row = session.get(HistoryComparison, created['comparison_id'])
        assert row.created_by == 'owner'
        assert row.baseline_sha256 == hashlib.sha256(image_bytes()).hexdigest()
        assert row.current_sha256 == hashlib.sha256(image_bytes()).hexdigest()


def test_missing_stored_image_returns_explicit_read_failure(tmp_path):
    store = MemoryObjectStore()
    client = make_client(tmp_path / 'history.sqlite3', store)
    comparison = create_comparison(client).json()
    store.objects.pop(next(iter(store.objects)))
    response = client.get('/api/v1/history-comparisons/' + comparison['comparison_id'],
                          headers={'X-Test-User': 'owner'})
    assert response.status_code == 503


@pytest.mark.parametrize('bad_image,expected_status', [
    (b'not-an-image', 422),
    (b'', 422),
    (image_bytes(size=(16, 16)), 422),
    (b'x' * (20 * 1024 * 1024 + 1), 413),
])
def test_rejected_upload_does_not_save_media_or_success_record(tmp_path, bad_image, expected_status):
    store = MemoryObjectStore()
    client = make_client(tmp_path / 'history.sqlite3', store)
    assert create_comparison(client, baseline=bad_image).status_code == expected_status
    assert store.objects == {}
    listing = client.get('/api/v1/history-comparisons', headers={'X-Test-User': 'owner'}).json()
    assert listing == {'comparisons': [], 'total': 0}


def test_authentication_ownership_and_pagination(tmp_path):
    client = make_client(tmp_path / 'history.sqlite3', MemoryObjectStore())
    assert client.post('/api/v1/history-comparisons').status_code == 401
    assert client.get('/api/v1/history-comparisons').status_code == 401
    assert client.get('/api/v1/history-comparisons/not-a-record').status_code == 401
    records = [create_comparison(client).json() for _ in range(3)]
    outsider = create_comparison(client, user='other').json()
    headers = {'X-Test-User': 'owner'}
    first = client.get('/api/v1/history-comparisons?page=1&limit=2', headers=headers).json()
    second = client.get('/api/v1/history-comparisons?page=2&limit=2', headers=headers).json()
    assert first['total'] == second['total'] == 3
    assert [r['comparison_id'] for r in first['comparisons'] + second['comparisons']] == [r['comparison_id'] for r in reversed(records)]
    assert all(r['status'] == 'matched' and r['change_count'] == 0 and r['diff_summary'] for r in first['comparisons'])
    assert client.get('/api/v1/history-comparisons/' + outsider['comparison_id'], headers=headers).status_code == 404
    assert client.get('/api/v1/history-comparisons/' + records[0]['comparison_id'], headers={'X-Test-User': 'other'}).status_code == 404
    assert client.get('/api/v1/history-comparisons?page=0', headers=headers).status_code == 422
    assert client.get('/api/v1/history-comparisons?limit=101', headers=headers).status_code == 422


@pytest.mark.parametrize('fail_at', [1, 2])
def test_object_storage_write_failure_is_503_without_success_record(tmp_path, fail_at):
    class FailingStore(MemoryObjectStore):
        def put_bytes(self, bucket, object_key, data, content_type):
            if len(self.objects) + 1 == fail_at:
                raise RuntimeError('private-storage-diagnostic')
            super().put_bytes(bucket, object_key, data, content_type)

    store = FailingStore()
    client = make_client(tmp_path / 'history.sqlite3', store)
    response = create_comparison(client)
    assert response.status_code == 503
    assert 'private-storage-diagnostic' not in response.text
    assert client.get('/api/v1/history-comparisons', headers={'X-Test-User': 'owner'}).json()['total'] == 0


def test_object_url_failure_is_explicit_and_does_not_save_success(tmp_path):
    class NoUrlsStore(MemoryObjectStore):
        def presigned_get_url(self, bucket, object_key, expires_seconds=900):
            return None

    client = make_client(tmp_path / 'history.sqlite3', NoUrlsStore())
    assert create_comparison(client).status_code == 503
    assert client.get('/api/v1/history-comparisons', headers={'X-Test-User': 'owner'}).json()['total'] == 0


def test_unalignable_pair_preserves_not_found_result(tmp_path):
    client = make_client(tmp_path / 'history.sqlite3', MemoryObjectStore())
    response = create_comparison(client, baseline=image_bytes(color=0), current=image_bytes(color=240))
    assert response.status_code == 201
    result = response.json()['result']
    assert result['status'] == 'not_found'
    assert result['polygons'] == []
    assert result['diff_summary'] != '没有明显变化'
    records = client.get('/api/v1/history-comparisons', headers={'X-Test-User': 'owner'}).json()
    assert records['comparisons'][0]['status'] == 'not_found'


def test_existing_database_migrates_without_losing_comparison_on_downgrade(tmp_path, monkeypatch):
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    database_url = f'sqlite:///{tmp_path / "existing.sqlite3"}'
    monkeypatch.setenv('DATABASE_URL', database_url)
    engine = build_engine(database_url)
    # Model an existing deployment at the previous revision, without letting
    # create_all silently install the new feature before its migration runs.
    User.__table__.create(engine)
    config = Config('alembic.ini')
    command.stamp(config, '0004_agent_review_fingerprint')
    command.upgrade(config, 'head')
    assert sa.inspect(engine).has_table('history_comparisons')
    factory = build_session_factory(engine)
    with factory() as session:
        session.add(User(id='owner', username='owner', password_hash='unused'))
        session.commit()
        session.add(HistoryComparison(
            id='comparison', created_by='owner', bucket='review',
            baseline_object_key='baseline', current_object_key='current',
            baseline_sha256='a' * 64, current_sha256='b' * 64, result_json='{}',
        ))
        session.commit()
    command.downgrade(config, '0004_agent_review_fingerprint')
    command.upgrade(config, 'head')
    with factory() as session:
        assert session.get(HistoryComparison, 'comparison').baseline_sha256 == 'a' * 64
