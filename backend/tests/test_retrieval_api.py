from __future__ import annotations

import io
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app

from .test_annotation_flow import _login, _settings


class MediaStore:
    def __init__(self):
        self.objects = {}

    def put_bytes(self, bucket, key, data, content_type):
        self.objects[(bucket, key)] = data

    def exists(self, bucket, key):
        return (bucket, key) in self.objects

    def presigned_get_url(self, bucket, key, expires_seconds=900):
        return f'https://media.invalid/{bucket}/{key}?X-Amz-Signature=private'


class Algorithm:
    def __init__(self):
        self.submissions = []
        self.cancelled = []
        self.snapshot = {'status': 'running', 'processed_count': 0, 'total_count': 1,
                         'total_hits': 0, 'results': [], 'file_errors': []}
        self.failure = None

    def submit(self, payload):
        self.submissions.append(payload)
        if self.failure:
            raise self.failure
        return 'ab12cd34ef56ab78'

    def get(self, remote_id):
        if self.failure:
            raise self.failure
        return self.snapshot

    def cancel(self, remote_id):
        self.cancelled.append(remote_id)
        return {'status': 'stopping'}


def _enabled(tmp_path):
    store, algorithm = MediaStore(), Algorithm()
    settings = replace(_settings(tmp_path), retrieval_enabled=True, retrieval_service_base_url='https://algorithm.invalid')
    app = create_app(settings, object_store=store)
    app.state.retrieval_adapter = algorithm
    client = TestClient(app)
    return client, _login(client), algorithm, store, settings


def _media(client, auth):
    stream = io.BytesIO()
    Image.new('RGB', (32, 24), 'white').save(stream, format='PNG')
    response = client.post('/api/v1/retrieval/media', headers=auth,
                           files={'file': ('sample.png', stream.getvalue(), 'image/png')})
    assert response.status_code == 201, response.text
    return response.json()['media_id']


def _create(client, auth, media_id, key='fixed-submission-key', **changes):
    return client.post('/api/v1/retrieval/tasks', headers={**auth, 'Idempotency-Key': key},
                       json={'query': '人员', 'media_ids': [media_id], **changes})


def test_capabilities_exposes_disabled_retrieval_without_enabling_work(tmp_path):
    client = TestClient(create_app(_settings(tmp_path)))
    auth = _login(client)
    response = client.get('/api/v1/capabilities', headers=auth)
    assert response.status_code == 200, response.text
    assert response.json()['retrieval_enabled'] is False
    response = client.post('/api/v1/retrieval/tasks', headers={**auth, 'Idempotency-Key': 'disabled-test'},
                           json={'query': '人员', 'media_ids': ['missing']})
    assert response.status_code == 403, response.text


def test_retrieval_endpoints_require_authentication(tmp_path):
    client = TestClient(create_app(_settings(tmp_path)))
    for method, path in [('get', '/api/v1/retrieval/tasks'),
                         ('post', '/api/v1/retrieval/tasks'),
                         ('get', '/api/v1/retrieval/tasks/unknown'),
                         ('post', '/api/v1/retrieval/tasks/unknown/cancel')]:
        response = getattr(client, method)(path)
        assert response.status_code == 401, (path, response.text)


def test_create_reconcile_and_reopen_terminal_results_after_app_restart(tmp_path):
    client, auth, algorithm, store, settings = _enabled(tmp_path)
    media_id = _media(client, auth)
    response = _create(client, auth, media_id)
    assert response.status_code == 202, response.text
    task_id = response.json()['task_id']
    assert response.json()['status'] == 'running'
    assert len(algorithm.submissions) == 1
    assert algorithm.submissions[0]['sample_interval_seconds'] == 3
    assert algorithm.submissions[0]['confidence'] == 0.2
    algorithm.snapshot = {'status': 'completed', 'processed_count': 1, 'total_count': 1,
                          'total_hits': 2, 'file_errors': [], 'results': [{
                              'file_index': 1, 'frame_index': 0, 'time_seconds': 0,
                              'image_url': 'https://media.invalid/result.jpg',
                              'file_url': algorithm.submissions[0]['file_urls'][0],
                              'detections': [{'class_name': 'person', 'confidence': 0.81, 'bbox': [1, 2, 8, 10]},
                                             {'class_name': 'person', 'confidence': 0.73, 'bbox': [9, 2, 12, 10]}]}]}
    result = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
    assert result.status_code == 200, result.text
    assert result.json()['status'] == 'completed'
    assert result.json()['total_hits'] == 2
    assert len(result.json()['results']) == 1
    assert 'X-Amz-Signature' not in result.text
    restarted = TestClient(create_app(settings, object_store=store))
    reopened = restarted.get(f'/api/v1/retrieval/tasks/{task_id}', headers=_login(restarted))
    assert reopened.status_code == 200
    assert reopened.json()['results'] == result.json()['results']
    assert reopened.json()['status'] == 'completed'
    assert restarted.get('/api/v1/retrieval/tasks', headers=_login(restarted)).json()['total'] == 1


def test_idempotency_is_durable_and_rejects_changed_parameters(tmp_path):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    media_id = _media(client, auth)
    first = _create(client, auth, media_id)
    second = _create(client, auth, media_id)
    assert second.status_code == 202, second.text
    assert second.json()['task_id'] == first.json()['task_id']
    assert len(algorithm.submissions) == 1
    changed = _create(client, auth, media_id, query='车辆')
    assert changed.status_code == 409, changed.text


def test_temporary_poll_failure_preserves_task_and_does_not_resubmit(tmp_path):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    algorithm.failure = httpx.ReadTimeout('secret=do-not-expose')
    response = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
    assert response.status_code == 200
    assert response.json()['status'] == 'running'
    assert response.json()['sync_error']
    assert 'do-not-expose' not in response.text
    assert len(algorithm.submissions) == 1
    algorithm.failure = None
    algorithm.snapshot = {'status': 'completed', 'processed_count': 1, 'total_count': 1,
                          'total_hits': 0, 'results': [], 'file_errors': []}
    assert client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth).json()['status'] == 'completed'


def test_submit_timeout_is_unknown_and_never_automatically_retried(tmp_path):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    media_id = _media(client, auth)
    algorithm.failure = httpx.ReadTimeout('secret=do-not-expose')
    first = _create(client, auth, media_id)
    assert first.status_code == 202, first.text
    assert first.json()['status'] == 'submission_unknown'
    second = _create(client, auth, media_id)
    assert second.json()['task_id'] == first.json()['task_id']
    assert len(algorithm.submissions) == 1
    assert 'do-not-expose' not in first.text


def test_cancel_waits_for_algorithm_and_keeps_partial_hits(tmp_path):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    response = client.post(f'/api/v1/retrieval/tasks/{task_id}/cancel', headers=auth)
    assert response.status_code == 200
    assert response.json()['status'] == 'stopping'
    client.post(f'/api/v1/retrieval/tasks/{task_id}/cancel', headers=auth)
    assert len(algorithm.cancelled) == 1
    algorithm.snapshot = {'status': 'cancelled', 'results': [{'file_index': 1, 'detections': [
        {'class_name': 'person', 'confidence': 0.9, 'bbox': [0, 0, 3, 4]}]}], 'total_hits': 1}
    response = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
    assert response.json()['status'] == 'cancelled'
    assert response.json()['total_hits'] == 1


@pytest.mark.parametrize('query', ['', '   '])
def test_invalid_query_is_rejected_before_algorithm_work(tmp_path, query):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    response = _create(client, auth, _media(client, auth), query=query)
    assert response.status_code == 422
    assert algorithm.submissions == []


def test_upload_rejects_fake_image_and_unowned_media(tmp_path):
    client, auth, algorithm, store, _ = _enabled(tmp_path)
    response = client.post('/api/v1/retrieval/media', headers=auth,
                           files={'file': ('fake.mp4', b'not-a-video', 'video/mp4')})
    assert response.status_code == 422
    assert store.objects == {}
    response = _create(client, auth, 'missing-media')
    assert response.status_code == 404
    assert algorithm.submissions == []


def test_missing_stored_media_does_not_start_unusable_task(tmp_path):
    client, auth, algorithm, store, _ = _enabled(tmp_path)
    media_id = _media(client, auth)
    store.objects.clear()
    response = _create(client, auth, media_id)
    assert response.status_code == 503
    assert algorithm.submissions == []


def test_unconfigured_service_is_rejected_before_submission(tmp_path):
    client, auth, algorithm, _, settings = _enabled(tmp_path)
    media_id = _media(client, auth)
    client.app.state.settings = replace(settings, retrieval_service_base_url='')
    response = _create(client, auth, media_id)
    assert response.status_code == 503
    assert algorithm.submissions == []


def test_video_upload_decodes_real_video_and_keeps_file_identity(tmp_path):
    import cv2
    import numpy as np
    client, auth, _, store, _ = _enabled(tmp_path)
    path = tmp_path / 'tiny.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (32, 24))
    try:
        for _ in range(5):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    payload = path.read_bytes()
    response = client.post('/api/v1/retrieval/media', headers=auth,
                           files={'file': ('tiny.avi', payload, 'video/x-msvideo')})
    assert response.status_code == 201, response.text
    assert response.json()['media_type'] == 'video'
    assert response.json()['size_bytes'] == len(payload)
    assert list(store.objects.values()) == [payload]


def test_completed_with_failed_files_is_partial_and_full_failures_remain_visible(tmp_path):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    algorithm.snapshot = {'status': 'completed', 'total_hits': 1, 'processed_count': 2, 'total_count': 2,
        'results': [{'detections': [{'class_name': 'person', 'confidence': 0.8, 'bbox': [0, 0, 2, 4]}]}],
        'file_errors': [{'file_index': 2, 'phase': 'decode_failed', 'error': 'signed-url=secret'}]}
    response = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
    assert response.json()['status'] == 'partial'
    assert response.json()['total_count'] == 2
    assert len(response.json()['file_errors']) == 1
    assert 'signed-url' not in response.text


def test_other_account_cannot_read_cancel_or_reuse_owned_media(tmp_path):
    from app.models import User
    from app.security import hash_password
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    media_id = _media(client, auth)
    task_id = _create(client, auth, media_id).json()['task_id']
    with client.app.state.session_factory() as session:
        session.add(User(username='other', password_hash=hash_password('other-test-password')))
        session.commit()
    login = client.post('/api/v1/auth/login', json={'username': 'other', 'password': 'other-test-password'})
    other = {'Authorization': 'Bearer ' + login.json()['access_token']}
    assert client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=other).status_code == 404
    assert client.post(f'/api/v1/retrieval/tasks/{task_id}/cancel', headers=other).status_code == 404
    assert _create(client, other, media_id).status_code == 404
    assert client.get('/api/v1/retrieval/tasks', headers=other).json()['total'] == 0
    assert len(algorithm.submissions) == 1


def test_private_result_images_are_resigned_and_missing_file_index_is_mapped(tmp_path):
    client, auth, algorithm, _, settings = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    algorithm.snapshot = {'status': 'completed', 'total_hits': 1, 'results': [{
        'file_url': algorithm.submissions[0]['file_urls'][0],
        'result_image_minio_url': f'minio://{settings.minio_bucket}/ab12cd34ef56ab78/hit.jpg',
        'timestamp_sec_approx': None, 'video_frame_id': None,
        'detections': [{'class_name': 'person', 'confidence': 0.8, 'bbox': [1, 1, 3, 4]}]}]}
    response = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
    result = response.json()['results'][0]
    assert result['file_index'] == 0
    assert result['image_url'].startswith('https://media.invalid/')
    assert result['timestamp_seconds'] is None


def test_interrupted_submission_becomes_unknown_without_restarting_remote_work(tmp_path):
    from datetime import timedelta
    from app.models import Job, utc_now
    from app.services.retrieval import RetrievalTask
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    with client.app.state.session_factory() as session:
        task = session.get(RetrievalTask, task_id)
        task.remote_id = None
        job = session.get(Job, task.job_id)
        job.status = 'submitting'
        job.started_at = utc_now() - timedelta(minutes=5)
        session.commit()
    response = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
    assert response.json()['status'] == 'submission_unknown'
    assert len(algorithm.submissions) == 1


def test_late_cancel_cannot_overwrite_completed_results(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    entered, release = Event(), Event()
    def slow_cancel(remote_id):
        entered.set()
        assert release.wait(5)
        return {'status': 'stopping'}
    algorithm.cancel = slow_cancel
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, f'/api/v1/retrieval/tasks/{task_id}/cancel', headers=auth)
        assert entered.wait(5)
        algorithm.snapshot = {'status': 'completed', 'total_hits': 1, 'results': [{'detections': [
            {'class_name': 'person', 'confidence': 0.8, 'bbox': [0, 0, 2, 4]}]}]}
        completed = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
        assert completed.json()['status'] == 'completed'
        release.set()
        response = pending.result(timeout=5)
    assert response.json()['status'] == 'completed'
    assert response.json()['total_hits'] == 1


def test_out_of_order_running_poll_cannot_erase_newer_hits(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    entered, release = Event(), Event()
    old = dict(algorithm.snapshot)
    def slow_get(remote_id):
        entered.set()
        assert release.wait(5)
        return old
    algorithm.get = slow_get
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.get, f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
        assert entered.wait(5)
        algorithm.get = lambda remote_id: {'status': 'running', 'total_hits': 1, 'results': [{'detections': [
            {'class_name': 'person', 'confidence': 0.8, 'bbox': [0, 0, 2, 4]}]}]}
        assert client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth).json()['total_hits'] == 1
        release.set()
        response = pending.result(timeout=5)
    assert response.json()['total_hits'] == 1


def test_remote_missing_job_is_explicit_failure_instead_of_endless_running(tmp_path):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    task_id = _create(client, auth, _media(client, auth)).json()['task_id']
    response = httpx.Response(404, request=httpx.Request('GET', 'https://algorithm.invalid/job'))
    algorithm.failure = httpx.HTTPStatusError('not found', request=response.request, response=response)
    result = client.get(f'/api/v1/retrieval/tasks/{task_id}', headers=auth)
    assert result.json()['status'] == 'failed'
    assert result.json()['error_code'] == 'algorithm_task_missing'
    assert len(algorithm.submissions) == 1


def test_explicit_submission_rejection_is_failed_without_an_automatic_retry(tmp_path):
    client, auth, algorithm, _, _ = _enabled(tmp_path)
    media_id = _media(client, auth)
    rejection = httpx.Response(401, request=httpx.Request('POST', 'https://algorithm.invalid/job'))
    algorithm.failure = httpx.HTTPStatusError('rejected', request=rejection.request, response=rejection)
    response = _create(client, auth, media_id)
    assert response.status_code == 202
    assert response.json()['status'] == 'failed'
    assert response.json()['error_code'] == 'submission_rejected'
    assert _create(client, auth, media_id).json()['task_id'] == response.json()['task_id']
    assert len(algorithm.submissions) == 1
