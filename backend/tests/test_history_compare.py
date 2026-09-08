import io

import numpy as np
import pytest
from PIL import Image

from app.services.history_compare import compare_geometry, decode_image


def scene():
    rng = np.random.default_rng(42)
    image = rng.integers(30, 210, size=(480, 640, 3), dtype=np.uint8)
    import cv2
    return cv2.GaussianBlur(image, (5, 5), 0)


def test_identical_has_no_regions():
    image = scene()
    result = compare_geometry(image, image.copy())
    assert result['status'] == 'matched'
    assert result['polygons'] == []


def test_small_identical_images_are_comparable():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    result = compare_geometry(image, image.copy())
    assert result['status'] == 'matched'
    assert result['polygons'] == []


def test_global_brightness_shift_is_not_a_change():
    baseline = scene()
    result = compare_geometry(baseline, baseline + 15)
    assert result['status'] == 'matched'
    assert result['polygons'] == []


def test_two_separate_changes_stay_separate():
    baseline = scene()
    current = baseline.copy()
    current[100:170, 150:240] = [15, 35, 70]
    current[280:350, 390:490] = [15, 35, 70]
    result = compare_geometry(baseline, current)
    assert result['status'] == 'matched'
    assert len(result['polygons']) == 2
    centers = [np.mean(polygon, axis=0) for polygon in result['polygons']]
    assert any(abs(x - 195 / 640) < 0.04 and abs(y - 135 / 480) < 0.04 for x, y in centers)
    assert any(abs(x - 440 / 640) < 0.04 and abs(y - 315 / 480) < 0.04 for x, y in centers)
    assert all(0 <= coordinate <= 1 for polygon in result['polygons'] for point in polygon for coordinate in point)


def test_translation_is_not_a_change():
    import cv2
    baseline = scene()
    current = cv2.warpAffine(baseline, np.float32([[1, 0, 8], [0, 1, 5]]), (640, 480))
    result = compare_geometry(baseline, current)
    assert result['status'] == 'matched'
    assert result['polygons'] == []


def test_unrelated_images_are_not_reported_unchanged():
    result = compare_geometry(scene(), np.zeros((480, 640, 3), dtype=np.uint8))
    assert result['status'] == 'not_found'
    assert result['polygons'] == []


def test_invalid_image_rejected():
    with pytest.raises(ValueError):
        decode_image(b'not an image')


def test_decoder_loads_rgb():
    buf = io.BytesIO()
    Image.new('L', (64, 48), 120).save(buf, format='PNG')
    assert decode_image(buf.getvalue()).shape == (48, 64, 3)


def test_api_requires_login_and_validates_uploads(tmp_path):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app

    settings = Settings(database_url=f'sqlite:///{tmp_path / "compare.sqlite3"}',
                        admin_password='test-secret', jwt_secret='test-secret-for-history-comparison',
                        minio_auto_create_bucket=False)
    app = create_app(settings, object_store=object())
    with TestClient(app) as client:
        assert client.post('/api/v1/history-compare').status_code == 401
        token = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'test-secret'}).json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        buf = io.BytesIO()
        Image.fromarray(scene()).save(buf, format='PNG')
        payload = buf.getvalue()
        response = client.post('/api/v1/history-compare', headers=headers,
                               files={'baseline': ('history.png', payload), 'current': ('current.png', payload)})
        assert response.status_code == 200
        assert response.json()['change_count'] == 0
        assert response.json()['semantic']['status'] == 'not_needed'
        response = client.post('/api/v1/history-compare', headers=headers,
                               files={'baseline': ('invalid.png', b'invalid'), 'current': ('current.png', payload)})
        assert response.status_code == 422
        from app.services.history_compare import MAX_IMAGE_BYTES
        response = client.post('/api/v1/history-compare', headers=headers,
                               files={'baseline': ('large.png', b'x' * (MAX_IMAGE_BYTES + 1)),
                                      'current': ('current.png', payload)})
        assert response.status_code == 413


def test_semantic_failure_does_not_discard_geometry(monkeypatch):
    import httpx
    from app.services.history_compare import describe_changes

    baseline = scene()
    current = baseline.copy()
    current[100:180, 150:260] = 0
    result = compare_geometry(baseline, current)
    polygons = result['polygons'].copy()

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout('timeout')

    monkeypatch.setattr(httpx.Client, 'post', timeout)
    semantic = describe_changes(result, base_url='http://model.invalid', model='test', timeout=1)
    assert semantic['status'] == 'failed'
    assert result['polygons'] == polygons
    assert polygons


@pytest.mark.parametrize('fenced', [True, False])
def test_semantic_accepts_json_with_optional_markdown(monkeypatch, fenced):
    import httpx
    from app.services.history_compare import describe_changes

    text = '{"summary":"Current image has additional solar panels"}'
    if fenced:
        text = '```json\n' + text + '\n```'
    monkeypatch.setattr(httpx.Client, 'post', lambda *args, **kwargs: httpx.Response(
        200, json={'message': {'content': text}}, request=httpx.Request('POST', 'http://model/api/chat')))
    result = {'status': 'matched', 'polygons': [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]],
              '_aligned': np.zeros((100, 100, 3), np.uint8), '_target': np.zeros((100, 100, 3), np.uint8)}
    semantic = describe_changes(result, base_url='http://model', model='test', timeout=1)
    assert semantic['status'] == 'completed'
    assert semantic['region_count'] == 1
    assert semantic['summary'] == 'Current image has additional solar panels'
