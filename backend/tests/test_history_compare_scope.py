import json

import httpx
import numpy as np
import pytest

from app.services.history_compare import filter_task_changes


def result(count):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return {'status': 'matched', 'polygons': [[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]] * count,
            '_aligned': image, '_target': image, 'diff_summary': 'old'}


def mock_model(monkeypatch, batches):
    calls = []
    def post(self, url, **kwargs):
        calls.append(kwargs['json'])
        reply = batches.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(200, request=httpx.Request('POST', url),
                              json={'message': {'content': json.dumps({'regions': reply})}})
    monkeypatch.setattr(httpx.Client, 'post', post)
    return calls


def rows(decisions):
    return [{'id': i + 1, 'decision': decision, 'summary': '局部判断'} for i, decision in enumerate(decisions)]


def review(data):
    return filter_task_changes(data, base_url='http://model.invalid', model='test', timeout=60,
                               task_description='对比一下光伏板的变化')


def test_all_eleven_candidates_reviewed_before_no_change(monkeypatch):
    calls = mock_model(monkeypatch, [rows(['ignore'] * 8), rows(['ignore'] * 3)])
    data = result(11)
    outcome = review(data)
    assert len(calls) == 2
    assert all('光伏板' in c['messages'][0]['content'] for c in calls)
    assert data['polygons'] == [] and data['diff_summary'] == '没有明显变化'
    assert data['status'] == 'matched' and outcome['error'] is None


def test_relevant_change_in_second_batch_retained(monkeypatch):
    mock_model(monkeypatch, [rows(['ignore'] * 8), rows(['keep'])])
    data = result(9)
    review(data)
    assert len(data['polygons']) == 1


@pytest.mark.parametrize('reply', [rows(['uncertain']), [],
                                  [{'id': True, 'decision': 'keep', 'summary': 'bad'}],
                                  [{'id': 1, 'decision': 'invalid', 'summary': 'bad'}],
                                  httpx.ReadTimeout('timeout')])
def test_uncertainty_and_invalid_responses_never_mean_no_change(monkeypatch, reply):
    mock_model(monkeypatch, [reply])
    data = result(1)
    outcome = review(data)
    assert data['status'] == 'not_found' and not data['polygons']
    assert '没有明显变化' not in data['diff_summary'] and outcome['error']


def test_late_failure_preserves_only_reviewed_changes(monkeypatch):
    mock_model(monkeypatch, [rows(['keep'] + ['ignore'] * 7), httpx.ReadTimeout('timeout')])
    data = result(9)
    outcome = review(data)
    assert len(data['polygons']) == 1 and outcome['error']


def test_no_geometry_candidates_has_no_obvious_change():
    data = result(0)
    review(data)
    assert data['diff_summary'] == '没有明显变化'
