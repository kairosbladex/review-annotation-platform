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
        payload = reply if isinstance(reply, dict) else {'regions': reply}
        return httpx.Response(200, request=httpx.Request('POST', url),
                              json={'message': {'content': json.dumps(payload)}})
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
    assert outcome['summary'] is None


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
    assert outcome['summary'] == '1. 局部判断'


def test_summary_contains_only_retained_regions_in_polygon_order(monkeypatch):
    first = rows(['ignore', 'keep', 'uncertain', 'ignore', 'ignore', 'ignore', 'ignore', 'keep'])
    first[1]['summary'] = '历史图无车辆，当前图出现白色车辆。'
    first[7]['summary'] = '历史图路面可见，当前图被堆放物遮挡'
    # Model output order need not match input order, but descriptions must match polygons.
    mock_model(monkeypatch, [list(reversed(first)), rows(['keep'])])
    data = result(9)
    outcome = review(data)
    assert len(data['polygons']) == 3
    assert outcome['summary'] == ('1. 历史图无车辆，当前图出现白色车辆；'
                                  '2. 历史图路面可见，当前图被堆放物遮挡；3. 局部判断')
    assert outcome['error'] == '1处候选区域未能确认，需人工复核'


def test_no_geometry_candidates_has_no_obvious_change():
    data = result(0)
    review(data)
    assert data['diff_summary'] == '没有明显变化'


def test_kept_change_text_is_one_conclusion(monkeypatch):
    mock_model(monkeypatch, [{'regions': rows(['keep', 'ignore']),
                              'conclusion': '当前图屋面新增蓝色光伏板'}])
    data = result(2)
    review(data)
    assert data['diff_summary'] == '检测到1处明显变化：当前图屋面新增蓝色光伏板'


def test_missing_conclusion_keeps_count_only_text(monkeypatch):
    mock_model(monkeypatch, [rows(['keep'])])
    data = result(1)
    review(data)
    assert data['diff_summary'] == '检测到1处符合任务的明显变化'


def test_overlong_conclusion_is_not_used(monkeypatch):
    mock_model(monkeypatch, [{'regions': rows(['keep']), 'conclusion': '长' * 81}])
    data = result(1)
    review(data)
    assert data['diff_summary'] == '检测到1处符合任务的明显变化'


def test_conclusion_of_ignored_batch_is_not_used(monkeypatch):
    mock_model(monkeypatch, [{'regions': rows(['ignore']), 'conclusion': '当前图新增光伏板'}])
    data = result(1)
    review(data)
    assert data['diff_summary'] == '没有明显变化'


def test_batches_join_distinct_conclusions(monkeypatch):
    mock_model(monkeypatch, [
        {'regions': rows(['keep'] * 8), 'conclusion': '当前图新增光伏板'},
        {'regions': rows(['keep']), 'conclusion': '当前图新增光伏板'},
    ])
    data = result(9)
    outcome = review(data)
    assert data['diff_summary'] == '检测到9处明显变化：当前图新增光伏板'
    assert len(data['polygons']) == 9 and outcome['region_count'] == 9
