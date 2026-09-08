import json

import httpx
import numpy as np
import pytest

from app.services.history_compare import compare_geometry, describe_changes
from .test_history_compare import scene


def test_local_color_shift_without_structure_is_not_marked():
    before = scene()
    after = before.copy()
    after[100:250, 160:320] = np.clip(after[100:250, 160:320].astype(int) + [40, 15, -15], 0, 255)
    result = compare_geometry(before, after)
    assert result['status'] == 'matched'
    assert result['polygons'] == []


def test_task_filters_unrelated_or_color_only_regions(monkeypatch):
    polygons = [[[.1, .1], [.4, .1], [.4, .4]], [[.6, .6], [.9, .6], [.9, .9]]]
    result = {'status': 'matched', 'polygons': polygons.copy(), 'change_count': 2,
              '_aligned': np.zeros((100, 100, 3), np.uint8), '_target': np.zeros((100, 100, 3), np.uint8)}

    def respond(self, url, **kwargs):
        assert '对比违建蓝色屋顶' in kwargs['json']['messages'][0]['content']
        content = {'regions': [
            {'id': 1, 'relevant': True, 'decision': 'keep', 'summary': '当前图中蓝色屋顶缺失'},
            {'id': 2, 'relevant': False, 'decision': 'color_only', 'summary': '路面形状不变，仅颜色变化'},
        ]}
        return httpx.Response(200, json={'message': {'content': json.dumps(content)}},
                              request=httpx.Request('POST', url))

    monkeypatch.setattr(httpx.Client, 'post', respond)
    semantic = describe_changes(result, base_url='http://model', model='test', timeout=1,
                                task_description='对比违建蓝色屋顶')
    assert result['polygons'] == polygons[:1]
    assert result['change_count'] == 1
    assert '路面' not in semantic['summary']
    assert semantic['region_count'] == 1


@pytest.mark.parametrize('decisions,expected_status', [
    ([{'id': 1, 'relevant': True, 'decision': 'color_only', 'summary': '颜色变化'}], 'matched'),
    ([{'id': 1, 'relevant': False, 'decision': 'keep', 'summary': '无关屋顶新增'}], 'matched'),
    ([{'id': 1, 'relevant': True, 'decision': 'uncertain', 'summary': '模糊无法确认'}], 'not_found'),
    ([], 'not_found'),
    ([{'id': True, 'decision': 'keep', 'summary': '物体新增'}], 'not_found'),
])
def test_review_decisions_do_not_fabricate_no_change(monkeypatch, decisions, expected_status):
    result = {'status': 'matched', 'polygons': [[[.1, .1], [.8, .1], [.8, .8]]], 'change_count': 1,
              '_aligned': np.zeros((100, 100, 3), np.uint8), '_target': np.zeros((100, 100, 3), np.uint8)}
    monkeypatch.setattr(httpx.Client, 'post', lambda self, url, **kwargs: httpx.Response(
        200, json={'message': {'content': json.dumps({'regions': decisions})}}, request=httpx.Request('POST', url)))
    semantic = describe_changes(result, base_url='http://model', model='test', timeout=1, task_description='比较道路')
    assert result['status'] == expected_status
    assert result['polygons'] == []
    if expected_status == 'not_found':
        assert '人工复核' in result['diff_summary']
        assert semantic['error']


def test_review_timeout_hides_unverified_boxes_with_explicit_failure(monkeypatch):
    result = {'status': 'matched', 'polygons': [[[.1, .1], [.8, .1], [.8, .8]]], 'change_count': 1,
              '_aligned': np.zeros((100, 100, 3), np.uint8), '_target': np.zeros((100, 100, 3), np.uint8)}
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout('busy')
    monkeypatch.setattr(httpx.Client, 'post', timeout)
    semantic = describe_changes(result, base_url='http://model', model='test', timeout=1, task_description='比较屋顶')
    assert semantic['status'] == 'failed'
    assert result['status'] == 'not_found' and result['polygons'] == []
    assert '复核失败或超时' in result['diff_summary']
