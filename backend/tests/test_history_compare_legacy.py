import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.services.history_compare_legacy import compare_legacy_pair


@pytest.mark.parametrize('one_to_many,baseline_is_current', [(True, False), (False, True), (False, False)])
def test_adapter_uses_current_pixels_and_correct_semantic_direction(monkeypatch, one_to_many, baseline_is_current):
    rng = np.random.default_rng(52)
    history = cv2.GaussianBlur(rng.integers(40, 210, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
    current = cv2.warpAffine(history, np.float32([[1, 0, 8], [0, 1, 5]]), (640, 480))
    current[110:180, 160:250] = 0

    def describe(result, **kwargs):
        assert result['_target'][140, 200].mean() == 0
        assert result['_aligned'][140, 200].mean() > 80
        return {'status': 'completed', 'summary': '当前图相较历史图，当前照片新增物体，历史图片无该物体',
                'error': None, 'region_count': 1}

    monkeypatch.setattr('app.services.history_compare_legacy.describe_changes', describe)
    first, second = (current, history) if one_to_many or baseline_is_current else (history, current)
    result = compare_legacy_pair(Image.fromarray(first), Image.fromarray(second),
                                 one_to_many=one_to_many, baseline_is_current=baseline_is_current,
                                 base_url='http://model.invalid', model='test', timeout=1)
    assert result['status'] == 'matched'
    assert len(result['polygons']) == 1
    x, y = np.mean(result['polygons'][0], axis=0)
    assert abs(x - 205) < 12 and abs(y - 145) < 12
    current_label, history_label = ('当前照片', '历史图片') if one_to_many else (
        ('基准照片', '对比照片') if baseline_is_current else ('对比照片', '基准照片'))
    assert f'{current_label}新增物体，{history_label}无该物体' in result['diff_summary']
    if not one_to_many:
        assert result['diff_summary'].startswith(f'{current_label}相较{history_label}，检测到1处')
        assert '当前' not in result['diff_summary'] and '历史' not in result['diff_summary']


def test_task_scoped_result_is_not_extended_with_region_list(monkeypatch):
    rng = np.random.default_rng(52)
    history = cv2.GaussianBlur(rng.integers(40, 210, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
    current = cv2.warpAffine(history, np.float32([[1, 0, 8], [0, 1, 5]]), (640, 480))
    current[110:180, 160:250] = 0

    def filter_task(result, **kwargs):
        assert kwargs['task_description'] == '对比一下光伏板的变化'
        result.update(polygons=result['polygons'][:1], change_count=1, status='matched',
                      diff_summary='检测到1处明显变化：当前图屋面新增蓝色光伏板')
        return {'summary': '1. 历史图：裸露混凝土；当前图：新增蓝色光伏板',
                'error': None, 'region_count': 1}

    monkeypatch.setattr('app.services.history_compare_legacy.filter_task_changes', filter_task)
    result = compare_legacy_pair(Image.fromarray(current), Image.fromarray(history),
                                 one_to_many=False, baseline_is_current=True,
                                 base_url='http://model.invalid', model='test', timeout=1,
                                 task_description='对比一下光伏板的变化')
    assert result['diff_summary'] == '检测到1处明显变化：基准照片屋面新增蓝色光伏板'
    assert '局部变化描述' not in result['diff_summary'] and len(result['polygons']) == 1


def test_overlay_refuses_unknown_source_and_keeps_legacy_branch():
    path = Path(__file__).resolve().parents[2] / 'scripts/prepare_history_overlay.py'
    spec = importlib.util.spec_from_file_location('history_overlay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = b'async def _compare_pair(prompt, current, history, client):\n    return "old_one"\n\nasync def _compare_baseline_compare_pair(prompt, baseline, compare, client):\n    return "old_many"\n'
    with pytest.raises(ValueError, match='digest changed'):
        module.patch_router(source)
    patched = module.patch_router(source, hashlib.sha256(source).hexdigest())
    assert patched.count('VL_COMPARE_ENGINE') == 2
    assert 'return "old_one"' in patched and 'return "old_many"' in patched
    namespace = {'_env': lambda key, default: default}
    exec(patched, namespace)
    import asyncio
    assert asyncio.run(namespace['_compare_pair'](None, None, None, None)) == 'old_one'


def test_overlay_passes_user_scope_to_registered_adapter(monkeypatch):
    import asyncio
    import json
    import sys
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[2] / 'scripts/prepare_history_overlay.py'
    spec = importlib.util.spec_from_file_location('history_overlay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = b'def _build_prompt(mode, req, extra_params):\n    return "old"\nasync def _compare_pair(prompt, current, history, client):\n    pass\nasync def _compare_baseline_compare_pair(prompt, baseline, compare, client):\n    pass\n'
    calls = []
    def compare(*args, **kwargs):
        calls.append(kwargs)
    monkeypatch.setitem(sys.modules, 'history_compare_runtime.services.history_compare_legacy',
                        SimpleNamespace(ComparisonBusy=RuntimeError, compare_legacy_pair=compare))
    namespace = {'asyncio': asyncio, 'json': json, '_env': lambda key, default: 'registered',
                 '_env_float': lambda key, default: default}
    exec(module.patch_router(source, hashlib.sha256(source).hexdigest()), namespace)
    prompt = namespace['_build_prompt'](None, SimpleNamespace(task_description='对比光伏板'), None)
    assert json.loads(prompt)['task_description'] == '对比光伏板'
    image = SimpleNamespace(image=None)
    client = SimpleNamespace(base_url='http://model.invalid', model='test')
    for helper in ('_compare_pair', '_compare_baseline_compare_pair'):
        asyncio.run(namespace[helper](prompt, image, image, client))
    assert all(c['task_description'] == prompt and c['timeout'] == 60 for c in calls)
