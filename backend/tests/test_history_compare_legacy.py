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


def test_overlay_refuses_unknown_source_and_keeps_legacy_branch():
    path = Path(__file__).resolve().parents[2] / 'scripts/prepare_history_overlay.py'
    spec = importlib.util.spec_from_file_location('history_overlay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = b'def _build_prompt(mode, req, extra_params):\n    return "old_prompt"\n\nasync def _compare_pair(prompt, current, history, client):\n    return "old_one"\n\nasync def _compare_baseline_compare_pair(prompt, baseline, compare, client):\n    return "old_many"\n'
    with pytest.raises(ValueError, match='digest changed'):
        module.patch_router(source)
    patched = module.patch_router(source, hashlib.sha256(source).hexdigest())
    assert patched.count('VL_COMPARE_ENGINE') == 3
    assert patched.count('task_description=prompt') == 2
    assert 'return "old_one"' in patched and 'return "old_many"' in patched
    namespace = {'_env': lambda key, default: default}
    exec(patched, namespace)
    import asyncio
    assert asyncio.run(namespace['_compare_pair'](None, None, None, None)) == 'old_one'
    assert namespace['_build_prompt'](None, None, None) == 'old_prompt'
    import json
    from types import SimpleNamespace
    namespace.update(_env=lambda key, default: 'registered', json=json)
    req = SimpleNamespace(task_description='只比较蓝色屋顶', compare_goal='比较变化', image_description=None)
    assert json.loads(namespace['_build_prompt']('many_to_many', req, {}))['task_description'] == '只比较蓝色屋顶'
