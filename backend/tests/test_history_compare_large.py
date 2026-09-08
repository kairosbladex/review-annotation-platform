import numpy as np

from app.services.history_compare import compare_geometry
from .test_history_compare import scene


def test_large_physical_change_produces_candidates_for_required_review():
    history = scene()
    current = history.copy()
    current[80:450, 80:590] = [215, 170, 115]
    result = compare_geometry(history, current, allow_large_changes=True)
    assert result['status'] == 'matched'
    assert result['polygons']
    assert all(min(y for x, y in polygon) > .14 for polygon in result['polygons'])
    assert max(max(y for x, y in polygon) for polygon in result['polygons']) > .9


def test_large_changes_without_review_keep_guard():
    history = scene()
    current = history.copy()
    current[80:420, 80:490] = [215, 170, 115]
    assert compare_geometry(history, current)['status'] == 'not_found'


def test_unalignable_large_scene_still_rejected():
    result = compare_geometry(scene(), np.zeros((480, 640, 3), np.uint8), allow_large_changes=True)
    assert result['status'] == 'not_found'
    assert result['polygons'] == []


def test_large_color_only_shift_is_not_a_land_change():
    history = scene()
    current = history.copy()
    current[80:450, 80:590] = np.clip(current[80:450, 80:590].astype(int) + [40, 15, -15], 0, 255)
    result = compare_geometry(history, current, allow_large_changes=True)
    assert result['status'] == 'matched'
    assert result['polygons'] == []
