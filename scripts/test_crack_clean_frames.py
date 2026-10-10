"""原图取帧脚本的选择、投影和来源标记。不读服务器，不改自动迭代资产。"""
from __future__ import annotations

import importlib.util
import pathlib
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('crack_clean_frames', ROOT / 'scripts' / 'crack_clean_frames.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(kind, usable=True, **extra):
    payload = {
        'source_kind': kind,
        'clean_usable_for_inference': usable,
        'homography_annotated_to_source': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'source_local_path': f'work/raw/{kind}/a.jpg',
    }
    if kind == 'video':
        payload['matched_raw_time'] = 1.5
        payload['clean_frame'] = 'work/clean_frames/a.jpg'
    payload.update(extra)
    return payload


class ChooseSourceTest(unittest.TestCase):
    def test_photo_wins_when_both_qualify(self):
        still, video = record('still'), record('video')
        chosen, reason = MODULE.choose_source(still, video, allow_still=True)
        self.assertIs(chosen, still)
        self.assertIsNone(reason)

    def test_video_used_when_photo_fails_gate(self):
        still = record('still', usable=False, clean_quality_gate_reason='candidate_region_or_global_correlation_insufficient')
        video = record('video')
        chosen, reason = MODULE.choose_source(still, video, allow_still=True)
        self.assertIs(chosen, video)
        self.assertIsNone(reason)

    def test_legacy_mode_ignores_qualified_photo(self):
        chosen, reason = MODULE.choose_source(record('still'), None, allow_still=False)
        self.assertIsNone(chosen)
        self.assertIn('视频', reason)

    def test_missing_media_is_explained(self):
        reason = MODULE.reject_reason(None, None, True, {'has_still': False, 'has_video': False})
        self.assertEqual(reason, '快照没有原照片或原视频')
        reason = MODULE.reject_reason(None, None, True, {'has_still': True, 'has_video': True})
        self.assertIn('有原照片但没有匹配记录', reason)
        self.assertIn('有原视频但没有匹配记录', reason)

    def test_gate_reason_is_chinese(self):
        still = record('still', usable=False, clean_quality_gate_reason='pre_gate_rejected')
        reason = MODULE.reject_reason(still, None, True, None)
        self.assertIn('几何预检未过', reason)
        self.assertIn('没有视频匹配记录', reason)


class ProjectTest(unittest.TestCase):
    def test_scale_homography_keeps_normalized_box(self):
        homography = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
        mapped = MODULE.project([0.1, 0.2, 0.3, 0.4], homography, 100, 100, 200, 200)
        self.assertTrue(np.allclose(mapped, [0.1, 0.2, 0.3, 0.4]))

    def test_box_past_the_edge_is_rejected(self):
        homography = [[1, 0, 500], [0, 1, 0], [0, 0, 1]]
        with self.assertRaises(ValueError):
            MODULE.project([0.1, 0.1, 0.2, 0.2], homography, 100, 100, 100, 100)


class AutotuneMarkTest(unittest.TestCase):
    def test_exact_box_on_same_screenshot_is_marked(self):
        index = [{'split': 'dev', 'key': 'gold0903:abc', 'sha256': 's', 'bbox': [0.1, 0.2, 0.3, 0.4]}]
        source, key, split = MODULE.match_autotune('s', [0.1, 0.2, 0.3, 0.4], index)
        self.assertEqual((source, key, split), ('autotune_dev', 'gold0903:abc', 'dev'))

    def test_different_box_stays_new(self):
        index = [{'split': 'holdout', 'key': 'gold0903:abc', 'sha256': 's', 'bbox': [0.5, 0.5, 0.6, 0.6]}]
        source, key, split = MODULE.match_autotune('s', [0.1, 0.2, 0.3, 0.4], index)
        self.assertEqual((source, key, split), ('new', None, None))


class PageCopyTest(unittest.TestCase):
    def test_hotspot_page_states_the_decision_rule(self):
        html = (ROOT / 'data' / 'hotspot-relabel-hd-20261010' / 'index.html').read_text()
        self.assertIn('局部异常亮斑', html)
        self.assertIn('不在光伏板上的一律不是', html)
        self.assertIn('整体反光', html)
        self.assertIn('是光斑', html)
        self.assertIn('不是光斑', html)
        self.assertIn('看不清', html)
        self.assertIn('不显示截图上的旧标签', html)
        self.assertNotIn('置信', html)
        self.assertNotIn('detector', html.lower())


if __name__ == '__main__':
    unittest.main()
