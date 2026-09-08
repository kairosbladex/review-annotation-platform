from __future__ import annotations

import re
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from .history_compare import MAX_PIXELS, compare_geometry, describe_changes

_slots = threading.BoundedSemaphore(2)


class ComparisonBusy(RuntimeError):
    pass


def compare_legacy_pair(first: Image.Image, second: Image.Image, *, one_to_many: bool,
                        baseline_is_current: bool, base_url: str, model: str, timeout: float,
                        task_description: str | None = None) -> dict:
    if not _slots.acquire(blocking=False):
        raise ComparisonBusy('历史对比繁忙，请稍后重试')
    started = time.perf_counter()
    try:
        for image in (first, second):
            if image.width * image.height > MAX_PIXELS or min(image.size) < 32:
                raise ValueError('图片尺寸须至少32像素，总像素不能超过2500万')
        first = ImageOps.exif_transpose(first).convert('RGB')
        second = ImageOps.exif_transpose(second).convert('RGB')
        current, history = (first, second) if one_to_many or baseline_is_current else (second, first)
        result = compare_geometry(np.asarray(history), np.asarray(current))
        semantic = describe_changes(result, base_url=base_url, model=model, timeout=timeout,
                                    task_description=task_description)
        summary = result['diff_summary']
        if semantic['summary']:
            summary += f"。局部变化描述（{semantic['region_count']}处）：{semantic['summary']}"
        if semantic['error'] and semantic['error'] != summary:
            summary += f"。{semantic['error']}"
        if not one_to_many:
            current_label, history_label = ('基准照片', '对比照片') if baseline_is_current else ('对比照片', '基准照片')
            summary = re.sub(r'当前(?:图片|照片|图像|图)|历史(?:图片|照片|图像|图)',
                             lambda match: current_label if match[0].startswith('当前') else history_label,
                             summary)
        polygons = [
            [[min(current.width - 1, round(x * current.width)),
              min(current.height - 1, round(y * current.height))] for x, y in polygon]
            for polygon in result['polygons']
        ]
        if one_to_many:
            dimensions = {'current_width': current.width, 'current_height': current.height,
                          'history_width': history.width, 'history_height': history.height}
        else:
            dimensions = {'baseline_width': first.width, 'baseline_height': first.height,
                          'compare_width': second.width, 'compare_height': second.height}
        return {
            **dimensions, 'status': result['status'], 'polygons': polygons,
            'match_score': None, 'diff_summary': summary,
            'reason': semantic['error'] if result['status'] == 'matched' else result['diff_summary'],
            'inference_time_seconds': round(time.perf_counter() - started, 3),
        }
    finally:
        _slots.release()
