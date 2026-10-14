from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_SIDE = 1600
# Change kinds the task-scoped review groups kept regions by; order controls the summary wording.
CHANGE_TYPES = ('新增', '破损', '缺失', '遮挡', '布局改变', '其他')
logger = logging.getLogger(__name__)


def parse_change_groups(payload: dict, kept_count: int) -> list[dict] | None:
    """Validate the model's per-kind grouping of kept regions; None means unusable."""
    groups = payload.get('changes')
    if not isinstance(groups, list) or (kept_count and not groups) or (not kept_count and groups):
        return None
    parsed = []
    for group in groups:
        if not isinstance(group, dict):
            return None
        kind, number, detail = group.get('type'), group.get('count'), group.get('detail')
        if kind not in CHANGE_TYPES or type(number) is not int or number < 1:
            return None
        if not isinstance(detail, str) or not detail.strip() or len(detail.strip()) > 60:
            return None
        parsed.append({'type': kind, 'count': number, 'detail': detail.strip().rstrip('。；;')})
    if sum(group['count'] for group in parsed) != kept_count:
        return None
    return parsed


def decode_image(data: bytes) -> np.ndarray:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('图片不能超过20MB')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS or min(image.size) < 32:
                raise ValueError('图片尺寸须至少32像素，总像素不能超过2500万')
            return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('无法读取图片，请使用有效的JPEG或PNG图片') from exc


def compare_geometry(baseline: np.ndarray, current: np.ndarray) -> dict:
    height, width = current.shape[:2]
    result = {
        'status': 'not_found', 'polygons': [], 'change_count': 0,
        'current_width': width, 'current_height': height,
        'baseline_width': baseline.shape[1], 'baseline_height': baseline.shape[0],
        'coordinate_space': 'normalized_current',
        'alignment': {'method': 'unavailable', 'inliers': 0, 'overlap': 0.0},
        'diff_summary': '两张图片无法可靠对齐',
    }
    if abs((baseline.shape[1] / baseline.shape[0]) / (width / height) - 1) > 0.05:
        result['diff_summary'] = '图片宽高比差异过大，无法可靠对齐'
        return result
    scale = min(1.0, MAX_SIDE / max(width, height))
    size = (round(width * scale), round(height * scale))
    target = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    reference = cv2.resize(baseline, size, interpolation=cv2.INTER_AREA)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    transform = np.eye(3)
    if np.array_equal(reference, target):
        result['alignment'] = {'method': 'identity', 'inliers': 0, 'overlap': 1.0}
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        key_reference, desc_reference = detector.detectAndCompute(gray_reference, None)
        key_target, desc_target = detector.detectAndCompute(gray_target, None)
        if desc_reference is None or desc_target is None or len(desc_target) < 2:
            return result
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_reference, desc_target, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        if len(good) < 20:
            return result
        src = np.float32([key_reference[m.queryIdx].pt for m in good])
        dst = np.float32([key_target[m.trainIdx].pt for m in good])
        transform, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
        if transform is None or inlier_mask is None or not np.isfinite(transform).all():
            return result
        inliers = int(inlier_mask.sum())
        if inliers < 20 or inliers / len(good) < 0.4:
            return result
        spread = np.ptp(dst[inlier_mask.ravel().astype(bool)], axis=0) / np.asarray(size)
        if min(spread) < 0.3:
            return result
        corners = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, transform)
        area_ratio = cv2.contourArea(projected) / (size[0] * size[1])
        if not cv2.isContourConvex(projected) or not 0.6 < area_ratio < 1.6:
            return result
        result['alignment'] = {'method': 'orb_ransac', 'inliers': inliers, 'overlap': 0.0}

    aligned = cv2.warpPerspective(reference, transform, size)
    valid = cv2.warpPerspective(np.full(reference.shape[:2], 255, np.uint8), transform, size)
    border = min(7, max(1, min(size) // 80))
    valid = cv2.erode(valid, np.ones((2 * border + 1, 2 * border + 1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    overlap = np.count_nonzero(valid) / valid.size
    result['alignment']['overlap'] = round(overlap, 4)
    if overlap < 0.7:
        result['diff_summary'] = '图片重叠范围不足，无法可靠比较'
        return result

    # Correct global exposure shifts before extracting spatially separate changes.
    first = cv2.GaussianBlur(aligned, (5, 5), 0).astype(np.float32)
    second = cv2.GaussianBlur(target, (5, 5), 0).astype(np.float32)
    offset = np.median((second - first)[valid > 0], axis=0)
    delta = np.max(np.abs(second - first - offset), axis=2)
    mask = ((delta > 28) & (valid > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if np.count_nonzero(mask) / max(np.count_nonzero(valid), 1) > 0.4:
        result['diff_summary'] = '大范围外观差异，无法可靠区分场景变化与成像差异'
        return result
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= mask.size * 0.0007]
    contours.sort(key=cv2.contourArea, reverse=True)
    polygons = []
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, max(1.0, cv2.arcLength(contour, True) * 0.007), True)
        if len(polygon) < 3:
            continue
        polygons.append([[round(float(x) / size[0], 6), round(float(y) / size[1], 6)] for x, y in polygon[:, 0]])
    result.update(status='matched', polygons=polygons, change_count=len(polygons),
                  diff_summary=f'当前图相较历史图，检测到{len(polygons)}处可定位变化' if polygons else '无变化')
    result['_aligned'] = aligned
    result['_target'] = target
    return result


def describe_changes(result: dict, *, base_url: str, model: str, timeout: float,
                     task_description: str | None = None) -> dict:
    if result['status'] != 'matched' or not result['polygons']:
        return {'status': 'not_needed', 'summary': None, 'error': None, 'region_count': 0}
    if not base_url:
        return {'status': 'unavailable', 'summary': None, 'error': '未配置语义描述服务，变化区域已完成定位', 'region_count': 0}
    reference, target = result['_aligned'], result['_target']
    count = min(len(result['polygons']), 8)
    sheet = Image.new('RGB', (768, count * 220), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, polygon in enumerate(result['polygons'][:count]):
        points = np.asarray(polygon) * [target.shape[1], target.shape[0]]
        lower = np.maximum(np.floor(points.min(axis=0)).astype(int) - 12, 0)
        upper = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 12, [target.shape[1], target.shape[0]])
        for column, source in enumerate((reference, target)):
            crop = Image.fromarray(source[lower[1]:upper[1], lower[0]:upper[0]])
            crop.thumbnail((372, 188))
            x, y = column * 384, index * 220
            sheet.paste(crop, (x + (384 - crop.width) // 2, y + 27 + (188 - crop.height) // 2))
            draw.text((x + 8, y + 7), f'{index + 1}: {"HISTORY (before)" if column == 0 else "CURRENT (after)"}', fill='black')
    buf = io.BytesIO()
    sheet.save(buf, format='JPEG', quality=85)
    prompt = Path(__file__).resolve().parents[1].joinpath('prompts/history_compare.txt').read_text(encoding='utf-8')
    if task_description:
        prompt = Path(__file__).resolve().parents[1].joinpath('prompts/history_compare_filter.txt').read_text(encoding='utf-8')
        prompt += '\n用户要求：' + task_description + f'\n本批共{count}行。'
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(f'{base_url.rstrip("/")}/api/chat', json={
                'model': model, 'stream': False, 'think': False, 'format': 'json',
                'messages': [{'role': 'user', 'content': prompt, 'images': [base64.b64encode(buf.getvalue()).decode()]}],
                'options': {'temperature': 0, 'num_predict': 1024 if task_description else 256},
            })
            response.raise_for_status()
            content = response.json()['message']['content']
            if isinstance(content, str) and content.strip().startswith('```'):
                lines = content.strip().splitlines()
                if len(lines) >= 3 and lines[-1].strip() == '```':
                    content = '\n'.join(lines[1:-1])
            parsed = json.loads(content)
            if task_description:
                regions = parsed.get('regions')
                if not isinstance(regions, list) or len(regions) != count:
                    raise ValueError('missing region decisions')
                decisions = {}
                for region in regions:
                    identifier = region.get('id')
                    if type(identifier) is not int or not 1 <= identifier <= count or identifier in decisions:
                        raise ValueError('invalid region id')
                    if region.get('decision') not in {'keep', 'ignore', 'uncertain'}:
                        raise ValueError('invalid region decision')
                    if not isinstance(region.get('summary'), str) or not region['summary'].strip() or len(region['summary']) > 300:
                        raise ValueError('invalid region summary')
                    decisions[identifier] = region
                kept = [i for i in range(1, count + 1) if decisions[i]['decision'] == 'keep']
                return {'status': 'completed', 'summary': '；'.join(decisions[i]['summary'] for i in kept) or None,
                        'error': None, 'region_count': len(kept), 'kept_indices': [i - 1 for i in kept],
                        'region_summaries': [decisions[i]['summary'].strip().rstrip('。；;') for i in kept],
                        'uncertain_count': sum(r['decision'] == 'uncertain' for r in regions),
                        'change_groups': parse_change_groups(parsed, len(kept)) or []}
            summary = parsed.get('summary')
            if not isinstance(summary, str) or not summary.strip() or len(summary) > 1000:
                raise ValueError('invalid summary')
        return {'status': 'completed', 'summary': summary.strip(), 'error': None, 'region_count': count}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning('History comparison description failed: %s', type(exc).__name__)
        return {'status': 'failed', 'summary': None, 'error': '语义描述请求失败或超时，变化区域已完成定位', 'region_count': 0}


def filter_task_changes(result: dict, *, base_url: str, model: str, timeout: float,
                        task_description: str) -> dict:
    import time
    if result['status'] != 'matched':
        return {'summary': None, 'error': None, 'region_count': 0}
    candidates = result['polygons']
    retained = []
    summaries = []
    grouped = {}
    uncertain = 0
    deadline = time.monotonic() + timeout
    # Review every candidate in readable batches; an unfinished review is not "no change".
    for start in range(0, len(candidates), 8):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            uncertain += len(candidates) - start
            break
        batch = {**result, 'polygons': candidates[start:start + 8]}
        review = describe_changes(batch, base_url=base_url, model=model, timeout=remaining,
                                  task_description=task_description)
        if review['status'] != 'completed':
            uncertain += len(candidates) - start
            break
        retained.extend(batch['polygons'][i] for i in review['kept_indices'])
        summaries.extend(review['region_summaries'])
        for group in review['change_groups']:
            slot = grouped.setdefault(group['type'], {'count': 0, 'details': []})
            slot['count'] += group['count']
            if group['detail'] not in slot['details']:
                slot['details'].append(group['detail'])
        uncertain += review['uncertain_count']
    error = f'{uncertain}处候选区域未能确认，需人工复核' if uncertain else None
    # Same-kind changes merge into one clause; the numbered per-region list stays in summary.
    brief = '；'.join(f'{kind}{grouped[kind]["count"]}处（{"、".join(grouped[kind]["details"])}）'
                      for kind in CHANGE_TYPES if kind in grouped)
    result.update(polygons=retained, change_count=len(retained),
                  status='not_found' if uncertain and not retained else 'matched',
                  diff_summary=(f'检测到{len(retained)}处明显变化：{brief}' if retained and brief else
                                f'检测到{len(retained)}处符合任务的明显变化' if retained else
                                '未能确认目标变化，请人工复核' if uncertain else '没有明显变化'))
    return {'summary': '；'.join(f'{i}. {text}' for i, text in enumerate(summaries, 1)) or None,
            'error': error, 'region_count': len(retained)}


def compare_images(baseline: bytes, current: bytes, *, base_url: str, model: str, timeout: float) -> dict:
    result = compare_geometry(decode_image(baseline), decode_image(current))
    result['semantic'] = describe_changes(result, base_url=base_url, model=model, timeout=timeout)
    result.pop('_aligned', None)
    result.pop('_target', None)
    return result
