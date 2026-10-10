#!/usr/bin/env python3
"""从 09-24 快照取原图并投影候选框，供原图复标。不调用模型，不写自动迭代资产，不连数据库。

默认（不给 --class-name）与道路裂纹包一致：自动迭代无标签池里的道路裂纹，只用合格原视频。
--events 可改成指定的快照事件，仍然只出道路裂纹、只用合格原视频。

--class-name 指定类别后，改为快照 inference_manifest 里该类别的全部事件（不抽样）：
照片和视频都接受，两者都合格时用照片（分辨率更高）。合格标准是快照字段
clean_usable_for_inference。合不上的事件写入 excluded.json，并写明原因。

视频在匹配时刻 ±1.5 秒内逐帧解码（内存只保留当前帧和最佳帧），用单应矩阵反投影后
与快照已核验的对齐参考帧做相关，取最相似的一帧。照片直接读 source_local_path，用同一单应矩阵投影。
自动迭代 items.json 里 dev/holdout 的同类别框按截图哈希和框的交并比标记来源，文件只读。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

SNAP = Path('/data/workspace/jzheng/codex-ktd-real-monitoring-20260924')
AUTO = Path('/data/workspace/jzheng/luna-prompt-autotune')
WINDOW = 1.5
MIN_VIDEO_NCC = 0.80
REASON_TEXT = {
    'candidate_region_or_global_correlation_insufficient': '候选区域或整图相关不够',
    'pre_gate_rejected': '几何预检未过',
    'full_ncc_below_0.80': '整图相关低于 0.80',
    'pass': '通过',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_memory():
    info = Path('/proc/meminfo')
    if not info.exists():
        return
    mem = {}
    for line in info.read_text().splitlines():
        key, value = line.split(':', 1)
        mem[key] = int(value.split()[0])
    if mem['MemAvailable'] / mem['MemTotal'] < 0.15:
        raise SystemExit('可用内存低于 15%，停止')


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def probe_size(video):
    data = json.loads(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', str(video)],
        timeout=30))
    stream = data['streams'][0]
    return int(stream['width']), int(stream['height'])


def best_frame(video, t, H, ref):
    """在匹配时刻 ±1.5 秒内逐帧解码，返回与已核验对齐帧最相似的一帧。"""
    width, height = probe_size(video)
    ref_h, ref_w = ref.shape
    inverse = np.linalg.inv(np.asarray(H, dtype=np.float64))
    start = max(0.0, t - WINDOW)
    cmd = ['timeout', '180', 'ffmpeg', '-v', 'error', '-threads', '1', '-ss', f'{start:.3f}', '-i', str(video),
           '-t', f'{2 * WINDOW:.3f}', '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes, index, best = width * height * 3, 0, None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            back = cv2.warpPerspective(frame, inverse, (ref_w, ref_h))
            score = ncc(cv2.cvtColor(back, cv2.COLOR_BGR2GRAY).astype(np.float32), ref)
            if best is None or score > best[1]:
                best = (frame.copy(), score, index)
            index += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=120)
    if best is None:
        raise RuntimeError('no_decoded_frame')
    return best, index, (width, height)


def project(box, H, sw, sh, dw, dh):
    """把截图归一化框经单应矩阵投到原图像素，再归一化。投出画面就抛 ValueError。"""
    x1, y1, x2, y2 = box
    pts = np.array([[[x1 * sw, y1 * sh]], [[x2 * sw, y1 * sh]], [[x1 * sw, y2 * sh]], [[x2 * sw, y2 * sh]]], dtype=np.float64)
    projected = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)
    if not np.isfinite(projected).all():
        raise ValueError('invalid_projection')
    mapped = [max(0.0, projected[:, 0].min() / dw), max(0.0, projected[:, 1].min() / dh),
              min(1.0, projected[:, 0].max() / dw), min(1.0, projected[:, 1].max() / dh)]
    if not (mapped[0] < mapped[2] and mapped[1] < mapped[3]):
        raise ValueError('outside')
    return mapped


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def usable(record, kind):
    if not record or not record.get('clean_usable_for_inference') or record.get('source_kind') != kind:
        return False
    if not record.get('homography_annotated_to_source') or not record.get('source_local_path'):
        return False
    if kind == 'video' and (record.get('matched_raw_time') is None or not record.get('clean_frame')):
        return False
    return True


def choose_source(still, video, allow_still):
    """两者都合格时返回照片。都不合格时返回 (None, 原因)。"""
    still_ok = allow_still and usable(still, 'still')
    video_ok = usable(video, 'video')
    if still_ok:
        return still, None
    if video_ok:
        return video, None
    return None, reject_reason(still, video, allow_still)


def _gate_text(record, label):
    if record is None:
        return f'没有{label}匹配记录'
    if record.get('clean_usable_for_inference') and not record.get('homography_annotated_to_source'):
        return f'{label}合格标记缺少单应矩阵'
    if record.get('clean_usable_for_inference') and not record.get('source_local_path'):
        return f'{label}合格标记缺少原片路径'
    if record.get('source_kind') == 'video' and record.get('clean_usable_for_inference') and record.get('matched_raw_time') is None:
        return '视频合格标记缺少匹配时刻'
    code = record.get('clean_quality_gate_reason') or record.get('status') or 'unknown'
    return f'{label}不合格：{REASON_TEXT.get(code, code)}'


def reject_reason(still, video, allow_still, source_flags=None):
    if still is None and video is None:
        flags = source_flags or {}
        if not flags.get('has_still') and not flags.get('has_video'):
            return '快照没有原照片或原视频'
        parts = []
        if flags.get('has_still'):
            parts.append('有原照片但没有匹配记录')
        if flags.get('has_video'):
            parts.append('有原视频但没有匹配记录')
        return '；'.join(parts)
    parts = []
    if allow_still:
        parts.append(_gate_text(still, '照片'))
    elif still is not None and not video:
        parts.append('照片不用于此类别的默认取帧')
    parts.append(_gate_text(video, '视频'))
    return '；'.join(parts)


def match_autotune(screenshot_sha, bbox, index):
    """index 来自只读的 items.json。交并比达到 0.99 视为同一个框。"""
    hits = [item for item in index if item['sha256'] == screenshot_sha and iou(item['bbox'], bbox) >= 0.99]
    if not hits:
        return 'new', None, None
    hit = sorted(hits, key=lambda item: item['key'])[0]
    return f"autotune_{hit['split']}", hit['key'], hit['split']


def crop(img, box, scale_min=96):
    height, width = img.shape[:2]
    x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    side = max(scale_min, (x2 - x1) * 2, (y2 - y1) * 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bounds = [int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(width, cx + side / 2)), int(min(height, cy + side / 2))]
    view = img[bounds[1]:bounds[3], bounds[0]:bounds[2]].copy()
    cv2.rectangle(view, (int(x1 - bounds[0]), int(y1 - bounds[1])), (int(x2 - bounds[0]), int(y2 - bounds[1])), (0, 255, 255), 2)
    return cv2.resize(view, (360, 360), interpolation=cv2.INTER_CUBIC)


def context(img, box):
    """原图大范围：框外扩 5 倍、至少 480 像素。"""
    height, width = img.shape[:2]
    x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    side = max(480, (x2 - x1) * 5, (y2 - y1) * 5)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bounds = [int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(width, cx + side / 2)), int(min(height, cy + side / 2))]
    view = img[bounds[1]:bounds[3], bounds[0]:bounds[2]].copy()
    cv2.rectangle(view, (int(x1 - bounds[0]), int(y1 - bounds[1])), (int(x2 - bounds[0]), int(y2 - bounds[1])), (0, 255, 255), 2)
    return cv2.resize(view, (480, 480), interpolation=cv2.INTER_AREA)


def file_key(candidate_id):
    return candidate_id.replace(':', '_')


def load_match_records():
    grouped = {}
    for name in ('still', 'video'):
        path = SNAP / 'output' / f'{name}_matches.jsonl'
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            grouped.setdefault(record['event_id'], {})[name] = record
    return grouped


def load_source_flags():
    flags = {}
    path = SNAP / 'input' / 'source_manifest.jsonl'
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raws = row.get('raw_sources') or []
        flags[row['event_id']] = {
            'has_still': any(item.get('kind') == 'still' for item in raws),
            'has_video': any(item.get('kind') == 'video' for item in raws),
            'still_resolutions': sorted({item.get('image_resolution') or '' for item in raws if item.get('kind') == 'still'}),
        }
    return flags


def load_autotune_index(class_name):
    items = json.loads((AUTO / 'assets' / 'items.json').read_text())
    index = []
    for unit in items:
        if unit.get('split') not in ('dev', 'holdout'):
            continue
        for candidate in unit.get('candidates') or []:
            if candidate.get('class_name') != class_name:
                continue
            index.append({
                'split': unit['split'],
                'key': candidate['key'],
                'unit': unit['unit'],
                'sha256': unit['sha256'],
                'bbox': candidate['bbox'],
            })
    return index


def load_snapshot_units(class_name, events):
    grouped = {}
    for line in (SNAP / 'input' / 'inference_manifest.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['class_name'] != class_name:
            continue
        if events is not None and row['event_id'] not in events:
            continue
        width, height = row['frame_size']
        x1, y1, x2, y2 = row['bbox_px']
        grouped.setdefault(row['event_id'], []).append({
            'key': row['candidate_id'],
            'class_name': row['class_name'],
            'bbox': [x1 / width, y1 / height, x2 / width, y2 / height],
        })
    units = []
    for event_id in sorted(grouped):
        units.append({
            'unit': event_id,
            'candidates': grouped[event_id],
            'image_path': str(SNAP / 'work' / 'annotated' / f'{event_id}.jpg'),
        })
    return units


def load_legacy_units():
    items = json.loads((AUTO / 'assets' / 'items.json').read_text())
    units = []
    for unit in items:
        if unit.get('split') != 'unlabeled':
            continue
        candidates = [dict(candidate) for candidate in unit.get('candidates') or []]
        if not any(candidate.get('class_name') == '道路裂纹' for candidate in candidates):
            continue
        units.append({
            'unit': unit['unit'],
            'candidates': candidates,
            'image_path': str(AUTO / 'assets' / 'images' / f"{unit['unit']}.jpg"),
            'screenshot_sha': unit.get('sha256'),
        })
    return units


def visual_of(record):
    path = record.get('source_local_path') or ''
    for attempt in record.get('attempts') or []:
        if attempt.get('local_path') == path and attempt.get('visual'):
            return attempt['visual']
    name = record.get('source_object_path') or ''
    for token in ('T', 'W', 'V', 'Z'):
        if f'_{token}_' in name or f'_{token}.' in name:
            return token
    return None


def mark_duplicates(units):
    first = {}
    for unit in units:
        digest = unit.get('screenshot_sha')
        if not digest and Path(unit['image_path']).is_file():
            digest = sha256_file(unit['image_path'])
            unit['screenshot_sha'] = digest
        boxes = tuple(sorted(tuple(round(value, 6) for value in candidate['bbox']) for candidate in unit['candidates']))
        signature = (digest, boxes)
        if digest and signature in first:
            unit['duplicate_of'] = first[signature]
        elif digest:
            first[signature] = unit['unit']
            unit['duplicate_of'] = None
        else:
            unit['duplicate_of'] = None


def tag_candidates(unit, index):
    for candidate in unit['candidates']:
        if index is None:
            candidate['sample_source'] = 'new'
            candidate['autotune_key'] = None
            candidate['autotune_split'] = None
            continue
        source, key, split = match_autotune(unit.get('screenshot_sha'), candidate['bbox'], index)
        candidate['sample_source'] = source
        candidate['autotune_key'] = key
        candidate['autotune_split'] = split


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n')


def open_source(record):
    path = SNAP / record['source_local_path']
    if record['source_kind'] == 'still':
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError('still_unreadable')
        height, width = frame.shape[:2]
        return frame, {'source_kind': 'still', 'source_width': width, 'source_height': height, 'source_visual': visual_of(record)}
    reference_path = SNAP / record['clean_frame']
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise RuntimeError('verified_clean_reference_missing')
    (frame, score, index), decoded, (width, height) = best_frame(
        path, float(record['matched_raw_time']), record['homography_annotated_to_source'], reference.astype(np.float32))
    audit = {
        'source_kind': 'video',
        'source_width': width,
        'source_height': height,
        'source_visual': None,
        'frames_decoded': decoded,
        'best_index': index,
        'ncc': round(score, 4),
    }
    if score < MIN_VIDEO_NCC:
        raise RuntimeError(f'decoded_frame_ncc_below_{MIN_VIDEO_NCC:.2f}:{score:.4f}')
    return frame, audit


def main():
    parser = argparse.ArgumentParser(description='原图取帧并投影候选框，供人工复标')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--class-name', help='快照里该类别的全部事件。不给则沿用道路裂纹无标签池，且只用视频')
    parser.add_argument('--events', type=Path, help='事件编号 JSON 列表，进一步限定范围')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    assert_memory()
    for name in ('clean', 'compare', 'context'):
        (args.out / name).mkdir(parents=True, exist_ok=True)

    events = set(json.loads(args.events.read_text())) if args.events else None
    allow_still = args.class_name is not None
    class_name = args.class_name or '道路裂纹'
    if args.class_name or events is not None:
        # --events 沿用快照事件列表。指定类别时照片和视频都要，照片优先。
        units = load_snapshot_units(class_name, events)
        autotune_index = load_autotune_index(class_name) if args.class_name else None
    else:
        units = load_legacy_units()
        autotune_index = None
    records = load_match_records()
    source_flags = load_source_flags() if allow_still else {}
    mark_duplicates(units)
    for unit in units:
        tag_candidates(unit, autotune_index)

    allowlist, audit, boxes, excluded = [], [], [], []
    for unit in units:
        assert_memory()
        still = records.get(unit['unit'], {}).get('still')
        video = records.get(unit['unit'], {}).get('video')
        chosen, reason = choose_source(still, video, allow_still)
        base = {
            'event_id': unit['unit'],
            'box_count': len(unit['candidates']),
            'duplicate_of': unit.get('duplicate_of'),
            'sample_sources': sorted({candidate['sample_source'] for candidate in unit['candidates']}),
        }
        if chosen is None:
            if not allow_still:
                reason = 'no_usable_video_match'
            else:
                reason = reject_reason(still, video, allow_still, source_flags.get(unit['unit']))
            excluded.append(dict(base, reason=reason, candidate_ids=[candidate['key'] for candidate in unit['candidates']]))
            print(unit['unit'], 'skip', reason, flush=True)
            continue
        shot = cv2.imread(unit['image_path'])
        if shot is None:
            excluded.append(dict(base, reason='截图无法读取', candidate_ids=[candidate['key'] for candidate in unit['candidates']]))
            print(unit['unit'], 'skip screenshot', flush=True)
            continue
        try:
            frame, frame_audit = open_source(chosen)
        except (RuntimeError, OSError, ValueError) as exc:
            excluded.append(dict(base, reason=f'原图未能打开：{exc}', candidate_ids=[candidate['key'] for candidate in unit['candidates']]))
            print(unit['unit'], 'skip', exc, flush=True)
            continue
        shot_h, shot_w = shot.shape[:2]
        projected, outside = [], []
        for candidate in unit['candidates']:
            original = list(candidate['bbox'])
            try:
                mapped = project(original, chosen['homography_annotated_to_source'], shot_w, shot_h, frame.shape[1], frame.shape[0])
            except (ValueError, cv2.error):
                outside.append(candidate['key'])
                continue
            projected.append(dict(candidate, bbox=mapped, screenshot_bbox=original))
        kept = [candidate for candidate in projected if candidate['class_name'] == class_name]
        if not kept:
            excluded.append(dict(base, reason='全部框投影后不在原图内', candidate_ids=[candidate['key'] for candidate in unit['candidates']], outside=outside))
            print(unit['unit'], 'skip outside', flush=True)
            continue
        cv2.imwrite(str(args.out / 'clean' / f"{unit['unit']}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for candidate in kept:
            key = file_key(candidate['key'])
            pair = np.concatenate([crop(shot, candidate['screenshot_bbox']), np.full((360, 8, 3), 255, np.uint8), crop(frame, candidate['bbox'])], axis=1)
            cv2.imwrite(str(args.out / 'compare' / f'{key}.jpg'), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(args.out / 'context' / f'{key}.jpg'), context(frame, candidate['bbox']), [cv2.IMWRITE_JPEG_QUALITY, 88])
            boxes.append({
                'key': key,
                'candidate_id': candidate['key'],
                'event_id': unit['unit'],
                'class_name': class_name,
                'sample_source': candidate['sample_source'],
                'autotune_split': candidate['autotune_split'],
                'autotune_key': candidate['autotune_key'],
                'duplicate_of': unit.get('duplicate_of'),
                'source_kind': frame_audit['source_kind'],
                'source_width': frame_audit['source_width'],
                'source_height': frame_audit['source_height'],
                'source_visual': frame_audit.get('source_visual'),
                'screenshot_bbox': candidate['screenshot_bbox'],
                'bbox': candidate['bbox'],
            })
        allowlist.append({'unit': unit['unit'], 'candidates': [{'key': candidate['key'], 'class_name': candidate['class_name'], 'bbox': candidate['bbox']} for candidate in projected]})
        audit.append(dict(base, source_kind=frame_audit['source_kind'], source_width=frame_audit['source_width'],
                          source_height=frame_audit['source_height'], source_visual=frame_audit.get('source_visual'),
                          frames_decoded=frame_audit.get('frames_decoded'), best_index=frame_audit.get('best_index'),
                          ncc=frame_audit.get('ncc'), outside=outside, kept=[candidate['key'] for candidate in kept]))
        print(unit['unit'], frame_audit['source_kind'], frame_audit['source_width'], frame_audit['source_height'],
              frame_audit.get('ncc'), outside, flush=True)
        del frame, shot

    counts = {}
    for box in boxes:
        counts[box['sample_source']] = counts.get(box['sample_source'], 0) + 1
    kind_counts, size_counts = {}, {}
    for box in boxes:
        kind_counts[box['source_kind']] = kind_counts.get(box['source_kind'], 0) + 1
        size = f"{box['source_width']}x{box['source_height']}"
        if box.get('source_visual'):
            size = f"{size}:{box['source_visual']}"
        size_counts[size] = size_counts.get(size, 0) + 1
    summary = {
        'class_name': class_name,
        'allow_still': allow_still,
        'qualification': 'clean_usable_for_inference；指定类别时照片优先于视频',
        'snapshot_events': len(units),
        'snapshot_boxes': sum(len(unit['candidates']) for unit in units),
        'included_events': len(allowlist),
        'included_boxes': len(boxes),
        'excluded_events': len(excluded),
        'sample_source_counts': counts,
        'source_kind_box_counts': kind_counts,
        'source_size_box_counts': size_counts,
        'duplicate_events': [unit['unit'] for unit in units if unit.get('duplicate_of')],
        'outside_boxes': [key for row in audit for key in (row.get('outside') or [])],
        'excluded_reason_counts': {},
        'autotune_index_boxes': None if autotune_index is None else {
            'dev': sum(1 for item in autotune_index if item['split'] == 'dev'),
            'holdout': sum(1 for item in autotune_index if item['split'] == 'holdout'),
        },
    }
    for row in excluded:
        summary['excluded_reason_counts'][row['reason']] = summary['excluded_reason_counts'].get(row['reason'], 0) + 1
    write_json(args.out / 'allowlist.json', allowlist)
    write_json(args.out / 'frames-audit.json', audit)
    write_json(args.out / 'excluded.json', excluded)
    write_json(args.out / 'boxes.json', {'class_name': class_name, 'seed': 20261010, 'boxes': boxes})
    write_json(args.out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
