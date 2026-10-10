#!/usr/bin/env python3
"""道路裂纹 40 张图：从 09-24 快照原视频取同一时刻 1920×1080 干净原帧，投影候选框（服务器运行，不读金标）。

每张图在快照记录的匹配时刻 ±1.5 秒内逐帧解码（逐帧读管道，内存只保留当前帧和最佳帧），
用快照单应矩阵把原帧反投影到截图坐标，与快照已核验的无标注对齐参考图做相关，取最相似一帧。
输出 <out>/clean/<unit>.jpg、<out>/allowlist.json（框为原帧归一化坐标）、<out>/frames-audit.json，
以及 <out>/compare/<key>.jpg（截图局部 | 原帧局部，供人工对照）。
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

SNAP = Path('/data/workspace/jzheng/codex-ktd-real-monitoring-20260924')
AUTO = Path('/data/workspace/jzheng/luna-prompt-autotune')
WINDOW = 1.5


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def best_frame(video: Path, t: float, H, ref):
    sh, sw = ref.shape
    inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    start = max(0.0, t - WINDOW)
    cmd = ['ffmpeg', '-v', 'error', '-threads', '2', '-ss', f'{start:.3f}', '-i', str(video), '-t', f'{2 * WINDOW:.3f}',
           '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size, k, best = 1920 * 1080 * 3, 0, None
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size: break
            frame = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
            back = cv2.cvtColor(cv2.warpPerspective(frame, inv, (sw, sh)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            score = ncc(back, ref)
            if best is None or score > best[1]: best = (frame.copy(), score, k)
            k += 1
    finally:
        proc.stdout.close(); proc.wait(timeout=60)
    return best, k


def project(box, H, sw, sh):
    x1, y1, x2, y2 = box
    pts = np.array([[[x1 * sw, y1 * sh]], [[x2 * sw, y1 * sh]], [[x1 * sw, y2 * sh]], [[x2 * sw, y2 * sh]]], dtype=np.float64)
    p = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)
    b = [max(0.0, p[:, 0].min() / 1920), max(0.0, p[:, 1].min() / 1080), min(1.0, p[:, 0].max() / 1920), min(1.0, p[:, 1].max() / 1080)]
    if not (b[0] < b[2] and b[1] < b[3]): raise ValueError('outside')
    return b


def crop(img, box, scale_min=96):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    s = max(scale_min, (x2 - x1) * 2, (y2 - y1) * 2); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    b = [int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(w, cx + s / 2)), int(min(h, cy + s / 2))]
    c = img[b[1]:b[3], b[0]:b[2]].copy()
    cv2.rectangle(c, (int(x1 - b[0]), int(y1 - b[1])), (int(x2 - b[0]), int(y2 - b[1])), (0, 255, 255), 2)
    return cv2.resize(c, (360, 360), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    (a.out / 'clean').mkdir(parents=True, exist_ok=True); (a.out / 'compare').mkdir(exist_ok=True)
    rec = {}
    for name in ('still', 'video'):
        for line in (SNAP / 'output' / f'{name}_matches.jsonl').read_text().splitlines():
            r = json.loads(line); rec[r['event_id']] = r
    items = json.loads((AUTO / 'assets/items.json').read_text())
    units = [u for u in items if u['split'] == 'unlabeled' and any(c['class_name'] == '道路裂纹' for c in u['candidates'])]
    allow, audit = [], []
    for u in units:
        r = rec.get(u['unit'])
        if not r or not r.get('clean_usable_for_inference') or r['source_kind'] != 'video':
            audit.append({'unit': u['unit'], 'skipped': 'no_usable_video_match'}); print(u['unit'], 'skip', flush=True); continue
        ref = cv2.imread(str(SNAP / r['clean_frame']), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        shot = cv2.imread(str(AUTO / 'assets/images' / (u['unit'] + '.jpg')))
        (frame, score, idx), n = best_frame(SNAP / r['source_local_path'], r['matched_raw_time'], r['homography_annotated_to_source'], ref)
        sh, sw = shot.shape[:2]
        cands, outside = [], []
        for c in u['candidates']:
            try: cands.append(dict(c, bbox=project(c['bbox'], r['homography_annotated_to_source'], sw, sh), screenshot_bbox=c['bbox']))
            except ValueError: outside.append(c['key'])
        cv2.imwrite(str(a.out / 'clean' / (u['unit'] + '.jpg')), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for c in cands:
            if c['class_name'] != '道路裂纹': continue
            pair = np.concatenate([crop(shot, c['screenshot_bbox']), np.full((360, 8, 3), 255, np.uint8), crop(frame, c['bbox'])], axis=1)
            cv2.imwrite(str(a.out / 'compare' / (c['key'].replace(':', '_') + '.jpg')), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
        allow.append({'unit': u['unit'], 'candidates': [{'key': c['key'], 'class_name': c['class_name'], 'bbox': c['bbox']} for c in cands]})
        audit.append({'unit': u['unit'], 'frames_decoded': n, 'best_index': idx, 'ncc': round(score, 4), 'outside': outside})
        print(u['unit'], n, idx, round(score, 4), outside, flush=True)
    (a.out / 'allowlist.json').write_text(json.dumps(allow, ensure_ascii=False))
    (a.out / 'frames-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
