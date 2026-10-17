#!/usr/bin/env python3
"""从 09-24 快照已核验的原视频（MinIO 原片的服务器缓存）取出与截图同一时刻的 1920×1080 原帧，
并用截图→原片单应矩阵把已标注框投影到原帧坐标。

输入 hd-plan.json：[{event_id, detections[{candidate_id, class_name, bbox(截图归一化)}], video, time, H}]
输出 <out>/hd/{event_id}.jpg 与 <out>/hd-allowlist.json（框改为原帧归一化坐标）。不读金标。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

SNAPSHOT = Path('/REDACTED_RUNTIME_ROOT/codex-ktd-real-monitoring-20260924')
ANNOTATED = SNAPSHOT / 'work' / 'annotated'
SHOT_W, SHOT_H = 960, 720
WINDOW = 1.5


def project(bbox: list[float], H: np.ndarray, w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    pts = np.array([[[x1 * SHOT_W, y1 * SHOT_H]], [[x2 * SHOT_W, y1 * SHOT_H]],
                    [[x1 * SHOT_W, y2 * SHOT_H]], [[x2 * SHOT_W, y2 * SHOT_H]]], dtype=np.float64)
    p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    nx1, ny1 = max(0.0, p[:, 0].min() / w), max(0.0, p[:, 1].min() / h)
    nx2, ny2 = min(1.0, p[:, 0].max() / w), min(1.0, p[:, 1].max() / h)
    if not (nx1 < nx2 and ny1 < ny2):
        raise ValueError('box projected outside frame')
    return [nx1, ny1, nx2, ny2]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def best_frame(video: Path, t: float, H: np.ndarray, shot: np.ndarray, dets: list[dict]):
    """快照记录的匹配时刻附近 ±WINDOW 秒逐帧解码，取与截图最相似的一帧（OpenCV 跳转不精确）。

    相似度 = 整图相关 与 各候选框周边相关最小值 的平均，避免小目标移动时只看背景。"""
    inv = np.linalg.inv(H)
    rois = []
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        m = 24
        rois.append((max(0, int(x1 * SHOT_W) - m), max(0, int(y1 * SHOT_H) - m),
                     min(SHOT_W, int(x2 * SHOT_W) + m), min(SHOT_H, int(y2 * SHOT_H) + m)))
    # OpenCV 遇到损坏包会提前停止解码、跳转也不精确，改用 ffmpeg 精确跳转后逐帧读出
    start = max(0.0, t - WINDOW)
    proc = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{start:.3f}', '-i', str(video), '-t', f'{2 * WINDOW:.3f}',
                           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-'],
                          capture_output=True, timeout=300)
    size = 1920 * 1080 * 3
    n = len(proc.stdout) // size
    fps = n / (2 * WINDOW) if n else 0
    best = None
    for k in range(n):
        frame = np.frombuffer(proc.stdout, np.uint8, size, k * size).reshape(1080, 1920, 3)
        back = cv2.cvtColor(cv2.warpPerspective(frame, inv, (SHOT_W, SHOT_H)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = ncc(back, shot)
        r = min(ncc(back[y1:y2, x1:x2], shot[y1:y2, x1:x2]) for x1, y1, x2, y2 in rois)
        if best is None or g + r > best[1] + best[2]:
            best = (frame.copy(), g, r, start + k / fps)
    if best is None:
        raise SystemExit(f'no frame decoded near {t} in {video}')
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    (args.out / 'hd').mkdir(parents=True, exist_ok=True)
    allow, audit = [], []
    for item in sorted(plan, key=lambda e: (e['video'], e['time'])):
        H = np.array(item['H'], dtype=np.float64)
        shot = cv2.cvtColor(cv2.imread(str(ANNOTATED / f"{item['event_id']}.jpg")), cv2.COLOR_BGR2GRAY).astype(np.float32)
        frame, ncc, roi, t_best = best_frame(SNAPSHOT / item['video'], item['time'], H, shot, item['detections'])
        h, w = frame.shape[:2]
        if (w, h) != (1920, 1080):
            raise SystemExit(f"unexpected size {item['event_id']} {w}x{h}")
        dets, outside = [], []
        for d in item['detections']:
            try:
                dets.append(dict(d, bbox=project(d['bbox'], H, w, h)))
            except ValueError:
                outside.append(d['candidate_id'])
        if not dets:
            print(item['event_id'], 'all boxes outside', flush=True)
            continue
        cv2.imwrite(str(args.out / 'hd' / f"{item['event_id']}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        allow.append(dict(event_id=item['event_id'], detections=dets))
        audit.append(dict(event_id=item['event_id'], matched_time=item['time'], frame_time=round(t_best, 3),
                          frame_ncc=round(ncc, 4), roi_ncc_min=round(roi, 4), outside=outside))
        print(item['event_id'], len(dets), round(t_best - item['time'], 2), round(ncc, 3), round(roi, 3), outside, flush=True)
    allow.sort(key=lambda e: e['event_id'])
    (args.out / 'hd-allowlist.json').write_text(json.dumps(allow, ensure_ascii=False, indent=1))
    (args.out / 'hd-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
