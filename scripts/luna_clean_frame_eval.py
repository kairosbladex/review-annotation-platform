#!/usr/bin/env python3
"""离线、配对、两次重复的 Luna 干净帧实验；不连接业务 API，不更新自动迭代状态。

prepare 在受限进程中冻结输入并提取原帧；run 只调用独立 worker；score 离线计分。
原图不绘框，clean_bbox 记录投影坐标；两组提示词、前三张图、候选顺序完全一致。
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).parent / 'luna_prompt_autotune'))
import promptlib

AUTO = Path('/data/workspace/jzheng/luna-prompt-autotune')
SNAP = Path('/data/workspace/jzheng/codex-ktd-real-monitoring-20260924')
PORT = 18398


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + '\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def health():
    mem = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    status = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}}', 'yoloe_service_test'], text=True, timeout=5).strip()
    result = {'at': time.time(), 'available_fraction': mem['MemAvailable'] / mem['MemTotal'], 'shared': status}
    if result['available_fraction'] < .15 or not status.startswith('running healthy '):
        raise RuntimeError('资源或共享服务健康门槛未通过')
    return result


def project(box, H, sw, sh, rw, rh):
    import numpy as np
    x1, y1, x2, y2 = box
    pts = np.array([[x1*sw, y1*sh, 1], [x2*sw, y1*sh, 1], [x2*sw, y2*sh, 1], [x1*sw, y2*sh, 1]]).T
    q = np.asarray(H) @ pts
    if not np.isfinite(q).all() or (abs(q[2]) < 1e-8).any():
        raise ValueError('invalid_projection')
    q = q[:2] / q[2]
    raw = [float(q[0].min()/rw), float(q[1].min()/rh), float(q[0].max()/rw), float(q[1].max()/rh)]
    clipped = [max(0, raw[0]), max(0, raw[1]), min(1, raw[2]), min(1, raw[3])]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError('box_outside_clean_frame')
    return clipped, raw


def ncc(a, b):
    import numpy as np
    x, y = a.astype(np.float32), b.astype(np.float32)
    x -= x.mean(); y -= y.mean()
    return float((x*y).sum() / (np.sqrt((x*x).sum() * (y*y).sum()) + 1e-9))


def video_frame(path, timestamp, H, shot):
    """用 ffmpeg 实际帧时间定位；流式读帧避免将三秒原始视频全部装入内存。"""
    import cv2
    import numpy as np
    info = read_probe(path)
    w, h = info['width'], info['height']
    start = max(0, timestamp - 1.5)
    inv = np.linalg.inv(np.asarray(H))
    # 参照快照已验证的无标注对齐帧。先完整投影再面积降采样，避免树木纹理混叠。
    small = cv2.resize(shot, (240, 180), interpolation=cv2.INTER_AREA)
    cmd = ['ffmpeg', '-v', 'error', '-threads', '1', '-ss', str(start), '-i', str(path), '-t', '3', '-an', '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    best, best_score, best_index = None, -2, None
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        k = 0
        while True:
            buf = proc.stdout.read(w*h*3)
            if not buf:
                break
            if len(buf) != w*h*3:
                raise RuntimeError('truncated_video_frame')
            f = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            back = cv2.warpPerspective(f, inv, (shot.shape[1], shot.shape[0]))
            back = cv2.resize(back, (240, 180), interpolation=cv2.INTER_AREA)
            score = ncc(cv2.cvtColor(back, cv2.COLOR_BGR2GRAY), small)
            if score > best_score:
                best, best_score, best_index = f.copy(), score, k
            k += 1
        if proc.wait(timeout=30):
            raise RuntimeError('ffmpeg_failed')
    if best is None:
        raise ValueError('no_decoded_frame')
    return best, {'search_start': start, 'decoded_frames': k, 'frame_index': best_index, 'match_ncc': best_score,
                  'nominal_frame_time': start + best_index/info['fps'], 'time_is_approximate': True}


def read_probe(path):
    d = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate', '-of', 'json', str(path)], timeout=30))['streams'][0]
    a, b = map(float, d['avg_frame_rate'].split('/'))
    return dict(width=d['width'], height=d['height'], fps=a/b)


def prepare(root):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-3:])
    import cv2
    cv2.setNumThreads(1)
    assets = AUTO / 'assets'
    evidence = root / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    items = [u for u in read(assets/'items.json') if u['split'] in ('dev', 'holdout')]
    labels = read(assets/'labels.json')
    best = read(AUTO/'state/best.json') if (AUTO/'state/best.json').exists() else None
    version = best['version'] if best and best['version'] not in ('p000', 'p000b') else 'p001'
    write(evidence/'prompt.json', read(AUTO/'state/prompts'/f'{version}.json'))
    write(evidence/'labels.json', labels)
    write(evidence/'all-items.json', items)
    write(evidence/'freeze.json', {'at': time.time(), 'prompt_version': version, 'best': best, 'items_sha256': sha(assets/'items.json'), 'labels_sha256': sha(assets/'labels.json'), 'prompt_sha256': sha(evidence/'prompt.json'), 'health': health(), 'port': PORT, 'model': 'gpt-5.6-luna', 'selection_uses_labels': False})
    records = {}
    for name in ('still', 'video'):
        for line in (SNAP/'output'/f'{name}_matches.jsonl').read_text().splitlines():
            r = json.loads(line)
            records[r['event_id']] = r
    hashes = collections.defaultdict(list)
    for p in sorted((SNAP/'work/annotated').glob('*.jpg')):
        hashes[sha(p)].append(p.stem)
    included, excluded = [], []
    (root/'clean').mkdir(exist_ok=True)
    for index, u in enumerate(items):
        health()
        image = assets/'images'/(u['unit']+'.jpg')
        base = {k: u[k] for k in ('unit', 'split', 'source')}
        try:
            assert sha(image) == u['sha256'], 'input_hash_mismatch'
            eids = hashes.get(u['sha256'], [])
            if not eids:
                raise ValueError('no_identical_snapshot_screenshot')
            eligible = [e for e in eids if records.get(e, {}).get('clean_usable_for_inference')]
            if not eligible:
                raise ValueError('snapshot_clean_quality_gate_rejected_or_missing')
            eid = u['unit'] if u['unit'] in eligible else sorted(eligible)[0]
            r = records[eid]
            shot = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            H = r['homography_annotated_to_source']
            if r['source_kind'] == 'video':
                reference = SNAP/r['clean_frame']
                clean_reference = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
                if clean_reference is None or clean_reference.shape != shot.shape:
                    raise ValueError('verified_clean_reference_missing_or_wrong_size')
                frame, audit = video_frame(SNAP/r['source_local_path'], r['matched_raw_time'], H, clean_reference)
                audit['reference_clean_sha256'] = sha(reference)
                audit['reference'] = 'snapshot_verified_aligned_clean_frame'
                if audit['match_ncc'] < .80:
                    raise ValueError('decoded_frame_ncc_below_0.80')
            else:
                frame = cv2.imread(str(SNAP/r['source_local_path']))
                audit = {'match_ncc': r.get('full_ncc'), 'source_kind': 'still'}
            if frame is None:
                raise ValueError('raw_source_unreadable')
            rh, rw = frame.shape[:2]
            cands = []
            for c in u['candidates']:
                try:
                    bbox, raw_bbox = project(c['bbox'], H, shot.shape[1], shot.shape[0], rw, rh)
                    cands.append(dict(c, clean_bbox=bbox, unclipped_clean_bbox=raw_bbox))
                except ValueError as e:
                    excluded.append(dict(base, key=c['key'], level='candidate', reason=str(e)))
            if not cands:
                raise ValueError('all_candidates_outside_clean_frame')
            dest = root/'clean'/(u['unit']+'.jpg')
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            included.append(dict(u, candidates=cands, event_id=eid, clean_sha256=sha(dest), clean_size=[rw, rh],
                                 screenshot_size=[shot.shape[1], shot.shape[0]], H=H, source_kind=r['source_kind'],
                                 source_local_path=r['source_local_path'], snapshot_quality_gate=r.get('clean_quality_gate'),
                                 snapshot_full_ncc=r.get('full_ncc'), snapshot_roi_correlations=r.get('roi_correlations'), frame_audit=audit))
            print(f'prepare {index+1}/{len(items)} {u["unit"]} included {len(cands)}', flush=True)
        except (ValueError, AssertionError) as e:
            excluded.append(dict(base, level='image', reason=str(e), candidate_keys=[c['key'] for c in u['candidates']]))
            print(f'prepare {index+1}/{len(items)} excluded {e}', flush=True)
        write(evidence/'included.json', included)
        write(evidence/'excluded.json', excluded)
    write(evidence/'prepare-complete.json', {'included_images': len(included), 'excluded_records': len(excluded), 'health': health()})


def busy():
    # 每次新发请求前检查，不停止/修改自动迭代进程或定时器。
    state = subprocess.check_output(['systemctl', '--user', 'show', 'luna-autotune-runner.service', '-p', 'ActiveState', '--value'], text=True, timeout=5).strip()
    return state in ('active', 'activating', 'reloading')


def audit_exclusions(root):
    """将无相同截图的样本与原片文件哈希核对，不凭分辨率猜测是否已经是干净图。"""
    e = root/'evidence'
    excluded, items = read(e/'excluded.json'), {u['unit']: u for u in read(e/'all-items.json')}
    raw_by_hash = collections.defaultdict(list)
    paths = set()
    for line in (SNAP/'output/still_matches.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('source_local_path'):
            paths.add(r['source_local_path'])
        for attempt in r.get('attempts', []):
            if attempt.get('local_path'):
                paths.add(attempt['local_path'])
    for rel in sorted(paths):
        p = SNAP/rel
        if p.is_file():
            raw_by_hash[sha(p)].append(rel)
    for x in excluded:
        if x['reason'] == 'no_identical_snapshot_screenshot':
            matches = raw_by_hash.get(items[x['unit']]['sha256'])
            x['raw_source_sha_matches'] = matches or []
            x['detail'] = 'already_clean_original_no_screenshot_comparison' if matches else 'no_verified_snapshot_lineage'
    write(e/'excluded.json', excluded)
    print('excluded detail', collections.Counter(x.get('detail', x['reason']) for x in excluded), flush=True)


def run(root):
    evidence = root/'evidence'
    assert (evidence/'prepare-complete.json').exists()
    units = read(evidence/'included.json')
    doc = read(evidence/'prompt.json')
    token = (root/'token').read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rng = random.Random(20261009)
    jobs = []
    # 每图交错四条件，随机顺序；避免整轮跨时段造成输入效果与服务变化混淆。
    rng.shuffle(units)
    for u in units:
        variants = [(arm, repeat) for arm in ('screenshot', 'plus_clean') for repeat in (1, 2)]
        rng.shuffle(variants)
        jobs.extend((u, arm, repeat) for arm, repeat in variants)
    write(evidence/'run-design.json', {'seed': 20261009, 'planned_requests': len(jobs), 'max_concurrency': 4, 'concurrency_while_autotune_active': 2, 'uncertain_is_rejected': True,
                                      'conditions': ['screenshot-1', 'screenshot-2', 'plus_clean-1', 'plus_clean-2'], 'prompt_cards_unchanged': True})

    def one(u, arm, repeat):
        target = evidence/'results'/f'{arm}-{repeat}'/(u['unit']+'.json')
        if target.exists() and read(target)['success']:
            return
        image = AUTO/'assets/images'/(u['unit']+'.jpg')
        clean = root/'clean'/(u['unit']+'.jpg')
        assert sha(image) == u['sha256'] and sha(clean) == u['clean_sha256']
        candidates = [dict(target_index=i, class_name=c['class_name'], bbox=c['bbox'], clean_bbox=c['clean_bbox']) for i, c in enumerate(u['candidates'])]
        prompt = promptlib.build(doc, candidates) + '\n输入坐标说明：bbox 是第一张截图的归一化框；clean_bbox 是第四张干净原图上的对应归一化投影框。第四张未提供时仅用前三张；不把框线文字当成物体。'
        payload = {'image': base64.b64encode(image.read_bytes()).decode(), 'candidates': candidates, 'prompt': prompt}
        if arm == 'plus_clean':
            payload['clean_image'] = base64.b64encode(clean.read_bytes()).decode()
        attempts = read(target).get('attempts', []) if target.exists() else []
        result = None
        for attempt in range(2):
            h = health()
            t = time.monotonic()
            try:
                req = urllib.request.Request(f'http://127.0.0.1:{PORT}/review', data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
                with opener.open(req, timeout=240) as response:
                    body = json.load(response)
                by = {x['target_index']: x for x in body['results']}
                assert set(by) == set(range(len(candidates)))
                result = [dict(key=c['key'], **by[i]) for i, c in enumerate(u['candidates'])]
                status = 'ok'
            except Exception as e:
                status = f'http_{e.code}' if hasattr(e, 'code') else type(e).__name__
            attempts.append({'status': status, 'seconds': round(time.monotonic()-t, 2), 'health': h})
            if result is not None:
                break
            time.sleep(20)
        write(target, {'unit': u['unit'], 'arm': arm, 'repeat': repeat, 'success': result is not None, 'results': result or [], 'attempts': attempts,
                       'execution_id': body.get('execution_id') if result is not None else None, 'requested_model': body.get('requested_model') if result is not None else None,
                       'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        print(f'{arm}-{repeat} {u["unit"]} {status}', flush=True)

    pending = iter(jobs)
    active, exhausted = set(), False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while active or not exhausted:
            limit = 2 if busy() else 4
            while len(active) < limit and not exhausted:
                try:
                    active.add(pool.submit(one, *next(pending)))
                except StopIteration:
                    exhausted = True
            if active:
                done, active = concurrent.futures.wait(active, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    f.result()
    write(evidence/'run-complete.json', {'at': time.time(), 'health': health()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'run', 'audit_exclusions'])
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    globals()[args.action](args.root)


if __name__ == '__main__':
    main()
