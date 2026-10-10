#!/usr/bin/env python3
"""复用 10-09 冻结样本；三张输入仅源自同一干净原帧，不读取平台截图。"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import socket
import subprocess
import threading
import time
import urllib.request

from luna_clean_frame_eval import SNAP, busy, health, promptlib, read, sha, write

PORT = 18399
PREFIX = 'luna-clean-only-20261010'
COORDINATES = '\n输入坐标说明：bbox 和 clean_bbox 都是第一张干净原图上的归一化投影框；定位图和目标局部裁剪图均由同一张干净原图生成。不把框线文字当成物体。'


def request_payload(root, unit, doc):
    image = root/'clean'/(unit['unit']+'.jpg')
    assert sha(image) == unit['clean_sha256'], 'clean_frame_hash_changed'
    candidates = [dict(target_index=i, class_name=c['class_name'], bbox=c['clean_bbox'], clean_bbox=c['clean_bbox'])
                  for i, c in enumerate(unit['candidates'])]
    prompt = promptlib.build(doc, candidates) + COORDINATES
    # 不接受截图路径；不发送第四张图。worker 用此 image 生成原图、定位图和裁剪图。
    return {'image': base64.b64encode(image.read_bytes()).decode(), 'candidates': candidates, 'prompt': prompt}


def restore_frame(unit):
    import cv2
    import numpy as np
    source = SNAP/unit['source_local_path']
    if unit['source_kind'] == 'still':
        return cv2.imread(str(source))
    audit = unit['frame_audit']; w, h = unit['clean_size']
    # 与上轮完全相同的解码命令和输出帧序号；不重新匹配、搜索或选帧。
    cmd = ['ffmpeg', '-v', 'error', '-threads', '1', '-ss', str(audit['search_start']), '-i', str(source),
           '-t', '3', '-an', '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    chosen = None; count = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        while True:
            raw = proc.stdout.read(w*h*3)
            if not raw: break
            assert len(raw) == w*h*3, 'truncated_frame'
            if count == audit['frame_index']:
                chosen = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            count += 1
        assert proc.wait(timeout=30) == 0
    assert count == audit['decoded_frames'] and chosen is not None, 'frame_sequence_changed'
    return chosen


def prepare(root):
    import cv2
    cv2.setNumThreads(1)
    e = root/'evidence'; e.mkdir(exist_ok=True)
    (root/'clean').mkdir(exist_ok=True)
    sentinel = e/'prepare-complete.json'
    if sentinel.exists(): sentinel.unlink()
    units = read(root/'baseline/included.json')
    manifest = []
    for index, unit in enumerate(units):
        health()
        target = root/'clean'/(unit['unit']+'.jpg')
        frame = restore_frame(unit)
        assert frame is not None and [frame.shape[1], frame.shape[0]] == unit['clean_size']
        assert cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert sha(target) == unit['clean_sha256'], 'restored_clean_frame_differs_from_previous_experiment'
        manifest.append({'unit': unit['unit'], 'split': unit['split'], 'image_sha256': sha(target),
                         'size': unit['clean_size'], 'candidates': request_payload(root, unit, read(root/'baseline/prompt.json'))['candidates']})
        print(f'prepare {index+1}/{len(units)} hash verified', flush=True)
    write(e/'input-manifest.json', manifest)
    write(sentinel, {'at': time.time(), 'images': len(manifest), 'all_clean_hashes_equal_previous': True, 'health': health()})


def run(root):
    e = root/'evidence'
    assert read(e/'prepare-complete.json')['images'] == 300
    units, doc = read(root/'baseline/included.json'), read(root/'baseline/prompt.json')
    token = (root/'token').read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rng = random.Random(20261010); rng.shuffle(units)
    jobs = []
    for u in units:
        repeats = [1, 2]; rng.shuffle(repeats)
        jobs.extend((u, n) for n in repeats)

    def one(unit, repeat):
        dest = e/'results'/f'clean_only-{repeat}'/(unit['unit']+'.json')
        if dest.exists() and read(dest)['success']: return
        payload = request_payload(root, unit, doc)
        attempts = read(dest)['attempts'] if dest.exists() else []
        result = None; body = {}
        for _ in range(2):
            state = health(); started = time.monotonic()
            try:
                req = urllib.request.Request(f'http://127.0.0.1:{PORT}/review', data=json.dumps(payload).encode(),
                                             headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
                with opener.open(req, timeout=240) as response: body = json.load(response)
                by = {x['target_index']: x for x in body['results']}
                assert set(by) == set(range(len(unit['candidates'])))
                assert all(x['decision'] in ('true', 'false', 'uncertain') for x in by.values())
                result = [dict(key=c['key'], **by[i]) for i, c in enumerate(unit['candidates'])]
                status = 'ok'
            except Exception as error:
                status = f'http_{error.code}' if hasattr(error, 'code') else type(error).__name__
            attempts.append({'at': time.time(), 'seconds': round(time.monotonic()-started, 2), 'status': status, 'health': state})
            if result is not None: break
            time.sleep(20)
        write(dest, {'unit': unit['unit'], 'repeat': repeat, 'arm': 'clean_only', 'success': result is not None,
                     'results': result or [], 'attempts': attempts, 'execution_id': body.get('execution_id') if result is not None else None,
                     'requested_model': body.get('requested_model') if result is not None else None,
                     'image_sha256': unit['clean_sha256'], 'prompt_sha256': hashlib.sha256(payload['prompt'].encode()).hexdigest(),
                     'image_count': 3, 'all_views_from_clean_frame': True})
        print(f'clean_only-{repeat} {unit["unit"]} {status}', flush=True)

    pending = iter(jobs); active = set(); exhausted = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while active or not exhausted:
            limit = 2 if busy() else 4
            while len(active) < limit and not exhausted:
                try: active.add(pool.submit(one, *next(pending)))
                except StopIteration: exhausted = True
            if active:
                done, active = concurrent.futures.wait(active, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done: future.result()
    write(e/'run-complete.json', {'at': time.time(), 'health': health()})


def properties(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ControlGroup'],
                            text=True, capture_output=True, timeout=10)
    return dict(x.split('=', 1) for x in result.stdout.splitlines() if '=' in x)


def launch(root, role, memory, cpu, command, extra=()):
    cpus = sorted(os.sched_getaffinity(0))[-3:]
    if cpu != '300%': cpus = cpus[-1:]
    name = PREFIX+'-'+role
    subprocess.run(['systemd-run', '--user', '--unit='+name, '-p', 'MemoryMax='+memory, '-p', 'MemorySwapMax=0',
                    '-p', 'CPUQuota='+cpu, '-p', 'KillMode=control-group', '-p', 'Restart=no',
                    '-p', 'Environment=OPENBLAS_NUM_THREADS=1', '-p', 'Environment=OMP_NUM_THREADS=1',
                    '-p', 'StandardOutput=append:'+str(root/(role+'.log')), '-p', 'StandardError=append:'+str(root/(role+'.log')),
                    *extra, '/usr/bin/taskset', '--cpu-list', ','.join(map(str, cpus)), *command], check=True)


def start(root):
    assert read(root/'evidence/prepare-complete.json')['images'] == 300
    with socket.socket() as sock: sock.bind(('127.0.0.1', PORT))
    os.umask(0o077)
    source = Path('/data/workspace/jzheng/luna-prompt-autotune/private/worker.env')
    env = dict(x.split('=', 1) for x in source.read_text().splitlines() if '=' in x and not x.startswith('#'))
    token = secrets.token_urlsafe(32); (root/'token').write_text(token)
    private = {k: env[k] for k in ('CODEX_REVIEW_BINARY', 'CODEX_REVIEW_HOME')}
    private.update(CODEX_REVIEW_WORKER_TOKEN=token, CODEX_REVIEW_BIND='127.0.0.1', CODEX_REVIEW_PORT=str(PORT),
                   CODEX_REVIEW_TMP_DIR=str(root/'tmp'), CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1')
    (root/'tmp').mkdir(exist_ok=True)
    (root/'worker.env').write_text(''.join(f'{k}={v}\n' for k, v in private.items()))
    wrapper = root/'src/clean_only_worker.py'
    wrapper.write_text('import sys,threading\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent/"luna_prompt_autotune"))\nimport luna_experiment_worker as w\nw.LIMIT=4\nw.SLOTS=threading.BoundedSemaphore(4)\nw.main()\n')
    try:
        launch(root, 'worker', '4G', '300%', ['/usr/bin/python3', str(wrapper)], ['-p', 'EnvironmentFile='+str(root/'worker.env')])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(20):
            try:
                with opener.open(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                    assert json.load(response)['max_concurrency'] == 4
                break
            except OSError: time.sleep(1)
        else: raise RuntimeError('worker_not_ready')
        props = properties(PREFIX+'-worker'); cg = Path('/sys/fs/cgroup')/props['ControlGroup'].lstrip('/')
        limits = {k: (cg/k).read_text().strip() for k in ('memory.max', 'memory.swap.max')}
        allowed = sorted(os.sched_getaffinity(int(props['MainPID'])))
        assert limits == {'memory.max': str(4*1024**3), 'memory.swap.max': '0'} and len(allowed) <= 3
        write(root/'evidence/effective-limits.json', {'at': time.time(), 'limits': limits, 'allowed_cpus': allowed,
                                                     'cpu_max': (cg/'cpu.max').read_text().strip() if (cg/'cpu.max').exists() else None,
                                                     'enforced_before_first_request': True})
        script = str(root/'src/luna_clean_only_eval.py')
        launch(root, 'runner', '1G', '100%', ['/usr/bin/python3', script, 'run', '--root', str(root)])
        launch(root, 'watch', '256M', '100%', ['/usr/bin/python3', script, 'watch', '--root', str(root)])
    except BaseException:
        subprocess.run(['systemctl', '--user', 'stop', PREFIX+'-runner', PREFIX+'-worker'], capture_output=True)
        raise


def watch(root):
    while True:
        mem = {x.split(':')[0]: int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
        worker, runner = properties(PREFIX+'-worker'), properties(PREFIX+'-runner')
        cg = Path('/sys/fs/cgroup')/worker.get('ControlGroup', '').lstrip('/')
        limits = {k: (cg/k).read_text().strip() for k in ('memory.current', 'memory.peak', 'memory.swap.current', 'memory.max', 'memory.swap.max') if (cg/k).exists()}
        shared = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}}', 'yoloe_service_test'], text=True, timeout=10).strip()
        state = {'at': time.time(), 'worker': worker, 'runner': runner, 'limits': limits, 'available_fraction': mem['MemAvailable']/mem['MemTotal'],
                 'load1': os.getloadavg()[0], 'shared': shared, 'autotune_busy': busy(),
                 'completed': sum(1 for _ in (root/'evidence/results').glob('*/*.json'))}
        with (root/'evidence/resource-samples.jsonl').open('a') as f: f.write(json.dumps(state)+'\n')
        danger = state['available_fraction'] < .15 or state['load1'] > 1.5*os.cpu_count() or not shared.startswith('running healthy ')
        finished = runner.get('ActiveState') not in ('active', 'activating', 'reloading')
        if danger or finished:
            if danger: subprocess.run(['systemctl', '--user', 'stop', PREFIX+'-runner'], check=True)
            subprocess.run(['systemctl', '--user', 'stop', PREFIX+'-worker'], check=True)
            write(root/'evidence/worker-stopped.json', {'at': time.time(), 'reason': 'resource_guard' if danger else 'runner_finished',
                                                       'worker': properties(PREFIX+'-worker'), 'runner': properties(PREFIX+'-runner')})
            return
        time.sleep(30)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['prepare', 'run', 'start', 'watch']); ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args(); globals()[args.action](args.root.resolve())
