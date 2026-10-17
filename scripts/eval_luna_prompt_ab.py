#!/usr/bin/env python3
"""Luna 提示词 A/B 新样本回放（服务器端执行，须在资源受限 scope 内运行）。

只读取任务目录 allowlist.json 与 images/，不读取金标。每张图最多两次调用，
持久化尝试账本，重启不重复调用。执行失败只记失败，不当作否定结论。
"""
from __future__ import annotations
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

MODEL = 'gpt-5.6-luna'
ALLOWED_TASKS = {Path('/REDACTED_RUNTIME_ROOT/codex-luna-prompt-v2-20261009'), Path('/REDACTED_RUNTIME_ROOT/codex-luna-prompt-v3-20261009')}


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def health():
    mem = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    state = subprocess.check_output(['docker', 'inspect', '--format',
        '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}', 'yoloe_service_test'], timeout=4, text=True).strip()
    value = dict(time=time.time(), available_fraction=mem['MemAvailable'] / mem['MemTotal'], shared_state=state)
    if value['available_fraction'] < .15 or state != 'running healthy':
        raise RuntimeError('resource_or_service_guard')
    return value


def normalize(body, n):
    items = body['results']
    if body.get('requested_model') != MODEL or len(items) != n or any(type(x.get('target_index')) is not int for x in items):
        raise ValueError('shape')
    indexed = {x['target_index']: x for x in items}
    if set(indexed) != set(range(n)) or any(x.get('decision') not in {'true', 'false', 'uncertain'} for x in items):
        raise ValueError('mapping')
    return [dict(target_index=i, rawDecision=indexed[i]['decision'], reasoning=indexed[i].get('reasoning'),
                 observedObject=indexed[i].get('observed_object'), imageLimitation=indexed[i].get('image_limitation')) for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=['baseline', 'v2', 'v3'])
    p.add_argument('--task-dir', type=Path, required=True)
    p.add_argument('--skip-source', default='', help='跳过该来源的图片（如已复用旧结果）')
    p.add_argument('--url', required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--concurrency', type=int, default=4)
    args = p.parse_args()
    root = args.task_dir.resolve()
    if root not in ALLOWED_TASKS: raise SystemExit('unexpected task directory')
    manifest = [u for u in json.loads((root / 'allowlist.json').read_text()) if u.get('source') != args.skip_source]
    token = None
    for line in args.token_file.read_text().splitlines():
        k, sep, v = line.partition('=')
        if sep and k.strip() == 'CODEX_REVIEW_WORKER_TOKEN': token = v.strip().strip('"\'')
    if not token: raise SystemExit('missing worker credential')
    out = root / ('out-' + args.arm); out.mkdir(exist_ok=True)
    ledger = root / ('attempts-' + args.arm); ledger.mkdir(exist_ok=True)
    health()
    stop = threading.Event()

    def guard():
        while not stop.wait(5):
            try:
                v = health()
                with (root / f'health-{args.arm}.jsonl').open('a') as f: f.write(json.dumps(v) + '\n')
            except Exception:
                write_json(root / f'guard-stop-{args.arm}.json', {'time': time.time()})
                os.kill(os.getpid(), signal.SIGTERM); return
    threading.Thread(target=guard, daemon=True).start()

    def run(u):
        uid = u['unit']; target = out / (uid + '.json'); log = ledger / (uid + '.json')
        if target.exists(): return
        attempts = json.loads(log.read_text()) if log.exists() else []
        raw = (root / 'images' / (uid + '.jpg')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == u['sha256']
        payload = json.dumps(dict(image=base64.b64encode(raw).decode(), candidates=[
            dict(target_index=i, class_name=c['class_name'], bbox=c['bbox']) for i, c in enumerate(u['candidates'])])).encode()
        results, started = None, time.monotonic()
        for attempt in range(len(attempts), 2):
            health()
            if attempt: time.sleep(30)
            attempts.append({'attempt': attempt + 1, 'started': time.time(), 'status': 'in_flight'}); write_json(log, attempts)
            t = time.monotonic()
            try:
                req = urllib.request.Request(args.url, data=payload, headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=240) as r: body = json.load(r)
                results = normalize(body, len(u['candidates'])); status = 'ok'
            except urllib.error.HTTPError as exc: status = 'http_' + str(exc.code)
            except Exception as exc: status = type(exc).__name__
            attempts[-1].update(status=status, elapsed=time.monotonic() - t); write_json(log, attempts)
            if results is not None: break
        write_json(target, dict(unit=uid, keys=[c['key'] for c in u['candidates']], success=results is not None,
                                results=results or [], elapsed=time.monotonic() - started, attempts=attempts))
        print(json.dumps({'unit': uid, 'success': results is not None, 'attempts': len(attempts)}), flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(run, manifest))
    finally:
        stop.set()
    write_json(root / f'final-health-{args.arm}.json', health())


if __name__ == '__main__':
    main()
