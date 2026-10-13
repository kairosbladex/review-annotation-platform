#!/usr/bin/env python3
"""Bounded, allowlisted offline replay; run only inside a resource-limited scope.

Never reads source videos or gold labels. A durable attempt ledger prevents
accidental extra calls when restarting. Errors never become negative decisions.
"""
from __future__ import annotations
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

MODEL = 'gpt-5.6-luna'
POLICY = 'only_true_is_true_v1'
ANNOTATED = Path('/data/workspace/jzheng/codex-ktd-real-monitoring-20260924/work/annotated')
TASK = Path('/data/workspace/jzheng/codex-luna-eval-20261008')
ALLOWED_TASKS = {TASK, Path('/data/workspace/jzheng/codex-luna-hd-20261008')}
WORKER = Path('/data/workspace/jzheng/codex-luna-review')
URL = 'http://192.168.130.238:18396/review'

def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def normalize(body, detections):
    items = body['results']
    if body.get('requested_model') != MODEL or len(items) != len(detections):
        raise ValueError('model_or_length_mismatch')
    if any(type(x.get('target_index')) is not int for x in items):
        raise ValueError('invalid_target_index')
    indexed = {x['target_index']: x for x in items}
    if len(indexed) != len(items) or set(indexed) != set(range(len(detections))):
        raise ValueError('target_mapping_mismatch')
    results = []
    for i, d in enumerate(detections):
        x = indexed[i]
        raw = x.get('decision')
        if raw not in {'true', 'false', 'uncertain'} or not isinstance(x.get('reasoning'), str) or not x['reasoning'].strip():
            raise ValueError('invalid_decision')
        results.append(dict(target_index=i, original_class=d['class_name'], original_bbox=d['bbox'],
                            decision='confirmed' if raw == 'true' else 'rejected', rawDecision=raw,
                            reasoning=x['reasoning'], observedObject=x.get('observed_object'),
                            imageLimitation=x.get('image_limitation')))
    return dict(success=True, results=results, modelVersion=MODEL, policy=POLICY,
                execution_id=body.get('execution_id'))

def health():
    mem = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    state = subprocess.check_output(['docker', 'inspect', '--format',
        '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}', 'yoloe_service_test'], timeout=4, text=True).strip()
    value = dict(time=time.time(), available_fraction=mem['MemAvailable']/mem['MemTotal'],
                 available_kib=mem['MemAvailable'], shared_state=state)
    if value['available_fraction'] < .15 or state != 'running healthy':
        raise RuntimeError('resource_or_service_guard')
    return value

def main():
    p = argparse.ArgumentParser();p.add_argument('--task-dir', type=Path, default=TASK)
    # 原帧评测：--image-dir 指向从原视频取出的 1920×1080 帧，--allowlist 为投影到原帧坐标的框
    p.add_argument('--image-dir', type=Path, default=ANNOTATED)
    p.add_argument('--allowlist', default='allowlist.json')
    p.add_argument('--image-size', default='960x720')
    p.add_argument('--events', type=int, default=113);p.add_argument('--boxes', type=int, default=212)
    args = p.parse_args();root = args.task_dir.resolve()
    if root not in ALLOWED_TASKS: raise SystemExit('unexpected task directory')
    image_dir = args.image_dir;size = tuple(int(x) for x in args.image_size.split('x'))
    # Exclusively created by the caller, contains no videos, labels, or credentials.
    manifest = json.loads((root/args.allowlist).read_text())
    assert len(manifest) == args.events and sum(len(e['detections']) for e in manifest) == args.boxes
    assert len({e['event_id'] for e in manifest}) == args.events
    images = {}
    for e in manifest:
        assert set(e) == {'event_id', 'detections'} and re.fullmatch('[0-9a-f]{32}', e['event_id'])
        for d in e['detections']:
            assert set(d) == {'candidate_id', 'class_name', 'bbox'}
            assert d['candidate_id'].startswith(e['event_id'] + ':')
            a,b,c,f = d['bbox'];assert 0 <= a < c <= 1 and 0 <= b < f <= 1
        path = image_dir/(e['event_id']+'.jpg')
        # Pillow validates dimensions without making a new image or changing bytes.
        from PIL import Image
        with Image.open(path) as im: assert im.size == size
        images[e['event_id']] = hashlib.sha256(path.read_bytes()).hexdigest()
    token = None
    for line in (WORKER/'private.env').read_text().splitlines():
        key, sep, val = line.partition('=')
        if sep and key.strip() == 'CODEX_REVIEW_WORKER_TOKEN': token = val.strip().strip('\"\'')
    if not token: raise SystemExit('missing worker credential')
    initial = health()
    write_json(root/'input-audit.json', dict(events=args.events, boxes=args.boxes, image_dir=str(image_dir),
        image_size=args.image_size, image_sha256=images, health=initial,
        worker_source_sha256=hashlib.sha256((WORKER/'source/scripts/run_codex_luna_worker.py').read_bytes()).hexdigest(),
        concurrency=4, max_attempts_per_event=2, model=MODEL, policy=POLICY, clean_image=False))
    out = root/'out-luna';out.mkdir(exist_ok=True)
    ledger = root/'attempts';ledger.mkdir(exist_ok=True)
    stop = threading.Event()
    def guard():
        while not stop.wait(5):
            try:
                v = health()
                with (root/'health.jsonl').open('a') as f: f.write(json.dumps(v)+'\n')
            except Exception:
                write_json(root/'guard-stop.json', {'time':time.time(), 'reason':'resource_or_service_guard'})
                # Kill only this client. Worker has its own 15% guard and 215s CLI limit.
                os.kill(os.getpid(), signal.SIGTERM)
                return
    threading.Thread(target=guard, daemon=True).start()
    def run(e):
        eid=e['event_id'];target=out/(eid+'.json');log=ledger/(eid+'.json')
        if target.exists(): return
        attempts=json.loads(log.read_text()) if log.exists() else []
        started=time.monotonic()
        response={'success':False, 'results':[], 'errorCode':'attempt_budget_exhausted'}
        for attempt in range(len(attempts),2):
            health()
            if attempt: time.sleep(30)
            health()
            raw=(image_dir/(eid+'.jpg')).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==images[eid]
            payload=dict(image=base64.b64encode(raw).decode(), candidates=[
                dict(target_index=i,class_name=d['class_name'],bbox=d['bbox']) for i,d in enumerate(e['detections'])])
            req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={
                'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            attempts.append({'attempt':attempt+1,'started':time.time(),'status':'in_flight'})
            write_json(log,attempts);t=time.monotonic()
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req,timeout=240) as r: body=json.load(r)
                response=normalize(body,e['detections']);status='ok'
            except urllib.error.HTTPError as exc:
                status='http_'+str(exc.code);response={'success':False,'results':[],'errorCode':status}
            except Exception as exc:
                # Never persist request/header/exception text, which could contain secrets.
                status=type(exc).__name__;response={'success':False,'results':[],'errorCode':status}
            attempts[-1].update(status=status,elapsed=time.monotonic()-t);write_json(log,attempts)
            if response['success']: break
        write_json(target,dict(event_id=eid,candidate_ids=[d['candidate_id'] for d in e['detections']],
                               elapsed=time.monotonic()-started,response=response,attempts=attempts))
        print(json.dumps({'event_id':eid,'success':response['success'],'attempts':len(attempts)}),flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run,manifest))
    finally: stop.set()
    write_json(root/'final-health.json',health())

if __name__=='__main__': main()
