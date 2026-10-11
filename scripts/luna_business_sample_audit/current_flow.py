"""当前算法 API 到 Codex CLI 的限量复检；平台数据库仅用于只读取样。"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import threading
import time
import urllib.request
import uuid


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.chmod(0o600)
    tmp.replace(path)


def normalize(rows):
    if isinstance(rows, str):
        rows = json.loads(rows)
    result = []
    for row in rows:
        if not {'frame_width', 'frame_height', 'width', 'height', 'x', 'y'} <= row.keys():
            raise ValueError('missing_complete_initial_bbox')
        w, h = float(row['frame_width']), float(row['frame_height'])
        box = [float(row['x']) / w, float(row['y']) / h,
               (float(row['x']) + float(row['width'])) / w,
               (float(row['y']) + float(row['height'])) / h]
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError('invalid_initial_box')
        result.append(dict(class_name=row['class_name'], bbox=box,
                           confidence=float(row.get('confidence', 1))))
    if not result:
        raise ValueError('empty_candidates')
    return result


def match_gold(candidates, labeled):
    """只允许同类、完全同框且唯一的匹配，不用近邻 IoU 猜配。"""
    matched = {}
    for gold in labeled:
        indices = [i for i, c in enumerate(candidates)
                   if c['class_name'] == gold['class_name']
                   and max(abs(a-b) for a, b in zip(c['bbox'], gold['bbox'])) <= 1e-8]
        if len(indices) != 1:
            continue
        i = indices[0]
        if i in matched and matched[i]['label'] != gold['label']:
            raise ValueError('conflicting_gold')
        matched[i] = gold
    return matched


def validate(body, candidates):
    if body.get('success') is not True:
        raise ValueError('execution_failed')
    rows = body.get('results', [])
    if len(rows) != len(candidates):
        raise ValueError('result_count_mismatch')
    execution_ids = set()
    for result, candidate in zip(rows, candidates):
        if result.get('original_class') != candidate['class_name'] or result.get('original_bbox') != candidate['bbox']:
            raise ValueError('target_mapping_mismatch')
        raw = result.get('rawDecision')
        if raw not in {'true', 'false', 'uncertain'}:
            raise ValueError('invalid_decision')
        if result.get('decision') != ('confirmed' if raw == 'true' else 'rejected'):
            raise ValueError('policy_mismatch')
        meta = result.get('codexReview', {})
        if meta.get('transport') != 'official_codex_cli' or meta.get('requestedModel') != 'gpt-5.6-luna':
            raise ValueError('transport_or_model_mismatch')
        if not meta.get('executionId'):
            raise ValueError('missing_execution_id')
        execution_ids.add(meta['executionId'])
    if len(execution_ids) != 1:
        raise ValueError('execution_id_mismatch')
    return rows


def prepare(args):
    from snapshot import connect
    items = json.loads((args.frozen/'items.json').read_text())
    labels = json.loads((args.frozen/'labels.json').read_text())
    selected = [x for x in items if re.fullmatch('[0-9a-f]{32}', x['unit'])
                and any(c['key'] in labels for c in x['candidates'])]
    assert len({x['unit'] for x in selected}) == len(selected)
    ids = [x['unit'] for x in selected]
    connection = connect(args.env)
    try:
        with connection.cursor() as cursor:
            for sql in ['SET SESSION MAX_EXECUTION_TIME=20000',
                        'SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ',
                        'SET SESSION TRANSACTION READ ONLY',
                        'START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY']:
                cursor.execute(sql)
            cursor.execute('SELECT NOW() snapshot_time,DATABASE() db,@@session.transaction_read_only ro')
            identity = cursor.fetchone()
            assert identity['db'] == 'ktd_lx' and identity['ro'] == 1
            placeholders = ','.join(['%s']*len(ids))
            cursor.execute(f'SELECT id,event_no,event_time,task_id,imgxy,recheck_status,recheck_time FROM bo_task_event WHERE id IN ({placeholders}) AND del_flag=0', ids)
            events = {x['id']: x for x in cursor.fetchall()}
            cursor.execute(f'SELECT id,busi_id,busi_type_code,file_path,imgxy,image_resolution FROM sys_annex_file WHERE busi_id IN ({placeholders}) AND del_flag=0', ids)
            annexes = cursor.fetchall()
    finally:
        connection.rollback()
        connection.close()
    save(args.out/'private/db-before.json', dict(identity=identity, events=events,
         annexes=annexes, transaction_end='rollback'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def one(item):
        uid = item['unit']
        event = events.get(uid)
        shots = [x for x in annexes if x['busi_id'] == uid and x['busi_type_code'] == 'SHOTIMG']
        if event is None or len(shots) != 1:
            return None, dict(unit=uid, reason='missing_or_ambiguous_event_image')
        shot = shots[0]
        try:
            candidates = normalize(shot['imgxy'] or event['imgxy'])
            gold = [dict(c, label=labels[c['key']]['label']) for c in item['candidates'] if c['key'] in labels]
            matches = match_gold(candidates, gold)
            if not matches:
                return None, dict(unit=uid, reason='no_exact_labeled_box')
            url = args.image_base.rstrip('/') + '/' + shot['file_path'].lstrip('/')
            with opener.open(url, timeout=25) as response:
                data = response.read(20_000_001)
            if len(data) > 20_000_000 or digest(data) != item['sha256']:
                return None, dict(unit=uid, reason='current_image_sha_mismatch')
            return dict(unit=uid, event_time=str(event['event_time']), task_id=event['task_id'],
                        image_url=url, image_sha256=digest(data), image_bytes=len(data),
                        candidates=candidates, gold={str(k): v for k, v in matches.items()},
                        label_candidates=len(gold), source=item['source'], split=item['split']), None
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ValueError) and str(exc) in {'missing_complete_initial_bbox', 'invalid_initial_box', 'empty_candidates', 'conflicting_gold'} else type(exc).__name__
            return None, dict(unit=uid, reason=reason)

    units, excluded = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for unit, failure in pool.map(one, selected):
            (units if unit else excluded).append(unit or failure)
    units.sort(key=lambda x: (x['event_time'], x['unit']))
    save(args.out/'private/units.json', units)
    save(args.out/'private/excluded.json', excluded)
    by_class = collections.defaultdict(collections.Counter)
    for unit in units:
        for gold in unit['gold'].values():
            by_class[gold['class_name']][gold['label']] += 1
    summary = dict(snapshot=identity, source_events=len(selected), eligible_events=len(units),
                   initial_candidates=sum(len(x['candidates']) for x in units),
                   labeled_candidates=sum(len(x['gold']) for x in units),
                   unmatched_labeled_candidates=sum(x['label_candidates']-len(x['gold']) for x in units),
                   excluded_reasons=collections.Counter(x['reason'] for x in excluded),
                   period=[min(x['event_time'] for x in units), max(x['event_time'] for x in units)] if units else [],
                   by_class=by_class, labels_sha256=digest((args.frozen/'labels.json').read_bytes()),
                   units_sha256=digest((args.out/'private/units.json').read_bytes()),
                   policy='only_true_is_true_v1', clean_image=False,
                   scope='原始初审截图与全部初审候选经当前算法API复检；仅人工已标注且精确匹配的框计分；无平台复检回写。')
    save(args.out/'selection.json', summary)
    print(json.dumps(summary, ensure_ascii=False, default=str))


def guard():
    memory = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    if memory['MemAvailable']/memory['MemTotal'] < .15:
        raise RuntimeError('resource_guard')
    return memory['MemAvailable']


def run(args):
    units = json.loads((args.out/'units.json').read_text())
    container = json.loads(subprocess.check_output(['docker', 'inspect', 'yoloe_service_test']))[0]
    assert container['State']['Health']['Status'] == 'healthy'
    env = dict(x.split('=', 1) for x in container['Config']['Env'] if '=' in x)
    headers = {'X-API-Key': env['API_KEY']}
    # 服务使用既有自签证书，仅允许本机固定地址；不扩大系统 TLS 策略。
    assert args.endpoint == 'https://127.0.0.1:8189/api/v1/review-agent/review'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                        urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    save(args.out/'runtime.json', dict(container_id=container['Id'], image_id=container['Image'],
         started_at=container['State']['StartedAt'], enabled=env.get('CODEX_REVIEW_ENABLED'),
         concurrency=4, memory_available_kib=guard(), request_attempts_per_event=1,
         started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    mounts = {m['Destination']: Path(m['Source']) for m in container['Mounts']}
    failure_state = {'consecutive': 0}
    lock = threading.Lock()
    stop = threading.Event()
    def one(pair):
        number, unit = pair
        uid = unit['unit']; target = args.out/'responses'/f'{uid}.json'
        ledger = args.out/'attempts'/f'{uid}.json'
        if ledger.exists() or stop.is_set():
            return  # 中断、失败均不自动重试。
        guard()
        boundary = uuid.uuid4().hex
        image = (args.image_root/(uid+'.jpg')).read_bytes()
        assert digest(image) == unit['image_sha256']
        payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="detections"\r\n\r\n'
                   + json.dumps(unit['candidates'])
                   + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="initial.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
        payload += image + f'\r\n--{boundary}--\r\n'.encode()
        attempt = dict(unit=uid, attempt_uid=uuid.uuid4().hex, started=time.time(), status='in_flight')
        save(ledger, attempt)
        request = urllib.request.Request(args.endpoint, data=payload,
                   headers=dict(headers, **{'Content-Type': f'multipart/form-data; boundary={boundary}'}))
        try:
            with opener.open(request, timeout=270) as response:
                body = json.load(response)
            save(target, body)
            validate(body, unit['candidates'])
            filename = body['original_url'].rsplit('/', 1)[-1]
            assert re.fullmatch('[0-9a-f]{8}_orig\\.jpg', filename)
            source_path = mounts['/app/uploads']/'review_agent'/filename
            result_path = mounts['/app/results']/'review_agent'/(body['session_id']+'.model-results.json')
            assert digest(source_path.read_bytes()) == unit['image_sha256']
            assert json.loads(result_path.read_text()) == body
            attempt.update(api_input_sha256=unit['image_sha256'], session_id=body['session_id'],
                           execution_id=body['results'][0]['codexReview']['executionId'],
                           api_source_file=str(source_path), api_result_file=str(result_path),
                           api_result_sha256=digest(result_path.read_bytes()))
            attempt['status'] = 'ok'
            with lock:
                failure_state['consecutive'] = 0
        except Exception as exc:
            attempt.update(status='failed', error_type=type(exc).__name__)
            if hasattr(exc, 'code'):
                attempt['http_status'] = exc.code
            with lock:
                failure_state['consecutive'] += 1
                if failure_state['consecutive'] >= 3:
                    stop.set()
        attempt['finished'] = time.time()
        save(ledger, attempt)
        print(json.dumps(dict(completed=number, planned=len(units), status=attempt['status'],
                              elapsed=round(attempt['finished']-attempt['started'], 2))), flush=True)
    selected = units[:args.limit] if args.limit else units
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, enumerate(selected, 1)))
    if stop.is_set():
        raise RuntimeError('three_consecutive_failures_no_retry')
    save(args.out/'finished.json', dict(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                      memory_available_kib=guard()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--env', type=Path)
    parser.add_argument('--image-base', default='https://lxtest.zhuoyilingxi.com:8088/minio-src')
    parser.add_argument('--endpoint', default='https://127.0.0.1:8189/api/v1/review-agent/review')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--image-root', type=Path, default=Path('/data/workspace/jzheng/luna-prompt-autotune/assets/images'))
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
