#!/usr/bin/env python3
"""Luna 提示词自动迭代执行器（宿主机 systemd 定时器每 5 分钟调用一次，flock 保证单实例）。

每次：检查停止条件 → 资源与共享服务健康 → 取队列中最早的提示词版本跑一整轮（dev + 无标签池）→
打分、对比基线、生成错判拼图 → 若为新最好版本，再跑锁定 holdout（结果只写 human/，不给 Hermes）。
`--brain-needed` 只判断是否需要唤醒 Hermes（退出码 0 需要，1 不需要），不调用模型。
"""
from __future__ import annotations
import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get('AUTOTUNE_ROOT', '/data/workspace/jzheng/luna-prompt-autotune'))
STATE, ASSETS, RUNS, HUMAN = ROOT / 'state', ROOT / 'assets', ROOT / 'runs', ROOT / 'human'
sys.path.insert(0, str(ROOT / 'src'))
import metrics  # noqa: E402
import promptlib  # noqa: E402
import samples  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=8))
WORKER_UNIT = 'luna-autotune-worker'
TIMERS = ['luna-autotune-runner.timer', 'luna-autotune-brain.timer']
URL = 'http://127.0.0.1:18397/review'
CONCURRENCY = 6
IDLE_TICKS_TO_DISABLE = 36  # 连续 3 小时无事可做且非每日限额等待 → 自动停用定时器


def now(): return dt.datetime.now(TZ)
def rj(p, d=None): return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d
def wj(p, v):
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding='utf-8'); t.replace(p)
def log(msg): print(f'{now():%m-%d %H:%M:%S} {msg}', flush=True)
def systemctl(*a): return subprocess.run(['systemctl', '--user', *a], capture_output=True, text=True, timeout=30)


def stop_reason():
    if (STATE / 'STOP').exists(): return 'STOP 文件：' + (STATE / 'STOP').read_text(encoding='utf-8').strip()[:200]
    deadline = rj(STATE / 'config/policy.json', {}).get('deadline')
    if deadline and now().date().isoformat() > deadline: return f'已过截止日期 {deadline}'
    return None


def disable_all(reason):
    if not (STATE / 'STOP').exists(): (STATE / 'STOP').write_text(reason + '\n', encoding='utf-8')
    systemctl('stop', WORKER_UNIT)
    for t in TIMERS: systemctl('disable', '--now', t)
    log('已停用定时器与实验 worker：' + reason)


def health():
    mem = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    frac = mem['MemAvailable'] / mem['MemTotal']
    state = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}',
                            'yoloe_service_test'], capture_output=True, text=True, timeout=5).stdout.strip()
    return frac >= 0.15 and state == 'running healthy', {'available_fraction': round(frac, 4), 'shared_state': state}


def rounds():
    return {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted((STATE / 'rounds').glob('*.json'))}


def ensure_worker():
    if systemctl('is-active', WORKER_UNIT).stdout.strip() == 'active': return
    systemctl('reset-failed', WORKER_UNIT)
    r = subprocess.run(['systemd-run', '--user', f'--unit={WORKER_UNIT}', f'-p', f'EnvironmentFile={ROOT}/private/worker.env',
                        '-p', 'MemoryMax=4G', '-p', 'MemorySwapMax=0', '-p', 'CPUQuota=300%', '-p', 'Restart=no', '-p', 'KillMode=control-group',
                        '/usr/bin/python3', str(ROOT / 'src/luna_experiment_worker.py')], capture_output=True, text=True, timeout=30)
    if r.returncode: raise RuntimeError('worker 启动失败：' + r.stderr[-300:])
    for _ in range(20):
        try:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(URL.replace('/review', '/health'), timeout=3).read(); return
        except Exception: time.sleep(1)
    raise RuntimeError('worker 未就绪')


def token():
    for line in (ROOT / 'private/worker.env').read_text().splitlines():
        k, _, v = line.partition('=')
        if k == 'CODEX_REVIEW_WORKER_TOKEN': return v.strip()
    raise RuntimeError('缺少 worker token')


def run_units(version, doc, units):
    out = RUNS / version / 'out'; out.mkdir(parents=True, exist_ok=True)
    tok, stop_flag = token(), {'stop': False}

    def one(u):
        target = out / (u['unit'] + '.json')
        if target.exists() or stop_flag['stop']: return
        cands = [{'target_index': i, 'class_name': c['class_name'], 'bbox': c['bbox']} for i, c in enumerate(u['candidates'])]
        payload = json.dumps({'image': base64.b64encode((ASSETS / 'images' / (u['unit'] + '.jpg')).read_bytes()).decode(),
                              'candidates': cands, 'prompt': promptlib.build(doc, cands)}).encode()
        attempts, results, t0 = [], None, time.monotonic()
        for attempt in range(2):
            ok, h = health()
            if not ok: stop_flag['stop'] = True; return
            t = time.monotonic()
            try:
                req = urllib.request.Request(URL, data=payload, headers={'Authorization': 'Bearer ' + tok, 'Content-Type': 'application/json'})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=240) as r: body = json.load(r)
                items = {x['target_index']: x for x in body['results']}
                if set(items) != set(range(len(cands))): raise ValueError('mapping')
                results = [{'key': c['key'], 'raw': items[i]['decision'], 'reasoning': items[i].get('reasoning'),
                            'observed': items[i].get('observed_object'), 'limitation': items[i].get('image_limitation')}
                           for i, c in enumerate(u['candidates'])]
                status = 'ok'
            except urllib.error.HTTPError as e: status = f'http_{e.code}'
            except Exception as e: status = type(e).__name__
            attempts.append({'status': status, 'elapsed': round(time.monotonic() - t, 2)})
            if results is not None: break
            time.sleep(60 if status == 'http_429' else 20)
        wj(target, {'unit': u['unit'], 'success': results is not None, 'results': results or [],
                    'elapsed': round(time.monotonic() - t0, 2), 'attempts': attempts})

    with concurrent.futures.ThreadPoolExecutor(CONCURRENCY) as pool: list(pool.map(one, units))
    if stop_flag['stop']: raise RuntimeError('资源或共享服务健康检查未通过，本轮中断（已完成的图片保留，下次续跑）')


def collect(version):
    res, detail = {}, {}
    for f in (RUNS / version / 'out').glob('*.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        for x in r['results']: res[x['key']] = x['raw']; detail[x['key']] = x
    return res, detail


def execute(version):
    rd = rj(STATE / 'rounds' / f'{version}.json'); doc = rj(STATE / 'prompts' / f'{version}.json')
    items, labels = rj(ASSETS / 'items.json'), rj(ASSETS / 'labels.json')
    rd.update(status='running', started_at=rd.get('started_at') or now().isoformat(timespec='seconds')); wj(STATE / 'rounds' / f'{version}.json', rd)
    ensure_worker()
    log(f'{version} 开始：dev + 无标签池')
    run_units(version, doc, [u for u in items if u['split'] in ('dev', 'unlabeled')])
    res, detail = collect(version)
    files = list((RUNS / version / 'out').glob('*.json'))
    failed_units = sum(not json.loads(f.read_text())['success'] for f in files)
    m = {'version': version, 'dev': metrics.score(res, labels, 'dev'), 'unlabeled': metrics.unlabeled(res, items),
         'failed_units': failed_units, 'units': len(files),
         'elapsed_median': sorted(json.loads(f.read_text())['elapsed'] for f in files)[len(files) // 2]}
    base = rj(STATE / 'metrics/p000.json')
    if version != 'p000' and base:
        m['vs_baseline'] = metrics.gates(m, base)
        parent = rj(STATE / 'metrics' / f"{doc.get('parent')}.json")
        if parent: m['parent'] = doc['parent']; m['parent_dev_accuracy'] = parent['dev']['overall']['accuracy']
    if version == 'p000b':
        m['noise'] = metrics.flips(collect('p000')[0], res)
    best = rj(STATE / 'best.json', {'version': 'p000', 'dev_accuracy': base['dev']['overall']['accuracy'] if base else 0})
    m['is_new_best'] = version not in ('p000', 'p000b') and bool(m.get('vs_baseline', {}).get('passed')) and m['dev']['overall']['accuracy'] > best['dev_accuracy'] and failed_units <= 0.05 * len(files)
    wj(STATE / 'metrics' / f'{version}.json', m)
    samples.build(version, items, labels, detail, STATE / 'samples' / version)
    if version == 'p000' or m['is_new_best']:
        log(f'{version} 跑锁定 holdout')
        run_units(version, doc, [u for u in items if u['split'] == 'holdout'])
        hres, _ = collect(version)
        wj(HUMAN / f'holdout-{version}.json', {'version': version, 'holdout': metrics.score(hres, labels, 'holdout')})
        if m['is_new_best']: wj(STATE / 'best.json', {'version': version, 'dev_accuracy': m['dev']['overall']['accuracy'], 'at': now().isoformat(timespec='seconds')})
    rd.update(status='done', finished_at=now().isoformat(timespec='seconds')); wj(STATE / 'rounds' / f'{version}.json', rd)
    log(f"{version} 完成：dev 准确率 {m['dev']['overall']['accuracy']}，无标签 uncertain {m['unlabeled']['overall']['uncertain_rate']}，新最好 {m['is_new_best']}")


def brain_needed():
    if stop_reason(): return False
    rs = rounds()
    if any(r['status'] == 'done' and not r.get('reported') and k not in ('p000', 'p000b') for k, r in rs.items()): return True
    if any(r['status'] in ('queued', 'running') for r in rs.values()): return False
    ledger = [json.loads(l) for l in (STATE / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()] if (STATE / 'ledger.jsonl').exists() else []
    if any(x.get('converged') for x in ledger): return False
    policy = rj(STATE / 'config/policy.json', {})
    today = now().date().isoformat()
    return sum(r.get('submitted_at', '').startswith(today) for r in rs.values()) < policy.get('daily_rounds', 8)


def tick():
    reason = stop_reason()
    if reason: disable_all(reason); return
    ok, h = health()
    if not ok: log(f'跳过：资源或共享服务不健康 {h}'); return
    queued = sorted((k, r) for k, r in rounds().items() if r['status'] in ('running', 'queued'))
    idle = STATE / 'idle_ticks'
    if not queued:
        systemctl('stop', WORKER_UNIT)
        n = int(idle.read_text()) + 1 if idle.exists() else 1
        idle.write_text(str(n))
        if not brain_needed() and n >= IDLE_TICKS_TO_DISABLE:
            ledger = (STATE / 'ledger.jsonl').read_text(encoding='utf-8') if (STATE / 'ledger.jsonl').exists() else ''
            if '"converged": true' in ledger: disable_all('Hermes 判定收敛，连续空闲后自动停用')
        return
    idle.write_text('0')
    version = queued[0][0]
    try:
        execute(version)
    except Exception as e:
        log(f'{version} 未完成：{type(e).__name__}: {str(e)[:300]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--brain-needed', action='store_true'); a = ap.parse_args()
    if a.brain_needed: sys.exit(0 if brain_needed() else 1)
    tick()
