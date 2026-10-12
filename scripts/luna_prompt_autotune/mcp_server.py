"""Luna 提示词自动迭代：供 Hermes 调用的受控 MCP 工具。

Hermes 只能读 dev 集与无标签池的结果、看错判拼图、读写提示词版本、写报告和台账。
锁定 holdout 的结果不在 state 目录，Hermes 看不到；评测、打分与合格标准由宿主机 runner 执行。
"""
from __future__ import annotations
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

STATE = Path(os.environ.get('AUTOTUNE_STATE', '/work/state'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import promptlib  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=8))
mcp = FastMCP('luna_autotune')
VER = re.compile(r'p\d{3}b?')


def _now(): return dt.datetime.now(TZ)
def _rj(p, d=None): return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d
def _wj(p, v): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding='utf-8')
def _append(p, row):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')
def _audit(tool, **kw): _append(STATE / 'audit.jsonl', {'at': _now().isoformat(timespec='seconds'), 'tool': tool, **kw})
def _rounds(): return {p.stem: _rj(p) for p in sorted((STATE / 'rounds').glob('*.json'))}
def _ledger(): return [json.loads(l) for l in (STATE / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()] if (STATE / 'ledger.jsonl').exists() else []


def _check(version):
    if not VER.fullmatch(version or ''): raise ValueError('version 格式应为 pNNN')
    if not (STATE / 'prompts' / f'{version}.json').exists(): raise ValueError(f'没有版本 {version}')


def _stop():
    if (STATE / 'STOP').exists(): return 'STOP 文件：' + (STATE / 'STOP').read_text(encoding='utf-8').strip()[:200]
    dl = _rj(STATE / 'config/policy.json', {}).get('deadline')
    if dl and _now().date().isoformat() > dl: return f'已过截止日期 {dl}'
    return None


@mcp.tool()
def get_state() -> dict:
    """每次唤醒第一步调用。返回停止状态、各版本进度、待写报告的版本、今日已提交数与上限、当前最好版本、随机波动参考。"""
    rs = _rounds(); policy = _rj(STATE / 'config/policy.json', {})
    today = _now().date().isoformat()
    ledger = _ledger()
    no_improve = 0
    for row in reversed(ledger):
        if row.get('is_new_best'): break
        no_improve += 1
    noise = (_rj(STATE / 'metrics/p000b.json', {}) or {}).get('noise')
    _audit('get_state')
    return {'stop_requested': _stop() is not None, 'stop_reason': _stop(),
            'rounds': {k: {'status': r['status'], 'parent': r.get('parent'), 'reported': r.get('reported', False)} for k, r in rs.items()},
            'unreported': [k for k, r in rs.items() if r['status'] == 'done' and not r.get('reported') and k not in ('p000', 'p000b')],
            'queued_or_running': [k for k, r in rs.items() if r['status'] in ('queued', 'running')],
            'submitted_today': sum(r.get('submitted_at', '').startswith(today) for r in rs.values()),
            'daily_limit': policy.get('daily_rounds', 8), 'best': _rj(STATE / 'best.json', {'version': 'p000'}),
            'rounds_since_new_best': no_improve, 'converged': any(x.get('converged') for x in ledger),
            'baseline_repeat_noise': noise}


@mcp.tool()
def get_round_result(version: str) -> dict:
    """读取某版本在 dev 集（有人工标签）和无标签池上的结果，以及与基线 p000 的合格标准对比。数字只引用这里返回的。"""
    _check(version)
    m = _rj(STATE / 'metrics' / f'{version}.json')
    if m is None: return {'available': False, 'reason': '该版本尚未跑完'}
    base = _rj(STATE / 'metrics/p000.json')
    _audit('get_round_result', version=version)
    return {'available': True, 'version': version, 'dev': m['dev'], 'unlabeled': m['unlabeled'],
            'baseline_p000': {'dev_overall': base['dev']['overall'], 'dev_by_class': base['dev']['by_class'],
                              'unlabeled_by_class': base['unlabeled']['by_class'], 'unlabeled_overall': base['unlabeled']['overall']} if base else None,
            'vs_baseline': m.get('vs_baseline'), 'parent': m.get('parent'), 'parent_dev_accuracy': m.get('parent_dev_accuracy'),
            'is_new_best': m.get('is_new_best'), 'failed_units': m.get('failed_units'), 'units': m.get('units'),
            'evaluation_status': m.get('evaluation_status'), 'promotion_errors': m.get('promotion_errors'),
            'paired_baselines': m.get('paired_baselines')}


@mcp.tool()
def list_error_samples(version: str) -> dict:
    """列出某版本可查看的错判拼图（类别-错误类型及数量）。kind：true_rejected=真目标被驳回，false_released=非目标被放过，uncertain=无标签池判看不清。"""
    _check(version)
    return {'available': (STATE / 'samples' / version / 'index.json').exists(), 'samples': _rj(STATE / 'samples' / version / 'index.json', {})}


@mcp.tool()
def view_error_samples(version: str, class_name: str, kind: str) -> dict:
    """取某类别某种错误的拼图路径（用 vision_analyze 查看）和每个样本的人工标签、模型理由。图中“人工”标签已由人确认，以它为准。"""
    _check(version)
    if kind not in ('true_rejected', 'false_released', 'uncertain'): raise ValueError('kind 只能是 true_rejected / false_released / uncertain')
    name = f'{class_name}-{kind}'
    img = STATE / 'samples' / version / f'{name}.jpg'
    if not img.exists(): return {'available': False, 'reason': '没有这类样本'}
    _audit('view_error_samples', version=version, name=name)
    return {'available': True, 'image_path': str(img), 'items': _rj(STATE / 'samples' / version / f'{name}.json')}


@mcp.tool()
def get_prompt(version: str) -> dict:
    """读取某版本提示词：common（通用部分）和 cards（各类别定义卡）。可用的类别卡名称见 known_classes。"""
    _check(version)
    return {'prompt': _rj(STATE / 'prompts' / f'{version}.json'), 'known_classes': promptlib.KNOWN_CLASSES,
            'limits': {'common': promptlib.MAX_COMMON, 'card': promptlib.MAX_CARD, 'total': promptlib.MAX_TOTAL}}


@mcp.tool()
def submit_prompt(parent: str, hypothesis: str, cards: dict, common: str = '') -> dict:
    """在 parent 版本基础上提交新提示词版本。cards 只写要修改的类别卡（类别名→新的完整卡片文字），未写的沿用 parent；
    common 留空表示沿用 parent。hypothesis 写明改了什么、预期哪个类别的哪个数字怎么变。提交后由执行器排队评测。"""
    _check(parent)
    if _stop(): return {'accepted': False, 'reason': _stop()}
    rs = _rounds(); policy = _rj(STATE / 'config/policy.json', {})
    if any(r['status'] in ('queued', 'running') for r in rs.values()): return {'accepted': False, 'reason': '已有版本在排队或运行，等它跑完'}
    today = _now().date().isoformat()
    if sum(r.get('submitted_at', '').startswith(today) for r in rs.values()) >= policy.get('daily_rounds', 8):
        return {'accepted': False, 'reason': '今日提交已达上限'}
    if any(x.get('converged') for x in _ledger()): return {'accepted': False, 'reason': '台账已标记收敛，不再提交'}
    if not isinstance(cards, dict) or not hypothesis.strip(): return {'accepted': False, 'reason': 'cards 须为对象，hypothesis 不能为空'}
    base = _rj(STATE / 'prompts' / f'{parent}.json')
    doc = {'common': common.strip() or base['common'], 'cards': dict(base.get('cards') or {}, **cards)}
    errors = promptlib.validate(doc)
    if errors: return {'accepted': False, 'reason': '；'.join(errors)}
    for k, other in rs.items():
        prev = _rj(STATE / 'prompts' / f'{k}.json')
        if prev and prev['common'] == doc['common'] and (prev.get('cards') or {}) == doc['cards']:
            return {'accepted': False, 'reason': f'与已有版本 {k} 完全相同'}
    n = max(int(k[1:4]) for k in rs) + 1
    version = f'p{n:03d}'
    changed = sorted(k for k in doc['cards'] if doc['cards'][k] != (base.get('cards') or {}).get(k)) + (['common'] if doc['common'] != base['common'] else [])
    _wj(STATE / 'prompts' / f'{version}.json', dict(doc, version=version, parent=parent, author='hermes', hypothesis=hypothesis.strip(), changed=changed))
    _wj(STATE / 'rounds' / f'{version}.json', {'status': 'queued', 'parent': parent, 'submitted_at': _now().isoformat(timespec='seconds')})
    _audit('submit_prompt', version=version, parent=parent, changed=changed)
    return {'accepted': True, 'version': version, 'changed': changed}


@mcp.tool()
def write_report(version: str, report: str) -> dict:
    """写某版本的评测报告（中文，先结论后数字）。写完后该版本标记为已报告。"""
    _check(version)
    (STATE / 'reports').mkdir(exist_ok=True)
    (STATE / 'reports' / f'{version}.md').write_text(report, encoding='utf-8')
    p = STATE / 'rounds' / f'{version}.json'; r = _rj(p)
    if r: r['reported'] = True; _wj(p, r)
    _audit('write_report', version=version, chars=len(report))
    return {'ok': True}


@mcp.tool()
def update_ledger(version: str, passed: bool, key_numbers: str, next_hypothesis: str, converged: bool = False) -> dict:
    """写台账一行。passed 抄 get_round_result 的 vs_baseline.passed；converged=true 表示判断已收敛、此后不再提交新版本。"""
    _check(version)
    m = _rj(STATE / 'metrics' / f'{version}.json', {})
    _append(STATE / 'ledger.jsonl', {'at': _now().isoformat(timespec='seconds'), 'version': version, 'passed': passed,
                                     'is_new_best': bool(m.get('is_new_best')), 'key_numbers': key_numbers,
                                     'next_hypothesis': next_hypothesis, 'converged': converged})
    return {'ok': True}


@mcp.tool()
def propose_playbook_change(text: str) -> dict:
    """对“迭代原则”提出修改建议（人工审核后才生效；合格标准和禁止事项不可修改）。"""
    _append(STATE / 'proposals.jsonl', {'at': _now().isoformat(timespec='seconds'), 'text': text})
    return {'ok': True}


if __name__ == '__main__':
    mcp.run()
