"""Luna 提示词 A/B 新样本打分（本地，不调用模型）。

线上规则：只有原始 true 判真，false/uncertain 判假；另给 uncertain 转人工口径。
执行失败单列，不计对错。
"""
from __future__ import annotations
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data/luna-prompt-v2-eval-20261009'
ARMS = ['baseline', 'v2']
GROUP = {'人员': '人员', '人': '人员', '违建': '违建'}


def group_of(cls):
    return GROUP.get(cls, '车辆')


def load(arm):
    out = {}
    for f in sorted((ROOT / ('out-' + arm)).glob('*.json')):
        r = json.loads(f.read_text())
        for k, res in zip(r['keys'], r['results'] if r['success'] else [None] * len(r['keys'])):
            out[k] = res
    return out


def score(labels, preds, keys):
    c = collections.Counter()
    for k in keys:
        p = preds.get(k)
        if p is None:
            c['failed'] += 1; continue
        gold, raw = labels[k]['label'], p['rawDecision']
        c['n'] += 1; c['uncertain'] += raw == 'uncertain'
        ok = (raw == 'true') == (gold == 'true')
        c['right' if ok else 'wrong'] += 1
        if not ok: c['true_rejected' if gold == 'true' else 'false_released'] += 1
        if raw == 'uncertain': c['manual'] += 1
        else: c['auto_right' if ok else 'auto_wrong'] += 1
        c['gold_' + gold] += 1
    c['accuracy'] = round(c['right'] / c['n'], 4) if c['n'] else None
    auto = c['auto_right'] + c['auto_wrong']
    c['auto_accuracy'] = round(c['auto_right'] / auto, 4) if auto else None
    return dict(c)


def main():
    labels = json.loads((ROOT / 'labels.json').read_text())
    preds = {a: load(a) for a in ARMS}
    groups = collections.defaultdict(list)
    for k, v in labels.items():
        g = group_of(v['class_name'])
        groups[g].append(k); groups[f"{g}/{v['source']}"].append(k)
    report = {'groups': {}, 'transitions': {}, 'examples': {}}
    for g, keys in sorted(groups.items()):
        both = [k for k in keys if all(preds[a].get(k) for a in ARMS)]
        report['groups'][g] = {a: score(labels, preds[a], keys) for a in ARMS}
        report['groups'][g]['paired_n'] = len(both)
        if '/' in g: continue
        t = collections.Counter()
        ex = collections.defaultdict(list)
        for k in both:
            okb = (preds['baseline'][k]['rawDecision'] == 'true') == (labels[k]['label'] == 'true')
            okv = (preds['v2'][k]['rawDecision'] == 'true') == (labels[k]['label'] == 'true')
            name = ('对' if okb else '错') + '→' + ('对' if okv else '错')
            t[name] += 1
            if okb != okv and len(ex[name]) < 15:
                ex[name].append({'key': k, 'gold': labels[k]['label'], 'class': labels[k]['class_name'],
                                 'baseline': preds['baseline'][k]['rawDecision'], 'v2': preds['v2'][k]['rawDecision'],
                                 'v2_reasoning': preds['v2'][k]['reasoning']})
        report['transitions'][g] = dict(t); report['examples'][g] = dict(ex)
    elapsed = {}
    for a in ARMS:
        vals = sorted(json.loads(f.read_text())['elapsed'] for f in (ROOT / ('out-' + a)).glob('*.json'))
        calls = [len(json.loads(f.read_text())['attempts']) for f in (ROOT / ('out-' + a)).glob('*.json')]
        elapsed[a] = {'units': len(vals), 'median': round(vals[len(vals) // 2], 2) if vals else None,
                      'p90': round(vals[max(0, -(-9 * len(vals) // 10) - 1)], 2) if vals else None,
                      'calls': sum(calls), 'retried_units': sum(c > 1 for c in calls)}
    report['elapsed'] = elapsed
    (ROOT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for g in ['人员', '车辆', '违建']:
        if g not in report['groups']: continue
        print(g, report['transitions'].get(g))
        for a in ARMS:
            s = report['groups'][g][a]
            print(f"  {a:8s} n={s.get('n')} 对={s.get('right',0)} 错={s.get('wrong',0)} 准确率={s.get('accuracy')} 真被驳={s.get('true_rejected',0)} 假被放={s.get('false_released',0)} uncertain={s.get('uncertain',0)} 失败={s.get('failed',0)} 转人工口径自动准确率={s.get('auto_accuracy')}")
    for g in sorted(report['groups']):
        if '/' in g:
            print(g, {a: (report['groups'][g][a].get('right', 0), report['groups'][g][a].get('wrong', 0), report['groups'][g][a].get('true_rejected', 0), report['groups'][g][a].get('false_released', 0)) for a in ARMS})
    print(json.dumps(elapsed, ensure_ascii=False))


if __name__ == '__main__':
    main()
