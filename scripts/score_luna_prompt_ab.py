"""Luna 提示词 A/B 新样本打分（本地，不调用模型）。

线上规则：只有原始 true 判真，false/uncertain 判假；另给 uncertain 转人工口径。
执行失败单列，不计对错。
"""
from __future__ import annotations
import collections
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'
ROOT = DATA / 'luna-prompt-v2-eval-20261009'
ARMS = ['baseline', 'v2']
GROUP = {'人员': '人员', '人': '人员', '违建': '违建', '光斑': '其他类别', '光伏板': '其他类别', '没戴头盔': '其他类别'}


def group_of(cls):
    return GROUP.get(cls, '车辆')


def result_files(arm):
    files = sorted((ROOT / ('out-' + arm)).glob('*.json'))
    if arm == 'baseline':  # 人员回归图片复用 v2 当天共享 worker（旧提示词）的结果
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        files += [DATA / u['reuse_baseline_from'] for u in manifest.values() if u.get('reuse_baseline_from')]
    return files


def load(arm):
    out = {}
    for f in result_files(arm):
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
    global ROOT, ARMS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='luna-prompt-v2-eval-20261009')
    ap.add_argument('--arms', default='baseline,v2')
    a = ap.parse_args()
    ROOT, ARMS = DATA / a.dir, a.arms.split(',')
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
            okv = (preds[ARMS[1]][k]['rawDecision'] == 'true') == (labels[k]['label'] == 'true')
            name = ('对' if okb else '错') + '→' + ('对' if okv else '错')
            t[name] += 1
            if okb != okv and len(ex[name]) < 15:
                ex[name].append({'key': k, 'gold': labels[k]['label'], 'class': labels[k]['class_name'],
                                 'baseline': preds['baseline'][k]['rawDecision'], ARMS[1]: preds[ARMS[1]][k]['rawDecision'],
                                 'new_reasoning': preds[ARMS[1]][k]['reasoning']})
        report['transitions'][g] = dict(t); report['examples'][g] = dict(ex)
    elapsed = {}
    for a in ARMS:
        vals = sorted(json.loads(f.read_text())['elapsed'] for f in (ROOT / ('out-' + a)).glob('*.json'))
        calls = [len(json.loads(f.read_text())['attempts']) for f in (ROOT / ('out-' + a)).glob('*.json')]  # 本轮实际调用，不含复用
        elapsed[a] = {'units': len(vals), 'median': round(vals[len(vals) // 2], 2) if vals else None,
                      'p90': round(vals[max(0, -(-9 * len(vals) // 10) - 1)], 2) if vals else None,
                      'calls': sum(calls), 'retried_units': sum(c > 1 for c in calls)}
    report['elapsed'] = elapsed
    (ROOT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for g in sorted(g for g in report['groups'] if '/' not in g):
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
