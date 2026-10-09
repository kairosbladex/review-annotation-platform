"""打分与合格标准（runner 调用；Hermes 只能读结果，不能修改）。

线上规则：只有原始 true 判真，false/uncertain 判假。执行失败单列，不计对错。
合格标准（相对基线 p000，dev 集）：
  G1 总体准确率高于基线；
  G2 金标 ≥10 框的类别，准确率下降不超过 3 个百分点；
  G3 人员“非人被放过”比基线多不超过 3 个（v2 教训：放宽人员规则会大量误放）；
  G4 无标签池总体 uncertain 比例不高于基线。
"""
from __future__ import annotations
import collections

MIN_CLASS_N = 10


def score(results: dict, labels: dict, split: str) -> dict:
    by_class = collections.defaultdict(collections.Counter)
    for key, lab in labels.items():
        if lab['split'] != split: continue
        c = by_class[lab['class_name']]
        raw = results.get(key)
        if raw is None:
            c['failed'] += 1; continue
        c['n'] += 1; c['gold_' + lab['label']] += 1; c[raw] += 1
        ok = (raw == 'true') == (lab['label'] == 'true')
        c['right'] += ok
        if not ok: c['true_rejected' if lab['label'] == 'true' else 'false_released'] += 1
    out = {}
    total = collections.Counter()
    for cls, c in by_class.items():
        total.update(c)
        out[cls] = _finish(c)
    return {'overall': _finish(total), 'by_class': dict(sorted(out.items(), key=lambda x: -x[1]['n']))}


def _finish(c: collections.Counter) -> dict:
    n = c['n']
    return {'n': n, 'gold_true': c['gold_true'], 'gold_false': c['gold_false'], 'right': c['right'],
            'accuracy': round(c['right'] / n, 4) if n else None, 'true_rejected': c['true_rejected'],
            'false_released': c['false_released'], 'uncertain': c['uncertain'],
            'uncertain_rate': round(c['uncertain'] / n, 4) if n else None, 'failed': c['failed']}


def unlabeled(results: dict, items: list) -> dict:
    by_class = collections.defaultdict(collections.Counter)
    for it in items:
        if it['split'] != 'unlabeled': continue
        for c in it['candidates']:
            raw = results.get(c['key'])
            by_class[c['class_name']]['failed' if raw is None else raw] += 1
    out, total = {}, collections.Counter()
    for cls, c in by_class.items():
        total.update(c)
        done = c['true'] + c['false'] + c['uncertain']
        out[cls] = dict(c, n=done, uncertain_rate=round(c['uncertain'] / done, 4) if done else None)
    done = total['true'] + total['false'] + total['uncertain']
    return {'overall': dict(total, n=done, uncertain_rate=round(total['uncertain'] / done, 4) if done else None),
            'by_class': dict(sorted(out.items(), key=lambda x: -x[1]['n']))}


def gates(cand: dict, base: dict) -> dict:
    cd, bd = cand['dev'], base['dev']
    g = {'G1_总体准确率高于基线': (cd['overall']['accuracy'] or 0) > (bd['overall']['accuracy'] or 0)}
    worse = []
    for cls, b in bd['by_class'].items():
        c = cd['by_class'].get(cls)
        if b['n'] >= MIN_CLASS_N and c and c['accuracy'] is not None and c['accuracy'] < b['accuracy'] - 0.03:
            worse.append(f"{cls} {b['accuracy']:.1%}→{c['accuracy']:.1%}")
    g['G2_各类准确率下降不超过3个百分点'] = not worse
    bp, cp = bd['by_class'].get('人员', {}), cd['by_class'].get('人员', {})
    g['G3_人员非人放过不比基线多3个以上'] = cp.get('false_released', 0) <= bp.get('false_released', 0) + 3
    g['G4_无标签uncertain比例不高于基线'] = (cand['unlabeled']['overall']['uncertain_rate'] or 0) <= (base['unlabeled']['overall']['uncertain_rate'] or 0)
    return {'passed': all(g.values()), 'gates': g, 'regressed_classes': worse}


def flips(a: dict, b: dict) -> dict:
    """同一组框两次结果的翻转数（用于估计随机波动）。"""
    keys = set(a) & set(b)
    changed = sum((a[k] == 'true') != (b[k] == 'true') for k in keys)
    return {'compared': len(keys), 'decision_flips': changed, 'flip_rate': round(changed / len(keys), 4) if keys else None}
