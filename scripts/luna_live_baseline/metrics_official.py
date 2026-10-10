"""打分与合格标准（runner 调用；Hermes 只能读结果，不能修改）。

线上规则：只有原始 true 判真，false/uncertain 判假。执行失败单列，不计对错。
合格标准（相对基线 p000 与重复运行 p000b，dev 集）：
  G1 总体准确率高于两次基线中较高者；
  G2 金标 ≥10 框的类别，准确率不低于两次基线较低者 3 个百分点以上；
  G3 人员“非人被放过”不比两次基线较多者多 3 个以上（v2 教训：放宽人员规则会大量误放）；
  G4 无标签池总体 uncertain 比例不高于两次基线平均。
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


def gates(cand: dict, bases: list) -> dict:
    """bases 为基线多次运行（p000、p000b）。同一提示词两次运行约 15% 结论翻转，标准按两次基线的范围设定，避免把随机波动当改进。"""
    cd = cand['dev']
    accs = [b['dev']['overall']['accuracy'] or 0 for b in bases]
    g = {'G1_总体准确率高于两次基线中较高者': (cd['overall']['accuracy'] or 0) > max(accs)}
    worse = []
    for cls, b0 in bases[0]['dev']['by_class'].items():
        c = cd['by_class'].get(cls)
        lows = [b['dev']['by_class'].get(cls, {}).get('accuracy') for b in bases]
        lows = [x for x in lows if x is not None]
        if b0['n'] >= MIN_CLASS_N and c and c['accuracy'] is not None and lows and c['accuracy'] < min(lows) - 0.03:
            worse.append(f"{cls} 基线{min(lows):.1%}~{max(lows):.1%}→{c['accuracy']:.1%}")
    g['G2_各类准确率不低于两次基线较低者3个百分点以上'] = not worse
    fr = max(b['dev']['by_class'].get('人员', {}).get('false_released', 0) for b in bases)
    g['G3_人员非人放过不比基线多3个以上'] = cd['by_class'].get('人员', {}).get('false_released', 0) <= fr + 3
    unc = sum(b['unlabeled']['overall']['uncertain_rate'] or 0 for b in bases) / len(bases)
    g['G4_无标签uncertain比例不高于两次基线平均'] = (cand['unlabeled']['overall']['uncertain_rate'] or 0) <= unc
    return {'passed': all(g.values()), 'gates': g, 'regressed_classes': worse}


def flips(a: dict, b: dict) -> dict:
    """同一组框两次结果的翻转数（用于估计随机波动）。"""
    keys = set(a) & set(b)
    changed = sum((a[k] == 'true') != (b[k] == 'true') for k in keys)
    return {'compared': len(keys), 'decision_flips': changed, 'flip_rate': round(changed / len(keys), 4) if keys else None}
