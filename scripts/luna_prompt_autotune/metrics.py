"""打分与合格标准（runner 调用；Hermes 只能读结果，不能修改）。

线上规则：只有原始 true 判真，false/uncertain 判假。执行失败单列，不计对错。
保留 G1—G4；G0 要求两次基线与候选使用相同冻结清单且完整计分；
G5 要求人员、道路裂纹有正负例，误拒/误放均不超过两次基线的较差值。
"""
from __future__ import annotations
import collections
import hashlib
import json

import promptlib

MIN_CLASS_N = 10
DECISIONS = ('true', 'false', 'uncertain')
CRITICAL_CLASSES = ('人员', '道路裂纹')
GATE_VERSION = 3


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def evaluation_identity(items: list, labels: dict, *, splits=('dev', 'unlabeled')) -> str:
    """默认只冻结开发/无标签池；锁定集由 runner 单独计算身份，不参与开发集比较。

    包含清单中的图片哈希、候选顺序与坐标；不代表已核验实际图片字节或严格同帧。
    """
    selected = [it for it in items if it['split'] in splits]
    units = [it['unit'] for it in selected]
    keys = [c['key'] for it in selected for c in it['candidates']]
    if len(units) != len(set(units)) or len(keys) != len(set(keys)):
        raise ValueError('冻结清单中 unit/key 重复，不能唯一对应结果')
    return _sha({'items': sorted(selected, key=lambda it: it['unit']),
                 'labels': {k: v for k, v in labels.items() if v['split'] in splits}})


def _coverage(results: dict, definitions: dict) -> dict:
    missing = sorted(k for k in definitions if results.get(k) is None)
    invalid = sorted(k for k in definitions if results.get(k) is not None and results[k] not in DECISIONS)
    return {'expected': len(definitions), 'completed': len(definitions) - len(missing) - len(invalid),
            'keys_sha256': _sha(sorted(definitions)), 'definition_sha256': _sha(definitions),
            'missing_keys': missing, 'invalid_keys': invalid, 'complete': not missing and not invalid}


def score(results: dict, labels: dict, split: str) -> dict:
    by_class = collections.defaultdict(collections.Counter)
    definitions = {}
    for key, lab in labels.items():
        if lab['split'] != split: continue
        if lab['label'] not in ('true', 'false'): raise ValueError(f'{key} 金标必须为 true/false')
        definitions[key] = {field: lab[field] for field in ('class_name', 'label', 'split')}
        c = by_class[lab['class_name']]
        c['expected_n'] += 1; c['expected_gold_' + lab['label']] += 1
        raw = results.get(key)
        if raw not in DECISIONS:
            c['failed'] += 1; c['invalid'] += raw is not None; continue
        c['n'] += 1; c['gold_' + lab['label']] += 1; c[raw] += 1
        c[lab['label'] + '_' + raw] += 1
        ok = (raw == 'true') == (lab['label'] == 'true')
        c['right'] += ok
        if not ok: c['true_rejected' if lab['label'] == 'true' else 'false_released'] += 1
    out = {}
    total = collections.Counter()
    for cls, c in by_class.items():
        total.update(c)
        out[cls] = _finish(c)
    return {'overall': _finish(total), 'by_class': dict(sorted(out.items(), key=lambda x: -x[1]['n'])),
            'coverage': _coverage(results, definitions)}


def _finish(c: collections.Counter) -> dict:
    n, expected = c['n'], c['expected_n']
    return {'n': n, 'gold_true': c['gold_true'], 'gold_false': c['gold_false'], 'right': c['right'],
            'accuracy': round(c['right'] / n, 4) if n else None, 'true_rejected': c['true_rejected'],
            'false_released': c['false_released'], 'uncertain': c['uncertain'],
            'uncertain_rate': round(c['uncertain'] / n, 4) if n else None, 'failed': c['failed'],
            'expected_n': expected, 'expected_gold_true': c['expected_gold_true'], 'expected_gold_false': c['expected_gold_false'],
            'coverage_rate': round(n / expected, 4) if expected else None, 'invalid': c['invalid'],
            'true_retained': c['true_true'],
            'true_retention_rate': round(c['true_true'] / c['gold_true'], 4) if c['gold_true'] else None,
            'false_release_rate': round(c['false_released'] / c['gold_false'], 4) if c['gold_false'] else None,
            'confusion': {gold: {raw: c[gold + '_' + raw] for raw in DECISIONS} for gold in ('true', 'false')}}


def unlabeled(results: dict, items: list) -> dict:
    by_class = collections.defaultdict(collections.Counter)
    definitions = {}
    for it in items:
        if it['split'] != 'unlabeled': continue
        for c in it['candidates']:
            if c['key'] in definitions: raise ValueError(f"无标签池重复 key：{c['key']}")
            definitions[c['key']] = c
            raw = results.get(c['key'])
            counts = by_class[c['class_name']]
            counts['expected_n'] += 1
            counts[raw if raw in DECISIONS else 'failed'] += 1
            counts['invalid'] += raw is not None and raw not in DECISIONS
    out, total = {}, collections.Counter()
    for cls, c in by_class.items():
        total.update(c)
        out[cls] = _unlabeled_finish(c)
    return {'overall': _unlabeled_finish(total), 'by_class': dict(sorted(out.items(), key=lambda x: -x[1]['n'])),
            'coverage': _coverage(results, definitions)}


def _unlabeled_finish(c: collections.Counter) -> dict:
    done = sum(c[raw] for raw in DECISIONS)
    return dict(c, n=done, failed=c['failed'], uncertain=c['uncertain'],
                uncertain_rate=round(c['uncertain'] / done, 4) if done else None,
                coverage_rate=round(done / c['expected_n'], 4) if c['expected_n'] else None)


def _comparison_errors(cand: dict, bases: list) -> list[str]:
    errors = []
    if len(bases) < 2: errors.append('需要两次独立基线运行，不能只用一次基线晋级')
    runs = [('候选', cand)] + [(f'基线{i + 1}', b) for i, b in enumerate(bases)]
    for name, run in runs:
        evidence = run.get('input_evidence') or {}
        reference = cand.get('input_evidence') or {}
        if (type(evidence.get('version')) is not int or evidence['version'] != 1 or evidence.get('complete') is not True
                or not isinstance(evidence.get('inputs_sha256'), str) or len(evidence['inputs_sha256']) != 64
                or type(evidence.get('expected_units')) is not int or evidence['expected_units'] < 1
                or type(evidence.get('verified_units')) is not int or evidence['verified_units'] != evidence['expected_units']):
            errors.append(f'{name}缺少完整的实际模型输入证据，不能用清单哈希代替')
        elif evidence['inputs_sha256'] != reference.get('inputs_sha256'):
            errors.append(f'{name}实际图片、裁剪、请求模型或输入处理实现与候选不一致')
        if not run.get('evaluation_sha256'):
            errors.append(f'{name}缺少冻结输入清单身份，旧汇总仅供参考')
        elif run['evaluation_sha256'] != cand.get('evaluation_sha256'):
            errors.append(f'{name}输入或标签清单与候选不一致')
        for split in ('dev', 'unlabeled'):
            cov = run[split].get('coverage')
            ref = cand[split].get('coverage')
            if not cov or not ref:
                errors.append(f'{name}/{split}缺少覆盖证据，不能直接比较旧汇总'); continue
            if any(cov[k] != ref[k] for k in ('keys_sha256', 'definition_sha256', 'expected')):
                errors.append(f'{name}/{split}样本或标签不一致')
            if not cov['complete'] or cov['completed'] != cov['expected']:
                errors.append(f"{name}/{split}评测不完整：{cov['completed']}/{cov['expected']}")
            if split == 'dev' and not cov['expected']:
                errors.append(f'{name}/dev 没有金标样本')
    return errors


def _critical_counts(scored: dict, cls: str) -> dict:
    counts = collections.Counter()
    for name, c in scored['by_class'].items():
        if promptlib.card_key(name) == cls:
            counts.update({k: c[k] for k in ('n', 'gold_true', 'gold_false', 'true_rejected', 'false_released')})
    return {k: counts[k] for k in ('n', 'gold_true', 'gold_false', 'true_rejected', 'false_released')}


def _critical_gates(cand: dict, bases: list) -> dict:
    out = {}
    for cls in CRITICAL_CLASSES:
        c = _critical_counts(cand['dev'], cls)
        bs = [_critical_counts(b['dev'], cls) for b in bases]
        sufficient = len(bs) >= 2 and all(x['n'] >= MIN_CLASS_N and x['gold_true'] and x['gold_false'] for x in [c, *bs])
        same_gold = all((b['gold_true'], b['gold_false']) == (c['gold_true'], c['gold_false']) for b in bs)
        if not sufficient or not same_gold:
            out[cls] = {'passed': False, 'status': 'insufficient_evidence', 'candidate': c,
                        'reason': '缺少同批正负金标或计分不完整；每类至少10框且须同时有正负例，不代表统计充分'}
            continue
        limits = {k: max(b[k] for b in bs) for k in ('true_rejected', 'false_released')}
        passed = all(c[k] <= limits[k] for k in limits)
        out[cls] = {'passed': passed, 'status': 'passed' if passed else 'regressed',
                    'candidate': c, 'baseline_worst': limits,
                    'reason': '真目标误拒和假目标误放均不得超出重复基线的较差值，不额外放宽容忍量'}
    return out


def gates(cand: dict, bases: list) -> dict:
    """缺证据不能晋级；原 G1—G4 不放宽，G5 不允许真假错误相抵。"""
    cd = cand['dev']
    errors = _comparison_errors(cand, bases)
    g = {'G0_同一冻结清单且候选与两次基线完整计分': not errors}
    accs = [b['dev']['overall']['accuracy'] or 0 for b in bases]
    g['G1_总体准确率高于两次基线中较高者'] = bool(accs) and (cd['overall']['accuracy'] or 0) > max(accs)
    worse = []
    for cls, b0 in (bases[0]['dev']['by_class'].items() if bases else []):
        c = cd['by_class'].get(cls)
        lows = [b['dev']['by_class'].get(cls, {}).get('accuracy') for b in bases]
        lows = [x for x in lows if x is not None]
        if b0['n'] >= MIN_CLASS_N and c and c['accuracy'] is not None and lows and c['accuracy'] < min(lows) - 0.03:
            worse.append(f"{cls} 基线{min(lows):.1%}~{max(lows):.1%}→{c['accuracy']:.1%}")
    g['G2_各类准确率不低于两次基线较低者3个百分点以上'] = not worse
    fr = max((_critical_counts(b['dev'], '人员')['false_released'] for b in bases), default=0)
    g['G3_人员非人放过不比基线多3个以上'] = bool(bases) and _critical_counts(cd, '人员')['false_released'] <= fr + 3
    unc = cand['unlabeled']['overall']['uncertain_rate']
    base_unc = [b['unlabeled']['overall']['uncertain_rate'] for b in bases]
    empty_pool = all(r['unlabeled'].get('coverage', {}).get('expected') == 0 for r in [cand, *bases])
    g['G4_无标签uncertain比例不高于两次基线平均'] = empty_pool or (
        bool(base_unc) and unc is not None and all(x is not None for x in base_unc) and unc <= sum(base_unc) / len(base_unc))
    critical = _critical_gates(cand, bases)
    g['G5_人员和道路裂纹误拒误放不超出重复基线范围'] = all(c['passed'] for c in critical.values())
    return {'passed': all(g.values()), 'gates': g, 'regressed_classes': worse,
            'comparison_errors': errors, 'critical_classes': critical}


def paired_comparison(candidate: dict, baseline: dict, labels: dict, split: str) -> dict:
    """只诊断共同有效框；不能用共同子集代替完整覆盖门禁。"""
    expected = {k: v for k, v in labels.items() if v['split'] == split}
    shared = {k: v for k, v in expected.items() if candidate.get(k) in DECISIONS and baseline.get(k) in DECISIONS}
    return {'diagnostic_only': True, 'expected_n': len(expected), 'shared_n': len(shared), 'excluded_n': len(expected) - len(shared),
            'candidate': score(candidate, shared, split), 'baseline': score(baseline, shared, split)}


def flips(a: dict, b: dict) -> dict:
    """同一组有效框两次结果的翻转数（用于估计随机波动）。"""
    keys = {k for k in set(a) & set(b) if a[k] in DECISIONS and b[k] in DECISIONS}
    changed = sum((a[k] == 'true') != (b[k] == 'true') for k in keys)
    return {'compared': len(keys), 'decision_flips': changed, 'flip_rate': round(changed / len(keys), 4) if keys else None}
