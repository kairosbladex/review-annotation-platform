"""人员视图实验协议。不修改、不替代 G0。

G0 仍要求附件一致。本协议把附件变化写成被研究因素，并要求原图、框和标签不变。
计分沿用 only_true_is_true_v1：只有 true 算保留，uncertain 不算答对，缺失不算判假。
两轮都要满足门禁，不挑选较好的一轮。

measurement_passed 只表示两轮都出现了可复验改善。未退化单独记录，相同结果不是改善。
release_allowed 还要求样本来源记录本身证明同帧人工正负样本；调用方的布尔值不起作用。
任一关键核对失败都同时关掉这两项。
"""
from __future__ import annotations
import statistics

import metrics

PROTOCOL = 'person_view_experiment_v1'
FACTORS = {'person_view', 'person_card_references'}
PERSON_FIELDS = ('n', 'gold_true', 'gold_false', 'true_retained', 'true_rejected', 'false_released',
                 'uncertain', 'failed', 'expected_n', 'right')


def _median(values):
    return round(statistics.median(values), 3) if values else None


def _p90(values):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, int(round(0.9 * len(ordered) + 0.5)) - 1)], 3)


def _attachment_map(evidence):
    return {item['role']: item['sha256'] for item in evidence['attachments']}


def _filled(value):
    return isinstance(value, str) and bool(value.strip())


def _evidence_gaps(evidence):
    """必要字段和附件角色必须先齐，缺了就不能拿空值互相比较。"""
    if not isinstance(evidence, dict):
        return ['实际输入证据不完整']
    gaps = []
    for name in ('source_sha256', 'candidates_sha256', 'source_kind', 'prompt_sha256'):
        if not _filled(evidence.get(name)):
            gaps.append(f'缺少{name}')
    attachments = evidence.get('attachments')
    if not isinstance(attachments, list) or not attachments:
        gaps.append('附件为空或缺失')
        return gaps
    found = {}
    for item in attachments:
        if not isinstance(item, dict) or not _filled(item.get('role')) or not _filled(item.get('sha256')):
            gaps.append('附件缺少角色或摘要')
            continue
        found.setdefault(item['role'], item['sha256'])
    missing = [role for role in ('source', 'locator', 'crops') if role not in found]
    if missing:
        gaps.append('附件角色覆盖不足：' + '、'.join(missing))
    return gaps


def factor_errors(baseline, candidate, factor):
    """原图、框、定位图必须相同。允许变化的只有本因素声明的附件或提示词。

    source_sha256 是收到的原始字节。附件 source 的 sha256 是重新编码后的 source.jpg。
    两者可以不同；基线和候选分别比较原始来源，以及实际提交的 source、locator。
    """
    if factor not in FACTORS:
        return ['未知实验因素']
    gaps = [f'基线{item}' for item in _evidence_gaps(baseline)]
    gaps += [f'候选{item}' for item in _evidence_gaps(candidate)]
    if gaps:
        return gaps
    errors = []
    for name in ('source_sha256', 'candidates_sha256', 'source_kind'):
        if baseline.get(name) != candidate.get(name):
            errors.append(f'{name} 发生变化，超出视图实验允许的因素')
    if baseline.get('source_kind') != 'screenshot' and baseline.get('source_kind') != 'bound_clean_frame':
        errors.append('图片来源不是截图或已绑定原帧')
    base_files, cand_files = _attachment_map(baseline), _attachment_map(candidate)
    for role in ('source', 'locator'):
        if base_files.get(role) != cand_files.get(role):
            errors.append(f'{role} 附件发生变化')
    if factor == 'person_view':
        if baseline.get('prompt_sha256') != candidate.get('prompt_sha256'):
            errors.append('视图实验改变了提示词')
        if 'references' in cand_files or 'references' in base_files:
            errors.append('视图实验不应附带参考图')
        if not candidate.get('person_changed') and base_files.get('crops') != cand_files.get('crops'):
            errors.append('没有人员时裁剪附件必须与基线一致')
    else:
        if base_files.get('crops') != cand_files.get('crops'):
            errors.append('类别卡实验改变了视图')
        if baseline.get('prompt_sha256') == candidate.get('prompt_sha256'):
            errors.append('类别卡实验没有改变提示词')
        if 'references' not in cand_files:
            errors.append('类别卡实验缺少参考图附件')
        if 'references' in base_files:
            errors.append('基线不应附带参考图')
    return errors


def labels_digest(labels, split):
    """只绑定本次评分实际使用的类别、真假和划分，供各轮互相核对。"""
    selected = {key: {field: value[field] for field in ('class_name', 'label', 'split')}
                for key, value in labels.items() if value.get('split') == split}
    return metrics._sha(selected)


def _person_counts(scored):
    totals = {key: 0 for key in PERSON_FIELDS}
    found = False
    for name, row in scored['by_class'].items():
        if metrics.promptlib.card_key(name) != '人员':
            continue
        found = True
        for key in PERSON_FIELDS:
            totals[key] += row.get(key, 0)
    return totals if found else {key: 0 for key in PERSON_FIELDS}


def _evidence_map(run):
    by_unit = run.get('evidence_by_unit')
    if isinstance(by_unit, dict) and by_unit:
        return by_unit
    one = run.get('input_evidence')
    if isinstance(one, dict) and one:
        return {'*': one}
    return {}


def _declared_labels(run, split):
    declared = run.get('labels_sha256')
    if isinstance(declared, dict):
        return declared.get(split)
    return declared


def _run_check_errors(role, index, run, split, labels_sha256):
    errors = []
    name = f'{role}{index}'
    if not _evidence_map(run):
        errors.append(f'{name}缺少实际输入证据')
    config = run.get('model_config')
    if not isinstance(config, dict):
        errors.append(f'{name}缺少模型配置')
    else:
        if not isinstance(config.get('requested_model'), str) or not config['requested_model'].strip():
            errors.append(f'{name}缺少请求模型')
        if config.get('reasoning_effort') != 'none':
            errors.append(f'{name}推理量不是冻结的 none')
        if config.get('decision_policy') != 'three_state':
            errors.append(f'{name}输出口径不是三态')
        schema = config.get('schema_sha256')
        if not isinstance(schema, str) or len(schema) != 64:
            errors.append(f'{name}缺少输出约束摘要')
    if _declared_labels(run, split) != labels_sha256:
        errors.append(f'{name}标签摘要与评分标签不一致')
    return errors


def _complete(scored):
    coverage = scored.get('coverage') or {}
    return bool(coverage.get('complete')) and coverage.get('expected', 0) > 0


def _frozen_input_errors(left, right, label):
    errors = []
    for name in ('source_sha256', 'candidates_sha256', 'source_kind', 'prompt_sha256'):
        if left.get(name) != right.get(name):
            errors.append(f'{label}{name} 不一致')
    try:
        if _attachment_map(left) != _attachment_map(right):
            errors.append(f'{label}附件不一致')
    except (KeyError, TypeError):
        errors.append(f'{label}实际输入证据不完整')
    return errors


def _link_evidence(baseline_runs, candidate_runs, factor):
    """基线各轮必须是同一输入。候选只允许因子声明的那一项变化。"""
    errors = []
    maps = {'基线': [_evidence_map(run) for run in baseline_runs],
            '候选': [_evidence_map(run) for run in candidate_runs]}
    if any(not item for group in maps.values() for item in group):
        return errors
    for role, group in maps.items():
        for index, current in enumerate(group[1:], start=2):
            if set(current) != set(group[0]):
                errors.append(f'{role}各轮的输入图片集合不一致')
                continue
            for unit in group[0]:
                errors.extend(_frozen_input_errors(group[0][unit], current[unit], f'{role}{index} {unit} '))
    base_map, cand_map = maps['基线'][0], maps['候选'][0]
    if set(base_map) != set(cand_map):
        errors.append('候选与基线的输入图片集合不一致')
        return errors
    for unit in sorted(base_map):
        for bi, base in enumerate(maps['基线'], start=1):
            for ci, cand in enumerate(maps['候选'], start=1):
                try:
                    found = factor_errors(base[unit], cand[unit], factor)
                except (KeyError, TypeError):
                    found = ['实际输入证据不完整']
                where = '' if unit == '*' else f'{unit} '
                errors.extend(f'候选{ci}相对基线{bi} {where}{item}' for item in found)
    return errors


def _model_agreement_errors(runs):
    configs = [run.get('model_config') for run in runs if isinstance(run.get('model_config'), dict)]
    if len(configs) != len(runs):
        return []
    errors = []
    first = configs[0]
    for key in ('requested_model', 'schema_sha256'):
        if any(item.get(key) != first.get(key) for item in configs[1:]):
            errors.append(f'各轮模型配置 {key} 不一致')
    return errors


def _count_limits_hold(candidates, baselines):
    limits = {key: max(row[key] for row in baselines) for key in ('true_rejected', 'false_released')}
    return all(row['true_rejected'] <= limits['true_rejected'] and row['false_released'] <= limits['false_released']
               for row in candidates)


def _repeatable_gain(candidates, baselines):
    """每一轮候选都要严格好于每一轮基线，且两类错误都不能比该轮基线更多。"""
    if len(candidates) < 2 or len(baselines) < 2:
        return False
    for cand in candidates:
        for base in baselines:
            if cand['true_rejected'] > base['true_rejected'] or cand['false_released'] > base['false_released']:
                return False
            if cand['true_rejected'] + cand['false_released'] >= base['true_rejected'] + base['false_released']:
                return False
    return True


def _independence_errors(exposure, labels, split):
    """不读取 independent_validation。来源记录对不上标签，就不是独立验证。"""
    source = exposure.get('sample_source') if isinstance(exposure, dict) else None
    if not isinstance(source, dict):
        reason = exposure.get('reason') if isinstance(exposure, dict) else None
        return [reason or '缺少样本来源记录，调用方声明不能代替独立验证']
    errors = []
    source_id = source.get('source_id')
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append('样本来源缺少可追溯编号')
    if source.get('kind') != 'same_frame_human':
        errors.append('样本来源不是可靠同帧人工真值')
    if source.get('frame_binding') != 'same_frame_verified':
        errors.append('缺少同帧绑定，不能用邻帧或未绑定截图充当独立样本')
    exposed_ids = exposure.get('exposed_source_ids') or []
    if source.get('exposed') is True or (source_id and source_id in exposed_ids):
        errors.append('样本来源已曝光，只能做回归')
    positives = source.get('positive_keys') if isinstance(source.get('positive_keys'), list) else []
    negatives = source.get('negative_keys') if isinstance(source.get('negative_keys'), list) else []
    if not positives or not negatives:
        errors.append('独立验证缺少人员正样本或负样本')
    expected = labels_digest(labels, split)
    if source.get('labels_sha256') != expected:
        errors.append('样本来源的标签摘要与评分标签不一致')
    for key in positives:
        lab = labels.get(key) or {}
        if lab.get('split') != split or lab.get('label') != 'true' or metrics.promptlib.card_key(lab.get('class_name', '')) != '人员':
            errors.append(f'正样本 {key} 与评分标签不一致')
    for key in negatives:
        lab = labels.get(key) or {}
        if lab.get('split') != split or lab.get('label') != 'false' or metrics.promptlib.card_key(lab.get('class_name', '')) != '人员':
            errors.append(f'负样本 {key} 与评分标签不一致')
    covered = set(positives) | set(negatives)
    missing = [key for key, value in labels.items()
               if value.get('split') == split and metrics.promptlib.card_key(value.get('class_name', '')) == '人员'
               and key not in covered]
    if missing:
        errors.append('独立来源没有覆盖本次评分的全部人员框')
    return errors


def evaluate(candidate_runs, baseline_runs, labels, *, factor, split='dev', exposure=None):
    """candidate_runs / baseline_runs 至少各两轮。每轮要带实际输入、模型配置和标签摘要。"""
    exposure = exposure if isinstance(exposure, dict) else {}
    critical = []
    if len(baseline_runs) < 2 or len(candidate_runs) < 2:
        critical.append('需要基线和候选各至少两轮，不能挑选最好一轮')
    if factor not in FACTORS:
        critical.append('未知实验因素')
    expected_labels = labels_digest(labels, split)
    named = [('基线', index, run) for index, run in enumerate(baseline_runs, start=1)]
    named += [('候选', index, run) for index, run in enumerate(candidate_runs, start=1)]
    for role, index, run in named:
        critical.extend(_run_check_errors(role, index, run, split, expected_labels))
    if factor in FACTORS:
        critical.extend(_link_evidence(baseline_runs, candidate_runs, factor))
    critical.extend(_model_agreement_errors([run for _, _, run in named]))
    scored_c = [metrics.score(run['results'], labels, split) for run in candidate_runs]
    scored_b = [metrics.score(run['results'], labels, split) for run in baseline_runs]
    for name, runs in (('候选', scored_c), ('基线', scored_b)):
        for index, scored in enumerate(runs, start=1):
            if not _complete(scored):
                critical.append(f'{name}{index} 计分不完整，缺失或非法输出没有算作判假')
    person_rows = [_person_counts(item) for item in scored_c]
    base_counts = [_person_counts(item) for item in scored_b]
    counts_hold = False
    gain = False
    complete = (len(scored_c) >= 2 and len(scored_b) >= 2 and all(_complete(item) for item in scored_c + scored_b))
    if complete:
        same_gold = all((row['gold_true'], row['gold_false']) == (base_counts[0]['gold_true'], base_counts[0]['gold_false'])
                        for row in person_rows + base_counts)
        enough = all(row['n'] >= metrics.MIN_CLASS_N and row['gold_true'] and row['gold_false'] for row in base_counts)
        if not same_gold or not enough:
            critical.append('人员正负样本不足或各轮分母不一致')
        else:
            limits = {key: max(row[key] for row in base_counts) for key in ('true_rejected', 'false_released')}
            for index, counts in enumerate(person_rows, start=1):
                if counts['true_rejected'] > limits['true_rejected']:
                    critical.append(f'候选{index} 真人误拒 {counts["true_rejected"]} 高于重复基线较差值 {limits["true_rejected"]}')
                if counts['false_released'] > limits['false_released']:
                    critical.append(f'候选{index} 非人误放 {counts["false_released"]} 高于重复基线较差值 {limits["false_released"]}')
            counts_hold = _count_limits_hold(person_rows, base_counts)
            gain = _repeatable_gain(person_rows, base_counts)
    other = _other_class_regression(scored_c, scored_b) if scored_c and scored_b else []
    critical.extend(other)
    no_regression = not critical and counts_hold and not other
    improvement = no_regression and gain
    independence_errors = _independence_errors(exposure, labels, split)
    errors = critical + independence_errors
    paired = pair_boxes(candidate_runs, baseline_runs, labels, split) if candidate_runs and baseline_runs else []
    flips = []
    if len(candidate_runs) >= 2:
        flips.append({'role': 'candidate', **metrics.flips(candidate_runs[0]['results'], candidate_runs[1]['results'])})
    if len(baseline_runs) >= 2:
        flips.append({'role': 'baseline', **metrics.flips(baseline_runs[0]['results'], baseline_runs[1]['results'])})
    elapsed = [value for run in candidate_runs for value in (run.get('elapsed_s') or {}).values()]
    return {
        'protocol': PROTOCOL,
        'factor': factor,
        'split': split,
        'g0_replaced': False,
        'checks_passed': not critical,
        'no_regression': no_regression,
        'improvement': improvement,
        'measurement_passed': improvement,
        'release_allowed': improvement and not independence_errors,
        'errors': errors,
        'person': person_rows,
        'baseline_person': base_counts,
        'candidate_scored': scored_c,
        'baseline_scored': scored_b,
        'paired': paired,
        'flips': flips,
        'elapsed_s': {'n': len(elapsed), 'median': _median(elapsed), 'p90': _p90(elapsed)},
        'exposure': exposure,
        'independence': {'accepted': not independence_errors, 'errors': independence_errors},
    }


def _other_class_regression(candidates, baselines):
    """人员图上的其他类别只在样本达到 10 且两轮都变差时拦截；样本不足单独说明。"""
    errors = []
    classes = set()
    for scored in candidates + baselines:
        classes.update(scored['by_class'])
    for name in sorted(classes):
        if metrics.promptlib.card_key(name) == '人员':
            continue
        base = [s['by_class'].get(name) for s in baselines]
        cand = [s['by_class'].get(name) for s in candidates]
        if any(row is None or row['n'] < metrics.MIN_CLASS_N for row in base + cand):
            continue
        worst_reject = max(row['true_rejected'] for row in base)
        worst_release = max(row['false_released'] for row in base)
        for index, row in enumerate(cand, start=1):
            if row['true_rejected'] > worst_reject or row['false_released'] > worst_release:
                errors.append(f'其他类别 {name} 第{index}轮相对重复基线变差')
    return errors


def pair_boxes(candidate_runs, baseline_runs, labels, split):
    rows = []
    keys = [key for key, label in labels.items() if label['split'] == split]
    for key in keys:
        label = labels[key]
        rows.append({
            'key': key,
            'class_name': label['class_name'],
            'label': label['label'],
            'baseline': [run['results'].get(key) for run in baseline_runs],
            'candidate': [run['results'].get(key) for run in candidate_runs],
        })
    return rows


def three_way_flips(a, b):
    keys = [key for key in set(a) & set(b) if a[key] in metrics.DECISIONS and b[key] in metrics.DECISIONS]
    changed = sum(a[key] != b[key] for key in keys)
    return {'compared': len(keys), 'three_way_flips': changed,
            'three_way_flip_rate': round(changed / len(keys), 4) if keys else None}
