#!/usr/bin/env python3
"""Freeze the audited local gold snapshot without reading images or calling models."""
import argparse
import collections
import json
import random
from datetime import datetime
from pathlib import Path

from consolidate_recheck_gold import SEED, TZ, dumps, population, read_rows, refresh_sums, statistics, write_rows


def stratified_assignments(gold):
    groups = collections.defaultdict(list)
    for row in gold:
        groups[row['group_id']].append(row)
    strata = collections.defaultdict(list)
    assignments = {}
    for gid, rows in sorted(groups.items()):
        exposure = sorted({e for r in rows for e in r['group_exposure']})
        unknown = any(r['gpt_history_unknown'] for r in rows)
        classes = sorted({r['class_name'] for r in rows})
        populations = sorted({p for r in rows for p in r['population']})
        has_false = any(r['label'] == 'false' for r in rows)
        labels = {r['label'] for r in rows}
        key = (tuple(classes), tuple(populations), has_false)
        reason = ('gpt_prompt_exposed' if 'gpt_prompt_exposed' in exposure else
                  'gpt_history_unknown' if unknown else
                  'uncertain_only' if labels == {'uncertain'} else 'stratified_group_random')
        assignments[gid] = {'group_id': gid, 'split': 'development', 'exposure': exposure,
                            'gpt_history_unknown': unknown, 'reason': reason,
                            'aliases': sorted({a for r in rows for a in r['group_aliases']}),
                            'stratum': {'classes': classes, 'population': populations, 'has_false': has_false},
                            'status': 'development_only', 'valid_for': []}
        if reason == 'stratified_group_random':
            strata[key].append(gid)
    rng = random.Random(SEED)
    audit = []
    for key, ids in sorted(strata.items()):
        rng.shuffle(ids)  # sorted groups and strata make input order irrelevant
        numerator = 7 if key[2] else 5
        count = len(ids) * numerator // 10
        for gid in ids[:count]:
            row = assignments[gid]
            row.update(split='locked_test_gpt', status='frozen',
                       valid_for=['gpt'] if 'qwen_exposed' in row['exposure'] else ['gpt', 'qwen'])
        audit.append({'classes': list(key[0]), 'population': list(key[1]), 'has_false': key[2],
                      'eligible_groups': len(ids), 'locked_groups': count,
                      'locked_fraction': numerator / 10, 'rounding': 'floor'})
    return assignments, audit


def table(stats):
    lines = ['|类别|全量 true/false/uncertain 框|locked true/false/uncertain 框|locked true/false/uncertain 组|locked 总组数|档位|',
             '|---|---|---|---|---|---|']
    for cls, counts in stats.items():
        fmt = lambda d: '/'.join(str(d[k]) for k in ('true', 'false', 'uncertain'))
        lines.append(f"|{cls}|{fmt(counts['samples'])}|{fmt(counts['locked_test_samples'])}|{fmt(counts['locked_test_groups'])}|{counts['locked_all_groups']}|{counts['tier']}|")
    return lines


def stats(rows):
    result = statistics(rows)
    for cls, counts in result.items():
        counts['locked_all_groups'] = len({r['group_id'] for r in rows if r['class_name'] == cls and r['split'] == 'locked_test_gpt'})
    return result


def freeze(out):
    report = json.loads((out / 'summary.json').read_text())
    if report['split_status'] == 'locked_test_gpt_frozen':
        raise ValueError('已冻结，禁止重新随机切分；后续回归ID核对只能追加整组退回记录')
    gold = read_rows(out / 'gold.jsonl')
    conflicts = read_rows(out / 'conflicts.jsonl')
    registry = read_rows(out / 'split_registry.jsonl')
    assignments, audit = stratified_assignments(gold)
    timestamp = datetime.now(TZ).isoformat()
    # Keep original registry history, including the complete original aliases.
    prior = {r['group_id']: r for r in registry}
    conflict_aliases = collections.defaultdict(set)
    for row in conflicts:
        conflict_aliases[row['group_id']].update(row['group_aliases'])
    for gid, assignment in sorted(assignments.items()):
        assignment['aliases'] = sorted(set(assignment['aliases']) | set(prior.get(gid, {}).get('aliases', [])) | conflict_aliases[gid])
        registry.append({**assignment, 'record_type': 'stratified_freeze', 'seed': SEED,
                         'recorded_at': timestamp, 'previous_split': prior.get(gid, {}).get('split'),
                         'regression_id_audit': 'pending_user_12_ids'})
    for row in gold:
        assignment = assignments[row['group_id']]
        row.update(split=assignment['split'], valid_for=assignment['valid_for'], split_status=assignment['status'])
    # Conflicting boxes never enter evaluation, even when other boxes share their group.
    for row in conflicts:
        row.update(split='quarantine', valid_for=[], split_status='requires_human_adjudication')
    report.update(split_status='locked_test_gpt_frozen', frozen_at=timestamp,
                  classes=stats(gold), population={p: stats([r for r in gold if p in r['population']])
                  for p in sorted({p for r in gold for p in r['population']})}, stratified_split=audit)
    locked_sources = collections.defaultdict(lambda: {'groups': set(), 'boxes': set(), 'population': collections.defaultdict(set)})
    for row in gold:
        if row['split'] != 'locked_test_gpt':
            continue
        for provenance in row['provenance']:
            source = provenance['source_file']
            v = locked_sources[source]
            v['groups'].add(row['group_id']); v['boxes'].add(row['gold_id'])
            v['population'][population(source)].add(row['gold_id'])
    report['locked_test_gpt_sources'] = [{'source': s, 'groups': len(v['groups']), 'boxes': len(v['boxes']),
                                         'population': {p: len(ids) for p, ids in v['population'].items()}}
                                        for s, v in sorted(locked_sources.items())]
    report['unresolved_exposure_manifests'] = [{'source': '12个复检回归样本',
        'reason': '用户将另行提供ID；先冻结，收到后核对，命中locked_test_gpt则整组退回development并追加registry记录'}]
    report['limitations'][0] = '12个复检回归样本ID待用户提供；冻结不代表这12个样本的重叠核对已经通过。收到后整组退回规则见README。'
    report['vehicle_evaluation_policy'] = {'formal_metric': '真目标被误判为假的比例',
        'false_target_pass_minimum': 100, 'below_minimum': '只报告数字和置信区间，不给通过结论'}
    old_readme = (out / 'README.md').read_text()
    suffix = old_readme[old_readme.index('## 补齐元数据'):]
    suffix = suffix.replace('但12个回归样本独立清单未定位，跨来源重叠核验仍未完成，不能把测试候选视为完整曝光审计通过。',
                            '12个回归样本ID待用户提供，跨来源重叠核验仍未完成。')
    lines = ['# 复检统一金标：第0步切分已冻结', '',
             f"冻结时间：{timestamp}。人工金标 {len(gold)} 框；冲突 {len(conflicts)} 框继续隔离，不阻塞冻结。", '',
             '固定种子20260930。分层键为类别 × population × 组内是否含false。按排序后的组ID、排序后的分层，使用同一个Python random.Random(seed)逐层shuffle；含false组70%进入locked_test_gpt，只含true的组50%；组数向下取整floor(N×比例)。跨类别或population的组以完整类别集合、population集合为独立分层，不拆组。true与uncertain混合但无false的组按50%；只有uncertain的组保留development。', '',
             'GPT已曝光或历史不明的整组继续进入development，原有曝光规则不变。仅Qwen曝光不阻止GPT锁定测试，valid_for=["gpt"]，禁止用于Qwen评测。冲突框隔离，不计入分层标签和评测框数。', '',
             'split_registry.jsonl保留旧记录并追加本次冻结记录；同一group_id按最后一条记录解释当前切分。locked_test_gpt现已冻结，不得重新随机切分。proposed_groups.jsonl是历史提案，当前权威为split_registry.jsonl与splits.jsonl。', '',
             '12个复检回归样本ID尚未提供。收到完整ID清单后按candidate_id/gold_id及来源追溯核对；若命中locked_test_gpt，将所在整组移回development，清空valid_for，在split_registry追加原切分、新切分、触发ID、原因和时间，并同步gold/splits/统计/校验和。禁止只移单框，也不补抽其他组。不明或一对多ID须先核清。', '',
             '车类当前只能正式评测“真目标被误判为假的比例”。“假目标判假率”要等locked_test_gpt中车类假目标达到100条后才给通过结论；现阶段只报告数字和置信区间。通用完整评测样本门槛仍是真/假各100框、各10独立组；uncertain不计真/假。样本档位不表示模型已通过。比例报告给出分子/分母和95%置信区间，并注明同组相关性，不能把每个框都当作独立样本。', '',
             '第1～3步线上复检链路尚未开始，必须等待用户确认数据外发授权。当前model_calls、image_uploads、database_writes、event_submissions均为0。', '']
    lines += table(report['classes'])
    lines += ['', '## 车类 locked_test_gpt population 分布', '',
              '|population|true框|false框|uncertain框|true组|false组|uncertain组|总组数|', '|---|---|---|---|---|---|---|---|']
    for p, counts in report['population'].items():
        c = counts.get('车')
        if c:
            lines.append('|' + '|'.join(map(str, [p, *c['locked_test_samples'].values(), *c['locked_test_groups'].values(), c['locked_all_groups']])) + '|')
    lines += ['', '同一框或组可能包含多个population，各行不可直接相加。', '', '## population分开清点（无准确率评测）', '']
    for p, counts in report['population'].items():
        lines += ['### ' + p, ''] + table(counts) + ['']
    lines += ['## locked_test_gpt来源清单', '', '同一金标若有多来源，分别计数，不可相加。', '',
              '|来源|组|框|population框分布|', '|---|---|---|---|']
    for v in report['locked_test_gpt_sources']:
        lines.append(f"|{v['source']}|{v['groups']}|{v['boxes']}|{dumps(v['population'])}|")
    write_rows(out / 'gold.jsonl', gold)
    write_rows(out / 'conflicts.jsonl', conflicts)
    write_rows(out / 'split_registry.jsonl', registry)
    write_rows(out / 'strata.jsonl', audit)
    write_rows(out / 'splits.jsonl', [{k: r[k] for k in ('gold_id', 'group_id', 'split', 'valid_for', 'split_status')} for r in gold])
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'README.md').write_text('\n'.join(lines) + '\n\n' + suffix)
    refresh_sums(out)
    print(json.dumps(report['classes']['车'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    freeze(parser.parse_args().out.resolve())
