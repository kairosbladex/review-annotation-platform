#!/usr/bin/env python3
"""One authorized pre-evaluation refreeze; refuses an already used test set."""
import argparse
import collections
import json
from datetime import datetime
from pathlib import Path
from consolidate_recheck_gold import TZ, read_rows, write_rows, refresh_sums
from freeze_recheck_gold_split import stratified_assignments, stats, table

REASON = '修正小层向下取整偏差 + 12 样本曝光核对，评测前重冻'
SESSION = '/Users/wangxiaolin/.codex/sessions/2026/09/30/rollout-2026-09-30T14-13-57-01a0f0f2-c708-7932-a95e-62c05f4206e7.jsonl'

def refreeze(out, source):
    if (out/'test_usage.jsonl').exists() and (out/'test_usage.jsonl').stat().st_size:
        raise ValueError('测试集已使用，禁止重冻')
    registry = read_rows(out/'split_registry.jsonl')
    if any(r.get('reason') == REASON for r in registry):
        raise ValueError('本次授权的重冻已完成')
    gold = read_rows(out/'gold.jsonl')
    before = stats(gold)
    selected = [r for r in read_rows(source/'candidate_pairs.jsonl')
                if r['partition'] == 'validation' and r.get('label_source') == 'wxl_hires_web_20260930']
    assert len(selected) == 12
    mapping = []
    for r in selected:
        matches = [g for g in gold if any(p.get('candidate_id') == r['origin_candidate_id'] for p in g['provenance'])]
        if len(matches) != 1:
            raise ValueError('回归样本映射不唯一: '+r['candidate_id'])
        g = matches[0]
        mapping.append({'sample_id': r['candidate_id'], 'candidate_id': r['origin_candidate_id'],
                        'gold_id': g['gold_id'], 'group_id': g['group_id'],
                        'evidence': [SESSION+':362', SESSION+':667', str(source/'candidate_pairs.jsonl')],
                        'selection': 'partition=validation AND label_source=wxl_hires_web_20260930'})
    exposed = {r['group_id'] for r in mapping}
    for r in gold:
        if r['group_id'] in exposed:
            r['group_exposure'] = sorted(set(r['group_exposure']) | {'gpt_prompt_exposed'})
            r['exposure'] = sorted(set(r['exposure']) | {'gpt_prompt_exposed'})
    assignments, audit = stratified_assignments(gold)
    timestamp = datetime.now(TZ).isoformat()
    prior = {r['group_id']: r for r in registry}
    for gid, a in sorted(assignments.items()):
        a['aliases'] = sorted(set(a['aliases']) | set(prior[gid]['aliases']))
        registry.append({**a, 'assignment_reason': a['reason'], 'reason': REASON,
                         'record_type': 'pre_evaluation_refreeze', 'seed': 20260930,
                         'recorded_at': timestamp, 'previous_split': prior[gid]['split']})
    for r in gold:
        a = assignments[r['group_id']]
        r.update(split=a['split'], valid_for=a['valid_for'], split_status=a['status'])
    report = json.loads((out/'summary.json').read_text())
    report.update(classes=stats(gold), population={p: stats([r for r in gold if p in r['population']])
                  for p in sorted({p for r in gold for p in r['population']})}, stratified_split=audit,
                  frozen_at=timestamp, regression_audit={'path': 'exact_12_found', 'mapped': 12, 'exposed_groups': len(exposed)},
                  unresolved_exposure_manifests=[])
    report['limitations'][0] = '12个回归样本全部唯一映射，3个任务组整体标记GPT曝光；只读会话核对。'
    locked_sources=collections.defaultdict(lambda: {'groups':set(),'boxes':set(),'population':collections.defaultdict(set)})
    from consolidate_recheck_gold import population
    for g in gold:
        if g['split']!='locked_test_gpt':continue
        for provenance in g['provenance']:
            name=provenance['source_file'];v=locked_sources[name]
            v['groups'].add(g['group_id']);v['boxes'].add(g['gold_id']);v['population'][population(name)].add(g['gold_id'])
    report['locked_test_gpt_sources']=[{'source':name,'groups':len(v['groups']),'boxes':len(v['boxes']),'population':{k:len(ids) for k,ids in v['population'].items()}} for name,v in sorted(locked_sources.items())]

    comparison = {'baseline_commit': 'bba595e', 'before': before, 'after': report['classes'],
                  'rates': {str(flag): {'locked': sum(s['locked_groups'] for s in audit if s['has_false']==flag),
                                        'eligible': sum(s['eligible_groups'] for s in audit if s['has_false']==flag)} for flag in (True, False)}}
    old = (out/'README.md').read_text(); suffix=old[old.index('## 补齐元数据'):]
    prefix = ['# 复检统一金标：评测前最后一次重冻', '', REASON, '',
              '找到12个回归样本：R045～R056。会话证据 '+SESSION+':362（样本选择）及 :667（仅开发回归）。逐条映射见 regression_exposure.jsonl，12/12唯一对应金标，3个整组标记 gpt_prompt_exposed。采用精确核对路径，未采用全输入保守兜底。', '',
              '主类别按组内框数最多确定；主population按出现框数最多确定；并列按字符串排序。种子20260930。分层键：主类别 × 主population × 是否含false。含false与无false分别计算总名额，四舍五入到最近整数；每层先floor，按余数降序补齐，并列按分层键排序。uncertain-only、GPT曝光、历史未知继续只进开发集；冲突继续隔离；valid_for不变。', '',
              '本次为尚未评测的最后一次重冻。后续已有组不重切；新组独立按同一规则分配并追加registry。新增来源连接两个既有组时停止并请人工裁决。', '',
              '对比bba595e见 refreeze_comparison.json。比例分母为符合GPT锁定条件的组，曝光组不参与分配。', '',
              '车类只对真目标误判假率给出通过/不通过；锁定集假目标达到100条前，假目标判假率只报告数字与置信区间。', '']
    prefix += table(report['classes'])
    prefix += ['', '## 车类锁定集 population', '', '|population|true框|false框|uncertain框|true组|false组|uncertain组|总组|', '|---|---|---|---|---|---|---|---|']
    for p, counts in report['population'].items():
        if '车' in counts:
            c=counts['车'];prefix.append('|'+ '|'.join(map(str,[p,*c['locked_test_samples'].values(),*c['locked_test_groups'].values(),c['locked_all_groups']]))+'|')
    prefix += ['', 'population可重叠，不可相加。', '', '## 实际分配比例', '']
    for flag,v in comparison['rates'].items(): prefix.append(f"含false={flag}: {v['locked']}/{v['eligible']} = {v['locked']/v['eligible']:.2%}")
    suffix=suffix.replace('12个回归样本ID待用户提供，跨来源重叠核验仍未完成。','12个回归样本核对已完成，详见回归曝光映射。')
    write_rows(out/'gold.jsonl',gold);write_rows(out/'split_registry.jsonl',registry);write_rows(out/'strata.jsonl',audit)
    write_rows(out/'regression_exposure.jsonl',mapping)
    write_rows(out/'splits.jsonl',[{k:r[k] for k in ('gold_id','group_id','split','valid_for','split_status')} for r in gold])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'refreeze_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text('\n'.join(prefix)+'\n\n'+suffix);refresh_sums(out)
    print(json.dumps({'rates':comparison['rates'],'vehicle':report['classes']['车']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();refreeze(a.out,a.source)
