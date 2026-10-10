#!/usr/bin/env python3
"""将干净原帧三视图的两次结果与 10-09 两组历史对照逐框配对。"""
import argparse
import collections
import json
from pathlib import Path
import statistics

from score_luna_clean_frame import CONDITIONS as OLD_CONDITIONS, compare, measure

NEW_CONDITIONS = ['clean_only-1', 'clean_only-2']
CONDITIONS = OLD_CONDITIONS + NEW_CONDITIONS


def read(path): return json.loads(path.read_text())


def paired(rows, baseline):
    # 复用已验证的计分器；其两个 arm 的内部名称仅用于计算。
    mapped = [dict(r, decisions={f'{alias}-{n}':r['decisions'][f'{arm}-{n}']
                                for alias,arm in [('screenshot',baseline),('plus_clean','clean_only')] for n in (1,2)}) for r in rows]
    result = compare(mapped)
    result['arm_mapping'] = {'screenshot':baseline,'plus_clean':'clean_only'}
    if rows and all(r['label']=='false' for r in rows):
        ci = result['mean_delta_pp_cluster_bootstrap95']
        result['false_acceptance_rate_delta_pp_cluster_bootstrap95'] = [-ci[1],-ci[0]]
        result['mean_false_acceptance_rate_delta_pp'] = sum(100*(result[f'plus_clean-{n}']['false_accepted']-result[f'screenshot-{n}']['false_accepted'])/len(rows) for n in (1,2))/2
    return result


def score(root, baseline):
    units, labels = read(baseline/'included.json'), read(baseline/'labels.json')
    records = {c:{p.stem:read(p) for p in ((root if c in NEW_CONDITIONS else baseline)/'results'/c).glob('*.json')} for c in CONDITIONS}
    new_records = [r for c in NEW_CONDITIONS for r in records[c].values()]
    ids = [r['execution_id'] for r in new_records if r['success']]
    old_ids = {r['execution_id'] for c in OLD_CONDITIONS for r in records[c].values() if r['success']}
    assert len(ids)==len(set(ids)) and not set(ids)&old_ids
    rows, unknown, incomplete = [], [], []
    for unit in units:
        uid=unit['unit']
        if not all(records[c].get(uid,{}).get('success') for c in CONDITIONS):
            incomplete.append(uid);continue
        assert len({records[c][uid]['prompt_sha256'] for c in NEW_CONDITIONS})==1
        assert all(records[c][uid]['image_sha256']==unit['clean_sha256'] for c in NEW_CONDITIONS)
        outputs={c:{r['key']:r for r in records[c][uid]['results']} for c in CONDITIONS}
        keys={c['key'] for c in unit['candidates']}
        assert all(set(o)==keys for o in outputs.values())
        for candidate in unit['candidates']:
            key=candidate['key'];lab=labels.get(key,{})
            row={'unit':uid,'key':key,'event_id':unit['event_id'],'split':unit['split'],'class_name':candidate['class_name'],
                 'source_local_path':unit['source_local_path'],'source_kind':unit['source_kind'],'label':lab.get('label'),
                 'decisions':{c:outputs[c][key]['decision'] for c in CONDITIONS},'new_details':{c:outputs[c][key] for c in NEW_CONDITIONS}}
            if lab.get('label') in ('true','false'):
                assert lab['split']==unit['split'] and lab['class_name']==candidate['class_name'];rows.append(row)
            else:unknown.append(row)
    classes=sorted({x['class_name'] for x in labels.values()})
    result={'conditions':CONDITIONS,'complete_images':len(units)-len(incomplete),'incomplete_images':incomplete,
            'scored_candidates':len(rows),'unscored_candidates':len(unknown),'splits':{},'comparisons':{},
            'requests':{'records':len(new_records),'success':sum(r['success'] for r in new_records),'failed':sum(not r['success'] for r in new_records),
                        'unique_execution_ids':len(ids),'attempt_status':dict(collections.Counter(a['status'] for r in new_records for a in r['attempts'])),
                        'requested_models':sorted({r['requested_model'] for r in new_records if r['success']}),
                        'latency_median_seconds':{c:statistics.median(a['seconds'] for r in records[c].values() for a in r['attempts']) if records[c] else None for c in NEW_CONDITIONS}}}
    for split in ('dev','holdout'):
        sub=[r for r in rows if r['split']==split]
        groups={'全部':sub,**{cls:[r for r in sub if r['class_name']==cls] for cls in classes}}
        result['splits'][split]={cls:{c:measure(group,c) for c in CONDITIONS} for cls,group in groups.items()}
    for old in ('screenshot','plus_clean'):
        result['comparisons'][old]={}
        for split in ('dev','holdout'):
            sub=[r for r in rows if r['split']==split]
            result['comparisons'][old][split]={'overall':paired(sub,old),
                'classes':{cls:paired([r for r in sub if r['class_name']==cls],old) for cls in classes},
                'vehicle_true':paired([r for r in sub if r['class_name']=='车' and r['label']=='true'],old),
                'people_false':paired([r for r in sub if r['class_name']=='人员' and r['label']=='false'],old)}
    all_rows=rows+unknown
    result['all_candidate_uncertain']={s:{cls:{'n':sum(r['split']==s and r['class_name']==cls for r in all_rows),
        **{c:sum(r['split']==s and r['class_name']==cls and r['decisions'][c]=='uncertain' for r in all_rows) for c in CONDITIONS}}
        for cls in classes} for s in ('dev','holdout')}
    for name,value in [('report.json',result),('per-box.json',rows),('unscored-candidates.json',unknown)]:
        (root/name).write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps({k:result[k] for k in ['complete_images','incomplete_images','scored_candidates','unscored_candidates','requests']},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    args=ap.parse_args();score(args.evidence,args.baseline)
