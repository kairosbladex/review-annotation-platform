#!/usr/bin/env python3
"""四组八条件完整性审计，以及按原片聚类的配对百分位区间。"""
import argparse, collections, json, random
from pathlib import Path
from score_luna_clean_frame import measure
CONDITIONS=[f'{a}-{n}' for a in 'ABCD' for n in (1,2)]
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=1)+'\n')
def paired(rows,a,b):
    metrics={}
    for metric in ('accuracy','uncertain','false_acceptance','true_rejection'):
        selected=[r for r in rows if metric not in ('false_acceptance','true_rejection') or r['label']==('false' if metric=='false_acceptance' else 'true')]
        def value(r,c):
            d=r['decisions'][c]
            return {'accuracy':(d=='true')==(r['label']=='true'),'uncertain':d=='uncertain','false_acceptance':d=='true','true_rejection':d!='true'}[metric]
        groups=collections.defaultdict(list)
        for r in selected:groups[r['source_local_path']].append(sum(value(r,f'{b}-{n}')-value(r,f'{a}-{n}') for n in (1,2))/2)
        totals=[(sum(v),len(v)) for v in groups.values()]
        if not totals:metrics[metric]={'n':0,'delta_pp':None,'ci95_pp':None};continue
        rng=random.Random(20261010);boot=[]
        for _ in range(2000):
            sample=rng.choices(totals,k=len(totals));boot.append(100*sum(v[0] for v in sample)/sum(v[1] for v in sample))
        boot.sort()
        metrics[metric]={'n':len(selected),'source_clusters':len(totals),'delta_pp':100*sum(v[0] for v in totals)/len(selected),'ci95_pp':[boot[49],boot[1949]],'single_cluster':len(totals)==1}
    return metrics

def score(root,baseline):
    units=read(baseline/'included.json');labels=read(baseline/'labels.json');protocol=read(root/'protocol.json')
    assert len(units)==300 and len({u['unit'] for u in units})==300
    records={c:{p.stem:read(p) for p in (root/'results'/c).glob('*.json')} for c in CONDITIONS}
    assert all(set(group)<={u['unit'] for u in units} for group in records.values())
    all_records=[v for group in records.values() for v in group.values()]
    ids=[r['execution_id'] for r in all_records if r['success']]
    assert len(ids)==len(set(ids)) and all(ids)
    assert all(r['binary_sha256']==protocol['binary_sha256'] for r in all_records)
    rows=[];unknown=[];incomplete=[]
    for u in units:
        uid=u['unit']; keys={c['key'] for c in u['candidates']}
        if not all(records[c].get(uid,{}).get('success') for c in CONDITIONS):incomplete.append(uid);continue
        outputs={c:{v['key']:v for v in records[c][uid]['results']} for c in CONDITIONS}
        assert all(set(o)==keys for o in outputs.values())
        for arm in 'ABCD':
            assert records[f'{arm}-1'][uid]['prompt_sha256']==records[f'{arm}-2'][uid]['prompt_sha256']
            assert all(records[f'{arm}-{n}'][uid]['image_sha256']==u['clean_sha256' if arm in 'CD' else 'sha256'] for n in (1,2))
        if any(c['class_name']!='违建' for c in u['candidates']):
            for a,b in [('A','B'),('C','D')]:assert records[f'{a}-1'][uid]['prompt_sha256']==records[f'{b}-1'][uid]['prompt_sha256']
        for candidate in u['candidates']:
            key=candidate['key'];lab=labels.get(key,{})
            r={k:u[k] for k in ('unit','split','source_local_path','event_id','source_kind')}
            r.update(key=key,class_name=candidate['class_name'],label=lab.get('label'),decisions={c:outputs[c][key]['decision'] for c in CONDITIONS})
            if r['label'] in ('true','false'):
                assert lab['split']==u['split'] and lab['class_name']==r['class_name'];rows.append(r)
            else:unknown.append(r)
    classes=sorted({v['class_name'] for v in labels.values()})
    report={'complete_images':300-len(incomplete),'incomplete_images':incomplete,'scored_candidates':len(rows),'unscored_candidates':len(unknown),'requests':{'records':len(all_records),'success':len(ids),'attempt_status':dict(collections.Counter(a['status'] for r in all_records for a in r['attempts'])),'models':sorted({r['requested_model'] for r in all_records}),'start':min((r['started'] for r in all_records),default=None),'finish':max((r['finished'] for r in all_records),default=None)},'splits':{},'comparisons':{}}
    for split in ('dev','holdout'):
        sub=[r for r in rows if r['split']==split];groups={'全部':sub,**{cls:[r for r in sub if r['class_name']==cls] for cls in classes}}
        report['splits'][split]={cls:{c:measure(group,c) for c in CONDITIONS} for cls,group in groups.items()}
        report['comparisons'][split]={f'{b}-{a}':{cls:paired(group,a,b) for cls,group in groups.items()} for a,b in [('A','B'),('C','D'),('A','C'),('B','D'),('A','D')]}
    sources={s:{u['source_local_path'] for u in units if u['split']==s} for s in ('dev','holdout')}
    overlap=sources['dev']&sources['holdout'];report['cross_split_sources']=sorted(overlap)
    report['holdout_without_cross_split_sources']={f'{b}-{a}':{cls:paired([r for r in rows if r['split']=='holdout' and r['source_local_path'] not in overlap and r['class_name']==cls],a,b) for cls in ('车','人员','违建')} for a,b in [('A','B'),('C','D'),('A','C'),('B','D')]}
    report['all_candidate_uncertain']={s:{cls:{c:sum(r['decisions'][c]=='uncertain' for r in rows+unknown if r['split']==s and r['class_name']==cls) for c in CONDITIONS} for cls in classes} for s in ('dev','holdout')}
    report['repeat_flips']={s:{cls:{a:sum(r['decisions'][a+'-1']!=r['decisions'][a+'-2'] for r in rows if r['split']==s and r['class_name']==cls) for a in 'ABCD'} for cls in classes} for s in ('dev','holdout')}
    save(root/'report.json',report);save(root/'per-box.json',rows);save(root/'unscored-candidates.json',unknown)
    print(json.dumps({k:report[k] for k in ('complete_images','scored_candidates','requests')},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();score(a.evidence,a.baseline)
