#!/usr/bin/env python3
"""Append independently human-confirmed groups without changing existing splits."""
import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path
from consolidate_recheck_gold import read_rows,write_rows,iou,sha,dumps,TZ,Union,refresh_sums
from freeze_recheck_gold_split import stratified_assignments,stats

def append(out,incoming):
    gold=read_rows(out/'gold.jsonl');registry=read_rows(out/'split_registry.jsonl');conflicts=read_rows(out/'conflicts.jsonl')
    latest={r['group_id']:r for r in registry};aliases={a:r['group_id'] for r in latest.values() for a in r['aliases']}
    source=read_rows(incoming);new=[];dedup=0;timestamp=datetime.now(TZ).isoformat();union=Union()
    for r in source:
        if r.get('label_source') not in {'human','human_confirmed','wxl_web'} or not r.get('labeled_by') or not r.get('labeled_at') or str(r.get('note','')).lower().startswith('auto annotation'):
            raise ValueError('incoming row is not confirmed human gold')
        if r.get('label') not in {'true','false','uncertain'}:raise ValueError('invalid label')
        bbox=r['bbox'];digest=r['image_sha256']
        import math,re
        if not re.fullmatch('[a-f0-9]{64}',digest) or len(bbox)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in bbox) or bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:raise ValueError('invalid image/bbox identity')
        match=[g for g in gold if g['image_sha256']==digest and g['class_name']==r['class_name'] and iou(g['bbox'],bbox)>=0.5]
        if match:
            if 'gpt_prompt_exposed' in r.get('exposure',[]) and any(g['split']=='locked_test_gpt' for g in match):raise ValueError('duplicate exposes a locked group; human adjudication required')
            if any(g['label']!=r['label'] for g in match):raise ValueError('label conflict requires human adjudication; original gold unchanged')
            dedup+=1;continue
        member=['image:'+digest,'event:'+str(r['event_id'])]
        if r.get('source_task_id'):member.append('task:'+str(r['source_task_id']))
        for alias in member:union.join(member[0],alias)
        duplicate=[x for x in new if x['image_sha256']==digest and x['class_name']==r['class_name'] and iou(x['bbox'],bbox)>=0.5]
        if duplicate:
            if any(x['label']!=r['label'] for x in duplicate):raise ValueError('incoming labels conflict; human adjudication required')
            dedup+=1;continue
        new.append({**r,'group_aliases':member})
    components=collections.defaultdict(list)
    for r in new:components[union.find(r['group_aliases'][0])].append(r)
    rows=[];newgroups=[]
    for group in components.values():
        members=sorted({a for r in group for a in r['group_aliases']});existing={aliases[a] for a in members if a in aliases}
        if len(existing)>1:raise ValueError('incoming source connects frozen groups; human adjudication required')
        gid=next(iter(existing)) if existing else 'group-'+sha(dumps(members).encode())[:20]
        exposed=sorted({e for r in group for e in r.get('exposure',[])})
        if existing and 'gpt_prompt_exposed' in exposed and latest[gid]['split']=='locked_test_gpt':raise ValueError('new GPT exposure affects a locked group; do not silently invalidate it')
        for r in group:
            row={'gold_id':sha(dumps([r['image_sha256'],r['class_name'],r['bbox']]).encode()),'group_id':gid,'group_aliases':members,
                 'bbox':r['bbox'],'class_name':r['class_name'],'label':r['label'],'image_sha256':r['image_sha256'],
                 'source_images':[r['source_image']] if r.get('source_image') else [],'population':r.get('population',['historical_review']),
                 'exposure':r.get('exposure',[]),'group_exposure':exposed,'gpt_history_unknown':r.get('gpt_history_unknown',True) is not False,
                 'duplicate_policy':'same_image_class_iou_ge_0.5_connected_component',
                 'provenance':[{'source_file':str(incoming),'source_sha256':sha(incoming.read_bytes()),'candidate_id':r['candidate_id'],'labeled_by':r['labeled_by'],'labeled_at':r['labeled_at'],'original_label':r['label'],'note':r.get('note',''),'revision':r.get('revision')}]}
            if existing:
                old=latest[gid];row.update(split=old['split'],valid_for=old['valid_for'],split_status=old['status'])
            else:newgroups.append(row)
            rows.append(row)
    assignments,audit=stratified_assignments(newgroups) if newgroups else ({},[])
    for row in rows:
        if row['group_id'] in assignments:
            a=assignments[row['group_id']];row.update(split=a['split'],valid_for=a['valid_for'],split_status=a['status'])
    for gid in sorted({r['group_id'] for r in rows}):
        a=assignments.get(gid,latest.get(gid));member=sorted({x for r in rows if r['group_id']==gid for x in r['group_aliases']}|set(a['aliases']))
        registry.append({**a,'aliases':member,'record_type':'human_gold_append','recorded_at':timestamp,'reason':'新增人工确认金标；已有组不重切','seed':20260930})
    gold+=rows
    report=json.loads((out/'summary.json').read_text());report.update(gold_samples=len(gold),classes=stats(gold),population={p:stats([g for g in gold if p in g['population']]) for p in sorted({p for g in gold for p in g['population']})})
    write_rows(out/'gold.jsonl',gold);write_rows(out/'split_registry.jsonl',registry)
    write_rows(out/'splits.jsonl',[{k:r[k] for k in ['gold_id','group_id','split','valid_for','split_status']} for r in gold])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (out/'README.md').open('a') as f:f.write(f'\n追加 {timestamp}: 新人工框{len(rows)}，重复{dedup}；已有组切分不变。GPT已曝光的新金标只进入开发集。\n')
    refresh_sums(out);return {'appended':len(rows),'deduplicated':dedup,'new_groups':len(assignments),'strata':audit}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--incoming',type=Path,required=True);a=p.parse_args();print(json.dumps(append(a.out,a.incoming),ensure_ascii=False))
