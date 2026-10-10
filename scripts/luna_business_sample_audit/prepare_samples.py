"""Prepare prediction-blind annotation queues; never import model decisions as labels."""
import argparse,csv,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
CLASSES=('人员','违建','道路裂纹');SEED=20261011

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def readj(p):return json.loads(p.read_text())
def run(source_root,out):
    private=out/'private'; sources={r['event_id']:r for r in map(json.loads,(source_root/'source_manifest.jsonl').open())}
    inf={r['candidate_id']:r for r in map(json.loads,(source_root/'input/inference_manifest.jsonl').open())}
    items=readj(private/'autotune.json');groups={r['id']:r for r in readj(private/'historical-groups.json')}
    def task(eid):return groups.get(eid,{}).get('task_id') or sources.get(eid,{}).get('task_id')
    hold_events={u['unit'] for u in items if u['split']=='holdout'};hold_tasks={task(e) for e in hold_events if task(e)}
    seen_tasks={task(u['unit']) for u in items if task(u['unit'])}; seen_events={u['unit'] for u in items}
    pools=defaultdict(list);excluded=Counter();overlap=Counter()
    for u in items:
        if u['split']=='holdout':continue
        eid=u['unit'];tid=task(eid)
        for b in u['candidates']:
            cls=b['class_name'];key=b['key']
            if cls not in CLASSES:continue
            if tid in hold_tasks:excluded[cls+':historical_holdout_task']+=1;overlap[u['split']]+=1;continue
            if not tid:excluded[cls+':unknown_task']+=1;continue
            meta=inf.get(key)
            if not meta or eid not in sources:excluded[cls+':missing_local_source_manifest']+=1;continue
            if not(meta.get('has_clean_source') and meta.get('clean_temporal_verified') and meta.get('clean_match_status')=='strong'):
                excluded[cls+':clean_match_not_verified']+=1;continue
            pools[cls].append((u,b,meta))
    queue=[]; transfers={}; remote='/data/workspace/jzheng/codex-ktd-real-monitoring-20260924/'
    for cls in CLASSES:
        pool=sorted(pools[cls],key=lambda x:sha(str(SEED)+x[1]['key']))
        # Round-robin events prevents a crowded image from filling the entire queue.
        byevent=defaultdict(list)
        for item in pool:byevent[item[0]['unit']].append(item)
        ordered=[]
        while byevent and len(ordered)<60:
            for eid in list(byevent):
                ordered.append(byevent[eid].pop(0))
                if not byevent[eid]:del byevent[eid]
                if len(ordered)==60:break
        for u,b,m in ordered:
            eid=u['unit'];opaque=sha(b['key'])[:20];eventkey=sha(eid)[:20];
            files={'screenshot':f'media/{eventkey}-screen.jpg','annotated_context':f'media/{opaque}-annotated.jpg','clean_context':f'media/{opaque}-clean.jpg'}
            transfers[files['screenshot']]=remote+'work/annotated/'+eid+'.jpg'
            transfers[files['annotated_context']]=remote+m['annotated_context'];transfers[files['clean_context']]=remote+m['clean_context']
            queue.append({'sample_id':opaque,'queue':'diagnostic','class_name':cls,'event_id':eid,'candidate_key':b['key'],'task_id':task(eid),'bbox':b['bbox'],'files':files,'annotated_source_box':m['annotated_context_source_box_px'],'clean_source_box':m['clean_context_source_box_px'],'target_bbox_px':m['bbox_px'],'source_frame_size':m['frame_size'],'clean_match_status':m['clean_match_status'],'clean_temporal_verified':m['clean_temporal_verified'],'clean_matched_raw_time':m['clean_matched_raw_time'],'clean_full_ncc':m['clean_full_ncc'],'clean_note':'已配准干净局部；不是原生高清完整帧','label':'','note':'','reviewer':'','label_basis':''})
    d=readj(private/'snapshot.json');extra=readj(private/'recent-extra.json');fresh=[]
    for e in extra['events']+d['recent_candidates']:
        eid=e['id'];tid=e['task_id'];reason=[]
        try:boxes=json.loads(e['imgxy'])
        except (ValueError,TypeError):excluded['recent:invalid_boxes']+=1;continue
        if eid in seen_events or tid in seen_tasks:reason.append('known_historical_event_or_task')
        # The four October 10 events were reviewed in the previous Claude session.
        if e['event_no'].startswith('Event-20261010'):reason.append('previous_business_review_exposure')
        annexes=[a for a in extra['annexes']+d['recent_annexes'] if a['busi_id']==eid]
        shots=[a for a in annexes if a['busi_type_code']=='SHOTIMG'];originals=[a for a in annexes if a['busi_type_code']=='PRIMARYIMG']
        shot=shots[0] if shots else None
        clean_distinct=bool(shot and any(a['file_path']!=shot['file_path'] for a in originals))
        if not clean_distinct:reason.append('no_distinct_clean_original')
        if not tid:reason.append('missing_flight_group')
        # Some historical events cannot be mapped to a flight; novelty cannot be certified.
        reason.append('historical_group_coverage_incomplete')
        for i,b in enumerate(boxes):
            cls=b.get('class_name');
            if cls not in CLASSES:continue
            opaque=sha(eid+':'+str(i))[:20];files={}
            if shot:files['screenshot']=shot['file_path']
            fresh.append({'sample_id':opaque,'queue':'recent_reserved','class_name':cls,'event_id':eid,'task_id':tid,'candidate_index':i,'box':b,'source_files':files,'eligibility':'pending','reasons':reason,'label':'','note':'','reviewer':'','label_basis':''})
    (private/'diagnostic.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2));(private/'recent-reserved.json').write_text(json.dumps(fresh,ensure_ascii=False,indent=2))
    (private/'media').mkdir(exist_ok=True)
    with (private/'sftp-batch.txt').open('w') as f:
        for dest,src in sorted(transfers.items()):f.write(f'-get "{src}" "{(private/dest).resolve()}"\n')
    public=[{'sample_id':r['sample_id'],'queue':r['queue'],'class_name':r['class_name'],'group_hash':sha(r['task_id'])[:16],'label':'','note':'','reviewer':'','label_basis':''} for r in queue+fresh]
    with (out/'annotation-template.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(public[0]));w.writeheader();w.writerows(public)
    summary={'seed':SEED,'selection':'hash order, event round-robin, max 60 per class; no model decisions or old labels used','recent_window':['2026-09-25','2026-10-11'],'recent_window_amendment':'initial Oct08-11 had only four events; metadata-only expansion before selection','diagnostic_counts':dict(Counter(r['class_name'] for r in queue)),'diagnostic_events':len({r['event_id'] for r in queue}),'recent_reserved_counts':dict(Counter(r['class_name'] for r in fresh)),'recent_reserved_events':len({r['event_id'] for r in fresh}),'certified_fresh_blind_samples':0,'exclusions':dict(excluded),'holdout_tasks_known':len(hold_tasks),'historical_items_without_task':sum(not task(u['unit']) for u in items),'label_status':'all blank; no human gold created','media_fetch_count':len(transfers),'negative_sample_status':'candidate queue only; actual negatives require human labeling','historical_split_overlap_excluded':dict(overlap)}
    (out/'sample-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source_root,a.out)
