"""Audit persisted results and bounded logs without changing business state."""
import argparse,csv,hashlib,json,math,re
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def normalized(b):
    if b.get('box_format')!='xywh_top_left':raise ValueError('unsupported bbox format')
    x,y,w,h,fw,fh=[float(b[k]) for k in ['x','y','width','height','frame_width','frame_height']]
    if min(w,h,fw,fh)<=0:raise ValueError('invalid geometry')
    return [x/fw,y/fh,(x+w)/fw,(y+h)/fh]
def match_boxes(boxes,results):
    if [r.get('index') for r in results]!=list(range(1,len(boxes)+1)):return False
    return all(b['class_name']==r.get('originalClass') and len(r.get('originalBbox',[]))==4 and all(math.isclose(x,y,abs_tol=1e-8) for x,y in zip(normalized(b),r['originalBbox'])) for b,r in zip(boxes,results))
def run(snapshot,worker,api,out):
    d=json.loads(snapshot.read_text()); es={e['event_no']:e for e in d['events']};logs=[]
    for line in worker.read_text().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if isinstance(r,dict) and r.get('time'):logs.append(r)
    api_rows=[]
    for line in api.read_text().splitlines():
        m=re.search(r'^(\S+).*POST /api/v1/review-agent/review HTTP/1.1" (\d+)',line)
        if m:api_rows.append((datetime.fromisoformat(m[1].replace('Z','+00:00')),int(m[2])))
    private=[];public=[];attempts=[]
    for event_no in json.loads(snapshot.with_name('expected-event-nos.json').read_text()):
        e=es[event_no]; eid=e['id']; rs=[r for r in d['image_results'] if r['event_id']==eid and not r['deleted']]
        orders=[r for r in d['orders'] if r['event_id']==eid and not r['deleted']];errs=[]
        if e['recheck_status']!='2' or e['deleted']:errs.append('event_not_completed')
        if len(rs)!=1:errs.append('image_result_count_not_one')
        r=rs[0];result=json.loads(r['recheck_result']);event_result=json.loads(e['recheck_result']);a=next(a for a in d['annexes'] if a['id']==r['annex_id']);boxes=json.loads(a['imgxy']);rows=result['resultList']
        if result!=event_result:errs.append('event_image_disagreement')
        if not match_boxes(boxes,rows):errs.append('candidate_identity_mismatch')
        if json.loads(e['imgxy'])!=boxes:errs.append('current_event_annex_input_mismatch')
        n=len(boxes)
        if n!=r['total'] or sum(r[k] or 0 for k in ['true_count','false_count','doubtful_count'])!=n:errs.append('count_mismatch')
        for key,col in [('trueCount','true_count'),('falseCount','false_count'),('doubtfulCount','doubtful_count')]:
            if result[key]!=r[col] or sum(x.get(key,0) for x in rows)!=r[col]:errs.append('detail_count_mismatch')
        if a['deleted'] or a['busi_id']!=eid:errs.append('annex_identity_mismatch')
        if r['recheck_time']!=e['recheck_time']:errs.append('timestamp_mismatch')
        for o in orders:
            if o['step_type_name']!='提交慧眼复核':continue
            st=datetime.fromisoformat(o['occurred_time']).replace(tzinfo=timezone(timedelta(hours=8)))
            cand=[l for l in logs if abs((datetime.fromisoformat(l['time'])-st).total_seconds())<=1.1 and l['candidate_count']==n and set(l['classes'])=={b['class_name'] for b in boxes}]
            terminal=[z for z in orders if z['occurred_time']>=o['occurred_time'] and z['step_type_name'] in ['慧眼复核完成','慧眼复核失败']]
            terminal=sorted(terminal,key=lambda x:x['occurred_time'])
            valid=False;api_count=0
            if len(cand)==1 and terminal:
                l=cand[0];end=datetime.fromisoformat(l['time'])+timedelta(milliseconds=l['duration_ms']);term=terminal[0]
                dt=datetime.fromisoformat(term['occurred_time']).replace(tzinfo=timezone(timedelta(hours=8)))
                expected='慧眼复核完成' if l['result']=='ok' else '慧眼复核失败'
                code=200 if l['result']=='ok' else int(l['result'])
                api_count=sum(abs((at-end).total_seconds())<=0.05 and ac==code for at,ac in api_rows)
                valid=abs((dt-end).total_seconds())<=1.1 and term['step_type_name']==expected and api_count==1
            attempts.append({'event_key':digest(eid)[:16],'submitted_at':o['occurred_time'],'unique_worker_match':len(cand)==1,'terminal_and_api_match':valid,'api_matches':api_count,'request_id':cand[0]['request_id'] if len(cand)==1 else None,'status':cand[0]['result'] if len(cand)==1 else None})
        row={'event_key':digest(eid)[:16],'category':e['upload_data_type_name'],'boxes':n,'confirmed':r['true_count'],'rejected':r['false_count'],'doubtful':r['doubtful_count'],'completed_at':e['recheck_time'],'image_rows':len(rs),'submission_count':sum(o['step_type_name']=='提交慧眼复核' for o in orders),'issues':';'.join(sorted(set(errs))),'database_verified':not errs,'page_verified':False,'input_sha256':digest(a['imgxy']),'result_sha256':digest(r['recheck_result'])}
        public.append(row);private.append(dict(row,event_id=eid,event_no=e['event_no'],annex_id=a['id']))
    summary={'snapshot':d['identity'][0]['snapshot_time'],'transaction_read_only':d['identity'][0]['read_only'],'transaction_end':d['transaction_end'],'events':len(public),'database_verified':sum(r['database_verified'] for r in public),'boxes':sum(r['boxes'] for r in public),'confirmed':sum(r['confirmed'] for r in public),'rejected':sum(r['rejected'] for r in public),'page_verified':0,'attempts':len(attempts),'unique_worker_and_terminal_and_api_matches':sum(r['unique_worker_match'] and r['terminal_and_api_match'] for r in attempts),'attempt_statuses':dict(Counter(str(r['status']) for r in attempts)),'api_match_tolerance_ms':50,'worker_versions':sorted(set(l['prompt_version'] for l in logs)),'association_basis':'unique submission/start time + candidate count/classes + terminal/end time + API response; no shared end-to-end ID','historical_inputs_limit':'current input verified against persisted result geometry; no captured original request body'}
    for f,rows in [('events.csv',public),('private/events.csv',private),('private/request-mapping.csv',attempts)]:
        with (out/f).open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'business-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--worker',type=Path,required=True);p.add_argument('--api',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.snapshot,a.worker,a.api,a.out)
