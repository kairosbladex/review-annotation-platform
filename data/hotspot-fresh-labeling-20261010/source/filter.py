from query import *
p=WORK;rs=json.loads((p/'events.json').read_text());ex=json.loads((p/'exposure.json').read_text());seen=set(ex['seen_events'])|set(ex['snapshot_events'])|set(json.loads((p/'exposure-ids.json').read_text())['ids'])
for name in ['codex-multiclass-review-20261008/final-result/source-records.json','codex-initial-review-20261008/source-records.json']:
 x=json.loads((ROOT/'data'/name).read_text());seen.update(x.keys() if isinstance(x,dict) else [r.get('id','') for r in x])
c=connect()
try:
 with c.cursor() as q:
  q.execute('START TRANSACTION READ ONLY');ids=list(seen);tasks=set(ex['seen_tasks'])
  for i in range(0,len(ids),200):
   chunk=ids[i:i+200];q.execute('SELECT id,task_id FROM bo_task_event WHERE project_id=%s AND id IN ('+','.join(['%s']*len(chunk))+')',[PROJECT]+chunk);tasks.update(r['task_id'] for r in q.fetchall() if r['task_id'])
finally:c.rollback();c.close()
fresh=[r for r in rs if r['id'] not in seen and r['task_id'] not in tasks and r['task_id']];a=json.loads((p/'attachments.json').read_text());jobs=[]
for e in fresh:
 annex=[x for x in a if x['busi_id']==e['id']];shots=[x for x in annex if x['busi_type_code']=='SHOTIMG'];raw=[x for x in annex if x['busi_type_code']=='PRIMARYIMG']
 if len(shots)==len(raw)==1:jobs.append({'event':e,'shot':shots[0],'raw':raw[0]})
(p/'download-jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2));print('unexposed metadata candidates',len(fresh),'paired annex records',len(jobs),'by task',dict(collections.Counter(j['event']['task_name'] for j in jobs)))
