from query import *
p=WORK;events=json.loads((p/'events.json').read_text());c=connect();out=[]
try:
 with c.cursor() as q:
  q.execute('START TRANSACTION READ ONLY')
  ids=[r['id'] for r in events];sql=','.join(['%s']*len(ids))
  q.execute(f'SELECT id,busi_id,busi_type_code,file_path,image_resolution,imgxy,date_time_original,visual FROM sys_annex_file WHERE del_flag=0 AND busi_id IN ({sql})',ids);out=q.fetchall()
finally:c.rollback();c.close()
(p/'attachments.json').write_text(json.dumps(out,default=str,ensure_ascii=False,indent=2))
ex=json.loads((p/'exposure.json').read_text());fresh=[r for r in events if r['id'] not in set(ex['snapshot_events'])|set(ex['seen_events']) and r['task_id'] not in ex['seen_tasks']]
for e in fresh:
 a=[x for x in out if x['busi_id']==e['id']];print(json.dumps({'event':e['id'],'date':e['event_time'],'task':e['task_name'],'files':[{k:x[k] for k in ['id','busi_type_code','file_path','image_resolution','imgxy']} for x in a]},ensure_ascii=False))
print('events',len(events),'annex',len(out),'outside_snapshot_candidates',len(fresh))
