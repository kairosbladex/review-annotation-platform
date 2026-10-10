"""Bounded metadata follow-up after the initial fresh-sample inventory was sparse."""
import argparse,json
from pathlib import Path
from snapshot import connect

def run(env,out):
 p=out/'private';items=json.loads((p/'autotune.json').read_text());building=json.loads((p/'building.json').read_text());ids=sorted({x['unit'] for x in items+building}); c=connect(env);queries=[];data={}
 try:
  with c.cursor() as q:
   for sql in ['SET SESSION MAX_EXECUTION_TIME=20000','SET SESSION TRANSACTION READ ONLY','START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY']:q.execute(sql)
   def query(sql,args):
    queries.append({'sql':sql,'args':args});q.execute(sql,args);return q.fetchall()
   data['recent-counts']=query('SELECT DATE(event_time) day,upload_data_type_name category,COUNT(*) n FROM bo_task_event WHERE event_time >= %s AND event_time < %s AND del_flag=0 AND (upload_data_type_name LIKE %s OR upload_data_type_name LIKE %s OR upload_data_type_name LIKE %s) GROUP BY DATE(event_time),upload_data_type_name ORDER BY day',('2026-09-25','2026-10-11','%人员%','%违建%','%裂%'))
   events=query('SELECT id,event_no,upload_data_type_name,event_time,task_id,terminal_id,imgxy FROM bo_task_event WHERE event_time >= %s AND event_time < %s AND del_flag=0 AND (upload_data_type_name LIKE %s OR upload_data_type_name LIKE %s OR upload_data_type_name LIKE %s) ORDER BY event_time,id LIMIT 101',('2026-09-25','2026-10-08','%人员%','%违建%','%裂%'))
   assert len(events)<=100,'metadata cap exceeded'
   ei=[e['id'] for e in events]
   annexes=query('SELECT id,busi_id,event_id,file_name,file_path,busi_type_code,imgxy,image_resolution,etag FROM sys_annex_file WHERE del_flag=0 AND busi_id IN ('+','.join(['%s']*len(ei))+') ORDER BY busi_id,id',ei) if ei else []
   data['recent-extra']={'events':events,'annexes':annexes};data['historical-groups']=[]
   for i in range(0,len(ids),100):
    batch=ids[i:i+100];data['historical-groups']+=query('SELECT id,task_id,terminal_id,event_time FROM bo_task_event WHERE id IN ('+','.join(['%s']*len(batch))+')',batch)
 finally:c.rollback();c.close()
 for name,val in data.items():(p/(name+'.json')).write_text(json.dumps(val,ensure_ascii=False,default=str,indent=2))
 (p/'enrichment-queries.json').write_text(json.dumps(queries,ensure_ascii=False,default=str,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--env',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.env,a.out)
