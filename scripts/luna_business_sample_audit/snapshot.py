"""Bounded, read-only business snapshot. Raw output must remain private."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pymysql

def connect(env: Path):
    cfg={}
    for line in env.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            k,v=line.split('=',1); cfg[k.strip()]=v.strip().strip('"').strip("'")
    p='KTD_SOURCE_MYSQL_'
    return pymysql.connect(host=cfg[p+'HOST'],port=int(cfg[p+'PORT']),user=cfg[p+'USER'],password=cfg[p+'PASSWORD'],database=cfg[p+'DATABASE'],charset='utf8mb4',autocommit=False,connect_timeout=10,read_timeout=25,cursorclass=pymysql.cursors.DictCursor)

def run(env:Path,out:Path):
    c=connect(env); data={}; query_log=[]
    try:
        with c.cursor() as q:
            def query(sql,args=()):
                query_log.append({'sql':sql,'params':args});q.execute(sql,args);return q.fetchall()
            for s in ['SET SESSION MAX_EXECUTION_TIME=20000','SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ','SET SESSION TRANSACTION READ ONLY','START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY']: q.execute(s)
            data['identity']=query('SELECT DATABASE() db,NOW() snapshot_time,@@session.time_zone time_zone,@@session.transaction_read_only read_only,@@session.max_execution_time max_execution_time')
            assert data['identity'][0]['db']=='ktd_lx' and data['identity'][0]['read_only']==1
            data['window_orders']=query("SELECT id,event_id,step_type,step_type_name,occurred_time,create_time,del_flag+0 deleted FROM bo_event_work_order WHERE occurred_time BETWEEN %s AND %s ORDER BY occurred_time,id",('2026-10-10 22:40:00','2026-10-10 23:10:00'))
            ids=sorted({r['event_id'] for r in data['window_orders'] if '复核' in (r['step_type_name'] or '')})
            assert 0<len(ids)<=40, f'unexpected event count: {len(ids)}'
            ph=','.join(['%s']*len(ids))
            data['events']=query(f'SELECT id,event_no,upload_data_type_name,event_time,task_id,terminal_id,recheck_status,recheck_time,recheck_result,imgxy,update_time,del_flag+0 deleted FROM bo_task_event WHERE id IN ({ph}) ORDER BY event_no',ids)
            data['orders']=query(f'SELECT id,event_id,step_type_name,occurred_time,create_time,del_flag+0 deleted FROM bo_event_work_order WHERE event_id IN ({ph}) AND step_type_name LIKE %s ORDER BY occurred_time,id',ids+['%复核%'])
            data['image_results']=query(f'SELECT id,event_id,annex_id,session_id,total,true_count,false_count,doubtful_count,elapsed_seconds,recheck_time,create_time,update_time,recheck_result,del_flag+0 deleted FROM bo_task_event_image_recheck_result WHERE event_id IN ({ph}) ORDER BY recheck_time,id',ids)
            data['annexes']=query(f'SELECT id,busi_id,event_id,file_name,file_path,busi_type_code,file_type,imgxy,image_resolution,etag,create_time,update_time,del_flag+0 deleted FROM sys_annex_file WHERE busi_id IN ({ph}) ORDER BY busi_id,id',ids)
            # Fixed predeclared window; only metadata and coordinates, no images/model calls.
            data['recent_candidates']=query("SELECT id,event_no,upload_data_type_name,event_time,task_id,terminal_id,imgxy FROM bo_task_event WHERE event_time >= %s AND event_time < %s AND del_flag=0 AND (upload_data_type_name LIKE %s OR upload_data_type_name LIKE %s OR upload_data_type_name LIKE %s) ORDER BY event_time,id LIMIT 2001",('2026-10-08 00:00:00','2026-10-11 00:00:00','%人员%','%违建%','%裂%'))
            data['recent_capped']=len(data['recent_candidates'])>2000
            if data['recent_capped']: data['recent_candidates']=data['recent_candidates'][:2000]
            recent_ids=[x['id'] for x in data['recent_candidates']]
            data['recent_annexes']=[]
            for start in range(0,len(recent_ids),100):
                batch=recent_ids[start:start+100];p=','.join(['%s']*len(batch))
                data['recent_annexes']+=query(f'SELECT id,busi_id,event_id,file_name,file_path,busi_type_code,imgxy,image_resolution,etag FROM sys_annex_file WHERE busi_id IN ({p}) AND del_flag=0 ORDER BY busi_id,id',batch)
    finally:
        c.rollback();c.close()
    data['transaction_end']='rollback'; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str));out.chmod(0o600)
    out.with_name('queries.json').write_text(json.dumps(query_log,ensure_ascii=False,indent=2,default=str));out.with_name('queries.json').chmod(0o600)
    print(json.dumps({'events':len(data['events']),'image_results':len(data['image_results']),'recent_candidates':len(data['recent_candidates']),'recent_capped':data['recent_capped'],'rollback':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--env',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.env,a.out)
