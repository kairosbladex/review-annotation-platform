import pymysql,json,pathlib,collections,os
ROOT=pathlib.Path(os.environ.get('HOTSPOT_ENV_ROOT','/Users/wangxiaolin/PycharmProjects/review-annotation-platform'))
WORK=pathlib.Path(os.environ.get('HOTSPOT_SEARCH_DIR',str(ROOT/'.tmp/hotspot-fresh-search')))
WORK.mkdir(parents=True,exist_ok=True)
def connect():
 cfg={}
 for s in (ROOT/'.env.ktd.local').read_text().splitlines():
  if '=' in s and not s.lstrip().startswith('#'):
   k,v=s.split('=',1);cfg[k.strip()]=v.strip().strip('\"').strip("'")
 p='KTD_SOURCE_MYSQL_'
 return pymysql.connect(host=cfg[p+'HOST'],port=int(cfg[p+'PORT']),user=cfg[p+'USER'],password=cfg[p+'PASSWORD'],database=cfg[p+'DATABASE'],connect_timeout=8,read_timeout=20,cursorclass=pymysql.cursors.DictCursor)
PROJECT='2c95908e958f3b75019593551f520126'
if __name__=='__main__':
 c=connect()
 try:
  with c.cursor() as q:
   q.execute('START TRANSACTION READ ONLY')
   q.execute("SELECT upload_data_type_name,COUNT(*) n,MAX(event_time) latest FROM bo_task_event WHERE project_id=%s AND del_flag=0 AND (upload_data_type_name LIKE '%%光%%' OR upload_data_type_name LIKE '%%热%%') GROUP BY upload_data_type_name",(PROJECT,));print(json.dumps(q.fetchall(),default=str,ensure_ascii=False))
   q.execute("SELECT e.id,e.task_id,e.event_time,e.upload_data_type_name,t.task_name,t.begin_time,t.fly_batch_no FROM bo_task_event e LEFT JOIN bo_task t ON t.id=e.task_id WHERE e.project_id=%s AND e.del_flag=0 AND (e.upload_data_type_name LIKE '%%光斑%%' OR e.upload_data_type_name LIKE '%%热斑%%') ORDER BY e.event_time DESC LIMIT 2000",(PROJECT,));rows=q.fetchall();(WORK/'events.json').write_text(json.dumps(rows,default=str,ensure_ascii=False,indent=2));groups={}
   for r in rows:
    g=groups.setdefault(r['task_id'],{'task_id':r['task_id'],'task_name':r['task_name'],'latest':str(r['event_time']),'n':0});g['n']+=1
   print(json.dumps(list(groups.values()),ensure_ascii=False,indent=2))
 finally:c.rollback();c.close()
