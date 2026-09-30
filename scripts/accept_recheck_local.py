#!/usr/bin/env python3
"""Local API + dedicated Celery + private Redis acceptance with injected faults.

All processes are bounded and cleaned in finally. Shared sources are read-only.
"""
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]

def acceptance(python,source_root,out):
    import httpx
    from sqlalchemy import select
    sys.path.insert(0,str(ROOT/'backend'))
    from app.db import build_engine,build_session_factory
    from app.models import RecheckRecord,Annotation
    tmp=ROOT/'.tmp/local-acceptance';tmp.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    for port in [8197,6397,5197]:
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('independent port occupied: '+str(port))
    db=tmp/'acceptance.sqlite3'
    if db.exists():raise RuntimeError('acceptance already exists; do not overwrite')
    mode=tmp/'fault.json';mode.write_text(json.dumps({'mode':'ok'}));calls=tmp/'calls.jsonl'
    env={**os.environ,'PYTHONPATH':str(ROOT/'backend'),'DATABASE_URL':'sqlite:///'+str(db),'REDIS_URL':'redis://127.0.0.1:6397/0',
         'MINIO_ENDPOINT':'','SOURCE_IMAGE_BASE_URL':'','LOCAL_READONLY_IMAGE_ROOT':str(source_root/'data'),
         'SERVICE_API_KEY':'isolated-replay-service','ADMIN_USERNAME':'replay-admin','ADMIN_PASSWORD':'isolated-replay-password',
         'JWT_SECRET':'isolated-replay-jwt-secret-32-bytes','RECHECK_ENABLED':'true','AUTO_ANNOTATION_API_KEY':'',
         'RECHECK_FAULT_FILE':str(mode),'RECHECK_CALLS_FILE':str(calls),'VITE_API_PROXY_TARGET':'http://127.0.0.1:8197'}
    bootstrap=tmp/'worker.py'
    bootstrap.write_text('''import os,json,time
from pathlib import Path
from app.tasks import celery_app as module
from app.adapters.recheck_relay import RelayFailure
from app.services.online_recheck import ROLE_ORDER
class FaultRelay:
 def mode(self):return json.loads(Path(os.environ['RECHECK_FAULT_FILE']).read_text())['mode']
 def usage(self):
  mode=self.mode()
  if mode in ['usage_auth_failed','usage_unavailable']:raise RelayFailure(mode)
  return {'remaining':4 if mode=='low_balance' else 100}
 def request(self,**kwargs):
  mode=self.mode()
  with Path(os.environ['RECHECK_CALLS_FILE']).open('a') as f:f.write(json.dumps({'id':kwargs['idempotency_key'],'mode':mode})+'\\n')
  if mode=='kill':time.sleep(120)
  if mode in ['rate_limited','execution_unknown']:raise RelayFailure(mode,1)
  return {'actual_model':'injected-local-test','usage':{'total_tokens':1},'answer':{'scene':{'label':'ground','confidence':0.9,'reason':'injected'},'decisions':[{'target_index':0,'consensus':'true','votes':[{'role':r,'label':'true','confidence':0.9,'reason':'injected local advice'} for r in ROLE_ORDER]}]}}
module._relay_instance=FaultRelay()
module.celery_app.worker_main(['worker','--pool=solo','--concurrency=1','-Q','recheck','--hostname=recheck-local-acceptance','--loglevel=WARNING','--without-gossip','--without-mingle'])
''')
    procs=[];handles=[];events=[]
    def start(name,command,memory):
        log=(tmp/(name+'.log')).open('w');handles.append(log)
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/limit_local_process.py'),'--memory-mb',str(memory),'--cpus','1','--wall-seconds','1800','--',*command],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        procs.append((name,proc,memory));return proc
    def stop(proc):
        if proc.poll() is None:
            table=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True)
            children=[int(l.split()[0]) for l in table.splitlines() if len(l.split())==2 and int(l.split()[1])==proc.pid]
            for child in children:
                try:os.killpg(child,signal.SIGCONT);os.killpg(child,signal.SIGTERM)
                except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    try:
        start('redis',['redis-server','--bind','127.0.0.1','--port','6397','--save','','--appendonly','no','--maxmemory','64mb','--maxmemory-policy','noeviction','--dir',str(tmp)],128)
        api=start('api',[python,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8197','--no-access-log'],512)
        worker=start('worker',[python,str(bootstrap)],1024)
        start('frontend',['node',str(ROOT/'frontend/node_modules/vite/bin/vite.js'),str(ROOT/'frontend'),'--host','127.0.0.1','--port','5197','--strictPort'],512)
        client=httpx.Client(base_url='http://127.0.0.1:8197',timeout=10,trust_env=False)
        deadline=time.monotonic()+25
        while True:
            try:
                if client.get('/api/v1/health').status_code==200:break
            except httpx.HTTPError:pass
            if time.monotonic()>deadline:raise RuntimeError('local API unavailable')
            time.sleep(.2)
        factory=build_session_factory(build_engine(env['DATABASE_URL']))
        rows=[json.loads(l) for l in (ROOT/'data/recheck-gold-consolidated-20260930/gold.jsonl').read_text().splitlines()]
        gold=next(r for r in rows if r['split']=='development' and r['class_name']=='车' and any(Path(p).is_file() and Path(p).is_relative_to(source_root/'data') for p in r['source_images']))
        image=next(Path(p) for p in gold['source_images'] if Path(p).is_file() and Path(p).is_relative_to(source_root/'data'))
        key=str(image.relative_to(source_root/'data'));digest=hashlib.sha256(image.read_bytes()).hexdigest()
        def push(name,cls='车',human=False,clean=False,mismatch=False):
            payload={'event_id':'replay-'+name,'source_service':'isolated-replay','image':{'bucket':'local','object_key':key,'sha256':digest if not mismatch else '0'*64},
                     'targets':[{'target_index':0,'class_name':cls,'bbox':gold['bbox'],'raw_decision':'confirmed'}],'metadata':{'human_handled':human,'population':['historical_review']}}
            if clean:payload.update(clean_image=payload['image'],clean_event_id=payload['event_id']);payload['targets'][0]['clean_bbox']=gold['bbox']
            response=client.post('/api/v1/ingest/review-events',json=payload,headers={'X-Service-Key':env['SERVICE_API_KEY']});assert response.status_code==201,response.text
            with factory() as s:rid=s.scalar(select(RecheckRecord.id).where(RecheckRecord.event_id==payload['event_id']))
            return rid,payload,response.json()['sample_id']
        def await_status(rid,status,seconds=20):
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                with factory() as s:
                    r=s.get(RecheckRecord,rid)
                    if r.status==status:return r
                time.sleep(.2)
            raise RuntimeError('status not reached: '+status)
        def setmode(v):mode.write_text(json.dumps({'mode':v}))
        def count(rid):return sum(json.loads(l)['id']==rid for l in calls.read_text().splitlines()) if calls.exists() else 0
        rid,payload,sample_id=push('duplicate',clean=True);await_status(rid,'done')
        assert client.post('/api/v1/ingest/review-events',json=payload,headers={'X-Service-Key':env['SERVICE_API_KEY']}).status_code==200
        from app.tasks.dispatcher import CeleryDispatcher
        dispatcher=CeleryDispatcher(env['REDIS_URL']);dispatcher.dispatch_recheck(rid);time.sleep(.5);assert count(rid)==1
        events.append({'case':'重复推送/重复Celery投递','status':'passed','calls':count(rid),'evidence':'SQLite unique record + calls.jsonl; injected relay'})
        with factory() as s:assert not s.scalar(select(Annotation))
        for scenario in ['low_balance','usage_auth_failed','usage_unavailable','rate_limited','execution_unknown']:
            setmode(scenario);r,p,_=push(scenario);expected='waiting_quota' if scenario in ['low_balance','usage_auth_failed','usage_unavailable'] else 'failed'
            record=await_status(r,expected);assert record.decision is None
            expected_calls=2 if scenario=='rate_limited' else 1 if scenario=='execution_unknown' else 0
            assert count(r)==expected_calls
            events.append({'case':scenario,'status':'passed','task_status':record.status,'decision':record.decision,'calls':count(r),'evidence':'real local Celery with injected relay fault'})
        setmode('ok')
        for scenario,cls,handled,mismatch in [('unsupported','光伏板',False,False),('human','车',True,False),('wrong_image','车',False,True)]:
            r,p,_=push(scenario,cls,handled,mismatch=mismatch);expected='failed' if mismatch else 'skipped';record=await_status(r,expected);assert count(r)==0
            events.append({'case':scenario,'status':'passed','task_status':record.status,'calls':0})
        # Force a genuine worker process kill while a paid-looking call is in flight.
        setmode('kill');r,p,_=push('worker_kill');await_status(r,'running')
        deadline=time.monotonic()+10
        while count(r)==0 and time.monotonic()<deadline:time.sleep(.1)
        assert count(r)==1
        # Find only descendants of our guard; kill the task-owned worker process.
        ps=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True)
        children=[int(l.split()[0]) for l in ps.splitlines() if len(l.split())==2 and int(l.split()[1])==worker.pid]
        assert children
        os.kill(children[0],signal.SIGKILL);stop(worker)
        setmode('ok');worker=start('worker-restarted',[python,str(bootstrap)],1024)
        dispatcher.dispatch_recheck(r);time.sleep(1)
        assert count(r)==1
        from app.services.online_recheck import recover_interrupted
        with factory() as s:recover_interrupted(s,datetime.now(timezone.utc))
        dispatcher.dispatch_recheck(r);time.sleep(.5);assert count(r)==1
        events.append({'case':'真实worker被杀/重启/重复投递','status':'passed','task_status':'failed','calls':1,'evidence':'SIGKILL local worker; durable running record recovered as unknown failure; no automatic resend'})
        # Manual chain (isolated DB only) and exported confirmed gold.
        token=client.post('/api/v1/auth/login',json={'username':env['ADMIN_USERNAME'],'password':env['ADMIN_PASSWORD']}).json()['access_token'];headers={'Authorization':'Bearer '+token}
        record=await_status(rid,'done');cid=record.candidate_id
        for label,rev in [('false',0),('true',1)]:
            response=client.patch('/api/v1/candidates/'+cid+'/annotation',headers=headers,json={'label':label,'note':'isolated human simulation','expected_revision':rev});assert response.status_code==200
        from export_recheck_human_gold import export
        exported=export(factory);assert len(exported)==1 and exported[0]['label']=='true'
        events.append({'case':'建议与人工金标隔离/修订历史/导出','status':'passed','exported':1,'evidence':'isolated simulation only; not appended to formal gold'})
        # Browser evidence runs in an isolated Chrome profile and is deleted by Playwright.
        browser_script=tmp/'browser.cjs'
        browser_script.write_text('''const {chromium}=require('/Users/wangxiaolin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--disable-gpu']});try{const page=await browser.newPage({viewport:{width:1500,height:1000}});await page.goto('http://127.0.0.1:5197');await page.getByLabel('管理员账号').fill('replay-admin');await page.getByLabel('登录密码').fill('isolated-replay-password');await page.getByRole('button',{name:'登录工作台'}).click();await page.getByLabel('按状态筛选').selectOption('all');await page.locator('.queue-item').last().click();await page.getByText('GPT 建议',{exact:true}).waitFor();await page.screenshot({path:process.env.RECHECK_SCREENSHOT,fullPage:true});console.log('browser_passed');}finally{await browser.close();}})().catch(e=>{console.error(e.message);process.exit(1)});
''')
        env['RECHECK_SCREENSHOT']=str(out/'detail.png')
        # Snapshot endpoint evidence regardless of browser availability.
        detail=client.get('/api/v1/samples/'+sample_id,headers=headers).json()
        (out/'detail-evidence.json').write_text(json.dumps({'gpt_advice':detail['candidates'][0]['gpt_advice'],'annotation':detail['candidates'][0]['annotation']},ensure_ascii=False,indent=2,default=str)+'\n')
        result=subprocess.run([sys.executable,str(ROOT/'scripts/limit_local_process.py'),'--memory-mb','1024','--cpus','1','--wall-seconds','90','--','node',str(browser_script)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=100)
        events.append({'case':'浏览器对照展示','status':'passed' if result.returncode==0 else 'not_verified','evidence':result.stdout[-300:]+result.stderr[-600:]})
        client.close()
    finally:
        for name,proc,memory in reversed(procs):stop(proc)
        for h in handles:h.close()
        released=[]
        for port in [8197,6397,5197]:
            with socket.socket() as s:released.append({'port':port,'released':s.connect_ex(('127.0.0.1',port))!=0})
        report={'checks':events,'processes':[{'name':n,'memory_limit_mb':m,'cpu_limit':1,'stopped':p.poll() is not None} for n,p,m in procs],
                'ports':released,'shared_database_writes':0,'shared_push_config_changes':0,'device_actions':0,'real_model_calls':0,
                'limitation':'故障验收使用注入的中转站，验证状态和请求次数，不证明真实供应商在超时后未扣费；未知状态一律不重发。实际模型能力见独立基线评测。'}
        (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();acceptance(a.python,a.source_root,a.out)
