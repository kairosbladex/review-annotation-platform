"""Bounded real-package A/B, isolated error contracts and two rollback drills.
Both worker processes and runner share ONE 4G/no-swap/3-core cgroup.
Only concurrency is overridden by an import harness; package bytes are unchanged.
"""
import base64,concurrent.futures,hashlib,json,os,pathlib,random,secrets,signal,subprocess,sys,threading,time,urllib.request,urllib.error
R=pathlib.Path('/data/workspace/jzheng/luna-building-release-validation-20261010');E=R/'evidence'
BINARY='/data/workspace/jzheng/codex-luna-review/bin/codex-0.162.0-alpha.17.2/codex'
EXPECTED='73b73b048546993298234a295de01ef2424e7a8b29def9d0ecb1ca7cb77500d4'
UNIT='luna-building-release-validation-20261010'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}));STOP=threading.Event();CHILDREN={}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(n,x):
 p=E/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def get(port):
 with OPENER.open(f'http://127.0.0.1:{port}/health',timeout=3) as f:return json.load(f)
def shared():
 return {'api':subprocess.check_output(['docker','inspect','--format','{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}} {{.State.StartedAt}}','yoloe_service_test'],text=True).strip(),'worker':subprocess.check_output(['systemctl','--user','show','codex-luna-review-worker','-p','MainPID','-p','ActiveEnterTimestamp','-p','NRestarts'],text=True).strip()}
def resources():
 m={l.split(':')[0]:int(l.split()[1]) for l in pathlib.Path('/proc/meminfo').read_text().splitlines()}
 return dict(at=time.time(),available=m['MemAvailable']/m['MemTotal'],swap_used=m['SwapTotal']-m['SwapFree'],load=os.getloadavg()[0],**{k:(CG/k).read_text().strip() for k in ['memory.current','memory.peak','memory.swap.current','memory.events']})
def guard():
 previous=None
 while not STOP.wait(1):
  s=resources()
  with (E/'resources.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
  vm={l.split()[0]:int(l.split()[1]) for l in pathlib.Path('/proc/vmstat').read_text().splitlines()};swap=(vm['pswpin'],vm['pswpout'])
  abnormal=previous and (swap[0]-previous[0]>4096 or swap[1]-previous[1]>4096)
  previous=swap
  if s['available']<.15 or s['load']>1.5*os.cpu_count() or int(s['memory.swap.current'])>0 or abnormal:
   write('resource-stop.json',s);subprocess.Popen(['systemctl','--user','stop',UNIT]);return

def launch(label,version,legacy=False):
 assert sha(BINARY)==EXPECTED
 port={'A':18401,'B':18402}[label]
 env=dict(os.environ,CODEX_REVIEW_BINARY=BINARY,CODEX_REVIEW_WORKER_TOKEN=TOKEN,CODEX_REVIEW_BIND='127.0.0.1',CODEX_REVIEW_PORT=str(port),CODEX_REVIEW_TMP_DIR=str(R/'tmp'),CODEX_REVIEW_PROMPT_VERSION=version,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 # Reuse auth location only, never copy credentials or print environment.
 private=dict(l.split('=',1) for l in pathlib.Path('/data/workspace/jzheng/luna-prompt-autotune/private/worker.env').read_text().splitlines() if '=' in l and not l.startswith('#'))
 env['CODEX_REVIEW_HOME']=private['CODEX_REVIEW_HOME']
 package=R/('legacy' if legacy else 'candidate')/'scripts'
 cmd=[sys.executable,str(R/'src/package_harness.py'),str(package)]
 log=open(R/f'{label}.log','a');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=log);log.close();CHILDREN[label]=p
 for _ in range(30):
  if p.poll() is not None:raise RuntimeError('worker failed to start')
  try:
   h=get(port);assert h['max_concurrency']==3;return h
  except OSError:time.sleep(.3)
 raise RuntimeError('health timeout')
def stop(label):
 p=CHILDREN.pop(label,None)
 if p:
  p.terminate()
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:p.kill();p.wait()
def request(u,arm,repeat=0,payload_override=None,auth=True):
 assert sha(BINARY)==EXPECTED
 b=(R/'images'/(u['unit']+'.jpg')).read_bytes();assert hashlib.sha256(b).hexdigest()==u['sha256']
 payload={'image':base64.b64encode(b).decode(),'candidates':[dict(target_index=i,class_name=c['class_name'],bbox=c['bbox']) for i,c in enumerate(u['candidates'])]}
 if payload_override is not None:payload.update(payload_override)
 started=time.time();body={};attempts=[]
 for _ in range(3):
  req=urllib.request.Request(f'http://127.0.0.1:{18401 if arm=="A" else 18402}/review',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+(TOKEN if auth else 'wrong'),'Content-Type':'application/json'})
  try:
   with OPENER.open(req,timeout=240) as f:body=json.load(f);status=200
  except urllib.error.HTTPError as err:status=err.code;body=json.loads(err.read())
  attempts.append({'at':time.time(),'status':status})
  if status not in [429,503]:break
  time.sleep(2)
 assert sha(BINARY)==EXPECTED
 return dict(unit=u['unit'],arm=arm,repeat=repeat,status=status,body=body,attempts=attempts,seconds=time.time()-started,binary_sha256=EXPECTED,image_sha256=u['sha256'])

def main():
 global TOKEN,CG
 E.mkdir(exist_ok=True);TOKEN=secrets.token_urlsafe(32)
 cg=next(l.split('::',1)[1] for l in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if '::' in l);CG=pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/')
 limits={k:(CG/k).read_text().strip() for k in ['memory.max','memory.swap.max']}
 assert limits=={'memory.max':str(4*1024**3),'memory.swap.max':'0'};assert len(os.sched_getaffinity(0))<=3;assert resources()['available']>=.15
 write('runtime-before.json',dict(shared=shared(),resources=resources(),limits=limits,cpus=sorted(os.sched_getaffinity(0)),cpu_max=(CG/'cpu.max').read_text().strip() if (CG/'cpu.max').exists() else 'not delegated; enforced by 3-core affinity'))
 threading.Thread(target=guard,daemon=True).start()
 units=json.loads((E/'units.json').read_text());hashes={str(p.relative_to(R)):sha(p) for folder in ['candidate','legacy'] for p in (R/folder).rglob('*') if p.is_file() and not p.name.startswith('._')};write('package-hashes.json',hashes)
 try:
  write('startup.json',{'A':launch('A','p000'),'B':launch('B','p000-building-only')})
  # Invalid/auth requests never reach CLI. Real CLI checks are in paired run and rollbacks.
  smoke=[]
  for name,over,auth,expected in [('auth',{'image':''},False,401),('empty',{'candidates':[]},True,400),('bad_image',{'image':'!'},True,400),('bad_index',{'candidates':[dict(target_index=99,class_name='违建',bbox=[0,0,1,1])]},True,400)]:
   z=request(units[0],'B',payload_override=over,auth=auth);assert z['status']==expected and 'results' not in z['body'];smoke.append(dict(name=name,**z))
  write('http-contracts.json',smoke)
  rng=random.Random(20261010);rng.shuffle(units);jobs=[]
  for u in units:
   conditions=[(a,n) for a in ['A','B'] for n in [1,2]];rng.shuffle(conditions);jobs.extend((u,a,n) for a,n in conditions)
  write('schedule.json',[dict(unit=u['unit'],arm=a,repeat=n) for u,a,n in jobs]);slots={a:threading.Semaphore(3) for a in ['A','B']}
  def one(job):
   u,a,n=job
   with slots[a]:z=request(u,a,n)
   if z['status']==200:assert {v['target_index'] for v in z['body']['results']}==set(range(len(u['candidates'])))
   write(f'results/{a}-{n}/{u["unit"]}.json',z);print(a,n,u['unit'],z['status'],flush=True);return z['status']==200
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:success=list(pool.map(one,jobs))
  write('run-complete.json',dict(total=len(jobs),success=sum(success),at=time.time()))
  # Configuration rollback and code rollback, ONLY this isolated B worker.
  drill=[];u=next(x for x in units if x['cohort']=='known_fp_regression')
  for name,version,legacy in [('config_rollback','p000',False),('code_rollback','p000',True),('candidate_restore','p000-building-only',False)]:
   stop('B');h=launch('B',version,legacy);z=request(u,'B');assert z['status']==200
   assert ('prompt_version' not in h) if legacy else h['prompt_version']==version
   drill.append(dict(step=name,health=h,response=z,script_sha256=sha(R/('legacy' if legacy else 'candidate')/'scripts/run_codex_luna_worker.py')))
  write('rollback-drill.json',drill)
  for p,h in hashes.items():assert sha(R/p)==h
  write('validation-complete.json',dict(at=time.time(),binary_sha256=sha(BINARY),package_unchanged=True,all_calls_success=all(success),shared_unchanged=shared()==json.loads((E/'runtime-before.json').read_text())['shared']))
 finally:
  for label in list(CHILDREN):stop(label)
  STOP.set();write('runtime-after.json',dict(shared=shared(),resources=resources(),children_stopped=True))
if __name__=='__main__':main()
