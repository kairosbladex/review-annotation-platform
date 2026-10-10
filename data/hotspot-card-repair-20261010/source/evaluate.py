"""Frozen paired experiment; development selection precedes a single locked holdout run."""
import argparse,base64,collections,concurrent.futures,csv,datetime,hashlib,importlib.util,json,os,secrets,signal,subprocess,sys,threading,time,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:18406'
STOP=threading.Event()
def digest(b):return hashlib.sha256(b).hexdigest()
def read(name):return json.loads((ROOT/name).read_text())
def write(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def load_worker():
 s=importlib.util.spec_from_file_location('production',ROOT/'source/production_worker.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
PROD=load_worker()
def doc(card):
 return {'version':'p000-pure-class-cards','common':PROD.load_prompt('p000')['common'],'cards':{'光斑':read('cards.json')['repair']} if card in ('repair','repair_context') else {}}
def candidates(u,mode):
 return [{'target_index':i,'class_name':'光斑','bbox':b['screenshot_bbox' if mode=='screenshot' else 'bbox']} for i,b in enumerate(u['boxes'])]
def shared():
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://192.168.130.238:18396/health',timeout=5) as f:return json.load(f)
def stats():
 m={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()};v={l.split()[0]:int(l.split()[1]) for l in Path('/proc/vmstat').read_text().splitlines()}
 cg=Path('/sys/fs/cgroup')
 return {'time':now(),'available_fraction':m['MemAvailable']/m['MemTotal'],'swap_free_kb':m['SwapFree'],'pswpout':v['pswpout'],'load':os.getloadavg(),'memory_current':int((cg/'memory.current').read_text()),'swap_current':int((cg/'memory.swap.current').read_text())}
def limits():
 cg=Path('/sys/fs/cgroup');d={k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','cpu.max']}
 assert d['memory.max']=='4294967296' and d['memory.swap.max']=='0'
 q,p=map(int,d['cpu.max'].split());assert q/p<=3
 return d

def run(stage):
 freeze=read('freeze.json')
 for name,h in freeze['files'].items():assert digest((ROOT/name).read_bytes())==h,name
 out=ROOT/'runs'/stage
 assert not out.exists(),'Stage already exists; holdout must never be repeated'
 out.mkdir(parents=True)
 us=[u for u in read('split.json') if u['split']==stage]
 if stage=='development':cards=['p000','repair','context','repair_context']
 else:cards=['p000',read('selection.json')['selected']]
 jobs=[]
 # Counterbalanced cyclic order per image, reversed in repetition 2; all conditions interleaved.
 for rep in [1,2]:
  for n,u in enumerate(us):
   combos=[(mode,c) for mode in ['screenshot','original'] for c in cards]
   shift=n%len(combos);combos=combos[shift:]+combos[:shift]
   if rep==2:combos=list(reversed(combos))
   for mode,card in combos:jobs.append({'event_id':u['event_id'],'mode':mode,'card':card,'repeat':rep})
 write(f'runs/{stage}/schedule.json',jobs)
 write(f'runs/{stage}/runtime-before.json',{'time':now(),'limits':limits(),'host':stats(),'shared':shared()})
 private=ROOT/'.private';private.mkdir(exist_ok=True,mode=0o700);home=private/'home';home.mkdir(exist_ok=True)
 if not (home/'auth.json').exists():(home/'auth.json').symlink_to('/auth.json')
 tmp=private/'tmp';tmp.mkdir(exist_ok=True)
 tok=secrets.token_hex(32)
 env=dict(os.environ,CODEX_REVIEW_BINARY='/cli/codex',CODEX_REVIEW_HOME=str(home),CODEX_HOME=str(home),CODEX_REVIEW_WORKER_TOKEN=tok,CODEX_REVIEW_BIND='127.0.0.1',CODEX_REVIEW_PORT='18406',CODEX_REVIEW_TMP_DIR=str(tmp),CODEX_REVIEW_PROMPT_VERSION='p000',CUDA_VISIBLE_DEVICES='')
 lf=(out/'worker.log').open('w');worker=subprocess.Popen([sys.executable,str(ROOT/'source/experiment_worker.py')],env=env,stdout=lf,stderr=lf,start_new_session=True)
 monitor_done=threading.Event();index={u['event_id']:u for u in us};lock=threading.Lock();done=0
 def monitor():
  previous=stats()
  while not monitor_done.wait(5):
   s=stats()
   with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
   if s['available_fraction']<.15 or s['swap_current']>0 or s['pswpout']-previous['pswpout']>65536 or s['load'][0]>128:
    STOP.set();worker.terminate();return
   previous=s
 def one(job):
  nonlocal done
  if STOP.is_set():return
  u=index[job['event_id']];cs=candidates(u,job['mode']);prompt=PROD.build_prompt(doc(job['card']),cs)
  image=Path(u['image'] if job['mode']=='screenshot' else u['clean']);expected=u['screenshot_sha256' if job['mode']=='screenshot' else 'clean_sha256'];raw=image.read_bytes();assert digest(raw)==expected
  payload={'image':base64.b64encode(raw).decode(),'candidates':cs,'prompt':prompt,'layout':'context' if job['card'] in ('context','repair_context') else 'legacy'};attempts=[];result=None;t0=time.monotonic()
  for attempt in range(2):
   if STOP.is_set():break
   try:
    req=urllib.request.Request(URL+'/review',json.dumps(payload).encode(),headers={'Authorization':'Bearer '+tok,'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=240) as f:result=json.load(f)
    assert len(result['results'])==len(cs) and {x['target_index'] for x in result['results']}==set(range(len(cs)))
    assert all(x['decision'] in ('true','false','uncertain') for x in result['results'])
    attempts.append({'status':'ok'});break
   except Exception as e:
    result=None
    attempts.append({'status':type(e).__name__,'http_code':getattr(e,'code',None)})
    if attempt==0:STOP.wait(3)
  name=f"{job['event_id']}-{job['mode']}-{job['card']}-{job['repeat']}"
  write(f'runs/{stage}/out/{name}.json',{**job,'keys':[b['key'] for b in u['boxes']],'success':result is not None,'result':result,'attempts':attempts,'elapsed':time.monotonic()-t0,'prompt':prompt,'prompt_sha256':digest(prompt.encode()),'image_sha256':expected,'candidates':cs})
  with lock:
   done+=1;write(f'runs/{stage}/progress.json',{'done':done,'total':len(jobs),'time':now()})
 try:
  for _ in range(30):
   try:
    with urllib.request.urlopen(URL+'/health',timeout=2) as f:health=json.load(f)
    assert health['max_concurrency']==6;break
   except Exception:time.sleep(1)
  else:raise RuntimeError('worker not ready')
  threading.Thread(target=monitor,daemon=True).start()
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(one,jobs))
  assert not STOP.is_set(),'resource guard stopped stage'
 finally:
  monitor_done.set();worker.terminate()
  try:worker.wait(timeout=15)
  except subprocess.TimeoutExpired:worker.kill();worker.wait()
  lf.close();write(f'runs/{stage}/runtime-after.json',{'time':now(),'host':stats(),'shared':shared(),'worker_exit':worker.returncode})
 write(f'runs/{stage}/complete.json',{'time':now(),'done':done,'total':len(jobs)})

def score(stage):
 units=[u for u in read('split.json') if u['split']==stage];labels={r['key']:{'是光斑':'true','不是光斑':'false','看不清':'uncertain'}[r['label']] for r in read('labels.json')}
 records=[json.loads(p.read_text()) for p in (ROOT/'runs'/stage/'out').glob('*.json')];schedule=read(f'runs/{stage}/schedule.json')
 assert len(records)==len(schedule) and all(r['success'] for r in records),'Incomplete calls cannot be scored as negatives'
 execution_ids=[r['result']['execution_id'] for r in records];assert len(set(execution_ids))==len(records)
 by={};detail=[]
 for r in records:
  key=(r['mode'],r['card'],r['repeat']);m=by.setdefault(key,collections.Counter())
  for x in r['result']['results']:
   k=r['keys'][x['target_index']];y=labels[k];pred=x['decision'];m['n']+=1;m['correct']+=y==pred;m['uncertain']+=pred=='uncertain';m['false_reject']+=y=='true' and pred!='true';m['explicit_false_reject']+=y=='true' and pred=='false';m['false_accept']+=y=='false' and pred=='true'
   detail.append({'event_id':r['event_id'],'key':k,'mode':r['mode'],'card':r['card'],'repeat':r['repeat'],'gold':y,'pred':pred})
 metrics=[{'mode':k[0],'card':k[1],'repeat':k[2],**dict(v),'accuracy':v['correct']/v['n']} for k,v in sorted(by.items())]
 # Pair at event level, average repeats; merge exact duplicate screenshot clusters.
 groups={u['event_id']:u['screenshot_sha256'] for u in units};cis=[]
 for mode in ['screenshot','original']:
  for card in sorted({d['card'] for d in detail}-{'p000'}):
   ds={(d['key'],d['card'],d['repeat']):d for d in detail if d['mode']==mode};agg={}
   for u in units:
    a=agg.setdefault(groups[u['event_id']],np.zeros(5))
    for b in u['boxes']:
     k=b['key'];a[0]+=1
     for rep in [1,2]:
      base=ds[k,'p000',rep];cand=ds[k,card,rep];y=base['gold'];p=base['pred'];q=cand['pred']
      a[1]+=.5*((q==y)-(p==y));a[2]+=.5*((y=='false' and q=='true')-(y=='false' and p=='true'));a[3]+=.5*((y=='true' and q!='true')-(y=='true' and p!='true'));a[4]+=.5*((q=='uncertain')-(p=='uncertain'))
   arr=np.array(list(agg.values()));rng=np.random.default_rng(20261010);sums=arr[rng.integers(0,len(arr),size=(20000,len(arr)))].sum(axis=1);dist=sums[:,1:]/sums[:,0,None]
   cis.append({'mode':mode,'card':card,'clusters':len(arr),'events':len(units),'n':int(arr[:,0].sum()),'delta':(arr[:,1:].sum(axis=0)/arr[:,0].sum()).tolist(),'ci95':np.quantile(dist,[.025,.975],axis=0).T.tolist(),'order':['accuracy','false_accept_per_box','false_reject_per_box','uncertain_per_box']})
 report={'stage':stage,'calls':len(records),'metrics':metrics,'paired':cis,'transport_retries':sum(len(r['attempts'])-1 for r in records)}
 if stage=='holdout':
  card=read('selection.json')['selected'];c=next(c for c in cis if c['mode']=='screenshot' and c['card']==card)
  fp_ok=all(by['screenshot',card,rep]['false_accept']<=by['screenshot','p000',rep]['false_accept'] for rep in [1,2])
  report['gate']={'passed':c['ci95'][0][0]>0 and fp_ok,'accuracy_lower':c['ci95'][0][0],'false_accept_not_increased_each_repeat':fp_ok}
 write(f'{stage}-report.json',report);write(f'{stage}-per-box.json',detail)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','score']);p.add_argument('stage',choices=['development']);a=p.parse_args()
 try:globals()[a.action](a.stage)
 except Exception as e:
  write(f'{a.stage}-error.json',{'error':str(e),'time':now()});raise
