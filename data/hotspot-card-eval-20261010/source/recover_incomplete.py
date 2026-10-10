"""Retry only slots with no valid response; preserve failures and never repeat a successful slot."""
import base64,concurrent.futures,importlib.util,json,os,secrets,subprocess,sys,threading,time,urllib.request
from pathlib import Path
s=importlib.util.spec_from_file_location('evaluation',Path(__file__).with_name('evaluate.py'));E=importlib.util.module_from_spec(s);s.loader.exec_module(E)
stage=sys.argv[1];root=E.ROOT;out=root/'runs'/stage
assert (out/'complete.json').exists(),'Primary batch must complete first'
failed=[(p,json.loads(p.read_text())) for p in (out/'out').glob('*.json') if not json.loads(p.read_text())['success']]
if not failed:print('No incomplete slots');sys.exit(0)
assert not (out/'recovery.json').exists(),'Only one bounded recovery batch allowed'
E.limits();before=E.stats();assert before['available_fraction']>=.15
private=root/'.private';home=private/'home';tok=secrets.token_hex(32)
env=dict(os.environ,CODEX_REVIEW_BINARY='/cli/codex',CODEX_REVIEW_HOME=str(home),CODEX_HOME=str(home),HOME=str(home),CODEX_REVIEW_WORKER_TOKEN=tok,CODEX_REVIEW_BIND='127.0.0.1',CODEX_REVIEW_PORT='18406',CODEX_REVIEW_TMP_DIR=str(private/'tmp'),CODEX_REVIEW_PROMPT_VERSION='p000',CUDA_VISIBLE_DEVICES='')
lf=(out/'recovery-worker.log').open('w');worker=subprocess.Popen([sys.executable,str(root/'source/experiment_worker.py')],env=env,stdout=lf,stderr=lf,start_new_session=True)
stop=threading.Event();index={u['event_id']:u for u in E.read('split.json')}
def monitor():
 previous=E.stats()
 while not stop.wait(5):
  d=E.stats()
  with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
  if d['available_fraction']<.15 or d['swap_current']>0 or d['pswpout']-previous['pswpout']>65536 or d['load'][0]>128:worker.terminate();stop.set();return
  previous=d
recovered=[]
def one(pair):
 p,d=pair;original=json.loads(p.read_text());assert not original['success'];u=index[d['event_id']];image=Path(u['image'] if d['mode']=='screenshot' else u['clean']).read_bytes();assert E.digest(image)==d['image_sha256'];assert E.digest(d['prompt'].encode())==d['prompt_sha256']
 t=time.monotonic();d['previous_incomplete']=original
 for _ in range(2):
  if stop.is_set():break
  try:
   payload={'image':base64.b64encode(image).decode(),'candidates':d['candidates'],'prompt':d['prompt']}
   req=urllib.request.Request(E.URL+'/review',json.dumps(payload).encode(),headers={'Authorization':'Bearer '+tok,'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=240) as f:result=json.load(f)
   assert len(result['results'])==len(d['keys']) and {x['target_index'] for x in result['results']}==set(range(len(d['keys'])))
   d.update(success=True,result=result);d['attempts'].append({'status':'ok','recovery':True});recovered.append(p.name);break
  except Exception as exc:d['attempts'].append({'status':type(exc).__name__,'http_code':getattr(exc,'code',None),'recovery':True});stop.wait(5)
 d['elapsed']+=time.monotonic()-t;E.write(str(p.relative_to(root)),d)
try:
 for _ in range(30):
  try:
   with urllib.request.urlopen(E.URL+'/health',timeout=2) as f:assert json.load(f)['max_concurrency']==6
   break
  except Exception:time.sleep(1)
 else:raise RuntimeError('worker not ready')
 threading.Thread(target=monitor,daemon=True).start()
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(one,failed))
finally:
 stop.set();worker.terminate()
 try:worker.wait(timeout=15)
 except subprocess.TimeoutExpired:worker.kill();worker.wait()
 lf.close();E.write(f'runs/{stage}/recovery.json',{'time':E.now(),'slots':[p.name for p,d in failed],'recovered':recovered,'before':before,'after':E.stats(),'shared':E.shared(),'worker_exit':worker.returncode})
print('Recovered',len(recovered),'of',len(failed))
